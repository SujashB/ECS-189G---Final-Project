"""
DeepRefusal intervention mechanisms.
Implements random refusal weakening during training and inference.
"""

import random
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    """Load a YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class RefusalInterventionHook:
    """
    Hook for applying refusal direction interventions during forward pass.
    Used for both training (random weakening) and stress testing.
    """
    
    def __init__(
        self,
        refusal_direction: torch.Tensor,
        alpha: float = 1.0,
        token_frac: float = 0.5,
        min_token_position: int = 10,
        random_token_positions: bool = True,
        mode: str = "weaken"  # "weaken" or "strengthen"
    ):
        """
        Initialize the intervention hook.
        
        Args:
            refusal_direction: Normalized refusal direction vector [hidden_dim]
            alpha: Intervention strength
            token_frac: Fraction of tokens to intervene on
            min_token_position: Skip first N tokens
            random_token_positions: Randomly sample positions or use all
            mode: "weaken" to remove projection, "strengthen" to add
        """
        self.refusal_direction = refusal_direction
        self.alpha = alpha
        self.token_frac = token_frac
        self.min_token_position = min_token_position
        self.random_token_positions = random_token_positions
        self.mode = mode
        
        # Will be set per-forward pass
        self.active = True
    
    def __call__(
        self,
        module: nn.Module,
        input: Tuple[torch.Tensor, ...],
        output: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply intervention to layer output.
        
        Args:
            module: The layer module
            input: Layer inputs
            output: Layer output hidden states [batch, seq_len, hidden_dim]
            
        Returns:
            Modified hidden states
        """
        if not self.active:
            return output
        
        # Handle different output formats
        if isinstance(output, tuple):
            hidden_states = output[0]
            rest = output[1:]
        else:
            hidden_states = output
            rest = None
        
        batch_size, seq_len, hidden_dim = hidden_states.shape
        device = hidden_states.device
        
        # Move refusal direction to correct device
        d = self.refusal_direction.to(device)
        
        # Create position mask
        position_mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
        valid_positions = list(range(self.min_token_position, seq_len))
        
        if valid_positions:
            if self.random_token_positions:
                # Randomly sample positions
                n_positions = max(1, int(len(valid_positions) * self.token_frac))
                selected = random.sample(valid_positions, min(n_positions, len(valid_positions)))
            else:
                # Use all valid positions with token_frac probability
                selected = [p for p in valid_positions if random.random() < self.token_frac]
            
            for pos in selected:
                position_mask[pos] = True
        
        # Apply intervention: h <- h - alpha * (h · d) * d
        # or for strengthen: h <- h + alpha * d
        if position_mask.any():
            # Get positions to modify [num_selected]
            positions = position_mask.nonzero(as_tuple=True)[0]
            
            # Extract hidden states at those positions [batch, num_selected, hidden_dim]
            h_selected = hidden_states[:, positions, :]
            
            if self.mode == "weaken":
                # Remove projection onto refusal direction
                # projection = (h · d) * d
                projection = torch.einsum("bsh,h->bs", h_selected, d)  # [batch, num_selected]
                projection = projection.unsqueeze(-1) * d  # [batch, num_selected, hidden_dim]
                h_modified = h_selected - self.alpha * projection
            else:  # strengthen
                # Add refusal direction
                h_modified = h_selected + self.alpha * d
            
            # Put back
            hidden_states = hidden_states.clone()
            hidden_states[:, positions, :] = h_modified
        
        if rest is not None:
            return (hidden_states,) + rest
        return hidden_states


class RandomRefusalWeakeningCallback:
    """
    Callback for applying random refusal weakening during training.
    Implements the DeepRefusal intervention strategy.
    """
    
    def __init__(
        self,
        model: nn.Module,
        refusal_direction: torch.Tensor,
        intervention_config: Dict[str, Any],
        layer_names: Optional[List[str]] = None
    ):
        """
        Initialize the callback.
        
        Args:
            model: The language model
            refusal_direction: Refusal direction vector
            intervention_config: Intervention configuration
            layer_names: Names of layers to potentially intervene on
        """
        self.model = model
        self.refusal_direction = refusal_direction
        self.config = intervention_config["intervention"]
        
        # Get layer modules
        self.layers = self._get_layer_modules(model, layer_names)
        
        # Parse layer indices from config
        self.layer_indices = self._parse_layer_indices(
            self.config["layers"],
            len(self.layers)
        )
        
        # Hooks storage
        self.hooks: List[Any] = []
        self.hook_handles: List[Any] = []
        
        # State
        self.enabled = True
    
    def _get_layer_modules(
        self,
        model: nn.Module,
        layer_names: Optional[List[str]] = None
    ) -> List[nn.Module]:
        """Get transformer layer modules from the model."""
        # Try common layer attribute names
        layer_attrs = ["model.layers", "transformer.h", "gpt_neox.layers"]
        
        layers = None
        for attr in layer_attrs:
            try:
                obj = model
                for part in attr.split("."):
                    obj = getattr(obj, part)
                layers = list(obj)
                break
            except AttributeError:
                continue
        
        if layers is None:
            # Fallback: look for ModuleList with "layer" in name
            for name, module in model.named_modules():
                if isinstance(module, nn.ModuleList) and "layer" in name.lower():
                    layers = list(module)
                    break
        
        if layers is None:
            raise ValueError("Could not find transformer layers in model")
        
        return layers
    
    def _parse_layer_indices(
        self,
        layer_spec: List[int],
        num_layers: int
    ) -> List[int]:
        """Convert layer specification (possibly negative) to actual indices."""
        indices = []
        for idx in layer_spec:
            if idx < 0:
                actual_idx = num_layers + idx
            else:
                actual_idx = idx
            if 0 <= actual_idx < num_layers:
                indices.append(actual_idx)
        return indices
    
    def on_step_begin(self) -> None:
        """Called at the beginning of each training step."""
        if not self.enabled:
            return
        
        # Remove previous hooks
        self._remove_hooks()
        
        # Decide whether to apply intervention this step
        if random.random() > self.config["p_weak"]:
            return
        
        # Sample layers to intervene on
        n_layers = min(
            self.config["num_layers_per_step"],
            len(self.layer_indices)
        )
        selected_layers = random.sample(self.layer_indices, n_layers)
        
        # Create and register hooks
        for layer_idx in selected_layers:
            hook = RefusalInterventionHook(
                refusal_direction=self.refusal_direction,
                alpha=self.config["alpha"],
                token_frac=self.config["token_frac"],
                min_token_position=self.config["min_token_position"],
                random_token_positions=self.config["random_token_positions"],
                mode="weaken"
            )
            self.hooks.append(hook)
            
            layer_module = self.layers[layer_idx]
            handle = layer_module.register_forward_hook(hook)
            self.hook_handles.append(handle)
    
    def on_step_end(self) -> None:
        """Called at the end of each training step."""
        self._remove_hooks()
    
    def _remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for handle in self.hook_handles:
            handle.remove()
        self.hooks.clear()
        self.hook_handles.clear()
    
    def disable(self) -> None:
        """Disable interventions."""
        self.enabled = False
        self._remove_hooks()
    
    def enable(self) -> None:
        """Enable interventions."""
        self.enabled = True


class InferenceInterventionManager:
    """
    Manager for applying interventions during inference (stress testing).
    """
    
    def __init__(
        self,
        model: nn.Module,
        refusal_direction: torch.Tensor,
        alpha: float = 1.0,
        layers: List[int] = None,
        mode: str = "weaken"
    ):
        """
        Initialize the manager.
        
        Args:
            model: The language model
            refusal_direction: Refusal direction vector
            alpha: Intervention strength
            layers: Layer indices to intervene on (negative = from end)
            mode: "weaken" or "strengthen"
        """
        self.model = model
        self.refusal_direction = refusal_direction
        self.alpha = alpha
        self.mode = mode
        
        # Get layers
        self.all_layers = self._get_layer_modules(model)
        self.layer_indices = self._parse_layer_indices(
            layers or [-1],
            len(self.all_layers)
        )
        
        self.hooks: List[RefusalInterventionHook] = []
        self.hook_handles: List[Any] = []
    
    def _get_layer_modules(self, model: nn.Module) -> List[nn.Module]:
        """Get transformer layer modules from the model."""
        layer_attrs = ["model.layers", "transformer.h", "gpt_neox.layers"]
        
        for attr in layer_attrs:
            try:
                obj = model
                for part in attr.split("."):
                    obj = getattr(obj, part)
                return list(obj)
            except AttributeError:
                continue
        
        for name, module in model.named_modules():
            if isinstance(module, nn.ModuleList) and "layer" in name.lower():
                return list(module)
        
        raise ValueError("Could not find transformer layers in model")
    
    def _parse_layer_indices(
        self,
        layer_spec: List[int],
        num_layers: int
    ) -> List[int]:
        """Convert layer specification to actual indices."""
        indices = []
        for idx in layer_spec:
            if idx < 0:
                actual_idx = num_layers + idx
            else:
                actual_idx = idx
            if 0 <= actual_idx < num_layers:
                indices.append(actual_idx)
        return indices
    
    def __enter__(self) -> "InferenceInterventionManager":
        """Apply hooks when entering context."""
        for layer_idx in self.layer_indices:
            hook = RefusalInterventionHook(
                refusal_direction=self.refusal_direction,
                alpha=self.alpha,
                token_frac=1.0,  # All tokens for inference
                min_token_position=0,
                random_token_positions=False,
                mode=self.mode
            )
            self.hooks.append(hook)
            
            handle = self.all_layers[layer_idx].register_forward_hook(hook)
            self.hook_handles.append(handle)
        
        return self
    
    def __exit__(self, *args) -> None:
        """Remove hooks when exiting context."""
        for handle in self.hook_handles:
            handle.remove()
        self.hooks.clear()
        self.hook_handles.clear()


def compute_refusal_direction(
    harmful_hidden_states: torch.Tensor,
    benign_hidden_states: torch.Tensor,
    normalize: bool = True
) -> torch.Tensor:
    """
    Compute the refusal direction from hidden states.
    
    d = normalize(mean(harmful_h) - mean(benign_h))
    
    Args:
        harmful_hidden_states: Hidden states for harmful prompts [n_harmful, hidden_dim]
        benign_hidden_states: Hidden states for benign prompts [n_benign, hidden_dim]
        normalize: Whether to normalize the direction
        
    Returns:
        Refusal direction vector [hidden_dim]
    """
    mean_harmful = harmful_hidden_states.mean(dim=0)
    mean_benign = benign_hidden_states.mean(dim=0)
    
    direction = mean_harmful - mean_benign
    
    if normalize:
        direction = direction / direction.norm()
    
    return direction


def project_onto_direction(
    hidden_states: torch.Tensor,
    direction: torch.Tensor
) -> torch.Tensor:
    """
    Compute the projection of hidden states onto a direction.
    
    Args:
        hidden_states: Hidden states [..., hidden_dim]
        direction: Direction vector [hidden_dim], should be normalized
        
    Returns:
        Projection scalars [...] (dot products)
    """
    return torch.einsum("...h,h->...", hidden_states, direction)


def remove_direction_component(
    hidden_states: torch.Tensor,
    direction: torch.Tensor,
    alpha: float = 1.0
) -> torch.Tensor:
    """
    Remove the component along a direction from hidden states.
    
    h_new = h - alpha * (h · d) * d
    
    Args:
        hidden_states: Hidden states [..., hidden_dim]
        direction: Direction vector [hidden_dim], should be normalized
        alpha: Scaling factor (1.0 = full removal)
        
    Returns:
        Modified hidden states [..., hidden_dim]
    """
    projection = project_onto_direction(hidden_states, direction)
    component = projection.unsqueeze(-1) * direction
    return hidden_states - alpha * component


def add_direction_component(
    hidden_states: torch.Tensor,
    direction: torch.Tensor,
    alpha: float = 1.0
) -> torch.Tensor:
    """
    Add a direction component to hidden states.
    
    h_new = h + alpha * d
    
    Args:
        hidden_states: Hidden states [..., hidden_dim]
        direction: Direction vector [hidden_dim]
        alpha: Scaling factor
        
    Returns:
        Modified hidden states [..., hidden_dim]
    """
    return hidden_states + alpha * direction
