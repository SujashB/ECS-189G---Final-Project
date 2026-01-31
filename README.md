# DeepRefusal Reproduction

**Reproducing: "Rebuilding LLM Safety Mechanisms via Probabilistic Ablation of Refusal Signals"**

This repository reproduces the DeepRefusal method for training more robust safety mechanisms in LLMs through probabilistic ablation of refusal signals during training.

## Research Question

> Does quantization amplify or dampen refusal signals, and how does that affect safety under probabilistic ablation?

## Hardware Requirements

- **Tested on:** ROG Strix G16 with NVIDIA RTX 5070 Ti Laptop GPU
- **VRAM:** 16GB+ recommended for training
- **RAM:** 32GB+ recommended

## Repository Structure

```
.
├── configs/                    # YAML configuration files
│   ├── models.yaml            # Model paths and settings
│   ├── train.yaml             # QLoRA training parameters
│   ├── intervention.yaml      # DeepRefusal intervention settings
│   ├── data.yaml              # Dataset configurations
│   └── eval.yaml              # Evaluation settings
├── scripts/                    # Runnable scripts
│   ├── prepare_data.py        # Data preparation pipeline
│   ├── extract_refusal_signal.py  # Stage 1: Extract refusal direction
│   ├── make_harmful_prefix_data.py # Generate harmful prefix augmentation
│   ├── train_variant.py       # Stage 2: Train model variants
│   ├── eval.py                # Evaluation script
│   └── stress_test.py         # Stress test for robustness
├── src/                        # Source modules
│   ├── data_utils.py          # Data loading utilities
│   ├── model_utils.py         # Model loading and generation
│   ├── refusal_detector.py    # Refusal classification
│   └── interventions.py       # DeepRefusal intervention mechanisms
├── data/processed/            # Processed datasets (generated)
├── artifacts/                 # Model artifacts (generated)
├── runs/                      # Training outputs (generated)
├── models/                    # Local model checkpoints (user-provided)
└── requirements.txt
```

## Setup

### 1. Create Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or: .venv\Scripts\activate  # Windows
```

### 2. Install Dependencies

**Important:** Install PyTorch first, then other dependencies.

```bash
# Install PyTorch with CUDA support (adjust for your CUDA version)
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Install remaining dependencies
pip install -r requirements.txt
```

### 3. Authenticate with HuggingFace

Some datasets (HarmBench) require authentication:

```bash
huggingface-cli login
```

Then accept the dataset license at: https://huggingface.co/datasets/walledai/HarmBench

### 4. Prepare Local Models

Place your downloaded models in the `./models/` directory:

```
models/
├── gemma-2b-it/
├── Qwen2.5-1.5B-Instruct/
└── deepseek-llm-1.5b-chat/
```

Update `configs/models.yaml` with your model paths if different.

## Training Variants

The reproduction implements 4 variants:

| Variant | Description |
|---------|-------------|
| `base_quantized` | No training; evaluate base model |
| `standard_refusal_training` | Harmful refusal + benign utility training |
| `refusal_plus_prefill_aug` | + Harmful-prefix augmentation |
| `deeprefusal_quantized` | + Random refusal weakening during training |

## Complete Pipeline Commands

### Step 1: Prepare Data

```bash
python scripts/prepare_data.py \
    --data-config configs/data.yaml \
    --datasets advbench ultrachat xstest harmbench \
    --seed 42
```

**Output:** `data/processed/` with JSONL files for training and evaluation.

### Step 2: Extract Refusal Direction (per model)

```bash
# For Gemma-2B
python scripts/extract_refusal_signal.py \
    --model gemma-2b \
    --models-config configs/models.yaml \
    --data-config configs/data.yaml \
    --intervention-config configs/intervention.yaml \
    --run-verification \
    --seed 42

# For Qwen-1.5B
python scripts/extract_refusal_signal.py \
    --model qwen-1.5b \
    --models-config configs/models.yaml \
    --data-config configs/data.yaml \
    --intervention-config configs/intervention.yaml \
    --run-verification \
    --seed 42

# For DeepSeek-1.5B
python scripts/extract_refusal_signal.py \
    --model deepseek-1.5b \
    --models-config configs/models.yaml \
    --data-config configs/data.yaml \
    --intervention-config configs/intervention.yaml \
    --run-verification \
    --seed 42
```

**Output:** `artifacts/<model>/refusal_dir.pt` and metadata.

### Step 3: Generate Harmful Prefix Data (per model)

```bash
python scripts/make_harmful_prefix_data.py \
    --model gemma-2b \
    --models-config configs/models.yaml \
    --data-config configs/data.yaml \
    --min-tokens 20 \
    --max-tokens 60 \
    --seed 42
```

**Output:** `data/processed/harmful_prefix_train.jsonl`

### Step 4: Train Variants (2-4)

```bash
# Variant 1: Base quantized (no training needed, just setup)
python scripts/train_variant.py \
    --model gemma-2b \
    --variant base_quantized \
    --models-config configs/models.yaml \
    --seed 42

# Variant 2: Standard refusal training
python scripts/train_variant.py \
    --model gemma-2b \
    --variant standard_refusal_training \
    --models-config configs/models.yaml \
    --train-config configs/train.yaml \
    --data-config configs/data.yaml \
    --seed 42

# Variant 3: Refusal + Prefill augmentation
python scripts/train_variant.py \
    --model gemma-2b \
    --variant refusal_plus_prefill_aug \
    --models-config configs/models.yaml \
    --train-config configs/train.yaml \
    --data-config configs/data.yaml \
    --seed 42

# Variant 4: DeepRefusal (full method)
python scripts/train_variant.py \
    --model gemma-2b \
    --variant deeprefusal_quantized \
    --models-config configs/models.yaml \
    --train-config configs/train.yaml \
    --data-config configs/data.yaml \
    --intervention-config configs/intervention.yaml \
    --seed 42
```

**Output:** `runs/<model>/<variant>/` with checkpoints and config.

### Step 5: Evaluate All Variants

```bash
# Evaluate each variant
for variant in base_quantized standard_refusal_training refusal_plus_prefill_aug deeprefusal_quantized; do
    python scripts/eval.py \
        --model gemma-2b \
        --variant $variant \
        --models-config configs/models.yaml \
        --data-config configs/data.yaml \
        --eval-config configs/eval.yaml \
        --datasets harmbench advbench xstest ultrachat \
        --seed 42
done
```

**Output:** `runs/<model>/<variant>/metrics.json` and `predictions.jsonl`

### Step 6: Run Stress Test

```bash
# Stress test each variant
for variant in base_quantized standard_refusal_training refusal_plus_prefill_aug deeprefusal_quantized; do
    python scripts/stress_test.py \
        --model gemma-2b \
        --variant $variant \
        --models-config configs/models.yaml \
        --data-config configs/data.yaml \
        --eval-config configs/eval.yaml \
        --intervention-config configs/intervention.yaml \
        --alpha-values 0.0 0.5 1.0 1.5 2.0 3.0 \
        --max-samples 100 \
        --seed 42
done
```

**Output:** `runs/<model>/<variant>/stress_test_results.json`

## Quick Start (Single Model)

Run the full pipeline for one model:

```bash
#!/bin/bash
MODEL="gemma-2b"

# 1. Prepare data
python scripts/prepare_data.py --datasets advbench ultrachat xstest harmbench

# 2. Extract refusal direction
python scripts/extract_refusal_signal.py --model $MODEL --run-verification

# 3. Generate harmful prefix data
python scripts/make_harmful_prefix_data.py --model $MODEL

# 4. Train all variants
for variant in base_quantized standard_refusal_training refusal_plus_prefill_aug deeprefusal_quantized; do
    python scripts/train_variant.py --model $MODEL --variant $variant
done

# 5. Evaluate all variants
for variant in base_quantized standard_refusal_training refusal_plus_prefill_aug deeprefusal_quantized; do
    python scripts/eval.py --model $MODEL --variant $variant
done

# 6. Stress test all variants
for variant in base_quantized standard_refusal_training refusal_plus_prefill_aug deeprefusal_quantized; do
    python scripts/stress_test.py --model $MODEL --variant $variant
done
```

## Expected Results

### Evaluation Metrics

| Metric | Dataset | Description |
|--------|---------|-------------|
| Attack Success Rate (ASR) | HarmBench | Lower is better (model refuses harmful requests) |
| Refusal Rate | AdvBench | Higher is better (model refuses harmful requests) |
| Over-refusal Rate | XSTest | Lower is better (model doesn't refuse safe requests) |
| Utility Rate | UltraChat | Higher is better (model helps with benign requests) |

### Stress Test

The stress test measures robustness by forcibly weakening refusal signals:
- **Baseline:** Normal inference
- **Intervention:** Remove refusal direction projection at inference time
- **Robustness Score:** 1 - avg_degradation (higher = more robust)

**Key hypothesis:** DeepRefusal-trained models (variant 4) should show less degradation under stress testing because their safety mechanisms are more distributed across layers.

## Configuration

### Key Parameters in `configs/intervention.yaml`

```yaml
intervention:
  p_weak: 0.3       # Probability of weakening per step
  alpha: 1.0        # Strength of projection removal
  layers: [-1, -2, -3, -4]  # Layers to intervene on
  num_layers_per_step: 2    # Layers sampled per step
  token_frac: 0.5   # Fraction of tokens to modify
```

### Key Parameters in `configs/train.yaml`

```yaml
qlora:
  r: 16             # LoRA rank
  lora_alpha: 32    # LoRA alpha
  lora_dropout: 0.05

training:
  learning_rate: 2.0e-4
  num_train_epochs: 3
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 4
```

## Troubleshooting

### CUDA Out of Memory

- Reduce `per_device_train_batch_size` in `configs/train.yaml`
- Reduce `max_seq_length`
- Ensure `gradient_checkpointing: true`

### HarmBench Access Denied

1. Log in: `huggingface-cli login`
2. Accept license at: https://huggingface.co/datasets/walledai/HarmBench

### Unsloth Installation Issues

```bash
# For CUDA 12.1
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```

## Citation

If you use this code, please cite the original paper:

```bibtex
@article{deeprefusal2024,
  title={Rebuilding LLM Safety Mechanisms via Probabilistic Ablation of Refusal Signals},
  author={...},
  journal={...},
  year={2024}
}
```

## License

This reproduction is for research purposes only. Please respect the licenses of the underlying models and datasets.