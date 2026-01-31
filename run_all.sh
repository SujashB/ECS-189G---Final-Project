#!/bin/bash
# Full pipeline for DeepRefusal reproduction
# Usage: ./run_all.sh [model_name]
# Example: ./run_all.sh gemma3-1b

set -e  # Exit on error

MODEL="${1:-gemma3-1b}"
echo "Running full pipeline for model: $MODEL"

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Check CUDA availability
echo "Checking CUDA availability..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')" || {
    echo "ERROR: CUDA not available. For RTX 5070 Ti:"
    echo "  1. Reboot your system (driver mismatch fix)"
    echo "  2. Ensure you have the latest NVIDIA driver (570+)"
    echo "  3. Install PyTorch with CUDA 12.8: pip install torch --index-url https://download.pytorch.org/whl/cu128"
    exit 1
}

echo "=============================================="
echo "Step 1: Prepare Data"
echo "=============================================="
python scripts/prepare_data.py \
    --data-config configs/data.yaml \
    --datasets advbench ultrachat xstest harmbench \
    --seed 42

echo "=============================================="
echo "Step 2: Extract Refusal Direction"
echo "=============================================="
python scripts/extract_refusal_signal.py \
    --model $MODEL \
    --models-config configs/models.yaml \
    --data-config configs/data.yaml \
    --intervention-config configs/intervention.yaml \
    --run-verification \
    --seed 42

echo "=============================================="
echo "Step 3: Generate Harmful Prefix Data"
echo "=============================================="
python scripts/make_harmful_prefix_data.py \
    --model $MODEL \
    --models-config configs/models.yaml \
    --data-config configs/data.yaml \
    --min-tokens 20 \
    --max-tokens 60 \
    --seed 42

echo "=============================================="
echo "Step 4: Train All Variants"
echo "=============================================="
for variant in base_quantized standard_refusal_training refusal_plus_prefill_aug deeprefusal_quantized; do
    echo "Training variant: $variant"
    python scripts/train_variant.py \
        --model $MODEL \
        --variant $variant \
        --models-config configs/models.yaml \
        --train-config configs/train.yaml \
        --data-config configs/data.yaml \
        --intervention-config configs/intervention.yaml \
        --seed 42
done

echo "=============================================="
echo "Step 5: Evaluate All Variants"
echo "=============================================="
for variant in base_quantized standard_refusal_training refusal_plus_prefill_aug deeprefusal_quantized; do
    echo "Evaluating variant: $variant"
    python scripts/eval.py \
        --model $MODEL \
        --variant $variant \
        --models-config configs/models.yaml \
        --data-config configs/data.yaml \
        --eval-config configs/eval.yaml \
        --datasets harmbench advbench xstest ultrachat \
        --seed 42
done

echo "=============================================="
echo "Step 6: Stress Test All Variants"
echo "=============================================="
for variant in base_quantized standard_refusal_training refusal_plus_prefill_aug deeprefusal_quantized; do
    echo "Stress testing variant: $variant"
    python scripts/stress_test.py \
        --model $MODEL \
        --variant $variant \
        --models-config configs/models.yaml \
        --data-config configs/data.yaml \
        --eval-config configs/eval.yaml \
        --intervention-config configs/intervention.yaml \
        --alpha-values 0.0 0.5 1.0 1.5 2.0 3.0 \
        --max-samples 100 \
        --seed 42
done

echo "=============================================="
echo "Pipeline Complete!"
echo "=============================================="
echo "Results saved to: runs/$MODEL/"
echo ""
echo "View metrics:"
for variant in base_quantized standard_refusal_training refusal_plus_prefill_aug deeprefusal_quantized; do
    echo "  cat runs/$MODEL/$variant/metrics.json"
done
