# DeepRefusal + KV-Cache Extension Summary

## 1. How the Existing DeepRefusal Algorithm Works

DeepRefusal (paper/repo baseline) is a **representation-level safety alignment** method.

### Core mechanism
1. **Extract refusal direction** `r_hat`:
- Compute hidden-state statistics for harmful vs benign prompts.
- Use the direction that best satisfies:
  - adding it induces refusals,
  - ablating it suppresses refusals.

2. **Probabilistic Activation Ablation (PAA)** during fine-tuning:
- Sample layers with Bernoulli probability `p`.
- Sample token positions with Bernoulli probability `p`.
- Apply intervention:
  - weaken refusal signal via projection removal along `r_hat`.
- This simulates jailbreak-like internal states.

3. **Train to recover refusal under perturbation**:
- Include normal benign utility pairs and harmful refusal pairs.
- Include harmful-prefix (prefilling) augmentation.
- Objective encourages robust refusal behavior even when internal refusal-related activations are weakened.

### Intended effect
- Safety is no longer concentrated in only the first few generated tokens.
- Refusal behavior should remain robust across token depth and layer depth.

---

## 2. New Modifications Added on Top

I implemented a **KV-cache robustness extension for quantized inference**, focused on deployment-style models served through Ollama, with optional HF KV instrumentation support.

### New modules/scripts
- `src/kv_eval/types.py`
- `src/kv_eval/attacks.py`
- `src/kv_eval/metrics.py`
- `src/kv_eval/report.py`
- `src/kv_eval/ollama_runner.py`
- `src/kv_eval/hf_runner.py`
- `scripts/kv_cache_eval.py`
- `scripts/kv_cache_make_tables.py`

### What is new algorithmically
Because Ollama is black-box (no layer hook access), the extension implements a **runtime DeepRefusal analog** plus cache stress testing:

1. **Runtime defense analog (for quantized black-box models)**
- Estimate refusal direction from response embeddings.
- Generate candidate responses under stochastic adversarial pressure (`p_weak`).
- Re-rank candidates using refusal detection + similarity to refusal direction.

2. **KV/cache-focused threat modes**
- `none`
- `prefill`
- `cache_contam` (context priming to contaminate session state)
- `cache_reuse` (persistent conversation/context carryover)

3. **New evaluation outputs**
- Safety metrics: ASR, refusal rate, over-refusal, utility.
- Systems metrics: p50/p95 latency, throughput.
- Uncertainty: bootstrap CIs.

4. **Paper-ready output generation**
- `table_core_asr.csv`
- `figure_pareto.csv`
- `table_baseline_delta.csv`

---

## 3. Summary of Executed Results

## Main combined run used for summary
- Combined directory: `artifacts/kv_cache_eval_combined/20260222_081807`
- Built from two full 20/20 runs:
  - `artifacts/kv_cache_eval/20260222_080443` (gemma3:1b)
  - `artifacts/kv_cache_eval/20260222_081222` (qwen3:1.7b)

### High-level findings

#### Gemma3 (1B)
- In `none` mode, baseline ASR was **0.25**.
- Defense improved to:
  - **0.10** with `p=0.3, a=0.5` (60% relative ASR reduction)
  - **0.05** with `p=0.7, a=1.5` (80% relative ASR reduction)
- Latency impact in `none` mode was favorable in this run (negative overhead vs baseline).
- Under `cache_contam`, harmful ASR was already low, but over-refusal increased for some defense settings.

#### Qwen3 (1.7B)
- ASR remained **1.0** across tested attack modes and defense settings in this configuration.
- Interpretation: the current runtime analog did not move refusal behavior for this model under these prompts/settings.
- Latency often improved or stayed similar, but safety did not improve.

---

## 4. What the Results Mean

1. **Model-dependent safety transfer**
- The same runtime DeepRefusal analog can strongly help one quantized model family (Gemma) but not another (Qwen).
- This suggests refusal-signal geometry and instruction-following dynamics differ materially by model family.

2. **Cache robustness needs explicit measurement**
- Cache-related attack modes expose different failure surfaces than plain prompts.
- Reporting only no-cache prompt ASR can miss behavior shifts in persistent sessions.

3. **Tradeoff behavior is non-uniform**
- Some settings reduce harmful ASR while increasing benign over-refusal (especially in contamination-like contexts).
- A paper should report both safety and utility, not ASR alone.

4. **Current success criterion is partially met**
- Gemma showed strong ASR reductions (meeting the direction of the target claim).
- Qwen did not; therefore the claim should be framed as **conditional/model-specific** unless broader settings or methods close that gap.

---

## 5. Practical Next Steps for Stronger Paper Claims

1. Increase prompt diversity and sample count for CIs with narrower variance.
2. Add stronger Qwen-specific defense variants (e.g., larger candidate set, prompt shields, model-specific refusal detector).
3. Run hybrid mode with HF KV instrumentation for direct causal cache analysis (`past_key_values` diagnostics).
4. Present per-model conclusions instead of one aggregated claim.

---

## 6. File/Artifact Map

### New code
- `src/kv_eval/`
- `scripts/kv_cache_eval.py`
- `scripts/kv_cache_make_tables.py`

### Main artifacts
- Combined final summary: `artifacts/kv_cache_eval_combined/20260222_081807/summary.json`
- Core table: `artifacts/kv_cache_eval_combined/20260222_081807/table_core_asr.csv`
- Delta table: `artifacts/kv_cache_eval_combined/20260222_081807/table_baseline_delta.csv`
- Pareto figure data: `artifacts/kv_cache_eval_combined/20260222_081807/figure_pareto.csv`
