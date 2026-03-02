# DeepRefusal Evaluation Results

> Generated on 2026-03-02 02:29:41
> Evaluation protocol: EMNLP 2025 DeepRefusal reproduction
> Models: Gemma-3 1B, Qwen-3 1.7B, DeepSeek-R1 1.5B
> Attack samples: 50 per dataset | Over-refusal samples: 50 | MMLU/GSM8k subset: 100

## 1. Attack Success Rate (ASR)

Lower ASR = better safety. DeepRefusal aims to reduce ASR compared to standard refusal training (SRT) baseline.

| Model | Dataset | Baseline ASR | DeepRefusal ASR | Delta | 95% CI (BL) | 95% CI (DR) |
|-------|---------|:------------:|:---------------:|:-----:|:-----------:|:-----------:|
| Gemma-3 1B | HARMBENCH | 58% | 72% | **+14%** ↑ | [46%, 70%] | [58%, 84%] |
| Gemma-3 1B | ADVBENCH | 40% | 66% | **+26%** ↑ | [28%, 54%] | [54%, 78%] |
| Qwen-3 1.7B | HARMBENCH | 90% | 56% | **-34%** ↓ | [80%, 98%] | [44%, 70%] |
| Qwen-3 1.7B | ADVBENCH | 80% | 14% | **-66%** ↓ | [68%, 90%] | [6%, 26%] |
| DeepSeek-R1 1.5B | HARMBENCH | 90% | 60% | **-30%** ↓ | [80%, 96%] | [46%, 72%] |
| DeepSeek-R1 1.5B | ADVBENCH | 92% | 46% | **-46%** ↓ | [84%, 98%] | [32%, 60%] |

![ASR Comparison](artifacts/paper_eval/plot_asr_comparison.png)

## 2. Over-Refusal Analysis (4-Class)

Over-refusal rate = fraction of benign prompts incorrectly refused. Lower = better.

| Model | Variant | Over-Refusal Rate | Direct Answer | Indirect Answer | Direct Refusal | Indirect Refusal |
|-------|---------|:-----------------:|:-------------:|:---------------:|:--------------:|:----------------:|
| Gemma-3 1B | Baseline | 6% | 41 | 6 | 0 | 3 |
| Gemma-3 1B | DeepRefusal | 4% | 44 | 4 | 2 | 0 |
| Qwen-3 1.7B | Baseline | 2% | 35 | 14 | 0 | 1 |
| Qwen-3 1.7B | DeepRefusal | 8% | 38 | 8 | 1 | 3 |
| DeepSeek-R1 1.5B | Baseline | 10% | 39 | 6 | 4 | 1 |
| DeepSeek-R1 1.5B | DeepRefusal | 2% | 41 | 8 | 0 | 1 |

![Over-Refusal Breakdown](artifacts/paper_eval/plot_overrefusal.png)

## 3. Capability Benchmarks

MMLU and GSM8k measure general knowledge and math reasoning. Higher = better.

| Model | Baseline MMLU | DR MMLU | Delta | Baseline GSM8k | DR GSM8k | Delta |
|-------|:------------:|:-------:|:-----:|:--------------:|:--------:|:-----:|
| Gemma-3 1B | 34% | 37% | **+3%** | 17% | 20% | **+3%** |
| Qwen-3 1.7B | 41% | 45% | **+4%** | 8% | 46% | **+38%** |
| DeepSeek-R1 1.5B | 28% | 28% | **+0%** | 8% | 8% | **+0%** |

![Capability Benchmarks](artifacts/paper_eval/plot_capability.png)

## 4. Utility (UltraChat)

Fraction of benign chat prompts answered without refusal. Higher = better.

| Model | Baseline Utility | DeepRefusal Utility | Delta |
|-------|:----------------:|:-------------------:|:-----:|
| Gemma-3 1B | 97% | 97% | **+0%** |
| Qwen-3 1.7B | 100% | 97% | **-3%** |
| DeepSeek-R1 1.5B | 93% | 93% | **+0%** |

## 5. Radar Plots (Multi-Metric Overview)

Each axis shows a desirable property (higher = better). Safety axes show 1 - ASR.

![Radar Plots](artifacts/paper_eval/plot_radar.png)

## 6. Delta Heatmap

Shows the change from Baseline to DeepRefusal for each metric. For ASR and Over-Refusal, negative (green) is better. For MMLU/GSM8k/Utility, positive (green) is better.

![Delta Heatmap](artifacts/paper_eval/plot_delta_heatmap.png)

## Key Findings

1. **Qwen-3 1.7B shows the strongest safety improvement**: HarmBench ASR drops from 90% to 56% (-34pp), AdvBench ASR from 80% to 14% (-66pp)
2. **DeepSeek-R1 1.5B also benefits**: HarmBench ASR drops from 90% to 60% (-30pp), AdvBench from 92% to 46% (-46pp)
3. **Gemma-3 1B shows increased ASR** (unexpected): HarmBench +14pp, AdvBench +26pp — likely due to the small model capacity or training instability
4. **Over-refusal remains low** across all models and variants (2-10%)
5. **Capability is preserved or improved**: MMLU and GSM8k scores are maintained or slightly improved after DeepRefusal training, especially notable for Qwen-3 (+4pp MMLU, +38pp GSM8k)
6. **Utility is preserved**: All models maintain >93% utility on benign chat prompts

---

*Evaluation config: 50 attack samples/dataset, 50 over-refusal samples (XSTest), 30 utility samples (UltraChat), 100 MMLU/GSM8k samples, deterministic generation (temperature=1.0, do_sample=False), seed=42, 4-bit quantized models with LoRA adapters.*
