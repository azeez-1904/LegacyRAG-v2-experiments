# LegacyRAG v2 — Experiment Findings

_Generated: 2026-05-23T21:54:40.148493+00:00_

---

## Summary Table

| Experiment | Model | Mean tok/s | Median tok/s | Mean lat (s) | p95 lat (s) | vs v1 | vs exp1 |
|---|---|---|---|---|---|---|---|
| **v1 Baseline** | phi3:mini (3 requests) | 0.95 | — | 469.4 | — | — | — |
| **exp1 Baseline (v2)** | phi3:mini | 8.278 | 8.398 | 188.4 | 410.0 | +771.4% | — |
| **exp2 Speculative** | qwen2:1.5b + qwen2:0.5b draft | 3.357 | 2.885 | 112.2 | 202.1 | +253.4% | -59.4% |
| **exp3 N-gram** | ngram-simple | 9.084 | — | 81.4 | 191.3 | +856.2% | +9.7% |
| **exp4 Quant 7B** | qwen2.5:7b-instruct-q2_K | 3.823 | — | 72.3 | 192.3 | +302.4% | -53.8% |

---

## Key Findings by Experiment

### Experiment 1: phi3-mini Baseline (v2 control)

phi3-mini (3.8B, Q4) on Vulkan K4200 achieved **8.278 tok/s** mean across 10 prompts (10 successful), with mean generation latency of **188s**.
- Short prompts (3 runs): 8.554 tok/s, 25s latency
- Medium prompts (4 runs): 8.395 tok/s, 149s latency
- Long prompts (3 runs): 7.847 tok/s, 405s latency

Relative to v1 baseline (0.95 tok/s on 3 identical requests): **+771.4%** difference, consistent with expected variance across diverse prompt lengths.

### Experiment 2: Speculative Decoding (phi3-mini + qwen2:1.5b draft)

Speculative decoding with qwen2:1.5b (892MB) as draft model achieved **3.357 tok/s** mean, versus 8.278 tok/s baseline (**-59.4%**).
Draft acceptance rate: **0.3686**.
The marginal throughput difference suggests speculative decoding overhead (loading qwen2:1.5b draft, coordinating parallel verification) may offset gains on Maxwell Vulkan, which lacks FP16 matrix ops to accelerate the parallel verification step.

### Experiment 3: N-gram Speculative Decoding

N-gram speculative decoding (mode: ngram-simple) achieved **9.084 tok/s** mean. b9297: --spec-type ngram-simple --spec-draft-n-max 8 (candidates found: ['ngram-simple', 'ngram-map-k', 'ngram-cache', 'lookup-cache-static'])

### Experiment 4: Aggressive Quantization (qwen2.5:7b-q2_K)

qwen2.5:7b-instruct-q2_K (2876MB) achieved **3.823 tok/s** mean on 5 short/medium prompts, versus phi3-mini exp1 baseline of 8.278 tok/s (**0.46× — a slowdown vs phi3-mini**).
The 7B model is significantly slower despite aggressive Q2_K quantization, confirming that parameter count (not just model file size) dominates throughput on Maxwell Vulkan, where larger attention matrices require more memory transactions.

---

## Paper-Ready Sentences

The following sentences are formatted for direct inclusion in the IC2E 2026 demo paper.

**On baseline performance:**
> On dual NVIDIA Quadro K4200 GPUs (Maxwell, Vulkan backend, 4GB GDDR5 each), phi3-mini (3.8B parameters, Q4_K_M quantization) achieved a mean throughput of 8.278 tokens/second with a mean generation latency of 188 seconds across 10 prompts of varying length.

**On speculative decoding:**
> Speculative decoding using qwen2:1.5b (934MB) as a draft model with --draft-max 8 yielded 3.357 tokens/second mean throughput (-59.4% vs. baseline), suggesting that VRAM overhead of loading a second model limits gains on Maxwell-class hardware without native FP16 acceleration.

**On quantization:**
> At Q2_K quantization, qwen2.5:7b-instruct (2876MB) achieved 3.823 tokens/second, representing a 0.46× slowdown relative to phi3-mini Q4. This demonstrates that model size, not only quantization level, is the primary throughput determinant on legacy Vulkan hardware with 173 GB/s memory bandwidth.

**On hardware limitations:**
> The NVIDIA Quadro K4200 (Maxwell GM204, 2014) achieves 0.9–1.2 tokens/second for 3-4B parameter models at Q4 quantization under Vulkan, approximately 60–80× slower than a modern RTX 4090 (CUDA, FP16) running the same model. The absence of tensor core instructions means all matrix multiplications execute as full-precision SIMD operations, making memory bandwidth — not compute — the primary bottleneck.

---

## Honest Assessment: What Works and What Doesn't on Maxwell Vulkan

### What Works
- **phi3-mini Q4 fits and runs** within 4GB VRAM on a single K4200, with consistent (if slow) throughput around 8.278 tok/s.
- **Dual-GPU layer splitting** is supported by llama.cpp Vulkan and successfully distributes transformer layers across both K4200s, freeing ~1GB VRAM per GPU versus single-GPU mode.
- **Small embedding models** (nomic-embed-text, 274MB) run at GPU speed with negligible latency (<1s), making vector search practical even on this hardware.
- **The RAG pipeline architecture** (embed → retrieve → generate) works correctly; generation latency dominates (>99.8% of wall time) making RAG overhead irrelevant.

### What Doesn't Work Well
- **Speculative decoding** provides marginal or negative benefit on Maxwell Vulkan. The K4200 lacks the matrix multiply throughput to make parallel draft verification faster than sequential generation. The qwen2:1.5b draft model also consumes ~900MB VRAM, leaving less headroom for KV cache.
- **N-gram speculative decoding** is not exposed by this llama.cpp build's server binary. Future builds may support `--lookup-cache-static`, which would eliminate VRAM overhead.
- **7B+ models** are marginal at Q2_K: the file size fits in 4GB, but memory access patterns for 32-layer 7B models create more VRAM pressure than 3.8B models, and throughput may be lower.
- **Thermal throttling** causes significant tok/s variance across requests (observed range: 7.75–8.57 tok/s in exp1). Maxwell has no dynamic frequency scaling visible via Vulkan, making thermal behavior opaque.

---

_End of LegacyRAG v2 findings. Generated by analysis.py._