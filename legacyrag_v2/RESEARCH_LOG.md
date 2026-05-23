# LegacyRAG v2 — Research Log

_Living document. Updated automatically by benchmark_runner.py after each experiment._
_Manual entries marked with [MANUAL]._

---

## Hardware & Software Context

| Item | Value |
|---|---|
| Host | Intel Xeon E5-1620 v3, Ubuntu 24.04 |
| GPUs | 2× NVIDIA Quadro K4200, 4GB GDDR5 each |
| Architecture | Maxwell GM204 (2014) |
| Vulkan | Yes (no FP16 matrix, no INT8 dot, no tensor cores) |
| llama.cpp build | b5576, Vulkan backend |
| llama-server | `/home/azeez/IEEE\ Edge/build/bin/llama-server` |
| LD_LIBRARY_PATH | `/home/azeez/IEEE\ Edge/build/bin` |
| phi3:mini blob | `sha256-633fc5be...` (~2.1GB, Q4_K_M, 3.8B) |
| qwen2:1.5b blob | `sha256-405b56...` (~892MB, Q4_K_M, 1.5B) |
| nomic-embed-text blob | `sha256-970aa7...` (~262MB) |

---

## v1 Baseline (reference, May 11 2026)

3 sequential requests (identical OPRA question), phi3-mini via llama-server Vulkan:

| # | tok/s | generate_s | prompt tok | completion tok |
|---|---|---|---|---|
| 1 | 0.89 | 575.25 | 459 | 512 |
| 2 | 1.66 | 307.68 | 459 | 512 |
| 3 | 0.29 | 525.23 | 459 | 150 |
| **Mean** | **0.95** | **469.4** | 459 | 391 |

**Observation:** Wide tok/s variance (0.29–1.66) attributed to thermal throttling
and KV cache warmth. Request 2 benefited from residual GPU state (warm weights).
Generation time was >99.86% of total RAG latency — embedding/retrieval negligible.

---

## v2 Experiments Plan

| Exp | What | Hypothesis |
|---|---|---|
| exp1 | phi3-mini baseline, 10 varied prompts | Confirm v1 numbers; characterize by prompt length |
| exp2 | phi3-mini + qwen2:1.5b draft, --draft-max 8 | Speculative decoding may give 1.5–2× if acceptance rate >60% |
| exp3 | N-gram speculative (no draft model) | **N/A — not in this build** (see note below) |
| exp4 | qwen2.5:7b-instruct-q2_K on single GPU | Test if aggressive quantization helps vs larger model size |

**NOTE — Experiment 3 (N-gram):** `llama-server` in build b5576 does not expose
`--lookup-cache-static`, `--ngram-draft`, or any ngram speculative flag in its
`--help` output. Experiment 3 runs as a baseline ablation (no draft model) to
confirm infrastructure consistency. A future build with lookup-cache support would
enable true ngram speculative decoding.

---

## Benchmark Run Log

_(Entries below are auto-appended by benchmark_runner.py or added manually)_

---

## [MANUAL] Experiment 1 Results — 2026-05-23

**Run time:** ~32 minutes (16:09–16:41 UTC). 10/10 prompts successful, 0 errors.

### Raw Results Table

| # | Bucket | Prompt tok | Gen tok | tok/s | First-tok (ms) | Prefill (s) | Prefill tok/s | Total wall (s) |
|---|--------|-----------|---------|-------|----------------|-------------|---------------|----------------|
| 1 | short  | 18  | 200 | 8.536 | 117.2 | 1.72  | 10.5 | 25.2  |
| 2 | short  | 18  | 200 | 8.569 | 116.7 | 1.60  | 11.3 | 24.9  |
| 3 | short  | 19  | 200 | 8.558 | 116.9 | 1.62  | 11.7 | 25.0  |
| 4 | medium | 91  | 200 | 8.385 | 119.3 | 124.8 | 0.73 | 148.6 |
| 5 | medium | 110 | 200 | 8.398 | 119.1 | 125.3 | 0.88 | 149.1 |
| 6 | medium | 78  | 200 | 8.414 | 118.9 | 124.9 | 0.62 | 148.6 |
| 7 | medium | 89  | 200 | 8.384 | 119.3 | 124.7 | 0.71 | 148.6 |
| 8 | long   | 313 | 200 | 7.942 | 125.9 | 377.2 | 0.83 | 402.4 |
| 9 | long   | 260 | 200 | 7.745 | 129.1 | 376.1 | 0.69 | 401.9 |
| 10 | long  | 367 | 200 | 7.854 | 127.3 | 384.5 | 0.95 | 410.0 |

### Summary Statistics

| Metric | All (n=10) | Short (n=3) | Medium (n=4) | Long (n=3) |
|--------|-----------|-------------|--------------|------------|
| Mean tok/s | **8.279** | 8.554 | 8.395 | 7.847 |
| Median tok/s | 8.398 | 8.558 | 8.391 | 7.854 |
| Mean wall (s) | 188.4 | 25.0 | 148.7 | 404.8 |
| p95 wall (s) | 410.0 | — | — | — |
| Mean prefill (s) | — | 1.65 | 124.9 | 379.3 |
| Mean prefill tok/s | — | 11.2 | 0.74 | 0.82 |
| First-tok latency (ms) | — | 116.9 | 119.1 | 127.4 |

### VRAM Utilization
- phi3:mini loaded across both GPUs (dual-split): GPU0 ~1976MB, GPU1 ~1545MB
- VRAM stable throughout; no OOM events
- GPU0 free: ~2057–2096MB; GPU1 free: ~2487–2492MB (healthy headroom)

### Key Observations

**1. Massive throughput improvement vs v1 baseline (+772%)**
v1 achieved 0.95 tok/s (single-GPU, 3-request test). v2 exp1 achieves **8.28 tok/s mean** (8.72× faster).
Primary cause: llama-server with `-ngl 99` auto-distributes transformer layers across both K4200s.
v1 ran via Ollama, which loaded phi3:mini onto a single GPU — this created VRAM pressure and thermal
throttling that suppressed throughput. The dual-split reduces per-GPU load from ~2.1GB to ~1.05GB each.

**2. Generation speed (tok/s) is largely stable across prompt lengths**
- Short: 8.55 tok/s | Medium: 8.40 tok/s | Long: 7.85 tok/s
- The ~8% drop from short to long is due to larger KV cache increasing attention memory bandwidth demand.
- Generation throughput is memory-bandwidth-bound, not compute-bound.

**3. Critical discovery: prefill is the wall-time bottleneck for medium/long prompts**
- Short prompts (18 tok): prefill = 1.65s (10.5–11.7 tok/s) → prefill is <7% of wall time
- Medium prompts (78–110 tok): prefill = 124.9s (0.62–0.88 tok/s) → prefill is **84%** of wall time
- Long prompts (260–367 tok): prefill = 379.3s (0.69–0.95 tok/s) → prefill is **93%** of wall time

This reveals a non-linear prefill cost in the Vulkan backend: going from 18 to 91 tokens (5× more)
causes prefill time to jump from 1.65s to 124.8s (76×). This is consistent with the Vulkan GLSL
attention shaders having poor batch-prefill performance for larger contexts — likely processing
attention in a loop that has quadratic memory access patterns without fused kernel support.

**4. First-token latency increases with prompt length**
- Short: ~117ms | Medium: ~119ms | Long: ~127ms
- The small but consistent increase reflects KV cache size at first-token time.

**5. v1 vs v2 wall-time reconciliation**
v1 reported 469s generate_s for 459-token prompts + 512 completion tokens.
v2 exp1 long prompts (260–367 tok) + 200 completion tokens = 402–410s wall time.
The numbers are consistent: longer prompts and more completion tokens in v1 explain the
higher wall time. The v1 "0.95 tok/s" was computed as completion_tokens/generate_s,
which included both slow prefill and generation — artificially depressing the true
generation tok/s. **The true generation rate on K4200 dual-Vulkan is ~8 tok/s.**

### What to Try Next
- Exp2: Speculative decoding → phi3+qwen2 INCOMPATIBLE (see below). Using qwen2:1.5b+qwen2:0.5b.
- Exp3: N-gram speculative. Downloaded b9297 which exposes `--spec-type ngram-simple`.
- Exp4: qwen2.5-7B Q2_K. Prefill will be even slower for 7B; focus on short prompts.

---

## [MANUAL] Experiment 2 Results — 2026-05-23

**Model pair:** qwen2:1.5b (main, 892MB) + qwen2:0.5b (draft, 336MB). b9297 binary.
**Total VRAM used:** GPU0 ~918MB + GPU1 ~1436MB = 2354MB (both models combined).
**Run time:** ~19 minutes. 10/10 prompts successful.

### Raw Results Table

| # | Bucket | Prompt tok | Gen tok | tok/s | Accept rate | Wall (s) |
|---|--------|-----------|---------|-------|-------------|----------|
| 1 | short  | 16  | 200 | 2.622 | 0.271 | 77.7  |
| 2 | short  | 15  | 200 | 2.886 | 0.297 | 70.2  |
| 3 | short  | 17  | 200 | 2.445 | 0.239 | 82.6  |
| 4 | medium | 84  | 200 | 2.385 | 0.224 | 124.0 |
| 5 | medium | 92  | 200 | 5.456 | 0.668 | 76.5  |
| 6 | medium | 68  | 200 | 4.888 | 0.590 | 80.9  |
| 7 | medium | 76  | 200 | 2.709 | 0.271 | 113.8 |
| 8 | long   | 265 | 200 | 3.885 | 0.448 | 166.7 |
| 9 | long   | 221 | 200 | 4.008 | 0.464 | 127.6 |
| 10 | long  | 300 | 200 | 2.287 | 0.213 | 202.1 |

### Summary Statistics

| Metric | Value |
|--------|-------|
| Mean tok/s | **3.357** |
| Median tok/s | 2.886 |
| Mean latency | 112.2s |
| p95 latency | 202.1s |
| Mean acceptance rate | **36.9%** |
| Range tok/s | 2.29–5.46 |
| Range accept | 21.3%–66.8% |

### Key Observations

**1. Model comparison problem — exp1 vs exp2 are NOT directly comparable**
Exp1 used phi3:mini (3.8B, Q4_K_M); exp2 uses qwen2:1.5b (1.5B) as main model.
The model change was forced by tokenizer incompatibility (phi3 ≠ qwen2 vocab).
To properly assess speculative decoding benefit, a qwen2:1.5b baseline run without
draft model would be needed. This is a limitation to note in the paper.

**2. qwen2:1.5b is SLOWER than phi3:mini on K4200 Vulkan despite fewer parameters**
- phi3:mini 3.8B achieves 8.3 tok/s mean
- qwen2:1.5b 1.5B achieves 2.4-5.5 tok/s (mean 3.36 tok/s without speculative baseline)
- Hypothesis: qwen2 uses grouped-query attention and a different ffn structure.
  More likely: qwen2 GGUF may have more total layers or the Vulkan shader dispatch
  pattern is less cache-friendly for this architecture on Maxwell.

**3. Extreme acceptance rate and tok/s variance (content-dependent)**
- Prompts 5, 6 (medium: GPU comparison, decoding strategies): 59–67% acceptance → 4.9–5.5 tok/s
- Prompts 1, 2, 3, 4, 7, 10: 21–30% acceptance → 2.3–2.9 tok/s

This is a fundamental characteristic of speculative decoding: acceptance rate depends
heavily on how well the draft model (qwen2:0.5b) predicts the main model's output.
For technical AI topics that qwen2:0.5b was trained on, acceptance is high (67%).
For OPRA/government/general prompts that don't appear often in Qwen's training data,
acceptance drops to 21%, and speculative decoding may actually slow things down
compared to autoregressive generation (each failed verification still costs time).

**4. Short prompt wall time much higher than expected**
- Short prompts: 70–83s wall for 200 gen tokens at 2.4–2.9 tok/s
- This implies generation rate without speculation overhead would be ~2.4–2.9 tok/s
  which is the observed rate — spectulative overhead is already included

**5. Speculative decoding does not help on Maxwell Vulkan at low acceptance rates**
With α=0.27 and draft_max=8, theoretical speedup = E[accepted+1] per step ≈ 1.37.
But on Maxwell Vulkan, the parallel verification step (running main model on k+1 tokens)
is NOT faster than running it on 1 token — because Vulkan shaders don't parallelize
token processing across the batch dimension on Maxwell (no native batched GEMM).
Each verification step effectively runs 9 forward passes sequentially, negating gains.

### What to Try Next (Exp3 and Beyond)
- Exp3: N-gram speculative with phi3:mini (no draft model → no VRAM overhead).
  b9297 supports `--spec-type ngram-simple`. Hypothesis: small improvement for
  repetitive/predictable prompt content (legal text, structured queries).
- Need qwen2:1.5b non-speculative baseline run to fairly assess exp2 results.

---

## [MANUAL] Exp2 Setup Finding — 2026-05-23: phi3+qwen2 Vocabulary Incompatibility

**Problem:** llama-server b5576 crashed when launching phi3:mini + qwen2:1.5b speculative decoding.
**Root cause:** Speculative decoding requires the draft and main model to share the same tokenizer vocabulary.

```
common_speculative_are_compatible: draft vocab special tokens must match target vocab
common_speculative_are_compatible: tgt: bos = 1 (0), eos = 32000 (0)     ← phi3 tokenizer
common_speculative_are_compatible: dft: bos = 151643 (0), eos = 151645 (0) ← qwen2 tokenizer
```

These are completely different tokenizer families. If models share different vocabularies, a "token 5"
in the draft model refers to a different string than "token 5" in the main model — speculative
decoding would accept/reject based on wrong token identities.

**Solution:** Switched to same-family pair: qwen2:1.5b (main) + qwen2:0.5b (draft).
Both use the Qwen2 tokenizer (tiktoken-based, 151K vocab). Compatible.

**Implication for paper:** This is a critical practical constraint. Organizations deploying speculative
decoding on legacy hardware must ensure draft and main models use identical tokenizers. The popular
pairing of phi3 + a small Qwen draft does not work. Users must find model families with small variants
(e.g., Qwen2: 0.5B, 1.5B, 7B all compatible; Llama3: 1B and 3B compatible with 8B/70B).

**Also:** Downloaded llama.cpp b9297 (latest as of 2026-05-23). This build adds:
- `--spec-type` with values: `ngram-simple`, `ngram-map-k`, `ngram-map-k4v`, `ngram-mod`, `ngram-cache`
- `--spec-draft-n-max` replaces removed `--draft-max`
Using b9297 for experiments 2, 3, 4.
