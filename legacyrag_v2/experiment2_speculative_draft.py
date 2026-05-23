#!/usr/bin/env python3
"""
Experiment 2: Speculative decoding with compatible vocab pair.

Original design: phi3:mini main + qwen2:1.5b draft.
INCOMPATIBLE: phi3 uses tokenizer bos=1/eos=32000; qwen2 uses bos=151643/eos=151645.
llama.cpp speculative decoding requires matching vocab between draft and main.

Updated design: qwen2:1.5b (main) + qwen2:0.5b (draft) — same Qwen tokenizer family.
Using b9297 binary (--spec-draft-n-max replaces removed --draft-max flag).
"""

import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
BIN_DIR = PROJECT_ROOT / "build_b9297"  # b9297 required: --spec-draft-n-max flag
LLAMA_SERVER = BIN_DIR / "llama-server"
SERVER_PORT = 8080
SERVER_URL = f"http://127.0.0.1:{SERVER_PORT}"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_FILE = RESULTS_DIR / "exp2_speculative.json"
MAX_TOKENS = 200
SERVER_TIMEOUT = 150
LIB_PATH = str(BIN_DIR)
BLOBS_DIR = Path("/usr/share/ollama/.ollama/models/blobs")


def find_model_gguf(manifest_path: str) -> Path:
    manifest = Path(
        "/usr/share/ollama/.ollama/models/manifests"
        f"/registry.ollama.ai/library/{manifest_path}"
    )
    with open(manifest) as f:
        data = json.load(f)
    for layer in data["layers"]:
        if layer["mediaType"] == "application/vnd.ollama.image.model":
            digest = layer["digest"].replace("sha256:", "sha256-")
            return BLOBS_DIR / digest
    raise FileNotFoundError(f"No model layer in manifest: {manifest_path}")


def get_vram() -> list[dict]:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.free,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        result = []
        for line in out.strip().splitlines():
            idx, free, used, total = [x.strip() for x in line.split(",")]
            result.append(
                {
                    "gpu": int(idx),
                    "free_mb": float(free),
                    "used_mb": float(used),
                    "total_mb": float(total),
                }
            )
        return result
    except Exception as e:
        return [{"error": str(e)}]


def start_server_speculative(
    main_model: Path, draft_model: Path
) -> subprocess.Popen:
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = LIB_PATH + ":" + env.get("LD_LIBRARY_PATH", "")

    cmd = [
        str(LLAMA_SERVER),
        "-m", str(main_model),
        "-ngl", "99",
        "-md", str(draft_model),
        "-ngld", "99",
        "--spec-draft-n-max", "8",  # b9297: --draft-max removed
        "--port", str(SERVER_PORT),
        "--host", "127.0.0.1",
        "--ctx-size", "2048",
        "--threads", "4",
        "--parallel", "1",
        "--log-disable",
    ]

    log_path = RESULTS_DIR / "server_exp2.log"
    log_file = open(log_path, "a")
    log_file.write(f"\n\n=== Exp2 Server start {datetime.now(timezone.utc).isoformat()} ===\n")
    log_file.write(f"CMD: {' '.join(cmd)}\n")
    log_file.flush()

    proc = subprocess.Popen(cmd, env=env, stdout=log_file, stderr=log_file)
    print(f"  Server PID {proc.pid}, waiting for health...", flush=True)

    deadline = time.time() + SERVER_TIMEOUT
    while time.time() < deadline:
        time.sleep(2)
        try:
            import urllib.request
            with urllib.request.urlopen(f"{SERVER_URL}/health", timeout=3) as r:
                if r.status == 200:
                    print("  Server healthy.", flush=True)
                    log_file.close()
                    return proc
        except Exception:
            pass
        if proc.poll() is not None:
            log_file.close()
            raise RuntimeError(
                f"llama-server exited early (code {proc.returncode}). Check {log_path}"
            )

    proc.terminate()
    log_file.close()
    raise TimeoutError(f"Server did not become healthy within {SERVER_TIMEOUT}s")


def stop_server(proc: subprocess.Popen, sleep_s: int = 30) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    print(f"  Server stopped. Sleeping {sleep_s}s for VRAM to clear...", flush=True)
    time.sleep(sleep_s)


def run_completion(prompt: str) -> tuple[dict, float]:
    import urllib.request
    payload = json.dumps(
        {
            "prompt": prompt,
            "n_predict": MAX_TOKENS,
            "temperature": 0.1,
            "stop": ["</s>", "<|end|>"],
            "stream": False,
        }
    ).encode()

    req = urllib.request.Request(
        f"{SERVER_URL}/completion",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=900) as r:
        raw = r.read()
    elapsed = time.perf_counter() - t0
    return json.loads(raw), elapsed


def get_draft_stats() -> dict:
    """Try to get draft acceptance stats from /slots endpoint."""
    try:
        import urllib.request
        with urllib.request.urlopen(f"{SERVER_URL}/slots", timeout=5) as r:
            return json.loads(r.read())
    except Exception:
        return {}


# Same 10 prompts as experiment 1
PROMPTS = [
    {"id": 1, "bucket": "short",
     "text": "What is speculative decoding in large language models and why does it improve inference speed?"},
    {"id": 2, "bucket": "short",
     "text": "How does post-training quantization reduce neural network memory footprint while preserving accuracy?"},
    {"id": 3, "bucket": "short",
     "text": "What are the main hardware limitations that make running large language models on consumer GPUs challenging?"},
    {"id": 4, "bucket": "medium",
     "text": (
         "Explain how retrieval-augmented generation (RAG) works and why it is useful for enterprise applications. "
         "Describe the role of embedding models and vector stores in a RAG pipeline. "
         "How is retrieved context injected into the prompt, and what are the main failure modes when the retrieved "
         "chunks are irrelevant or too long? Provide a concrete example with a government records use case where "
         "citizens query a public records database using natural language."
     )},
    {"id": 5, "bucket": "medium",
     "text": (
         "Compare the performance characteristics of modern NVIDIA Ampere or Ada Lovelace GPUs versus legacy Maxwell "
         "architecture GPUs when running transformer model inference. What specific hardware features are absent in "
         "Maxwell that limit throughput? Consider FP16 tensor cores, INT8 dot product support, and memory bandwidth. "
         "How does the NVIDIA Quadro K4200 specifically perform relative to these newer architectures? "
         "What is the practical implication for an organization running open-source LLMs on older hardware?"
     )},
    {"id": 6, "bucket": "medium",
     "text": (
         "What is the difference between greedy decoding, beam search, and temperature-based sampling in language "
         "model text generation? How does each strategy affect output diversity, factual accuracy, and token "
         "generation speed on resource-constrained hardware? Explain the relationship between inference speed "
         "and decoding strategy for a model running at under 2 tokens per second on a legacy Vulkan GPU."
     )},
    {"id": 7, "bucket": "medium",
     "text": (
         "Describe the Open Public Records Act (OPRA) in New Jersey. What obligations does it place on government "
         "agencies regarding document disclosure? How many business days do agencies have to respond to a request? "
         "What categories of records are exempt from disclosure, and what is the role of the Government Records "
         "Council in adjudicating disputes? How might an AI-assisted retrieval system improve compliance efficiency?"
     )},
    {"id": 8, "bucket": "long",
     "text": (
         "You are a research assistant preparing a technical analysis of GPU hardware suitability for large language "
         "model inference. Write a detailed analysis of the NVIDIA Quadro K4200's capabilities and limitations.\n\n"
         "The K4200 is based on the Maxwell architecture (GM204), featuring 1344 CUDA cores, 4GB of GDDR5 VRAM "
         "with 173 GB/s memory bandwidth, and Vulkan 1.3 support via the open-source Mesa/RADV stack. "
         "It lacks hardware FP16 matrix operations, INT8 dot product instructions, and tensor core accelerators "
         "found in Volta and later architectures.\n\n"
         "Address the following in your analysis:\n"
         "1. How do these architectural limitations affect transformer inference throughput?\n"
         "2. What quantization strategies (Q4_K_M, Q2_K, Q8_0) are most practical given 4GB VRAM?\n"
         "3. What is the theoretical maximum tokens per second given the 173 GB/s memory bandwidth constraint "
         "for a 3.8B parameter model at Q4 quantization?\n"
         "4. How does running two such GPUs in a tensor-split configuration via llama.cpp affect throughput?\n"
         "5. What is the practical use case for such hardware in 2025, given that modern alternatives exist?\n\n"
         "Provide specific numbers and calculations where possible."
     )},
    {"id": 9, "bucket": "long",
     "text": (
         "Write a comprehensive technical overview of speculative decoding techniques for accelerating large language "
         "model inference, suitable for inclusion in an IEEE conference paper.\n\n"
         "Cover the following topics in depth:\n"
         "1. The basic mechanism: how a small draft model proposes token sequences and the main model verifies them "
         "in a single forward pass, achieving parallel verification.\n"
         "2. The acceptance rate alpha and how it determines actual speedup versus theoretical maximum speedup. "
         "Derive the relationship between acceptance rate and the expected number of accepted tokens per draft step.\n"
         "3. N-gram speculative decoding as a draft-model-free alternative: how it uses previously generated "
         "context to predict likely continuations without a second model.\n"
         "4. Trade-offs between draft model size, acceptance quality, and VRAM overhead on memory-constrained "
         "devices with 4-8GB total VRAM.\n"
         "5. How speculative decoding performs differently when the main model is GPU-resident versus CPU-offloaded.\n"
         "6. Current research challenges: applying speculative decoding to heavily quantized (Q2-Q4) models where "
         "the draft and main model token distributions may diverge significantly.\n\n"
         "Include references to relevant literature where appropriate."
     )},
    {"id": 10, "bucket": "long",
     "text": (
         "Analyze the following deployment scenario for an AI-assisted public records retrieval system:\n\n"
         "A mid-size New Jersey municipal government receives approximately 500 OPRA requests per month. "
         "Staff currently spend 2-4 hours per request manually searching physical archives and scanned PDFs. "
         "The IT department has two NVIDIA Quadro K4200 GPUs (4GB VRAM each, Maxwell architecture, Vulkan-only) "
         "available in an existing on-premises server. No new hardware procurement is budgeted. "
         "Staff have no machine learning expertise. Legal compliance requires all data to remain on-premises.\n\n"
         "Design a complete technical architecture for this system addressing:\n"
         "1. Model selection: which open-source LLM and embedding model fit within 8GB combined VRAM?\n"
         "2. Document ingestion pipeline: how to chunk, embed, and index scanned government records.\n"
         "3. Query handling: how the RAG pipeline processes citizen queries and generates OPRA-compliant summaries.\n"
         "4. Expected performance: given benchmark data showing 0.95 tok/s mean throughput on phi3-mini Q4 "
         "with 469s mean generation latency, what is the realistic daily capacity of this system?\n"
         "5. Bottleneck analysis: which components limit throughput and what optimizations are feasible?\n"
         "6. Staff workflow integration: how would municipal employees interact with and validate AI responses?\n"
         "7. Risk assessment: what are the failure modes specific to legacy Vulkan GPU hardware?\n\n"
         "Be specific and pragmatic — this is a real deployment constraint, not a theoretical exercise."
     )},
]


def run_experiment() -> None:
    print("=" * 60)
    print("Experiment 2: Speculative Decoding (qwen2:1.5b main + qwen2:0.5b draft)")
    print("NOTE: phi3+qwen2 incompatible (different tokenizers). Using same-family pair.")
    print("=" * 60)

    RESULTS_DIR.mkdir(exist_ok=True)

    qwen2_main_path = find_model_gguf("qwen2/1.5b")
    qwen2_draft_path = find_model_gguf("qwen2/0.5b")
    print(f"Main model (qwen2:1.5b): {qwen2_main_path}")
    print(f"Draft model (qwen2:0.5b): {qwen2_draft_path}")

    vram_pre_server = get_vram()
    print(f"VRAM before server start: {vram_pre_server}")

    print("\nStarting llama-server with speculative decoding (qwen2:1.5b + qwen2:0.5b draft)...")
    try:
        server_proc = start_server_speculative(qwen2_main_path, qwen2_draft_path)
    except Exception as e:
        output = {
            "experiment": "exp2_speculative",
            "error": f"Server failed to start: {e}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        with open(RESULTS_FILE, "w") as f:
            json.dump(output, f, indent=2)
        print(f"FATAL: {e}")
        return

    time.sleep(3)

    results = []
    for p in PROMPTS:
        print(f"\n[{p['id']}/10] bucket={p['bucket']}")
        vram_before = get_vram()
        t_start = datetime.now(timezone.utc).isoformat()

        try:
            resp, wall_time = run_completion(p["text"])

            timings = resp.get("timings", {})
            tokens_eval = timings.get("prompt_n", 0) or resp.get("tokens_evaluated", 0)
            tokens_pred = timings.get("predicted_n", 0) or resp.get("tokens_predicted", 0)
            toks_per_sec = timings.get("predicted_per_second", None)
            if toks_per_sec is None and tokens_pred > 0 and wall_time > 0:
                toks_per_sec = tokens_pred / wall_time

            # Draft acceptance info — available in timings if server exposes it
            draft_accepted = timings.get("draft_n_accepted", None)
            draft_total = timings.get("draft_n", None)
            acceptance_rate = (
                draft_accepted / draft_total
                if draft_accepted is not None and draft_total and draft_total > 0
                else None
            )

            vram_after = get_vram()
            record = {
                "id": p["id"],
                "bucket": p["bucket"],
                "timestamp": t_start,
                "prompt_words": len(p["text"].split()),
                "tokens_evaluated": tokens_eval,
                "tokens_predicted": tokens_pred,
                "tok_per_sec": round(toks_per_sec, 4) if toks_per_sec else None,
                "total_latency_s": round(wall_time, 3),
                "draft_tokens_accepted": draft_accepted,
                "draft_tokens_total": draft_total,
                "draft_acceptance_rate": round(acceptance_rate, 4) if acceptance_rate is not None else None,
                "vram_before": vram_before,
                "vram_after": vram_after,
                "timings": timings,
                "error": None,
            }
            print(
                f"  tok/s={record['tok_per_sec']}  "
                f"draft_accept={acceptance_rate}  wall={wall_time:.1f}s",
                flush=True,
            )
        except Exception as e:
            vram_after = get_vram()
            record = {
                "id": p["id"],
                "bucket": p["bucket"],
                "timestamp": t_start,
                "prompt_words": len(p["text"].split()),
                "error": str(e),
                "vram_before": vram_before,
                "vram_after": vram_after,
            }
            print(f"  ERROR: {e}", flush=True)

        results.append(record)

    vram_post = get_vram()
    stop_server(server_proc, sleep_s=30)

    good = [r for r in results if r.get("tok_per_sec") is not None]
    summary = {}
    if good:
        tps_list = [r["tok_per_sec"] for r in good]
        lat_list = [r["total_latency_s"] for r in good]
        acc_rates = [r["draft_acceptance_rate"] for r in good if r.get("draft_acceptance_rate") is not None]
        summary = {
            "n_success": len(good),
            "n_error": len(results) - len(good),
            "mean_tok_per_sec": round(sum(tps_list) / len(tps_list), 4),
            "median_tok_per_sec": round(sorted(tps_list)[len(tps_list) // 2], 4),
            "mean_latency_s": round(sum(lat_list) / len(lat_list), 2),
            "p95_latency_s": round(sorted(lat_list)[min(int(len(lat_list) * 0.95), len(lat_list) - 1)], 2),
            "mean_draft_acceptance_rate": round(sum(acc_rates) / len(acc_rates), 4) if acc_rates else "unavailable",
        }

    output = {
        "experiment": "exp2_speculative",
        "model_main": "qwen2:1.5b",
        "model_draft": "qwen2:0.5b",
        "incompatibility_note": (
            "Original design (phi3:mini main + qwen2:1.5b draft) failed: "
            "different tokenizer families (phi bos=1 vs qwen bos=151643). "
            "Switched to same-family pair: qwen2:1.5b main + qwen2:0.5b draft."
        ),
        "draft_max": 8,
        "llama_cpp_build": "b9297",
        "backend": "vulkan",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "server_args": ["-ngl", "99", "-ngld", "99", "--spec-draft-n-max", "8"],
        "vram_pre_server": vram_pre_server,
        "vram_post_experiment": vram_post,
        "summary": summary,
        "results": results,
    }

    with open(RESULTS_FILE, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {RESULTS_FILE}")
    print(f"Summary: {json.dumps(summary, indent=2)}")


if __name__ == "__main__":
    run_experiment()
