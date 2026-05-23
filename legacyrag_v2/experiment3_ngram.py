#!/usr/bin/env python3
"""
Experiment 3: N-gram speculative decoding using llama.cpp b9297.

b9297 exposes --spec-type with variants: ngram-simple, ngram-map-k,
ngram-map-k4v, ngram-mod, ngram-cache. No draft model needed.
Uses --spec-type ngram-simple --spec-draft-n-max 8.

b9297 binary lives in build_b9297/ (downloaded from ggml-org/llama.cpp b9297).
exp1/exp2 used b5576 (build/bin/). We note this in results for reproducibility.
"""

import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
# b9297 binary — has ngram speculative support
BIN_DIR = PROJECT_ROOT / "build_b9297"
LLAMA_SERVER = BIN_DIR / "llama-server"
SERVER_PORT = 8080
SERVER_URL = f"http://127.0.0.1:{SERVER_PORT}"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_FILE = RESULTS_DIR / "exp3_ngram.json"
MAX_TOKENS = 200
SERVER_TIMEOUT = 120
LIB_PATH = str(BIN_DIR)
BLOBS_DIR = Path("/usr/share/ollama/.ollama/models/blobs")

NGRAM_FLAG_CANDIDATES = [
    "ngram-simple",
    "ngram-map-k",
    "ngram-cache",
    "lookup-cache-static",
]


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
    raise FileNotFoundError(f"No model layer: {manifest_path}")


def get_vram() -> list[dict]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.free,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            text=True,
        )
        result = []
        for line in out.strip().splitlines():
            idx, free, used, total = [x.strip() for x in line.split(",")]
            result.append({"gpu": int(idx), "free_mb": float(free),
                           "used_mb": float(used), "total_mb": float(total)})
        return result
    except Exception as e:
        return [{"error": str(e)}]


def probe_ngram_support() -> tuple[bool, list[str]]:
    """Check if this binary supports ngram speculative via --spec-type."""
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = LIB_PATH + ":" + env.get("LD_LIBRARY_PATH", "")

    help_out = subprocess.run(
        [str(LLAMA_SERVER), "--help"],
        env=env, capture_output=True, text=True
    )
    full_help = help_out.stdout + help_out.stderr

    # b9297 exposes these as values to --spec-type
    found = [f for f in NGRAM_FLAG_CANDIDATES if f in full_help]
    return bool(found), found


def start_server(model_path: Path, extra_args: list[str]) -> subprocess.Popen:
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = LIB_PATH + ":" + env.get("LD_LIBRARY_PATH", "")

    cmd = [
        str(LLAMA_SERVER),
        "-m", str(model_path),
        "-ngl", "99",
        "--port", str(SERVER_PORT),
        "--host", "127.0.0.1",
        "--ctx-size", "2048",
        "--threads", "4",
        "--parallel", "1",
        "--log-disable",
    ] + extra_args

    log_path = RESULTS_DIR / "server_exp3.log"
    log_file = open(log_path, "a")
    log_file.write(f"\n\n=== Exp3 Server start {datetime.now(timezone.utc).isoformat()} ===\n")
    log_file.write(f"CMD: {' '.join(cmd)}\n")
    log_file.flush()

    proc = subprocess.Popen(cmd, env=env, stdout=log_file, stderr=log_file)
    print(f"  Server PID {proc.pid}", flush=True)

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
            raise RuntimeError(f"Server exited (code {proc.returncode}). See {log_path}")

    proc.terminate()
    log_file.close()
    raise TimeoutError(f"Server not healthy after {SERVER_TIMEOUT}s")


def stop_server(proc: subprocess.Popen, sleep_s: int = 30) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    print(f"  Server stopped. Sleeping {sleep_s}s...", flush=True)
    time.sleep(sleep_s)


def run_completion(prompt: str) -> tuple[dict, float]:
    import urllib.request
    payload = json.dumps({
        "prompt": prompt, "n_predict": MAX_TOKENS, "temperature": 0.1,
        "stop": ["</s>", "<|end|>"], "stream": False,
    }).encode()
    req = urllib.request.Request(
        f"{SERVER_URL}/completion", data=payload,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=900) as r:
        raw = r.read()
    return json.loads(raw), time.perf_counter() - t0


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
         "chunks are irrelevant or too long? Provide a concrete example with a government records use case."
     )},
    {"id": 5, "bucket": "medium",
     "text": (
         "Compare the performance characteristics of modern NVIDIA Ampere or Ada Lovelace GPUs versus legacy Maxwell "
         "architecture GPUs when running transformer model inference. What specific hardware features are absent in "
         "Maxwell that limit throughput? How does the Quadro K4200 specifically perform? "
         "What is the practical implication for organizations running open-source LLMs on older hardware?"
     )},
    {"id": 6, "bucket": "medium",
     "text": (
         "What is the difference between greedy decoding, beam search, and temperature-based sampling in language "
         "model text generation? How does each strategy affect output diversity, factual accuracy, and token "
         "generation speed on resource-constrained hardware running under 2 tokens per second?"
     )},
    {"id": 7, "bucket": "medium",
     "text": (
         "Describe the Open Public Records Act (OPRA) in New Jersey. What obligations does it place on government "
         "agencies? How many days do they have to respond? What exemptions exist and how might an AI system "
         "improve compliance efficiency for municipal governments?"
     )},
    {"id": 8, "bucket": "long",
     "text": (
         "You are a research assistant preparing a technical analysis of the NVIDIA Quadro K4200 for LLM inference. "
         "The K4200 uses Maxwell architecture (GM204), 1344 CUDA cores, 4GB GDDR5, 173 GB/s bandwidth, Vulkan 1.3. "
         "It lacks FP16 matrix ops, INT8 dot products, and tensor cores. Analyze: "
         "(1) how architectural limits affect transformer throughput, "
         "(2) which quantization strategies fit 4GB VRAM, "
         "(3) theoretical max tok/s from 173 GB/s bandwidth for a 3.8B Q4 model, "
         "(4) how dual-GPU tensor-split affects throughput, "
         "(5) practical use cases in 2025 for this hardware class. "
         "Provide specific numbers and calculations."
     )},
    {"id": 9, "bucket": "long",
     "text": (
         "Write a technical overview of speculative decoding for an IEEE conference paper. Cover: "
         "(1) draft-then-verify mechanism and parallel verification, "
         "(2) acceptance rate alpha and its relationship to actual speedup, "
         "(3) ngram-based speculative decoding as a draft-model-free alternative, "
         "(4) VRAM trade-offs for draft models on 4-8GB devices, "
         "(5) GPU vs CPU performance differences, "
         "(6) challenges with Q2-Q4 quantized models where draft and main distributions diverge. "
         "Include literature references."
     )},
    {"id": 10, "bucket": "long",
     "text": (
         "A New Jersey municipality receives 500 OPRA requests/month. Staff spend 2-4 hours per request. "
         "Available hardware: two Quadro K4200 GPUs (4GB VRAM each, Maxwell, Vulkan-only). No new budget. "
         "No ML expertise on staff. Must be on-premises for legal compliance. "
         "Design a technical architecture covering: model selection for 8GB VRAM, document ingestion, "
         "query handling, expected performance (baseline: 0.95 tok/s, 469s latency), bottleneck analysis, "
         "staff workflow integration, and risk assessment for legacy Vulkan hardware."
     )},
]


def run_experiment() -> None:
    print("=" * 60)
    print("Experiment 3: N-gram Speculative Decoding (or baseline ablation)")
    print("=" * 60)

    RESULTS_DIR.mkdir(exist_ok=True)

    # Check if this build supports ngram mode
    ngram_supported, ngram_flags = probe_ngram_support()
    mode_note = ""

    if ngram_supported:
        # b9297: use --spec-type ngram-simple with --spec-draft-n-max 8
        extra_args = ["--spec-type", "ngram-simple", "--spec-draft-n-max", "8"]
        mode = "ngram-simple"
        mode_note = f"b9297: --spec-type ngram-simple --spec-draft-n-max 8 (candidates found: {ngram_flags})"
        print(f"  Ngram support confirmed in b9297: {ngram_flags}")
    else:
        extra_args = []
        mode = "baseline_ablation_no_draft"
        mode_note = "Ngram spec-type not found in binary help output. Running baseline ablation."
        print(f"  WARNING: {mode_note}")

    phi3_path = find_model_gguf("phi3/mini")
    vram_pre_server = get_vram()

    print(f"\nStarting llama-server (mode={mode})...")
    try:
        server_proc = start_server(phi3_path, extra_args)
    except Exception as e:
        output = {
            "experiment": "exp3_ngram",
            "mode": mode,
            "mode_note": mode_note,
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
            tokens_pred = timings.get("predicted_n", 0) or resp.get("tokens_predicted", 0)
            toks_per_sec = timings.get("predicted_per_second", None)
            if toks_per_sec is None and tokens_pred > 0 and wall_time > 0:
                toks_per_sec = tokens_pred / wall_time

            vram_after = get_vram()
            record = {
                "id": p["id"],
                "bucket": p["bucket"],
                "timestamp": t_start,
                "prompt_words": len(p["text"].split()),
                "tokens_predicted": tokens_pred,
                "tok_per_sec": round(toks_per_sec, 4) if toks_per_sec else None,
                "total_latency_s": round(wall_time, 3),
                "vram_before": vram_before,
                "vram_after": vram_after,
                "timings": timings,
                "error": None,
            }
            print(f"  tok/s={record['tok_per_sec']}  wall={wall_time:.1f}s", flush=True)
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
        summary = {
            "n_success": len(good),
            "n_error": len(results) - len(good),
            "mean_tok_per_sec": round(sum(tps_list) / len(tps_list), 4),
            "mean_latency_s": round(sum(lat_list) / len(lat_list), 2),
            "p95_latency_s": round(sorted(lat_list)[min(int(len(lat_list) * 0.95), len(lat_list) - 1)], 2),
        }

    output = {
        "experiment": "exp3_ngram",
        "mode": mode,
        "mode_note": mode_note,
        "ngram_flags_checked": NGRAM_FLAG_CANDIDATES,
        "ngram_flags_found": ngram_flags,
        "extra_server_args": extra_args,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "vram_pre_server": vram_pre_server,
        "vram_post_experiment": vram_post,
        "summary": summary,
        "results": results,
    }

    with open(RESULTS_FILE, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {RESULTS_FILE}")
    print(f"Mode: {mode}")
    print(f"Summary: {json.dumps(summary, indent=2)}")


if __name__ == "__main__":
    run_experiment()
