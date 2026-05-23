#!/usr/bin/env python3
"""
Experiment 4: qwen2.5:7b-instruct-q2_K quantization test.

Pulls qwen2.5:7b-instruct-q2_K via Ollama if not present, finds GGUF path,
checks it fits within 8GB combined VRAM, runs 5 medium prompts (skip long).
"""

import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
BIN_DIR = PROJECT_ROOT / "build_b9297"  # use b9297 for exp4
LLAMA_SERVER = BIN_DIR / "llama-server"
SERVER_PORT = 8080
SERVER_URL = f"http://127.0.0.1:{SERVER_PORT}"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_FILE = RESULTS_DIR / "exp4_quant.json"
MAX_TOKENS = 150  # shorter for 7B
SERVER_TIMEOUT = 180  # larger model takes longer to load
LIB_PATH = str(BIN_DIR)
BLOBS_DIR = Path("/usr/share/ollama/.ollama/models/blobs")

# 7B Q2_K fits in ~3.2GB — within 4GB on a single K4200
VRAM_LIMIT_MB = 8000


def find_model_gguf_by_manifest(manifest_path: str) -> Path | None:
    """Return GGUF blob path from manifest, or None if manifest missing."""
    manifest = Path(
        "/usr/share/ollama/.ollama/models/manifests"
        f"/registry.ollama.ai/library/{manifest_path}"
    )
    if not manifest.exists():
        return None
    with open(manifest) as f:
        data = json.load(f)
    for layer in data["layers"]:
        if layer["mediaType"] == "application/vnd.ollama.image.model":
            digest = layer["digest"].replace("sha256:", "sha256-")
            p = BLOBS_DIR / digest
            return p if p.exists() else None
    return None


def pull_model(model_tag: str) -> bool:
    """Pull a model via ollama CLI. Returns True on success."""
    print(f"  Pulling {model_tag} via ollama...", flush=True)
    result = subprocess.run(
        ["ollama", "pull", model_tag],
        capture_output=False,
        timeout=1800,
    )
    return result.returncode == 0


def find_gguf_by_size_range(min_gb: float, max_gb: float) -> Path | None:
    """Fallback: find a blob in the expected size range for Q2_K 7B (~2.7-3.5GB)."""
    min_bytes = int(min_gb * 1024**3)
    max_bytes = int(max_gb * 1024**3)
    for blob in sorted(BLOBS_DIR.iterdir()):
        try:
            size = blob.stat().st_size
            if min_bytes <= size <= max_bytes and blob.name.startswith("sha256-"):
                return blob
        except OSError:
            continue
    return None


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


def check_vram_feasibility(model_path: Path) -> tuple[bool, str]:
    """Check if model file size suggests it fits in 8GB combined VRAM."""
    size_bytes = model_path.stat().st_size
    size_mb = size_bytes / (1024 ** 2)
    vram = get_vram()
    total_free_mb = sum(g.get("free_mb", 0) for g in vram if "free_mb" in g)
    fits = size_mb < VRAM_LIMIT_MB and size_mb < total_free_mb + 1500  # 1.5GB headroom
    msg = (
        f"Model size: {size_mb:.0f}MB, "
        f"Total free VRAM: {total_free_mb:.0f}MB, "
        f"VRAM limit: {VRAM_LIMIT_MB}MB — {'OK' if fits else 'MAY NOT FIT'}"
    )
    return fits, msg


def start_server(model_path: Path) -> subprocess.Popen:
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = LIB_PATH + ":" + env.get("LD_LIBRARY_PATH", "")

    cmd = [
        str(LLAMA_SERVER),
        "-m", str(model_path),
        "-ngl", "99",
        "--port", str(SERVER_PORT),
        "--host", "127.0.0.1",
        "--ctx-size", "1024",
        "--threads", "4",
        "--parallel", "1",
        "--log-disable",
    ]

    log_path = RESULTS_DIR / "server_exp4.log"
    log_file = open(log_path, "a")
    log_file.write(f"\n\n=== Exp4 Server start {datetime.now(timezone.utc).isoformat()} ===\n")
    log_file.write(f"CMD: {' '.join(cmd)}\n")
    log_file.flush()

    proc = subprocess.Popen(cmd, env=env, stdout=log_file, stderr=log_file)
    print(f"  Server PID {proc.pid}", flush=True)

    deadline = time.time() + SERVER_TIMEOUT
    while time.time() < deadline:
        time.sleep(3)
        try:
            import urllib.request
            with urllib.request.urlopen(f"{SERVER_URL}/health", timeout=5) as r:
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
        "stop": ["</s>", "<|end|>", "<|im_end|>"], "stream": False,
    }).encode()
    req = urllib.request.Request(
        f"{SERVER_URL}/completion", data=payload,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=900) as r:
        raw = r.read()
    return json.loads(raw), time.perf_counter() - t0


# 5 prompts — short and medium only (skip long for 7B on 4GB)
PROMPTS = [
    {"id": 1, "bucket": "short",
     "text": "What is speculative decoding in large language models and why does it improve inference speed?"},
    {"id": 2, "bucket": "short",
     "text": "How does post-training quantization reduce neural network memory footprint while preserving accuracy?"},
    {"id": 3, "bucket": "medium",
     "text": (
         "Explain how retrieval-augmented generation (RAG) works and why it is useful for enterprise applications. "
         "Describe the role of embedding models and vector stores in a RAG pipeline. "
         "How is retrieved context injected into the prompt, and what are the main failure modes?"
     )},
    {"id": 4, "bucket": "medium",
     "text": (
         "Compare the NVIDIA Quadro K4200 Maxwell GPU with modern Ampere/Ada Lovelace GPUs for LLM inference. "
         "What hardware features are missing, and what is the practical performance impact? "
         "What quantization level (Q2, Q4, Q8) makes a 7B model fit in 4GB VRAM?"
     )},
    {"id": 5, "bucket": "medium",
     "text": (
         "Describe the Open Public Records Act (OPRA) in New Jersey. What obligations does it place on agencies? "
         "How many business days do they have to respond? What categories of records are exempt? "
         "How could an on-premises AI system running on legacy GPU hardware assist with OPRA compliance?"
     )},
]


def run_experiment() -> None:
    print("=" * 60)
    print("Experiment 4: qwen2.5:7b-instruct-q2_K Quantization Test")
    print("=" * 60)

    RESULTS_DIR.mkdir(exist_ok=True)

    # Try to find the model, pull if needed
    model_tag = "qwen2.5:7b-instruct-q2_K"
    manifest_path = "qwen2.5/7b-instruct-q2_K"

    model_path = find_model_gguf_by_manifest(manifest_path)
    pull_attempted = False
    pull_success = None

    if model_path is None:
        print(f"  Model not found. Attempting: ollama pull {model_tag}")
        pull_attempted = True
        try:
            pull_success = pull_model(model_tag)
        except subprocess.TimeoutExpired:
            pull_success = False
            print("  Pull timed out after 30 minutes.")

        if pull_success:
            model_path = find_model_gguf_by_manifest(manifest_path)

    if model_path is None:
        # Try fallback: find a blob in 2.5-3.5GB range (Q2_K 7B)
        print("  Trying size-range fallback for Q2_K 7B blob...")
        model_path = find_gguf_by_size_range(2.5, 3.8)

    if model_path is None:
        output = {
            "experiment": "exp4_quant",
            "error": f"Could not find or pull {model_tag}. Pull attempted: {pull_attempted}, success: {pull_success}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        with open(RESULTS_FILE, "w") as f:
            json.dump(output, f, indent=2)
        print(f"FATAL: Model unavailable. Logged to {RESULTS_FILE}.")
        return

    print(f"  Model path: {model_path}")
    size_mb = model_path.stat().st_size / (1024 ** 2)
    print(f"  Model size: {size_mb:.0f} MB")

    feasible, feasibility_msg = check_vram_feasibility(model_path)
    print(f"  VRAM check: {feasibility_msg}")

    if not feasible:
        print("  WARNING: Model may exceed VRAM. Attempting anyway — will log OOM if it fails.")

    vram_pre_server = get_vram()

    print("\nStarting llama-server with 7B Q2_K model...")
    try:
        server_proc = start_server(model_path)
    except Exception as e:
        output = {
            "experiment": "exp4_quant",
            "model": model_tag,
            "model_path": str(model_path),
            "model_size_mb": round(size_mb, 1),
            "feasibility_check": feasibility_msg,
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
        print(f"\n[{p['id']}/5] bucket={p['bucket']}")
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
    # Compare to exp1 baseline (0.95 tok/s)
    baseline_tps = 0.95
    summary = {}
    if good:
        tps_list = [r["tok_per_sec"] for r in good]
        lat_list = [r["total_latency_s"] for r in good]
        mean_tps = sum(tps_list) / len(tps_list)
        summary = {
            "n_success": len(good),
            "n_error": len(results) - len(good),
            "mean_tok_per_sec": round(mean_tps, 4),
            "mean_latency_s": round(sum(lat_list) / len(lat_list), 2),
            "p95_latency_s": round(sorted(lat_list)[min(int(len(lat_list) * 0.95), len(lat_list) - 1)], 2),
            "speedup_vs_phi3_baseline": round(mean_tps / baseline_tps, 3),
            "baseline_reference": {"model": "phi3:mini", "mean_tok_per_sec": baseline_tps},
        }

    output = {
        "experiment": "exp4_quant",
        "model": model_tag,
        "model_path": str(model_path),
        "model_size_mb": round(size_mb, 1),
        "pull_attempted": pull_attempted,
        "pull_success": pull_success,
        "feasibility_check": feasibility_msg,
        "backend": "vulkan",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "server_args": ["-ngl", "99", "--ctx-size", "1024"],
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
