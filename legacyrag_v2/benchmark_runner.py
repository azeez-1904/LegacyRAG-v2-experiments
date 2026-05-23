#!/usr/bin/env python3
"""
benchmark_runner.py — Run all 4 LegacyRAG v2 experiments sequentially.

60s cooldown between experiments to reset GPU thermal state.
Updates RESEARCH_LOG.md after each experiment.
Commits to git after each successful experiment.
Generates results/v2_summary.json at the end.
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

EXPERIMENTS_DIR = Path(__file__).parent
RESULTS_DIR = EXPERIMENTS_DIR / "results"
RESEARCH_LOG = EXPERIMENTS_DIR / "RESEARCH_LOG.md"
SUMMARY_FILE = RESULTS_DIR / "v2_summary.json"
COOLDOWN_S = 60
V1_BASELINE = {"mean_tok_per_sec": 0.95, "mean_latency_s": 469.4}

EXPERIMENT_FILES = [
    ("exp1", "experiment1_baseline.py", "results/exp1_baseline.json"),
    ("exp2", "experiment2_speculative_draft.py", "results/exp2_speculative.json"),
    ("exp3", "experiment3_ngram.py", "results/exp3_ngram.json"),
    ("exp4", "experiment4_quantization.py", "results/exp4_quant.json"),
]


def run_nvidia_smi() -> str:
    try:
        return subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,memory.free,memory.used,temperature.gpu,utilization.gpu",
             "--format=csv,noheader"],
            text=True,
        ).strip()
    except Exception as e:
        return f"nvidia-smi error: {e}"


def append_research_log(section: str) -> None:
    with open(RESEARCH_LOG, "a") as f:
        f.write(section)


def git_commit(message: str) -> bool:
    try:
        repo_root = EXPERIMENTS_DIR.parent
        subprocess.run(
            ["git", "add", str(EXPERIMENTS_DIR.relative_to(repo_root))],
            cwd=str(repo_root), check=True, capture_output=True
        )
        result = subprocess.run(
            ["git", "commit", "-m", message, "--no-gpg-sign"],
            cwd=str(repo_root), capture_output=True, text=True
        )
        if result.returncode == 0:
            print(f"  Git commit: {message[:60]}", flush=True)
            return True
        else:
            print(f"  Git commit skipped: {result.stderr.strip()[:100]}", flush=True)
            return False
    except Exception as e:
        print(f"  Git commit failed: {e}", flush=True)
        return False


def run_experiment(label: str, script: str) -> tuple[bool, dict | None]:
    """Run a single experiment script. Returns (success, summary_dict)."""
    script_path = EXPERIMENTS_DIR / script
    print(f"\n{'='*60}")
    print(f"Running {label}: {script}")
    print(f"GPU state before:\n{run_nvidia_smi()}")
    print(f"{'='*60}", flush=True)

    t_start = time.time()
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(EXPERIMENTS_DIR),
        capture_output=False,
    )
    elapsed = time.time() - t_start

    success = result.returncode == 0
    print(f"\n{label} finished in {elapsed:.0f}s, exit code {result.returncode}", flush=True)
    print(f"GPU state after:\n{run_nvidia_smi()}", flush=True)
    return success, elapsed


def load_results(json_path: str) -> dict | None:
    p = EXPERIMENTS_DIR / json_path
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def build_summary_row(label: str, data: dict | None) -> dict:
    if data is None:
        return {"experiment": label, "status": "missing", "error": "no result file"}
    if "error" in data and data.get("summary") is None:
        return {"experiment": label, "status": "error", "error": data["error"]}

    s = data.get("summary", {})
    mean_tps = s.get("mean_tok_per_sec")
    improvement = (
        round((mean_tps - V1_BASELINE["mean_tok_per_sec"]) / V1_BASELINE["mean_tok_per_sec"] * 100, 1)
        if mean_tps else None
    )
    return {
        "experiment": label,
        "status": "success" if s else "partial",
        "model": data.get("model") or data.get("model_main", "unknown"),
        "mean_tok_per_sec": mean_tps,
        "median_tok_per_sec": s.get("median_tok_per_sec"),
        "mean_latency_s": s.get("mean_latency_s"),
        "p95_latency_s": s.get("p95_latency_s"),
        "improvement_vs_v1_pct": improvement,
        "n_success": s.get("n_success"),
        "n_error": s.get("n_error"),
        "notes": data.get("mode_note") or data.get("error") or "",
    }


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    ts = datetime.now(timezone.utc).isoformat()
    append_research_log(
        f"\n\n---\n## Benchmark Run: {ts}\n\n"
        f"**Hardware:** Dual Quadro K4200, 4GB VRAM each, Maxwell, Vulkan\n"
        f"**v1 Baseline:** {V1_BASELINE['mean_tok_per_sec']} tok/s mean, {V1_BASELINE['mean_latency_s']}s latency\n\n"
        f"**Initial GPU state:**\n```\n{run_nvidia_smi()}\n```\n\n"
    )

    run_results = {}
    for label, script, result_path in EXPERIMENT_FILES:
        success, elapsed = run_experiment(label, script)
        run_results[label] = (success, result_path)

        data = load_results(result_path)
        summary_row = build_summary_row(label, data)

        # Update RESEARCH_LOG
        status_icon = "✓" if success else "✗"
        log_entry = (
            f"### {status_icon} {label}: {script}\n\n"
            f"- **Elapsed:** {elapsed:.0f}s\n"
            f"- **Exit code:** {'0 (success)' if success else 'non-zero'}\n"
        )
        if summary_row.get("mean_tok_per_sec"):
            log_entry += (
                f"- **Mean tok/s:** {summary_row['mean_tok_per_sec']}\n"
                f"- **Mean latency:** {summary_row.get('mean_latency_s')}s\n"
                f"- **p95 latency:** {summary_row.get('p95_latency_s')}s\n"
                f"- **vs v1 baseline:** {summary_row.get('improvement_vs_v1_pct')}%\n"
            )
        if summary_row.get("notes"):
            log_entry += f"- **Notes:** {summary_row['notes']}\n"
        log_entry += f"\n**GPU after {label}:**\n```\n{run_nvidia_smi()}\n```\n\n"

        append_research_log(log_entry)

        if success:
            git_commit(
                f"LegacyRAG v2: {label} complete — "
                f"{summary_row.get('mean_tok_per_sec', 'N/A')} tok/s mean"
            )

        if label != "exp4":
            print(f"\nCooldown: {COOLDOWN_S}s between experiments...", flush=True)
            time.sleep(COOLDOWN_S)

    # Generate v2_summary.json
    rows = []
    for label, _, result_path in EXPERIMENT_FILES:
        data = load_results(result_path)
        rows.append(build_summary_row(label, data))

    summary = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "v1_baseline": V1_BASELINE,
        "experiments": rows,
        "comparison_table": {
            "headers": [
                "Experiment", "Model", "Mean tok/s", "Median tok/s",
                "Mean latency (s)", "p95 latency (s)", "vs v1 (%)", "Notes"
            ],
            "rows": [
                [
                    r["experiment"], r.get("model", ""), r.get("mean_tok_per_sec"),
                    r.get("median_tok_per_sec"), r.get("mean_latency_s"),
                    r.get("p95_latency_s"), r.get("improvement_vs_v1_pct"),
                    (r.get("notes") or "")[:80],
                ]
                for r in rows
            ],
        },
    }

    with open(SUMMARY_FILE, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n\nAll experiments complete. Summary: {SUMMARY_FILE}")
    print("\n--- Comparison Table ---")
    print(f"{'Exp':<8} {'Model':<22} {'tok/s':<8} {'vs v1':<10} {'Status'}")
    print("-" * 65)
    for r in rows:
        tps = r.get("mean_tok_per_sec") or 0
        vs = r.get("improvement_vs_v1_pct")
        vs_str = f"{vs:+.1f}%" if vs is not None else "N/A"
        print(f"{r['experiment']:<8} {str(r.get('model',''))[:22]:<22} {tps:<8.3f} {vs_str:<10} {r['status']}")

    append_research_log(
        f"\n### Final Summary\n\n"
        f"```\n{json.dumps(summary['comparison_table']['rows'], indent=2)}\n```\n\n"
        f"**Benchmark run complete:** {datetime.now(timezone.utc).isoformat()}\n"
    )

    git_commit("LegacyRAG v2: all experiments complete, generate v2_summary.json")


if __name__ == "__main__":
    main()
