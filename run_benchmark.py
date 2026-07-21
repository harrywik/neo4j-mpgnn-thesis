#!/usr/bin/env python3
"""run_benchmark.py — orchestrate the papers100M benchmark.

Runs 5 repetitions of:
  1. Training 1 epoch — PyG in-memory (baseline_pyg)
  2. Training 1 epoch — Neo4j + PyG (java_neo4j)
  3. Inference 2048 nodes — PyG in-memory (NeighborLoader)
  4. Inference 2048 nodes — Neo4j + PyG (neighborhood_sampling + in_db_java)

Each variant is run as a subprocess.  OOM kills (exit code 137 / -9) are
caught and recorded.  Final results are written to results_summary.json.

Usage
-----
    PYTHONPATH=src python run_benchmark.py [--results_dir RESULTS_DIR]
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev

SRC_DIR = Path(__file__).resolve().parent / "src"
PROJECT_ROOT = Path(__file__).resolve().parent


def run_cmd(cmd, timeout=None, env=None):
    """Run a command as a subprocess.

    Returns (returncode, stdout, stderr, oom_killed).
    """
    run_env = os.environ.copy()
    if env:
        run_env.update(env)
    run_env["PYTHONPATH"] = str(SRC_DIR)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=run_env,
            cwd=str(PROJECT_ROOT),
        )
        oom = result.returncode in (137, -9)
        return result.returncode, result.stdout, result.stderr, oom
    except subprocess.TimeoutExpired:
        return -1, "", "TIMEOUT", False
    except Exception as e:
        return -1, "", str(e), False


def check_oom_from_dmesg():
    """Check dmesg for recent OOM kills (best-effort, may need root)."""
    try:
        result = subprocess.run(
            ["dmesg", "--time-format=iso", "-T"],
            capture_output=True, text=True, timeout=5,
        )
        lines = result.stdout.strip().split("\n")
        recent = [l for l in lines if "Out of memory" in l or "oom-kill" in l.lower()]
        return recent[-5:] if recent else []
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def run_training_variant(implementation, dataset, run_idx, results_dir):
    """Run one training repetition. Returns dict with timing or OOM status."""
    print(f"\n{'='*60}")
    print(f"  TRAINING [{implementation}] run {run_idx + 1}")
    print(f"{'='*60}")

    cmd = [
        sys.executable, "-m", "training.Main",
        "--dataset", dataset,
        "--implementation", implementation,
    ]

    t0 = time.monotonic()
    rc, stdout, stderr, oom = run_cmd(cmd, timeout=7200)
    wall = time.monotonic() - t0

    if oom:
        print(f"  ✗ OOM killed after {wall:.1f}s")
        return {"status": "OOM", "wall_time_s": round(wall, 2)}
    if rc != 0:
        print(f"  ✗ Failed (rc={rc}) after {wall:.1f}s")
        tail = (stderr or stdout or "")[-500:]
        print(f"  tail: {tail}")
        return {"status": "ERROR", "return_code": rc, "wall_time_s": round(wall, 2), "tail": tail}

    print(f"  ✓ Completed in {wall:.1f}s")

    result = {"status": "OK", "wall_time_s": round(wall, 2)}

    measurements = Path("measurements")
    if measurements.exists():
        mj = measurements / "measurements.json"
        if mj.exists():
            try:
                data = json.loads(mj.read_text())
                epoch_times = data.get("epoch_times", [])
                if epoch_times:
                    result["epoch_time_s"] = epoch_times[-1]
                    result["mean_epoch_time_s"] = round(mean(epoch_times), 3)
                    if len(epoch_times) > 1:
                        result["std_epoch_time_s"] = round(stdev(epoch_times), 3)
            except Exception as e:
                print(f"  Warning: could not parse measurements: {e}")

    return result


# ---------------------------------------------------------------------------
# Inference — PyG
# ---------------------------------------------------------------------------

def run_pyg_inference(run_idx, results_dir, n_nodes=2048):
    """Run PyG-only inference on n_nodes seed nodes."""
    print(f"\n{'='*60}")
    print(f"  INFERENCE [pyg_in_memory] run {run_idx + 1} ({n_nodes} nodes)")
    print(f"{'='*60}")

    out_json = results_dir / f"pyg_inference_run{run_idx}.json"
    cmd = [
        sys.executable, "-m", "benchmarking_tools.pyg_inference_bench",
        "--n_nodes", str(n_nodes),
        "--output_json", str(out_json),
    ]

    t0 = time.monotonic()
    rc, stdout, stderr, oom = run_cmd(cmd, timeout=3600)
    wall = time.monotonic() - t0

    if oom:
        print(f"  ✗ OOM killed after {wall:.1f}s")
        return {"status": "OOM", "wall_time_s": round(wall, 2)}
    if rc != 0:
        print(f"  ✗ Failed (rc={rc}) after {wall:.1f}s")
        tail = (stderr or stdout or "")[-500:]
        print(f"  tail: {tail}")
        return {"status": "ERROR", "return_code": rc, "wall_time_s": round(wall, 2), "tail": tail}

    print(f"  ✓ Completed in {wall:.1f}s")

    result = {"status": "OK", "wall_time_s": round(wall, 2)}
    if out_json.exists():
        try:
            data = json.loads(out_json.read_text())
            result["inference_time_s"] = data.get("total_time_s")
            result["ms_per_node"] = data.get("ms_per_node")
            result["throughput_nodes_per_s"] = data.get("throughput_nodes_per_s")
        except Exception:
            pass
    return result


# ---------------------------------------------------------------------------
# Inference — Neo4j
# ---------------------------------------------------------------------------

def run_neo4j_inference(run_idx, results_dir, strategies=None, n_nodes=2048):
    """Run Neo4j-backed inference experiment.

    strategies: list of strategy names, e.g. ["neighborhood_sampling", "in_db_java"]
    """
    if strategies is None:
        strategies = ["neighborhood_sampling", "in_db_java"]

    label = "+".join(strategies)
    print(f"\n{'='*60}")
    print(f"  INFERENCE [neo4j: {label}] run {run_idx + 1} ({n_nodes} nodes)")
    print(f"{'='*60}")

    output_dir = results_dir / f"neo4j_inference_run{run_idx}"
    dataset_cfg = "src/configs/inference/datasets/papers100M_bench.json"
    model_cfg = "src/configs/inference/models/gcn_papers100M.json"

    cmd = [
        sys.executable, "-m", "comparison_experiments.inference_experiment",
        "--dataset", dataset_cfg,
        "--model", model_cfg,
        "--output_dir", str(output_dir),
        "--strategies", *strategies,
    ]

    env = {}
    model_dir = os.environ.get("NEO4J_GNN_MODEL_DIR", "")
    if model_dir:
        env["NEO4J_GNN_MODEL_DIR"] = model_dir

    t0 = time.monotonic()
    rc, stdout, stderr, oom = run_cmd(cmd, timeout=7200, env=env)
    wall = time.monotonic() - t0

    if oom:
        print(f"  ✗ OOM killed after {wall:.1f}s")
        return {"status": "OOM", "wall_time_s": round(wall, 2)}
    if rc != 0:
        print(f"  ✗ Failed (rc={rc}) after {wall:.1f}s")
        tail = (stderr or stdout or "")[-500:]
        print(f"  tail: {tail}")
        return {"status": "ERROR", "return_code": rc, "wall_time_s": round(wall, 2), "tail": tail}

    print(f"  ✓ Completed in {wall:.1f}s")

    result = {"status": "OK", "wall_time_s": round(wall, 2)}

    # Parse per-strategy results from the experiment output JSON
    if output_dir.exists():
        jsons = sorted(output_dir.glob("*.json"))
        for jf in jsons:
            try:
                data = json.loads(jf.read_text())
                for strategy in strategies:
                    if strategy in data.get("results", {}):
                        sdata = data["results"][strategy]
                        # Look up the entry for our node count
                        for entry in sdata:
                            if entry.get("n_nodes") == n_nodes:
                                times = entry.get("times_s", [])
                                if times:
                                    result[f"{strategy}_time_s"] = round(mean(times), 4)
                                    if len(times) > 1:
                                        result[f"{strategy}_std_s"] = round(stdev(times), 4)
                                break
            except Exception:
                pass

    return result


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_runs(runs, time_key):
    """Compute mean ± std from a list of run dicts.

    Returns {"mean": ..., "std": ..., "n_ok": ..., "n_oom": ..., "n_error": ...}
    or {"status": "OOM"/"ERROR"} if no runs succeeded.
    """
    ok = [r for r in runs if r.get("status") == "OK" and time_key in r]
    n_oom = sum(1 for r in runs if r.get("status") == "OOM")
    n_err = sum(1 for r in runs if r.get("status") == "ERROR")

    if not ok:
        if n_oom > 0:
            return {"status": "OOM", "n_oom": n_oom, "n_error": n_err}
        return {"status": "ERROR", "n_error": n_err}

    vals = [r[time_key] for r in ok]
    out = {
        "mean": round(mean(vals), 4),
        "n_ok": len(ok),
        "n_oom": n_oom,
        "n_error": n_err,
    }
    if len(vals) > 1:
        out["std"] = round(stdev(vals), 4)
    else:
        out["std"] = 0.0
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run papers100M benchmark")
    parser.add_argument("--results_dir", type=str, default="experiment_results/gcp_benchmark")
    parser.add_argument("--n_runs", type=int, default=5)
    parser.add_argument("--n_nodes", type=int, default=2048)
    parser.add_argument("--skip_pyg_training", action="store_true",
                        help="Skip PyG in-memory training (expected OOM on large datasets)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    dataset = "papers100M_bench"
    n_runs = args.n_runs

    print("=" * 60)
    print("  papers100M Benchmark")
    print(f"  Runs: {n_runs}  |  Inference nodes: {args.n_nodes}")
    print(f"  Results dir: {results_dir}")
    print(f"  Started: {datetime.now().isoformat()}")
    print("=" * 60)

    all_results = {
        "metadata": {
            "started": datetime.now().isoformat(),
            "n_runs": n_runs,
            "n_inference_nodes": args.n_nodes,
            "dataset": dataset,
        },
        "training": {},
        "inference": {},
    }

    # ---- Training: PyG in-memory ----
    if not args.skip_pyg_training:
        pyg_train_runs = []
        for i in range(n_runs):
            r = run_training_variant("baseline_pyg", dataset, i, results_dir)
            pyg_train_runs.append(r)
        all_results["training"]["pyg_in_memory"] = {
            "runs": pyg_train_runs,
            "summary": aggregate_runs(pyg_train_runs, "mean_epoch_time_s"),
        }
    else:
        all_results["training"]["pyg_in_memory"] = {"summary": {"status": "SKIPPED"}}

    # ---- Training: Neo4j + PyG ----
    neo4j_train_runs = []
    for i in range(n_runs):
        r = run_training_variant("java_neo4j", dataset, i, results_dir)
        neo4j_train_runs.append(r)
    all_results["training"]["neo4j_java"] = {
        "runs": neo4j_train_runs,
        "summary": aggregate_runs(neo4j_train_runs, "mean_epoch_time_s"),
    }

    # ---- Inference: PyG in-memory ----
    pyg_inf_runs = []
    for i in range(n_runs):
        r = run_pyg_inference(i, results_dir, n_nodes=args.n_nodes)
        pyg_inf_runs.append(r)
    all_results["inference"]["pyg_in_memory"] = {
        "runs": pyg_inf_runs,
        "summary": aggregate_runs(pyg_inf_runs, "inference_time_s"),
    }

    # ---- Inference: Neo4j (neighborhood_sampling + in_db_java) ----
    neo4j_inf_runs = []
    for i in range(n_runs):
        r = run_neo4j_inference(
            i, results_dir,
            strategies=["neighborhood_sampling", "in_db_java"],
            n_nodes=args.n_nodes,
        )
        neo4j_inf_runs.append(r)
    all_results["inference"]["neo4j"] = {
        "runs": neo4j_inf_runs,
        "summary": {
            "neighborhood_sampling": aggregate_runs(neo4j_inf_runs, "neighborhood_sampling_time_s"),
            "in_db_java": aggregate_runs(neo4j_inf_runs, "in_db_java_time_s"),
        },
    }

    # ---- Write results ----
    all_results["metadata"]["finished"] = datetime.now().isoformat()
    out_path = results_dir / "results_summary.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # ---- Print summary ----
    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)

    for section in ("training", "inference"):
        print(f"\n  {section.upper()}:")
        for variant, vdata in all_results[section].items():
            summary = vdata.get("summary", {})
            status = summary.get("status", "")
            if status in ("OOM", "ERROR", "SKIPPED"):
                print(f"    {variant}: {status}")
            else:
                m = summary.get("mean", "?")
                s = summary.get("std", 0)
                n = summary.get("n_ok", 0)
                oom = summary.get("n_oom", 0)
                line = f"    {variant}: {m:.4f}s ± {s:.4f}s (n={n}"
                if oom:
                    line += f", {oom} OOM"
                line += ")"
                print(line)

    print(f"\n  Results → {out_path}")
    print(f"  Finished: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
