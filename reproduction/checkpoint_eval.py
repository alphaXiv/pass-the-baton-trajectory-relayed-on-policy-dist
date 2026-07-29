#!/usr/bin/env python3
"""Evaluate base and trained actors on one disjoint DAPO slice."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_SCRIPT = REPO_ROOT / "relay-opd" / "opd" / "eval" / "math_benchmarks.py"
GRADER = REPO_ROOT / "relay-opd" / "opd" / "reward" / "grader"
N_SHARDS = 8
N_PROBLEMS = 128
MAX_NEW = 2048
SEED = 314159


def evaluate(label: str, model: Path, data: Path) -> dict:
    output_root = data.parent / f"checkpoint_eval_{label}"
    log_root = data.parent / f"checkpoint_eval_logs_{label}"
    output_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)
    processes: list[tuple[int, subprocess.Popen, object, Path]] = []
    for shard_id in range(N_SHARDS):
        shard_out = output_root / f"shard_{shard_id}"
        shard_out.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            str(EVAL_SCRIPT),
            "--model",
            str(model),
            "--bench",
            "dapo128",
            "--tag",
            "dapoheldout",
            "--data_dir",
            str(data.parent),
            "--n_samples",
            "1",
            "--max_new",
            str(MAX_NEW),
            "--temperature",
            "0.0",
            "--top_p",
            "1.0",
            "--gpu_mem",
            "0.88",
            "--tp",
            "1",
            "--max_model_len",
            "4096",
            "--n_problems",
            str(N_PROBLEMS),
            "--num_shards",
            str(N_SHARDS),
            "--shard_id",
            str(shard_id),
            "--seed",
            str(SEED),
            "--disable_thinking",
            "--out_dir",
            str(shard_out),
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(shard_id)
        env["MATH_GRADER_PATH"] = str(GRADER)
        env["VLLM_ALLREDUCE_USE_FLASHINFER"] = "0"
        env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
        log_path = log_root / f"shard_{shard_id}.log"
        log_handle = log_path.open("w", encoding="utf-8")
        print(
            f"[checkpoint-eval-launch] label={label} shard={shard_id}/8 "
            f"gpu={shard_id} model={model}",
            flush=True,
        )
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT / "relay-opd",
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        processes.append((shard_id, process, log_handle, log_path))

    failures: list[int] = []
    for shard_id, process, log_handle, log_path in processes:
        return_code = process.wait()
        log_handle.close()
        print(
            f"[checkpoint-eval-complete] label={label} shard={shard_id} "
            f"rc={return_code}",
            flush=True,
        )
        if return_code:
            failures.append(shard_id)
            print(log_path.read_text(encoding="utf-8", errors="replace"), flush=True)
    if failures:
        raise RuntimeError(f"{label} evaluation shards failed: {failures}")

    records: list[dict] = []
    wall_seconds: list[float] = []
    for shard_id in range(N_SHARDS):
        shard_dir = output_root / f"shard_{shard_id}"
        with (shard_dir / "dapoheldout.jsonl").open(encoding="utf-8") as handle:
            records.extend(json.loads(line) for line in handle if line.strip())
        summary = json.loads(
            (shard_dir / "dapoheldout.summary.json").read_text(encoding="utf-8")
        )
        wall_seconds.append(float(summary["wall_seconds"]))

    lengths = [int(record["gen_len"]) for record in records]
    correct = [int(bool(record["correct"])) for record in records]
    sorted_lengths = sorted(lengths)
    result = {
        "label": label,
        "model": str(model),
        "n": len(records),
        "accuracy": sum(correct) / len(correct),
        "correct": sum(correct),
        "mean_gen_tokens": statistics.fmean(lengths),
        "median_gen_tokens": statistics.median(lengths),
        "p90_gen_tokens": sorted_lengths[
            max(0, int(0.9 * len(sorted_lengths)) - 1)
        ],
        "truncation_rate": sum(length >= MAX_NEW for length in lengths)
        / len(lengths),
        "wall_seconds": max(wall_seconds),
    }
    print("CHECKPOINT_EVAL_RESULT " + json.dumps(result, sort_keys=True), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=Path, required=True)
    parser.add_argument("--trained_model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    args = parser.parse_args()
    base = evaluate("base", args.base_model, args.data)
    trained = evaluate("trained", args.trained_model, args.data)
    comparison = {
        "accuracy_delta": trained["accuracy"] - base["accuracy"],
        "correct_delta": trained["correct"] - base["correct"],
        "mean_gen_tokens_delta": (
            trained["mean_gen_tokens"] - base["mean_gen_tokens"]
        ),
        "base_accuracy": base["accuracy"],
        "trained_accuracy": trained["accuracy"],
    }
    print(
        "CHECKPOINT_EVAL_COMPARISON "
        + json.dumps(comparison, sort_keys=True),
        flush=True,
    )


if __name__ == "__main__":
    main()
