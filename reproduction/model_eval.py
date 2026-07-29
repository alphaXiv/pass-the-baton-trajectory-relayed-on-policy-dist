#!/usr/bin/env python3
"""Eight-way data-parallel DAPO-128 rollout evaluation."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download, snapshot_download


STUDENT_REPO = "Qwen/Qwen3-1.7B"
TEACHER_REPO = "Qwen/Qwen3-4B-Instruct-2507"
DATASET_REPO = "BytedTsinghua-SIA/DAPO-Math-17k"
DATASET_FILE = "data/dapo-math-17k.parquet"
N_PROBLEMS = 128
N_SHARDS = 8
MAX_NEW = 4096
SEED = 42


def download_inputs(work_dir: Path, mode: str) -> tuple[Path, Path | None, Path]:
    token = os.environ.get("HF_TOKEN")
    model_dir = work_dir / "models"
    data_dir = work_dir / "data"
    model_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"[download] student={STUDENT_REPO}", flush=True)
    student = Path(
        snapshot_download(
            repo_id=STUDENT_REPO,
            local_dir=model_dir / "student",
            token=token,
        )
    )
    teacher = None
    if mode == "relay":
        print(f"[download] teacher={TEACHER_REPO}", flush=True)
        teacher = Path(
            snapshot_download(
                repo_id=TEACHER_REPO,
                local_dir=model_dir / "teacher",
                token=token,
            )
        )

    source = Path(
        hf_hub_download(
            repo_id=DATASET_REPO,
            filename=DATASET_FILE,
            repo_type="dataset",
            local_dir=data_dir / "source",
            token=token,
        )
    )
    frame = pd.read_parquet(source).head(N_PROBLEMS).copy()
    target = data_dir / "dapo128.parquet"
    frame.to_parquet(target, index=False)
    print(
        f"[data] source={DATASET_REPO}/{DATASET_FILE} rows={len(frame)} "
        f"columns={list(frame.columns)} target={target}",
        flush=True,
    )
    return student, teacher, data_dir


def launch_shards(
    repo_root: Path,
    work_dir: Path,
    mode: str,
    student: Path,
    teacher: Path | None,
    data_dir: Path,
) -> list[dict]:
    eval_script = repo_root / "relay-opd" / "opd" / "eval" / "math_benchmarks.py"
    grader = repo_root / "relay-opd" / "opd" / "reward" / "grader"
    patch = repo_root / "relay-opd" / "opd" / "patches" / "vllm"
    output_root = work_dir / f"eval_{mode}"
    log_root = work_dir / f"logs_{mode}"
    output_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)

    processes: list[tuple[int, subprocess.Popen, object, Path]] = []
    for shard_id in range(N_SHARDS):
        shard_out = output_root / f"shard_{shard_id}"
        shard_out.mkdir(parents=True, exist_ok=True)
        model = teacher if mode == "relay" else student
        assert model is not None
        cmd = [
            sys.executable,
            str(eval_script),
            "--model",
            str(model),
            "--bench",
            "dapo128",
            "--data_dir",
            str(data_dir),
            "--n_samples",
            "1",
            "--max_new",
            str(MAX_NEW),
            "--temperature",
            "1.0",
            "--top_p",
            "1.0",
            "--gpu_mem",
            "0.88",
            "--tp",
            "1",
            "--max_model_len",
            "6144",
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
        if mode == "relay":
            cmd.extend(
                [
                    "--draft_model",
                    str(student),
                    "--num_spec_tokens",
                    "4",
                    "--draft_tp",
                    "1",
                    "--rollout_mode",
                    "relay",
                    "--trigger_topk",
                    "5",
                    "--max_takeovers",
                    "2",
                    "--paragraphs_per_takeover",
                    "4",
                    "--save_rollout_trace",
                    "--trace_mode",
                    "mask",
                ]
            )
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(shard_id)
        env["MATH_GRADER_PATH"] = str(grader)
        env["VLLM_ALLREDUCE_USE_FLASHINFER"] = "0"
        env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
        if mode == "relay":
            env["PYTHONPATH"] = f"{patch}:{env.get('PYTHONPATH', '')}"
        log_path = log_root / f"shard_{shard_id}.log"
        log_handle = log_path.open("w", encoding="utf-8")
        print(
            f"[launch] mode={mode} shard={shard_id}/{N_SHARDS} "
            f"gpu={shard_id} output={shard_out}",
            flush=True,
        )
        process = subprocess.Popen(
            cmd,
            cwd=repo_root / "relay-opd",
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
            f"[complete] mode={mode} shard={shard_id} rc={return_code}",
            flush=True,
        )
        if return_code != 0:
            failures.append(shard_id)
            print(log_path.read_text(encoding="utf-8", errors="replace"), flush=True)
    if failures:
        raise RuntimeError(f"Evaluation shards failed: {failures}")

    summaries: list[dict] = []
    for shard_id in range(N_SHARDS):
        summary_path = output_root / f"shard_{shard_id}" / "dapo128.summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summaries.append(summary)
        print(
            "MODEL_EVAL_SHARD "
            f"mode={mode} shard={shard_id} n={summary['n_problems']} "
            f"accuracy={summary['pass@1']:.6f} wall_seconds={summary['wall_seconds']:.3f}",
            flush=True,
        )
    return summaries


def aggregate(work_dir: Path, mode: str, summaries: list[dict]) -> dict:
    records: list[dict] = []
    for shard_id in range(N_SHARDS):
        path = work_dir / f"eval_{mode}" / f"shard_{shard_id}" / "dapo128.jsonl"
        with path.open(encoding="utf-8") as handle:
            records.extend(json.loads(line) for line in handle if line.strip())

    lengths = [int(record["gen_len"]) for record in records]
    correct = [int(bool(record["correct"])) for record in records]
    truncations = [
        int(record.get("finish_reason") == "length" or record["gen_len"] >= MAX_NEW)
        for record in records
    ]
    stats: dict[str, float] = {}
    for summary in summaries:
        for key, value in summary.get("opd_stats", {}).items():
            stats[key] = stats.get(key, 0.0) + float(value)
    teacher_tokens = stats.get("teacher_tokens", 0.0)
    student_tokens = stats.get("student_tokens", 0.0)
    source_tokens = teacher_tokens + student_tokens
    wall_seconds = max(float(summary["wall_seconds"]) for summary in summaries)
    sorted_lengths = sorted(lengths)
    result = {
        "mode": mode,
        "student": STUDENT_REPO,
        "teacher": TEACHER_REPO if mode == "relay" else None,
        "n": len(records),
        "accuracy": sum(correct) / len(correct),
        "mean_gen_tokens": statistics.fmean(lengths),
        "median_gen_tokens": statistics.median(lengths),
        "p90_gen_tokens": sorted_lengths[max(0, int(0.9 * len(sorted_lengths)) - 1)],
        "truncation_rate": sum(truncations) / len(truncations),
        "wall_seconds": wall_seconds,
        "throughput_tokens_per_second": sum(lengths) / max(wall_seconds, 1e-9),
        "teacher_token_ratio": teacher_tokens / source_tokens if source_tokens else 0.0,
        "relay_divergence_triggers": stats.get("relay_divergence_triggers", 0.0),
        "relay_takeover_triggers": stats.get("relay_takeover_triggers", 0.0),
        "relay_takeovers_completed": stats.get("relay_takeovers_completed", 0.0),
        "opd_stats": stats,
    }
    print("MODEL_EVAL_RESULT " + json.dumps(result, sort_keys=True), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["student", "relay"])
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    work_dir = Path(os.environ.get("REPRO_WORK_DIR", "/tmp/relay-opd-reproduction"))
    work_dir.mkdir(parents=True, exist_ok=True)
    student, teacher, data_dir = download_inputs(work_dir, args.mode)
    summaries = launch_shards(
        repo_root, work_dir, args.mode, student, teacher, data_dir
    )
    aggregate(work_dir, args.mode, summaries)


if __name__ == "__main__":
    main()
