#!/usr/bin/env python3
"""Evaluate released-trigger K7 on 512 held-out problems."""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download, snapshot_download


REPO_ROOT = Path(__file__).resolve().parents[1]
VERL_OPD_DIR = REPO_ROOT / "relay-opd"
WORK_DIR = Path("/tmp/relay-opd-released-k7-m1-l4-response1536-eval512")
STUDENT_REPO = "Qwen/Qwen3-1.7B"
TEACHER_REPO = "Qwen/Qwen3-4B-Instruct-2507"
DATA_REPO = "BytedTsinghua-SIA/DAPO-Math-17k"
DATA_FILE = "data/dapo-math-17k.parquet"


def prepare_inputs() -> tuple[Path, Path, Path, Path, Path]:
    models_dir = WORK_DIR / "models"
    data_dir = WORK_DIR / "data"
    bench_dir = WORK_DIR / "bench"
    output_dir = WORK_DIR / "output"
    for path in (models_dir, data_dir, bench_dir, output_dir):
        path.mkdir(parents=True, exist_ok=True)

    print(f"[download] student={STUDENT_REPO}", flush=True)
    student = Path(
        snapshot_download(
            repo_id=STUDENT_REPO,
            local_dir=models_dir / "student",
        )
    )
    print(f"[download] teacher={TEACHER_REPO}", flush=True)
    teacher = Path(
        snapshot_download(
            repo_id=TEACHER_REPO,
            local_dir=models_dir / "teacher",
        )
    )
    source = Path(
        hf_hub_download(
            repo_id=DATA_REPO,
            filename=DATA_FILE,
            repo_type="dataset",
        )
    )
    frame = pd.read_parquet(source).head(16512).copy()
    train = frame.iloc[:2048].copy()
    heldout = frame.iloc[16000:16512].copy()
    train_path = data_dir / "dapo2048.parquet"
    # The official evaluator maps its generic DAPO entry to dapo128.parquet.
    heldout_path = bench_dir / "dapo128.parquet"
    train.to_parquet(train_path, index=False)
    heldout.to_parquet(heldout_path, index=False)

    # The official trainer constructs validation datasets even when validation
    # is disabled. These schema-compatible placeholders are never evaluated.
    validation = heldout.head(8).copy()
    validation.to_parquet(bench_dir / "aime-24_verl.parquet", index=False)
    validation.to_parquet(bench_dir / "aime-2025_verl.parquet", index=False)
    print(
        f"[data] source={DATA_REPO}/{DATA_FILE} train_rows={len(train)} "
        f"heldout_rows={len(heldout)} train_slice=0:2048 "
        f"heldout_slice=16000:16512 "
        f"train={train_path} heldout={heldout_path} validation_enabled=False",
        flush=True,
    )
    return student, teacher, train_path, heldout_path, output_dir


def main() -> None:
    started = time.monotonic()
    student, teacher, train_path, heldout_path, output_dir = prepare_inputs()
    env = os.environ.copy()
    env.update(
        {
            "STUDENT_MODEL": str(student),
            "TEACHER_MODEL": str(teacher),
            "TRAIN_DATA": str(train_path),
            "BENCH": str(WORK_DIR / "bench"),
            "OUTPUT_DIR": str(output_dir),
            "EXP_ID": "released_trigger_relay_k7_m1_l4_response1536_eval512",
            "TRAIN_BATCH_SIZE": "128",
            "PPO_MINI_BATCH_SIZE": "128",
            "MAX_PROMPT_LENGTH": "2048",
            "MAX_RESPONSE_LENGTH": "1536",
            "VAL_MAX_RESPONSE_LENGTH": "2048",
            "ROLLOUT_MAX_MODEL_LEN": "3585",
            "TEACHER_MAX_MODEL_LEN": "3585",
            "ACTOR_GPUS_PER_NODE": "4",
            "TEACHER_GPUS_PER_NODE": "4",
            "ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU": "8192",
            "ROLLOUT_LOG_PROB_MAX_TOKEN_LEN_PER_GPU": "8192",
            "TEACHER_MAX_NUM_BATCHED_TOKENS": "4096",
            "RELAY_OPD_TRIGGER_TOPK": "7",
            "RELAY_OPD_MAX_TAKEOVERS": "1",
            "RELAY_OPD_PARAGRAPHS_PER_TAKEOVER": "4",
            "ROLLOUT_GPU_MEMORY_UTILIZATION": "0.45",
            "TEACHER_GPU_MEMORY_UTILIZATION": "0.45",
            "SAVE_FREQ": "16",
            "TEST_FREQ": "-1",
            "VAL_BEFORE_TRAIN": "False",
            "TOTAL_EPOCHS": "1",
            "WANDB_MODE": "disabled",
        }
    )
    command = [
        "bash",
        "opd/scripts/relay_opd/train.sh",
        "trainer.total_training_steps=16",
    ]
    print(
        "TRAINING_CONFIG "
        + json.dumps(
            {
                "student": STUDENT_REPO,
                "teacher": TEACHER_REPO,
                "train_rows": 2048,
                "heldout_rows": 512,
                "heldout_slice": "16000:16512",
                "response_budget": 1536,
                "updates": 16,
                "actor_gpus": 4,
                "teacher_gpus": 4,
                "trigger_topk": 7,
                "max_takeovers": 1,
                "paragraphs_per_takeover": 4,
                "formula_correct_trigger": False,
                "method": "released_relay_opd",
            },
            sort_keys=True,
        ),
        flush=True,
    )
    subprocess.run(command, cwd=VERL_OPD_DIR, env=env, check=True)
    checkpoint = output_dir / "global_step_16" / "actor" / "huggingface"
    if not (checkpoint / "config.json").is_file():
        raise FileNotFoundError(f"Expected full HF checkpoint at {checkpoint}")
    checkpoint_bytes = sum(
        path.stat().st_size for path in checkpoint.rglob("*") if path.is_file()
    )
    print(
        "CHECKPOINT_EVIDENCE "
        + json.dumps(
            {
                "path": str(checkpoint),
                "bytes": checkpoint_bytes,
                "global_step": 16,
                "config_present": True,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    subprocess.run(["ray", "stop", "--force"], check=False)
    eval_command = [
        "python3",
        str(REPO_ROOT / "reproduction" / "checkpoint_eval.py"),
        "--base_model",
        str(student),
        "--trained_model",
        str(checkpoint),
        "--data",
        str(heldout_path),
    ]
    subprocess.run(eval_command, cwd=REPO_ROOT, env=env, check=True)
    print(
        "TRAINING_EVIDENCE "
        + json.dumps(
            {
                "status": "PASS",
                "optimizer_updates": 16,
                "checkpoint_evaluated": True,
                "wall_seconds": time.monotonic() - started,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
