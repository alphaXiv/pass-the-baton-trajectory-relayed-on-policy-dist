#!/usr/bin/env python3
"""Run one matched standard-OPD optimizer update on the paper's models."""

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
WORK_DIR = Path("/tmp/relay-opd-one-update")
STUDENT_REPO = "Qwen/Qwen3-1.7B"
TEACHER_REPO = "Qwen/Qwen3-4B-Instruct-2507"
DATA_REPO = "BytedTsinghua-SIA/DAPO-Math-17k"
DATA_FILE = "data/dapo-math-17k.parquet"


def prepare_inputs() -> tuple[Path, Path, Path, Path]:
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
    frame = pd.read_parquet(source).head(128).copy()
    train_path = data_dir / "dapo128.parquet"
    frame.to_parquet(train_path, index=False)

    # The official trainer constructs validation datasets even when validation
    # is disabled. These schema-compatible placeholders are never evaluated.
    validation = frame.head(8).copy()
    validation.to_parquet(bench_dir / "aime-24_verl.parquet", index=False)
    validation.to_parquet(bench_dir / "aime-2025_verl.parquet", index=False)
    print(
        f"[data] source={DATA_REPO}/{DATA_FILE} rows={len(frame)} "
        f"train={train_path} validation_enabled=False",
        flush=True,
    )
    return student, teacher, train_path, output_dir


def main() -> None:
    started = time.monotonic()
    student, teacher, train_path, output_dir = prepare_inputs()
    env = os.environ.copy()
    env.update(
        {
            "STUDENT_MODEL": str(student),
            "TEACHER_MODEL": str(teacher),
            "TRAIN_DATA": str(train_path),
            "BENCH": str(WORK_DIR / "bench"),
            "OUTPUT_DIR": str(output_dir),
            "EXP_ID": "standard_opd_one_update",
            "TRAIN_BATCH_SIZE": "128",
            "PPO_MINI_BATCH_SIZE": "128",
            "MAX_PROMPT_LENGTH": "2048",
            "MAX_RESPONSE_LENGTH": "2048",
            "VAL_MAX_RESPONSE_LENGTH": "2048",
            "ROLLOUT_MAX_MODEL_LEN": "4097",
            "TEACHER_MAX_MODEL_LEN": "4097",
            "ACTOR_GPUS_PER_NODE": "4",
            "TEACHER_GPUS_PER_NODE": "4",
            "ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU": "8192",
            "ROLLOUT_LOG_PROB_MAX_TOKEN_LEN_PER_GPU": "8192",
            "TEACHER_MAX_NUM_BATCHED_TOKENS": "4096",
            "ROLLOUT_GPU_MEMORY_UTILIZATION": "0.45",
            "TEACHER_GPU_MEMORY_UTILIZATION": "0.45",
            "SAVE_FREQ": "-1",
            "TEST_FREQ": "-1",
            "VAL_BEFORE_TRAIN": "False",
            "TOTAL_EPOCHS": "1",
            "WANDB_MODE": "disabled",
        }
    )
    command = [
        "bash",
        "opd/scripts/baselines/opd.sh",
        "trainer.total_training_steps=1",
    ]
    print(
        "TRAINING_CONFIG "
        + json.dumps(
            {
                "student": STUDENT_REPO,
                "teacher": TEACHER_REPO,
                "train_rows": 128,
                "response_budget": 2048,
                "updates": 1,
                "actor_gpus": 4,
                "teacher_gpus": 4,
                "method": "standard_k1_opd",
            },
            sort_keys=True,
        ),
        flush=True,
    )
    subprocess.run(command, cwd=VERL_OPD_DIR, env=env, check=True)
    print(
        "TRAINING_EVIDENCE "
        + json.dumps(
            {
                "status": "PASS",
                "optimizer_updates": 1,
                "wall_seconds": time.monotonic() - started,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
