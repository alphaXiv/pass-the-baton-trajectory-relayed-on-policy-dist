#!/usr/bin/env python3
"""Compare the paper's reflection-set equation to the released sampler."""

from __future__ import annotations

import os
from types import SimpleNamespace

import torch
import torch.multiprocessing as mp


def _worker(rank: int, world_size: int) -> None:
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    os.environ.update(
        {
            "VERL_OPD_ROLLOUT_MODE": "relay",
            "VERL_OPD_SOURCE_METRICS": "1",
            "VERL_OPD_TRACE": "0",
            "RELAY_OPD_TRIGGER_TOPK": "5",
            "RELAY_OPD_MAX_TAKEOVERS": "2",
            "RELAY_OPD_PARAGRAPHS_PER_TAKEOVER": "3",
        }
    )

    from opd.patches.vllm import speculative_decode as relay

    reflection_ids = torch.tensor([1, 10], dtype=torch.long, device=device)
    relay._get_reflection_token_ids = lambda _: reflection_ids
    relay._get_paragraph_boundary_token_ids = lambda _: torch.empty(
        0, dtype=torch.long, device=device
    )
    relay._get_internal_stop_token_id = lambda _: 15
    relay._TAKEOVER_TOKENS_REMAINING.clear()
    relay._TAKEOVER_PARAGRAPHS_REMAINING.clear()
    relay._COMPLETED_TAKEOVERS.clear()
    relay.pop_opd_stats()

    vocab = 16
    target_logits = torch.full((1, vocab), -20.0, device=device)
    target_logits[0, 1] = 20.0
    student_top_ids = [10, 2, 3, 4, 5]
    draft_probs = torch.full((1, vocab), 1e-8, device=device)
    for score, token_id in enumerate(reversed(student_top_ids), start=1):
        draft_probs[0, token_id] = float(score)
    draft_probs /= draft_probs.sum(dim=-1, keepdim=True)

    output = relay._relay_opd_rejection_sample(
        draft_token_ids=torch.tensor([2], dtype=torch.int32, device=device),
        num_draft_tokens=[1],
        max_spec_len=1,
        cu_num_draft_tokens=torch.tensor([1], dtype=torch.int32, device=device),
        draft_probs=draft_probs,
        target_logits=target_logits,
        bonus_token_ids=torch.tensor([0], dtype=torch.int32, device=device),
        sampling_metadata=SimpleNamespace(req_ids=[f"audit-{rank}"]),
    )
    stats = relay.pop_opd_stats()

    paper_trigger = (1 in {1, 10}) and not bool({10, 2, 3, 4, 5} & {1, 10})
    released_trigger = output[0, 0].item() == 1
    equivalent = paper_trigger == released_trigger

    assert paper_trigger is False
    assert released_trigger is True
    assert stats["relay_divergence_triggers"] == 1.0
    print(
        "FORMULA_AUDIT "
        f"gpu={rank}/{world_size} teacher_reflection=1 "
        f"student_other_reflection_topk=1 paper_trigger={int(paper_trigger)} "
        f"released_trigger={int(released_trigger)} equivalent={int(equivalent)}",
        flush=True,
    )


def main() -> None:
    gpu_count = torch.cuda.device_count()
    if gpu_count < 1:
        raise RuntimeError("No CUDA devices visible")
    mp.spawn(_worker, args=(gpu_count,), nprocs=gpu_count, join=True)
    print(
        "FORMULA_AUDIT summary=PASS paper_set_intersection_equivalent=0 "
        "released_check=teacher_argmax_only",
        flush=True,
    )


if __name__ == "__main__":
    main()
