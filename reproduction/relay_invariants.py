#!/usr/bin/env python3
"""Exercise the released Relay sampler state machine on every visible GPU."""

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
            "RELAY_OPD_MAX_TAKEOVER_TOKENS": "256",
        }
    )

    from opd.patches.vllm import speculative_decode as relay

    reflection_ids = torch.tensor([1, 10], dtype=torch.long, device=device)
    paragraph_ids = torch.tensor([8], dtype=torch.long, device=device)
    relay._get_reflection_token_ids = lambda _: reflection_ids
    relay._get_paragraph_boundary_token_ids = lambda _: paragraph_ids
    relay._get_internal_stop_token_id = lambda _: 15

    def reset() -> None:
        relay._TAKEOVER_TOKENS_REMAINING.clear()
        relay._TAKEOVER_PARAGRAPHS_REMAINING.clear()
        relay._COMPLETED_TAKEOVERS.clear()
        relay.pop_opd_stats()

    def sample(
        request_id: str,
        student_top_ids: list[int],
        draft_tokens: tuple[int, int] = (2, 3),
    ) -> torch.Tensor:
        vocab = 16
        target_logits = torch.full((2, vocab), -20.0, device=device)
        target_logits[0, 1] = 20.0
        target_logits[1, 7] = 20.0
        draft_probs = torch.full((2, vocab), 1e-8, device=device)
        for score, token_id in enumerate(reversed(student_top_ids), start=1):
            draft_probs[0, token_id] = float(score)
        draft_probs[1, 3] = 10.0
        draft_probs /= draft_probs.sum(dim=-1, keepdim=True)
        return relay._relay_opd_rejection_sample(
            draft_token_ids=torch.tensor(draft_tokens, dtype=torch.int32, device=device),
            num_draft_tokens=[2],
            max_spec_len=2,
            cu_num_draft_tokens=torch.tensor([2], dtype=torch.int32, device=device),
            draft_probs=draft_probs,
            target_logits=target_logits,
            bonus_token_ids=torch.tensor([0], dtype=torch.int32, device=device),
            sampling_metadata=SimpleNamespace(req_ids=[request_id]),
        )

    reset()
    triggered = sample("divergent", [2, 3, 4, 5, 6])
    trigger_stats = relay.pop_opd_stats()
    assert triggered[0, 0].item() == 1
    assert triggered[0, 1].item() == relay.PLACEHOLDER_TOKEN_ID
    assert trigger_stats["relay_divergence_triggers"] == 1.0
    assert trigger_stats["relay_new_triggers"] == 1.0
    assert trigger_stats["teacher_tokens"] == 1.0
    assert relay._TAKEOVER_TOKENS_REMAINING["divergent"] == 256
    assert relay._TAKEOVER_PARAGRAPHS_REMAINING["divergent"] == 3

    reset()
    suppressed = sample("aligned", [1, 2, 3, 4, 5])
    suppression_stats = relay.pop_opd_stats()
    assert suppressed[0, :2].tolist() == [2, 3]
    assert suppression_stats["relay_divergence_triggers"] == 0.0
    assert suppression_stats["relay_teacher_argmax_in_student_topk"] == 1.0

    reset()
    request_id = "two-takeovers"
    first_trigger = sample(request_id, [2, 3, 4, 5, 6])
    relay.pop_opd_stats()
    assert first_trigger[0, 0].item() == 1

    def fake_teacher_leg(*_args, **_kwargs) -> torch.Tensor:
        return torch.tensor(
            [[8, 9, relay.PLACEHOLDER_TOKEN_ID]],
            dtype=torch.int32,
            device=device,
        )

    relay._standard_rejection_sample_output = fake_teacher_leg
    relay._TAKEOVER_TOKENS_REMAINING[request_id] = 2
    relay._TAKEOVER_PARAGRAPHS_REMAINING[request_id] = 1
    first_leg = sample(request_id, [2, 3, 4, 5, 6], (4, 5))
    first_leg_stats = relay.pop_opd_stats()
    assert first_leg[0, 0].item() == 8
    assert first_leg[0, 1].item() == relay.PLACEHOLDER_TOKEN_ID
    assert first_leg_stats["relay_takeovers_completed"] == 1.0
    assert relay._COMPLETED_TAKEOVERS[request_id] == 1

    second_trigger = sample(request_id, [2, 3, 4, 5, 6])
    relay.pop_opd_stats()
    assert second_trigger[0, 0].item() == 1
    relay._TAKEOVER_TOKENS_REMAINING[request_id] = 2
    relay._TAKEOVER_PARAGRAPHS_REMAINING[request_id] = 1
    second_leg = sample(request_id, [2, 3, 4, 5, 6], (4, 5))
    second_leg_stats = relay.pop_opd_stats()
    assert second_leg[0, :2].tolist() == [8, 15]
    assert second_leg_stats["relay_stopped_after_max_takeovers"] == 1.0
    assert request_id not in relay._COMPLETED_TAKEOVERS

    checksum = torch.arange(1, 2049, dtype=torch.float32, device=device).square().sum()
    torch.cuda.synchronize(device)
    print(
        "INVARIANT_PASS "
        f"gpu={rank}/{world_size} trigger=1 suppression=1 "
        f"teacher_leg=1 budget_M2=1 checksum={checksum.item():.0f}",
        flush=True,
    )


def main() -> None:
    gpu_count = torch.cuda.device_count()
    if gpu_count < 1:
        raise RuntimeError("No CUDA devices visible")
    mp.spawn(_worker, args=(gpu_count,), nprocs=gpu_count, join=True)
    print(
        "RELAY_INVARIANTS summary=PASS "
        f"gpus={gpu_count} K=5 M=2 L=3 official_sampler=1",
        flush=True,
    )


if __name__ == "__main__":
    main()
