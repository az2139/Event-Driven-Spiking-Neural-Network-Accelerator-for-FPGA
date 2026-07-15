#!/usr/bin/env python3
"""Host-only checks for event-aware BP training semantics."""

from __future__ import annotations

import torch

from onchip_stdp_prune import BPPruneNet, BPPruneTrainer


def test_hidden_fires_once_and_score_readout_is_non_spiking() -> None:
    net = BPPruneNet(
        n_input=2, hidden_size=1, n_classes=2,
        hidden_threshold=1.0, output_threshold=1.0,
        leak=0.0, surrogate_scale=10.0,
    )
    with torch.no_grad():
        # Ascending event order: +1.2 fires first; the later inhibition cannot
        # retract the fire. Fire-once prevents another hidden event.
        net.fc1.weight.copy_(torch.tensor([[1.2, -1.2]]))
        net.fc2.weight.copy_(torch.tensor([[2.0], [0.5]]))
    net.eval()
    spikes = torch.ones(1, 1, 2)
    score, active = net(spikes, random_event_order=False)
    assert score.tolist() == [[2.0, 0.5]]
    assert active.tolist() == [[1.0, 0.0]]


def test_output_weights_remain_nonnegative_after_projection() -> None:
    trainer = BPPruneTrainer(
        n_input=4, hidden_size=3, n_classes=2,
        timesteps=1, device=torch.device("cpu"),
    )
    with torch.no_grad():
        trainer.net.fc2.weight.fill_(-1.0)
    trainer.enforce_hardware_constraints()
    assert torch.all(trainer.net.fc2.weight >= 0)

