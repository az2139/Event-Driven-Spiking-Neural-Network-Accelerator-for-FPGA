#!/usr/bin/env python3
"""Small host-only checks for the 784->1000->10 board reference helpers."""

from __future__ import annotations

import numpy as np
import sys

import fpga_10class_coregroup_inference as cg
from prepare_bp_coregroup_deployment import encode_score_output


def _deployment() -> dict:
    # 2 inputs -> 2 hidden -> 2 outputs, threshold 5.
    return {
        "logical_to_hw": np.arange(6, dtype=np.int32),
        "input_hw_ids": np.array([0, 1], dtype=np.int32),
        "output_hw_ids": np.array([4, 5], dtype=np.int32),
        "edge_src": np.array([0, 1, 2, 3], dtype=np.int32),
        "edge_dst": np.array([2, 3, 4, 5], dtype=np.int32),
        "edge_weight": np.array([5, 5, 5, 5], dtype=np.uint8),
        "edge_exc": np.ones(4, dtype=np.uint8),
        "hw_threshold": np.array(5, dtype=np.int32),
        "input_spike_weight": np.array(5, dtype=np.uint8),
        "input_threshold": np.array(0.3, dtype=np.float32),
        "n_input": np.array(2, dtype=np.int32),
        "n_hidden": np.array(2, dtype=np.int32),
        "n_output": np.array(2, dtype=np.int32),
    }


def test_bp_references_and_output_decode() -> None:
    deployment = _deployment()
    image = np.array([1.0, 0.0], dtype=np.float32)
    sync = cg.bp_sync_reference(image, deployment)
    event = cg.bp_event_reference(image, deployment, fire_once=True)
    assert sync["pred"] == 0
    assert sync["hidden_spikes"] == 1
    assert event["pred"] == 0
    assert event["output_order"] == [0]

    decoded = cg.classify_bp_output_spikes(
        [{"neuron_id": 99}, {"neuron_id": 5}, {"neuron_id": 4}],
        deployment["output_hw_ids"],
    )
    assert decoded["order"] == [1, 0]
    assert decoded["pred"] == 1


def test_bp_input_packet_uses_hls_axis_width() -> None:
    deployment = _deployment()
    words = cg.build_bp_input_spike_words(
        np.array([1.0, 0.0], dtype=np.float32),
        deployment["input_hw_ids"], 0.3, 255, 13,
    )
    assert words.tolist() == [255 << 13]


def test_score_output_configuration_word() -> None:
    addr, data = encode_score_output(class_id=7, output_hw_id=0x345)
    assert addr >> 28 == 2
    assert (data >> 31) & 1 == 1
    assert (data >> 24) & 0xF == 7
    assert data & 0x7FF == 0x345


def test_event_hidden_score_readout_reference() -> None:
    deployment = _deployment()
    deployment["output_readout"] = np.array("nonnegative_weight_score_argmax")
    ref = cg.bp_event_reference(
        np.array([1.0, 0.0], dtype=np.float32), deployment, fire_once=True)
    assert ref["pred"] == 0
    assert ref["fired_hidden"] == 1
    assert ref["output_score"] == [5, 0]


def test_main_auto_dispatches_bp_package(tmp_path, monkeypatch) -> None:
    package = tmp_path / "bp.npz"
    np.savez(package, format_name=np.array("bp_coregroup_deployment"))
    called = {}

    def fake_run(args, deployment, deploy_path, *unused):
        called["path"] = deploy_path
        called["format"] = str(deployment["format_name"].item())

    monkeypatch.setattr(cg, "run_bp_deployment", fake_run)
    monkeypatch.setattr(
        sys, "argv",
        ["fpga_10class_coregroup_inference.py", "--weights", str(package),
         "--no-program", "--profile-output", ""],
    )
    cg.main()
    assert called == {"path": str(package), "format": "bp_coregroup_deployment"}


def test_resolve_data_path_prefers_data_directory(tmp_path) -> None:
    data_dir = str(tmp_path)
    assert cg.resolve_data_path(data_dir, None, "model.npz") == str(tmp_path / "model.npz")
    assert cg.resolve_data_path(data_dir, "model.npz", "unused") == str(tmp_path / "model.npz")


if __name__ == "__main__":
    test_bp_references_and_output_decode()
    print("BP core-group board reference: PASS")
