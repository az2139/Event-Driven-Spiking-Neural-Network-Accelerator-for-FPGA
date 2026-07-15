#!/usr/bin/env python3
"""Generate a tiny deterministic graph in the full 784->1000->10 ID space.

The package is for RTL/software parity checks only, not accuracy measurement.
Pixels 0 and 1 target hidden neuron 0. Fire-once must limit its class-3
contribution to 255 rather than 510.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from prepare_bp_coregroup_deployment import (
    encode_ct,
    encode_intra,
    encode_score_output,
)


N_INPUT = 784
N_HIDDEN = 1000
N_OUTPUT = 10
LOCAL_WIDTH = 7
MAX_FANOUT = 16


def build_smoke_package() -> dict:
    total = N_INPUT + N_HIDDEN + N_OUTPUT
    logical_to_hw = np.arange(total, dtype=np.int32)
    hidden0 = N_INPUT
    output3 = N_INPUT + N_HIDDEN + 3
    edges = {
        0: [(hidden0, 255)],
        1: [(hidden0, 255)],
        hidden0: [(output3, 255)],
    }
    cfg_addr: list[int] = []
    cfg_data: list[int] = []
    cfg_kind: list[int] = []

    for src in range(total):
        src_hw = int(logical_to_hw[src])
        src_group = src_hw >> LOCAL_WIDTH
        local_edges = []
        cross_edges = []
        for dst, weight in edges.get(src, []):
            dst_hw = int(logical_to_hw[dst])
            target = local_edges if (dst_hw >> LOCAL_WIDTH) == src_group else cross_edges
            target.append((dst_hw, weight))

        for is_local, entries, kind in ((True, local_edges, 1), (False, cross_edges, 0)):
            for fanout, (dst_hw, weight) in enumerate(entries):
                if is_local:
                    addr, data = encode_intra(
                        src_hw, dst_hw, fanout, weight, True, True, LOCAL_WIDTH)
                else:
                    addr, data = encode_ct(
                        src_hw, dst_hw, fanout, weight, True, True, LOCAL_WIDTH)
                cfg_addr.append(addr)
                cfg_data.append(data)
                cfg_kind.append(kind)
            if len(entries) < MAX_FANOUT:
                if is_local:
                    addr, data = encode_intra(
                        src_hw, src_hw, len(entries), 0, True, False, LOCAL_WIDTH)
                else:
                    addr, data = encode_ct(
                        src_hw, 0, len(entries), 0, True, False, LOCAL_WIDTH)
                cfg_addr.append(addr)
                cfg_data.append(data)
                cfg_kind.append(kind)

    output_hw_ids = logical_to_hw[N_INPUT + N_HIDDEN:]
    for class_id, output_hw_id in enumerate(output_hw_ids):
        addr, data = encode_score_output(class_id, int(output_hw_id))
        cfg_addr.append(addr)
        cfg_data.append(data)
        cfg_kind.append(2)

    return {
        "format_version": np.array(2, dtype=np.int32),
        "format_name": np.array("bp_coregroup_deployment"),
        "logical_to_hw": logical_to_hw,
        "input_hw_ids": logical_to_hw[:N_INPUT],
        "hidden_hw_ids": logical_to_hw[N_INPUT:N_INPUT + N_HIDDEN],
        "output_hw_ids": output_hw_ids,
        "edge_src": np.array([0, 1, hidden0], dtype=np.int32),
        "edge_dst": np.array([hidden0, hidden0, output3], dtype=np.int32),
        "edge_weight": np.array([255, 255, 255], dtype=np.uint8),
        "edge_exc": np.ones(3, dtype=np.uint8),
        "edge_same_group": np.zeros(3, dtype=np.uint8),
        "cfg_addr": np.asarray(cfg_addr, dtype=np.uint32),
        "cfg_wdata": np.asarray(cfg_data, dtype=np.uint32),
        "cfg_kind": np.asarray(cfg_kind, dtype=np.uint8),
        "hw_threshold": np.array(255, dtype=np.int32),
        "input_spike_weight": np.array(255, dtype=np.uint8),
        "weight_scale": np.array(255.0, dtype=np.float32),
        "software_leak": np.array(0.0, dtype=np.float32),
        "timesteps": np.array(1, dtype=np.int32),
        "input_encoding": np.array("deterministic_threshold"),
        "input_threshold": np.array(0.3, dtype=np.float32),
        "hidden_execution": np.array("event_serial_fire_once_reset_zero"),
        "output_readout": np.array("nonnegative_weight_score_argmax"),
        "n_input": np.array(N_INPUT, dtype=np.int32),
        "n_hidden": np.array(N_HIDDEN, dtype=np.int32),
        "n_output": np.array(N_OUTPUT, dtype=np.int32),
        "num_groups": np.array(16, dtype=np.int32),
        "neurons_per_group": np.array(128, dtype=np.int32),
        "local_id_width": np.array(LOCAL_WIDTH, dtype=np.int32),
        "max_fanout": np.array(MAX_FANOUT, dtype=np.int32),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/cache/bp_score_smoke_deployment.npz")
    parser.add_argument("--dataset", default="data/cache/bp_score_smoke_dataset.npz")
    args = parser.parse_args()

    output = Path(args.output)
    dataset = Path(args.dataset)
    output.parent.mkdir(parents=True, exist_ok=True)
    dataset.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **build_smoke_package())
    image = np.zeros((1, 28, 28), dtype=np.float32)
    image[0, 0, 0] = 1.0
    image[0, 0, 1] = 1.0
    np.savez_compressed(dataset, test_imgs=image, test_lbls=np.array([3], dtype=np.int64))
    print(f"Saved smoke deployment: {output}")
    print(f"Saved smoke dataset:    {dataset}")
    print("Expected score: [0, 0, 0, 255, 0, 0, 0, 0, 0, 0], prediction=3")


if __name__ == "__main__":
    main()
