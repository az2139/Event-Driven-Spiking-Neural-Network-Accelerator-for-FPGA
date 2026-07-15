#!/usr/bin/env python3
"""Generate a nonnegative random graph for RTL/software score parity testing."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from prepare_bp_coregroup_deployment import encode_ct, encode_intra, encode_score_output


N_INPUT = 784
N_HIDDEN = 1000
N_OUTPUT = 10
TOTAL_NODES = N_INPUT + N_HIDDEN + N_OUTPUT
NUM_GROUPS = 16
NEURONS_PER_GROUP = 128
LOCAL_WIDTH = 7
MAX_FANOUT = 16
THRESHOLD = 255


def build_graph(seed: int, active_hidden: int, input_fanout: int,
                output_fanout: int) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(seed)
    hidden_nodes = N_INPUT + np.arange(active_hidden, dtype=np.int32)
    output_nodes = N_INPUT + N_HIDDEN + np.arange(N_OUTPUT, dtype=np.int32)
    src: list[int] = []
    dst: list[int] = []
    weight: list[int] = []

    for input_id in range(N_INPUT):
        targets = rng.choice(hidden_nodes, size=input_fanout, replace=False)
        for target in targets:
            src.append(input_id)
            dst.append(int(target))
            weight.append(int(rng.integers(48, 129)))
    for hidden_id in hidden_nodes:
        targets = rng.choice(output_nodes, size=output_fanout, replace=False)
        for target in targets:
            src.append(int(hidden_id))
            dst.append(int(target))
            weight.append(int(rng.integers(32, 128)))
    return (
        np.asarray(src, dtype=np.int32),
        np.asarray(dst, dtype=np.int32),
        np.asarray(weight, dtype=np.uint8),
    )


def build_config(logical_to_hw: np.ndarray, edge_src: np.ndarray,
                 edge_dst: np.ndarray, edge_weight: np.ndarray) -> tuple[np.ndarray, ...]:
    src_hw = logical_to_hw[edge_src]
    dst_hw = logical_to_hw[edge_dst]
    same_group = (src_hw >> LOCAL_WIDTH) == (dst_hw >> LOCAL_WIDTH)
    cfg_addr: list[int] = []
    cfg_data: list[int] = []
    cfg_kind: list[int] = []

    for logical_src in range(TOTAL_NODES):
        selected = np.flatnonzero(edge_src == logical_src)
        for is_local, kind in ((True, 1), (False, 0)):
            entries = selected[same_group[selected] == is_local]
            if len(entries) > MAX_FANOUT:
                raise ValueError(f"source {logical_src} exceeds hardware fanout")
            for fanout, edge_idx in enumerate(entries):
                if is_local:
                    addr, data = encode_intra(
                        int(src_hw[edge_idx]), int(dst_hw[edge_idx]), fanout,
                        int(edge_weight[edge_idx]), True, True, LOCAL_WIDTH)
                else:
                    addr, data = encode_ct(
                        int(src_hw[edge_idx]), int(dst_hw[edge_idx]), fanout,
                        int(edge_weight[edge_idx]), True, True, LOCAL_WIDTH)
                cfg_addr.append(addr)
                cfg_data.append(data)
                cfg_kind.append(kind)
            if len(entries) < MAX_FANOUT:
                source_hw = int(logical_to_hw[logical_src])
                if is_local:
                    addr, data = encode_intra(
                        source_hw, source_hw, len(entries), 0, True, False, LOCAL_WIDTH)
                else:
                    addr, data = encode_ct(
                        source_hw, 0, len(entries), 0, True, False, LOCAL_WIDTH)
                cfg_addr.append(addr)
                cfg_data.append(data)
                cfg_kind.append(kind)

    output_hw_ids = logical_to_hw[N_INPUT + N_HIDDEN:]
    for class_id, output_hw_id in enumerate(output_hw_ids):
        addr, data = encode_score_output(class_id, int(output_hw_id))
        cfg_addr.append(addr)
        cfg_data.append(data)
        cfg_kind.append(2)
    return (
        np.asarray(cfg_addr, dtype=np.uint32),
        np.asarray(cfg_data, dtype=np.uint32),
        np.asarray(cfg_kind, dtype=np.uint8),
        same_group.astype(np.uint8),
    )


def software_scores(images: np.ndarray, edge_src: np.ndarray,
                    edge_dst: np.ndarray, edge_weight: np.ndarray) -> np.ndarray:
    ih = edge_src < N_INPUT
    ho = (edge_src >= N_INPUT) & (edge_src < N_INPUT + N_HIDDEN)
    scores = np.zeros((len(images), N_OUTPUT), dtype=np.int64)
    for sample, image in enumerate(images):
        active = image.reshape(-1) > 0.3
        hidden_mem = np.zeros(N_HIDDEN, dtype=np.int64)
        ih_idx = np.flatnonzero(ih & active[np.minimum(edge_src, N_INPUT - 1)])
        np.add.at(hidden_mem, edge_dst[ih_idx] - N_INPUT, edge_weight[ih_idx])
        hidden_fire = hidden_mem >= THRESHOLD
        ho_candidates = np.flatnonzero(ho)
        ho_idx = ho_candidates[hidden_fire[edge_src[ho_candidates] - N_INPUT]]
        np.add.at(
            scores[sample], edge_dst[ho_idx] - N_INPUT - N_HIDDEN,
            edge_weight[ho_idx])
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--active-hidden", type=int, default=128)
    parser.add_argument("--input-fanout", type=int, default=6)
    parser.add_argument("--output-fanout", type=int, default=4)
    parser.add_argument("--pixel-probability", type=float, default=0.18)
    parser.add_argument("--output", default="data/cache/bp_random_parity_deployment.npz")
    parser.add_argument("--dataset", default="data/cache/bp_random_parity_dataset.npz")
    args = parser.parse_args()
    if not 1 <= args.input_fanout <= MAX_FANOUT:
        parser.error("--input-fanout must be in [1, 16]")
    if not 1 <= args.output_fanout <= min(MAX_FANOUT, N_OUTPUT):
        parser.error("--output-fanout must be in [1, 10]")
    if not 1 <= args.active_hidden <= N_HIDDEN:
        parser.error("--active-hidden must be in [1, 1000]")

    rng = np.random.default_rng(args.seed)
    # Random injective placement in the 2048-neuron hardware ID space.
    logical_to_hw = rng.choice(
        NUM_GROUPS * NEURONS_PER_GROUP, size=TOTAL_NODES, replace=False
    ).astype(np.int32)
    edge_src, edge_dst, edge_weight = build_graph(
        args.seed + 1, args.active_hidden, args.input_fanout, args.output_fanout)
    cfg_addr, cfg_wdata, cfg_kind, same_group = build_config(
        logical_to_hw, edge_src, edge_dst, edge_weight)
    images = (rng.random((args.samples, 28, 28)) < args.pixel_probability).astype(np.float32)
    expected_scores = software_scores(images, edge_src, edge_dst, edge_weight)
    labels = expected_scores.argmax(axis=1).astype(np.int64)

    package = {
        "format_version": np.array(2, dtype=np.int32),
        "format_name": np.array("bp_coregroup_deployment"),
        "logical_to_hw": logical_to_hw,
        "input_hw_ids": logical_to_hw[:N_INPUT],
        "hidden_hw_ids": logical_to_hw[N_INPUT:N_INPUT + N_HIDDEN],
        "output_hw_ids": logical_to_hw[N_INPUT + N_HIDDEN:],
        "edge_src": edge_src,
        "edge_dst": edge_dst,
        "edge_weight": edge_weight,
        "edge_exc": np.ones(len(edge_src), dtype=np.uint8),
        "edge_same_group": same_group,
        "cfg_addr": cfg_addr,
        "cfg_wdata": cfg_wdata,
        "cfg_kind": cfg_kind,
        "hw_threshold": np.array(THRESHOLD, dtype=np.int32),
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
        "num_groups": np.array(NUM_GROUPS, dtype=np.int32),
        "neurons_per_group": np.array(NEURONS_PER_GROUP, dtype=np.int32),
        "local_id_width": np.array(LOCAL_WIDTH, dtype=np.int32),
        "max_fanout": np.array(MAX_FANOUT, dtype=np.int32),
        "parity_seed": np.array(args.seed, dtype=np.int32),
    }
    output = Path(args.output)
    dataset = Path(args.dataset)
    output.parent.mkdir(parents=True, exist_ok=True)
    dataset.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **package)
    np.savez_compressed(
        dataset, test_imgs=images, test_lbls=labels,
        expected_scores=expected_scores)
    print(f"Saved random deployment: {output}")
    print(f"Saved random dataset:    {dataset}")
    print(f"seed={args.seed} samples={args.samples} edges={len(edge_src)} "
          f"local={int(same_group.sum())} cross={int(len(same_group)-same_group.sum())} "
          f"writes={len(cfg_addr)}")


if __name__ == "__main__":
    main()
