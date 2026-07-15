#!/usr/bin/env python3
"""Convert a pruned two-layer BP SNN into a core-group deployment package.

The converter consumes the model saved by ``onchip_stdp_prune.py`` and a
mapping snapshot written by ``coregroup_mapping.py``.  It emits both a logical
edge list and the exact AXI configuration words expected by
``snn_core_group_top``.

Example:
    python3 tests/prepare_bp_coregroup_deployment.py \
        --model data/cache/bp_prune_model_1000h_10c.npz \
        --mapping data/cache/bp_coregroup_mapping_epoch020.npz \
        --output data/cache/bp_coregroup_deployment.npz
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


FORMAT_VERSION = 2


def _scalar(data: np.lib.npyio.NpzFile, key: str, default=None):
    if key not in data.files:
        return default
    return np.asarray(data[key]).item()


def load_model(path: str) -> dict:
    data = np.load(path, allow_pickle=True)
    required = {"fc1_weight", "fc2_weight"}
    missing = required.difference(data.files)
    if missing:
        raise ValueError(f"{path} is missing model arrays: {sorted(missing)}")

    fc1 = np.asarray(data["fc1_weight"], dtype=np.float32)
    fc2 = np.asarray(data["fc2_weight"], dtype=np.float32)
    if fc1.ndim != 2 or fc2.ndim != 2 or fc2.shape[1] != fc1.shape[0]:
        raise ValueError(
            f"invalid two-layer shapes: fc1={fc1.shape}, fc2={fc2.shape}"
        )
    return {
        "fc1": fc1,
        "fc2": fc2,
        "n_input": int(fc1.shape[1]),
        "n_hidden": int(fc1.shape[0]),
        "n_output": int(fc2.shape[0]),
        "hidden_threshold": float(_scalar(data, "hidden_threshold", 1.0)),
        "output_threshold": float(_scalar(data, "output_threshold", 1.0)),
        "leak": float(_scalar(data, "leak", 0.0)),
        "timesteps": int(_scalar(data, "timesteps", 1)),
        "input_encoding": str(_scalar(data, "input_encoding", "rate_bernoulli")),
        "input_threshold": float(_scalar(data, "input_threshold", 0.3)),
        "hidden_execution": str(_scalar(
            data, "hidden_execution", "layer_synchronous")),
        "output_readout": str(_scalar(
            data, "output_readout", "spiking_threshold")),
    }


def load_mapping(path: str, model: dict) -> dict:
    data = np.load(path, allow_pickle=True)
    required = {"neuron_global_id", "edge_src", "edge_dst"}
    missing = required.difference(data.files)
    if missing:
        raise ValueError(f"{path} is missing mapping arrays: {sorted(missing)}")

    dims = (
        int(_scalar(data, "n_input", -1)),
        int(_scalar(data, "n_hidden", -1)),
        int(_scalar(data, "n_output", -1)),
    )
    expected = (model["n_input"], model["n_hidden"], model["n_output"])
    if dims != expected:
        raise ValueError(f"mapping dimensions {dims} do not match model {expected}")

    logical_to_hw = np.asarray(data["neuron_global_id"], dtype=np.int32)
    total_nodes = sum(expected)
    if logical_to_hw.shape != (total_nodes,):
        raise ValueError(
            f"neuron_global_id shape {logical_to_hw.shape} != ({total_nodes},)"
        )
    if np.unique(logical_to_hw).size != logical_to_hw.size:
        raise ValueError("mapping contains duplicate hardware neuron IDs")

    num_groups = int(_scalar(data, "num_groups", 16))
    neurons_per_group = int(_scalar(data, "neurons_per_group", 128))
    local_id_width = int(_scalar(data, "local_id_width", 7))
    capacity = num_groups * neurons_per_group
    if np.any(logical_to_hw < 0) or np.any(logical_to_hw >= capacity):
        raise ValueError(f"hardware neuron ID outside [0, {capacity})")

    src = np.asarray(data["edge_src"], dtype=np.int32)
    dst = np.asarray(data["edge_dst"], dtype=np.int32)
    if src.shape != dst.shape or src.ndim != 1:
        raise ValueError("edge_src and edge_dst must be equal-length 1-D arrays")
    if np.any(src < 0) or np.any(src >= total_nodes):
        raise ValueError("edge source is outside the logical neuron range")
    if np.any(dst < 0) or np.any(dst >= total_nodes):
        raise ValueError("edge destination is outside the logical neuron range")

    return {
        "logical_to_hw": logical_to_hw,
        "edge_src": src,
        "edge_dst": dst,
        "num_groups": num_groups,
        "neurons_per_group": neurons_per_group,
        "local_id_width": local_id_width,
    }


def edge_weights(model: dict, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    n_input = model["n_input"]
    n_hidden = model["n_hidden"]
    hidden_offset = n_input
    output_offset = n_input + n_hidden
    weights = np.empty(src.size, dtype=np.float32)

    ih = (src < n_input) & (dst >= hidden_offset) & (dst < output_offset)
    ho = ((src >= hidden_offset) & (src < output_offset) &
          (dst >= output_offset))
    if not np.all(ih | ho):
        bad = int(np.flatnonzero(~(ih | ho))[0])
        raise ValueError(
            f"edge {bad} ({int(src[bad])}->{int(dst[bad])}) is not input->hidden "
            "or hidden->output"
        )
    weights[ih] = model["fc1"][dst[ih] - hidden_offset, src[ih]]
    weights[ho] = model["fc2"][dst[ho] - output_offset, src[ho] - hidden_offset]
    return weights


def encode_ct(src_hw: int, dst_hw: int, fanout: int, magnitude: int,
              exc: bool, valid: bool, local_width: int) -> tuple[int, int]:
    local_mask = (1 << local_width) - 1
    src_group, src_local = src_hw >> local_width, src_hw & local_mask
    dst_group, dst_local = dst_hw >> local_width, dst_hw & local_mask
    addr = src_group & 0xF
    data = ((int(valid) & 1) << 31) | ((dst_group & 0xF) << 27)
    data |= (dst_local & 0x7F) << 20
    data |= (magnitude & 0xFF) << 12
    data |= (int(exc) & 1) << 11
    data |= (fanout & 0xF) << 7
    data |= src_local & 0x7F
    return addr, data


def encode_intra(src_hw: int, dst_hw: int, fanout: int, magnitude: int,
                 exc: bool, valid: bool, local_width: int) -> tuple[int, int]:
    local_mask = (1 << local_width) - 1
    src_group, src_local = src_hw >> local_width, src_hw & local_mask
    dst_group, dst_local = dst_hw >> local_width, dst_hw & local_mask
    if src_group != dst_group:
        raise ValueError("intra-group entry crosses a group boundary")
    addr = 0x1 << 28
    data = (src_local & 0x7F) << 25
    data |= (dst_local & 0x7F) << 18
    data |= ((magnitude if valid else 0) & 0xFF) << 10
    data |= (int(exc) & 1) << 9
    data |= (src_group & 0xF) << 5
    data |= fanout & 0xF
    return addr, data


def encode_score_output(class_id: int, output_hw_id: int) -> tuple[int, int]:
    """Configure one mapped output ID for the RTL score accumulator."""
    addr = 0x2 << 28
    data = (1 << 31) | ((int(class_id) & 0xF) << 24)
    data |= int(output_hw_id) & 0x7FF
    return addr, data


def build_package(model: dict, mapping: dict, max_fanout: int,
                  weight_scale: float | None, input_weight: int) -> dict:
    hidden_th = model["hidden_threshold"]
    output_th = model["output_threshold"]
    if hidden_th <= 0 or output_th <= 0:
        raise ValueError("neuron thresholds must be positive")
    if not np.isclose(hidden_th, output_th, rtol=1e-5, atol=1e-7):
        raise ValueError(
            "the RTL exposes one global threshold, but hidden_threshold="
            f"{hidden_th:g} and output_threshold={output_th:g}; retrain with equal "
            "thresholds or provide hardware with per-layer thresholds"
        )
    if not 1 <= input_weight <= 255:
        raise ValueError("input_weight must be in [1, 255]")
    if weight_scale is None:
        weight_scale = input_weight / hidden_th
    if weight_scale <= 0:
        raise ValueError("weight_scale must be positive")

    src = mapping["edge_src"]
    dst = mapping["edge_dst"]
    float_weight = edge_weights(model, src, dst)
    q_magnitude = np.clip(
        np.rint(np.abs(float_weight) * weight_scale), 0, 255
    ).astype(np.uint8)
    keep = q_magnitude > 0
    src, dst = src[keep], dst[keep]
    float_weight, q_magnitude = float_weight[keep], q_magnitude[keep]
    exc = (float_weight >= 0).astype(np.uint8)

    logical_to_hw = mapping["logical_to_hw"]
    src_hw = logical_to_hw[src]
    dst_hw = logical_to_hw[dst]
    local_width = mapping["local_id_width"]
    same_group = (src_hw >> local_width) == (dst_hw >> local_width)

    order = np.lexsort((dst, ~same_group, src))
    src, dst = src[order], dst[order]
    src_hw, dst_hw = src_hw[order], dst_hw[order]
    float_weight, q_magnitude = float_weight[order], q_magnitude[order]
    exc, same_group = exc[order], same_group[order]

    cfg_addr: list[int] = []
    cfg_data: list[int] = []
    cfg_kind: list[int] = []  # 0: CT, 1: local sparse fanout
    max_seen = {"intra": 0, "inter": 0}
    # Program every mapped source, including output or quantized-to-zero nodes.
    # Their fanout-0 terminators make repeated deployment deterministic when a
    # previous model left valid entries in either hardware table.
    for logical_src in range(logical_to_hw.size):
        selected = np.flatnonzero(src == logical_src)
        for is_local, name, kind in ((True, "intra", 1), (False, "inter", 0)):
            entries = selected[same_group[selected] == is_local]
            max_seen[name] = max(max_seen[name], int(entries.size))
            if entries.size > max_fanout:
                raise ValueError(
                    f"source {int(logical_src)} has {entries.size} {name} fanouts; "
                    f"hardware limit is {max_fanout}"
                )
            for fanout, edge_idx in enumerate(entries):
                if is_local:
                    addr, data = encode_intra(
                        int(src_hw[edge_idx]), int(dst_hw[edge_idx]), fanout,
                        int(q_magnitude[edge_idx]), bool(exc[edge_idx]), True,
                        local_width,
                    )
                else:
                    addr, data = encode_ct(
                        int(src_hw[edge_idx]), int(dst_hw[edge_idx]), fanout,
                        int(q_magnitude[edge_idx]), bool(exc[edge_idx]), True,
                        local_width,
                    )
                cfg_addr.append(addr)
                cfg_data.append(data)
                cfg_kind.append(kind)

            # A zero/invalid entry terminates scans shorter than the full table.
            if entries.size < max_fanout:
                source_hw = int(logical_to_hw[int(logical_src)])
                if is_local:
                    addr, data = encode_intra(
                        source_hw, source_hw, int(entries.size), 0, True, False,
                        local_width,
                    )
                else:
                    addr, data = encode_ct(
                        source_hw, 0, int(entries.size), 0, True, False,
                        local_width,
                    )
                cfg_addr.append(addr)
                cfg_data.append(data)
                cfg_kind.append(kind)

    n_input = model["n_input"]
    n_hidden = model["n_hidden"]
    output_hw_ids = logical_to_hw[n_input + n_hidden:]
    for class_id, output_hw_id in enumerate(output_hw_ids):
        addr, data = encode_score_output(class_id, int(output_hw_id))
        cfg_addr.append(addr)
        cfg_data.append(data)
        cfg_kind.append(2)

    hw_threshold = int(round(hidden_th * weight_scale))
    if not 1 <= hw_threshold <= 65535:
        raise ValueError(f"quantized threshold {hw_threshold} is outside uint16")

    return {
        "format_version": np.array(FORMAT_VERSION, dtype=np.int32),
        "format_name": np.array("bp_coregroup_deployment"),
        "logical_to_hw": logical_to_hw.astype(np.int32),
        "input_hw_ids": logical_to_hw[:n_input].astype(np.int32),
        "hidden_hw_ids": logical_to_hw[n_input:n_input + n_hidden].astype(np.int32),
        "output_hw_ids": output_hw_ids.astype(np.int32),
        "edge_src": src.astype(np.int32),
        "edge_dst": dst.astype(np.int32),
        "edge_src_hw": src_hw.astype(np.int32),
        "edge_dst_hw": dst_hw.astype(np.int32),
        "edge_weight_float": float_weight.astype(np.float32),
        "edge_weight": q_magnitude.astype(np.uint8),
        "edge_exc": exc.astype(np.uint8),
        "edge_same_group": same_group.astype(np.uint8),
        "cfg_addr": np.asarray(cfg_addr, dtype=np.uint32),
        "cfg_wdata": np.asarray(cfg_data, dtype=np.uint32),
        "cfg_kind": np.asarray(cfg_kind, dtype=np.uint8),
        "hw_threshold": np.array(hw_threshold, dtype=np.int32),
        "input_spike_weight": np.array(input_weight, dtype=np.uint8),
        "weight_scale": np.array(weight_scale, dtype=np.float32),
        "software_leak": np.array(model["leak"], dtype=np.float32),
        "timesteps": np.array(model["timesteps"], dtype=np.int32),
        "input_encoding": np.array(model["input_encoding"]),
        "input_threshold": np.array(model["input_threshold"], dtype=np.float32),
        "hidden_execution": np.array(model["hidden_execution"]),
        "output_readout": np.array(model["output_readout"]),
        "n_input": np.array(n_input, dtype=np.int32),
        "n_hidden": np.array(n_hidden, dtype=np.int32),
        "n_output": np.array(model["n_output"], dtype=np.int32),
        "num_groups": np.array(mapping["num_groups"], dtype=np.int32),
        "neurons_per_group": np.array(mapping["neurons_per_group"], dtype=np.int32),
        "local_id_width": np.array(local_width, dtype=np.int32),
        "max_fanout": np.array(max_fanout, dtype=np.int32),
        "max_intra_fanout_used": np.array(max_seen["intra"], dtype=np.int32),
        "max_inter_fanout_used": np.array(max_seen["inter"], dtype=np.int32),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a pruned BP SNN and mapping into core-group RTL writes"
    )
    parser.add_argument("--model", required=True, help="BP prune model .npz")
    parser.add_argument("--mapping", required=True, help="Matching core-group mapping .npz")
    parser.add_argument("--output", required=True, help="Output deployment .npz")
    parser.add_argument("--max-fanout", type=int, default=16)
    parser.add_argument(
        "--weight-scale", type=float, default=0.0,
        help="Float-to-uint8 scale; 0 maps the common threshold to input-weight",
    )
    parser.add_argument(
        "--input-weight", type=int, default=255,
        help="Excitatory weight used to make an input source neuron fire",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 1 <= args.max_fanout <= 16:
        raise SystemExit("--max-fanout must be in [1, 16] for the current RTL")

    model = load_model(args.model)
    mapping = load_mapping(args.mapping, model)
    package = build_package(
        model=model,
        mapping=mapping,
        max_fanout=args.max_fanout,
        weight_scale=None if args.weight_scale == 0 else args.weight_scale,
        input_weight=args.input_weight,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **package)

    local_edges = int(np.count_nonzero(package["edge_same_group"]))
    edge_count = int(package["edge_src"].size)
    manifest = {
        "format": str(package["format_name"].item()),
        "format_version": int(package["format_version"]),
        "model": os.path.abspath(args.model),
        "mapping": os.path.abspath(args.mapping),
        "output": os.path.abspath(output),
        "nodes": {
            "input": int(package["n_input"]),
            "hidden": int(package["n_hidden"]),
            "output": int(package["n_output"]),
        },
        "edges": {
            "total": edge_count,
            "intra_group": local_edges,
            "inter_group": edge_count - local_edges,
        },
        "configuration_writes": int(package["cfg_addr"].size),
        "quantization": {
            "weight_scale": float(package["weight_scale"]),
            "hardware_threshold": int(package["hw_threshold"]),
            "input_spike_weight": int(package["input_spike_weight"]),
        },
        "input": {
            "encoding": str(package["input_encoding"].item()),
            "threshold": float(package["input_threshold"]),
            "timesteps": int(package["timesteps"]),
        },
        "hardware": {
            "num_groups": int(package["num_groups"]),
            "neurons_per_group": int(package["neurons_per_group"]),
            "max_fanout": int(package["max_fanout"]),
            "max_intra_fanout_used": int(package["max_intra_fanout_used"]),
            "max_inter_fanout_used": int(package["max_inter_fanout_used"]),
        },
        "notes": [
            "cfg_addr/cfg_wdata are ordered AXI configuration writes.",
            "Input events target input_hw_ids with input_spike_weight.",
            "The software subtractive leak is recorded but is not exactly represented by the RTL shift leak.",
        ],
    }
    manifest_path = output.with_suffix(".json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"Saved deployment: {output}")
    print(f"Saved manifest:   {manifest_path}")
    print(f"Nodes: {sum(manifest['nodes'].values())}  Edges: {edge_count} "
          f"(local={local_edges}, inter={edge_count - local_edges})")
    print(f"Config writes: {manifest['configuration_writes']}  "
          f"threshold={manifest['quantization']['hardware_threshold']}  "
          f"scale={manifest['quantization']['weight_scale']:.6g}")
    if model["leak"] != 0:
        print("WARNING: software leak is non-zero; current RTL shift-based leak is not bit-exact.")
    if model["input_encoding"] != "deterministic_threshold" or model["timesteps"] != 1:
        print("WARNING: model input encoding/timesteps do not match the current board path; retrain with current defaults.")


if __name__ == "__main__":
    main()
