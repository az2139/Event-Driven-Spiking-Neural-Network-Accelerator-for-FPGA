#!/usr/bin/env python3
"""
10-class MNIST inference for the hierarchical core-group/profile bitstream.

This script is intentionally separate from fpga_10class_inference.py.  The old
script programs the legacy spike_router connection-memory format.  The
core-group route uses event_router_ng, direct external delivery, a sparse
connectivity table (CT), and per-group intra weight memories.
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import os
import re
import struct
import sys
import time

import numpy as np

import fpga_10class_inference as legacy

# AXI-Lite bases are fixed by rebuild_core_group_integrated.tcl.
HLS_BASE = legacy.HLS_BASE
CFG_BASE = legacy.CFG_BASE
DMA_BASE = legacy.DMA_BASE

HLS_AP_CTRL = legacy.HLS_AP_CTRL
HLS_CTRL_REG = legacy.HLS_CTRL_REG
HLS_CONFIG_REG = legacy.HLS_CONFIG_REG
HLS_MODE_REG = legacy.HLS_MODE_REG
HLS_TIME_STEPS = legacy.HLS_TIME_STEPS
HLS_STATUS_REG = legacy.HLS_STATUS_REG
HLS_SPIKE_COUNT = legacy.HLS_SPIKE_COUNT
HLS_VERSION_REG = legacy.HLS_VERSION_REG

CFG_CONFIG_CTRL = legacy.CFG_CONFIG_CTRL
CFG_CONFIG_ADDR = legacy.CFG_CONFIG_ADDR
CFG_CONFIG_WDATA = legacy.CFG_CONFIG_WDATA
CFG_CONFIG_RDATA = legacy.CFG_CONFIG_RDATA
CFG_THRESHOLD = legacy.CFG_THRESHOLD
CFG_NEURON_PARAMS = legacy.CFG_NEURON_PARAMS
CFG_ROUTER_SPKS = legacy.CFG_ROUTER_SPKS
CFG_NEURON_SPKS = legacy.CFG_NEURON_SPKS
CFG_STATUS = legacy.CFG_STATUS
CFG_THROUGHPUT = legacy.CFG_THROUGHPUT
CFG_VERSION = legacy.CFG_VERSION
CFG_SERVICE_CYCLES = legacy.CFG_SERVICE_CYCLES
CFG_PROFILE_CTRL = legacy.CFG_PROFILE_CTRL
CFG_PROFILE_INDEX = legacy.CFG_PROFILE_INDEX
CFG_PROFILE_DATA = legacy.CFG_PROFILE_DATA
CFG_PROFILE_INFO = legacy.CFG_PROFILE_INFO

DMA_MM2S_DMASR = legacy.DMA_MM2S_DMASR
DMA_S2MM_DMASR = legacy.DMA_S2MM_DMASR

CTRL_ENABLE = legacy.CTRL_ENABLE
CTRL_FIRST_SPIKE_ONLY = legacy.CTRL_FIRST_SPIKE_ONLY
EXPECTED_HLS_VERSION = legacy.EXPECTED_HLS_VERSION

DMA_BUF_IN = legacy.DMA_BUF_IN
DMA_BUF_OUT = legacy.DMA_BUF_OUT

DEFAULT_GROUPS = 16
DEFAULT_LOCAL_ID_WIDTH = 7
DEFAULT_GROUP_ID_WIDTH = 4
DEFAULT_HLS_PACKET_ID_WIDTH = legacy.HLS_SPIKE_PKT_ID_W
DEFAULT_PL_CLK_MHZ = legacy.PL_CLK_MHZ_DEFAULT
DEFAULT_MAX_FANOUT_INTER = 16


def require_compatible_legacy() -> None:
    sig = inspect.signature(legacy.run_inference)
    required = ["wait_hls_input_count", "hls_input_timeout_s"]
    missing = [name for name in required if name not in sig.parameters]
    if missing:
        legacy_path = getattr(legacy, "__file__", "UNKNOWN")
        print("ERROR: fpga_10class_coregroup_inference.py is newer than the imported "
              "fpga_10class_inference.py helper.")
        print(f"  imported helper: {legacy_path}")
        print(f"  missing run_inference() args: {', '.join(missing)}")
        print("  Sync tests/fpga_10class_inference.py together with "
              "tests/fpga_10class_coregroup_inference.py on the board.")
        sys.exit(1)


def read_profile_class_debug(cfg: legacy.MMIO, num_groups: int, n_classes: int) -> dict:
    """Read router/top class profile windows directly from the detailed v9 layout."""
    result: dict[str, list[int]] = {}
    profile_info = int(cfg.read(CFG_PROFILE_INFO))
    profile_count = profile_info & 0xFFFF
    profile_version = (profile_info >> 24) & 0xFF
    result["available"] = False
    if profile_version >= 10:
        return result

    router_score_base = 11 + 2 * int(num_groups)
    router_event_base = router_score_base + int(n_classes)
    core_base = router_event_base + int(n_classes)
    core_metric_count = 6
    classifier_base = core_base + int(num_groups) * core_metric_count
    class_count_base = classifier_base + 3
    top_score_base = class_count_base + int(n_classes)
    top_event_base = top_score_base + int(n_classes)
    if profile_count < top_event_base + int(n_classes):
        return result

    def read_window(base: int) -> list[int]:
        values = []
        for cls in range(int(n_classes)):
            cfg.write(CFG_PROFILE_INDEX, base + cls)
            values.append(int(cfg.read(CFG_PROFILE_DATA)))
        return values

    result["router_class_scores"] = read_window(router_score_base)
    result["router_class_events"] = read_window(router_event_base)
    result["top_class_scores"] = read_window(top_score_base)
    result["top_class_events"] = read_window(top_event_base)
    result["available"] = True
    return result


def parse_hwh_metadata(hwh_path: str) -> dict:
    meta = {
        "path": hwh_path,
        "exists": os.path.exists(hwh_path),
        "hls_modtype": "UNKNOWN",
        "hls_vlnv": "UNKNOWN",
        "apnone_id_width": 11,
        "has_axi_dma_1": False,
        "has_weight_stream": False,
        "has_learn_weight": False,
        "has_s_axis_data": False,
    }
    if not meta["exists"]:
        return meta
    with open(hwh_path, "r", encoding="utf-8", errors="ignore") as f:
        txt = f.read()

    m = re.search(r'INSTANCE="snn_top_hls_0"[^>]*MODTYPE="([^"]+)"[^>]*VLNV="([^"]+)"', txt)
    if m:
        meta["hls_modtype"] = m.group(1)
        meta["hls_vlnv"] = m.group(2)

    w = legacy.detect_hls_spike_id_width(hwh_path, default=11)
    meta["apnone_id_width"] = int(w)
    meta["has_axi_dma_1"] = "axi_dma_1" in txt
    meta["has_weight_stream"] = ("s_axis_weights" in txt) or ("m_axis_weights" in txt)
    meta["has_learn_weight"] = "learn_weight" in txt
    meta["has_s_axis_data"] = "s_axis_data" in txt
    return meta


def _parse_report_number(value: str) -> int | float | None:
    value = str(value).strip().replace(",", "")
    if value in ("", "---", "NA", "Unspecified*"):
        return None
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return None


def parse_vivado_utilization_report(path: str | None) -> dict:
    """Extract a compact LUT/FF/BRAM/DSP summary from report_utilization."""
    info = {
        "path": path,
        "available": False,
        "lut": None,
        "ff": None,
        "bram": None,
        "dsp": None,
        "lut_util_percent": None,
        "ff_util_percent": None,
        "bram_util_percent": None,
        "dsp_util_percent": None,
    }
    if not path or not os.path.exists(path):
        return info

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        txt = f.read()

    patterns = {
        "lut": r"\|\s*Slice LUTs\s*\|\s*([0-9,]+)\s*\|[^|]*\|[^|]*\|\s*([0-9,]+)\s*\|\s*([0-9.]+)",
        "ff": r"\|\s*Slice Registers\s*\|\s*([0-9,]+)\s*\|[^|]*\|[^|]*\|\s*([0-9,]+)\s*\|\s*([0-9.]+)",
        "bram": r"\|\s*Block RAM Tile\s*\|\s*([0-9,]+)\s*\|[^|]*\|[^|]*\|\s*([0-9,]+)\s*\|\s*([0-9.]+)",
        "dsp": r"\|\s*DSPs\s*\|\s*([0-9,]+)\s*\|[^|]*\|[^|]*\|\s*([0-9,]+)\s*\|\s*([0-9.]+)",
    }
    for key, pat in patterns.items():
        m = re.search(pat, txt)
        if not m:
            continue
        info[key] = int(str(m.group(1)).replace(",", ""))
        info[f"{key}_available"] = int(str(m.group(2)).replace(",", ""))
        info[f"{key}_util_percent"] = float(m.group(3))
        info["available"] = True
    return info


def parse_vivado_power_report(path: str | None) -> dict:
    """Extract Vivado power summary and estimate PL dynamic by excluding PS7."""
    info = {
        "path": path,
        "available": False,
        "total_on_chip_w": None,
        "dynamic_w": None,
        "device_static_w": None,
        "ps7_dynamic_w": None,
        "pl_dynamic_w": None,
        "confidence": None,
    }
    if not path or not os.path.exists(path):
        return info

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        txt = f.read()

    fields = {
        "total_on_chip_w": r"\|\s*Total On-Chip Power \(W\)\s*\|\s*([0-9.]+)",
        "dynamic_w": r"\|\s*Dynamic \(W\)\s*\|\s*([0-9.]+)",
        "device_static_w": r"\|\s*Device Static \(W\)\s*\|\s*([0-9.]+)",
        "confidence": r"\|\s*Confidence Level\s*\|\s*([^|]+?)\s*\|",
    }
    for key, pat in fields.items():
        m = re.search(pat, txt)
        if not m:
            continue
        value = m.group(1).strip()
        info[key] = value if key == "confidence" else float(value)

    ps = re.search(r"\|\s*processing_system7_0\s*\|\s*([0-9.]+)\s*\|", txt)
    if not ps:
        ps = re.search(r"\|\s*PS7\s*\|\s*([0-9.]+)\s*\|", txt)
    if ps:
        info["ps7_dynamic_w"] = float(ps.group(1))

    if info["dynamic_w"] is not None:
        if info["ps7_dynamic_w"] is not None:
            info["pl_dynamic_w"] = max(0.0, info["dynamic_w"] - info["ps7_dynamic_w"])
        else:
            info["pl_dynamic_w"] = info["dynamic_w"]
    info["available"] = any(info[k] is not None for k in ("total_on_chip_w", "dynamic_w", "pl_dynamic_w"))
    return info


def default_report_path(data_dir: str, filename: str) -> str | None:
    candidates = [
        os.path.join(data_dir, filename),
        os.path.join(os.getcwd(), "outputs", filename),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def decode_global_id(nid: int, local_id_width: int) -> tuple[int, int]:
    return int(nid >> local_id_width), int(nid & ((1 << local_id_width) - 1))


def group_hist_from_hw_ids(hw_ids, local_id_width: int, num_groups: int) -> list[int]:
    hist = [0 for _ in range(int(num_groups))]
    for hw_id in hw_ids:
        group, _ = decode_global_id(int(hw_id), int(local_id_width))
        if 0 <= group < int(num_groups):
            hist[group] += 1
    return hist


def build_neuron_mapping(n_neurons: int,
                         num_groups: int,
                         local_id_width: int,
                         mode: str) -> tuple[np.ndarray, dict[int, int]]:
    """Map logical classifier neuron IDs to hardware global neuron IDs."""
    mode = str(mode).lower()
    if mode == "contiguous":
        logical_to_hw = np.arange(n_neurons, dtype=np.int32)
    elif mode == "round-robin":
        group_capacity = 1 << int(local_id_width)
        logical_to_hw = np.zeros(n_neurons, dtype=np.int32)
        for logical_id in range(n_neurons):
            group = logical_id % int(num_groups)
            local = logical_id // int(num_groups)
            if local >= group_capacity:
                raise ValueError(
                    f"round-robin map exceeds local capacity: logical_id={logical_id} "
                    f"group={group} local={local} capacity={group_capacity}"
                )
            logical_to_hw[logical_id] = (group << int(local_id_width)) | local
    else:
        raise ValueError(f"unsupported neuron map mode: {mode}")

    hw_to_logical = {int(hw): int(logical) for logical, hw in enumerate(logical_to_hw)}
    if len(hw_to_logical) != int(n_neurons):
        raise ValueError(f"neuron map {mode} produced duplicate hardware IDs")
    return logical_to_hw, hw_to_logical


def classify_hw_first_spike_mapped(result: dict,
                                   hw_to_logical: dict[int, int],
                                   n_classes: int,
                                   fps_per_class: int) -> int | None:
    """Decode the first classifiable S2MM spike through the logical classifier map."""
    spikes = result.get("output_spikes", [])
    for spike in spikes:
        hw_id = int(spike.get("neuron_id", -1))
        logical_id = hw_to_logical.get(hw_id)
        if logical_id is None:
            continue
        pred = int(logical_id) // int(fps_per_class)
        if 0 <= pred < int(n_classes):
            return pred
    return None


def classify_profile_first_classifier(result: dict,
                                      hw_to_logical: dict[int, int],
                                      n_classes: int,
                                      fps_per_class: int) -> int | None:
    """Decode RTL-latched first classifier spike from the profile snapshot."""
    profile = result.get("profile", {})
    if int(profile.get("first_classifier_spike_valid", 0) or 0) == 0:
        return None
    hw_id = int(profile.get("first_classifier_spike_id", -1))
    logical_id = hw_to_logical.get(hw_id)
    if logical_id is None:
        return None
    pred = int(logical_id) // int(fps_per_class)
    if 0 <= pred < int(n_classes):
        return pred
    return None


def classify_profile_count(result: dict, n_classes: int) -> tuple[int | None, list[int]]:
    """Decode count-based classifier result from RTL profile class counters."""
    profile = result.get("profile", {})
    counts = [
        int(profile.get(f"class_{cls}_spike_count", 0) or 0)
        for cls in range(int(n_classes))
    ]
    if max(counts) <= 0:
        return None, counts
    return int(np.argmax(np.asarray(counts))), counts


def classify_profile_score(result: dict, n_classes: int) -> tuple[int | None, list[int], list[int]]:
    """Decode router-delivered weighted class-score readout from profile."""
    profile = result.get("profile", {})
    scores = [
        int(profile.get(f"class_{cls}_score", 0) or 0)
        for cls in range(int(n_classes))
    ]
    events = [
        int(profile.get(f"class_{cls}_event_count", 0) or 0)
        for cls in range(int(n_classes))
    ]
    if max(scores) <= 0:
        return None, scores, events
    return int(np.argmax(np.asarray(scores))), scores, events


def classify_count_with_score_tiebreak(counts: list[int],
                                       scores: list[int]) -> int | None:
    """Select max count; when tied, select max score among tied classes."""
    if not counts:
        return None
    counts_i = [int(x) for x in counts]
    scores_i = [int(x) for x in scores]
    if max(counts_i) <= 0:
        return None
    max_count = max(counts_i)
    candidates = [idx for idx, val in enumerate(counts_i) if val == max_count]
    if len(candidates) == 1:
        return int(candidates[0])
    return int(max(candidates, key=lambda idx: scores_i[idx] if idx < len(scores_i) else 0))


def direct_sw_count_by_class(potential: np.ndarray,
                             n_classes: int,
                             fps_per_class: int) -> tuple[int | None, list[int]]:
    """SW per-class count baseline for direct external-target inference."""
    fired = np.asarray(potential) > 0
    counts = [0 for _ in range(int(n_classes))]
    sums = [0 for _ in range(int(n_classes))]
    for nid, did_fire in enumerate(fired):
        cls = int(nid) // int(fps_per_class)
        if 0 <= cls < int(n_classes):
            if bool(did_fire):
                counts[cls] += 1
            sums[cls] += int(potential[int(nid)])
    if max(counts) <= 0:
        return None, counts
    pred = int(np.argmax(np.asarray(counts)))
    if counts.count(counts[pred]) > 1:
        pred = int(np.argmax(np.asarray(sums)))
    return pred, counts


def build_dummy_sink_ids(logical_to_hw: np.ndarray,
                         num_groups: int,
                         local_id_width: int) -> list[int | None]:
    """Pick one unused high local ID per group for non-classifying CT fanout sinks."""
    group_capacity = 1 << int(local_id_width)
    used = [set() for _ in range(int(num_groups))]
    for hw_id in logical_to_hw:
        group, local = decode_global_id(int(hw_id), local_id_width)
        if 0 <= group < int(num_groups):
            used[group].add(local)

    sinks: list[int | None] = []
    for group in range(int(num_groups)):
        sink_local = None
        for local in range(group_capacity - 1, -1, -1):
            if local not in used[group]:
                sink_local = local
                break
        sinks.append(None if sink_local is None else ((group << int(local_id_width)) | sink_local))
    return sinks


def parse_ct_mode(ct_mode: str) -> dict:
    mode = str(ct_mode).strip().lower().replace("_", "-")
    if mode in ("none", "terminator-only"):
        return {"kind": "none", "fanout": 0, "canonical": "none"}
    if mode in ("same", "same-group", "dummy-same"):
        return {"kind": "same", "fanout": 1, "canonical": "dummy-same"}

    m = re.fullmatch(r"(?:dummy-)?cross-?(\d+)", mode)
    if m:
        fanout = int(m.group(1))
        if fanout < 1:
            raise ValueError("cross fanout must be >= 1")
        return {"kind": "cross", "fanout": fanout, "canonical": f"dummy-cross{fanout}"}

    if mode in ("random", "dummy-random", "sparse-p"):
        return {"kind": "random", "fanout": None, "canonical": "dummy-random"}

    if mode in ("real-full", "real"):
        return {"kind": "real", "variant": "full", "fanout": None, "canonical": "real-full"}
    if mode in ("real-random25", "real-random-25"):
        return {"kind": "real", "variant": "random", "density": 0.25, "fanout": None,
                "canonical": "real-random25"}
    if mode in ("real-random", "real-sparse"):
        return {"kind": "real", "variant": "random", "density": None, "fanout": None,
                "canonical": "real-random"}

    raise ValueError(f"unsupported ct_mode: {ct_mode}")


def build_real_fanout_mask(q_weights: np.ndarray,
                           classifier_logical_to_hw: np.ndarray,
                           input_logical_to_hw: np.ndarray,
                           local_id_width: int,
                           ct_mode: str,
                           real_density: float = 0.25,
                           random_seed: int = 1,
                           max_fanout_inter: int = DEFAULT_MAX_FANOUT_INTER,
                           overflow_policy: str = "topk") -> dict:
    """Build the exact pixel-source fanout list used by real CT modes."""
    mode = parse_ct_mode(ct_mode)
    if mode["kind"] != "real":
        raise ValueError(f"build_real_fanout_mask requires real mode, got {ct_mode}")

    max_valid_fanout = max(0, int(max_fanout_inter) - 1)
    if max_valid_fanout <= 0:
        raise ValueError("MAX_FANOUT_INTER must leave at least one valid fanout slot")

    if mode["variant"] == "full":
        density = 1.0
    else:
        density = float(mode.get("density") if mode.get("density") is not None else real_density)
    if not (0.0 <= density <= 1.0):
        raise ValueError("--real-density must be in [0, 1]")

    overflow_policy = str(overflow_policy).lower()
    if overflow_policy not in ("topk", "error"):
        raise ValueError("--real-overflow-policy must be topk or error")

    rng = np.random.default_rng(int(random_seed))
    q = np.asarray(q_weights)
    n_neurons, n_pixels = q.shape
    if len(classifier_logical_to_hw) < n_neurons or len(input_logical_to_hw) < n_pixels:
        raise ValueError("real CT mapping arrays are too small")

    fanouts: list[list[tuple[int, int, int, int]]] = []
    valid_entries = 0
    cross_entries = 0
    same_entries = 0
    pre_prune_entries = 0
    truncated_entries = 0
    source_with_overflow = 0

    for pixel_idx in range(n_pixels):
        src_hw = int(input_logical_to_hw[pixel_idx])
        src_group, _ = decode_global_id(src_hw, local_id_width)
        entries = []
        for neuron_idx in range(n_neurons):
            weight = int(q[neuron_idx, pixel_idx])
            if weight <= 0:
                continue
            if density < 1.0 and rng.random() >= density:
                continue
            dst_hw = int(classifier_logical_to_hw[neuron_idx])
            entries.append((int(neuron_idx), dst_hw, min(weight, 255)))

        pre_prune_entries += len(entries)
        entries.sort(key=lambda item: (-item[2], item[0]))
        if len(entries) > max_valid_fanout:
            source_with_overflow += 1
            if overflow_policy == "error":
                raise ValueError(
                    f"pixel source {pixel_idx} has {len(entries)} fanouts, "
                    f"but hardware supports {max_valid_fanout}; use --real-overflow-policy topk "
                    "or rebuild with larger MAX_FANOUT_INTER"
                )
            truncated_entries += len(entries) - max_valid_fanout
            entries = entries[:max_valid_fanout]

        packed_entries = []
        for neuron_idx, dst_hw, weight in entries:
            dst_group, _ = decode_global_id(dst_hw, local_id_width)
            valid_entries += 1
            if dst_group == src_group:
                same_entries += 1
            else:
                cross_entries += 1
            packed_entries.append((int(neuron_idx), int(dst_hw), int(weight), int(dst_group)))
        fanouts.append(packed_entries)

    return {
        "fanouts": fanouts,
        "valid_entries": int(valid_entries),
        "invalid_terminators": int(n_pixels),
        "cross_entries": int(cross_entries),
        "same_entries": int(same_entries),
        "real_density": float(density),
        "real_pre_prune_entries": int(pre_prune_entries),
        "real_truncated_entries": int(truncated_entries),
        "real_sources_with_overflow": int(source_with_overflow),
        "real_overflow_policy": overflow_policy,
    }


def sw_ct_pruned_predict(image: np.ndarray,
                         fanout_mask: dict,
                         hw_threshold: int,
                         n_classes: int,
                         fps_per_class: int,
                         pixel_th: float = 0.3,
                         input_source_weight: int = 0x7F,
                         active_pixels_override: np.ndarray | None = None) -> int | None:
    """Functional SW emulation of the real CT path.

    This mirrors the architectural behavior, not cycle timing:
      pixel event -> input-source neuron threshold -> CT fanout ->
      classifier membrane accumulate -> first classifier threshold crossing.

    It intentionally does not model router RR, group FIFO timing, stalls, leak,
    or refractory timing. The runtime config currently sets leak=0/refrac=0.
    """
    fanouts = fanout_mask.get("fanouts") if fanout_mask is not None else None
    if fanouts is None:
        return None

    n_classifier = int(n_classes) * int(fps_per_class)
    classifier_mem = np.zeros(n_classifier, dtype=np.int32)
    source_mem = np.zeros(len(fanouts), dtype=np.int32)
    active_pixels = (np.asarray(active_pixels_override, dtype=np.int64)
                     if active_pixels_override is not None
                     else np.flatnonzero(image.reshape(-1) > pixel_th))
    for pixel_idx in active_pixels:
        # The board injects an event into the input-source neuron first. Only a
        # fired input-source neuron becomes a CT lookup source.
        source_mem[int(pixel_idx)] += int(input_source_weight)
        if source_mem[int(pixel_idx)] < int(hw_threshold):
            continue
        source_mem[int(pixel_idx)] = 0

        for neuron_idx, _dst_hw, weight, _dst_group in fanouts[int(pixel_idx)]:
            if 0 <= int(neuron_idx) < n_classifier:
                classifier_mem[int(neuron_idx)] += int(weight)
                if classifier_mem[int(neuron_idx)] >= int(hw_threshold):
                    classifier_mem[int(neuron_idx)] = 0
                    pred = int(neuron_idx) // int(fps_per_class)
                    if 0 <= pred < int(n_classes):
                        return pred
    return None


def sw_ct_pruned_count_predict(image: np.ndarray,
                               fanout_mask: dict,
                               hw_threshold: int,
                               n_classes: int,
                               fps_per_class: int,
                               pixel_th: float = 0.3,
                               input_source_weight: int = 0x7F,
                               active_pixels_override: np.ndarray | None = None) -> tuple[int | None, list[int]]:
    """Count-based functional SW emulation of the real CT path."""
    fanouts = fanout_mask.get("fanouts") if fanout_mask is not None else None
    if fanouts is None:
        return None, [0 for _ in range(int(n_classes))]

    n_classifier = int(n_classes) * int(fps_per_class)
    classifier_mem = np.zeros(n_classifier, dtype=np.int32)
    source_mem = np.zeros(len(fanouts), dtype=np.int32)
    class_counts = [0 for _ in range(int(n_classes))]
    active_pixels = (np.asarray(active_pixels_override, dtype=np.int64)
                     if active_pixels_override is not None
                     else np.flatnonzero(image.reshape(-1) > pixel_th))

    for pixel_idx in active_pixels:
        source_mem[int(pixel_idx)] += int(input_source_weight)
        if source_mem[int(pixel_idx)] < int(hw_threshold):
            continue
        source_mem[int(pixel_idx)] = 0

        for neuron_idx, _dst_hw, weight, _dst_group in fanouts[int(pixel_idx)]:
            if 0 <= int(neuron_idx) < n_classifier:
                classifier_mem[int(neuron_idx)] += int(weight)
                if classifier_mem[int(neuron_idx)] >= int(hw_threshold):
                    classifier_mem[int(neuron_idx)] = 0
                    pred = int(neuron_idx) // int(fps_per_class)
                    if 0 <= pred < int(n_classes):
                        class_counts[pred] += 1

    if max(class_counts) <= 0:
        return None, class_counts
    return int(np.argmax(np.asarray(class_counts))), class_counts


def sw_ct_pruned_score_predict(image: np.ndarray,
                               fanout_mask: dict,
                               hw_threshold: int,
                               n_classes: int,
                               fps_per_class: int,
                               pixel_th: float = 0.3,
                               input_source_weight: int = 0x7F,
                               active_pixels_override: np.ndarray | None = None) -> tuple[int | None, list[int], list[int]]:
    """Weighted class-score SW reference for router-delivered CT fanouts."""
    fanouts = fanout_mask.get("fanouts") if fanout_mask is not None else None
    if fanouts is None:
        return None, [0 for _ in range(int(n_classes))], [0 for _ in range(int(n_classes))]

    source_mem = np.zeros(len(fanouts), dtype=np.int32)
    class_scores = [0 for _ in range(int(n_classes))]
    class_events = [0 for _ in range(int(n_classes))]
    active_pixels = (np.asarray(active_pixels_override, dtype=np.int64)
                     if active_pixels_override is not None
                     else np.flatnonzero(image.reshape(-1) > pixel_th))

    for pixel_idx in active_pixels:
        source_mem[int(pixel_idx)] += int(input_source_weight)
        if source_mem[int(pixel_idx)] < int(hw_threshold):
            continue
        source_mem[int(pixel_idx)] = 0

        for neuron_idx, _dst_hw, weight, _dst_group in fanouts[int(pixel_idx)]:
            cls = int(neuron_idx) // int(fps_per_class)
            if 0 <= cls < int(n_classes):
                class_scores[cls] += int(weight)
                class_events[cls] += 1

    if max(class_scores) <= 0:
        return None, class_scores, class_events
    return int(np.argmax(np.asarray(class_scores))), class_scores, class_events


def select_sw_active_pixels_for_valid_budget(image: np.ndarray,
                                             fanout_mask: dict,
                                             target_valid_events: int,
                                             pixel_th: float = 0.3) -> tuple[np.ndarray, int]:
    """Select input-order active pixels whose fanouts match the HW valid budget."""
    fanouts = fanout_mask.get("fanouts") if fanout_mask is not None else None
    active_pixels = np.flatnonzero(image.reshape(-1) > pixel_th)
    if fanouts is None or int(target_valid_events) <= 0 or active_pixels.size == 0:
        return np.asarray([], dtype=np.int64), 0

    selected = []
    valid_sum = 0
    for pixel_idx in active_pixels:
        fanout_count = len(fanouts[int(pixel_idx)])
        if valid_sum + fanout_count > int(target_valid_events):
            break
        selected.append(int(pixel_idx))
        valid_sum += int(fanout_count)
        if valid_sum == int(target_valid_events):
            break
    return np.asarray(selected, dtype=np.int64), int(valid_sum)


def encode_ct_entry(src_global: int,
                    fanout_idx: int,
                    dst_global: int,
                    weight: int,
                    valid: bool = True,
                    exc: bool = True,
                    local_id_width: int = DEFAULT_LOCAL_ID_WIDTH) -> tuple[int, int]:
    """Encode snn_core_group_top connectivity-table write format."""
    src_group, src_neuron = decode_global_id(src_global, local_id_width)
    dst_group, dst_neuron = decode_global_id(dst_global, local_id_width)
    addr = (0x0 << 28) | (src_group & 0xF)
    data = ((1 if valid else 0) << 31)
    data |= (dst_group & 0xF) << 27
    data |= (dst_neuron & 0x7F) << 20
    data |= (int(weight) & 0xFF) << 12
    data |= (1 if exc else 0) << 11
    data |= (int(fanout_idx) & 0xF) << 7
    data |= src_neuron & 0x7F
    return addr, data


def encode_intra_sparse_entry(src_global: int,
                              fanout_idx: int,
                              dst_global: int,
                              weight: int,
                              valid: bool = True,
                              exc: bool = True,
                              local_id_width: int = DEFAULT_LOCAL_ID_WIDTH) -> tuple[int, int]:
    """Encode one sparse intra-group fanout-table entry.

    The RTL stores intra entries as {valid, exc, weight, dst_local}. The
    current config word has no separate valid bit for intra writes, so
    valid=False is represented as weight=0 terminator.
    """
    src_group, src_neuron = decode_global_id(src_global, local_id_width)
    dst_group, dst_neuron = decode_global_id(dst_global, local_id_width)
    if src_group != dst_group:
        raise ValueError("intra sparse entry requires src and dst in the same group")
    addr = (0x1 << 28)
    data = (src_neuron & 0x7F) << 25
    data |= (dst_neuron & 0x7F) << 18
    data |= ((int(weight) if valid else 0) & 0xFF) << 10
    data |= (1 if exc else 0) << 9
    data |= (src_group & 0xF) << 5
    data |= int(fanout_idx) & 0xF
    return addr, data


def encode_intra_weight(src_global: int,
                        dst_global: int,
                        weight: int,
                        exc: bool = True,
                        local_id_width: int = DEFAULT_LOCAL_ID_WIDTH,
                        fanout_idx: int = 0) -> tuple[int, int]:
    """Backward-compatible wrapper for the intra config word."""
    return encode_intra_sparse_entry(
        src_global=src_global,
        fanout_idx=fanout_idx,
        dst_global=dst_global,
        weight=weight,
        valid=(int(weight) != 0),
        exc=exc,
        local_id_width=local_id_width,
    )


def cfg_write_pair(cfg: legacy.MMIO, addr: int, data: int) -> None:
    cfg.write(CFG_CONFIG_CTRL, 0)
    cfg.write(CFG_CONFIG_ADDR, addr)
    # CONFIG_WDATA triggers the actual RTL config write using CONFIG_ADDR.
    # Readbacks provide an ordering barrier for /dev/mem mmap stores so large
    # CT programming sweeps cannot race WDATA ahead of the intended address.
    cfg.read(CFG_CONFIG_ADDR)
    cfg.write(CFG_CONFIG_WDATA, data)
    cfg.read(CFG_CONFIG_WDATA)


def clear_ct_fanout0(cfg: legacy.MMIO,
                     n_neurons: int,
                     local_id_width: int = DEFAULT_LOCAL_ID_WIDTH,
                     logical_to_hw: np.ndarray | None = None) -> None:
    """Write invalid fanout-0 terminators for all output neurons used here."""
    for nid in range(n_neurons):
        hw_nid = int(logical_to_hw[nid]) if logical_to_hw is not None else int(nid)
        addr, data = encode_ct_entry(
            src_global=hw_nid,
            fanout_idx=0,
            dst_global=0,
            weight=0,
            valid=False,
            exc=True,
            local_id_width=local_id_width,
        )
        cfg_write_pair(cfg, addr, data)


def clear_ct_all_fanout0(cfg: legacy.MMIO,
                         num_groups: int,
                         local_id_width: int = DEFAULT_LOCAL_ID_WIDTH) -> int:
    writes = 0
    group_capacity = 1 << int(local_id_width)
    for src_group in range(int(num_groups)):
        for src_local in range(group_capacity):
            addr, data = encode_ct_entry(
                src_global=(src_group << int(local_id_width)) | src_local,
                fanout_idx=0,
                dst_global=0,
                weight=0,
                valid=False,
                exc=True,
                local_id_width=local_id_width,
            )
            cfg_write_pair(cfg, addr, data)
            writes += 1
    return writes


def program_real_ct_fanout(cfg: legacy.MMIO,
                           q_weights: np.ndarray,
                           classifier_logical_to_hw: np.ndarray,
                           input_logical_to_hw: np.ndarray,
                           local_id_width: int,
                           ct_mode: str,
                           real_density: float = 0.25,
                           random_seed: int = 1,
                           max_fanout_inter: int = DEFAULT_MAX_FANOUT_INTER,
                           overflow_policy: str = "topk",
                           intra_mode: str = "off") -> dict:
    """Program CT with real pixel-source -> classifier-neuron positive weights."""
    mode = parse_ct_mode(ct_mode)
    if mode["kind"] != "real":
        raise ValueError(f"program_real_ct_fanout requires real mode, got {ct_mode}")
    intra_mode = str(intra_mode).strip().lower().replace("_", "-")
    split_same_group = intra_mode == "split-same-group"
    if intra_mode not in ("off", "split-same-group"):
        raise ValueError(f"unsupported intra_mode for real CT fanout: {intra_mode}")

    mask = build_real_fanout_mask(
        q_weights=q_weights,
        classifier_logical_to_hw=classifier_logical_to_hw,
        input_logical_to_hw=input_logical_to_hw,
        local_id_width=local_id_width,
        ct_mode=ct_mode,
        real_density=real_density,
        random_seed=random_seed,
        max_fanout_inter=max_fanout_inter,
        overflow_policy=overflow_policy,
    )

    intra_entries = 0
    intra_split_same_entries = 0
    for pixel_idx, entries in enumerate(mask["fanouts"]):
        src_hw = int(input_logical_to_hw[pixel_idx])
        src_group, _ = decode_global_id(src_hw, local_id_width)
        ct_entries = []
        intra_fanout_idx = 0
        for neuron_idx, dst_hw, weight, dst_group in entries:
            dst_group_decoded, _ = decode_global_id(int(dst_hw), local_id_width)
            if split_same_group and int(dst_group_decoded) == int(src_group):
                addr, data = encode_intra_sparse_entry(
                    src_global=src_hw,
                    fanout_idx=intra_fanout_idx,
                    dst_global=int(dst_hw),
                    weight=int(weight),
                    valid=True,
                    exc=True,
                    local_id_width=local_id_width,
                )
                cfg_write_pair(cfg, addr, data)
                intra_entries += 1
                intra_fanout_idx += 1
            else:
                ct_entries.append((neuron_idx, int(dst_hw), int(weight), int(dst_group)))

        if split_same_group:
            if intra_fanout_idx >= int(max_fanout_inter):
                raise ValueError(
                    f"sparse intra fanout for pixel source {pixel_idx} uses "
                    f"{intra_fanout_idx} valid entries, leaving no terminator slot"
                )
            src_dummy_global = int(src_hw)
            src_group_for_term, _src_local_for_term = decode_global_id(src_dummy_global, local_id_width)
            term_dst_global = src_group_for_term << int(local_id_width)
            addr, data = encode_intra_sparse_entry(
                src_global=src_dummy_global,
                fanout_idx=intra_fanout_idx,
                dst_global=term_dst_global,
                weight=0,
                valid=False,
                exc=True,
                local_id_width=local_id_width,
            )
            cfg_write_pair(cfg, addr, data)

        for fanout_idx, (_neuron_idx, dst_hw, weight, _dst_group) in enumerate(ct_entries):
            addr, data = encode_ct_entry(
                src_global=src_hw,
                fanout_idx=fanout_idx,
                dst_global=dst_hw,
                weight=weight,
                valid=True,
                exc=True,
                local_id_width=local_id_width,
            )
            cfg_write_pair(cfg, addr, data)

        addr, data = encode_ct_entry(
            src_global=src_hw,
            fanout_idx=len(ct_entries),
            dst_global=0,
            weight=0,
            valid=False,
            exc=True,
            local_id_width=local_id_width,
        )
        cfg_write_pair(cfg, addr, data)
        intra_split_same_entries += len(entries) - len(ct_entries)

    return {
        "valid_entries": int(mask["valid_entries"] - intra_entries),
        "invalid_terminators": int(mask["invalid_terminators"]),
        "cross_entries": int(mask["cross_entries"]),
        "same_entries": int(0 if split_same_group else mask["same_entries"]),
        "intra_entries": int(intra_entries),
        "intra_split_same_entries": int(intra_split_same_entries),
        "real_density": float(mask["real_density"]),
        "real_pre_prune_entries": int(mask["real_pre_prune_entries"]),
        "real_truncated_entries": int(mask["real_truncated_entries"]),
        "real_sources_with_overflow": int(mask["real_sources_with_overflow"]),
        "real_overflow_policy": str(mask["real_overflow_policy"]),
        "fanout_mask": mask,
    }


def program_ct_dummy_fanout(cfg: legacy.MMIO,
                            n_neurons: int,
                            local_id_width: int,
                            logical_to_hw: np.ndarray,
                            ct_mode: str,
                            dummy_sink_ids: list[int | None],
                            dummy_weight: int = 0,
                            fanout_density: float = 1.0,
                            random_seed: int = 1,
                            max_fanout_inter: int = DEFAULT_MAX_FANOUT_INTER) -> tuple[int, int, int, int]:
    """Program harmless valid CT fanouts plus invalid terminators for non-dummy sources."""
    mode = parse_ct_mode(ct_mode)
    num_groups = len(dummy_sink_ids)
    max_valid_fanout = max(0, int(max_fanout_inter) - 1)
    valid_entries = 0
    invalid_terminators = 0
    cross_entries = 0
    same_entries = 0
    rng = np.random.default_rng(int(random_seed))
    group_capacity = 1 << int(local_id_width)

    if mode["kind"] in ("same", "cross") and int(mode["fanout"]) > max_valid_fanout:
        raise ValueError(
            f"{mode['canonical']} requests {mode['fanout']} valid fanouts, "
            f"but max supported before terminator is {max_valid_fanout}"
        )

    density = float(fanout_density)
    if not (0.0 <= density <= 1.0):
        raise ValueError("--fanout-density must be in [0, 1]")

    # Cover every possible non-dummy source in each group. This keeps the dummy
    # communication load orthogonal to the logical classifier mapping and avoids
    # under-counting if a non-classifier local source emits an output spike.
    for src_group in range(num_groups):
        src_dummy_hw = dummy_sink_ids[src_group]
        src_dummy_local = None
        if src_dummy_hw is not None:
            _, src_dummy_local = decode_global_id(int(src_dummy_hw), local_id_width)
        for src_local in range(group_capacity):
            if src_dummy_local is not None and src_local == src_dummy_local:
                addr, data = encode_ct_entry(
                    src_global=(src_group << int(local_id_width)) | src_local,
                    fanout_idx=0,
                    dst_global=0,
                    weight=0,
                    valid=False,
                    exc=True,
                    local_id_width=local_id_width,
                )
                cfg_write_pair(cfg, addr, data)
                invalid_terminators += 1
                continue

            src_hw = (src_group << int(local_id_width)) | src_local

            dst_groups: list[int] = []
            if mode["kind"] == "same":
                dst_groups = [src_group]
            elif mode["kind"] == "cross":
                dst_groups = [
                    (src_group + offset) % num_groups
                    for offset in range(1, int(mode["fanout"]) + 1)
                ]
            elif mode["kind"] == "random":
                candidates = [
                    (src_group + offset) % num_groups
                    for offset in range(1, num_groups)
                ]
                rng.shuffle(candidates)
                for dst_group in candidates[:max_valid_fanout]:
                    if rng.random() < density:
                        dst_groups.append(int(dst_group))
            else:
                raise ValueError(f"unsupported dummy CT mode: {ct_mode}")

            for fanout_idx, dst_group in enumerate(dst_groups):
                dst_hw = dummy_sink_ids[dst_group]
                if dst_hw is None:
                    raise ValueError(
                        f"no free dummy sink in group {dst_group}; use round-robin mapping "
                        "or reduce classifier neurons per group"
                    )

                addr, data = encode_ct_entry(
                    src_global=src_hw,
                    fanout_idx=fanout_idx,
                    dst_global=int(dst_hw),
                    weight=int(dummy_weight),
                    valid=True,
                    exc=True,
                    local_id_width=local_id_width,
                )
                cfg_write_pair(cfg, addr, data)
                valid_entries += 1
                if dst_group != src_group:
                    cross_entries += 1
                else:
                    same_entries += 1

            addr, data = encode_ct_entry(
                src_global=src_hw,
                fanout_idx=len(dst_groups),
                dst_global=0,
                weight=0,
                valid=False,
                exc=True,
                local_id_width=local_id_width,
            )
            cfg_write_pair(cfg, addr, data)
            invalid_terminators += 1

    return valid_entries, invalid_terminators, cross_entries, same_entries


def clear_intra_rows(cfg: legacy.MMIO,
                     n_neurons: int,
                     local_id_width: int = DEFAULT_LOCAL_ID_WIDTH,
                     group_size: int = 128,
                     logical_to_hw: np.ndarray | None = None) -> int:
    """Clear local recurrent rows for neurons used by the 10-class model."""
    writes = 0
    for src in range(n_neurons):
        src_hw = int(logical_to_hw[src]) if logical_to_hw is not None else int(src)
        src_group, _ = decode_global_id(src_hw, local_id_width)
        row_start = src_group << int(local_id_width)
        for local_dst in range(group_size):
            dst = row_start | local_dst
            addr, data = encode_intra_weight(src_hw, dst, 0, True, local_id_width)
            cfg_write_pair(cfg, addr, data)
            writes += 1
    return writes


def clear_intra_hw_rows(cfg: legacy.MMIO,
                        source_hw_ids,
                        local_id_width: int = DEFAULT_LOCAL_ID_WIDTH,
                        group_size: int = 128) -> int:
    """Clear local recurrent rows for explicit hardware-global source IDs."""
    writes = 0
    for src_hw_raw in source_hw_ids:
        src_hw = int(src_hw_raw)
        src_group, _ = decode_global_id(src_hw, local_id_width)
        row_start = src_group << int(local_id_width)
        for local_dst in range(int(group_size)):
            dst = row_start | local_dst
            addr, data = encode_intra_weight(src_hw, dst, 0, True, local_id_width)
            cfg_write_pair(cfg, addr, data)
            writes += 1
    return writes


def clear_intra_sparse_rows(cfg: legacy.MMIO,
                            source_hw_ids,
                            local_id_width: int = DEFAULT_LOCAL_ID_WIDTH,
                            max_fanout_inter: int = DEFAULT_MAX_FANOUT_INTER) -> int:
    """Clear sparse intra fanout slots for explicit hardware-global source IDs."""
    writes = 0
    for src_hw_raw in source_hw_ids:
        src_hw = int(src_hw_raw)
        src_group, _ = decode_global_id(src_hw, local_id_width)
        term_dst_global = src_group << int(local_id_width)
        for fanout_idx in range(int(max_fanout_inter)):
            addr, data = encode_intra_sparse_entry(
                src_global=src_hw,
                fanout_idx=fanout_idx,
                dst_global=term_dst_global,
                weight=0,
                valid=False,
                exc=True,
                local_id_width=local_id_width,
            )
            cfg_write_pair(cfg, addr, data)
            writes += 1
    return writes


def clear_intra_sparse_fanout0_all(cfg: legacy.MMIO,
                                   num_groups: int,
                                   local_id_width: int = DEFAULT_LOCAL_ID_WIDTH) -> int:
    """Install a fanout-0 terminator for every possible local source row."""
    writes = 0
    group_capacity = 1 << int(local_id_width)
    for src_group in range(int(num_groups)):
        term_dst_global = src_group << int(local_id_width)
        for src_local in range(group_capacity):
            src_hw = (src_group << int(local_id_width)) | src_local
            addr, data = encode_intra_sparse_entry(
                src_global=src_hw,
                fanout_idx=0,
                dst_global=term_dst_global,
                weight=0,
                valid=False,
                exc=True,
                local_id_width=local_id_width,
            )
            cfg_write_pair(cfg, addr, data)
            writes += 1
    return writes


def program_coregroup_for_inference(cfg: legacy.MMIO,
                                    n_neurons: int,
                                    local_id_width: int,
                                    q_weights: np.ndarray | None = None,
                                    logical_to_hw: np.ndarray | None = None,
                                    input_logical_to_hw: np.ndarray | None = None,
                                    ct_mode: str = "terminator-only",
                                    dummy_sink_ids: list[int | None] | None = None,
                                    dummy_weight: int = 0,
                                    fanout_density: float = 1.0,
                                    real_density: float = 0.25,
                                    random_seed: int = 1,
                                    max_fanout_inter: int = DEFAULT_MAX_FANOUT_INTER,
                                    real_overflow_policy: str = "topk",
                                    clear_intra: bool = False,
                                    intra_mode: str = "off") -> dict:
    """Configure the new route for direct external delivery inference."""
    t0 = time.perf_counter()
    mode = parse_ct_mode(ct_mode)
    ct_mode = mode["canonical"]
    intra_mode_requested = str(intra_mode).strip().lower().replace("_", "-")
    if intra_mode_requested == "auto":
        intra_mode = "split-same-group" if mode["kind"] == "real" else "off"
    else:
        intra_mode = intra_mode_requested
    if intra_mode not in ("off", "split-same-group"):
        raise ValueError(f"unsupported intra_mode: {intra_mode}")
    if intra_mode != "off" and mode["kind"] != "real":
        raise ValueError("--intra-mode is currently supported only with real CT modes")
    if intra_mode == "split-same-group" and clear_intra:
        raise ValueError("--clear-intra would erase split-same-group weights; omit --clear-intra")
    valid_entries = 0
    cross_entries = 0
    same_entries = 0
    real_cfg = {}
    intra_writes = 0
    intra_nonzero_entries = 0
    if mode["kind"] == "none":
        invalid_terminators = clear_ct_all_fanout0(
            cfg,
            num_groups=(len(dummy_sink_ids) if dummy_sink_ids is not None else DEFAULT_GROUPS),
            local_id_width=local_id_width,
        )
    elif mode["kind"] == "real":
        if q_weights is None or logical_to_hw is None or input_logical_to_hw is None:
            raise ValueError("real CT modes require q_weights, logical_to_hw, and input_logical_to_hw")
        clear_ct_all_fanout0(
            cfg,
            num_groups=(len(dummy_sink_ids) if dummy_sink_ids is not None else DEFAULT_GROUPS),
            local_id_width=local_id_width,
        )
        if intra_mode == "split-same-group":
            intra_writes += clear_intra_sparse_fanout0_all(
                cfg,
                num_groups=(len(dummy_sink_ids) if dummy_sink_ids is not None else DEFAULT_GROUPS),
                local_id_width=local_id_width,
            )
        real_cfg = program_real_ct_fanout(
            cfg,
            q_weights=q_weights,
            classifier_logical_to_hw=logical_to_hw,
            input_logical_to_hw=input_logical_to_hw,
            local_id_width=local_id_width,
            ct_mode=ct_mode,
            real_density=real_density,
            random_seed=random_seed,
            max_fanout_inter=max_fanout_inter,
            overflow_policy=real_overflow_policy,
            intra_mode=intra_mode,
        )
        valid_entries = int(real_cfg["valid_entries"])
        invalid_terminators = int(real_cfg["invalid_terminators"])
        cross_entries = int(real_cfg["cross_entries"])
        same_entries = int(real_cfg["same_entries"])
        intra_nonzero_entries = int(real_cfg.get("intra_entries", 0))
    elif mode["kind"] in ("same", "cross", "random"):
        if logical_to_hw is None or dummy_sink_ids is None:
            raise ValueError("dummy CT modes require logical_to_hw and dummy_sink_ids")
        valid_entries, invalid_terminators, cross_entries, same_entries = program_ct_dummy_fanout(
            cfg,
            n_neurons=n_neurons,
            local_id_width=local_id_width,
            logical_to_hw=logical_to_hw,
            ct_mode=ct_mode,
            dummy_sink_ids=dummy_sink_ids,
            dummy_weight=dummy_weight,
            fanout_density=fanout_density,
            random_seed=random_seed,
            max_fanout_inter=max_fanout_inter,
        )
    else:
        raise ValueError(f"unsupported ct_mode: {ct_mode}")

    if clear_intra:
        intra_writes += clear_intra_rows(
            cfg, n_neurons, local_id_width=local_id_width, logical_to_hw=logical_to_hw
        )
    return {
        "ct_mode": ct_mode,
        "ct_valid_entries": int(valid_entries),
        "ct_invalid_terminators": int(invalid_terminators),
        "ct_cross_entries": int(cross_entries),
        "ct_same_entries": int(same_entries),
        "ct_dummy_weight": int(dummy_weight),
        "ct_fanout_density": float(fanout_density),
        "ct_random_seed": int(random_seed),
        "ct_max_fanout_inter": int(max_fanout_inter),
        "ct_pattern_kind": mode["kind"],
        "ct_pattern_variant": mode.get("variant"),
        "ct_pattern_fanout": (None if mode["fanout"] is None else int(mode["fanout"])),
        "ct_real_density": float(real_density),
        "ct_real_overflow_policy": str(real_overflow_policy),
        "ct_real_pre_prune_entries": int(real_cfg["real_pre_prune_entries"]) if mode["kind"] == "real" else 0,
        "ct_real_truncated_entries": int(real_cfg["real_truncated_entries"]) if mode["kind"] == "real" else 0,
        "ct_real_sources_with_overflow": int(real_cfg["real_sources_with_overflow"]) if mode["kind"] == "real" else 0,
        "ct_real_fanout_mask": real_cfg.get("fanout_mask") if mode["kind"] == "real" else None,
        "intra_mode": intra_mode,
        "intra_mode_requested": intra_mode_requested,
        "intra_nonzero_entries": int(intra_nonzero_entries),
        "intra_clear_writes": int(intra_writes),
        "ct_expected_dummy_entries": int(
            (valid_entries if mode["kind"] == "random" else
             ((1 << int(local_id_width)) * len(dummy_sink_ids) - len(dummy_sink_ids)) *
             (int(mode["fanout"]) if mode["fanout"] is not None else 0))
            if dummy_sink_ids is not None and mode["kind"] != "none" else 0
        ),
        "intra_zero_writes": int(intra_writes),
        "config_ms": (time.perf_counter() - t0) * 1000.0,
    }


def build_direct_spike_words(image: np.ndarray,
                             q_weights: np.ndarray,
                             hls_spike_pkt_id_w: int,
                             pixel_th: float = 0.3,
                             potential: np.ndarray | None = None,
                             logical_to_hw: np.ndarray | None = None) -> np.ndarray:
    """Emit TTFS-ordered classifier events using mapped hardware global IDs."""
    if potential is None:
        potential = legacy.compute_positive_potential(image, q_weights, pixel_th=pixel_th)
    order = legacy.ttfs_order_from_potential(potential)
    pos_order = order[potential[order] > 0]
    if pos_order.size == 0:
        return np.zeros(1, dtype=np.uint32)
    hw_order = logical_to_hw[pos_order] if logical_to_hw is not None else pos_order
    id_mask = (1 << hls_spike_pkt_id_w) - 1
    return (hw_order.astype(np.uint32) & np.uint32(id_mask)) | np.uint32(0x7F << hls_spike_pkt_id_w)


def real_input_active_pixels(image: np.ndarray,
                             pixel_th: float = 0.3) -> np.ndarray:
    """Return the Python-owned unique input-source set for one sample."""
    # np.flatnonzero returns sorted unique indexes for a bitmap condition.
    # np.unique keeps the uniqueness contract explicit if this helper later
    # accepts a non-bitmap source list.
    return np.unique(np.flatnonzero(image.reshape(-1) > pixel_th)).astype(np.int64)


def build_real_input_spike_words(image: np.ndarray,
                                 input_logical_to_hw: np.ndarray,
                                 hls_spike_pkt_id_w: int,
                                 pixel_th: float = 0.3) -> np.ndarray:
    """Emit one event per active input-source neuron in deterministic pixel order.

    The router no longer keeps an input_source_seen bitmap. Real CT modes rely
    on this software-side source set to be unique per sample.
    """
    active_pixels = real_input_active_pixels(image, pixel_th=pixel_th)
    if active_pixels.size == 0:
        return np.zeros(1, dtype=np.uint32)
    hw_order = input_logical_to_hw[active_pixels]
    id_mask = (1 << hls_spike_pkt_id_w) - 1
    return (hw_order.astype(np.uint32) & np.uint32(id_mask)) | np.uint32(0x7F << hls_spike_pkt_id_w)


def decode_cfg_status(status: int) -> dict:
    return {
        "fifo_overflow": bool(status & 0x1),
        "router_busy": bool(status & legacy.STATUS_ROUTER_BUSY),
        "any_core_group_busy": bool(status & legacy.STATUS_GROUP_BUSY),
        "rtl_snn_ready": bool(status & legacy.STATUS_SNN_READY),
        "profile_active": bool(status & legacy.STATUS_PROFILE_ACTIVE),
        "profile_done": bool(status & legacy.STATUS_PROFILE_DONE),
        "active_neurons": int((status >> 6) & 0xFF),
    }


def fmt_bool(v: bool) -> str:
    return "1" if v else "0"


def row_profile_value(row: dict, name: str) -> int:
    return int(row.get("profile", {}).get(name, 0) or 0)


def ct_mode_expected_counts(output_count: int, ct_mode: str) -> dict | None:
    mode = parse_ct_mode(ct_mode)
    if mode["kind"] == "none":
        return {
            "lookup": output_count,
            "valid": 0,
            "invalid": output_count,
            "cross": 0,
            "same": 0,
        }
    if mode["kind"] == "same":
        return {
            "lookup": output_count * (int(mode["fanout"]) + 1),
            "valid": output_count,
            "invalid": output_count,
            "cross": 0,
            "same": output_count,
        }
    if mode["kind"] == "cross":
        fanout = int(mode["fanout"])
        return {
            "lookup": output_count * (fanout + 1),
            "valid": output_count * fanout,
            "invalid": output_count,
            "cross": output_count * fanout,
            "same": 0,
        }
    if mode["kind"] == "random":
        return None
    if mode["kind"] == "real":
        return None
    raise ValueError(f"unsupported ct_mode: {ct_mode}")


def ct_mode_row_ok(row: dict, ct_mode: str) -> bool:
    output_count = row_profile_value(row, "output_spike_count")
    expected = ct_mode_expected_counts(output_count, ct_mode)
    if expected is None:
        return True
    return (
        row_profile_value(row, "ct_lookup_count") == expected["lookup"] and
        row_profile_value(row, "ct_valid_entry_count") == expected["valid"] and
        row_profile_value(row, "ct_invalid_entry_count") == expected["invalid"] and
        row_profile_value(row, "cross_group_event_count") == expected["cross"] and
        row_profile_value(row, "same_group_event_count") == expected["same"]
    )


def print_diagnostics(hls: legacy.MMIO,
                      cfg: legacy.MMIO,
                      dma: legacy.MMIO,
                      hwh_meta: dict,
                      profile_info: int | None) -> None:
    cfg_status = cfg.read(CFG_STATUS)
    hls_status = hls.read(HLS_STATUS_REG)
    mm2s = dma.read(DMA_MM2S_DMASR)
    s2mm = dma.read(DMA_S2MM_DMASR)
    ds = decode_cfg_status(cfg_status)
    hs = legacy.decode_hls_status(hls_status)

    print("\nRuntime interface diagnostics:")
    print(f"  HWH: {hwh_meta['path']}")
    print(f"  HLS MODTYPE: {hwh_meta['hls_modtype']}  VLNV: {hwh_meta['hls_vlnv']}")
    print(f"  HWH ports: apnone_id_width={hwh_meta['apnone_id_width']} "
          f"axi_dma_1={int(hwh_meta['has_axi_dma_1'])} "
          f"weight_stream={int(hwh_meta['has_weight_stream'])} "
          f"learn_weight={int(hwh_meta['has_learn_weight'])} "
          f"s_axis_data={int(hwh_meta['has_s_axis_data'])}")
    print(f"  CFG_STATUS=0x{cfg_status:08X}: "
          f"router_busy={fmt_bool(ds['router_busy'])} "
          f"group_busy={fmt_bool(ds['any_core_group_busy'])} "
          f"rtl_ready={fmt_bool(ds['rtl_snn_ready'])} "
          f"profile_active={fmt_bool(ds['profile_active'])} "
          f"profile_done={fmt_bool(ds['profile_done'])} "
          f"overflow={fmt_bool(ds['fifo_overflow'])} "
          f"active_neurons={ds['active_neurons']}")
    print(f"  HLS_STATUS=0x{hls_status:08X}: ready={fmt_bool(hs['snn_ready'])} "
          f"busy={fmt_bool(hs['snn_busy'])} first_only={fmt_bool(hs['first_spike_only'])} "
          f"mode={hs['op_mode']} pending={fmt_bool(hs['first_spike_pending'])}")
    print(f"  DMA status: MM2S_DMASR=0x{mm2s:08X} S2MM_DMASR=0x{s2mm:08X}")
    print(f"  Counters: router={cfg.read(CFG_ROUTER_SPKS)} neuron={cfg.read(CFG_NEURON_SPKS)} "
          f"hls_in={hls.read(HLS_SPIKE_COUNT)} "
          f"latency={cfg.read(CFG_THROUGHPUT)} service={cfg.read(CFG_SERVICE_CYCLES)}")
    if profile_info is not None:
        version = (profile_info >> 24) & 0xFF
        groups = (profile_info >> 16) & 0xFF
        count = profile_info & 0xFFFF
        print(f"  PROFILE_INFO=0x{profile_info:08X}: version={version} groups={groups} counters={count}")


def parse_args():
    p = argparse.ArgumentParser(description="10-class MNIST inference on snn_core_group_profile bitstream")
    p.add_argument("--data", default="/home/xilinx/snn", help="Directory containing bit/hwh/npz")
    p.add_argument("--bitstream", default=None, help="Override bitstream path")
    p.add_argument("--weights", default=None, help="Override deployment .npz path")
    p.add_argument("--n", type=int, default=0, help="Number of images to run (0=all)")
    p.add_argument("--no-program", action="store_true", help="Skip FPGA programming")
    p.add_argument("--output", default=None, help="JSON output path")
    p.add_argument("--profile-output", default="/home/xilinx/snn/profile_coregroup.csv", help="CSV profile output path")
    p.add_argument("--packet-id-width", type=int, default=DEFAULT_HLS_PACKET_ID_WIDTH)
    p.add_argument("--local-id-width", type=int, default=DEFAULT_LOCAL_ID_WIDTH)
    p.add_argument("--neuron-map", choices=["contiguous", "round-robin"], default="round-robin",
                   help="Map logical classifier neurons to hardware global IDs")
    p.add_argument("--ct-mode", default="none",
                   help=("CT pattern after classifier output spikes: none, dummy-same, "
                         "dummy-cross1, dummy-cross4, dummy-crossK, dummy-random, "
                         "real-full, real-random25"))
    p.add_argument("--ct-dummy-weight", type=int, default=0,
                   help="Weight for dummy CT fanout sinks; keep 0 to avoid functional impact")
    p.add_argument("--fanout-density", type=float, default=0.25,
                   help="Probability for each candidate cross-group fanout in dummy-random mode")
    p.add_argument("--real-density", type=float, default=0.25,
                   help="Probability for retaining real pixel->classifier connections in real-random mode")
    p.add_argument("--real-overflow-policy", choices=["topk", "error"], default="topk",
                   help="Handle real fanout count above MAX_FANOUT_INTER-1")
    p.add_argument("--intra-mode", choices=["auto", "off", "split-same-group"], default="auto",
                   help=("auto splits same-group real fanouts into core_group weight_mem "
                         "and keeps cross-group fanouts in CT; off keeps all real fanouts in CT"))
    p.add_argument("--ct-random-seed", type=int, default=1,
                   help="Seed for dummy-random CT pattern generation")
    p.add_argument("--max-fanout-inter", type=int, default=DEFAULT_MAX_FANOUT_INTER,
                   help="Hardware MAX_FANOUT_INTER; one slot is reserved for the invalid terminator")
    p.add_argument("--capture-all-spikes", action="store_true")
    p.add_argument("--assert-hls-reset", action="store_true")
    p.add_argument("--clear-intra", action="store_true",
                   help="Explicitly zero intra-group recurrent rows used by the model")
    p.add_argument("--check-hls-version", action="store_true")
    p.add_argument("--benchmark-fast", action="store_true")
    p.add_argument("--no-per-image-reset", action="store_true")
    p.add_argument("--first-spike-timeout-ms", type=float, default=legacy.FIRST_SPIKE_TIMEOUT_MS_DEFAULT)
    p.add_argument("--further-spike-timeout-ms", type=float, default=legacy.FURTHER_SPIKE_TIMEOUT_MS_DEFAULT)
    p.add_argument("--mm2s-tail-timeout-ms", type=float, default=legacy.MM2S_TAIL_TIMEOUT_MS_DEFAULT)
    p.add_argument("--hls-input-timeout-ms", type=float, default=legacy.MM2S_TAIL_TIMEOUT_MS_DEFAULT,
                   help="Wait for HLS_SPIKE_COUNT to reach the sample input count before profile stop")
    p.add_argument("--settle-cap-ms", type=float, default=legacy.SETTLE_CAP_MS_DEFAULT)
    p.add_argument("--stop-sleep-ms", type=float, default=legacy.STOP_SLEEP_MS_DEFAULT)
    p.add_argument("--pl-clock-mhz", type=float, default=DEFAULT_PL_CLK_MHZ)
    p.add_argument("--pl-clock-hz", type=float, default=0.0)
    p.add_argument("--power-report", default=None,
                   help="Vivado report_power .rpt path for PL power/efficiency summary")
    p.add_argument("--util-report", default=None,
                   help="Vivado report_utilization .rpt path for LUT/FF/BRAM/DSP summary")
    p.add_argument("--peak-router-lanes", type=int, default=1,
                   help="Theoretical router SOP lanes per cycle for peak GSOP/S")
    p.add_argument("--peak-intra-lanes-per-group", type=int, default=1,
                   help="Theoretical intra SOP lanes per group per cycle for peak GSOP/S")
    p.add_argument("--print-every", type=int, default=100)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    require_compatible_legacy()
    if args.benchmark_fast:
        if args.first_spike_timeout_ms == legacy.FIRST_SPIKE_TIMEOUT_MS_DEFAULT:
            args.first_spike_timeout_ms = legacy.FIRST_SPIKE_TIMEOUT_MS_FAST
        if args.further_spike_timeout_ms == legacy.FURTHER_SPIKE_TIMEOUT_MS_DEFAULT:
            args.further_spike_timeout_ms = legacy.FURTHER_SPIKE_TIMEOUT_MS_FAST
        if args.mm2s_tail_timeout_ms == legacy.MM2S_TAIL_TIMEOUT_MS_DEFAULT:
            args.mm2s_tail_timeout_ms = legacy.MM2S_TAIL_TIMEOUT_MS_FAST
        if args.settle_cap_ms == legacy.SETTLE_CAP_MS_DEFAULT:
            args.settle_cap_ms = legacy.SETTLE_CAP_MS_FAST
        if args.stop_sleep_ms == legacy.STOP_SLEEP_MS_DEFAULT:
            args.stop_sleep_ms = legacy.STOP_SLEEP_MS_FAST

    data_dir = args.data
    bit_path = args.bitstream or os.path.join(data_dir, "snn_core_group_profile.bit")
    hwh_path = bit_path.replace(".bit", ".hwh")
    deploy_path = args.weights or os.path.join(data_dir, "mnist_10class_deployment.npz")
    result_path = args.output or os.path.join(data_dir, "mnist_10class_coregroup_results.json")
    profile_path = args.profile_output
    util_report_path = args.util_report or default_report_path(
        data_dir, "snn_core_group_profile_utilization.rpt")
    power_report_path = args.power_report or default_report_path(
        data_dir, "snn_core_group_profile_power.rpt")
    util_report = parse_vivado_utilization_report(util_report_path)
    power_report = parse_vivado_power_report(power_report_path)

    print("=" * 72)
    print("10-Class MNIST FPGA Inference  (core-group/profile route)")
    print("=" * 72)
    print(f"\nLoading: {deploy_path}")
    if not os.path.exists(deploy_path):
        print(f"ERROR: deployment file not found: {deploy_path}")
        sys.exit(1)
    data = np.load(deploy_path, allow_pickle=True)
    q_weights = data["q_weights"]
    test_imgs = data["test_imgs"]
    test_lbls = data["test_lbls"]
    hw_threshold = int(data["hw_threshold"])
    n_classes = int(data.get("n_classes", 10))
    fps_per_class = int(data.get("fps_per_class", 15))
    n_neurons = n_classes * fps_per_class
    n_pixels = int(q_weights.shape[1])
    n_images = int(args.n) if int(args.n) > 0 else len(test_imgs)
    print(f"  q_weights: {q_weights.shape}  hw_threshold: {hw_threshold}")
    print(f"  n_neurons: {n_neurons}  n_classes: {n_classes}  fps: {fps_per_class}")
    print(f"  Test images: {n_images}")

    ct_pattern = parse_ct_mode(args.ct_mode)
    args.ct_mode = ct_pattern["canonical"]
    total_logical_nodes = n_neurons + (n_pixels if ct_pattern["kind"] == "real" else 0)
    all_logical_to_hw, _ = build_neuron_mapping(
        n_neurons=total_logical_nodes,
        num_groups=DEFAULT_GROUPS,
        local_id_width=int(args.local_id_width),
        mode=args.neuron_map,
    )
    logical_to_hw = all_logical_to_hw[:n_neurons]
    input_logical_to_hw = all_logical_to_hw[n_neurons:n_neurons + n_pixels]
    hw_to_logical = {int(hw): int(logical) for logical, hw in enumerate(logical_to_hw)}
    group_hist = [0 for _ in range(DEFAULT_GROUPS)]
    for hw_id in all_logical_to_hw:
        group, _ = decode_global_id(int(hw_id), int(args.local_id_width))
        if 0 <= group < DEFAULT_GROUPS:
            group_hist[group] += 1
    used_groups = [g for g, count in enumerate(group_hist) if count > 0]
    dummy_sink_ids = build_dummy_sink_ids(
        logical_to_hw=all_logical_to_hw,
        num_groups=DEFAULT_GROUPS,
        local_id_width=int(args.local_id_width),
    )
    print(f"  Neuron map: {args.neuron_map}  used_groups={used_groups}")
    print(f"  Neuron map group counts: {group_hist}")
    print(f"  CT mode: {ct_pattern['canonical']}  dummy_weight={int(args.ct_dummy_weight)} "
          f"density={float(args.fanout_density):.3f} real_density={float(args.real_density):.3f} "
          f"seed={int(args.ct_random_seed)}")
    if ct_pattern["kind"] != "none":
        dummy_summary = [
            (g, int(hw)) for g, hw in enumerate(dummy_sink_ids) if hw is not None
        ]
        print(f"  CT dummy sink ids: {dummy_summary}")

    if not args.no_program:
        print(f"\nProgramming FPGA: {bit_path}")
        if not os.path.exists(bit_path):
            print(f"ERROR: bitstream not found: {bit_path}")
            sys.exit(1)
        if not legacy.program_fpga(bit_path):
            print("ERROR: FPGA programming failed")
            sys.exit(1)
    else:
        print("\n(--no-program: skipping FPGA programming)")

    hwh_meta = parse_hwh_metadata(hwh_path)
    if hwh_meta["hls_modtype"] != "snn_inference_profile_hls":
        print(f"WARNING: HWH MODTYPE is {hwh_meta['hls_modtype']}, expected snn_inference_profile_hls")
    if hwh_meta["has_weight_stream"] or hwh_meta["has_learn_weight"] or hwh_meta["has_axi_dma_1"]:
        print("WARNING: HWH still exposes old learning/weight interfaces.")

    hls_packet_id_w = int(args.packet_id_width)
    local_id_width = int(args.local_id_width)
    pl_clock_hz = float(args.pl_clock_hz) if float(args.pl_clock_hz) > 0.0 else float(args.pl_clock_mhz) * 1_000_000.0
    pl_clock_mhz = pl_clock_hz / 1_000_000.0

    print(f"  HLS packet_id_width: {hls_packet_id_w}  apnone_id_width: {hwh_meta['apnone_id_width']}")
    print(f"  Core-group ID split: group_id_width={DEFAULT_GROUP_ID_WIDTH} local_id_width={local_id_width}")
    print(f"  Input mapping: logical neuron_id -> hardware global neuron_id ({args.neuron_map}), no source_offset")
    print(f"  PL clock for latency conversion: {pl_clock_hz:.0f} Hz ({pl_clock_mhz:.6f} MHz)")
    print(f"  S2MM capture mode: {'all-spikes' if args.capture_all_spikes else 'first-spike-only'}")

    hls = legacy.MMIO(HLS_BASE, 0x100)
    cfg = legacy.MMIO(CFG_BASE, 0x100)
    dma = legacy.MMIO(DMA_BASE, 0x100)

    cfg_ver = cfg.read(CFG_VERSION)
    print(f"  snn_config_regs version: 0x{cfg_ver:08X}  ({'OK' if cfg_ver == 0x534E4E01 else 'UNEXPECTED'})")
    hls_ver = None
    if args.check_hls_version:
        hls_ver = hls.read(HLS_VERSION_REG)
        print(f"  hls version_reg: 0x{hls_ver:08X}  ({'OK' if hls_ver == EXPECTED_HLS_VERSION else 'UNEXPECTED'})")
    else:
        print("  hls version_reg: (SKIPPED; use --check-hls-version to enable)")

    profile_info = cfg.read(CFG_PROFILE_INFO)
    profile_enabled = profile_path is not None
    profile_num_groups = 0
    profile_counter_count = 0
    profile_version = (profile_info >> 24) & 0xFF
    if profile_info != 0xDEADBEEF and profile_version >= 1:
        profile_num_groups = (profile_info >> 16) & 0xFF
        profile_counter_count = profile_info & 0xFFFF
        print(
            f"  profile info: 0x{profile_info:08X}  "
            f"version={profile_version} groups={profile_num_groups} "
            f"counters={profile_counter_count}"
        )
        if profile_enabled and profile_version not in (9, 10, 11, 12):
            print("ERROR: this script expects the current core-group/profile "
                  "bitstream with PROFILE_INFO version=9 or BASIC version=10/11/12.")
            print("  version=6 was the temporary relay experiment and can deadlock/slow the run; "
                  "rebuild and redeploy the current RTL/HWH/bit files.")
            sys.exit(1)
    elif profile_enabled:
        print("ERROR: --profile-output requested, but profile window is not available.")
        sys.exit(1)
    profile_basic_only = profile_version >= 10

    print("\nWarm-up HLS control path ...")
    legacy.warmup_hls(hls, poll_sleep_s=(0.001 if args.benchmark_fast else 0.005),
                      post_sleep_s=(0.0 if args.benchmark_fast else 0.010))
    print("  Done.")

    print(f"\nProgramming core-group route ({n_neurons} direct external targets) ...")
    legacy.reset_system(hls, cfg, assert_hls_reset=args.assert_hls_reset,
                        poll_sleep_s=(0.0002 if args.benchmark_fast else 0.001),
                        post_sleep_s=(0.0 if args.benchmark_fast else 0.002))
    route_cfg = program_coregroup_for_inference(
        cfg,
        n_neurons=n_neurons,
        local_id_width=local_id_width,
        q_weights=q_weights,
        logical_to_hw=logical_to_hw,
        input_logical_to_hw=input_logical_to_hw,
        ct_mode=args.ct_mode,
        dummy_sink_ids=dummy_sink_ids,
        dummy_weight=int(args.ct_dummy_weight),
        fanout_density=float(args.fanout_density),
        real_density=float(args.real_density),
        random_seed=int(args.ct_random_seed),
        max_fanout_inter=int(args.max_fanout_inter),
        real_overflow_policy=str(args.real_overflow_policy),
        clear_intra=bool(args.clear_intra),
        intra_mode=str(args.intra_mode),
    )
    legacy.configure_neurons(cfg, threshold=hw_threshold, leak=0, refrac=0)
    print(f"  Done. CT invalid terminators={route_cfg['ct_invalid_terminators']} "
          f"valid_entries={route_cfg['ct_valid_entries']} "
          f"expected_dummy_entries={route_cfg['ct_expected_dummy_entries']} "
          f"cross_entries={route_cfg['ct_cross_entries']} "
          f"same_entries={route_cfg['ct_same_entries']} "
          f"intra_mode={route_cfg['intra_mode']} "
          f"intra_nonzero_entries={route_cfg['intra_nonzero_entries']} "
          f"intra_clear_writes={route_cfg['intra_clear_writes']} "
          f"config_ms={route_cfg['config_ms']:.3f}")
    if route_cfg["ct_pattern_kind"] == "real":
        print(f"  Real CT: pre_prune_entries={route_cfg['ct_real_pre_prune_entries']} "
              f"programmed={route_cfg['ct_valid_entries']} "
              f"truncated={route_cfg['ct_real_truncated_entries']} "
              f"overflow_sources={route_cfg['ct_real_sources_with_overflow']} "
              f"policy={route_cfg['ct_real_overflow_policy']} "
              f"intra={route_cfg['intra_mode_requested']}->{route_cfg['intra_mode']} "
              f"intra_nonzero={route_cfg['intra_nonzero_entries']}")
    if route_cfg["ct_pattern_kind"] in ("same", "cross"):
        if int(route_cfg["ct_valid_entries"]) != int(route_cfg["ct_expected_dummy_entries"]):
            print("ERROR: dummy CT coverage is incomplete before inference.")
            print(f"  ct_valid_entries={route_cfg['ct_valid_entries']} "
                  f"expected={route_cfg['ct_expected_dummy_entries']}")
            print("  Sync tests/fpga_10class_coregroup_inference.py to the board and rerun.")
            sys.exit(1)

    print_diagnostics(hls, cfg, dma, hwh_meta, profile_info)

    real_ct_mode = route_cfg["ct_pattern_kind"] == "real"
    if real_ct_mode and not args.capture_all_spikes:
        print("  NOTE: real CT mode forces all-spikes capture so input-source spikes can be filtered in Python.")
        args.capture_all_spikes = True
    capture_capacity = (n_neurons + n_pixels + 32) if real_ct_mode else (n_neurons + 16)
    n_buf_words = capture_capacity
    buf_in = legacy.make_dma_buffer(DMA_BUF_IN, n_buf_words * 4 + 64, label="MM2S")
    buf_out = legacy.make_dma_buffer(DMA_BUF_OUT, n_buf_words * 4 + 64, label="S2MM")

    print(f"\nRunning {n_images} inference{'s' if n_images != 1 else ''} ...")
    print("-" * 100)
    print(f"{'idx':>5} {'lbl':>4} {'sw_t':>5} {'sw_c':>5} {'sw_ct':>5} {'hw':>4} {'src':>4} {'acc':>5} {'cteq':>5} | "
          f"{'router':>7} {'neuron':>7} {'hls_in':>7} {'lat':>6} {'svc':>6} "
          f"{'mm2s':>10} {'s2mm':>10} {'cfg':>8} {'#s2':>4}")
    print("-" * 100)

    results = []
    sw_correct = 0
    sw_ct_pruned_correct = 0
    sw_ct_pruned_samples = 0
    hw_matches_sw_ct_pruned = 0
    hw_matches_sw_ct_count_score = 0
    hw_matches_sw_hwct_count_score = 0
    sw_hwct_count_score_samples = 0
    sw_ct_count_score_samples = 0
    direct_count_pred_matches = 0
    direct_count_score_pred_matches = 0
    direct_count_vector_matches = 0
    hw_correct = 0
    s2mm_correct = 0
    s2mm_samples = 0
    fallback_count = 0
    t_start = time.time()

    iter_ms = []
    mm2s_tail_ms = []

    for idx in range(n_images):
        t_iter0 = time.perf_counter()
        img = test_imgs[idx]
        lbl = int(test_lbls[idx])

        t_sw0 = time.perf_counter()
        potential = legacy.compute_positive_potential(img, q_weights)
        sw = legacy.sw_reference_10class(
            img, q_weights, hw_threshold, n_classes, fps_per_class, potential=potential
        )
        t_sw1 = time.perf_counter()
        sw_pred = int(sw["pred"])
        sw_count = int(sw["pred_count"])
        sw_direct_count_pred, sw_direct_class_counts = direct_sw_count_by_class(
            potential,
            n_classes=n_classes,
            fps_per_class=fps_per_class,
        )
        if sw_pred == lbl:
            sw_correct += 1
        sw_ct_pred = None
        sw_ct_count_score_pred = None
        sw_ct_counts = [0 for _ in range(n_classes)]
        sw_ct_scores = [0 for _ in range(n_classes)]
        sw_ct_events = [0 for _ in range(n_classes)]
        sw_hwct_pred = None
        sw_hwct_count_score_pred = None
        sw_hwct_counts = [0 for _ in range(n_classes)]
        sw_hwct_scores = [0 for _ in range(n_classes)]
        sw_hwct_events = [0 for _ in range(n_classes)]
        sw_hwct_source_fire_count = 0
        sw_hwct_valid_event_budget = 0
        sw_ct_source_fire_count = 0
        sw_ct_valid_event_budget = 0
        sw_ct_input_source_group_hist = [0 for _ in range(profile_num_groups)]

        t_pack0 = time.perf_counter()
        if real_ct_mode:
            spike_words = build_real_input_spike_words(
                img,
                input_logical_to_hw=input_logical_to_hw,
                hls_spike_pkt_id_w=hls_packet_id_w,
            )
        else:
            spike_words = build_direct_spike_words(
                img, q_weights, hls_spike_pkt_id_w=hls_packet_id_w,
                potential=potential, logical_to_hw=logical_to_hw
            )
        t_pack1 = time.perf_counter()

        reset_ms = 0.0
        if not args.no_per_image_reset:
            t_reset0 = time.perf_counter()
            legacy.reset_system(hls, cfg, assert_hls_reset=args.assert_hls_reset,
                                poll_sleep_s=(0.0002 if args.benchmark_fast else 0.001),
                                post_sleep_s=(0.0 if args.benchmark_fast else 0.002))
            t_reset1 = time.perf_counter()
            reset_ms = (t_reset1 - t_reset0) * 1000.0

        ctr_router_base = cfg.read(CFG_ROUTER_SPKS)
        ctr_neuron_base = cfg.read(CFG_NEURON_SPKS)

        hw = legacy.run_inference(
            hls, cfg, dma,
            spike_words=spike_words,
            n_neurons=capture_capacity,
            hw_threshold=hw_threshold,
            hls_spike_pkt_id_w=hls_packet_id_w,
            capture_all_spikes=bool(args.capture_all_spikes),
            buf_in=buf_in,
            buf_out=buf_out,
            ctr_router_base=ctr_router_base,
            ctr_neuron_base=ctr_neuron_base,
            first_spike_timeout_s=float(args.first_spike_timeout_ms) / 1000.0,
            further_spike_timeout_s=float(args.further_spike_timeout_ms) / 1000.0,
            mm2s_tail_timeout_s=float(args.mm2s_tail_timeout_ms) / 1000.0,
            mm2s_tail_poll_s=(0.0002 if args.benchmark_fast else 0.005),
            settle_cap_s=float(args.settle_cap_ms) / 1000.0,
            stop_sleep_s=float(args.stop_sleep_ms) / 1000.0,
            dma_reset_sleep_s=(0.0002 if args.benchmark_fast else 0.002),
            profile_enabled=profile_enabled,
            profile_num_groups=profile_num_groups,
            profile_drain_timeout_s=max(0.100, float(args.mm2s_tail_timeout_ms) / 1000.0),
            wait_hls_input_count=True,
            hls_input_timeout_s=max(0.100, float(args.hls_input_timeout_ms) / 1000.0),
        )

        if profile_enabled and profile_num_groups:
            class_dbg = read_profile_class_debug(cfg, profile_num_groups, n_classes)
            if class_dbg.get("available", False):
                profile = hw.setdefault("profile", {})
                profile["router_class_scores"] = class_dbg["router_class_scores"]
                profile["router_class_events"] = class_dbg["router_class_events"]
                profile["top_class_scores"] = class_dbg["top_class_scores"]
                profile["top_class_events"] = class_dbg["top_class_events"]
                for cls in range(n_classes):
                    profile[f"class_{cls}_score"] = int(class_dbg["top_class_scores"][cls])
                    profile[f"class_{cls}_event_count"] = int(class_dbg["top_class_events"][cls])

        hw_pred_rtl = classify_profile_first_classifier(hw, hw_to_logical, n_classes, fps_per_class)
        hw_pred_count_raw, hw_class_counts = classify_profile_count(hw, n_classes)
        hw_pred_score, hw_class_scores, hw_class_events = classify_profile_score(hw, n_classes)
        hw_pred_count_score = classify_count_with_score_tiebreak(
            hw_class_counts,
            hw_class_scores,
        )
        hw_pred_count = hw_pred_count_score if hw_pred_count_score is not None else hw_pred_count_raw
        hw_pred_s2 = classify_hw_first_spike_mapped(hw, hw_to_logical, n_classes, fps_per_class)
        if real_ct_mode:
            sw_ct_active_pixels = real_input_active_pixels(img)
            real_fanouts = route_cfg.get("ct_real_fanout_mask", {}).get("fanouts", [])
            sw_ct_valid_event_budget = sum(
                len(real_fanouts[int(pixel_idx)])
                for pixel_idx in sw_ct_active_pixels
                if int(pixel_idx) < len(real_fanouts)
            )
            sw_ct_source_fire_count = int(len(sw_ct_active_pixels))
            sw_ct_input_source_group_hist = group_hist_from_hw_ids(
                input_logical_to_hw[sw_ct_active_pixels],
                local_id_width,
                profile_num_groups,
            )
            _sw_ct_count_pred, sw_ct_counts = sw_ct_pruned_count_predict(
                img,
                fanout_mask=route_cfg.get("ct_real_fanout_mask"),
                hw_threshold=hw_threshold,
                n_classes=n_classes,
                fps_per_class=fps_per_class,
                active_pixels_override=sw_ct_active_pixels,
            )
            sw_ct_pred, sw_ct_scores, sw_ct_events = sw_ct_pruned_score_predict(
                img,
                fanout_mask=route_cfg.get("ct_real_fanout_mask"),
                hw_threshold=hw_threshold,
                n_classes=n_classes,
                fps_per_class=fps_per_class,
                active_pixels_override=sw_ct_active_pixels,
            )
            sw_ct_count_score_pred = classify_count_with_score_tiebreak(
                sw_ct_counts,
                sw_ct_scores,
            )
            hw_ct_active_pixels = hw.get("profile", {}).get("input_source_ct_active_pixels", None)
            if hw_ct_active_pixels is not None:
                sw_hwct_active_pixels = np.asarray(hw_ct_active_pixels, dtype=np.int64)
            elif profile_basic_only:
                # BASIC profile hides the input-source CT bitmap to keep timing light.
                # Fall back to the programmed SW active-pixel set so the count+score
                # comparison remains visible in compact logs/CSVs.
                sw_hwct_active_pixels = np.asarray(sw_ct_active_pixels, dtype=np.int64)
            else:
                sw_hwct_active_pixels = None

            if sw_hwct_active_pixels is not None:
                real_fanouts = route_cfg.get("ct_real_fanout_mask", {}).get("fanouts", [])
                sw_hwct_valid_event_budget = sum(
                    len(real_fanouts[int(pixel_idx)])
                    for pixel_idx in sw_hwct_active_pixels
                    if int(pixel_idx) < len(real_fanouts)
                )
                sw_hwct_source_fire_count = int(len(sw_hwct_active_pixels))
                _sw_hwct_count_pred, sw_hwct_counts = sw_ct_pruned_count_predict(
                    img,
                    fanout_mask=route_cfg.get("ct_real_fanout_mask"),
                    hw_threshold=hw_threshold,
                    n_classes=n_classes,
                    fps_per_class=fps_per_class,
                    active_pixels_override=sw_hwct_active_pixels,
                )
                sw_hwct_pred, sw_hwct_scores, sw_hwct_events = sw_ct_pruned_score_predict(
                    img,
                    fanout_mask=route_cfg.get("ct_real_fanout_mask"),
                    hw_threshold=hw_threshold,
                    n_classes=n_classes,
                    fps_per_class=fps_per_class,
                    active_pixels_override=sw_hwct_active_pixels,
                )
                sw_hwct_count_score_pred = classify_count_with_score_tiebreak(
                    sw_hwct_counts,
                    sw_hwct_scores,
                )
            if sw_ct_pred is not None:
                sw_ct_pruned_samples += 1
                if int(sw_ct_pred) == lbl:
                    sw_ct_pruned_correct += 1
            if sw_ct_count_score_pred is not None:
                sw_ct_count_score_samples += 1
            if sw_hwct_count_score_pred is not None:
                sw_hwct_count_score_samples += 1
        direct_count_pred_match = (
            (not real_ct_mode) and
            hw_pred_count is not None and
            sw_direct_count_pred is not None and
            int(hw_pred_count) == int(sw_direct_count_pred)
        )
        direct_count_score_pred_match = (
            (not real_ct_mode) and
            hw_pred_count_score is not None and
            sw_direct_count_pred is not None and
            int(hw_pred_count_score) == int(sw_direct_count_pred)
        )
        direct_count_vector_match = (
            (not real_ct_mode) and
            [int(x) for x in hw_class_counts] == [int(x) for x in sw_direct_class_counts]
        )
        if direct_count_pred_match:
            direct_count_pred_matches += 1
        if direct_count_score_pred_match:
            direct_count_score_pred_matches += 1
        if direct_count_vector_match:
            direct_count_vector_matches += 1
        if hw_pred_s2 is not None:
            s2mm_samples += 1
            if int(hw_pred_s2) == lbl:
                s2mm_correct += 1
        if real_ct_mode and hw_pred_count_score is not None:
            hw_pred = int(hw_pred_count_score)
            src = "C+S"
        elif real_ct_mode and hw_pred_score is not None:
            hw_pred = int(hw_pred_score)
            src = "SCR"
        elif real_ct_mode and hw_pred_count is not None:
            hw_pred = int(hw_pred_count)
            src = "CNT"
        elif real_ct_mode and hw_pred_rtl is not None:
            hw_pred = int(hw_pred_rtl)
            src = "RTL"
        elif hw_pred_s2 is None:
            hw_pred = sw_pred
            src = "SW"
            fallback_count += 1
        else:
            hw_pred = int(hw_pred_s2)
            src = "S2"
        if hw_pred == lbl and int(hw.get("neuron_spikes", 0)) > 0:
            hw_correct += 1
        if real_ct_mode and sw_ct_pred is not None and int(hw_pred) == int(sw_ct_pred):
            hw_matches_sw_ct_pruned += 1
        if (real_ct_mode and sw_ct_count_score_pred is not None and
                hw_pred_count_score is not None and
                int(hw_pred_count_score) == int(sw_ct_count_score_pred)):
            hw_matches_sw_ct_count_score += 1
        if (real_ct_mode and sw_hwct_count_score_pred is not None and
                hw_pred_count_score is not None and
                int(hw_pred_count_score) == int(sw_hwct_count_score_pred)):
            hw_matches_sw_hwct_count_score += 1

        t_iter1 = time.perf_counter()
        timing = hw.get("timing", {})
        iter_ms.append((t_iter1 - t_iter0) * 1000.0)
        mm2s_tail_ms.append(float(timing.get("mm2s_tail_wait_ms", 0.0)))

        cfg_status = int(hw.get("status", cfg.read(CFG_STATUS))) if "status" in hw else cfg.read(CFG_STATUS)
        pl_latency = int(hw.get("pl_latency_cycles", 0))
        pl_service = hw.get("pl_service_cycles", None)
        pl_service_print = int(pl_service) if pl_service is not None else 0
        hw_input_source_group_hist = [
            int(hw.get("profile", {}).get(f"input_source_group_{group}_spike_count", 0) or 0)
            for group in range(profile_num_groups)
        ]

        print_every = max(1, int(args.print_every))
        if idx % print_every == 0 or idx == n_images - 1:
            sw_ct_print = "-" if sw_ct_pred is None else str(int(sw_ct_pred))
            cteq = "-" if sw_ct_pred is None else ("Y" if int(hw_pred) == int(sw_ct_pred) else "N")
            print(f"{idx:>5d} {lbl:>4d} {sw_pred:>5d} {sw_count:>5d} {sw_ct_print:>5s} "
                  f"{hw_pred:>4d} {src:>4s} {'OK' if sw_pred == lbl else 'DIFF':>5s} {cteq:>5s} | "
                  f"{int(hw.get('router_spikes', 0)):>7d} {int(hw.get('neuron_spikes', 0)):>7d} "
                  f"{int(hw.get('hls_spike_count', 0)):>7d} {pl_latency:>6d} {pl_service_print:>6d} "
                  f"0x{int(hw.get('mm2s_sr', 0)):08X} 0x{int(hw.get('s2mm_sr', 0)):08X} "
                  f"0x{cfg_status:06X} {len(hw.get('output_spikes', [])):>4d}")

        row = {
            "idx": idx,
            "label": lbl,
            "input_words": int(len(spike_words)),
            "sw_pred_ttfs": sw_pred,
            "sw_pred_count": sw_count,
            "sw_direct_count_pred": (None if sw_direct_count_pred is None else int(sw_direct_count_pred)),
            "sw_direct_class_counts": [int(x) for x in sw_direct_class_counts],
            "direct_count_pred_match": bool(direct_count_pred_match),
            "direct_count_score_pred_match": bool(direct_count_score_pred_match),
            "direct_count_vector_match": bool(direct_count_vector_match),
            "sw_ct_pruned_pred": (None if sw_ct_pred is None else int(sw_ct_pred)),
            "sw_ct_count_score_pred": (None if sw_ct_count_score_pred is None else int(sw_ct_count_score_pred)),
            "sw_ct_source_fire_count": int(sw_ct_source_fire_count),
            "sw_ct_valid_event_budget": int(sw_ct_valid_event_budget),
            "sw_ct_input_source_group_hist": [int(x) for x in sw_ct_input_source_group_hist],
            "sw_ct_pruned_class_counts": [int(x) for x in sw_ct_counts],
            "sw_ct_pruned_class_scores": [int(x) for x in sw_ct_scores],
            "sw_ct_pruned_class_events": [int(x) for x in sw_ct_events],
            "sw_hwct_pruned_pred": (None if sw_hwct_pred is None else int(sw_hwct_pred)),
            "sw_hwct_count_score_pred": (None if sw_hwct_count_score_pred is None else int(sw_hwct_count_score_pred)),
            "sw_hwct_source_fire_count": int(sw_hwct_source_fire_count),
            "sw_hwct_valid_event_budget": int(sw_hwct_valid_event_budget),
            "sw_hwct_pruned_class_counts": [int(x) for x in sw_hwct_counts],
            "sw_hwct_pruned_class_scores": [int(x) for x in sw_hwct_scores],
            "sw_hwct_pruned_class_events": [int(x) for x in sw_hwct_events],
            "hw_pred": int(hw_pred),
            "hw_pred_source": src,
            "hw_pred_s2mm": (None if hw_pred_s2 is None else int(hw_pred_s2)),
            "hw_pred_rtl_first_classifier": (None if hw_pred_rtl is None else int(hw_pred_rtl)),
            "hw_pred_count": (None if hw_pred_count is None else int(hw_pred_count)),
            "hw_pred_count_raw": (None if hw_pred_count_raw is None else int(hw_pred_count_raw)),
            "hw_pred_score": (None if hw_pred_score is None else int(hw_pred_score)),
            "hw_pred_count_score": (None if hw_pred_count_score is None else int(hw_pred_count_score)),
            "hw_class_counts": [int(x) for x in hw_class_counts],
            "hw_class_scores": [int(x) for x in hw_class_scores],
            "hw_class_events": [int(x) for x in hw_class_events],
            "hw_input_source_group_hist": [int(x) for x in hw_input_source_group_hist],
            "router_spikes": int(hw.get("router_spikes", 0)),
            "neuron_spikes": int(hw.get("neuron_spikes", 0)),
            "hls_spike_count": int(hw.get("hls_spike_count", 0)),
            "pl_latency_cycles": int(hw.get("pl_latency_cycles", 0)),
            "pl_service_cycles": (int(hw.get("pl_service_cycles")) if hw.get("pl_service_cycles") is not None else None),
            "mm2s_sr": int(hw.get("mm2s_sr", 0)),
            "s2mm_sr": int(hw.get("s2mm_sr", 0)),
            "mm2s_done": bool(hw.get("mm2s_done", False)),
            "hls_input_done": bool(hw.get("hls_input_done", False)),
            "hls_spike_target": int(hw.get("hls_spike_target", len(spike_words))),
            "profile_incomplete": bool(hw.get("profile_incomplete", False)),
            "profile": hw.get("profile", {}),
            "timing": {
                **timing,
                "sw_ref_ms": (t_sw1 - t_sw0) * 1000.0,
                "spike_pack_ms": (t_pack1 - t_pack0) * 1000.0,
                "reset_ms": reset_ms,
                "iter_wall_ms": (t_iter1 - t_iter0) * 1000.0,
            },
        }
        results.append(row)

    elapsed = time.time() - t_start
    total_input = sum(r["input_words"] for r in results)
    total_router = sum(r["router_spikes"] for r in results)
    total_neuron = sum(r["neuron_spikes"] for r in results)
    total_s2mm = sum(1 for r in results if r.get("hw_pred_s2mm") is not None)
    total_count_score = sum(1 for r in results if r["hw_pred_source"] == "C+S")
    total_score = sum(1 for r in results if r["hw_pred_source"] == "SCR")
    total_count = sum(1 for r in results if r["hw_pred_source"] == "CNT")
    total_rtl = sum(1 for r in results if r["hw_pred_source"] == "RTL")
    profile_incomplete = sum(1 for r in results if r["profile_incomplete"])
    hls_input_incomplete = sum(1 for r in results if not r["hls_input_done"])
    ct_output_total = sum(row_profile_value(r, "output_spike_count") for r in results)
    ct_lookup_total = sum(row_profile_value(r, "ct_lookup_count") for r in results)
    ct_valid_total = sum(row_profile_value(r, "ct_valid_entry_count") for r in results)
    ct_invalid_total = sum(row_profile_value(r, "ct_invalid_entry_count") for r in results)
    ct_cross_total = sum(row_profile_value(r, "cross_group_event_count") for r in results)
    ct_same_total = sum(row_profile_value(r, "same_group_event_count") for r in results)
    router_busy_total = sum(row_profile_value(r, "router_busy_cycles") for r in results)
    router_stall_total = sum(row_profile_value(r, "router_stall_cycles") for r in results)
    total_latency_total = sum(row_profile_value(r, "total_latency_cycles") for r in results)
    has_core_group_busy_sum = any("core_group_busy_cycles_sum" in r.get("profile", {}) for r in results)
    has_intra_route_cycles_sum = any("intra_route_cycles_sum" in r.get("profile", {}) for r in results)
    has_intra_weight_nonzero = any("intra_weight_nonzero_count" in r.get("profile", {}) for r in results)
    core_group_busy_total = (
        sum(row_profile_value(r, "core_group_busy_cycles_sum") for r in results)
        if has_core_group_busy_sum else None
    )
    intra_route_cycles_total = (
        sum(row_profile_value(r, "intra_route_cycles_sum") for r in results)
        if has_intra_route_cycles_sum else None
    )
    intra_weight_nonzero_total = (
        sum(row_profile_value(r, "intra_weight_nonzero_count") for r in results)
        if has_intra_weight_nonzero else 0
    )
    out_fifo_overflow_drop_total = sum(row_profile_value(r, "drop_spike_count") for r in results)
    if out_fifo_overflow_drop_total == 0:
        out_fifo_overflow_drop_total = sum(row_profile_value(r, "out_fifo_overflow_drop_count") for r in results)
    if out_fifo_overflow_drop_total == 0:
        out_fifo_overflow_drop_total = sum(
        int(r.get("profile", {}).get(f"group_{group}_out_fifo_overflow_drop_count", 0) or 0)
        for r in results
        for group in range(profile_num_groups)
        )
    input_source_hw_total = sum(sum(r.get("hw_input_source_group_hist", [])) for r in results)
    input_source_sw_total = sum(sum(r.get("sw_ct_input_source_group_hist", [])) for r in results)
    input_source_hist_mismatch = sum(
        1 for r in results
        if r.get("hw_input_source_group_hist", []) != r.get("sw_ct_input_source_group_hist", [])
    )
    input_source_ct_hw_total = sum(
        int(r.get("profile", {}).get("input_source_ct_active_pixel_count", 0) or 0)
        for r in results
    )
    input_source_ct_source_gap = input_source_hw_total - input_source_ct_hw_total
    ct_expected_total = None if profile_basic_only else ct_mode_expected_counts(ct_output_total, args.ct_mode)
    ct_mode_mismatch = sum(
        1 for r in results
        if not ct_mode_row_ok(r, args.ct_mode)
    ) if not profile_basic_only else 0
    ct_mode_ok = (
        True if ct_expected_total is None else
        ct_lookup_total == ct_expected_total["lookup"] and
        ct_valid_total == ct_expected_total["valid"] and
        ct_invalid_total == ct_expected_total["invalid"] and
        ct_cross_total == ct_expected_total["cross"] and
        ct_same_total == ct_expected_total["same"] and
        ct_mode_mismatch == 0
    )
    mean_iter = sum(iter_ms) / max(len(iter_ms), 1)
    mean_tail = sum(mm2s_tail_ms) / max(len(mm2s_tail_ms), 1)
    pl_cycles_samples = [
        int(r.get("pl_latency_cycles", 0))
        for r in results
        if int(r.get("pl_latency_cycles", 0)) > 0
    ]
    pl_service_cycles_samples = [
        int(r.get("pl_service_cycles"))
        for r in results
        if r.get("pl_service_cycles") is not None and int(r.get("pl_service_cycles")) > 0
    ]

    def mean_or_none(samples):
        return (float(sum(samples)) / float(len(samples))) if samples else None

    def timing_mean(name: str):
        return mean_or_none([
            float(r.get("timing", {}).get(name, 0.0))
            for r in results
            if name in r.get("timing", {})
        ])

    if pl_cycles_samples:
        pl_cycles_mean = mean_or_none(pl_cycles_samples)
        pl_cycles_min = min(pl_cycles_samples)
        pl_cycles_max = max(pl_cycles_samples)
        pl_ms_mean = (pl_cycles_mean / pl_clock_hz) * 1000.0
        pl_ms_min = (pl_cycles_min / pl_clock_hz) * 1000.0
        pl_ms_max = (pl_cycles_max / pl_clock_hz) * 1000.0
    else:
        pl_cycles_mean = pl_cycles_min = pl_cycles_max = None
        pl_ms_mean = pl_ms_min = pl_ms_max = None

    if pl_service_cycles_samples:
        pl_service_cycles_mean = mean_or_none(pl_service_cycles_samples)
        pl_service_cycles_min = min(pl_service_cycles_samples)
        pl_service_cycles_max = max(pl_service_cycles_samples)
        pl_service_ms_mean = (pl_service_cycles_mean / pl_clock_hz) * 1000.0
        pl_service_ms_min = (pl_service_cycles_min / pl_clock_hz) * 1000.0
        pl_service_ms_max = (pl_service_cycles_max / pl_clock_hz) * 1000.0
    else:
        pl_service_cycles_mean = pl_service_cycles_min = pl_service_cycles_max = None
        pl_service_ms_mean = pl_service_ms_min = pl_service_ms_max = None

    pl_first_spike_tput_img_s = (
        1000.0 / pl_ms_mean
        if pl_ms_mean is not None and pl_ms_mean > 0.0 else None
    )
    pl_service_tput_img_s = (
        1000.0 / pl_service_ms_mean
        if pl_service_ms_mean is not None and pl_service_ms_mean > 0.0 else None
    )
    synaptic_ops_total = int(ct_valid_total + intra_weight_nonzero_total)
    peak_router_lanes = max(0, int(args.peak_router_lanes))
    peak_intra_lanes_per_group = max(0, int(args.peak_intra_lanes_per_group))
    peak_parallel_sops_per_cycle = (
        peak_router_lanes + peak_intra_lanes_per_group * int(profile_num_groups)
    )
    peak_gsops = (pl_clock_hz * peak_parallel_sops_per_cycle) / 1.0e9
    gsops_first_spike = (
        (synaptic_ops_total * pl_first_spike_tput_img_s) / 1.0e9 / max(n_images, 1)
        if pl_first_spike_tput_img_s is not None else None
    )
    gsops_service = (
        (synaptic_ops_total * pl_service_tput_img_s) / 1.0e9 / max(n_images, 1)
        if pl_service_tput_img_s is not None else None
    )
    gsops_service_util_percent = (
        gsops_service / peak_gsops * 100.0
        if gsops_service is not None and peak_gsops > 0.0 else None
    )
    pl_dynamic_w = power_report.get("pl_dynamic_w") if power_report.get("available") else None
    gsops_per_w_service = (
        gsops_service / float(pl_dynamic_w)
        if gsops_service is not None and pl_dynamic_w is not None and float(pl_dynamic_w) > 0.0 else None
    )

    sw_ref_ms_mean = timing_mean("sw_ref_ms")
    reset_ms_mean = timing_mean("reset_ms")
    spike_pack_ms_mean = timing_mean("spike_pack_ms")
    hw_run_ms_mean = timing_mean("run_total_ms")
    hw_dma_reset_ms_mean = timing_mean("dma_reset_ms")
    hw_stream_poll_ms_mean = timing_mean("stream_poll_ms")
    hw_mm2s_tail_ms_mean = timing_mean("mm2s_tail_wait_ms")
    hw_hls_input_ms_mean = timing_mean("hls_input_wait_ms")
    hw_settle_ms_mean = timing_mean("settle_wait_ms")
    hw_stop_ms_mean = timing_mean("stop_wait_ms")
    hw_readback_ms_mean = timing_mean("counter_read_ms")
    hw_tracked_sleep_ms_mean = timing_mean("tracked_sleep_ms_total")
    if hw_tracked_sleep_ms_mean is None:
        hw_tracked_sleep_ms_mean = timing_mean("tracked_sleep_ms")
    non_pl_overhead_ms_mean = None
    non_pl_overhead_ratio = None
    if pl_service_ms_mean is not None:
        non_pl_overhead_ms_mean = mean_iter - pl_service_ms_mean
        non_pl_overhead_ratio = (non_pl_overhead_ms_mean / mean_iter) * 100.0 if mean_iter > 0.0 else None

    print("=" * 100)
    print(f"\nResults ({n_images} images, {elapsed:.1f}s, {elapsed/max(n_images,1)*1000:.1f}ms/img):")
    print(f"  SW TTFS acc:             {sw_correct}/{n_images} ({sw_correct/max(n_images,1)*100:.2f}%)")
    if real_ct_mode:
        print(f"  SW CT-pruned acc:        {sw_ct_pruned_correct}/{max(sw_ct_pruned_samples,1)} "
              f"({sw_ct_pruned_correct/max(sw_ct_pruned_samples,1)*100:.2f}%)")
        print(f"  HW vs SW CT-pruned:      {hw_matches_sw_ct_pruned}/{max(sw_ct_pruned_samples,1)} "
              f"({hw_matches_sw_ct_pruned/max(sw_ct_pruned_samples,1)*100:.2f}%)")
        print(f"  HW count+score vs SW:    {hw_matches_sw_ct_count_score}/{max(sw_ct_count_score_samples,1)} "
              f"({hw_matches_sw_ct_count_score/max(sw_ct_count_score_samples,1)*100:.2f}%)")
        print(f"  HW count+score vs SW-HWCT:{hw_matches_sw_hwct_count_score}/{max(sw_hwct_count_score_samples,1)} "
              f"({hw_matches_sw_hwct_count_score/max(sw_hwct_count_score_samples,1)*100:.2f}%)")
        print(f"  Input-source group hist: hw={input_source_hw_total} sw={input_source_sw_total} "
              f"mismatch={input_source_hist_mismatch}/{n_images}")
        print(f"  Input-source CT bitmap:  consumed={input_source_ct_hw_total} "
              f"fired_minus_consumed={input_source_ct_source_gap}")
    else:
        print(f"  HW count vs SW count pred:{direct_count_pred_matches}/{n_images} "
              f"({direct_count_pred_matches/max(n_images,1)*100:.2f}%)")
        print(f"  HW count+score vs SW:    {direct_count_score_pred_matches}/{n_images} "
              f"({direct_count_score_pred_matches/max(n_images,1)*100:.2f}%)")
        print(f"  HW/SW class-count vector: {direct_count_vector_matches}/{n_images} "
              f"({direct_count_vector_matches/max(n_images,1)*100:.2f}%)")
    print(f"  HW acc confirmed path:   {hw_correct}/{n_images} ({hw_correct/max(n_images,1)*100:.2f}%)")
    print(f"  S2MM first-spike acc:    {s2mm_correct}/{max(s2mm_samples,1)} ({s2mm_correct/max(s2mm_samples,1)*100:.2f}%)")
    print(f"  HW pred source:          count+score={total_count_score}, score={total_score}, "
          f"count={total_count}, rtl={total_rtl}, s2mm={total_s2mm}, fallback={fallback_count}")
    print(f"  Total input words:       {total_input}")
    print(f"  Total router deliveries: {total_router} ({100.0*total_router/max(total_input,1):.2f}% of input)")
    print(f"  Total neuron fires:      {total_neuron} ({100.0*total_neuron/max(total_input,1):.2f}% of input)")
    print(f"  HLS input incomplete:    {hls_input_incomplete}/{n_images}")
    print(f"  Profile incomplete:      {profile_incomplete}/{n_images}")
    print("  Communication profile totals:")
    print(f"    router_busy_cycles:    {router_busy_total}")
    print(f"    router_stall_cycles:   {router_stall_total}")
    if core_group_busy_total is not None:
        print(f"    core_group_busy_cycles:{core_group_busy_total}")
    if intra_route_cycles_total is not None:
        print(f"    intra_route_cycles:    {intra_route_cycles_total}")
    print(f"    total_latency_cycles:  {total_latency_total}")
    print(f"    out_fifo_drop_count:   {out_fifo_overflow_drop_total}")
    print(f"\n  CT mode check ({args.ct_mode}):")
    if profile_basic_only:
        print("    unavailable in BASIC profile mode "
              "(output/invalid/per-group detail counters are hidden)")
        print(f"    ct_lookup_count:       {ct_lookup_total}")
        print(f"    ct_valid_entry_count:  {ct_valid_total}")
        print(f"    cross_group_event_count:{ct_cross_total}")
        print(f"    same_group_event_count: {ct_same_total}")
    else:
        print(f"    output_spike_count:    {ct_output_total}")
    if (not profile_basic_only) and ct_expected_total is None:
        print(f"    ct_lookup_count:       {ct_lookup_total}")
        print(f"    ct_valid_entry_count:  {ct_valid_total}")
        print(f"    ct_invalid_entry_count:{ct_invalid_total}")
        print(f"    cross/same fanout:     cross={ct_cross_total} same={ct_same_total}")
        print("    per-sample mismatch:   N/A (measured-only pattern)")
    elif not profile_basic_only:
        print(f"    ct_lookup_count:       {ct_lookup_total} "
              f"(expected {ct_expected_total['lookup']})")
        print(f"    ct_valid_entry_count:  {ct_valid_total} "
              f"(expected {ct_expected_total['valid']})")
        print(f"    ct_invalid_entry_count:{ct_invalid_total} "
              f"(expected {ct_expected_total['invalid']})")
        print(f"    cross/same fanout:     cross={ct_cross_total}/{ct_expected_total['cross']} "
              f"same={ct_same_total}/{ct_expected_total['same']}")
        print(f"    per-sample mismatch:   {ct_mode_mismatch}/{n_images} "
              f"({'OK' if ct_mode_ok else 'CHECK'})")
    if pl_cycles_mean is not None:
        print(f"  PL-only latency (first input->first output): "
              f"mean={pl_cycles_mean:.1f} cyc ({pl_ms_mean:.4f} ms), "
              f"min={pl_cycles_min} cyc ({pl_ms_min:.4f} ms), "
              f"max={pl_cycles_max} cyc ({pl_ms_max:.4f} ms) @ {pl_clock_mhz:.6f} MHz")
    else:
        print("  PL-only latency (first input->first output): unavailable (all-zero cycle samples)")
    if pl_service_cycles_mean is not None:
        print(f"  PL-only service time (first input->idle): "
              f"mean={pl_service_cycles_mean:.1f} cyc ({pl_service_ms_mean:.4f} ms), "
              f"min={pl_service_cycles_min} cyc ({pl_service_ms_min:.4f} ms), "
              f"max={pl_service_cycles_max} cyc ({pl_service_ms_max:.4f} ms) @ {pl_clock_mhz:.6f} MHz")
    else:
        print("  PL-only service time (first input->idle): unavailable (register missing or all-zero samples)")
    if pl_first_spike_tput_img_s is not None:
        print(f"  PL-only throughput (first-spike window): {pl_first_spike_tput_img_s:.2f} img/s")
    if pl_service_tput_img_s is not None:
        print(f"  PL-only throughput (service window): {pl_service_tput_img_s:.2f} img/s")
    print("  PL compute estimate:")
    print(f"    synaptic_ops_total:     {synaptic_ops_total} "
          f"(ct_valid={ct_valid_total}, intra_nonzero={intra_weight_nonzero_total})")
    print(f"    peak_sops_per_cycle:    {peak_parallel_sops_per_cycle} "
          f"(router_lanes={peak_router_lanes}, "
          f"intra_lanes={peak_intra_lanes_per_group} x groups={profile_num_groups})")
    print(f"    peak GSOP/S:            {peak_gsops:.6f}")
    if gsops_first_spike is not None:
        print(f"    GSOP/S first-spike:     {gsops_first_spike:.6f}")
    if gsops_service is not None:
        print(f"    GSOP/S service:         {gsops_service:.6f}")
    if gsops_service_util_percent is not None:
        print(f"    service/peak util:      {gsops_service_util_percent:.4f}%")
    if power_report.get("available"):
        total_w = power_report.get("total_on_chip_w")
        dyn_w = power_report.get("dynamic_w")
        static_w = power_report.get("device_static_w")
        ps_w = power_report.get("ps7_dynamic_w")
        pl_w = power_report.get("pl_dynamic_w")
        print("  Vivado power estimate:")
        print(f"    report:                 {power_report.get('path')}")
        print(f"    total_on_chip_w:        {total_w:.3f}" if total_w is not None else "    total_on_chip_w:        N/A")
        print(f"    dynamic_w:              {dyn_w:.3f}" if dyn_w is not None else "    dynamic_w:              N/A")
        print(f"    device_static_w:        {static_w:.3f}" if static_w is not None else "    device_static_w:        N/A")
        print(f"    ps7_dynamic_w:          {ps_w:.3f}" if ps_w is not None else "    ps7_dynamic_w:          N/A")
        print(f"    pl_dynamic_w_excl_ps7:  {pl_w:.3f}" if pl_w is not None else "    pl_dynamic_w_excl_ps7:  N/A")
        print(f"    confidence:             {power_report.get('confidence') or 'N/A'}")
        if gsops_per_w_service is not None:
            print(f"    service GSOP/S/W:       {gsops_per_w_service:.6f}")
    else:
        print("  Vivado power estimate:    unavailable "
              "(pass --power-report or copy snn_core_group_profile_power.rpt)")
    if util_report.get("available"):
        print("  Vivado resource utilization:")
        print(f"    report:                 {util_report.get('path')}")
        print(f"    LUT:                    {util_report.get('lut')} / "
              f"{util_report.get('lut_available')} ({util_report.get('lut_util_percent'):.2f}%)")
        print(f"    FF:                     {util_report.get('ff')} / "
              f"{util_report.get('ff_available')} ({util_report.get('ff_util_percent'):.2f}%)")
        print(f"    BRAM tile:              {util_report.get('bram')} / "
              f"{util_report.get('bram_available')} ({util_report.get('bram_util_percent'):.2f}%)")
        print(f"    DSP:                    {util_report.get('dsp')} / "
              f"{util_report.get('dsp_available')} ({util_report.get('dsp_util_percent'):.2f}%)")
    else:
        print("  Vivado resource utilization: unavailable "
              "(pass --util-report or copy snn_core_group_profile_utilization.rpt)")

    print("\n  Timing decomposition (per image, host-side measured):")
    print(f"    iter_wall_ms_mean:    {mean_iter:.6f}")
    print(f"    sw_ref_ms_mean:       {sw_ref_ms_mean:.6f}" if sw_ref_ms_mean is not None else "    sw_ref_ms_mean:       N/A")
    print(f"    reset_ms_mean:        {reset_ms_mean:.6f}" if reset_ms_mean is not None else "    reset_ms_mean:        N/A")
    print(f"    spike_pack_ms_mean:   {spike_pack_ms_mean:.6f}" if spike_pack_ms_mean is not None else "    spike_pack_ms_mean:   N/A")
    print(f"    hw_run_ms_mean:       {hw_run_ms_mean:.6f}" if hw_run_ms_mean is not None else "    hw_run_ms_mean:       N/A")
    print(f"      dma_reset_ms_mean:  {hw_dma_reset_ms_mean:.6f}" if hw_dma_reset_ms_mean is not None else "      dma_reset_ms_mean:  N/A")
    print(f"      stream_poll_ms_mean:{hw_stream_poll_ms_mean:.6f}" if hw_stream_poll_ms_mean is not None else "      stream_poll_ms_mean:N/A")
    print(f"      mm2s_tail_ms_mean:  {hw_mm2s_tail_ms_mean:.6f}" if hw_mm2s_tail_ms_mean is not None else "      mm2s_tail_ms_mean:  N/A")
    print(f"      hls_input_ms_mean:  {hw_hls_input_ms_mean:.6f}" if hw_hls_input_ms_mean is not None else "      hls_input_ms_mean:  N/A")
    print(f"      settle_ms_mean:     {hw_settle_ms_mean:.6f}" if hw_settle_ms_mean is not None else "      settle_ms_mean:     N/A")
    print(f"      stop_ms_mean:       {hw_stop_ms_mean:.6f}" if hw_stop_ms_mean is not None else "      stop_ms_mean:       N/A")
    print(f"      readback_ms_mean:   {hw_readback_ms_mean:.6f}" if hw_readback_ms_mean is not None else "      readback_ms_mean:   N/A")
    print(f"      tracked_sleep_ms:   {hw_tracked_sleep_ms_mean:.6f}" if hw_tracked_sleep_ms_mean is not None else "      tracked_sleep_ms:   N/A")
    if non_pl_overhead_ms_mean is not None and non_pl_overhead_ratio is not None:
        print(f"    non_pl_overhead_ms_mean (iter - pl_service): {non_pl_overhead_ms_mean:.6f} "
              f"({non_pl_overhead_ratio:.4f}% of iter)")

    print_diagnostics(hls, cfg, dma, hwh_meta, profile_info)

    route_cfg_summary = {
        k: v for k, v in route_cfg.items()
        if k != "ct_real_fanout_mask"
    }

    summary = {
        "n_images": n_images,
        "elapsed_s": elapsed,
        "ms_per_image": elapsed / max(n_images, 1) * 1000.0,
        "bitstream": bit_path,
        "hwh": hwh_meta,
        "hls_version": hls_ver,
        "profile_info": int(profile_info),
        "profile_basic_only": bool(profile_basic_only),
        "route_config": route_cfg_summary,
        "neuron_map_mode": args.neuron_map,
        "neuron_map_group_counts": group_hist,
        "logical_to_hw": [int(x) for x in logical_to_hw.tolist()],
        "input_logical_to_hw": [int(x) for x in input_logical_to_hw.tolist()],
        "hw_threshold": hw_threshold,
        "n_neurons": n_neurons,
        "total_input_words": total_input,
        "total_router_spikes": total_router,
        "total_neuron_spikes": total_neuron,
        "hw_pred_source_count_score_count": int(total_count_score),
        "hw_pred_source_score_count": int(total_score),
        "hw_pred_source_count_count": int(total_count),
        "hw_pred_source_rtl_count": int(total_rtl),
        "hw_pred_source_s2mm_count": int(total_s2mm),
        "direct_count_pred_matches": int(direct_count_pred_matches),
        "direct_count_score_pred_matches": int(direct_count_score_pred_matches),
        "direct_count_vector_matches": int(direct_count_vector_matches),
        "sw_ct_pruned_correct": int(sw_ct_pruned_correct),
        "sw_ct_pruned_samples": int(sw_ct_pruned_samples),
        "hw_matches_sw_ct_pruned": int(hw_matches_sw_ct_pruned),
        "sw_ct_count_score_samples": int(sw_ct_count_score_samples),
        "hw_matches_sw_ct_count_score": int(hw_matches_sw_ct_count_score),
        "sw_hwct_count_score_samples": int(sw_hwct_count_score_samples),
        "hw_matches_sw_hwct_count_score": int(hw_matches_sw_hwct_count_score),
        "input_source_hw_total": int(input_source_hw_total),
        "input_source_sw_total": int(input_source_sw_total),
        "input_source_hist_mismatch": int(input_source_hist_mismatch),
        "input_source_ct_hw_total": int(input_source_ct_hw_total),
        "input_source_ct_source_gap": int(input_source_ct_source_gap),
        "hls_input_incomplete_count": hls_input_incomplete,
        "profile_incomplete_count": profile_incomplete,
        "ct_mode": args.ct_mode,
        "ct_mode_check": {
            "ok": bool(ct_mode_ok),
            "per_sample_mismatch": int(ct_mode_mismatch),
            "output_spike_count": int(ct_output_total),
            "ct_lookup_count": int(ct_lookup_total),
            "ct_valid_entry_count": int(ct_valid_total),
            "ct_invalid_entry_count": int(ct_invalid_total),
            "cross_group_event_count": int(ct_cross_total),
            "same_group_event_count": int(ct_same_total),
            "intra_weight_nonzero_count": int(intra_weight_nonzero_total),
            "router_busy_cycles": int(router_busy_total),
            "router_stall_cycles": int(router_stall_total),
            "total_latency_cycles": int(total_latency_total),
            "out_fifo_overflow_drop_count": int(out_fifo_overflow_drop_total),
            "expected": ct_expected_total,
        },
        "pl_compute_estimate": {
            "synaptic_ops_total": int(synaptic_ops_total),
            "synaptic_ops_definition": "ct_valid_entry_count + intra_weight_nonzero_count(if present)",
            "peak_sops_per_cycle": int(peak_parallel_sops_per_cycle),
            "peak_router_lanes": int(peak_router_lanes),
            "peak_intra_lanes_per_group": int(peak_intra_lanes_per_group),
            "peak_groups": int(profile_num_groups),
            "peak_gsops": peak_gsops,
            "gsops_first_spike": gsops_first_spike,
            "gsops_service": gsops_service,
            "gsops_service_util_percent": gsops_service_util_percent,
            "gsops_per_w_service": gsops_per_w_service,
        },
        "vivado_power_report": power_report,
        "vivado_utilization_report": util_report,
        "ct_dummy_sink_ids": [
            (None if hw is None else int(hw)) for hw in dummy_sink_ids
        ],
        "pl_clock_hz": pl_clock_hz,
        "pl_clock_mhz": pl_clock_mhz,
        "pl_latency_cycles_mean": pl_cycles_mean,
        "pl_latency_cycles_min": pl_cycles_min,
        "pl_latency_cycles_max": pl_cycles_max,
        "pl_latency_ms_mean": pl_ms_mean,
        "pl_latency_ms_min": pl_ms_min,
        "pl_latency_ms_max": pl_ms_max,
        "pl_service_cycles_mean": pl_service_cycles_mean,
        "pl_service_cycles_min": pl_service_cycles_min,
        "pl_service_cycles_max": pl_service_cycles_max,
        "pl_service_ms_mean": pl_service_ms_mean,
        "pl_service_ms_min": pl_service_ms_min,
        "pl_service_ms_max": pl_service_ms_max,
        "pl_throughput_img_s_first_spike": pl_first_spike_tput_img_s,
        "pl_throughput_img_s_service": pl_service_tput_img_s,
        "timing_breakdown_ms_mean": {
            "iter_wall": mean_iter,
            "sw_ref": sw_ref_ms_mean,
            "reset": reset_ms_mean,
            "spike_pack": spike_pack_ms_mean,
            "hw_run": hw_run_ms_mean,
            "hw_dma_reset": hw_dma_reset_ms_mean,
            "hw_stream_poll": hw_stream_poll_ms_mean,
            "hw_mm2s_tail": hw_mm2s_tail_ms_mean,
            "hw_hls_input": hw_hls_input_ms_mean,
            "hw_settle": hw_settle_ms_mean,
            "hw_stop": hw_stop_ms_mean,
            "hw_counter_read": hw_readback_ms_mean,
            "hw_tracked_sleep": hw_tracked_sleep_ms_mean,
            "non_pl_overhead": non_pl_overhead_ms_mean,
            "non_pl_overhead_ratio_percent": non_pl_overhead_ratio,
        },
        "results": results,
    }
    with open(result_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {result_path}")

    if profile_path is not None:
        rows = []
        for row in results:
            out = {
                "idx": row["idx"],
                "label": row["label"],
                "sw_pred_ttfs": row.get("sw_pred_ttfs"),
                "sw_pred_count": row.get("sw_pred_count"),
                "sw_direct_count_pred": row.get("sw_direct_count_pred"),
                "direct_count_pred_match": int(row.get("direct_count_pred_match", False)),
                "direct_count_score_pred_match": int(row.get("direct_count_score_pred_match", False)),
                "direct_count_vector_match": int(row.get("direct_count_vector_match", False)),
                "sw_ct_pruned_pred": row.get("sw_ct_pruned_pred"),
                "sw_ct_count_score_pred": row.get("sw_ct_count_score_pred"),
                "sw_ct_source_fire_count": row.get("sw_ct_source_fire_count"),
                "sw_ct_valid_event_budget": row.get("sw_ct_valid_event_budget"),
                "sw_hwct_pruned_pred": row.get("sw_hwct_pruned_pred"),
                "sw_hwct_count_score_pred": row.get("sw_hwct_count_score_pred"),
                "sw_hwct_source_fire_count": row.get("sw_hwct_source_fire_count"),
                "sw_hwct_valid_event_budget": row.get("sw_hwct_valid_event_budget"),
                "hw_pred": row.get("hw_pred"),
                "hw_pred_score": row.get("hw_pred_score"),
                "hw_pred_count": row.get("hw_pred_count"),
                "hw_pred_count_raw": row.get("hw_pred_count_raw"),
                "hw_pred_count_score": row.get("hw_pred_count_score"),
                "hw_pred_s2mm": row.get("hw_pred_s2mm"),
                "hw_pred_rtl_first_classifier": row.get("hw_pred_rtl_first_classifier"),
                "input_words": row.get("input_words"),
                "router_spikes": row.get("router_spikes"),
                "neuron_spikes": row.get("neuron_spikes"),
                "hls_spike_count": row.get("hls_spike_count"),
                "hls_spike_target": row.get("hls_spike_target"),
                "hls_input_done": int(row.get("hls_input_done", False)),
                "mm2s_done": int(row.get("mm2s_done", False)),
                "pl_latency_cycles": row.get("pl_latency_cycles"),
                "pl_service_cycles": row.get("pl_service_cycles"),
                "mm2s_sr": row.get("mm2s_sr"),
                "s2mm_sr": row.get("s2mm_sr"),
                "hw_matches_sw_ct_pruned": int(
                    row.get("sw_ct_pruned_pred") is not None and
                    row.get("hw_pred") == row.get("sw_ct_pruned_pred")
                ),
                "hw_matches_sw_ct_count_score": int(
                    row.get("sw_ct_count_score_pred") is not None and
                    row.get("hw_pred_count_score") == row.get("sw_ct_count_score_pred")
                ),
                "hw_matches_sw_hwct_count_score": int(
                    row.get("sw_hwct_count_score_pred") is not None and
                    row.get("hw_pred_count_score") == row.get("sw_hwct_count_score_pred")
                ),
                "profile_incomplete": int(row["profile_incomplete"]),
            }
            sw_ct_counts = row.get("sw_ct_pruned_class_counts", [])
            sw_direct_counts = row.get("sw_direct_class_counts", [])
            sw_ct_scores = row.get("sw_ct_pruned_class_scores", [])
            sw_ct_events = row.get("sw_ct_pruned_class_events", [])
            sw_hwct_counts = row.get("sw_hwct_pruned_class_counts", [])
            sw_hwct_scores = row.get("sw_hwct_pruned_class_scores", [])
            sw_hwct_events = row.get("sw_hwct_pruned_class_events", [])
            hw_counts = row.get("hw_class_counts", [])
            hw_scores = row.get("hw_class_scores", [])
            hw_events = row.get("hw_class_events", [])
            router_scores = row.get("profile", {}).get("router_class_scores", [])
            router_events = row.get("profile", {}).get("router_class_events", [])
            top_scores = row.get("profile", {}).get("top_class_scores", [])
            top_events = row.get("profile", {}).get("top_class_events", [])
            sw_input_source_hist = row.get("sw_ct_input_source_group_hist", [])
            hw_input_source_hist = row.get("hw_input_source_group_hist", [])
            for cls in range(n_classes):
                out[f"sw_direct_class_{cls}_count"] = int(sw_direct_counts[cls]) if cls < len(sw_direct_counts) else 0
                out[f"sw_ct_class_{cls}_count"] = int(sw_ct_counts[cls]) if cls < len(sw_ct_counts) else 0
                out[f"hw_class_{cls}_count"] = int(hw_counts[cls]) if cls < len(hw_counts) else 0
                out[f"hw_minus_sw_direct_class_{cls}_count"] = (
                    int(hw_counts[cls]) - int(sw_direct_counts[cls])
                    if cls < len(hw_counts) and cls < len(sw_direct_counts) else 0
                )
                out[f"sw_ct_class_{cls}_score"] = int(sw_ct_scores[cls]) if cls < len(sw_ct_scores) else 0
                out[f"sw_hwct_class_{cls}_count"] = int(sw_hwct_counts[cls]) if cls < len(sw_hwct_counts) else 0
                out[f"sw_hwct_class_{cls}_score"] = int(sw_hwct_scores[cls]) if cls < len(sw_hwct_scores) else 0
                out[f"sw_hwct_class_{cls}_event_count"] = int(sw_hwct_events[cls]) if cls < len(sw_hwct_events) else 0
                out[f"hw_class_{cls}_score"] = int(hw_scores[cls]) if cls < len(hw_scores) else 0
                out[f"sw_ct_class_{cls}_event_count"] = int(sw_ct_events[cls]) if cls < len(sw_ct_events) else 0
                out[f"hw_class_{cls}_event_count"] = int(hw_events[cls]) if cls < len(hw_events) else 0
                out[f"router_class_{cls}_score"] = int(router_scores[cls]) if cls < len(router_scores) else 0
                out[f"router_class_{cls}_event_count"] = int(router_events[cls]) if cls < len(router_events) else 0
                out[f"top_class_{cls}_score"] = int(top_scores[cls]) if cls < len(top_scores) else 0
                out[f"top_class_{cls}_event_count"] = int(top_events[cls]) if cls < len(top_events) else 0
            for group in range(profile_num_groups):
                sw_group_count = int(sw_input_source_hist[group]) if group < len(sw_input_source_hist) else 0
                hw_group_count = int(hw_input_source_hist[group]) if group < len(hw_input_source_hist) else 0
                out[f"sw_input_source_group_{group}_spike_count"] = sw_group_count
                out[f"hw_input_source_group_{group}_spike_count"] = hw_group_count
                out[f"input_source_group_{group}_spike_diff"] = hw_group_count - sw_group_count
            out.update(row.get("profile", {}))
            expected = None if profile_basic_only else ct_mode_expected_counts(
                int(out.get("output_spike_count", 0) or 0),
                args.ct_mode,
            )
            if expected is None:
                out["ct_lookup_minus_expected"] = ""
                out["ct_valid_minus_expected"] = ""
                out["ct_invalid_minus_expected"] = ""
                out["ct_cross_minus_expected"] = ""
                out["ct_same_minus_expected"] = ""
                out["ct_mode_ok"] = "" if profile_basic_only else 1
            else:
                out["ct_lookup_minus_expected"] = int(out.get("ct_lookup_count", 0) or 0) - expected["lookup"]
                out["ct_valid_minus_expected"] = int(out.get("ct_valid_entry_count", 0) or 0) - expected["valid"]
                out["ct_invalid_minus_expected"] = int(out.get("ct_invalid_entry_count", 0) or 0) - expected["invalid"]
                out["ct_cross_minus_expected"] = int(out.get("cross_group_event_count", 0) or 0) - expected["cross"]
                out["ct_same_minus_expected"] = int(out.get("same_group_event_count", 0) or 0) - expected["same"]
                out["ct_mode_ok"] = int(
                    out["ct_lookup_minus_expected"] == 0 and
                    out["ct_valid_minus_expected"] == 0 and
                    out["ct_invalid_minus_expected"] == 0 and
                    out["ct_cross_minus_expected"] == 0 and
                    out["ct_same_minus_expected"] == 0
                )
            rows.append(out)
        fieldnames = list(rows[0].keys()) if rows else ["idx", "label", "profile_incomplete"]
        with open(profile_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved profile CSV: {profile_path}")

    hls.close()
    cfg.close()
    dma.close()
    buf_in.close()
    buf_out.close()


if __name__ == "__main__":
    main()
