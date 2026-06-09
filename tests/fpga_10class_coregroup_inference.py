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


def decode_global_id(nid: int, local_id_width: int) -> tuple[int, int]:
    return int(nid >> local_id_width), int(nid & ((1 << local_id_width) - 1))


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


def encode_intra_weight(src_global: int,
                        dst_global: int,
                        weight: int,
                        exc: bool = True,
                        local_id_width: int = DEFAULT_LOCAL_ID_WIDTH) -> tuple[int, int]:
    """Encode snn_core_group_top intra-group weight write format."""
    src_group, src_neuron = decode_global_id(src_global, local_id_width)
    dst_group, dst_neuron = decode_global_id(dst_global, local_id_width)
    if src_group != dst_group:
        raise ValueError("intra weight requires src and dst in the same group")
    addr = (0x1 << 28)
    data = (src_neuron & 0x7F) << 25
    data |= (dst_neuron & 0x7F) << 18
    data |= (int(weight) & 0xFF) << 10
    data |= (1 if exc else 0) << 9
    data |= (src_group & 0xF) << 5
    return addr, data


def cfg_write_pair(cfg: legacy.MMIO, addr: int, data: int) -> None:
    cfg.write(CFG_CONFIG_CTRL, 0)
    cfg.write(CFG_CONFIG_ADDR, addr)
    cfg.write(CFG_CONFIG_WDATA, data)


def clear_ct_fanout0(cfg: legacy.MMIO,
                     n_neurons: int,
                     local_id_width: int = DEFAULT_LOCAL_ID_WIDTH) -> None:
    """Write invalid fanout-0 terminators for all output neurons used here."""
    for nid in range(n_neurons):
        addr, data = encode_ct_entry(
            src_global=nid,
            fanout_idx=0,
            dst_global=0,
            weight=0,
            valid=False,
            exc=True,
            local_id_width=local_id_width,
        )
        cfg_write_pair(cfg, addr, data)


def clear_intra_rows(cfg: legacy.MMIO,
                     n_neurons: int,
                     local_id_width: int = DEFAULT_LOCAL_ID_WIDTH,
                     group_size: int = 128) -> int:
    """Clear local recurrent rows for neurons used by the 10-class model."""
    writes = 0
    for src in range(n_neurons):
        src_group, _ = decode_global_id(src, local_id_width)
        row_start = src_group * group_size
        for local_dst in range(group_size):
            dst = row_start + local_dst
            addr, data = encode_intra_weight(src, dst, 0, True, local_id_width)
            cfg_write_pair(cfg, addr, data)
            writes += 1
    return writes


def program_coregroup_for_inference(cfg: legacy.MMIO,
                                    n_neurons: int,
                                    local_id_width: int,
                                    clear_intra: bool = False) -> dict:
    """Configure the new route for direct external delivery inference."""
    t0 = time.perf_counter()
    clear_ct_fanout0(cfg, n_neurons, local_id_width=local_id_width)
    intra_writes = 0
    if clear_intra:
        intra_writes = clear_intra_rows(cfg, n_neurons, local_id_width=local_id_width)
    return {
        "ct_invalid_terminators": int(n_neurons),
        "intra_zero_writes": int(intra_writes),
        "config_ms": (time.perf_counter() - t0) * 1000.0,
    }


def build_direct_spike_words(image: np.ndarray,
                             q_weights: np.ndarray,
                             hls_spike_pkt_id_w: int,
                             pixel_th: float = 0.3,
                             potential: np.ndarray | None = None) -> np.ndarray:
    """Use global neuron IDs directly; event_router_ng delivers to that group."""
    if potential is None:
        potential = legacy.compute_positive_potential(image, q_weights, pixel_th=pixel_th)
    order = legacy.ttfs_order_from_potential(potential)
    pos_order = order[potential[order] > 0]
    if pos_order.size == 0:
        return np.zeros(1, dtype=np.uint32)
    id_mask = (1 << hls_spike_pkt_id_w) - 1
    return (pos_order.astype(np.uint32) & np.uint32(id_mask)) | np.uint32(0x7F << hls_spike_pkt_id_w)


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
    n_images = int(args.n) if int(args.n) > 0 else len(test_imgs)
    print(f"  q_weights: {q_weights.shape}  hw_threshold: {hw_threshold}")
    print(f"  n_neurons: {n_neurons}  n_classes: {n_classes}  fps: {fps_per_class}")
    print(f"  Test images: {n_images}")

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
    print(f"  Input mapping: spike packet neuron_id = target global neuron_id, no source_offset")
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
    if profile_info != 0xDEADBEEF and ((profile_info >> 24) & 0xFF) == 1:
        profile_num_groups = (profile_info >> 16) & 0xFF
        print(f"  profile info: 0x{profile_info:08X}  groups={profile_num_groups}")
    elif profile_enabled:
        print("ERROR: --profile-output requested, but profile window is not available.")
        sys.exit(1)

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
        clear_intra=bool(args.clear_intra),
    )
    legacy.configure_neurons(cfg, threshold=hw_threshold, leak=0, refrac=0)
    print(f"  Done. CT invalid terminators={route_cfg['ct_invalid_terminators']} "
          f"intra_zero_writes={route_cfg['intra_zero_writes']} "
          f"config_ms={route_cfg['config_ms']:.3f}")

    print_diagnostics(hls, cfg, dma, hwh_meta, profile_info)

    n_buf_words = n_neurons + 16
    buf_in = legacy.make_dma_buffer(DMA_BUF_IN, n_buf_words * 4 + 64, label="MM2S")
    buf_out = legacy.make_dma_buffer(DMA_BUF_OUT, n_buf_words * 4 + 64, label="S2MM")

    print(f"\nRunning {n_images} inference{'s' if n_images != 1 else ''} ...")
    print("-" * 100)
    print(f"{'idx':>5} {'lbl':>4} {'sw_t':>5} {'sw_c':>5} {'hw':>4} {'src':>4} {'acc':>5} | "
          f"{'router':>7} {'neuron':>7} {'hls_in':>7} {'lat':>6} {'svc':>6} "
          f"{'mm2s':>10} {'s2mm':>10} {'cfg':>8} {'#s2':>4}")
    print("-" * 100)

    results = []
    sw_correct = 0
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
        if sw_pred == lbl:
            sw_correct += 1

        t_pack0 = time.perf_counter()
        spike_words = build_direct_spike_words(
            img, q_weights, hls_spike_pkt_id_w=hls_packet_id_w, potential=potential
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
            n_neurons=n_neurons,
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

        hw_pred_s2 = legacy.classify_hw_s2mm_first_spike(hw, n_classes, fps_per_class)
        if hw_pred_s2 is None:
            hw_pred = sw_pred
            src = "SW"
            fallback_count += 1
        else:
            hw_pred = int(hw_pred_s2)
            src = "S2"
            s2mm_samples += 1
            if hw_pred == lbl:
                s2mm_correct += 1
        if hw_pred == lbl and int(hw.get("neuron_spikes", 0)) > 0:
            hw_correct += 1

        t_iter1 = time.perf_counter()
        timing = hw.get("timing", {})
        iter_ms.append((t_iter1 - t_iter0) * 1000.0)
        mm2s_tail_ms.append(float(timing.get("mm2s_tail_wait_ms", 0.0)))

        cfg_status = int(hw.get("status", cfg.read(CFG_STATUS))) if "status" in hw else cfg.read(CFG_STATUS)
        pl_latency = int(hw.get("pl_latency_cycles", 0))
        pl_service = hw.get("pl_service_cycles", None)
        pl_service_print = int(pl_service) if pl_service is not None else 0

        if n_images <= 200 or idx % max(1, int(args.print_every)) == 0 or hw_pred != sw_pred:
            print(f"{idx:>5d} {lbl:>4d} {sw_pred:>5d} {sw_count:>5d} {hw_pred:>4d} {src:>4s} "
                  f"{'OK' if sw_pred == lbl else 'DIFF':>5s} | "
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
            "hw_pred": int(hw_pred),
            "hw_pred_source": src,
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
    total_s2mm = sum(1 for r in results if r["hw_pred_source"] == "S2")
    profile_incomplete = sum(1 for r in results if r["profile_incomplete"])
    hls_input_incomplete = sum(1 for r in results if not r["hls_input_done"])
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
    print(f"  HW acc confirmed path:   {hw_correct}/{n_images} ({hw_correct/max(n_images,1)*100:.2f}%)")
    print(f"  S2MM first-spike acc:    {s2mm_correct}/{max(s2mm_samples,1)} ({s2mm_correct/max(s2mm_samples,1)*100:.2f}%)")
    print(f"  HW pred source:          s2mm={total_s2mm}, fallback={fallback_count}")
    print(f"  Total input words:       {total_input}")
    print(f"  Total router deliveries: {total_router} ({100.0*total_router/max(total_input,1):.2f}% of input)")
    print(f"  Total neuron fires:      {total_neuron} ({100.0*total_neuron/max(total_input,1):.2f}% of input)")
    print(f"  HLS input incomplete:    {hls_input_incomplete}/{n_images}")
    print(f"  Profile incomplete:      {profile_incomplete}/{n_images}")
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

    summary = {
        "n_images": n_images,
        "elapsed_s": elapsed,
        "ms_per_image": elapsed / max(n_images, 1) * 1000.0,
        "bitstream": bit_path,
        "hwh": hwh_meta,
        "hls_version": hls_ver,
        "profile_info": int(profile_info),
        "route_config": route_cfg,
        "hw_threshold": hw_threshold,
        "n_neurons": n_neurons,
        "total_input_words": total_input,
        "total_router_spikes": total_router,
        "total_neuron_spikes": total_neuron,
        "hls_input_incomplete_count": hls_input_incomplete,
        "profile_incomplete_count": profile_incomplete,
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
                "profile_incomplete": int(row["profile_incomplete"]),
            }
            out.update(row.get("profile", {}))
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
