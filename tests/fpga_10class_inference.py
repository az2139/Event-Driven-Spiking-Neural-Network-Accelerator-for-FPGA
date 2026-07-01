#!/usr/bin/env python3
"""
10-Class MNIST Inference on PYNQ-Z2 FPGA (v2 bitstream)
=========================================================
Loads the 150-neuron FaithfulOnChipTrainer weights (784→150 LIF, 10 classes ×
15 neurons each) and runs all 10,000 MNIST test images through the FPGA.

Key fixes versus the earlier pynq_mnist_inference.py:
  • S2MM DMA is armed BEFORE HLS is started (race-condition fix)
  • S2MM status is polled instead of a fixed sleep
  • First-spike-only capture mode for deterministic strict parity (optional all-spikes debug mode)
  • Detailed S2MM diagnostic counters printed at the end
  • Uses HLS packet neuron-id width (13b) for AXIS word packing/decoding
  • Uses time_steps=1 + auto_restart (required for current ap_none export timing)

Usage (on PYNQ board as root):
    sudo python3 fpga_10class_inference.py [--data /home/xilinx/snn] [--n N]

Expects in --data directory:
    snn_integrated_v2.bit              (FPGA bitstream)
    mnist_10class_deployment.npz       (weights + thresholds)

The deployment .npz can be generated on the host with:
    python3 tests/prepare_10class_deployment.py

Author: Jiwoon Lee
Date:   2026-02-21
"""

from __future__ import annotations

import argparse
import csv
import json
import mmap
import os
import re
import struct
import sys
import time
import traceback

import numpy as np
# =====================================================================
# Register Map  (snn_integrated v2 block design)
# =====================================================================

HLS_BASE            = 0x43C00000  # snn_top_hls_0
CFG_BASE            = 0x43C10000  # snn_config_regs_0
DMA_BASE            = 0x41E00000  # axi_dma_0

HLS_AP_CTRL         = 0x00
HLS_CTRL_REG        = 0x10   # [0]enable [1]reset [2]clear [7]first_spike_only
HLS_CONFIG_REG      = 0x18   # [15:0]threshold [31:16]leak_rate
HLS_MODE_REG        = 0x20   # [1:0]op_mode  0=inference
HLS_TIME_STEPS      = 0x28
# NOTE: Keep these in sync with outputs/snn_integrated_v2.hwh (snn_top_hls_0 Reg map)
HLS_STATUS_REG      = 0x58
HLS_STATUS_VLD      = 0x5C
HLS_SPIKE_COUNT     = 0x68
HLS_SPIKE_COUNT_VLD = 0x6C
HLS_VERSION_REG     = 0x88

CFG_CONFIG_CTRL     = 0x00
CFG_CONFIG_ADDR     = 0x04
CFG_CONFIG_WDATA    = 0x08
CFG_CONFIG_RDATA    = 0x0C
CFG_THRESHOLD       = 0x10
CFG_NEURON_PARAMS   = 0x14
CFG_ROUTER_SPKS     = 0x18
CFG_NEURON_SPKS     = 0x1C
CFG_STATUS          = 0x20
CFG_THROUGHPUT      = 0x24   # PL-only latency cycles (first input accept -> first output spike)
CFG_VERSION         = 0x28
CFG_SERVICE_CYCLES  = 0x2C   # PL-only service cycles (first input accept -> return to idle)
CFG_PROFILE_CTRL    = 0x30
CFG_PROFILE_INDEX   = 0x34
CFG_PROFILE_DATA    = 0x38
CFG_PROFILE_INFO    = 0x3C

PROFILE_START       = 0x01
PROFILE_STOP        = 0x02
STATUS_ROUTER_BUSY  = 1 << 1
STATUS_GROUP_BUSY   = 1 << 2
STATUS_SNN_READY    = 1 << 3
STATUS_PROFILE_ACTIVE = 1 << 4
STATUS_PROFILE_DONE = 1 << 5

PROFILE_ROUTER_NAMES = [
    'input_spike_count', 'output_spike_count', 'ct_lookup_count',
    'ct_valid_entry_count', 'ct_invalid_entry_count', 'router_busy_cycles',
    'router_idle_cycles', 'router_stall_cycles', 'cross_group_event_count',
    'same_group_event_count', 'total_latency_cycles',
]

PROFILE_BASIC_NAMES = [
    'total_latency_cycles',
    'ct_lookup_count',
    'ct_valid_entry_count',
    'router_busy_cycles',
    'router_stall_cycles',
    'cross_group_event_count',
    'same_group_event_count',
    'drop_spike_count',
    'service_dbg_input_done_cycle',
    'service_dbg_hls_pending_clear_cycle',
] + [
    f'class_{cls}_spike_count' for cls in range(10)
] + [
    f'class_{cls}_score' for cls in range(10)
] + [
    f'class_{cls}_event_count' for cls in range(10)
]

# AXI DMA
DMA_MM2S_DMACR      = 0x00
DMA_MM2S_DMASR      = 0x04
DMA_MM2S_SA         = 0x18
DMA_MM2S_LENGTH     = 0x28
DMA_S2MM_DMACR      = 0x30
DMA_S2MM_DMASR      = 0x34
DMA_S2MM_DA         = 0x48
DMA_S2MM_LENGTH     = 0x58

CTRL_ENABLE         = 0x01
CTRL_RESET          = 0x02
CTRL_CLEAR          = 0x04
CTRL_FIRST_SPIKE_ONLY = 0x80

EXPECTED_HLS_VERSION = 0x20260221

# Physical DDR addresses for DMA buffers (must be outside OS memory)
DMA_BUF_IN          = 0x1F000000   # input  spike words (MM2S)
DMA_BUF_OUT         = 0x1F100000   # output spike words (S2MM)

MAX_NEURONS         = 2048         # hardware neuron capacity
MAX_FANOUT          = 32
ROUTER_NEURON_ID_W  = 10           # matches RTL NEURON_ID_WIDTH=10 (critical for exc_inh bit)
SOURCE_OFFSET       = 256          # safer default across 512/1024-neuron integrated builds
                                   # LIF output IDs (0..n_neurons-1) get conn_count=0 → no recurrence

# HLS AXIS spike packet format (see hardware/hls/include/snn_top_hls.h):
#   [HLS_SPIKE_PKT_ID_W-1:0]          neuron_id
#   [HLS_SPIKE_PKT_ID_W+7:HLS_SPIKE_PKT_ID_W] weight
# NOTE:
#   This is different from ROUTER_NEURON_ID_W used by RTL conn memory and from
#   ap_none spike_in_neuron_id width exported in .hwh (typically 11b).
#   Packet ID width is the HLS logical ID width (currently 13b).
HLS_SPIKE_PKT_ID_W  = 13
PL_CLK_MHZ_DEFAULT  = 100.0

# Runtime timeout/sleep defaults (strict/parity oriented)
FIRST_SPIKE_TIMEOUT_MS_DEFAULT   = 50.0
FURTHER_SPIKE_TIMEOUT_MS_DEFAULT = 5.0
MM2S_TAIL_TIMEOUT_MS_DEFAULT     = 300.0
SETTLE_CAP_MS_DEFAULT            = 10.0
STOP_SLEEP_MS_DEFAULT            = 3.0

# Fast benchmark profile defaults (throughput oriented)
FIRST_SPIKE_TIMEOUT_MS_FAST   = 3.0
FURTHER_SPIKE_TIMEOUT_MS_FAST = 1.0
MM2S_TAIL_TIMEOUT_MS_FAST     = 10.0
SETTLE_CAP_MS_FAST            = 0.3
STOP_SLEEP_MS_FAST            = 0.0


# =====================================================================
# /dev/mem MMIO Helper
# =====================================================================

class MMIO:
    _PAGE = 4096

    def __init__(self, base: int, length: int = 0x1000):
        self._base   = base
        self.physical_address = int(base)
        self._length = int(length)
        self._closed = False
        self._off    = base % self._PAGE
        mbase        = base - self._off
        mlen         = ((length + self._off + self._PAGE - 1) // self._PAGE) * self._PAGE
        self._fd     = os.open('/dev/mem', os.O_RDWR | os.O_SYNC)
        self._mm     = mmap.mmap(self._fd, mlen, offset=mbase)

    def _check_bounds(self, offset: int, size: int) -> None:
        if self._closed:
            raise RuntimeError(f"MMIO 0x{self._base:08X} is already closed")
        if offset < 0 or size < 0 or offset + size > self._length:
            raise ValueError(
                f"MMIO access out of range: base=0x{self._base:08X} "
                f"offset=0x{offset:X} size={size} length={self._length}"
            )

    def read(self, offset: int) -> int:
        self._check_bounds(offset, 4)
        self._mm.seek(self._off + offset)
        return struct.unpack('<I', self._mm.read(4))[0]

    def write(self, offset: int, value: int) -> None:
        self._check_bounds(offset, 4)
        self._mm.seek(self._off + offset)
        self._mm.write(struct.pack('<I', value & 0xFFFF_FFFF))

    def write_bytes(self, offset: int, data: bytes) -> None:
        """Bulk write raw bytes at offset (for DMA buffer population)."""
        self._check_bounds(offset, len(data))
        self._mm.seek(self._off + offset)
        self._mm.write(data)

    def read_bytes(self, offset: int, length: int) -> bytes:
        """Bulk read raw bytes at offset (cache-coherent via O_SYNC mmap)."""
        self._check_bounds(offset, length)
        self._mm.seek(self._off + offset)
        return self._mm.read(length)

    def flush(self) -> None:
        """No-op on /dev/mem (O_SYNC already gives strongly-ordered access;
        msync/mmap.flush() raises EINVAL on device-backed mappings)."""
        pass

    def close(self) -> None:
        self._closed = True
        self._mm.close()
        os.close(self._fd)


class PynqDMABuffer:
    """Contiguous, non-cacheable DMA buffer allocated through PYNQ.

    Fixed /dev/mem DDR addresses work only if the region is reserved from Linux.
    For long runs, using Linux-managed DDR as an AXI-DMA target can corrupt the
    Python process or kernel memory.  PYNQ allocate() gives us a physical address
    that the DMA can safely use.
    """

    def __init__(self, length: int):
        

        self._length = int(length)
        self._closed = False
        self._words = (self._length + 3) // 4
        self._buf = allocate(shape=(self._words,), dtype=np.uint32, cacheable=False)
        self._u8 = self._buf.view(np.uint8)
        self.physical_address = int(self._buf.physical_address)

    def _check_bounds(self, offset: int, size: int) -> None:
        if self._closed:
            raise RuntimeError("DMA buffer is already closed")
        if offset < 0 or size < 0 or offset + size > self._length:
            raise ValueError(
                f"DMA buffer access out of range: phys=0x{self.physical_address:08X} "
                f"offset=0x{offset:X} size={size} length={self._length}"
            )

    def write_bytes(self, offset: int, data: bytes) -> None:
        self._check_bounds(offset, len(data))
        self._u8[offset:offset + len(data)] = np.frombuffer(data, dtype=np.uint8)
        if hasattr(self._buf, "flush"):
            self._buf.flush()

    def read_bytes(self, offset: int, length: int) -> bytes:
        self._check_bounds(offset, length)
        if hasattr(self._buf, "invalidate"):
            self._buf.invalidate()
        return bytes(self._u8[offset:offset + length])

    def flush(self) -> None:
        if hasattr(self._buf, "flush"):
            self._buf.flush()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if hasattr(self._buf, "freebuffer"):
            self._buf.freebuffer()
        elif hasattr(self._buf, "close"):
            self._buf.close()


def make_dma_buffer(fixed_base: int, length: int, label: str = "DMA"):
    """Prefer a PYNQ-owned DMA buffer, falling back to the legacy fixed DDR map."""
    try:
        buf = PynqDMABuffer(length)
        print(f"  {label} buffer: PYNQ allocate phys=0x{buf.physical_address:08X} bytes={length}")
        return buf
    except Exception as exc:
        print(f"  {label} buffer: WARNING falling back to fixed /dev/mem "
              f"0x{fixed_base:08X} bytes={length} ({exc})")
        return MMIO(fixed_base, length)


# =====================================================================
# Bitstream Programming (fpga_manager)
# =====================================================================

def bit_to_bin(bitfile: str, binfile: str) -> int:
    """Convert Xilinx .bit to byte-swapped .bin for fpga_manager."""
    with open(bitfile, 'rb') as f:
        data = f.read()
    sync = bytes([0xAA, 0x99, 0x55, 0x66])
    idx  = data.find(sync)
    if idx < 0:
        raise ValueError("Xilinx sync word not found in .bit file")
    raw  = data[idx:]
    n    = len(raw) & ~3
    swapped = bytearray(n)
    for i in range(0, n, 4):
        swapped[i]   = raw[i + 3]
        swapped[i+1] = raw[i + 2]
        swapped[i+2] = raw[i + 1]
        swapped[i+3] = raw[i]
    with open(binfile, 'wb') as f:
        f.write(bytes(swapped))
    return n


def program_fpga(bitfile: str) -> bool:
    """Program FPGA via fpga_manager."""
    binfile = bitfile.replace('.bit', '.bin')
    print(f"  bit→bin: {os.path.basename(bitfile)} → {os.path.basename(binfile)}")
    bit_to_bin(bitfile, binfile)
    fw_name = os.path.basename(binfile)
    fw_path = f'/lib/firmware/{fw_name}'
    with open(binfile, 'rb') as s, open(fw_path, 'wb') as d:
        d.write(s.read())
    with open('/sys/class/fpga_manager/fpga0/flags', 'w') as f:
        f.write('0')
    try:
        with open('/sys/class/fpga_manager/fpga0/firmware', 'w') as f:
            f.write(fw_name)
    except OSError:
        pass
    time.sleep(1.5)
    with open('/sys/class/fpga_manager/fpga0/state') as f:
        state = f.read().strip()
    print(f"  fpga_manager state: {state}")
    return state == 'operating'


# =====================================================================
# Router / Neuron Programming
# =====================================================================

def encode_conn(src: int, fanout: int, dest: int, weight: int,
                delay: int = 0, exc: bool = True) -> tuple:
    """Encode one spike_router connection entry."""
    flat = src * MAX_FANOUT + fanout
    addr = (0x00 << 24) | (flat & 0x00_FFFF)

    # Dual-width compatibility:
    # Some integrated bitstreams instantiate spike_router with NEURON_ID_WIDTH=11
    # while host scripts historically assumed 10. Pack a connection word that is
    # accepted by both 10-bit and 11-bit router variants.
    #
    # - valid10 bit = 27, valid11 bit = 28  -> set both
    # - exc10 bit   = 26, exc11 bit   = 27  -> keep valid10 high and set exc10 by `exc`
    # - dest IDs in this inference flow are < 1024, so [10] stays 0 and delay overlap is safe
    # - keep delay=0 in inference path
    if delay == 0:
        w7 = ((weight & 0xFE) >> 1) & 0x7F  # overlap-safe magnitude
        data = 0
        data |= (dest & 0x7FF)              # supports both 10b/11b IDs for this range
        data |= (w7 << 19)                  # shared weight region
        if exc:
            data |= (1 << 26)               # exc10 / contributes to weight11 MSB
        data |= (1 << 27)                   # valid10 + exc11
        data |= (1 << 28)                   # valid11
    else:
        n = ROUTER_NEURON_ID_W
        data = ((1) << (n + 17)) | \
               ((1 if exc else 0) << (n + 16)) | \
               ((weight & 0xFF) << (n + 8)) | \
               ((delay & 0xFF) << n) | \
               (dest & ((1 << n) - 1))

    return addr, data


def encode_conn_count(neuron: int, count: int) -> tuple:
    """Encode a conn_count write for source neuron."""
    addr = (0x01 << 24) | (neuron & 0x00_FF_FF)
    return addr, count & 0xFF


def reset_system(hls: MMIO, cfg: MMIO,
                 assert_hls_reset: bool = False,
                 poll_sleep_s: float = 0.001,
                 post_sleep_s: float = 0.002) -> float:
    """
    Per-image maintenance between inferences.

    Default behavior uses CTRL_CLEAR only (no HLS reset assertion) to avoid
    runtime reset-hold corner cases on ap_none `snn_reset` in some bitstreams.
    Set `assert_hls_reset=True` only for targeted debug.

    Router connectivity BRAM (conn_memory/conn_count) is preserved because this
    reset path only clears runtime state/counters.
    """
    t_reset_start = time.perf_counter()

    def run_hls_oneshot(ctrl_word: int, timeout_s: float = 0.05) -> None:
        hls.write(HLS_AP_CTRL, 0x00)
        hls.write(HLS_MODE_REG, 0)
        hls.write(HLS_TIME_STEPS, 1)
        hls.write(HLS_CTRL_REG, ctrl_word)
        hls.write(HLS_AP_CTRL, 0x01)

        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            ap = hls.read(HLS_AP_CTRL)
            if ap & 0x06:              # ap_done or ap_idle
                break
            time.sleep(poll_sleep_s)

        hls.write(HLS_AP_CTRL, 0x00)

    if assert_hls_reset:
        # 1) Assert reset/clear pulse into RTL
        run_hls_oneshot(CTRL_RESET | CTRL_CLEAR)
        # 2) Explicitly deassert reset on ap_none outputs
        run_hls_oneshot(0x00)
    else:
        # Counter/state maintenance without toggling HLS reset output.
        run_hls_oneshot(CTRL_CLEAR)
        run_hls_oneshot(0x00)

    hls.write(HLS_CTRL_REG, 0x00)
    time.sleep(post_sleep_s)
    return time.perf_counter() - t_reset_start


def program_router_for_inference(cfg: MMIO, n_neurons: int,
                                 source_offset: int = SOURCE_OFFSET) -> None:
    """
    Program router for external-input TTFS inference:
      - Source (source_offset + j) → destination j, weight=127, exc, count=1
      - LIF output neurons j=0..n_neurons-1: conn_count=0  (blocks recurrent loop)

    Using source IDs in the SOURCE_OFFSET range keeps LIF output neuron IDs
    (0..n_neurons-1) from triggering connections, breaking the feedback path.
    """
    cfg.write(CFG_CONFIG_CTRL, 0)   # target = router

    # Step 1: zero conn_count for actual LIF neuron IDs 0..n_neurons-1
    # (prevents any residual identity connections from previous runs)
    for j in range(n_neurons):
        cca, ccd = encode_conn_count(j, 0)
        cfg.write(CFG_CONFIG_ADDR, cca)
        cfg.write(CFG_CONFIG_WDATA, ccd)
        time.sleep(0.000_05)

    # Step 2: program connections source_offset+j → j
    for j in range(n_neurons):
        src = source_offset + j
        ca, cd = encode_conn(src, 0, j, 127, exc=True)
        cfg.write(CFG_CONFIG_ADDR, ca)
        cfg.write(CFG_CONFIG_WDATA, cd)
        time.sleep(0.000_05)
        cca, ccd = encode_conn_count(src, 1)
        cfg.write(CFG_CONFIG_ADDR, cca)
        cfg.write(CFG_CONFIG_WDATA, ccd)
        time.sleep(0.000_05)


def configure_neurons(cfg: MMIO, threshold: int,
                      leak: int = 0, refrac: int = 0) -> None:
    cfg.write(CFG_THRESHOLD,   threshold & 0xFFFF)
    cfg.write(CFG_NEURON_PARAMS, (leak & 0xFF) | ((refrac & 0xFF) << 8))


def detect_hls_spike_id_width(hwh_path: str, default: int = 11) -> int:
    """
    Parse exported .hwh to detect spike_in_neuron_id bus width.
    """
    try:
        with open(hwh_path, 'r', encoding='utf-8', errors='ignore') as f:
            txt = f.read()
        m = re.search(
            r'PORT\s+DIR="O"\s+LEFT="(\d+)"\s+NAME="spike_in_neuron_id"\s+RIGHT="(\d+)"',
            txt
        )
        if not m:
            return default
        left = int(m.group(1))
        right = int(m.group(2))
        w = abs(left - right) + 1
        if w <= 0 or w > 16:
            return default
        return w
    except Exception:
        return default


def choose_source_offset(n_neurons: int,
                         hls_spike_pkt_id_w: int,
                         preferred: int = SOURCE_OFFSET) -> int:
    """
    Choose source-offset that is valid for both HLS ID width and router ID width.
    """
    id_space = 1 << min(hls_spike_pkt_id_w, ROUTER_NEURON_ID_W)
    max_start = id_space - n_neurons
    if max_start <= 0:
        return 0
    # Prefer non-overlap with neuron output IDs [0, n_neurons-1].
    if max_start > n_neurons:
        # Enforce source_offset >= n_neurons when possible so routed input IDs
        # do not overlap with neuron output IDs (avoids recurrent feedback).
        return min(max(preferred, n_neurons), max_start)
    # If non-overlap is impossible (very small ID space), best-effort fallback.
    return min(preferred, max_start)


def router_readback(cfg: MMIO, addr: int) -> int:
    """
    Read one router register/data word through snn_config_regs CONFIG_RDATA path.
    """
    cfg.write(CFG_CONFIG_CTRL, 0)  # router target
    cfg.write(CFG_CONFIG_ADDR, addr & 0xFFFF_FFFF)
    time.sleep(0.000_05)
    return cfg.read(CFG_CONFIG_RDATA)


def read_profile_snapshot(cfg: MMIO, num_groups: int) -> dict:
    """Read the latched per-sample profile snapshot through the indirect window."""
    result = {}
    profile_info = int(cfg.read(CFG_PROFILE_INFO))
    profile_count = profile_info & 0xFFFF
    profile_version = (profile_info >> 24) & 0xFF

    if profile_version >= 10:
        for index, name in enumerate(PROFILE_BASIC_NAMES[:profile_count]):
            cfg.write(CFG_PROFILE_INDEX, index)
            result[name] = int(cfg.read(CFG_PROFILE_DATA))
        if 'drop_spike_count' in result:
            result['out_fifo_overflow_drop_count'] = result['drop_spike_count']
        return result

    for index, name in enumerate(PROFILE_ROUTER_NAMES):
        if index >= profile_count:
            return result
        cfg.write(CFG_PROFILE_INDEX, index)
        result[name] = int(cfg.read(CFG_PROFILE_DATA))
    router_group_base = 11
    router_class_score_base = 11 + 2 * num_groups
    router_class_event_base = router_class_score_base + 10
    router_class_profile_count = router_class_event_base + 10
    core_names = [
        'intra_route_start_count',
        'intra_weight_lookup_count',
        'intra_weight_nonzero_count',
        'intra_fifo_push_count',
        'intra_fifo_blocked_event_count',
        'out_fifo_overflow_drop_count',
    ]

    legacy_core_base = router_class_score_base
    new_core_base = router_class_profile_count
    new_classifier_base = new_core_base + num_groups * len(core_names)
    has_extended_router_profile = profile_version >= 2
    has_router_class_profile = (
        profile_version >= 2 and profile_count >= router_class_profile_count
    )
    core_base = new_core_base if has_extended_router_profile else legacy_core_base

    if has_router_class_profile:
        for cls in range(10):
            cfg.write(CFG_PROFILE_INDEX, router_class_score_base + cls)
            result[f'class_{cls}_score'] = int(cfg.read(CFG_PROFILE_DATA))
        for cls in range(10):
            cfg.write(CFG_PROFILE_INDEX, router_class_event_base + cls)
            result[f'class_{cls}_event_count'] = int(cfg.read(CFG_PROFILE_DATA))

    for group in range(num_groups):
        cfg.write(CFG_PROFILE_INDEX, router_group_base + group)
        result[f'group_{group}_in_event_count'] = int(cfg.read(CFG_PROFILE_DATA))
        cfg.write(CFG_PROFILE_INDEX, router_group_base + num_groups + group)
        result[f'group_{group}_stall_cycles'] = int(cfg.read(CFG_PROFILE_DATA))
        for metric, name in enumerate(core_names):
            cfg.write(CFG_PROFILE_INDEX, core_base + group * len(core_names) + metric)
            result[f'group_{group}_{name}'] = int(cfg.read(CFG_PROFILE_DATA))

    classifier_base = core_base + num_groups * len(core_names)
    if profile_count >= classifier_base + 3:
        cfg.write(CFG_PROFILE_INDEX, classifier_base)
        result['first_classifier_spike_valid'] = int(cfg.read(CFG_PROFILE_DATA)) & 0x1
        cfg.write(CFG_PROFILE_INDEX, classifier_base + 1)
        result['first_classifier_spike_id'] = int(cfg.read(CFG_PROFILE_DATA))
        cfg.write(CFG_PROFILE_INDEX, classifier_base + 2)
        result['first_classifier_spike_cycle'] = int(cfg.read(CFG_PROFILE_DATA))
    class_count_base = classifier_base + 3
    if profile_count >= class_count_base + 10:
        for cls in range(10):
            cfg.write(CFG_PROFILE_INDEX, class_count_base + cls)
            result[f'class_{cls}_spike_count'] = int(cfg.read(CFG_PROFILE_DATA))
    class_score_base = class_count_base + 10
    has_top_class_profile = (
        (not has_router_class_profile) and
        profile_version >= 3 and
        profile_count >= class_score_base + 20
    )
    if has_top_class_profile or ((not has_router_class_profile) and profile_count >= class_score_base + 10):
        for cls in range(10):
            cfg.write(CFG_PROFILE_INDEX, class_score_base + cls)
            result[f'class_{cls}_score'] = int(cfg.read(CFG_PROFILE_DATA))
    class_event_base = class_score_base + 10
    if has_top_class_profile or ((not has_router_class_profile) and profile_count >= class_event_base + 10):
        for cls in range(10):
            cfg.write(CFG_PROFILE_INDEX, class_event_base + cls)
            result[f'class_{cls}_event_count'] = int(cfg.read(CFG_PROFILE_DATA))
    input_source_group_base = class_event_base + 10
    if profile_version >= 5 and profile_count >= input_source_group_base + num_groups:
        for group in range(num_groups):
            cfg.write(CFG_PROFILE_INDEX, input_source_group_base + group)
            result[f'input_source_group_{group}_spike_count'] = int(cfg.read(CFG_PROFILE_DATA))
    input_source_bitmap_base = input_source_group_base + num_groups
    input_source_neurons = 784
    input_source_bitmap_words = (input_source_neurons + 31) // 32
    if profile_version >= 7 and profile_count >= input_source_bitmap_base + input_source_bitmap_words:
        bitmap_words = []
        active_pixels = []
        for word_idx in range(input_source_bitmap_words):
            cfg.write(CFG_PROFILE_INDEX, input_source_bitmap_base + word_idx)
            word = int(cfg.read(CFG_PROFILE_DATA))
            bitmap_words.append(word)
            for bit in range(32):
                pixel_idx = word_idx * 32 + bit
                if pixel_idx < input_source_neurons and ((word >> bit) & 0x1):
                    active_pixels.append(pixel_idx)
        result['input_source_bitmap_words'] = bitmap_words
        result['input_source_active_pixels'] = active_pixels
        result['input_source_active_pixel_count'] = len(active_pixels)
    input_source_ct_bitmap_base = input_source_bitmap_base + input_source_bitmap_words
    if profile_version >= 8 and profile_count >= input_source_ct_bitmap_base + input_source_bitmap_words:
        bitmap_words = []
        active_pixels = []
        for word_idx in range(input_source_bitmap_words):
            cfg.write(CFG_PROFILE_INDEX, input_source_ct_bitmap_base + word_idx)
            word = int(cfg.read(CFG_PROFILE_DATA))
            bitmap_words.append(word)
            for bit in range(32):
                pixel_idx = word_idx * 32 + bit
                if pixel_idx < input_source_neurons and ((word >> bit) & 0x1):
                    active_pixels.append(pixel_idx)
        result['input_source_ct_bitmap_words'] = bitmap_words
        result['input_source_ct_active_pixels'] = active_pixels
        result['input_source_ct_active_pixel_count'] = len(active_pixels)
    return result


def verify_router_for_inference(cfg: MMIO, n_neurons: int,
                                source_offset: int = SOURCE_OFFSET) -> tuple:
    """
    Spot-check router programming after `program_router_for_inference`.

    Returns:
      (ok, checks)
      checks: list[(j, src, conn_count, valid, dest)]
    """
    if n_neurons <= 0:
        return True, []

    idxs = sorted(set([0, n_neurons // 2, n_neurons - 1]))
    checks = []
    ok = True
    nid_mask10 = (1 << 10) - 1
    nid_mask11 = (1 << 11) - 1

    for j in idxs:
        src = source_offset + j
        cnt_addr = (0x01 << 24) | (src & 0x00_FFFF)
        conn_addr = (0x00 << 24) | (((src * MAX_FANOUT) + 0) & 0x00_FFFF)

        conn_count = router_readback(cfg, cnt_addr) & 0xFF
        conn_word = router_readback(cfg, conn_addr)

        valid10 = (conn_word >> 27) & 0x1
        valid11 = (conn_word >> 28) & 0x1
        dest10 = conn_word & nid_mask10
        dest11 = conn_word & nid_mask11

        ok10 = (valid10 == 1) and (dest10 == (j & nid_mask10))
        ok11 = (valid11 == 1) and (dest11 == (j & nid_mask11))

        checks.append((j, src, conn_count, valid10, valid11, dest10, dest11, conn_word))
        if conn_count != 1 or not (ok10 or ok11):
            ok = False

    return ok, checks


# =====================================================================
# Shared TTFS Ordering
# =====================================================================

def ttfs_order_from_potential(potential: np.ndarray) -> np.ndarray:
    """
    Return TTFS ordering used for both input spike generation and SW reference.

    Stable descending sort makes equal-potential ties deterministic and keeps
    strict identity checks reproducible across runs.
    """
    return np.argsort(-potential, kind='stable')


# =====================================================================
# Shared Potential Computation
# =====================================================================

def compute_positive_potential(image: np.ndarray,
                               q_weights: np.ndarray,
                               pixel_th: float = 0.3) -> np.ndarray:
    """
    Compute per-neuron positive-only potential for the current image.

    This is the dominant host-side cost in both `sw_reference_10class()` and
    `build_spike_words()`. Keep it shared so the inference loop does the work
    exactly once per image.
    """
    active_mask = image.reshape(-1) > pixel_th
    if not np.any(active_mask):
        return np.zeros(q_weights.shape[0], dtype=np.int32)

    active_cols = q_weights[:, active_mask]
    return np.maximum(active_cols, 0).sum(axis=1, dtype=np.int32)


# =====================================================================
# Spike Word Preparation
# =====================================================================

def build_spike_words(image: np.ndarray,
                      q_weights: np.ndarray,
                      pixel_th: float = 0.3,
                      source_offset: int = SOURCE_OFFSET,
                      hls_spike_pkt_id_w: int = HLS_SPIKE_PKT_ID_W,
                      potential: np.ndarray | None = None) -> np.ndarray:
    """
    Convert one MNIST image + int8 weight matrix to AER spike words (TTFS encoding).

    Hardware routing:
      source (source_offset + j) → LIF neuron j  (router weight=127, exc=1)
      LIF neuron j (0..n-1) has conn_count=0       (no recurrent feedback)

    TTFS ordering: compute potential[j] = sum(W[j,i] for active i with W[j,i]>0),
    then sort neurons by potential DESCENDING.  The first spike word sent fires the
    neuron with the highest potential → first output spike = best predicted class.

    One spike word per unique neuron j (the single spike fires it immediately since
    router_weight=127 > threshold=120).  Neurons with zero potential are omitted.

    word = ((source_offset + j) & hls_spike_id_mask) | (0x7F << hls_spike_wgt_shift)
           (weight field unused by router)
    """
    n_out  = q_weights.shape[0]
    hls_spike_id_mask = (1 << hls_spike_pkt_id_w) - 1
    hls_spike_wgt_shift = hls_spike_pkt_id_w

    if potential is None:
        potential = compute_positive_potential(image, q_weights, pixel_th=pixel_th)

    # Sort neurons by potential DESCENDING (highest fires first → TTFS classification)
    # with the exact same tie-break rule as SW reference.
    order = ttfs_order_from_potential(potential)
    pos_order = order[potential[order] > 0]
    if pos_order.size == 0:
        return np.zeros(1, dtype=np.uint32)
    src_ids = (source_offset + pos_order) & hls_spike_id_mask
    return src_ids.astype(np.uint32) | np.uint32(0x7F << hls_spike_wgt_shift)


# =====================================================================
# HLS Warmup
# =====================================================================

def warmup_hls(hls: MMIO,
               timeout_s: float = 0.25,
               poll_sleep_s: float = 0.005,
               post_sleep_s: float = 0.010) -> None:
    """
    Run HLS once without DMA before the first sample.

    This remains useful for both the legacy learning HLS and the lightweight
    inference/profile HLS. The lightweight variant has no weight-memory
    initialization, so the call is only a control-path readiness check.
    """
    hls.write(HLS_CTRL_REG, CTRL_ENABLE)
    hls.write(HLS_AP_CTRL, 0x01)   # ap_start (no auto_restart)
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        ap = hls.read(HLS_AP_CTRL)
        if ap & 0x06:              # ap_done (bit1) or ap_idle (bit2)
            break
        time.sleep(poll_sleep_s)
    hls.write(HLS_AP_CTRL, 0x00)
    hls.write(HLS_CTRL_REG, 0x00)
    time.sleep(post_sleep_s)


# =====================================================================
# DMA Inference with S2MM DMA Fix
# =====================================================================


def run_inference(hls: MMIO, cfg: MMIO, dma: MMIO,
                  spike_words: np.ndarray,
                  n_neurons: int,
                  hw_threshold: int,
                  hls_spike_pkt_id_w: int = HLS_SPIKE_PKT_ID_W,
                  capture_all_spikes: bool = False,
                  buf_in:  MMIO = None,
                  buf_out: MMIO = None,
                  ctr_router_base: int = 0,
                  ctr_neuron_base: int = 0,
                  first_spike_timeout_s: float = 0.050,
                  further_spike_timeout_s: float = 0.005,
                  mm2s_deadline_s: float = 1.5,
                  mm2s_tail_timeout_s: float = 0.300,
                  mm2s_tail_poll_s: float = 0.005,
                  settle_cap_s: float = 0.010,
                  stop_sleep_s: float = 0.003,
                  dma_reset_sleep_s: float = 0.002,
                  poll_sleep_first_s: float = 0.0005,
                  zero_retry_sleep_s: float = 0.00002,
                  settle_poll_sleep_s: float = 0.0002,
                  settle_stable_cycles_req: int = 5,
                  profile_enabled: bool = False,
                  profile_num_groups: int = 0,
                  profile_drain_timeout_s: float = 0.100,
                  wait_hls_input_count: bool = False,
                  hls_input_timeout_s: float = 0.300) -> dict:
    """
    Inject spike_words via MM2S DMA and collect output from S2MM.

    Key timing fix:
      • S2MM drain loop runs CONCURRENTLY with MM2S (not after it).
        Output spikes can appear as soon as the first input spike clears
        the router + LIF pipeline (~10 µs RTL), which is while MM2S is
        still sending the remaining input spike words.
      • By default, captures only the first HW spike (deterministic TTFS parity).
        Set capture_all_spikes=True for legacy multi-spike debug behavior.
      • First-spike timeout and settle windows are configurable for benchmark mode.
      • Counter deltas are computed against provided baselines so callers
        get per-image counts even though RTL counters are cumulative.
    """
    n_words  = len(spike_words)
    nbytes   = n_words * 4
    hls_spike_id_mask = (1 << hls_spike_pkt_id_w) - 1
    hls_spike_wgt_shift = hls_spike_pkt_id_w
    dma_buf_in_addr = int(getattr(buf_in, "physical_address", DMA_BUF_IN))
    dma_buf_out_addr = int(getattr(buf_out, "physical_address", DMA_BUF_OUT))
    t_total0 = time.perf_counter()
    sleep_s_total = 0.0
    hls_spike_base = int(hls.read(HLS_SPIKE_COUNT))

    def mm2s_is_done(sr: int) -> bool:
        # MM2S completion is reliably indicated by IDLE=1 (bit1).
        # Requiring IOC+IDLE can add unnecessary tail wait depending on IRQ state.
        return bool(sr & 0x0002)

    def tracked_sleep(sec: float) -> None:
        nonlocal sleep_s_total
        if sec > 0:
            time.sleep(sec)
            sleep_s_total += sec

    # CRITICAL: with current ap_none export behavior, spike_in_* is committed at
    # TIME_LOOP exit. Keep time_steps=1 and rely on auto_restart so one AXIS word
    # is consumed per invocation and exported to RTL each iteration.
    time_steps_hw = 1

    # --- Write input spikes + zero output buffer via mmap (cache-coherent) ---
    # Using MMIO (mmap + O_SYNC) guarantees the ARM CPU's writes reach DDR
    # before MM2S DMA reads them, and that DMA writes to DMA_BUF_OUT are
    # visible to the CPU when we read back via buf_out.read_bytes().
    buf_in.write_bytes(0, spike_words.tobytes())
    buf_out.write_bytes(0, b'\x00' * (n_neurons * 4 + 64))

    # --- Hard-reset both DMA channels ---
    t_dma_reset0 = time.perf_counter()
    dma.write(DMA_MM2S_DMACR, 0x04)   # MM2S reset
    tracked_sleep(dma_reset_sleep_s)
    dma.write(DMA_S2MM_DMACR, 0x04)   # S2MM reset
    tracked_sleep(dma_reset_sleep_s)
    t_dma_reset1 = time.perf_counter()

    # ── STEP 1: Configure HLS ──────────────────────────────────────────
    hls_ctrl_word = CTRL_ENABLE | (0 if capture_all_spikes else CTRL_FIRST_SPIKE_ONLY)
    hls.write(HLS_CTRL_REG,   hls_ctrl_word)
    hls.write(HLS_MODE_REG,   0)                # inference mode
    hls.write(HLS_TIME_STEPS, time_steps_hw)
    hls.write(HLS_CONFIG_REG, hw_threshold & 0xFFFF)

    if profile_enabled:
        cfg.write(CFG_PROFILE_CTRL, PROFILE_START | ((n_words & 0xFFFF) << 16))

    # ── STEP 2: Arm first S2MM (4 bytes = 1 spike word, TLAST per spike)
    dest_ptr = dma_buf_out_addr
    max_output_spikes = (n_neurons + 16) if capture_all_spikes else 1

    def arm_s2mm(dest):
        """
        Arm S2MM for exactly one 4-byte spike word.

        bit12 (IOC_Irq) in DMASR is STICKY (write-1-to-clear).  If we do not
        clear it before each arm, the next poll immediately sees bit12=1 and
        falsely concludes a new spike arrived, reading stale DDR zeros.
        Fix: W1C-clear bit12 FIRST, THEN set up the new transfer.
        """
        dma.write(DMA_S2MM_DMASR, 0x1000)   # W1C clear IOC_Irq BEFORE re-arm
        dma.write(DMA_S2MM_DMACR, 0x01)     # run
        dma.write(DMA_S2MM_DA,    dest)
        dma.write(DMA_S2MM_LENGTH, 4)        # write LENGTH last → triggers transfer

    arm_s2mm(dest_ptr)

    # ── STEP 3: Start MM2S first ──────────────────────────────────────
    dma.write(DMA_MM2S_DMACR, 0x01)
    dma.write(DMA_MM2S_SA,     dma_buf_in_addr)
    dma.write(DMA_MM2S_LENGTH, nbytes)

    # ── STEP 4: Start HLS after MM2S is armed ─────────────────────────
    # Use auto-restart so the kernel stays alive while stream traffic arrives.
    hls.write(HLS_AP_CTRL, 0x81)

    # ── STEP 5+6: Drain S2MM concurrently with MM2S ───────────────────
    # The LIF pipeline latency (router BRAM + LIF BRAM) is ~10-30 µs at
    # 100 MHz, so the first output spike appears while MM2S is still
    # sending input words.  We must poll S2MM NOW, not after MM2S ends.
    #
    # Timeouts:
    #   first spike  : 50 ms  (Python + AXI-MM overhead dominates)
    #   further spikes: 5 ms  (tight polling while keeping margin)
    output_spikes: list = []
    s2mm_sr       = 0
    mm2s_done     = False
    next_dest     = dest_ptr
    mm2s_deadline = time.monotonic() + mm2s_deadline_s
    s2mm_spurious_ioc = 0

    first_spike_timeout  = first_spike_timeout_s
    further_spike_timeout= further_spike_timeout_s
    this_timeout = first_spike_timeout

    t_stream0 = time.perf_counter()
    spike_n = 0
    while spike_n < max_output_spikes:
        deadline = time.monotonic() + this_timeout
        got_completion = False

        while time.monotonic() < deadline:
            sr = dma.read(DMA_S2MM_DMASR)

            if sr & 0x1000:          # bit12 = IOC_Irq (W1C): transfer completed
                got_completion = True
                s2mm_sr = sr
                break
            if sr & 0x0050:          # bit6=SGErr, bit4=DMAErr, bit3=SIErr
                s2mm_sr = sr
                break                # DMA error – abort

            # Also check MM2S completion opportunistically
            if not mm2s_done:
                if mm2s_is_done(dma.read(DMA_MM2S_DMASR)):
                    mm2s_done = True

            # Tight spin for first spike (sleep only on very first wait)
            if this_timeout > 0.010 and poll_sleep_first_s > 0:
                tracked_sleep(poll_sleep_first_s)

        if not got_completion:
            # Final check: did MM2S at least finish?
            if not mm2s_done and time.monotonic() < mm2s_deadline:
                if mm2s_is_done(dma.read(DMA_MM2S_DMASR)):
                    mm2s_done = True
            break                    # No more output spikes

        # Read captured spike word from DDR via mmap (cache-coherent)
        # os.pread reads cached data; MMIO.read_bytes uses the O_SYNC mmap
        # which creates an uncached mapping on Zynq → sees DMA-written data.
        raw4 = buf_out.read_bytes(next_dest - dma_buf_out_addr, 4)
        w    = struct.unpack('<I', raw4)[0]

        # Re-read once to absorb rare read-after-write visibility lag.
        # In strict first-spike mode, a zero word can be a valid spike
        # (neuron_id=0, weight=0), so do not discard it unconditionally.
        if w == 0:
            tracked_sleep(zero_retry_sleep_s)
            raw4_retry = buf_out.read_bytes(next_dest - dma_buf_out_addr, 4)
            w_retry = struct.unpack('<I', raw4_retry)[0]
            if w_retry != 0:
                w = w_retry
            elif capture_all_spikes:
                # Legacy debug mode can ignore empty completions and keep scanning.
                s2mm_spurious_ioc += 1
                arm_s2mm(next_dest)
                this_timeout = first_spike_timeout if len(output_spikes) == 0 else further_spike_timeout
                continue

        nid = w & hls_spike_id_mask
        wt  = (w >> hls_spike_wgt_shift) & 0xFF
        output_spikes.append({'neuron_id': int(nid), 'weight': int(wt)})
        spike_n += 1
        this_timeout = further_spike_timeout

        next_dest += 4
        if next_dest - dma_buf_out_addr >= max_output_spikes * 4:
            break
        arm_s2mm(next_dest)   # re-arm immediately for next spike
    t_stream1 = time.perf_counter()

    # Confirm MM2S done if not already seen
    t_mm2s_tail0 = time.perf_counter()
    if not mm2s_done:
        tail_poll_s = max(0.00001, float(mm2s_tail_poll_s))
        tail_iters = max(1, int(mm2s_tail_timeout_s / tail_poll_s))
        for _ in range(tail_iters):
            tracked_sleep(tail_poll_s)
            if mm2s_is_done(dma.read(DMA_MM2S_DMASR)):
                mm2s_done = True
                break
    t_mm2s_tail1 = time.perf_counter()

    # MM2S IDLE only proves the DMA has pushed words into the AXIS path.
    # The HLS bridge may still be draining that stream into the RTL router.
    t_hls_input0 = time.perf_counter()
    hls_input_done = not wait_hls_input_count
    hls_spike_target = hls_spike_base + n_words
    if wait_hls_input_count:
        hls_input_deadline = time.monotonic() + max(0.0, hls_input_timeout_s)
        while time.monotonic() < hls_input_deadline:
            if int(hls.read(HLS_SPIKE_COUNT)) >= hls_spike_target:
                hls_input_done = True
                break
            tracked_sleep(settle_poll_sleep_s)
    t_hls_input1 = time.perf_counter()

    # Give router/neuron pipeline time to drain after MM2S completion.
    # Without this settle window, a few tail spikes can be dropped when HLS is
    # stopped immediately, causing small neuron_spike under-counts (e.g. 149/150).
    t_settle0 = time.perf_counter()
    if settle_cap_s > 0:
        settle_deadline = time.monotonic() + settle_cap_s
        stable_cycles = 0
        stable_cycles_req = max(1, int(settle_stable_cycles_req))
        last_router = cfg.read(CFG_ROUTER_SPKS)
        last_neuron = cfg.read(CFG_NEURON_SPKS)
        while time.monotonic() < settle_deadline:
            tracked_sleep(settle_poll_sleep_s)
            cur_router = cfg.read(CFG_ROUTER_SPKS)
            cur_neuron = cfg.read(CFG_NEURON_SPKS)
            if cur_router == last_router and cur_neuron == last_neuron:
                stable_cycles += 1
                if stable_cycles >= stable_cycles_req:
                    break
            else:
                stable_cycles = 0
                last_router = cur_router
                last_neuron = cur_neuron
    t_settle1 = time.perf_counter()

    # A profile snapshot is valid only after all input and RTL work has drained.
    profile_incomplete = False
    profile_status = 0
    profile_snapshot = {}
    if profile_enabled:
        drain_deadline = time.monotonic() + max(0.0, profile_drain_timeout_s)
        while time.monotonic() < drain_deadline:
            mm2s_done = mm2s_done or mm2s_is_done(dma.read(DMA_MM2S_DMASR))
            profile_status = cfg.read(CFG_STATUS)
            router_idle = not bool(profile_status & STATUS_ROUTER_BUSY)
            groups_idle = not bool(profile_status & STATUS_GROUP_BUSY)
            snn_ready = bool(profile_status & STATUS_SNN_READY)
            if mm2s_done and hls_input_done and router_idle and groups_idle and snn_ready:
                break
            tracked_sleep(settle_poll_sleep_s)
        else:
            profile_incomplete = True

        cfg.write(CFG_PROFILE_CTRL, PROFILE_STOP)
        profile_snapshot = read_profile_snapshot(cfg, profile_num_groups)

    # ── STEP 7: Stop HLS + DMA ────────────────────────────────────────
    t_stop0 = time.perf_counter()
    hls.write(HLS_CTRL_REG, 0x00)
    tracked_sleep(settle_poll_sleep_s)
    hls.write(HLS_AP_CTRL, 0x00)
    dma.write(DMA_S2MM_DMACR, 0x00)
    tracked_sleep(stop_sleep_s)
    t_stop1 = time.perf_counter()

    # ── STEP 8: Read HW counters (delta from baselines) ───────────────
    t_readback0 = time.perf_counter()
    router_abs = cfg.read(CFG_ROUTER_SPKS)
    neuron_abs = cfg.read(CFG_NEURON_SPKS)
    status     = cfg.read(CFG_STATUS)
    hls_status = hls.read(HLS_STATUS_REG)
    hls_spike_count = hls.read(HLS_SPIKE_COUNT)
    pl_latency_cycles = cfg.read(CFG_THROUGHPUT)
    pl_service_cycles_raw = cfg.read(CFG_SERVICE_CYCLES)
    # Old bitstreams may return default decode (0xDEADBEEF) for unknown address.
    pl_service_cycles = None if int(pl_service_cycles_raw) == 0xDEADBEEF else int(pl_service_cycles_raw)
    mm2s_sr_end = dma.read(DMA_MM2S_DMASR)
    s2mm_sr_end = dma.read(DMA_S2MM_DMASR)
    t_readback1 = time.perf_counter()
    t_total1 = time.perf_counter()


    return {
        'input_words':    int(n_words),
        'router_spikes':  router_abs - ctr_router_base,
        'neuron_spikes':  neuron_abs - ctr_neuron_base,
        'router_abs':     router_abs,
        'neuron_abs':     neuron_abs,
        'time_steps_hw':  int(time_steps_hw),
        'hls_status':     int(hls_status),
        'hls_spike_count': int(hls_spike_count),
        'pl_latency_cycles': int(pl_latency_cycles),
        'pl_service_cycles': pl_service_cycles,
        'overflow':       bool(status & 0x01),
        'output_spikes':  output_spikes,
        's2mm_sr':        int(s2mm_sr_end if s2mm_sr_end != 0 else s2mm_sr),
        's2mm_spurious_ioc': int(s2mm_spurious_ioc),
        's2mm_err':       bool((s2mm_sr_end if s2mm_sr_end != 0 else s2mm_sr) & 0x0050),
        'mm2s_sr':        int(mm2s_sr_end),
        'mm2s_done':      mm2s_done,
        'hls_input_done': bool(hls_input_done),
        'hls_spike_target': int(hls_spike_target),
        'capture_all_spikes': bool(capture_all_spikes),
        'profile_incomplete': bool(profile_incomplete),
        'profile_status': int(profile_status),
        'profile': profile_snapshot,
        'timing': {
            'run_total_ms': (t_total1 - t_total0) * 1000.0,
            'dma_reset_ms': (t_dma_reset1 - t_dma_reset0) * 1000.0,
            'stream_poll_ms': (t_stream1 - t_stream0) * 1000.0,
            'mm2s_tail_wait_ms': (t_mm2s_tail1 - t_mm2s_tail0) * 1000.0,
            'hls_input_wait_ms': (t_hls_input1 - t_hls_input0) * 1000.0,
            'settle_wait_ms': (t_settle1 - t_settle0) * 1000.0,
            'stop_wait_ms': (t_stop1 - t_stop0) * 1000.0,
            'counter_read_ms': (t_readback1 - t_readback0) * 1000.0,
            'tracked_sleep_ms_total': sleep_s_total * 1000.0,
        },
    }


# =====================================================================
# SW Reference (bit-accurate match to FaithfulOnChipTrainer in eval mode)
# =====================================================================

def sw_reference_10class(image: np.ndarray,
                          q_weights: np.ndarray,
                          threshold: int,
                          n_classes: int  = 10,
                          fps_per_class: int = 15,
                          pixel_th: float = 0.3,
                          potential: np.ndarray | None = None) -> dict:
    """
    Replicate the FPGA inference in pure Python.

    Two classification strategies are returned:

    HW semantics (TTFS first-spike, what hardware measures):
      • neuron j fires iff potential[j] = Σ_{active i, W[j,i]>0} W[j,i] > 0
        (router weight=127 ≥ hw_threshold → unconditional fire on input spike)
      • first_spike_pred = class of neuron with highest individual potential
        This mirrors what m_axis_spikes output_spikes[0] captures: the first
        neuron to fire is the one whose input spike was sent first by
        build_spike_words (spikes sorted by potential DESCENDING).

    Group-count baseline (more robust, for reference):
      • group_counts[cls] = number of neurons in class group that fired
      • count_pred = argmax(group_counts) with group_sums tiebreak
    """
    n_neurons = q_weights.shape[0]

    if potential is None:
        potential = compute_positive_potential(image, q_weights, pixel_th=pixel_th)

    fired = potential > 0

    # TTFS first-spike prediction must use the same ordering as build_spike_words.
    # argmax() can disagree in equal-potential tie cases.
    order = ttfs_order_from_potential(potential)
    best_j = -1
    for j in order:
        if potential[j] > 0:
            best_j = int(j)
            break
    if best_j < 0:
        best_j = 0
    first_spike_pred = best_j // fps_per_class

    # Group-count prediction (more robust baseline)
    if n_neurons == n_classes * fps_per_class:
        fired_2d = fired.reshape(n_classes, fps_per_class)
        potential_2d = potential.reshape(n_classes, fps_per_class)
        group_counts = fired_2d.sum(axis=1, dtype=np.int32)
        group_sums = potential_2d.sum(axis=1, dtype=np.int64)
    else:
        group_counts = np.zeros(n_classes, dtype=np.int32)
        group_sums   = np.zeros(n_classes, dtype=np.int64)
        for j in range(n_neurons):
            cls = j // fps_per_class
            if fired[j]:
                group_counts[cls] += 1
            group_sums[cls] += int(potential[j])

    count_pred = int(np.argmax(group_counts))
    if np.sum(group_counts == group_counts[count_pred]) > 1:
        count_pred = int(np.argmax(group_sums))

    # Keep return payload minimal in the hot path.
    return {
        'pred': first_spike_pred,      # primary: TTFS (matches HW)
        'pred_count': count_pred,      # secondary: group count
    }


# =====================================================================
# Classify from HW output
# =====================================================================

def classify_hw(result: dict,
                n_classes: int = 10,
                fps_per_class: int = 15,
                sw_pred: int = -1) -> tuple:
    """
    Hardware classification with explicit source tracking.

    Priority:
      1) First S2MM spike decode (true HW first-spike path)
      2) SW TTFS fallback when S2MM produced no output

    Returns:
      (hw_pred, source) where source in {"s2mm_first_spike", "sw_ttfs_fallback"}.
    """
    pred_s2mm = classify_hw_s2mm_first_spike(result, n_classes, fps_per_class)
    if pred_s2mm is not None:
        return int(pred_s2mm), "s2mm_first_spike"
    return int(sw_pred), "sw_ttfs_fallback"


def classify_hw_s2mm_first_spike(result: dict,
                                 n_classes: int = 10,
                                 fps_per_class: int = 15):
    """
    Decode class from first S2MM spike word if available.

    Returns:
      int class [0, n_classes-1] or None when S2MM produced no output.
    """
    out = result.get('output_spikes', [])
    if not out:
        return None
    nid = int(out[0].get('neuron_id', -1))
    if nid < 0:
        return None
    cls = nid // max(int(fps_per_class), 1)
    if cls < 0:
        cls = 0
    if cls >= n_classes:
        cls = n_classes - 1
    return int(cls)


def decode_hls_status(status: int) -> dict:
    return {
        'snn_ready': bool(status & (1 << 0)),
        'snn_busy': bool(status & (1 << 1)),
        'stdp_active': bool(status & (1 << 2)),
        'first_spike_only': bool(status & (1 << 3)),
        'rstdp_enable': bool(status & (1 << 4)),
        'encoder_enable': bool(status & (1 << 5)),
        'op_mode': int((status >> 6) & 0x3),
        'update_counter_8b': int((status >> 8) & 0xFF),
        'first_spike_pending': bool(status & (1 << 16)),
    }


# =====================================================================
# Deployment data preparation (host-side helper, not used on board)
# =====================================================================

def save_deployment_npz(save_path: str,
                        weights: np.ndarray,
                        thresholds: np.ndarray,
                        test_imgs: np.ndarray,
                        test_lbls: np.ndarray,
                        hw_threshold: int,
                        weight_scale: float = 127.0) -> None:
    """
    Save full deployment package to .npz (run on HOST before uploading to board).

    Parameters
    ----------
    weights     : float32 (n_neurons, 784)   FaithfulOnChipTrainer weights
    thresholds  : float32 (n_neurons,)       per-neuron thresholds
    test_imgs   : float32 (10000, 28, 28)    MNIST test images
    test_lbls   : int64   (10000,)           labels
    hw_threshold: int                        HW threshold value (from training)
    weight_scale: float                      scale before int8 clipping
    """
    q_weights = np.clip(np.round(weights * weight_scale), -127, 127).astype(np.int8)
    np.savez_compressed(
        save_path,
        q_weights   = q_weights,
        thresholds  = thresholds,
        test_imgs   = test_imgs,
        test_lbls   = test_lbls,
        hw_threshold= np.array(hw_threshold, dtype=np.int32),
        weight_scale= np.array(weight_scale, dtype=np.float32),
        n_classes   = np.array(10, dtype=np.int32),
        fps_per_class = np.array(15, dtype=np.int32),
    )
    print(f"Saved deployment package: {save_path}")
    print(f"  q_weights: {q_weights.shape}  range [{q_weights.min()}, {q_weights.max()}]")
    print(f"  hw_threshold: {hw_threshold}")


# =====================================================================
# Main
# =====================================================================

def parse_args():
    p = argparse.ArgumentParser(description="10-class MNIST inference on PYNQ FPGA")
    p.add_argument('--data',       default='/home/xilinx/snn',
                   help='Directory with .bit and .npz files')
    p.add_argument('--bitstream',  default=None,
                   help='Override bitstream path')
    p.add_argument('--weights',    default=None,
                   help='Override deployment .npz path')
    p.add_argument('--n',          type=int, default=0,
                   help='Number of images to test (0 = all 10000)')
    p.add_argument('--no-program', action='store_true',
                   help='Skip FPGA programming (bitstream already loaded)')
    p.add_argument('--output',     default=None,
                   help='JSON result file path')
    p.add_argument('--profile-output', default=None,
                   help='Write per-sample hardware profile snapshots to CSV')
    p.add_argument('--source-offset', type=int, default=-1,
                   help='Override router source offset (default: auto from .hwh)')
    p.add_argument('--packet-id-width', type=int, default=HLS_SPIKE_PKT_ID_W,
                   help='HLS AXIS packet neuron-id width (default: 13)')
    p.add_argument('--capture-all-spikes', action='store_true',
                   help='Debug mode: keep capturing all S2MM spikes (default captures first spike only)')
    p.add_argument('--assert-hls-reset', action='store_true',
                   help='Use CTRL_RESET pulse between images (debug only)')
    p.add_argument('--strict-identical', action='store_true',
                   help='Require per-image SW/HW identity (delivery + first-spike parity); non-zero exit on failure')
    p.add_argument('--check-hls-version', action='store_true',
                   help='Read HLS version_reg (0x88). Disable by default to avoid SIGBUS on legacy bitstreams.')
    p.add_argument('--pl-clock-mhz', type=float, default=PL_CLK_MHZ_DEFAULT,
                   help='PL clock in MHz for cycle->time conversion (default: 100.0)')
    p.add_argument('--pl-clock-hz', type=float, default=0.0,
                   help='PL clock in Hz (overrides --pl-clock-mhz when > 0)')
    p.add_argument('--benchmark-fast', action='store_true',
                   help='Throughput-oriented profile (shorter timeout/settle sleeps; keep PL counters unchanged)')
    p.add_argument('--no-per-image-reset', action='store_true',
                   help='Skip reset_system() between images (benchmark-only; may reduce strict reproducibility)')
    p.add_argument('--first-spike-timeout-ms', type=float, default=FIRST_SPIKE_TIMEOUT_MS_DEFAULT,
                   help='S2MM first-spike timeout in ms (default: 50.0)')
    p.add_argument('--further-spike-timeout-ms', type=float, default=FURTHER_SPIKE_TIMEOUT_MS_DEFAULT,
                   help='S2MM timeout for subsequent spikes in ms (default: 5.0)')
    p.add_argument('--mm2s-tail-timeout-ms', type=float, default=MM2S_TAIL_TIMEOUT_MS_DEFAULT,
                   help='MM2S completion wait timeout in ms after stream loop (default: 300.0)')
    p.add_argument('--settle-cap-ms', type=float, default=SETTLE_CAP_MS_DEFAULT,
                   help='Post-MM2S drain settle cap in ms (default: 10.0)')
    p.add_argument('--stop-sleep-ms', type=float, default=STOP_SLEEP_MS_DEFAULT,
                   help='Sleep after stopping HLS/DMA in ms (default: 3.0)')
    p.add_argument('--print-every', type=int, default=100,
                   help='Print progress row every N images when N>200 (default: 100)')
    return p.parse_args()


def main():
    args = parse_args()

    if args.benchmark_fast:
        if args.first_spike_timeout_ms == FIRST_SPIKE_TIMEOUT_MS_DEFAULT:
            args.first_spike_timeout_ms = FIRST_SPIKE_TIMEOUT_MS_FAST
        if args.further_spike_timeout_ms == FURTHER_SPIKE_TIMEOUT_MS_DEFAULT:
            args.further_spike_timeout_ms = FURTHER_SPIKE_TIMEOUT_MS_FAST
        if args.mm2s_tail_timeout_ms == MM2S_TAIL_TIMEOUT_MS_DEFAULT:
            args.mm2s_tail_timeout_ms = MM2S_TAIL_TIMEOUT_MS_FAST
        if args.settle_cap_ms == SETTLE_CAP_MS_DEFAULT:
            args.settle_cap_ms = SETTLE_CAP_MS_FAST
        if args.stop_sleep_ms == STOP_SLEEP_MS_DEFAULT:
            args.stop_sleep_ms = STOP_SLEEP_MS_FAST

    DATA_DIR     = args.data
    BIT_PATH     = args.bitstream or os.path.join(DATA_DIR, 'snn_integrated_v2.bit')
    DEPLOY_PATH  = args.weights   or os.path.join(DATA_DIR, 'mnist_10class_deployment.npz')
    RESULT_PATH  = args.output    or os.path.join(DATA_DIR, 'mnist_10class_results.json')
    PROFILE_PATH = args.profile_output

    SEP = '=' * 72
    print(SEP)
    print("10-Class MNIST FPGA Inference  (v2 bitstream, S2MM DMA fix)")
    print(SEP)

    # ── Load deployment data ────────────────────────────────────────────
    print(f"\nLoading: {DEPLOY_PATH}")
    if not os.path.exists(DEPLOY_PATH):
        print(f"ERROR: {DEPLOY_PATH} not found.")
        print("Generate it on the host with:")
        print("  python3 tests/prepare_10class_deployment.py")
        sys.exit(1)

    data          = np.load(DEPLOY_PATH, allow_pickle=True)
    q_weights     = data['q_weights']          # int8  (150, 784)
    thresholds    = data['thresholds']         # float (150,)
    test_imgs     = data['test_imgs']          # float (10000, 28, 28)
    test_lbls     = data['test_lbls']          # int   (10000,)
    hw_threshold  = int(data['hw_threshold'])
    n_classes     = int(data.get('n_classes',     10))
    fps_per_class = int(data.get('fps_per_class', 15))
    n_neurons     = n_classes * fps_per_class

    N = int(args.n) if args.n > 0 else len(test_imgs)
    print(f"  q_weights: {q_weights.shape}  hw_threshold: {hw_threshold}")
    print(f"  n_neurons: {n_neurons}  n_classes: {n_classes}  fps: {fps_per_class}")
    print(f"  Test images: {N}")

    # ── FPGA programming ────────────────────────────────────────────────
    if not args.no_program:
        print(f"\nProgramming FPGA: {BIT_PATH}")
        if not os.path.exists(BIT_PATH):
            print(f"ERROR: bitstream not found: {BIT_PATH}")
            sys.exit(1)
        if not program_fpga(BIT_PATH):
            print("ERROR: FPGA programming failed")
            sys.exit(1)
    else:
        print("\n(--no-program: skipping FPGA programming)")

    # ── Runtime packet format / source-offset selection ────────────────
    hwh_path = BIT_PATH.replace('.bit', '.hwh')
    hls_apnone_id_w = detect_hls_spike_id_width(hwh_path, default=11)
    hls_spike_pkt_id_w = int(args.packet_id_width)
    auto_source_offset = choose_source_offset(n_neurons, hls_spike_pkt_id_w, preferred=SOURCE_OFFSET)
    source_offset = auto_source_offset if args.source_offset < 0 else int(args.source_offset)
    pl_clock_hz = float(args.pl_clock_hz) if float(args.pl_clock_hz) > 0.0 else (float(args.pl_clock_mhz) * 1_000_000.0)
    pl_clock_mhz_runtime = pl_clock_hz / 1_000_000.0

    print(f"  HLS packet_id_width: {hls_spike_pkt_id_w}  apnone_id_width: {hls_apnone_id_w}  source_offset: {source_offset}")
    print(f"  PL clock for latency conversion: {pl_clock_hz:.0f} Hz ({pl_clock_mhz_runtime:.6f} MHz)")
    print(f"  S2MM capture mode: {'all-spikes (debug)' if args.capture_all_spikes else 'first-spike-only (strict/parity)'}")
    print(
        "  Runtime profile: "
        f"{'benchmark-fast' if args.benchmark_fast else 'strict-default'} "
        f"(first={args.first_spike_timeout_ms:.3f}ms, further={args.further_spike_timeout_ms:.3f}ms, "
        f"mm2s_tail={args.mm2s_tail_timeout_ms:.3f}ms, settle_cap={args.settle_cap_ms:.3f}ms, "
        f"stop_sleep={args.stop_sleep_ms:.3f}ms, per_image_reset={not args.no_per_image_reset})"
    )
    if source_offset + n_neurons >= (1 << min(hls_spike_pkt_id_w, ROUTER_NEURON_ID_W)):
        print("  WARNING: source_offset overlaps router ID space limit; recurrent aliasing may occur.")

    # ── MMIO open ───────────────────────────────────────────────────────
    hls = MMIO(HLS_BASE, 0x100)
    cfg = MMIO(CFG_BASE, 0x100)
    dma = MMIO(DMA_BASE, 0x100)

    ver = cfg.read(CFG_VERSION)
    print(f"  snn_config_regs version: 0x{ver:08X}  "
          f"({'OK' if ver == 0x534E4E01 else 'UNEXPECTED'})")
    profile_enabled = PROFILE_PATH is not None
    profile_num_groups = 0
    if profile_enabled:
        profile_info = cfg.read(CFG_PROFILE_INFO)
        profile_version = (profile_info >> 24) & 0xFF
        if profile_info == 0xDEADBEEF or profile_version < 1:
            print("ERROR: --profile-output requested, but this bitstream has no profile window.")
            sys.exit(1)
        profile_num_groups = (profile_info >> 16) & 0xFF
        profile_counter_count = profile_info & 0xFFFF
        print(
            f"  profile info: 0x{profile_info:08X}  "
            f"version={profile_version} groups={profile_num_groups} "
            f"counters={profile_counter_count}"
        )
    if args.check_hls_version:
        hls_ver = hls.read(HLS_VERSION_REG)
        print(f"  hls version_reg: 0x{hls_ver:08X}  "
              f"({'OK' if hls_ver == EXPECTED_HLS_VERSION else 'UNEXPECTED'})")
    else:
        print("  hls version_reg: (SKIPPED; use --check-hls-version to enable)")

    # ── Full reset + router setup ───────────────────────────────────────
    print(f"\nWarm-up HLS control path ...")
    warmup_hls(
        hls,
        poll_sleep_s=(0.001 if args.benchmark_fast else 0.005),
        post_sleep_s=(0.0 if args.benchmark_fast else 0.010),
    )
    print(f"  Done.")

    print(f"\nProgramming router ({n_neurons} source-offset connections) ...")
    reset_system(
        hls, cfg,
        assert_hls_reset=args.assert_hls_reset,
        poll_sleep_s=(0.0002 if args.benchmark_fast else 0.001),
        post_sleep_s=(0.0 if args.benchmark_fast else 0.002),
    )
    program_router_for_inference(cfg, n_neurons, source_offset=source_offset)
    configure_neurons(cfg, threshold=hw_threshold, leak=0, refrac=0)
    router_ok, router_checks = verify_router_for_inference(cfg, n_neurons, source_offset=source_offset)
    if router_ok:
        print("  Done. (router readback: OK)")
        if router_checks:
            j, src, cnt, v10, v11, d10, d11, word = router_checks[0]
            print(f"    sample j={j:3d} src={src:4d} cnt={cnt:3d} "
                  f"v10={v10} v11={v11} d10={d10:4d} d11={d11:4d} word=0x{word:08X}")
    else:
        print("  Done. (router readback: WARNING)")
        for j, src, cnt, v10, v11, d10, d11, word in router_checks:
            print(f"    j={j:3d} src={src:4d} cnt={cnt:3d} "
                  f"v10={v10} v11={v11} d10={d10:4d} d11={d11:4d} word=0x{word:08X}")
        print("  Retrying router init once (reset -> program -> verify) ...")
        reset_system(
            hls, cfg,
            assert_hls_reset=args.assert_hls_reset,
            poll_sleep_s=(0.0002 if args.benchmark_fast else 0.001),
            post_sleep_s=(0.0 if args.benchmark_fast else 0.002),
        )
        program_router_for_inference(cfg, n_neurons, source_offset=source_offset)
        configure_neurons(cfg, threshold=hw_threshold, leak=0, refrac=0)
        router_ok2, router_checks2 = verify_router_for_inference(cfg, n_neurons, source_offset=source_offset)
        if router_ok2:
            print("  Retry result: OK")
            router_ok = True
        else:
            print("  Retry result: WARNING")
            for j, src, cnt, v10, v11, d10, d11, word in router_checks2:
                print(f"    j={j:3d} src={src:4d} cnt={cnt:3d} "
                      f"v10={v10} v11={v11} d10={d10:4d} d11={d11:4d} word=0x{word:08X}")

    # ── DDR buffer MMIO objects (opened once, O_SYNC → cache-coherent) ─
    # On Zynq-7020 (ARM Cortex-A9), opening /dev/mem with O_SYNC and
    # mmapping a DDR region creates a strongly-ordered (nGnRE) mapping.
    # All reads/writes bypass the L1/L2 cache, so we always see what
    # AXI-DMA actually wrote instead of stale cache-line data.
    # Input buffer: up to n_neurons spike words (784 pixels max); output:
    # n_neurons LIF output spikes (one 32-bit word each).  Add 64-byte pad.
    n_buf_words = n_neurons + 16
    buf_in  = make_dma_buffer(DMA_BUF_IN,  n_buf_words * 4 + 64, label="MM2S")
    buf_out = make_dma_buffer(DMA_BUF_OUT, n_buf_words * 4 + 64, label="S2MM")

    # ── Inference loop ─────────────────────────────────────────────────
    print(f"\nRunning {N} inference{'s' if N != 1 else ''} ...")
    print('-' * 80)
    print(f"{'idx':>5} {'lbl':>4} {'sw_t':>5} {'sw_c':>5} {'hw':>4} {'src':>4} {'acc':>5} | "
          f"{'router':>7} {'neuron':>7} {'hls_in':>7} {'hs':>6} {'#s2':>4} {'s2':>3} {'ok':>3}")
    print('-' * 80)

    sw_correct_ttfs  = 0
    sw_correct_count = 0
    hw_correct        = 0
    hw_sw_match_ttfs  = 0
    hw_sw_match_count = 0
    hw_s2mm_correct   = 0
    hw_s2mm_samples   = 0
    hw_pred_s2mm_used = 0
    hw_pred_fallback  = 0
    s2mm_zeros    = 0   # samples where S2MM produced no output
    s2mm_spurious_total = 0
    strict_failures = []
    results_log   = []
    t_start       = time.time()
    iter_ms_samples = []
    sw_ref_ms_samples = []
    reset_ms_samples = []
    spike_pack_ms_samples = []
    hw_run_ms_samples = []
    hw_dma_reset_ms_samples = []
    hw_stream_poll_ms_samples = []
    hw_mm2s_tail_ms_samples = []
    hw_settle_ms_samples = []
    hw_stop_ms_samples = []
    hw_counter_read_ms_samples = []
    hw_tracked_sleep_ms_samples = []

    for idx in range(N):
        t_iter0 = time.perf_counter()
        img = test_imgs[idx]
        lbl = int(test_lbls[idx])

        # SW reference
        t_sw0 = time.perf_counter()
        potential = compute_positive_potential(img, q_weights)
        sw_res        = sw_reference_10class(img, q_weights, hw_threshold,
                                              n_classes, fps_per_class,
                                              potential=potential)
        t_sw1 = time.perf_counter()
        sw_pred_ttfs  = sw_res['pred']          # TTFS first-spike (matches HW)
        sw_pred_count = sw_res['pred_count']    # group-count baseline

        # FPGA inference
        if args.no_per_image_reset:
            reset_elapsed_s = 0.0
        else:
            reset_elapsed_s = reset_system(
                hls, cfg,
                assert_hls_reset=args.assert_hls_reset,
                poll_sleep_s=(0.0002 if args.benchmark_fast else 0.001),
                post_sleep_s=(0.0 if args.benchmark_fast else 0.002),
            )

        # Snapshot counters BEFORE this image (they're cumulative in RTL)
        ctr_router_pre = cfg.read(CFG_ROUTER_SPKS)
        ctr_neuron_pre = cfg.read(CFG_NEURON_SPKS)

        t_pack0 = time.perf_counter()
        spike_words = build_spike_words(
            img,
            q_weights,
            source_offset=source_offset,
            hls_spike_pkt_id_w=hls_spike_pkt_id_w,
            potential=potential,
        )
        t_pack1 = time.perf_counter()
        hw_res      = run_inference(hls, cfg, dma, spike_words, n_neurons,
                                    hw_threshold,
                                    hls_spike_pkt_id_w=hls_spike_pkt_id_w,
                                    capture_all_spikes=args.capture_all_spikes,
                                    buf_in=buf_in, buf_out=buf_out,
                                    ctr_router_base=ctr_router_pre,
                                    ctr_neuron_base=ctr_neuron_pre,
                                    first_spike_timeout_s=args.first_spike_timeout_ms / 1000.0,
                                    further_spike_timeout_s=args.further_spike_timeout_ms / 1000.0,
                                    mm2s_tail_timeout_s=args.mm2s_tail_timeout_ms / 1000.0,
                                    mm2s_tail_poll_s=(0.0002 if args.benchmark_fast else 0.005),
                                    settle_cap_s=args.settle_cap_ms / 1000.0,
                                    stop_sleep_s=args.stop_sleep_ms / 1000.0,
                                    dma_reset_sleep_s=(0.0 if args.benchmark_fast else 0.002),
                                    poll_sleep_first_s=(0.0001 if args.benchmark_fast else 0.0005),
                                    zero_retry_sleep_s=0.00002,
                                    settle_poll_sleep_s=(0.00005 if args.benchmark_fast else 0.0002),
                                    settle_stable_cycles_req=(2 if args.benchmark_fast else 5),
                                    profile_enabled=profile_enabled,
                                    profile_num_groups=profile_num_groups,
                                    profile_drain_timeout_s=max(0.100, args.mm2s_tail_timeout_ms / 1000.0))
        t_iter1 = time.perf_counter()

        hw_pred, hw_pred_source = classify_hw(hw_res, n_classes, fps_per_class, sw_pred_ttfs)
        hw_pred_s2mm = hw_pred if hw_pred_source == "s2mm_first_spike" else None
        hw_confirmed = hw_res.get('neuron_spikes', 0) > 0
        pl_latency_cycles = int(hw_res.get('pl_latency_cycles', 0))
        pl_latency_ms = (pl_latency_cycles / pl_clock_hz) * 1000.0 if pl_clock_hz > 0 else 0.0
        pl_service_cycles = hw_res.get('pl_service_cycles', None)
        pl_service_ms = ((int(pl_service_cycles) / pl_clock_hz) * 1000.0
                         if (pl_service_cycles is not None and pl_clock_hz > 0)
                         else None)
        s2mm_spurious_total += int(hw_res.get('s2mm_spurious_ioc', 0))

        s2mm_ok  = len(hw_res['output_spikes']) > 0
        if not s2mm_ok:
            s2mm_zeros += 1
            hw_pred_fallback += 1
        else:
            hw_pred_s2mm_used += 1
            hw_s2mm_samples += 1
            if hw_pred_s2mm == lbl:
                hw_s2mm_correct += 1

        match_hw_ttfs  = (hw_pred == sw_pred_ttfs)
        match_hw_count = (hw_pred == sw_pred_count)
        sw_ok_ttfs  = (sw_pred_ttfs  == lbl)
        sw_ok_count = (sw_pred_count == lbl)
        hw_ok = (hw_pred == lbl)  # HW pred = SW TTFS when hw_confirmed

        if sw_ok_ttfs:   sw_correct_ttfs  += 1
        if sw_ok_count:  sw_correct_count += 1
        if hw_ok and hw_confirmed: hw_correct += 1
        if match_hw_ttfs:  hw_sw_match_ttfs  += 1
        if match_hw_count: hw_sw_match_count += 1

        hw_timing = hw_res.get('timing', {}) if isinstance(hw_res.get('timing', {}), dict) else {}
        iter_ms_samples.append((t_iter1 - t_iter0) * 1000.0)
        sw_ref_ms_samples.append((t_sw1 - t_sw0) * 1000.0)
        reset_ms_samples.append(reset_elapsed_s * 1000.0)
        spike_pack_ms_samples.append((t_pack1 - t_pack0) * 1000.0)
        hw_run_ms_samples.append(float(hw_timing.get('run_total_ms', 0.0)))
        hw_dma_reset_ms_samples.append(float(hw_timing.get('dma_reset_ms', 0.0)))
        hw_stream_poll_ms_samples.append(float(hw_timing.get('stream_poll_ms', 0.0)))
        hw_mm2s_tail_ms_samples.append(float(hw_timing.get('mm2s_tail_wait_ms', 0.0)))
        hw_settle_ms_samples.append(float(hw_timing.get('settle_wait_ms', 0.0)))
        hw_stop_ms_samples.append(float(hw_timing.get('stop_wait_ms', 0.0)))
        hw_counter_read_ms_samples.append(float(hw_timing.get('counter_read_ms', 0.0)))
        hw_tracked_sleep_ms_samples.append(float(hw_timing.get('tracked_sleep_ms_total', 0.0)))

        if args.strict_identical:
            strict_reasons = []
            input_words = int(hw_res.get('input_words', len(spike_words)))
            if int(hw_res.get('router_spikes', 0)) != input_words:
                strict_reasons.append('router_delivery')
            if int(hw_res.get('neuron_spikes', 0)) != input_words:
                strict_reasons.append('neuron_delivery')
            if hw_pred_s2mm is None:
                strict_reasons.append('missing_s2mm_first_spike')
            elif hw_pred_s2mm != sw_pred_ttfs:
                strict_reasons.append('first_spike_parity')
            if strict_reasons:
                strict_failures.append({
                    'idx': idx,
                    'label': lbl,
                    'sw_pred_ttfs': int(sw_pred_ttfs),
                    'hw_pred_s2mm': (int(hw_pred_s2mm) if hw_pred_s2mm is not None else None),
                    'input_words': input_words,
                    'router_spikes': int(hw_res.get('router_spikes', 0)),
                    'neuron_spikes': int(hw_res.get('neuron_spikes', 0)),
                    'reasons': strict_reasons,
                })

        # Print every image (or every 100 for speed)
        if N <= 200 or idx % max(1, int(args.print_every)) == 0 or not match_hw_ttfs:
            hs = hw_res.get('hls_status', 0)
            src_tag = 'S2' if hw_pred_source == "s2mm_first_spike" else 'SW'
            print(f"{idx:>5d} {lbl:>4d} {sw_pred_ttfs:>5d} {sw_pred_count:>5d} {hw_pred:>4d} "
                  f"{src_tag:>4s} "
                  f"{'OK' if sw_ok_ttfs else 'DIFF':>5s} | "
                  f"{hw_res['router_spikes']:>7d} {hw_res['neuron_spikes']:>7d} {hw_res['hls_spike_count']:>7d} "
                  f"{hs:>6X} "
                  f"{len(hw_res['output_spikes']):>4d} "
                  f"{'Y' if s2mm_ok else 'N':>4s} "
                  f"{'Y' if hw_confirmed else 'N':>4s}")

        results_log.append({
            'idx':           idx,
            'lbl':           lbl,
            'input_words':   int(hw_res.get('input_words', len(spike_words))),
            'sw_pred_ttfs':  sw_pred_ttfs,
            'sw_pred_count': sw_pred_count,
            'hw_pred':       hw_pred,
            'hw_pred_source': hw_pred_source,
            'hw_pred_s2mm':  (int(hw_pred_s2mm) if hw_pred_s2mm is not None else None),
            'hw_confirmed':  hw_confirmed,
            'match_ttfs':    bool(match_hw_ttfs),
            'match_count':   bool(match_hw_count),
            'router':        hw_res['router_spikes'],
            'neuron':        hw_res['neuron_spikes'],
            'hls_spike_count': hw_res['hls_spike_count'],
            'pl_latency_cycles': pl_latency_cycles,
            'pl_latency_ms': pl_latency_ms,
            'pl_service_cycles': (int(pl_service_cycles) if pl_service_cycles is not None else None),
            'pl_service_ms': (float(pl_service_ms) if pl_service_ms is not None else None),
            'hls_status':    hw_res['hls_status'],
            's2mm_n':        len(hw_res['output_spikes']),
            's2mm_spurious_ioc': int(hw_res.get('s2mm_spurious_ioc', 0)),
            's2mm_sr':       hw_res['s2mm_sr'],
            'mm2s_sr':       hw_res['mm2s_sr'],
            'mm2s_ok':       hw_res['mm2s_done'],
            'iter_ms':       (t_iter1 - t_iter0) * 1000.0,
            'sw_ref_ms':     (t_sw1 - t_sw0) * 1000.0,
            'reset_ms':      reset_elapsed_s * 1000.0,
            'spike_pack_ms': (t_pack1 - t_pack0) * 1000.0,
            'hw_run_ms':     float(hw_timing.get('run_total_ms', 0.0)),
            'hw_timing':     hw_timing,
            'profile_incomplete': bool(hw_res.get('profile_incomplete', False)),
            'profile':       hw_res.get('profile', {}),
        })

    elapsed = time.time() - t_start

    # ── Results summary ────────────────────────────────────────────────
    print('=' * 80)
    n_hw_confirmed = sum(1 for r in results_log if r.get('hw_confirmed', False))
    print(f"\nResults ({N} images, {elapsed:.1f}s, {elapsed/N*1000:.1f}ms/img):")
    print(f"  HW confirmed (neuron_spikes>0): {n_hw_confirmed}/{N}")
    print(f"  HW accuracy (S2MM first-spike when captured; SW fallback otherwise):  {hw_correct}/{n_hw_confirmed}  "
          f"({hw_correct/max(n_hw_confirmed,1)*100:.2f}%)")
    print(f"  HW S2MM first-spike acc (debug path): {hw_s2mm_correct}/{hw_s2mm_samples}  "
          f"({(hw_s2mm_correct/max(hw_s2mm_samples,1))*100:.2f}%)")
    print(f"  SW TTFS acc (primary metric):  {sw_correct_ttfs}/{N}  ({sw_correct_ttfs/N*100:.2f}%)")
    print(f"  SW count-per-cls acc:          {sw_correct_count}/{N}  ({sw_correct_count/N*100:.2f}%)"
          f"   [note: TTFS >= count when model trained with TTFS objective]")
    print(f"\n  S2MM captures:  {N - s2mm_zeros}/{N} images")
    print(f"  HW pred source: s2mm={hw_pred_s2mm_used}, fallback={hw_pred_fallback}")
    total_input = sum(int(r.get('input_words', 0)) for r in results_log)
    total_router = sum(r['router'] for r in results_log)
    total_neuron = sum(r['neuron'] for r in results_log)
    total_s2mm   = sum(r['s2mm_n'] for r in results_log)
    avg_s2mm     = total_s2mm / max(N - s2mm_zeros, 1)
    router_ratio = (100.0 * total_router / max(total_input, 1))
    neuron_ratio = (100.0 * total_neuron / max(total_input, 1))
    pl_cycles_samples = [int(r.get('pl_latency_cycles', 0)) for r in results_log if int(r.get('pl_latency_cycles', 0)) > 0]
    pl_service_cycles_samples = [
        int(r.get('pl_service_cycles'))
        for r in results_log
        if (r.get('pl_service_cycles') is not None and int(r.get('pl_service_cycles')) > 0)
    ]
    if pl_cycles_samples:
        pl_cycles_mean = sum(pl_cycles_samples) / len(pl_cycles_samples)
        pl_cycles_min = min(pl_cycles_samples)
        pl_cycles_max = max(pl_cycles_samples)
        pl_ms_mean = (pl_cycles_mean / pl_clock_hz) * 1000.0
        pl_ms_min = (pl_cycles_min / pl_clock_hz) * 1000.0
        pl_ms_max = (pl_cycles_max / pl_clock_hz) * 1000.0
    else:
        pl_cycles_mean = None
        pl_cycles_min = None
        pl_cycles_max = None
        pl_ms_mean = None
        pl_ms_min = None
        pl_ms_max = None
    if pl_service_cycles_samples:
        pl_service_cycles_mean = sum(pl_service_cycles_samples) / len(pl_service_cycles_samples)
        pl_service_cycles_min = min(pl_service_cycles_samples)
        pl_service_cycles_max = max(pl_service_cycles_samples)
        pl_service_ms_mean = (pl_service_cycles_mean / pl_clock_hz) * 1000.0
        pl_service_ms_min = (pl_service_cycles_min / pl_clock_hz) * 1000.0
        pl_service_ms_max = (pl_service_cycles_max / pl_clock_hz) * 1000.0
    else:
        pl_service_cycles_mean = None
        pl_service_cycles_min = None
        pl_service_cycles_max = None
        pl_service_ms_mean = None
        pl_service_ms_min = None
        pl_service_ms_max = None

    pl_first_spike_tput_img_s = (1000.0 / pl_ms_mean) if (pl_ms_mean is not None and pl_ms_mean > 0.0) else None
    pl_service_tput_img_s = (1000.0 / pl_service_ms_mean) if (pl_service_ms_mean is not None and pl_service_ms_mean > 0.0) else None
    print(f"  Total input words:    {total_input}")
    print(f"  Total router spikes:  {total_router}  (avg {total_router/N:.0f}/img)")
    print(f"  Total neuron fires:   {total_neuron}  (avg {total_neuron/N:.0f}/img)")
    print(f"  Delivery ratio:       router {router_ratio:.2f}%  neuron {neuron_ratio:.2f}%  (vs input words)")
    print(f"  Total S2MM outputs:   {total_s2mm}  (avg {avg_s2mm:.1f}/img over captured samples)")
    print(f"  S2MM spurious IOC:    {s2mm_spurious_total}  (re-armed on zero-word completion)")
    if pl_cycles_mean is not None:
        print(f"  PL-only latency (first input->first output): "
              f"mean={pl_cycles_mean:.1f} cyc ({pl_ms_mean:.4f} ms), "
              f"min={pl_cycles_min} cyc ({pl_ms_min:.4f} ms), "
              f"max={pl_cycles_max} cyc ({pl_ms_max:.4f} ms) @ {pl_clock_mhz_runtime:.6f} MHz")
    else:
        print("  PL-only latency (first input->first output): unavailable (all-zero cycle samples)")
    if pl_service_cycles_mean is not None:
        print(f"  PL-only service time (first input->idle): "
              f"mean={pl_service_cycles_mean:.1f} cyc ({pl_service_ms_mean:.4f} ms), "
              f"min={pl_service_cycles_min} cyc ({pl_service_ms_min:.4f} ms), "
              f"max={pl_service_cycles_max} cyc ({pl_service_ms_max:.4f} ms) @ {pl_clock_mhz_runtime:.6f} MHz")
    else:
        print("  PL-only service time (first input->idle): unavailable (register missing or all-zero samples)")
    if pl_first_spike_tput_img_s is not None:
        print(f"  PL-only throughput (first-spike window): {pl_first_spike_tput_img_s:.2f} img/s")
    if pl_service_tput_img_s is not None:
        print(f"  PL-only throughput (service window): {pl_service_tput_img_s:.2f} img/s")

    def mean_or_none(samples):
        return (float(sum(samples)) / float(len(samples))) if samples else None

    iter_ms_mean = mean_or_none(iter_ms_samples)
    sw_ref_ms_mean = mean_or_none(sw_ref_ms_samples)
    reset_ms_mean = mean_or_none(reset_ms_samples)
    spike_pack_ms_mean = mean_or_none(spike_pack_ms_samples)
    hw_run_ms_mean = mean_or_none(hw_run_ms_samples)
    hw_dma_reset_ms_mean = mean_or_none(hw_dma_reset_ms_samples)
    hw_stream_poll_ms_mean = mean_or_none(hw_stream_poll_ms_samples)
    hw_mm2s_tail_ms_mean = mean_or_none(hw_mm2s_tail_ms_samples)
    hw_settle_ms_mean = mean_or_none(hw_settle_ms_samples)
    hw_stop_ms_mean = mean_or_none(hw_stop_ms_samples)
    hw_counter_read_ms_mean = mean_or_none(hw_counter_read_ms_samples)
    hw_tracked_sleep_ms_mean = mean_or_none(hw_tracked_sleep_ms_samples)

    non_pl_overhead_ms_mean = None
    non_pl_overhead_ratio = None
    if iter_ms_mean is not None and pl_service_ms_mean is not None:
        non_pl_overhead_ms_mean = iter_ms_mean - pl_service_ms_mean
        if iter_ms_mean > 0:
            non_pl_overhead_ratio = (non_pl_overhead_ms_mean / iter_ms_mean) * 100.0

    print("\n  Timing decomposition (per image, host-side measured):")
    print(f"    iter_wall_ms_mean:    {iter_ms_mean:.6f}" if iter_ms_mean is not None else "    iter_wall_ms_mean:    N/A")
    print(f"    sw_ref_ms_mean:       {sw_ref_ms_mean:.6f}" if sw_ref_ms_mean is not None else "    sw_ref_ms_mean:       N/A")
    print(f"    reset_ms_mean:        {reset_ms_mean:.6f}" if reset_ms_mean is not None else "    reset_ms_mean:        N/A")
    print(f"    spike_pack_ms_mean:   {spike_pack_ms_mean:.6f}" if spike_pack_ms_mean is not None else "    spike_pack_ms_mean:   N/A")
    print(f"    hw_run_ms_mean:       {hw_run_ms_mean:.6f}" if hw_run_ms_mean is not None else "    hw_run_ms_mean:       N/A")
    print(f"      dma_reset_ms_mean:  {hw_dma_reset_ms_mean:.6f}" if hw_dma_reset_ms_mean is not None else "      dma_reset_ms_mean:  N/A")
    print(f"      stream_poll_ms_mean:{hw_stream_poll_ms_mean:.6f}" if hw_stream_poll_ms_mean is not None else "      stream_poll_ms_mean:N/A")
    print(f"      mm2s_tail_ms_mean:  {hw_mm2s_tail_ms_mean:.6f}" if hw_mm2s_tail_ms_mean is not None else "      mm2s_tail_ms_mean:  N/A")
    print(f"      settle_ms_mean:     {hw_settle_ms_mean:.6f}" if hw_settle_ms_mean is not None else "      settle_ms_mean:     N/A")
    print(f"      stop_ms_mean:       {hw_stop_ms_mean:.6f}" if hw_stop_ms_mean is not None else "      stop_ms_mean:       N/A")
    print(f"      readback_ms_mean:   {hw_counter_read_ms_mean:.6f}" if hw_counter_read_ms_mean is not None else "      readback_ms_mean:   N/A")
    print(f"      tracked_sleep_ms:   {hw_tracked_sleep_ms_mean:.6f}" if hw_tracked_sleep_ms_mean is not None else "      tracked_sleep_ms:   N/A")
    if non_pl_overhead_ms_mean is not None and non_pl_overhead_ratio is not None:
        print(f"    non_pl_overhead_ms_mean (iter - pl_service): {non_pl_overhead_ms_mean:.6f} "
              f"({non_pl_overhead_ratio:.4f}% of iter)")

    if (total_neuron < total_input) and (not args.assert_hls_reset):
        print("  NOTE: neuron fires are slightly below input words.")
        print("  Try `--assert-hls-reset` for stricter per-image state reset.")

    # ── Warn about S2MM path ────────────────────────────────────────────
    if s2mm_zeros == N:
        print("\n  WARNING: S2MM DMA path produced NO output on any sample.")
        print("  HW accuracy above uses SW TTFS fallback (not true HW first-spike).")
        if results_log:
            st = decode_hls_status(int(results_log[-1].get('hls_status', 0)))
            print(f"  Last HLS status decode: ready={int(st['snn_ready'])} busy={int(st['snn_busy'])} "
                  f"stdp={int(st['stdp_active'])} first_only={int(st['first_spike_only'])} "
                  f"mode={st['op_mode']} enc={int(st['encoder_enable'])} "
                  f"upd8={st['update_counter_8b']} pending={int(st['first_spike_pending'])}")

    if args.strict_identical:
        n_fail = len(strict_failures)
        n_pass = N - n_fail
        verdict = "PASS" if n_fail == 0 else "FAIL"
        print(f"\n  Strict identity check: {n_pass}/{N} {verdict}")
        if n_fail > 0:
            print("  First failures:")
            for fail in strict_failures[:5]:
                print(f"    idx={fail['idx']} reasons={','.join(fail['reasons'])} "
                      f"(in={fail['input_words']} router={fail['router_spikes']} neuron={fail['neuron_spikes']} "
                      f"sw={fail['sw_pred_ttfs']} hw_s2={fail['hw_pred_s2mm']})")

    # ── Save results ────────────────────────────────────────────────────
    summary = {
        'n_images':           N,
        'sw_acc_ttfs':        sw_correct_ttfs  / N,
        'sw_acc_count':       sw_correct_count / N,
        'hw_acc_ttfs':        hw_correct / max(n_hw_confirmed, 1),
        'n_hw_confirmed':     n_hw_confirmed,
        'hw_s2mm_acc':        hw_s2mm_correct / max(hw_s2mm_samples, 1),
        'n_hw_s2mm_samples':  hw_s2mm_samples,
        'n_hw_pred_s2mm':     hw_pred_s2mm_used,
        'n_hw_pred_fallback': hw_pred_fallback,
        'strict_identical':   bool(args.strict_identical),
        'strict_failures':    strict_failures,
        'strict_pass_count':  N - len(strict_failures),
        'strict_pass_rate':   (N - len(strict_failures)) / N,
        'strict_identical_pass': bool(args.strict_identical) and (len(strict_failures) == 0),
        'benchmark_fast':     bool(args.benchmark_fast),
        'elapsed_s':          elapsed,
        'ms_per_image':       elapsed / N * 1000,
        'pl_clock_hz':        pl_clock_hz,
        'pl_clock_mhz':       pl_clock_mhz_runtime,
        'runtime_profile': {
            'first_spike_timeout_ms': float(args.first_spike_timeout_ms),
            'further_spike_timeout_ms': float(args.further_spike_timeout_ms),
            'mm2s_tail_timeout_ms': float(args.mm2s_tail_timeout_ms),
            'settle_cap_ms': float(args.settle_cap_ms),
            'stop_sleep_ms': float(args.stop_sleep_ms),
            'per_image_reset': bool(not args.no_per_image_reset),
            'assert_hls_reset': bool(args.assert_hls_reset),
        },
        'pl_latency_cycles_mean': pl_cycles_mean,
        'pl_latency_cycles_min': pl_cycles_min,
        'pl_latency_cycles_max': pl_cycles_max,
        'pl_latency_ms_mean':  pl_ms_mean,
        'pl_latency_ms_min':   pl_ms_min,
        'pl_latency_ms_max':   pl_ms_max,
        'pl_service_cycles_mean': pl_service_cycles_mean,
        'pl_service_cycles_min': pl_service_cycles_min,
        'pl_service_cycles_max': pl_service_cycles_max,
        'pl_service_ms_mean':  pl_service_ms_mean,
        'pl_service_ms_min':   pl_service_ms_min,
        'pl_service_ms_max':   pl_service_ms_max,
        'pl_throughput_img_s_first_spike': pl_first_spike_tput_img_s,
        'pl_throughput_img_s_service': pl_service_tput_img_s,
        'timing_breakdown_ms_mean': {
            'iter_wall': iter_ms_mean,
            'sw_ref': sw_ref_ms_mean,
            'reset': reset_ms_mean,
            'spike_pack': spike_pack_ms_mean,
            'hw_run': hw_run_ms_mean,
            'hw_dma_reset': hw_dma_reset_ms_mean,
            'hw_stream_poll': hw_stream_poll_ms_mean,
            'hw_mm2s_tail': hw_mm2s_tail_ms_mean,
            'hw_settle': hw_settle_ms_mean,
            'hw_stop': hw_stop_ms_mean,
            'hw_counter_read': hw_counter_read_ms_mean,
            'hw_tracked_sleep': hw_tracked_sleep_ms_mean,
            'non_pl_overhead': non_pl_overhead_ms_mean,
            'non_pl_overhead_ratio_percent': non_pl_overhead_ratio,
        },
        'hw_threshold': hw_threshold,
        'n_neurons':    n_neurons,
        'total_input_words': total_input,
        'total_router_spikes': total_router,
        'total_neuron_spikes': total_neuron,
        'total_s2mm_spurious_ioc': s2mm_spurious_total,
        'results':      results_log,
    }
    with open(RESULT_PATH, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {RESULT_PATH}")

    if PROFILE_PATH is not None:
        profile_rows = []
        for row in results_log:
            profile_row = {
                'idx': row['idx'],
                'label': row['lbl'],
                'profile_incomplete': int(row.get('profile_incomplete', False)),
            }
            profile_row.update(row.get('profile', {}))
            profile_rows.append(profile_row)
        fieldnames = list(profile_rows[0].keys()) if profile_rows else [
            'idx', 'label', 'profile_incomplete'
        ]
        with open(PROFILE_PATH, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(profile_rows)
        print(f"Saved profile CSV: {PROFILE_PATH}")

    hls.close()
    cfg.close()
    dma.close()
    buf_in.close()
    buf_out.close()

    if args.strict_identical and strict_failures:
        sys.exit(2)


if __name__ == '__main__':
    main()
