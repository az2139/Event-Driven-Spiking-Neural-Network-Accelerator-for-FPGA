# Architecture

System architecture of the Event-Driven SNN FPGA Accelerator.

## Overview

```
┌─────────────────────────────────────────────────────────┐
│                Software (Python/PyTorch)                │
│  - Model training                                       │
│  - Spike encoding                                       │
│  - Configuration                                        │
└───────────────────────┬─────────────────────────────────┘
                        │ AXI Bus
┌───────────────────────┴─────────────────────────────────┐
│                 FPGA (PYNQ-Z2)                          │
│  ┌────────────┐  ┌──────────────────┐  ┌───────────┐    │
│  │ AXI        │→ │  Event Router    │→ │ Core Group│    │
│  │ Interface  │  │  (NG, 16-port)   │  │ ×16       │    │
│  └────────────┘  └──────┬───────────┘  │ (128 LIF  │    │
│         ↓               │              │  neurons) │    │
│  ┌──────────────┐  ┌────┴────────┐     └───────────┘    │
│  │ STDP/R-STDP  │  │ Synaptic    │                      │
│  │ Learning     │  │ Connectivity│                      │
│  │ Engine (HLS) │  │ Table (BRAM)│                      │
│  └──────────────┘  └─────────────┘                      │
│                                                         │
│  Total: 2,048 neurons, ~65 BRAM36, ~10K LUT             │
└─────────────────────────────────────────────────────────┘
```

**Design Principles**:
- Hierarchical Core Group architecture (IEEE-inspired)
- **Spike-triggered processing** — neuron state updates are gated by incoming
  AER spike events; only neuron targets of an arriving spike are active per cycle
  (⚠️ not "asynchronous event-driven" — FPGA is clock-synchronous at 100 MHz)
- AC-based operations (accumulate-only, no multiply)
- Dense intra-group + sparse inter-group connectivity
- Fixed-point arithmetic (8-bit weights)

**Hardware**: Xilinx Zynq-7020 (xc7z020clg400-2) on PYNQ-Z2

**Core Group Configuration** (16 groups × 128 neurons):

| Resource | Per Group | ×16 + Router + CT | Available | Util% |
|----------|-----------|---------------------|-----------|-------|
| LUT | 557 | ~9,777 | 53,200 | 18.4% |
| FF | 317 | ~5,456 | 106,400 | 5.1% |
| BRAM36 | 3 | ~65 | 140 | 46.4% |
| DSP | 0 | 0 | 220 | 0% |

**Timing**: Target 100 MHz, Synthesis clean (0 errors, 0 critical warnings)

---

## Core Group Architecture

The core group is the fundamental processing unit, inspired by hierarchical
neuromorphic architectures described in recent IEEE literature.

### Block Diagram

```
                     ┌──────────────────────────────────────────────┐
                     │             snn_core_group_top               │
                     │                                              │
          ┌──────────┤   ┌─────────────────────────────────────┐    │
  AXI ──→ │ Config   │   │         Event Router (NG)           │    │
  Lite    │ Decoder  │   │  ┌────────────────────────────┐     │    │
          │          ├───│  │  Round-Robin Arbiter       │     │    │
          └──────────┤   │  │  (16 sources + external)   │     │    │
                     │   │  └──────┬─────────────────────┘     │    │
  HLS ◄──────────────│───│  learn_spike (observation port)     │    │
  Learning           │   │                                     │    │
                     │   └──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬────┘    │
                     │      │  │  │  │  │  │  │  │  │  │  │         │
                     │     ┌┴┐┌┴┐┌┴┐┌┴┐┌┴┐┌┴┐                       │
                     │     │0││1││2││3││4││5│ ... (16 groups)       │
                     │     │C││C││C││C││C││C│                       │
                     │     │G││G││G││G││G││G│                       │
                     │     └┬┘└┬┘└┬┘└┬┘└┬┘└┬┘                       │
                     │      │  │  │  │  │  │                        │
                     │   ┌──┴──┴──┴──┴──┴──┴───────────────────┐    │
                     │   │    Synaptic Connectivity Table      │    │
                     │   │    (32K × 17b BRAM, sparse xbar)    │    │
                     │   └─────────────────────────────────────┘    │
                     └──────────────────────────────────────────────┘
```

### Core Group (core_group.v)

Each core group contains 128 time-multiplexed LIF neurons with dense
local synaptic connectivity.

**Internal Architecture**:
```
ext_spike_in → [Input FIFO (64)] → [Processing FSM] → Neuron State BRAM
                                        ↑    ↓ (fire)
                   [intra-group recurrence] ← [Local Weight BRAM read]
                                                    ↓ (non-zero weight)
                                               [push to FIFO]
                                  + spike_flag_bitmap → output scan → Event Router
```

**Memory Resources (per group)**:
- **Neuron State BRAM**: 128 × 24b (16b membrane + 8b refractory) → 1 RAMB18
- **Weight Memory**: 128 × 128 × 5b (4b weight + 1b exc flag) → 2 RAMB36 + 1 RAMB18
- **Spike FIFO**: 64 entries → LUTRAM (16 RAMD64E)

**FSM States**:
```
IDLE → { SPIKE_RD → SPIKE_CMP → SPIKE_WR → [INTRA_READ → INTRA_ROUTE]* }
     → { LEAK_RD → LEAK_CMP → LEAK_WR (128 iterations) }
```

- Incoming spikes **preempt** leak cycles (low-latency event processing)
- Neuron firing triggers intra-group weight row scan (128 lookups)
- Non-zero weights are pushed back into the input FIFO for local propagation

**Parameters**:
| Parameter | Value | Description |
|-----------|-------|-------------|
| NEURONS_PER_GROUP | 128 | Neurons per group |
| WEIGHT_WIDTH | 4 | Synaptic weight bits |
| DATA_WIDTH | 16 | Membrane potential bits |
| REFRAC_WIDTH | 8 | Refractory counter bits |
| SPIKE_BUFFER_DEPTH | 64 | Input FIFO depth |

**File**: `hardware/hdl/rtl/core/core_group.v`

### Synaptic Connectivity Table (synaptic_connectivity_table.v)

Sparse inter-group connection storage using dual-port BRAM.

**Address Scheme** (15 bits for 16 groups):
```
addr = {src_group[3:0], src_neuron[6:0], fanout_idx[3:0]}
     = 2^15 = 32,768 entries
```

**Data Format** (17 bits):
```
[16]     valid       — entry is active
[15:12]  dst_group   — destination core group ID (4-bit)
[11:5]   dst_neuron  — destination neuron within group (7-bit)
[8:1]    weight      — 8-bit synaptic weight
[0]      exc_inh     — 1=excitatory, 0=inhibitory
```

**BRAM Usage**: 32K × 17b → 17 RAMB36E1

**Read Latency**: 2 cycles (BRAM read + unpack pipeline)

**File**: `hardware/hdl/rtl/core/synaptic_connectivity_table.v`

### Event Router (event_router_ng.v)

Central spike routing hub with round-robin arbitration.

**FSM**:
```
IDLE → ARB_SELECT → CT_LOOKUP → CT_WAIT1 → CT_WAIT2 → CT_DELIVER → CT_NEXT
                                                                          ↓
                                                                   LEARN_NOTIFY → IDLE
     → EXT_ROUTE → IDLE  (direct external spike routing)
     → WEIGHT_FWD        (learning engine weight updates)
```

**Features**:
- **Round-robin arbiter** across 16 group ports + external source
- **Fanout iteration**: Scans up to 16 CT entries per spike event
- **Learning observation port**: Forwards all spike events to HLS
- **Weight update passthrough**: Routes learning updates to groups or CT
- **Backpressure handling**: Waits when destination group FIFO is full

**Resources**: 862 LUT, 382 FF, 0 BRAM (all state in registers)

**File**: `hardware/hdl/rtl/core/event_router_ng.v`

### Top-Level Integration (snn_core_group_top.v)

Integrates all components with PS/HLS interface.

**Config Register Mapping** (from AXI-Lite):
```
cfg_router_config_addr[31:28]:
  0x0: Connectivity table write (wdata format below)
  0x1: Intra-group weight write
  0x2: Read routed_spike_count
  0x3: Read total_neuron_spikes

CT write wdata[31:0]:
  [31]    valid, [30:27] dst_group, [26:20] dst_neuron,
  [19:16] weight, [15] exc_inh, [14:11] fanout_idx,
  [10:4]  src_neuron, [3:0] src_group

Intra-group weight wdata[31:0]:
  [31:25] src_neuron, [24:18] dst_neuron, [17:14] weight,
  [13] exc, [12:9] group_id
```

**HLS Bridge**: Converts 11-bit global neuron IDs between HLS and core group
addressing ({group_id[3:0], local_id[6:0]}).

**File**: `hardware/hdl/rtl/top/snn_core_group_top.v`

---

## Bug Fixes Applied

### core_group.v — ST_SPIKE_WR Deadlock Fix
- **Issue**: When `ref_rd > 0` in ST_SPIKE_WR, no state transition was assigned,
  causing the FSM to hang indefinitely in ST_SPIKE_WR.
- **Fix**: Added `state <= ST_IDLE` in the refractory branch.

### core_group.v — FIFO Write Collision Fix
- **Issue**: External spike FIFO writes could collide with intra-group routing
  FIFO writes on the same clock cycle (both writing to `fifo_wr_ptr`).
- **Fix**: Added `intra_routing` guard — `ext_spike_ready` deasserts during
  `ST_INTRA_ROUTE` and `ST_INTRA_READ` states, preventing simultaneous writes.

### synaptic_connectivity_table.v — Read Pipeline Alignment
- **Issue**: `result_valid` was misaligned with `rd_data` (off by one cycle).
- **Fix**: Added `lookup_en_d1` pipeline stage to align valid signal with data.

---

## Verification

### Testbench Summary (55/55 PASS)

| Testbench | Tests | Status |
|-----------|-------|--------|
| Core Group (tb_core_group.v) | 15 | 15/15 PASS |
| Router + CT (tb_router_ct.v) | 24 | 24/24 PASS |
| Integration (tb_integration.v) | 16 | 16/16 PASS |

**Core Group Tests**:
1. Reset state, 2. Weight load, 3. Sub-threshold, 4. Supra-threshold,
5. Output spike detection, 6. Refractory period, 7. Accumulation (3×4>10),
8. Inhibitory, 9. Intra-group recurrence (50→51 chain), 10. Backpressure,
11. Zero weight, 12. Burst (10 spikes), 13. Exact threshold,
14. Multi-neuron diverse, 15. Leak decay

**Router+CT Tests**:
1-4. Reset & CT write/read, 5-8. CT CRUD, 9-12. Spike routing via CT,
13-14. Learning notifications, 15-16. Multi-fanout, 17-18. Round-robin,
19-20. Weight forwarding (intra/inter), 21-22. Backpressure,
23. Max fanout (16), 24. Empty CT handling

**Integration Tests**:
1-3. Reset & enable, 4-6. External spike injection, 7-8. Intra+inter combined,
9-11. Multi-group fanout, 12. Sub-threshold inter-group, 13. Learning notifications,
14-15. Counter consistency, 16. Stress test

## LIF Neuron Model

Each core group implements 128 time-multiplexed LIF neurons.

**State per Neuron** (24 bits stored in BRAM):
- `v_mem`: 16-bit unsigned membrane potential
- `refrac_counter`: 8-bit refractory counter

**Operation within Core Group FSM**:
```
// Spike arrives (ST_SPIKE_RD/CMP/WR)
if refrac_counter > 0:
    refrac_counter -= 1     // skip, neuron refractory
else:
    if exc_flag:
        v_mem += weight     // excitatory (saturate at 2^16-1)
    else:
        v_mem -= weight     // inhibitory (floor at 0)
    if v_mem >= threshold:
        spike_out = 1
        v_mem = reset_potential
        refrac_counter = refractory_period

// Leak cycle (ST_LEAK_RD/CMP/WR, 128 iterations)
leak1 = v_mem >> shift1
leak2 = v_mem >> shift2  (if enabled)
v_mem -= (leak1 + leak2)
```

**Shift-Based Leak** (no multiplier):

`tau = 1 - 2^(-shift1) - 2^(-shift2)`

| tau | shift1 | shift2 | Usage |
|-----|--------|--------|-------|
| 0.500 | 1 | 0 | Fast decay |
| 0.875 | 3 | 0 | Moderate |
| 0.906 | 4 | 5 | Typical |
| 0.953 | 5 | 6 | Slow decay |

**Parameters**:
- threshold: 16-bit (typical 100-2000)
- refractory_period: 8-bit (0-255 timesteps)
- reset_potential: 16-bit (typically 0)

## STDP Learning Engine

On-chip learning using Spike-Timing-Dependent Plasticity (HLS).

**Algorithm**: Mozafari weight-dependent STDP

$$\Delta w_{LTP} = a^+ \cdot \frac{(w_{max} - w)^{\mu}}{scale}$$

$$\Delta w_{LTD} = -a^- \cdot \frac{(w - w_{min})^{\mu}}{scale}$$

**Per-Neuron Traces** (Memory-efficient):

```cpp
// O(N+M) instead of O(N×M)
static neuron_trace_t pre_traces[MAX_NEURONS];   // 720 entries (HLS limit)
static neuron_trace_t post_traces[MAX_NEURONS];  // 720 entries (HLS limit)

struct neuron_trace_t {
    ap_uint<8> trace;              // 8-bit exponential trace
    ap_uint<16> last_spike_time;   // Timestamp for lazy update
};
```

**Lazy Update**: Traces are only recomputed on spike arrival using a 16-entry
LUT for exponential decay, avoiding per-cycle updates.

**R-STDP**: Reward-modulated variant: $\Delta w = eligibility \cdot reward$

**Integration with Core Group**: The Event Router's `learn_spike` output
forwards all spike events to the HLS learning engine. Weight updates
flow back through the router to the appropriate core group (intra-group)
or connectivity table (inter-group).

**Parameters**:
- a_plus, a_minus: Learning rates (8-bit fixed-point)
- w_min, w_max: Weight bounds (8-bit)
- tau_pre, tau_post: Trace decay time constants
- mu: Weight-dependence exponent (Q4.4 fixed-point)

**File**: `hardware/hls/src/snn_top_hls.cpp`

## Synaptic Weight Memory

Two-tier weight storage reflecting the hierarchical architecture.

### Intra-Group Weights (Dense)

Each core group stores a full 128×128 weight matrix in local BRAM.

```
Address: weight_addr = {src_neuron[6:0], dst_neuron[6:0]}
Data:    5 bits = {exc_flag[4], weight[3:0]}
Size:    128 × 128 × 5b = 81,920 bits per group → 2 RAMB36 + 1 RAMB18
Total:   16 groups × 3 BRAM tiles = 48 BRAM tiles
```

### Inter-Group Weights (Sparse)

The Synaptic Connectivity Table stores sparse connections between groups.

```
Address: {src_group[3:0], src_neuron[6:0], fanout_idx[3:0]} = 15 bits
Data:    17 bits = {valid, dst_group[3:0], dst_neuron[6:0], weight[3:0], exc_inh}
Size:    32K × 17b → 17 RAMB36
Max fanout per neuron: 16 destinations
```

**Total Weight Memory**: 48 + 17 = 65 BRAM36 (~46.4% of xc7z020)

## Communication Interfaces

### AXI4-Lite (Control)

32-bit register access for configuration.

**Config Commands** (via `cfg_router_config_addr`):
| cmd [31:28] | Function | Data Format |
|-------------|----------|-------------|
| 0x0 | CT entry write | {valid, dst_grp, dst_nrn, wt, exc, fanout, src_nrn, src_grp} |
| 0x1 | Intra weight write | {src_nrn, dst_nrn, weight, exc, group_id} |
| 0x2 | Read routed_spike_count | — |
| 0x3 | Read total_neuron_spikes | — |

### AXI4-Stream (Data)

Spike streaming between PS and PL.

**Global Neuron ID** (11-bit, supports 2048 neurons):
```
global_id[10:7] = group_id (0-15)
global_id[6:0]  = local_neuron_id (0-127)
```

## Data Flow

### Inference

1. PS sends input spikes via AXI Stream with 11-bit global neuron IDs
2. Event Router routes spikes to destination core groups
3. Core group FSM integrates weight into target neuron membrane
4. If neuron fires → spike bitmap set, intra-group weights scanned
5. Non-zero intra-group connections pushed to local FIFO
6. Output spikes forwarded to Event Router for inter-group propagation
7. Router queries CT for sparse inter-group connections (up to 16 per source)
8. Output spikes sent back to PS

### Learning

1. Event Router forwards all spikes to HLS learning engine (`learn_spike`)
2. HLS updates pre/post traces using lazy exponential decay
3. LTP/LTD weight deltas computed per Mozafari STDP rule
4. Weight updates routed back through Event Router:
   - Intra-group: forwarded to target core group's weight BRAM
   - Inter-group: forwarded to connectivity table BRAM
5. (R-STDP) Eligibility traces modulated by reward signal

## HLS Pipelining Optimizations

| Loop | Before | After | Improvement |
|------|--------|-------|-------------|
| LTD_LOOP | II=2, UNROLL=2 | II=1, UNROLL=4 | 2× throughput |
| LTP_LOOP | II=2 | II=1 | 2× throughput |
| RSTDP_INNER | No unroll | UNROLL=4 | 4× throughput |
| DECAY_PRE/POST | UNROLL=2 | UNROLL=4 | 2× throughput |
| WEIGHT_SUM | II=2 | II=1 | 2× throughput |

## Power Efficiency

### AC-Based Architecture

- **MAC operation**: ~4.6 pJ (multiply + add)
- **AC operation**: ~0.9 pJ (add only)
- **Savings**: ~5× per synaptic operation

**Estimated PL Breakdown** (Vivado `report_power`, pre-route estimate):
- HLS IP: ~108 mW (65%)
- Verilog RTL (16 groups): ~35 mW (21%)
- PS interface: ~15 mW (9%)
- Clocking: ~8 mW (5%)
- PL total (Vivado estimate): ~166 mW

**Measured Board Power** (PYNQ-Z2, XADC, 2026-02-21):

| XADC Rail | V_meas | I_typ | P_est | Function |
|-----------|--------|-------|-------|----------|
| `vccint`  | 1.017 V | 500 mA | 508.5 mW | PS + PL fabric |
| `vccaux`  | 1.808 V |  60 mA | 108.5 mW | I/O banks |
| `vccbram` | 1.018 V |  20 mA |  20.4 mW | Block RAM |
| `vccpint` | 1.017 V | 150 mA | 152.5 mW | PS (ARM) core |
| `vccpaux` | 1.809 V |  30 mA |  54.3 mW | PS I/O |
| **Total** | | | **844 mW** | ±20% (XADC method) |

> **Note**: XADC estimates P = V_measured × I_typical (fixed datasheet values).
> Rail voltages are regulated and nearly load-independent, so XADC cannot detect
> dynamic switching increments. Idle and inference-active power read identically
> (Δ = +0.14 mW, within 0.5 mW noise floor). For accurate PL-only dynamic power,
> use an external INA226 on the 5 V input rail or Vivado power analysis with a
> switching-activity (.saif) file.

### Energy Optimizations

- Shift-based leak (no multiplier)
- Spike-triggered processing (state updates gated by AER spike events)
- Per-neuron traces (reduce memory access)
- Lazy trace update (compute on-demand)
- 8-bit weights (reduced memory bandwidth)
- Sparse inter-group connectivity (reduced BRAM)

## Build Details

**Target Device**: xc7z020clg400-2
**Clock**: 100 MHz
**Neuron Count**: 2,048 (16 groups × 128)

**Build Command**:
```bash
cd hardware/scripts
./build_integrated.sh
```

**Synthesis Verification**:
```bash
# Run RTL synthesis check (16-group configuration)
cd hardware/scripts
vivado -mode batch -source synth_core_group.tcl
```

**Output Files**: `outputs/snn_integrated.bit`, `outputs/snn_integrated.hwh`

See [developer_guide.md](developer_guide.md) for detailed build instructions.

## References

- [User Guide](user_guide.md) - Usage examples
- [Developer Guide](developer_guide.md) - Development workflow
- [API Reference](api_reference.md) - Python API
