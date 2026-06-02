"""
SNN Accelerator Parameters — AUTO-GENERATED from snn_params.yaml

Generated: 2026-02-22 16:30:32
DO NOT EDIT — modify config/snn_params.yaml and run generate_params.py
"""

# ─── Core Architecture ─────────────────────────────────────────────
NUM_GROUPS          = 16
NEURONS_PER_GROUP   = 128   # max(GROUP_SIZES) — backward compat
MAX_NEURONS_PER_GROUP = 128
MAX_FANOUT_INTER    = 16
SPIKE_BUFFER_DEPTH  = 64

# ─── Per-Group Neuron Counts ────────────────────────────────────────
GROUP_SIZES         = [128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128]
TOTAL_NEURONS       = 2048

# ─── Data Widths ────────────────────────────────────────────────────
DATA_WIDTH          = 16
WEIGHT_WIDTH        = 8
THRESHOLD_WIDTH     = 16
LEAK_WIDTH          = 8
REFRAC_WIDTH        = 8

# ─── Derived Bit Widths ─────────────────────────────────────────────
GROUP_ID_WIDTH      = 4   # clog2(16)
LOCAL_ID_WIDTH      = 7   # clog2(128)
GLOBAL_ID_WIDTH     = 11  # GROUP_ID_WIDTH + LOCAL_ID_WIDTH
FANOUT_IDX_WIDTH    = 4   # clog2(16)

# ─── Derived Counts ─────────────────────────────────────────────────
MAX_NEURONS         = 2048  # sum(GROUP_SIZES)
CT_DATA_WIDTH       = 21  # 1+GROUP_ID+LOCAL_ID+WEIGHT+1
NEURON_STATE_WIDTH  = 24  # DATA_WIDTH + REFRAC_WIDTH

# ─── Weight Representation ────────────────────────────────────────
MAX_WEIGHT          = 255  # (1 << WEIGHT_WIDTH) - 1
MIN_WEIGHT          = 0
WEIGHT_FLAG_WIDTH   = 9   # WEIGHT_WIDTH + 1
MAX_WEIGHT_DELTA    = 255

# ─── HLS Interface ────────────────────────────────────────────────
HLS_NEURON_ID_WIDTH = 13
HLS_MAX_NEURONS     = 2048
HLS_WEIGHT_WIDTH    = 8
NEURON_ID_WIDTH     = GLOBAL_ID_WIDTH  # Alias

# ─── NeuronGroup Connection Topology (Brian2-style) ────────────────
NUM_NEURON_GROUPS       = 9
NUM_CONNECTIONS         = 8
MAX_WEIGHT_BUFFER_SIZE  = 843776
MAX_SRC_NEURONS         = 1024
MAX_DST_NEURONS         = 1024
TOTAL_LOGICAL_NEURONS   = 4890

NEURON_GROUP_NAMES  = ['input_0', 'input_1', 'input_2', 'input_3', 'hidden_0', 'hidden_1', 'hidden_2', 'hidden_3', 'output']
NEURON_GROUP_SIZES  = [196, 196, 196, 196, 1024, 1024, 1024, 1024, 10]
NEURON_GROUP_ID_START = [0, 196, 392, 588, 784, 1808, 2832, 3856, 4880, 4890]

# Per-Connection metadata: list of dicts
CONNECTIONS = [
    {"name": "in0_to_hid0", "src_group": 0, "dst_group": 4, "src_size": 196, "dst_size": 1024, "weight_offset": 0, "num_weights": 200704, "src_id_start": 0, "dst_id_start": 784},
    {"name": "in1_to_hid1", "src_group": 1, "dst_group": 5, "src_size": 196, "dst_size": 1024, "weight_offset": 200704, "num_weights": 200704, "src_id_start": 196, "dst_id_start": 1808},
    {"name": "in2_to_hid2", "src_group": 2, "dst_group": 6, "src_size": 196, "dst_size": 1024, "weight_offset": 401408, "num_weights": 200704, "src_id_start": 392, "dst_id_start": 2832},
    {"name": "in3_to_hid3", "src_group": 3, "dst_group": 7, "src_size": 196, "dst_size": 1024, "weight_offset": 602112, "num_weights": 200704, "src_id_start": 588, "dst_id_start": 3856},
    {"name": "hid0_to_output", "src_group": 4, "dst_group": 8, "src_size": 1024, "dst_size": 10, "weight_offset": 802816, "num_weights": 10240, "src_id_start": 784, "dst_id_start": 4880},
    {"name": "hid1_to_output", "src_group": 5, "dst_group": 8, "src_size": 1024, "dst_size": 10, "weight_offset": 813056, "num_weights": 10240, "src_id_start": 1808, "dst_id_start": 4880},
    {"name": "hid2_to_output", "src_group": 6, "dst_group": 8, "src_size": 1024, "dst_size": 10, "weight_offset": 823296, "num_weights": 10240, "src_id_start": 2832, "dst_id_start": 4880},
    {"name": "hid3_to_output", "src_group": 7, "dst_group": 8, "src_size": 1024, "dst_size": 10, "weight_offset": 833536, "num_weights": 10240, "src_id_start": 3856, "dst_id_start": 4880},
]

# ─── Fixed-Point (HLS ap_fixed<16,8>) ─────────────────────────────
FIXED_POINT_FRAC_BITS = 8
FIXED_POINT_SCALE     = 1 << FIXED_POINT_FRAC_BITS  # 256

# ─── Legacy 8-bit weight constants (backward compat) ──────────────
LEGACY_MAX_WEIGHT   = 127
LEGACY_MIN_WEIGHT   = -128
LEGACY_WEIGHT_SCALE = 128
WEIGHT_SCALE        = 256

# ─── Weight Memory Optimization (Loihi/TrueNorth/KIST) ──────────
WEIGHT_BITS             = 4
PACKED_MAX_WEIGHT       = 7
PACKED_MIN_WEIGHT       = -8
TIME_EMBEDDING          = 1
AUXILIARY_LUTRAM         = 1
PACKED_BUFFER_BYTES     = 421888

# ─── FPGA Target ──────────────────────────────────────────────────
FPGA_PART           = "xc7z020clg400-2"
CLOCK_PERIOD_NS     = 10
BOARD               = "tul.com.tw:pynq-z2:part0:1.0"
