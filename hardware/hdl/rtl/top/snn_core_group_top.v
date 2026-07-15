//-----------------------------------------------------------------------------
// Title         : SNN Core Group Top - Hierarchical Neuromorphic Processor
// Project       : PYNQ-Z2 SNN Accelerator
// File          : snn_core_group_top.v
// Author        : Jiwoon Lee (@metr0jw)
// Organization  : Kwangwoon University, Seoul, South Korea
// Contact       : jwlee@linux.com
// Description   : Top-level integration of the Core Group architecture:
//
//                 ┌─────────────────────────────────────────────────────┐
//                 │              snn_core_group_top                     │
//                 │                                                     │
//                 │  ┌─────────────────┐    ┌──────────────────┐        │
//                 │  │ design_1_wrapper│    │  Synaptic Conn.  │        │
//                 │  │  (PS + HLS IP)  │    │  Table (BRAM)    │        │
//                 │  └──────┬──────────┘    └────────┬─────────┘        │
//                 │         │                        │                  │
//                 │         ▼                        ▼                  │
//                 │  ┌──────────────────────────────────────────┐       │
//                 │  │         Event Router (NG)                │       │
//                 │  │  - Round-robin arbitration               │       │
//                 │  │  - Sparse inter-group routing            │       │
//                 │  │  - Learning engine interface             │       │
//                 │  │  - External sensor/host input            │       │
//                 │  └──┬───┬───┬───┬───┬───┬────────┬───┬──────┘       │
//                 │     │   │   │   │   │   │        │   │              │
//                 │     ▼   ▼   ▼   ▼   ▼   ▼  ...   ▼   ▼              │
//                 │  ┌───┐┌───┐┌───┐┌───┐┌───┐     ┌────┐┌────┐         │
//                 │  │CG0││CG1││CG2││CG3││CG4│ ... │CG14││CG15│         │
//                 │  │ N0││ N1││ N2││ N3││ N4│     │ N14││ N15│         │
//                 │  └───┘└───┘└───┘└───┘└───┘     └────┘└────┘         │
//                 │                                                     │
//                 │  Variable group sizes: Ni = SNN_GROUP_SIZE_i        │
//                 │  Default: all 128 (16 × 128 = 2048 neurons)         │
//                 │  Bus width: LOCAL_ID_WIDTH = clog2(max(Ni))         │
//                 └─────────────────────────────────────────────────────┘
//
// Config Register Mapping (from AXI-Lite):
//   cfg_router_config_addr[31:28]:
//     0x0: Connectivity table write (inter-group)
//     0x1: Intra-group weight write (selects group via addr[27:25])
//     0x2: Status readback
//
//   cfg_router_config_wdata format for connectivity table (8-bit weight):
//     [31]    = valid
//     [30:27] = dst_group  (4-bit, supports up to 16 groups)
//     [26:20] = dst_neuron (7-bit, max neurons/group)
//     [19:12] = weight     (8-bit unsigned magnitude)
//     [11]    = exc_inh
//     [10:7]  = fanout_idx (4-bit, max 16 fanout)
//     [6:0]   = src_neuron (7-bit)
//     addr[3:0] = src_group (4-bit)
//
//   cfg_router_config_wdata format for intra-group weight (8-bit weight):
//     [31:25] = src_neuron
//     [24:18] = dst_neuron
//     [17:10] = weight     (8-bit unsigned magnitude)
//     [9]     = exc
//     [8:5]   = group_id   (4-bit, supports up to 16 groups)
//     [4]     = reserved
//     [3:0]   = sparse fanout_idx (ignored by dense weight_mem)
//
// Resource Budget (xc7z020clg400-2):
//   - 16 Core Groups:      ~48 BRAM36, ~9,120 LUT
//   - Connectivity Table:  ~16 BRAM36 (32K×17b)
//   - Event Router:        ~300 LUT
//   - Block Design (PS+HLS): existing
//   Total RTL: ~64 BRAM36, ~9,420 LUT
//   Leaves ~54% BRAM, ~82% LUT for HLS IP
//-----------------------------------------------------------------------------

`timescale 1ns / 1ps
`include "snn_params.vh"

module snn_core_group_top #(
    // Core Group Parameters (defaults from snn_params.yaml via snn_params.vh)
    parameter NUM_GROUPS            = `SNN_NUM_GROUPS,
    parameter NEURONS_PER_GROUP     = `SNN_NEURONS_PER_GROUP,
    parameter WEIGHT_WIDTH          = `SNN_WEIGHT_WIDTH,
    parameter MAX_FANOUT_INTER      = `SNN_MAX_FANOUT_INTER,
    parameter GROUP_ID_WIDTH        = `SNN_GROUP_ID_WIDTH,
    parameter LOCAL_ID_WIDTH        = `SNN_LOCAL_ID_WIDTH,
    parameter GLOBAL_ID_WIDTH       = `SNN_GLOBAL_ID_WIDTH,

    // Neuron Parameters
    parameter DATA_WIDTH            = `SNN_DATA_WIDTH,
    parameter THRESHOLD_WIDTH       = `SNN_THRESHOLD_WIDTH,
    parameter LEAK_WIDTH            = `SNN_LEAK_WIDTH,
    parameter REFRAC_WIDTH          = `SNN_REFRAC_WIDTH,
    parameter SPIKE_BUFFER_DEPTH    = `SNN_SPIKE_BUFFER_DEPTH,

    // HLS Compatibility
    parameter HLS_NEURON_ID_WIDTH   = `SNN_HLS_NEURON_ID_WIDTH,
    parameter HLS_MAX_NEURONS       = `SNN_TOTAL_NEURONS,
    parameter HLS_WEIGHT_WIDTH      = `SNN_HLS_WEIGHT_WIDTH,

    // Current runtime maps classifier neurons round-robin into the first
    // CLASSIFIER_NEURONS logical IDs. The latch below ignores input-source
    // proxy neurons used by real CT modes.
    parameter CLASSIFIER_NEURONS    = 150,
    parameter INPUT_SOURCE_NEURONS  = 784,
    parameter NUM_CLASSES           = 10,
    parameter FPS_PER_CLASS         = 15,

    // Profile build-time switches. BASIC keeps only low-cost aggregate
    // counters in the AXI read window; DETAIL/PER_GROUP retain the larger
    // debug windows when explicitly enabled for bring-up.
    parameter ENABLE_PROFILE_BASIC     = 1,
    parameter ENABLE_PROFILE_DETAIL    = 0,
    parameter ENABLE_PROFILE_PER_GROUP = 0,

    // Intra-group recurrent fanout implementation switches. Stage 1 keeps the
    // sparse table instantiated/writable while the sparse routing FSM is added
    // in a later step.
    parameter ENABLE_INTRA_SPARSE      = 1,
    parameter ENABLE_INTRA_DENSE       = 0,
    parameter INTRA_MAX_FANOUT         = MAX_FANOUT_INTER
)(
    //-------------------------------------------------------------------------
    // DDR Interface (directly from PS)
    //-------------------------------------------------------------------------
    inout  wire [14:0]  DDR_addr,
    inout  wire [2:0]   DDR_ba,
    inout  wire         DDR_cas_n,
    inout  wire         DDR_ck_n,
    inout  wire         DDR_ck_p,
    inout  wire         DDR_cke,
    inout  wire         DDR_cs_n,
    inout  wire [3:0]   DDR_dm,
    inout  wire [31:0]  DDR_dq,
    inout  wire [3:0]   DDR_dqs_n,
    inout  wire [3:0]   DDR_dqs_p,
    inout  wire         DDR_odt,
    inout  wire         DDR_ras_n,
    inout  wire         DDR_reset_n,
    inout  wire         DDR_we_n,

    //-------------------------------------------------------------------------
    // Fixed IO (PS)
    //-------------------------------------------------------------------------
    inout  wire         FIXED_IO_ddr_vrn,
    inout  wire         FIXED_IO_ddr_vrp,
    inout  wire [53:0]  FIXED_IO_mio,
    inout  wire         FIXED_IO_ps_clk,
    inout  wire         FIXED_IO_ps_porb,
    inout  wire         FIXED_IO_ps_srstb
);

    //=========================================================================
    // Derived Parameters
    //=========================================================================
    localparam FANOUT_IDX_WIDTH = $clog2(MAX_FANOUT_INTER);
    localparam TOTAL_NEURONS    = `SNN_TOTAL_NEURONS;

    //=========================================================================
    // Clock / Reset from PS Block Design
    //=========================================================================
    wire clk_100mhz;
    wire rst_n_sync;
    wire debug_learning_active;

    //=========================================================================
    // HLS <-> RTL Interface Signals (from Block Design)
    //=========================================================================
    wire                            hls_spike_out_valid;
    wire [HLS_NEURON_ID_WIDTH-1:0]  hls_spike_out_neuron_id;
    wire [HLS_WEIGHT_WIDTH-1:0]     hls_spike_out_weight;
    wire                            rtl_spike_in_ready;
    reg                             hls_spike_out_toggle_d;
    wire                            hls_spike_out_pulse;
    wire                            router_ext_spike_ready;
    localparam EXT_INPUT_FIFO_DEPTH = 32;
    localparam EXT_INPUT_FIFO_AW    = $clog2(EXT_INPUT_FIFO_DEPTH);
    localparam [EXT_INPUT_FIFO_AW:0] EXT_INPUT_FIFO_DEPTH_COUNT = EXT_INPUT_FIFO_DEPTH;
    wire                            hls_ext_pending;
    wire [GLOBAL_ID_WIDTH-1:0]      hls_ext_pending_id;
    wire [WEIGHT_WIDTH-1:0]         hls_ext_pending_weight;
    reg  [GLOBAL_ID_WIDTH-1:0]      ext_input_fifo_id [0:EXT_INPUT_FIFO_DEPTH-1];
    reg  [WEIGHT_WIDTH-1:0]         ext_input_fifo_weight [0:EXT_INPUT_FIFO_DEPTH-1];
    reg  [EXT_INPUT_FIFO_AW-1:0]    ext_input_fifo_wr_ptr;
    reg  [EXT_INPUT_FIFO_AW-1:0]    ext_input_fifo_rd_ptr;
    reg  [EXT_INPUT_FIFO_AW:0]      ext_input_fifo_count;
    wire                            ext_input_fifo_empty;
    wire                            ext_input_fifo_full;
    wire                            ext_input_fifo_push;
    wire                            ext_input_fifo_pop;
    integer                         ext_fifo_i;

    wire                            rtl_spike_out_valid;
    wire [HLS_NEURON_ID_WIDTH-1:0]  rtl_spike_out_neuron_id;
    wire [HLS_WEIGHT_WIDTH-1:0]     rtl_spike_out_weight;
    wire                            hls_spike_in_ready;

    // Inference-only HLS has no learned-weight update channel.
    wire                            hls_learn_weight_valid;
    wire [GROUP_ID_WIDTH-1:0]       hls_learn_weight_group;
    wire [LOCAL_ID_WIDTH-1:0]       hls_learn_weight_src;
    wire [LOCAL_ID_WIDTH-1:0]       hls_learn_weight_dst;
    wire [WEIGHT_WIDTH-1:0]         hls_learn_weight_data;
    wire                            hls_learn_weight_exc;
    wire                            hls_learn_weight_is_inter;
    wire [GROUP_ID_WIDTH-1:0]       hls_learn_weight_dst_group;
    wire [FANOUT_IDX_WIDTH-1:0]     hls_learn_weight_fanout_idx;
    wire                            rtl_learn_weight_ready;

    assign hls_learn_weight_valid      = 1'b0;
    assign hls_learn_weight_group      = {GROUP_ID_WIDTH{1'b0}};
    assign hls_learn_weight_src        = {LOCAL_ID_WIDTH{1'b0}};
    assign hls_learn_weight_dst        = {LOCAL_ID_WIDTH{1'b0}};
    assign hls_learn_weight_data       = {WEIGHT_WIDTH{1'b0}};
    assign hls_learn_weight_exc        = 1'b0;
    assign hls_learn_weight_is_inter   = 1'b0;
    assign hls_learn_weight_dst_group  = {GROUP_ID_WIDTH{1'b0}};
    assign hls_learn_weight_fanout_idx = {FANOUT_IDX_WIDTH{1'b0}};

    wire                            hls_snn_enable;
    wire                            hls_snn_reset;
    wire                            rtl_snn_ready;
    wire                            rtl_snn_busy;
    wire [15:0]                     hls_threshold_out;
    wire [15:0]                     hls_leak_rate_out;

    //=========================================================================
    // Config Register Interface (from AXI-Lite in Block Design)
    //=========================================================================
    wire                            cfg_router_config_we;
    wire [31:0]                     cfg_router_config_addr;
    wire [31:0]                     cfg_router_config_wdata;
    wire [31:0]                     cfg_router_config_rdata;

    wire                            cfg_neuron_config_we;
    wire [9:0]                      cfg_neuron_config_addr;
    wire [31:0]                     cfg_neuron_config_wdata;

    wire [15:0]                     cfg_global_threshold;
    wire [7:0]                      cfg_global_leak_rate;
    wire [7:0]                      cfg_global_refrac_period;
    wire                            cfg_profile_start;
    wire                            cfg_profile_stop;
    wire [15:0]                     cfg_profile_expected_count;
    wire [7:0]                      cfg_profile_index;
    reg  [31:0]                     cfg_profile_data;
    wire [31:0]                     cfg_profile_info;

    //=========================================================================
    // Internal Wiring: Event Router <-> Core Groups
    //=========================================================================

    // Core group output spikes → event router
    wire [NUM_GROUPS-1:0]                       grp_spike_valid;
    wire [NUM_GROUPS*LOCAL_ID_WIDTH-1:0]        grp_spike_neuron_id;
    wire [NUM_GROUPS-1:0]                       grp_spike_ready;

    // Event router → core group input spikes
    wire [NUM_GROUPS-1:0]                       grp_in_valid;
    wire [NUM_GROUPS*LOCAL_ID_WIDTH-1:0]        grp_in_dest_id;
    wire [NUM_GROUPS*WEIGHT_WIDTH-1:0]          grp_in_weight;
    wire [NUM_GROUPS-1:0]                       grp_in_exc;
    wire [NUM_GROUPS-1:0]                       grp_in_ready;
    wire [NUM_GROUPS-1:0]                       grp_sample_clear_done;

    // Weight config from event router → core groups
    wire [NUM_GROUPS-1:0]                       grp_weight_we;
    wire [LOCAL_ID_WIDTH-1:0]                   grp_weight_src;
    wire [LOCAL_ID_WIDTH-1:0]                   grp_weight_dst;
    wire [WEIGHT_WIDTH-1:0]                     grp_weight_data;
    wire                                        grp_weight_exc;

    // Core group status
    wire [NUM_GROUPS*32-1:0]                    grp_spike_count;
    wire [NUM_GROUPS-1:0]                       grp_busy;
    wire [NUM_GROUPS*8*32-1:0]                  grp_profile_snapshot;
    wire [NUM_GROUPS*10*32-1:0]                 grp_score_live;

    //=========================================================================
    // Internal Wiring: Event Router <-> Connectivity Table
    //=========================================================================
    wire                          ct_lookup_en;
    wire [GROUP_ID_WIDTH-1:0]     ct_lookup_src_group;
    wire [LOCAL_ID_WIDTH-1:0]     ct_lookup_src_neuron;
    wire [FANOUT_IDX_WIDTH-1:0]   ct_lookup_fanout_idx;

    wire                          ct_result_valid;
    wire [GROUP_ID_WIDTH-1:0]     ct_result_dst_group;
    wire [LOCAL_ID_WIDTH-1:0]     ct_result_dst_neuron;
    wire [WEIGHT_WIDTH-1:0]       ct_result_weight;
    wire                          ct_result_exc_inh;
    wire                          ct_result_entry_valid;

    // Connectivity table config (from event router or decode logic)
    wire                          ct_cfg_we;
    wire [GROUP_ID_WIDTH-1:0]     ct_cfg_src_group;
    wire [LOCAL_ID_WIDTH-1:0]     ct_cfg_src_neuron;
    wire [FANOUT_IDX_WIDTH-1:0]   ct_cfg_fanout_idx;
    wire                          ct_cfg_valid_bit;
    wire [GROUP_ID_WIDTH-1:0]     ct_cfg_dst_group;
    wire [LOCAL_ID_WIDTH-1:0]     ct_cfg_dst_neuron;
    wire [WEIGHT_WIDTH-1:0]       ct_cfg_weight;
    wire                          ct_cfg_exc_inh;

    // Learning engine interface
    wire                          learn_spike_valid;
    wire [GLOBAL_ID_WIDTH-1:0]    learn_spike_src_id;
    wire                          learn_spike_ready;
    wire                          first_spike_tap_valid;
    wire [GLOBAL_ID_WIDTH-1:0]    first_spike_tap_id;
    wire [HLS_WEIGHT_WIDTH-1:0]   first_spike_tap_weight;

    // The inference/profile build observes routing only and never updates
    // synaptic storage from HLS.
    localparam                    LEARN_WEIGHT_BRIDGE_ENABLE = 1'b0;
    wire                          learn_weight_valid_br;
    wire [GROUP_ID_WIDTH-1:0]     learn_weight_group_br;
    wire [LOCAL_ID_WIDTH-1:0]     learn_weight_src_br;
    wire [LOCAL_ID_WIDTH-1:0]     learn_weight_dst_br;
    wire [WEIGHT_WIDTH-1:0]       learn_weight_data_br;
    wire                          learn_weight_exc_br;
    wire                          learn_weight_is_inter_br;
    wire [GROUP_ID_WIDTH-1:0]     learn_weight_dst_group_br;
    wire [FANOUT_IDX_WIDTH-1:0]   learn_weight_fanout_idx_br;
    wire                          learn_weight_ready_br;

    assign learn_weight_valid_br      = LEARN_WEIGHT_BRIDGE_ENABLE ? hls_learn_weight_valid : 1'b0;
    assign learn_weight_group_br      = LEARN_WEIGHT_BRIDGE_ENABLE ? hls_learn_weight_group : {GROUP_ID_WIDTH{1'b0}};
    assign learn_weight_src_br        = LEARN_WEIGHT_BRIDGE_ENABLE ? hls_learn_weight_src : {LOCAL_ID_WIDTH{1'b0}};
    assign learn_weight_dst_br        = LEARN_WEIGHT_BRIDGE_ENABLE ? hls_learn_weight_dst : {LOCAL_ID_WIDTH{1'b0}};
    assign learn_weight_data_br       = LEARN_WEIGHT_BRIDGE_ENABLE ? hls_learn_weight_data : {WEIGHT_WIDTH{1'b0}};
    assign learn_weight_exc_br        = LEARN_WEIGHT_BRIDGE_ENABLE ? hls_learn_weight_exc : 1'b0;
    assign learn_weight_is_inter_br   = LEARN_WEIGHT_BRIDGE_ENABLE ? hls_learn_weight_is_inter : 1'b0;
    assign learn_weight_dst_group_br  = LEARN_WEIGHT_BRIDGE_ENABLE ? hls_learn_weight_dst_group : {GROUP_ID_WIDTH{1'b0}};
    assign learn_weight_fanout_idx_br = LEARN_WEIGHT_BRIDGE_ENABLE ? hls_learn_weight_fanout_idx : {FANOUT_IDX_WIDTH{1'b0}};

    // Router status
    wire [31:0]                   routed_spike_count;
    wire                          router_busy;
    wire [(11+2*NUM_GROUPS+2*NUM_CLASSES)*32-1:0] router_profile_snapshot;
    wire                          router_profile_fanout_valid;
    wire [GROUP_ID_WIDTH-1:0]     router_profile_fanout_src_group;
    wire [LOCAL_ID_WIDTH-1:0]     router_profile_fanout_src_neuron;
    wire [GROUP_ID_WIDTH-1:0]     router_profile_fanout_dst_group;
    wire [LOCAL_ID_WIDTH-1:0]     router_profile_fanout_dst_neuron;
    wire [WEIGHT_WIDTH-1:0]       router_profile_fanout_weight;
    wire                          router_profile_class_valid;
    wire [3:0]                    router_profile_class_id;
    wire [WEIGHT_WIDTH-1:0]       router_profile_class_weight;
    //=========================================================================
    // Global first-spike tap for TTFS first-spike classification
    //=========================================================================
    // event_router_ng round-robins among group outputs before forwarding the
    // observation to HLS. For first-spike-only classification, tap the earliest
    // group output directly so the HLS latch is not biased by router RR order.
    reg                          first_spike_tap_pending;
    reg                          first_spike_tap_done;
    reg [GLOBAL_ID_WIDTH-1:0]    first_spike_tap_id_reg;
    reg                          hls_spike_in_ready_d;
    reg                          first_spike_tap_seen;
    reg [GLOBAL_ID_WIDTH-1:0]    first_spike_tap_candidate;
    reg                          first_classifier_spike_seen;
    reg [GLOBAL_ID_WIDTH-1:0]    first_classifier_spike_candidate;
    integer                      first_spike_tap_i;

    function is_classifier_global_id;
        input [GROUP_ID_WIDTH-1:0] group_id;
        input [LOCAL_ID_WIDTH-1:0] local_id;
        integer logical_id;
        begin
            logical_id = local_id * NUM_GROUPS + group_id;
            is_classifier_global_id = (logical_id < CLASSIFIER_NEURONS);
        end
    endfunction

    function is_input_source_global_id;
        input [GROUP_ID_WIDTH-1:0] group_id;
        input [LOCAL_ID_WIDTH-1:0] local_id;
        integer logical_id;
        begin
            logical_id = local_id * NUM_GROUPS + group_id;
            is_input_source_global_id =
                (logical_id >= CLASSIFIER_NEURONS) &&
                (logical_id < CLASSIFIER_NEURONS + INPUT_SOURCE_NEURONS);
        end
    endfunction

    function [3:0] classifier_class_id;
        input [GROUP_ID_WIDTH-1:0] group_id;
        input [LOCAL_ID_WIDTH-1:0] local_id;
        integer logical_id;
        reg [LOCAL_ID_WIDTH:0] local_plus_group;
        begin
            if (NUM_GROUPS == 16 && FPS_PER_CLASS == 15) begin
                // floor((16*local+group)/15) = local + floor((local+group)/15).
                // Classifier IDs are below 150, so floor((local+group)/15) is 0 or 1.
                local_plus_group = local_id + group_id;
                classifier_class_id = local_id[3:0] +
                    ((local_plus_group >= 15) ? 4'd1 : 4'd0);
            end else begin
                logical_id = local_id * NUM_GROUPS + group_id;
                classifier_class_id = logical_id / FPS_PER_CLASS;
            end
        end
    endfunction

    always @(*) begin
        first_spike_tap_seen = 1'b0;
        first_spike_tap_candidate = {GLOBAL_ID_WIDTH{1'b0}};
        first_classifier_spike_seen = 1'b0;
        first_classifier_spike_candidate = {GLOBAL_ID_WIDTH{1'b0}};
        for (first_spike_tap_i = 0; first_spike_tap_i < NUM_GROUPS; first_spike_tap_i = first_spike_tap_i + 1) begin
            if (!first_spike_tap_seen && grp_spike_valid[first_spike_tap_i]) begin
                first_spike_tap_seen = 1'b1;
                first_spike_tap_candidate = {
                    first_spike_tap_i[GROUP_ID_WIDTH-1:0],
                    grp_spike_neuron_id[first_spike_tap_i*LOCAL_ID_WIDTH +: LOCAL_ID_WIDTH]
                };
            end
            if (ENABLE_PROFILE_DETAIL && !first_classifier_spike_seen && grp_spike_valid[first_spike_tap_i] &&
                is_classifier_global_id(
                    first_spike_tap_i[GROUP_ID_WIDTH-1:0],
                    grp_spike_neuron_id[first_spike_tap_i*LOCAL_ID_WIDTH +: LOCAL_ID_WIDTH]
                )) begin
                first_classifier_spike_seen = 1'b1;
                first_classifier_spike_candidate = {
                    first_spike_tap_i[GROUP_ID_WIDTH-1:0],
                    grp_spike_neuron_id[first_spike_tap_i*LOCAL_ID_WIDTH +: LOCAL_ID_WIDTH]
                };
            end
        end
    end

    always @(posedge clk_100mhz) begin
        if (!rst_n_sync || hls_snn_reset || cfg_profile_start) begin
            first_spike_tap_pending <= 1'b0;
            first_spike_tap_done    <= 1'b0;
            first_spike_tap_id_reg  <= {GLOBAL_ID_WIDTH{1'b0}};
            hls_spike_in_ready_d    <= 1'b0;
        end else begin
            hls_spike_in_ready_d <= hls_spike_in_ready;

            if (first_spike_tap_pending && (hls_spike_in_ready != hls_spike_in_ready_d)) begin
                first_spike_tap_pending <= 1'b0;
            end else if (!first_spike_tap_pending && !first_spike_tap_done && first_spike_tap_seen) begin
                first_spike_tap_pending <= 1'b1;
                first_spike_tap_done    <= 1'b1;
                first_spike_tap_id_reg  <= first_spike_tap_candidate;
            end
        end
    end

    assign first_spike_tap_valid  = first_spike_tap_pending;
    assign first_spike_tap_id     = first_spike_tap_id_reg;
    assign first_spike_tap_weight = {HLS_WEIGHT_WIDTH{1'b0}};

    //=========================================================================
    // Per-sample profile control and snapshot readback
    //=========================================================================
    localparam PROFILE_ROUTER_TOTAL_COUNT  = 11 + 2*NUM_GROUPS + 2*NUM_CLASSES;
    localparam PROFILE_CORE_BASE           = PROFILE_ROUTER_TOTAL_COUNT;
    localparam PROFILE_CORE_METRIC_COUNT   = 8;
    localparam PROFILE_CLASSIFIER_BASE     = PROFILE_CORE_BASE + PROFILE_CORE_METRIC_COUNT*NUM_GROUPS;
    localparam PROFILE_CLASS_COUNT_BASE    = PROFILE_CLASSIFIER_BASE + 3;
    localparam PROFILE_CLASS_SCORE_BASE    = PROFILE_CLASS_COUNT_BASE + NUM_CLASSES;
    localparam PROFILE_CLASS_EVENT_BASE    = PROFILE_CLASS_SCORE_BASE + NUM_CLASSES;
    localparam PROFILE_INPUT_SOURCE_GROUP_BASE = PROFILE_CLASS_EVENT_BASE + NUM_CLASSES;
    localparam PROFILE_INPUT_SOURCE_BITMAP_BASE = PROFILE_INPUT_SOURCE_GROUP_BASE + NUM_GROUPS;
    localparam PROFILE_INPUT_SOURCE_BITMAP_WORDS = (INPUT_SOURCE_NEURONS + 31) / 32;
    localparam PROFILE_INPUT_SOURCE_BITMAP_BITS = PROFILE_INPUT_SOURCE_BITMAP_WORDS * 32;
    localparam PROFILE_INPUT_SOURCE_PIXEL_WIDTH = $clog2(INPUT_SOURCE_NEURONS);
    localparam PROFILE_INPUT_SOURCE_CT_BITMAP_BASE = PROFILE_INPUT_SOURCE_BITMAP_BASE + PROFILE_INPUT_SOURCE_BITMAP_WORDS;
    localparam PROFILE_FULL_COUNT          = PROFILE_INPUT_SOURCE_CT_BITMAP_BASE + PROFILE_INPUT_SOURCE_BITMAP_WORDS;
    localparam PROFILE_BASIC_SCALAR_COUNT = 10;
    localparam PROFILE_BASIC_CLASS_COUNT_BASE = PROFILE_BASIC_SCALAR_COUNT;
    localparam PROFILE_BASIC_CLASS_SCORE_BASE = PROFILE_BASIC_CLASS_COUNT_BASE + NUM_CLASSES;
    localparam PROFILE_BASIC_CLASS_EVENT_BASE = PROFILE_BASIC_CLASS_SCORE_BASE + NUM_CLASSES;
    localparam PROFILE_BASIC_COUNT         = PROFILE_BASIC_CLASS_EVENT_BASE + NUM_CLASSES;
    localparam PROFILE_BASIC_ONLY          =
        (ENABLE_PROFILE_BASIC != 0) &&
        (ENABLE_PROFILE_DETAIL == 0) &&
        (ENABLE_PROFILE_PER_GROUP == 0);
    localparam PROFILE_CLASS_ENABLE        =
        (ENABLE_PROFILE_BASIC != 0) || (ENABLE_PROFILE_DETAIL != 0);
    localparam PROFILE_TOTAL_COUNT         = PROFILE_BASIC_ONLY ? PROFILE_BASIC_COUNT : PROFILE_FULL_COUNT;
    localparam [7:0] PROFILE_NUM_GROUPS_INFO = NUM_GROUPS;
    localparam [15:0] PROFILE_TOTAL_COUNT_INFO = PROFILE_TOTAL_COUNT;
    localparam [7:0] PROFILE_VERSION_INFO = PROFILE_BASIC_ONLY ? 8'h0D : 8'h09;

    reg        profile_active;
    reg        profile_done;
    reg        bp_score_mode;
    reg        profile_stop_local;
    reg        sample_clear_pending;
    reg [31:0] total_latency_live;
    reg [31:0] total_latency_snapshot;
    reg [31:0] first_spike_latency_live;
    reg [31:0] first_spike_latency_snapshot;
    reg [31:0] service_cycles_live;
    reg [31:0] service_cycles_snapshot;
    reg [15:0] sample_input_count;
    reg        sample_input_done;
    reg        sample_service_done;
    reg        sample_seen_input;
    reg        sample_seen_output;
    reg [31:0] service_dbg_input_done_cycle_live;
    reg [31:0] service_dbg_input_done_cycle_snapshot;
    reg [31:0] service_dbg_hls_pending_clear_cycle_live;
    reg [31:0] service_dbg_hls_pending_clear_cycle_snapshot;
    reg        service_dbg_input_done_seen;
    reg        service_dbg_hls_pending_clear_seen;
    reg        first_classifier_spike_valid_live;
    reg        first_classifier_spike_valid_snapshot;
    reg [GLOBAL_ID_WIDTH-1:0] first_classifier_spike_id_live;
    reg [GLOBAL_ID_WIDTH-1:0] first_classifier_spike_id_snapshot;
    reg [31:0] first_classifier_spike_cycle_live;
    reg [31:0] first_classifier_spike_cycle_snapshot;
    reg [31:0] classifier_class_count_live [0:NUM_CLASSES-1];
    reg [31:0] classifier_class_count_snapshot [0:NUM_CLASSES-1];
    reg [31:0] classifier_class_score_live [0:NUM_CLASSES-1];
    reg [31:0] classifier_class_score_snapshot [0:NUM_CLASSES-1];
    reg [31:0] classifier_class_event_live [0:NUM_CLASSES-1];
    reg [31:0] classifier_class_event_snapshot [0:NUM_CLASSES-1];
    reg signed [31:0] bp_score_aggregate [0:NUM_CLASSES-1];
    reg [GROUP_ID_WIDTH-1:0] bp_score_group_for_class [0:NUM_CLASSES-1];
    integer bp_score_gi;
    integer bp_score_ci;

    always @(*) begin
        for (bp_score_ci = 0; bp_score_ci < NUM_CLASSES; bp_score_ci = bp_score_ci + 1) begin
            bp_score_aggregate[bp_score_ci] = 32'sd0;
            for (bp_score_gi = 0; bp_score_gi < NUM_GROUPS; bp_score_gi = bp_score_gi + 1)
                if (bp_score_group_for_class[bp_score_ci] == bp_score_gi)
                    bp_score_aggregate[bp_score_ci] =
                        $signed(grp_score_live[(bp_score_gi*NUM_CLASSES+bp_score_ci)*32 +: 32]);
        end
    end
    reg [31:0] input_source_group_count_live [0:NUM_GROUPS-1];
    reg [31:0] input_source_group_count_snapshot [0:NUM_GROUPS-1];
    reg [PROFILE_INPUT_SOURCE_BITMAP_BITS-1:0] input_source_bitmap_live;
    reg [PROFILE_INPUT_SOURCE_BITMAP_BITS-1:0] input_source_bitmap_snapshot;
    reg [PROFILE_INPUT_SOURCE_BITMAP_BITS-1:0] input_source_ct_bitmap_live;
    reg [PROFILE_INPUT_SOURCE_BITMAP_BITS-1:0] input_source_ct_bitmap_snapshot;
    reg        profile_group_event_valid_comb;
    reg [GROUP_ID_WIDTH-1:0] profile_group_event_group_comb;
    reg [LOCAL_ID_WIDTH-1:0] profile_group_event_neuron_comb;
    reg        profile_group_event_class_valid_comb;
    reg [3:0]  profile_group_event_class_id_comb;
    reg        profile_group_event_input_source_comb;
    reg [PROFILE_INPUT_SOURCE_PIXEL_WIDTH-1:0] profile_group_event_pixel_comb;
    reg        profile_group_event_valid_d;
    reg [GROUP_ID_WIDTH-1:0] profile_group_event_group_d;
    reg        profile_group_event_class_valid_d;
    reg [3:0]  profile_group_event_class_id_d;
    reg        profile_group_event_input_source_d;
    reg [PROFILE_INPUT_SOURCE_PIXEL_WIDTH-1:0] profile_group_event_pixel_d;
    reg        profile_class_valid_d;
    reg [3:0]  profile_class_id_d;
    reg [WEIGHT_WIDTH-1:0] profile_class_weight_d;
    reg        profile_fanout_input_source_valid_comb;
    reg [PROFILE_INPUT_SOURCE_PIXEL_WIDTH-1:0] profile_fanout_input_source_pixel_comb;
    reg        profile_fanout_input_source_valid_d;
    reg [PROFILE_INPUT_SOURCE_PIXEL_WIDTH-1:0] profile_fanout_input_source_pixel_d;
    reg [31:0] basic_drop_spike_sum;
    reg [31:0] basic_drop_spike_snapshot;
    integer profile_sel;
    integer class_count_i;
    integer class_count_gi;
    integer input_source_pixel_id;
    integer basic_profile_gi;

    always @(*) begin
        basic_drop_spike_sum = 32'd0;
        for (basic_profile_gi = 0; basic_profile_gi < NUM_GROUPS; basic_profile_gi = basic_profile_gi + 1) begin
            basic_drop_spike_sum =
                basic_drop_spike_sum +
                grp_profile_snapshot[(basic_profile_gi*PROFILE_CORE_METRIC_COUNT + 5)*32 +: 32];
        end
    end

    // Register the cross-group reduction before the AXI profile read mux.
    // Core snapshots update at profile_stop; the host reads only after the
    // subsequent state-clear interval, so this pipeline stage is settled.
    always @(posedge clk_100mhz) begin
        if (!rst_n_sync)
            basic_drop_spike_snapshot <= 32'd0;
        else
            basic_drop_spike_snapshot <= basic_drop_spike_sum;
    end

    always @(*) begin
        input_source_pixel_id = 0;
        profile_group_event_valid_comb = 1'b0;
        profile_group_event_group_comb = {GROUP_ID_WIDTH{1'b0}};
        profile_group_event_neuron_comb = {LOCAL_ID_WIDTH{1'b0}};
        profile_group_event_class_valid_comb = 1'b0;
        profile_group_event_class_id_comb = 4'd0;
        profile_group_event_input_source_comb = 1'b0;
        profile_group_event_pixel_comb = {PROFILE_INPUT_SOURCE_PIXEL_WIDTH{1'b0}};
        profile_fanout_input_source_valid_comb = 1'b0;
        profile_fanout_input_source_pixel_comb = {PROFILE_INPUT_SOURCE_PIXEL_WIDTH{1'b0}};

        if (PROFILE_CLASS_ENABLE || ENABLE_PROFILE_PER_GROUP) begin
            for (class_count_gi = 0; class_count_gi < NUM_GROUPS; class_count_gi = class_count_gi + 1) begin
                if (!profile_group_event_valid_comb &&
                    grp_spike_valid[class_count_gi] &&
                    grp_spike_ready[class_count_gi]) begin
                    profile_group_event_valid_comb = 1'b1;
                    profile_group_event_group_comb = class_count_gi[GROUP_ID_WIDTH-1:0];
                    profile_group_event_neuron_comb =
                        grp_spike_neuron_id[class_count_gi*LOCAL_ID_WIDTH +: LOCAL_ID_WIDTH];
                    profile_group_event_class_valid_comb = is_classifier_global_id(
                        class_count_gi[GROUP_ID_WIDTH-1:0],
                        grp_spike_neuron_id[class_count_gi*LOCAL_ID_WIDTH +: LOCAL_ID_WIDTH]
                    );
                    profile_group_event_class_id_comb = classifier_class_id(
                        class_count_gi[GROUP_ID_WIDTH-1:0],
                        grp_spike_neuron_id[class_count_gi*LOCAL_ID_WIDTH +: LOCAL_ID_WIDTH]
                    );
                    profile_group_event_input_source_comb = is_input_source_global_id(
                        class_count_gi[GROUP_ID_WIDTH-1:0],
                        grp_spike_neuron_id[class_count_gi*LOCAL_ID_WIDTH +: LOCAL_ID_WIDTH]
                    );
                    input_source_pixel_id =
                        (grp_spike_neuron_id[class_count_gi*LOCAL_ID_WIDTH +: LOCAL_ID_WIDTH] *
                         NUM_GROUPS + class_count_gi) - CLASSIFIER_NEURONS;
                    if (input_source_pixel_id >= 0 && input_source_pixel_id < INPUT_SOURCE_NEURONS)
                        profile_group_event_pixel_comb =
                            input_source_pixel_id[PROFILE_INPUT_SOURCE_PIXEL_WIDTH-1:0];
                end
            end
        end

        if (ENABLE_PROFILE_DETAIL && router_profile_fanout_valid &&
            is_input_source_global_id(
                router_profile_fanout_src_group,
                router_profile_fanout_src_neuron
            )) begin
            input_source_pixel_id =
                (router_profile_fanout_src_neuron * NUM_GROUPS +
                 router_profile_fanout_src_group) - CLASSIFIER_NEURONS;
            if (input_source_pixel_id >= 0 && input_source_pixel_id < INPUT_SOURCE_NEURONS) begin
                profile_fanout_input_source_valid_comb = 1'b1;
                profile_fanout_input_source_pixel_comb =
                    input_source_pixel_id[PROFILE_INPUT_SOURCE_PIXEL_WIDTH-1:0];
            end
        end
    end

    task automatic bump_class_count;
        input [3:0] class_id;
        begin
            case (class_id)
                4'd0: classifier_class_count_live[0] <= classifier_class_count_live[0] + 1'b1;
                4'd1: classifier_class_count_live[1] <= classifier_class_count_live[1] + 1'b1;
                4'd2: classifier_class_count_live[2] <= classifier_class_count_live[2] + 1'b1;
                4'd3: classifier_class_count_live[3] <= classifier_class_count_live[3] + 1'b1;
                4'd4: classifier_class_count_live[4] <= classifier_class_count_live[4] + 1'b1;
                4'd5: classifier_class_count_live[5] <= classifier_class_count_live[5] + 1'b1;
                4'd6: classifier_class_count_live[6] <= classifier_class_count_live[6] + 1'b1;
                4'd7: classifier_class_count_live[7] <= classifier_class_count_live[7] + 1'b1;
                4'd8: classifier_class_count_live[8] <= classifier_class_count_live[8] + 1'b1;
                4'd9: classifier_class_count_live[9] <= classifier_class_count_live[9] + 1'b1;
                default: begin
                end
            endcase
        end
    endtask

    task automatic bump_class_score_event;
        input [3:0] class_id;
        input [WEIGHT_WIDTH-1:0] weight;
        begin
            case (class_id)
                4'd0: begin
                    classifier_class_score_live[0] <= classifier_class_score_live[0] + weight;
                    classifier_class_event_live[0] <= classifier_class_event_live[0] + 1'b1;
                end
                4'd1: begin
                    classifier_class_score_live[1] <= classifier_class_score_live[1] + weight;
                    classifier_class_event_live[1] <= classifier_class_event_live[1] + 1'b1;
                end
                4'd2: begin
                    classifier_class_score_live[2] <= classifier_class_score_live[2] + weight;
                    classifier_class_event_live[2] <= classifier_class_event_live[2] + 1'b1;
                end
                4'd3: begin
                    classifier_class_score_live[3] <= classifier_class_score_live[3] + weight;
                    classifier_class_event_live[3] <= classifier_class_event_live[3] + 1'b1;
                end
                4'd4: begin
                    classifier_class_score_live[4] <= classifier_class_score_live[4] + weight;
                    classifier_class_event_live[4] <= classifier_class_event_live[4] + 1'b1;
                end
                4'd5: begin
                    classifier_class_score_live[5] <= classifier_class_score_live[5] + weight;
                    classifier_class_event_live[5] <= classifier_class_event_live[5] + 1'b1;
                end
                4'd6: begin
                    classifier_class_score_live[6] <= classifier_class_score_live[6] + weight;
                    classifier_class_event_live[6] <= classifier_class_event_live[6] + 1'b1;
                end
                4'd7: begin
                    classifier_class_score_live[7] <= classifier_class_score_live[7] + weight;
                    classifier_class_event_live[7] <= classifier_class_event_live[7] + 1'b1;
                end
                4'd8: begin
                    classifier_class_score_live[8] <= classifier_class_score_live[8] + weight;
                    classifier_class_event_live[8] <= classifier_class_event_live[8] + 1'b1;
                end
                4'd9: begin
                    classifier_class_score_live[9] <= classifier_class_score_live[9] + weight;
                    classifier_class_event_live[9] <= classifier_class_event_live[9] + 1'b1;
                end
                default: begin
                end
            endcase
        end
    endtask

    assign cfg_profile_info = {PROFILE_VERSION_INFO, PROFILE_NUM_GROUPS_INFO, PROFILE_TOTAL_COUNT_INFO};

    always @(posedge clk_100mhz) begin
        if (!rst_n_sync || hls_snn_reset) begin
            profile_active          <= 1'b0;
            profile_done            <= 1'b0;
            profile_stop_local      <= 1'b0;
            sample_clear_pending    <= 1'b0;
            total_latency_live      <= 32'd0;
            total_latency_snapshot  <= 32'd0;
            first_spike_latency_live     <= 32'd0;
            first_spike_latency_snapshot <= 32'd0;
            service_cycles_live          <= 32'd0;
            service_cycles_snapshot      <= 32'd0;
            sample_input_count           <= 16'd0;
            sample_input_done            <= 1'b0;
            sample_service_done          <= 1'b0;
            sample_seen_input            <= 1'b0;
            sample_seen_output           <= 1'b0;
            service_dbg_input_done_cycle_live <= 32'd0;
            service_dbg_input_done_cycle_snapshot <= 32'd0;
            service_dbg_hls_pending_clear_cycle_live <= 32'd0;
            service_dbg_hls_pending_clear_cycle_snapshot <= 32'd0;
            service_dbg_input_done_seen <= 1'b0;
            service_dbg_hls_pending_clear_seen <= 1'b0;
            first_classifier_spike_valid_live     <= 1'b0;
            first_classifier_spike_valid_snapshot <= 1'b0;
            first_classifier_spike_id_live        <= {GLOBAL_ID_WIDTH{1'b0}};
            first_classifier_spike_id_snapshot    <= {GLOBAL_ID_WIDTH{1'b0}};
            first_classifier_spike_cycle_live     <= 32'd0;
            first_classifier_spike_cycle_snapshot <= 32'd0;
            for (class_count_i = 0; class_count_i < NUM_CLASSES; class_count_i = class_count_i + 1) begin
                classifier_class_count_live[class_count_i] <= 32'd0;
                classifier_class_count_snapshot[class_count_i] <= 32'd0;
                classifier_class_score_live[class_count_i] <= 32'd0;
                classifier_class_score_snapshot[class_count_i] <= 32'd0;
                classifier_class_event_live[class_count_i] <= 32'd0;
                classifier_class_event_snapshot[class_count_i] <= 32'd0;
            end
            for (class_count_i = 0; class_count_i < NUM_GROUPS; class_count_i = class_count_i + 1) begin
                input_source_group_count_live[class_count_i] <= 32'd0;
                input_source_group_count_snapshot[class_count_i] <= 32'd0;
            end
            input_source_bitmap_live <= {PROFILE_INPUT_SOURCE_BITMAP_BITS{1'b0}};
            input_source_bitmap_snapshot <= {PROFILE_INPUT_SOURCE_BITMAP_BITS{1'b0}};
            input_source_ct_bitmap_live <= {PROFILE_INPUT_SOURCE_BITMAP_BITS{1'b0}};
            input_source_ct_bitmap_snapshot <= {PROFILE_INPUT_SOURCE_BITMAP_BITS{1'b0}};
            profile_group_event_valid_d <= 1'b0;
            profile_group_event_group_d <= {GROUP_ID_WIDTH{1'b0}};
            profile_group_event_class_valid_d <= 1'b0;
            profile_group_event_class_id_d <= 4'd0;
            profile_group_event_input_source_d <= 1'b0;
            profile_group_event_pixel_d <= {PROFILE_INPUT_SOURCE_PIXEL_WIDTH{1'b0}};
            profile_class_valid_d <= 1'b0;
            profile_class_id_d <= 4'd0;
            profile_class_weight_d <= {WEIGHT_WIDTH{1'b0}};
            profile_fanout_input_source_valid_d <= 1'b0;
            profile_fanout_input_source_pixel_d <= {PROFILE_INPUT_SOURCE_PIXEL_WIDTH{1'b0}};
        end else begin
            profile_stop_local <= cfg_profile_stop;
            if (cfg_profile_stop)
                profile_done <= 1'b0;
            if (cfg_profile_start) begin
                profile_active               <= 1'b1;
                profile_done                 <= 1'b0;
                profile_stop_local           <= 1'b0;
                sample_clear_pending         <= 1'b0;
                total_latency_live           <= 32'd0;
                first_spike_latency_live     <= 32'd0;
                first_spike_latency_snapshot <= 32'd0;
                service_cycles_live          <= 32'd0;
                service_cycles_snapshot      <= 32'd0;
                sample_input_count           <= 16'd0;
                sample_input_done            <= (cfg_profile_expected_count == 16'd0);
                sample_service_done          <= 1'b0;
                sample_seen_input            <= 1'b0;
                sample_seen_output           <= 1'b0;
                service_dbg_input_done_cycle_live <= 32'd0;
                service_dbg_input_done_cycle_snapshot <= 32'd0;
                service_dbg_hls_pending_clear_cycle_live <= 32'd0;
                service_dbg_hls_pending_clear_cycle_snapshot <= 32'd0;
                service_dbg_input_done_seen <= (cfg_profile_expected_count == 16'd0);
                service_dbg_hls_pending_clear_seen <= 1'b0;
                first_classifier_spike_valid_live     <= 1'b0;
                first_classifier_spike_valid_snapshot <= 1'b0;
                first_classifier_spike_id_live        <= {GLOBAL_ID_WIDTH{1'b0}};
                first_classifier_spike_id_snapshot    <= {GLOBAL_ID_WIDTH{1'b0}};
                first_classifier_spike_cycle_live     <= 32'd0;
                first_classifier_spike_cycle_snapshot <= 32'd0;
                for (class_count_i = 0; class_count_i < NUM_CLASSES; class_count_i = class_count_i + 1) begin
                    classifier_class_count_live[class_count_i] <= 32'd0;
                    classifier_class_count_snapshot[class_count_i] <= 32'd0;
                    classifier_class_score_live[class_count_i] <= 32'd0;
                    classifier_class_score_snapshot[class_count_i] <= 32'd0;
                    classifier_class_event_live[class_count_i] <= 32'd0;
                    classifier_class_event_snapshot[class_count_i] <= 32'd0;
                end
                for (class_count_i = 0; class_count_i < NUM_GROUPS; class_count_i = class_count_i + 1) begin
                    input_source_group_count_live[class_count_i] <= 32'd0;
                    input_source_group_count_snapshot[class_count_i] <= 32'd0;
                end
                input_source_bitmap_live <= {PROFILE_INPUT_SOURCE_BITMAP_BITS{1'b0}};
                input_source_bitmap_snapshot <= {PROFILE_INPUT_SOURCE_BITMAP_BITS{1'b0}};
                input_source_ct_bitmap_live <= {PROFILE_INPUT_SOURCE_BITMAP_BITS{1'b0}};
                input_source_ct_bitmap_snapshot <= {PROFILE_INPUT_SOURCE_BITMAP_BITS{1'b0}};
                profile_group_event_valid_d <= 1'b0;
                profile_group_event_group_d <= {GROUP_ID_WIDTH{1'b0}};
                profile_group_event_class_valid_d <= 1'b0;
                profile_group_event_class_id_d <= 4'd0;
                profile_group_event_input_source_d <= 1'b0;
                profile_group_event_pixel_d <= {PROFILE_INPUT_SOURCE_PIXEL_WIDTH{1'b0}};
                profile_class_valid_d <= 1'b0;
                profile_class_id_d <= 4'd0;
                profile_class_weight_d <= {WEIGHT_WIDTH{1'b0}};
                profile_fanout_input_source_valid_d <= 1'b0;
                profile_fanout_input_source_pixel_d <= {PROFILE_INPUT_SOURCE_PIXEL_WIDTH{1'b0}};
            end else if (profile_active) begin
                total_latency_live <= total_latency_live + 1'b1;
                if (hls_spike_out_pulse && !sample_seen_input) begin
                    sample_seen_input <= 1'b1;
                end
                if (hls_spike_out_pulse && cfg_profile_expected_count != 16'd0) begin
                    sample_input_count <= sample_input_count + 1'b1;
                    if (sample_input_count + 1'b1 >= cfg_profile_expected_count)
                        sample_input_done <= 1'b1;
                end
                if (sample_seen_input) begin
                    if (!sample_service_done)
                        service_cycles_live <= service_cycles_live + 1'b1;
                    if (!sample_seen_output)
                        first_spike_latency_live <= first_spike_latency_live + 1'b1;
                    if (!first_classifier_spike_valid_live)
                        first_classifier_spike_cycle_live <= first_classifier_spike_cycle_live + 1'b1;
                end
                if (sample_seen_input && !service_dbg_input_done_seen &&
                    (sample_input_done ||
                     (hls_spike_out_pulse && cfg_profile_expected_count != 16'd0 &&
                      (sample_input_count + 1'b1 >= cfg_profile_expected_count)))) begin
                    service_dbg_input_done_seen <= 1'b1;
                    service_dbg_input_done_cycle_live <= service_cycles_live;
                end
                if (sample_seen_input && sample_input_done &&
                    !service_dbg_hls_pending_clear_seen && !hls_ext_pending) begin
                    service_dbg_hls_pending_clear_seen <= 1'b1;
                    service_dbg_hls_pending_clear_cycle_live <= service_cycles_live;
                end
                if (sample_seen_input && !sample_seen_output && rtl_spike_out_valid) begin
                    sample_seen_output <= 1'b1;
                    first_spike_latency_snapshot <= first_spike_latency_live;
                end
                if (ENABLE_PROFILE_DETAIL &&
                    sample_seen_input && !first_classifier_spike_valid_live && first_classifier_spike_seen) begin
                    first_classifier_spike_valid_live <= 1'b1;
                    first_classifier_spike_id_live    <= first_classifier_spike_candidate;
                    first_classifier_spike_cycle_snapshot <= first_classifier_spike_cycle_live;
                end

                if (PROFILE_CLASS_ENABLE &&
                    profile_group_event_valid_d && profile_group_event_class_valid_d)
                    bump_class_count(profile_group_event_class_id_d);

                if (PROFILE_CLASS_ENABLE && profile_class_valid_d)
                    bump_class_score_event(profile_class_id_d, profile_class_weight_d);

                if (ENABLE_PROFILE_PER_GROUP &&
                    profile_group_event_valid_d && profile_group_event_input_source_d) begin
                    case (profile_group_event_group_d)
                        4'd0:  input_source_group_count_live[0]  <= input_source_group_count_live[0] + 1'b1;
                        4'd1:  input_source_group_count_live[1]  <= input_source_group_count_live[1] + 1'b1;
                        4'd2:  input_source_group_count_live[2]  <= input_source_group_count_live[2] + 1'b1;
                        4'd3:  input_source_group_count_live[3]  <= input_source_group_count_live[3] + 1'b1;
                        4'd4:  input_source_group_count_live[4]  <= input_source_group_count_live[4] + 1'b1;
                        4'd5:  input_source_group_count_live[5]  <= input_source_group_count_live[5] + 1'b1;
                        4'd6:  input_source_group_count_live[6]  <= input_source_group_count_live[6] + 1'b1;
                        4'd7:  input_source_group_count_live[7]  <= input_source_group_count_live[7] + 1'b1;
                        4'd8:  input_source_group_count_live[8]  <= input_source_group_count_live[8] + 1'b1;
                        4'd9:  input_source_group_count_live[9]  <= input_source_group_count_live[9] + 1'b1;
                        4'd10: input_source_group_count_live[10] <= input_source_group_count_live[10] + 1'b1;
                        4'd11: input_source_group_count_live[11] <= input_source_group_count_live[11] + 1'b1;
                        4'd12: input_source_group_count_live[12] <= input_source_group_count_live[12] + 1'b1;
                        4'd13: input_source_group_count_live[13] <= input_source_group_count_live[13] + 1'b1;
                        4'd14: input_source_group_count_live[14] <= input_source_group_count_live[14] + 1'b1;
                        4'd15: input_source_group_count_live[15] <= input_source_group_count_live[15] + 1'b1;
                        default: begin
                        end
                    endcase
                    input_source_bitmap_live[profile_group_event_pixel_d] <= 1'b1;
                end

                if (ENABLE_PROFILE_DETAIL && profile_fanout_input_source_valid_d)
                    input_source_ct_bitmap_live[profile_fanout_input_source_pixel_d] <= 1'b1;

                if (PROFILE_CLASS_ENABLE || ENABLE_PROFILE_PER_GROUP) begin
                    profile_group_event_valid_d <= profile_group_event_valid_comb;
                    profile_group_event_group_d <= profile_group_event_group_comb;
                    profile_group_event_class_valid_d <= profile_group_event_class_valid_comb;
                    profile_group_event_class_id_d <= profile_group_event_class_id_comb;
                    profile_group_event_input_source_d <= profile_group_event_input_source_comb;
                    profile_group_event_pixel_d <= profile_group_event_pixel_comb;
                end else begin
                    profile_group_event_valid_d <= 1'b0;
                    profile_group_event_group_d <= {GROUP_ID_WIDTH{1'b0}};
                    profile_group_event_class_valid_d <= 1'b0;
                    profile_group_event_class_id_d <= 4'd0;
                    profile_group_event_input_source_d <= 1'b0;
                    profile_group_event_pixel_d <= {PROFILE_INPUT_SOURCE_PIXEL_WIDTH{1'b0}};
                end
                if (PROFILE_CLASS_ENABLE) begin
                    profile_class_valid_d <= router_profile_class_valid;
                    profile_class_id_d <= router_profile_class_id;
                    profile_class_weight_d <= router_profile_class_weight;
                    profile_fanout_input_source_valid_d <= profile_fanout_input_source_valid_comb;
                    profile_fanout_input_source_pixel_d <= profile_fanout_input_source_pixel_comb;
                end else begin
                    profile_class_valid_d <= 1'b0;
                    profile_class_id_d <= 4'd0;
                    profile_class_weight_d <= {WEIGHT_WIDTH{1'b0}};
                    profile_fanout_input_source_valid_d <= 1'b0;
                    profile_fanout_input_source_pixel_d <= {PROFILE_INPUT_SOURCE_PIXEL_WIDTH{1'b0}};
                end

                if (sample_seen_input && !sample_service_done && sample_input_done &&
                    !hls_ext_pending && !router_busy && (grp_busy == {NUM_GROUPS{1'b0}})) begin
                    sample_service_done <= 1'b1;
                    service_cycles_snapshot <= service_cycles_live;
                end
            end
            if (profile_stop_local) begin
                profile_active         <= 1'b0;
                profile_done           <= 1'b0;
                sample_clear_pending   <= 1'b1;
                total_latency_snapshot <= total_latency_live;
                if (!sample_service_done)
                    service_cycles_snapshot <= service_cycles_live;
                service_dbg_input_done_cycle_snapshot <= service_dbg_input_done_cycle_live;
                service_dbg_hls_pending_clear_cycle_snapshot <= service_dbg_hls_pending_clear_cycle_live;
                if (PROFILE_CLASS_ENABLE) begin
                    first_classifier_spike_valid_snapshot <= first_classifier_spike_valid_live;
                    first_classifier_spike_id_snapshot    <= first_classifier_spike_id_live;
                    if (!first_classifier_spike_valid_live)
                        first_classifier_spike_cycle_snapshot <= first_classifier_spike_cycle_live;
                    for (class_count_i = 0; class_count_i < NUM_CLASSES; class_count_i = class_count_i + 1)
                        classifier_class_count_snapshot[class_count_i] <= classifier_class_count_live[class_count_i];
                    for (class_count_i = 0; class_count_i < NUM_CLASSES; class_count_i = class_count_i + 1) begin
                        classifier_class_score_snapshot[class_count_i] <= bp_score_mode ?
                            bp_score_aggregate[class_count_i] :
                            classifier_class_score_live[class_count_i];
                        classifier_class_event_snapshot[class_count_i] <= classifier_class_event_live[class_count_i];
                    end
                end
                if (ENABLE_PROFILE_PER_GROUP) begin
                    for (class_count_i = 0; class_count_i < NUM_GROUPS; class_count_i = class_count_i + 1)
                        input_source_group_count_snapshot[class_count_i] <= input_source_group_count_live[class_count_i];
                    input_source_bitmap_snapshot <= input_source_bitmap_live;
                end
                if (ENABLE_PROFILE_DETAIL)
                    input_source_ct_bitmap_snapshot <= input_source_ct_bitmap_live;
            end
            if (sample_clear_pending && (&grp_sample_clear_done)) begin
                sample_clear_pending <= 1'b0;
                profile_done         <= 1'b1;
            end
        end
    end

    always @(*) begin
        cfg_profile_data = 32'd0;
        profile_sel = cfg_profile_index;
        if (PROFILE_BASIC_ONLY) begin
            case (profile_sel)
                0: cfg_profile_data = total_latency_snapshot;
                1: cfg_profile_data = router_profile_snapshot[2*32 +: 32]; // ct_lookup_count
                2: cfg_profile_data = router_profile_snapshot[3*32 +: 32]; // ct_valid_entry_count
                3: cfg_profile_data = router_profile_snapshot[5*32 +: 32]; // router_busy_cycles
                4: cfg_profile_data = router_profile_snapshot[7*32 +: 32]; // router_stall_cycles
                5: cfg_profile_data = router_profile_snapshot[8*32 +: 32]; // cross_group_event_count
                6: cfg_profile_data = router_profile_snapshot[9*32 +: 32]; // same_group_event_count
                7: cfg_profile_data = basic_drop_spike_snapshot;
                8: cfg_profile_data = service_dbg_input_done_cycle_snapshot;
                9: cfg_profile_data = service_dbg_hls_pending_clear_cycle_snapshot;
                default: cfg_profile_data = 32'd0;
            endcase
            if (profile_sel >= PROFILE_BASIC_CLASS_COUNT_BASE &&
                profile_sel < PROFILE_BASIC_CLASS_SCORE_BASE)
                cfg_profile_data = classifier_class_count_snapshot[
                    profile_sel-PROFILE_BASIC_CLASS_COUNT_BASE
                ];
            else if (profile_sel >= PROFILE_BASIC_CLASS_SCORE_BASE &&
                     profile_sel < PROFILE_BASIC_CLASS_EVENT_BASE)
                cfg_profile_data = classifier_class_score_snapshot[
                    profile_sel-PROFILE_BASIC_CLASS_SCORE_BASE
                ];
            else if (profile_sel >= PROFILE_BASIC_CLASS_EVENT_BASE &&
                     profile_sel < PROFILE_BASIC_COUNT)
                cfg_profile_data = classifier_class_event_snapshot[
                    profile_sel-PROFILE_BASIC_CLASS_EVENT_BASE
                ];
        end else begin
            if (profile_sel == 10)
                cfg_profile_data = total_latency_snapshot;
            else if (profile_sel < PROFILE_ROUTER_TOTAL_COUNT)
                cfg_profile_data = router_profile_snapshot[profile_sel*32 +: 32];
            else if (profile_sel < PROFILE_TOTAL_COUNT)
                if (profile_sel < PROFILE_CLASSIFIER_BASE)
                    cfg_profile_data = grp_profile_snapshot[(profile_sel-PROFILE_CORE_BASE)*32 +: 32];
                else if (profile_sel == PROFILE_CLASSIFIER_BASE)
                    cfg_profile_data = {31'd0, first_classifier_spike_valid_snapshot};
                else if (profile_sel == PROFILE_CLASSIFIER_BASE + 1)
                    cfg_profile_data = {{(32-GLOBAL_ID_WIDTH){1'b0}}, first_classifier_spike_id_snapshot};
                else if (profile_sel == PROFILE_CLASSIFIER_BASE + 2)
                    cfg_profile_data = first_classifier_spike_cycle_snapshot;
                else if (profile_sel < PROFILE_CLASS_SCORE_BASE)
                    cfg_profile_data = classifier_class_count_snapshot[profile_sel-PROFILE_CLASS_COUNT_BASE];
                else if (profile_sel < PROFILE_CLASS_EVENT_BASE)
                    cfg_profile_data = classifier_class_score_snapshot[profile_sel-PROFILE_CLASS_SCORE_BASE];
                else if (profile_sel < PROFILE_INPUT_SOURCE_GROUP_BASE)
                    cfg_profile_data = classifier_class_event_snapshot[profile_sel-PROFILE_CLASS_EVENT_BASE];
                else if (profile_sel < PROFILE_INPUT_SOURCE_BITMAP_BASE)
                    cfg_profile_data = input_source_group_count_snapshot[profile_sel-PROFILE_INPUT_SOURCE_GROUP_BASE];
                else if (profile_sel < PROFILE_TOTAL_COUNT) begin
                    if (profile_sel < PROFILE_INPUT_SOURCE_CT_BITMAP_BASE)
                        cfg_profile_data = input_source_bitmap_snapshot[
                            (profile_sel-PROFILE_INPUT_SOURCE_BITMAP_BASE)*32 +: 32
                        ];
                    else
                        cfg_profile_data = input_source_ct_bitmap_snapshot[
                            (profile_sel-PROFILE_INPUT_SOURCE_CT_BITMAP_BASE)*32 +: 32
                        ];
                end
        end
    end

    //=========================================================================
    // Config Register Decode Logic
    //=========================================================================
    // Decode cfg_router_config for connectivity table & intra-group weights

    wire [3:0] cfg_cmd = cfg_router_config_addr[31:28];

    // Connectivity table write: cmd = 0x0
    reg                          ct_cfg_we_reg;
    reg [GROUP_ID_WIDTH-1:0]     ct_cfg_src_group_reg;
    reg [LOCAL_ID_WIDTH-1:0]     ct_cfg_src_neuron_reg;
    reg [FANOUT_IDX_WIDTH-1:0]   ct_cfg_fanout_idx_reg;
    reg                          ct_cfg_valid_reg;
    reg [GROUP_ID_WIDTH-1:0]     ct_cfg_dst_group_reg;
    reg [LOCAL_ID_WIDTH-1:0]     ct_cfg_dst_neuron_reg;
    reg [WEIGHT_WIDTH-1:0]       ct_cfg_weight_reg;
    reg                          ct_cfg_exc_inh_reg;

    // Intra-group weight write: cmd = 0x1
    reg [NUM_GROUPS-1:0]         intra_weight_we_reg;
    reg [LOCAL_ID_WIDTH-1:0]     intra_weight_src_reg;
    reg [LOCAL_ID_WIDTH-1:0]     intra_weight_dst_reg;
    reg [WEIGHT_WIDTH-1:0]       intra_weight_data_reg;
    reg                          intra_weight_exc_reg;
    reg [FANOUT_IDX_WIDTH-1:0]   intra_weight_fanout_idx_reg;

    // BP score readout configuration: cmd=0x2, one mapped output ID per class.
    reg [NUM_GROUPS-1:0]         score_cfg_we_reg;
    reg [3:0]                    score_cfg_class_reg;
    reg [LOCAL_ID_WIDTH-1:0]     score_cfg_local_id_reg;
    reg                          score_cfg_valid_reg;
    integer                      score_cfg_i;

    always @(posedge clk_100mhz) begin
        if (!rst_n_sync) begin
            ct_cfg_we_reg       <= 0;
            intra_weight_we_reg <= {NUM_GROUPS{1'b0}};
            score_cfg_we_reg    <= {NUM_GROUPS{1'b0}};
            score_cfg_class_reg <= 4'd0;
            score_cfg_local_id_reg <= {LOCAL_ID_WIDTH{1'b0}};
            score_cfg_valid_reg <= 1'b0;
            bp_score_mode       <= 1'b0;
            for (score_cfg_i = 0; score_cfg_i < NUM_CLASSES; score_cfg_i = score_cfg_i + 1)
                bp_score_group_for_class[score_cfg_i] <= {GROUP_ID_WIDTH{1'b0}};
        end else begin
            ct_cfg_we_reg       <= 0;
            intra_weight_we_reg <= {NUM_GROUPS{1'b0}};
            score_cfg_we_reg    <= {NUM_GROUPS{1'b0}};

            if (cfg_router_config_we) begin
                case (cfg_cmd)
                    4'h0: begin
                        // Connectivity table write (8-bit weight format)
                        // src_group is in address register [3:0]
                        ct_cfg_we_reg         <= 1;
                        ct_cfg_valid_reg      <= cfg_router_config_wdata[31];
                        ct_cfg_dst_group_reg  <= cfg_router_config_wdata[30:27];
                        ct_cfg_dst_neuron_reg <= cfg_router_config_wdata[26:20];
                        ct_cfg_weight_reg     <= cfg_router_config_wdata[19:12];
                        ct_cfg_exc_inh_reg    <= cfg_router_config_wdata[11];
                        ct_cfg_fanout_idx_reg <= cfg_router_config_wdata[10:7];
                        ct_cfg_src_neuron_reg <= cfg_router_config_wdata[6:0];
                        ct_cfg_src_group_reg  <= cfg_router_config_addr[3:0];
                    end
                    4'h1: begin
                        // Intra-group weight write (8-bit weight format)
                        intra_weight_src_reg  <= cfg_router_config_wdata[31:25];
                        intra_weight_dst_reg  <= cfg_router_config_wdata[24:18];
                        intra_weight_data_reg <= cfg_router_config_wdata[17:10];
                        intra_weight_exc_reg  <= cfg_router_config_wdata[9];
                        intra_weight_fanout_idx_reg <= cfg_router_config_wdata[3:0];
                        intra_weight_we_reg[cfg_router_config_wdata[8:5]] <= 1'b1;
                    end
                    4'h2: begin
                        score_cfg_valid_reg <= cfg_router_config_wdata[31];
                        score_cfg_class_reg <= cfg_router_config_wdata[27:24];
                        score_cfg_local_id_reg <= cfg_router_config_wdata[LOCAL_ID_WIDTH-1:0];
                        score_cfg_we_reg[
                            cfg_router_config_wdata[LOCAL_ID_WIDTH +: GROUP_ID_WIDTH]
                        ] <= 1'b1;
                        if (cfg_router_config_wdata[27:24] < NUM_CLASSES)
                            bp_score_group_for_class[cfg_router_config_wdata[27:24]] <=
                                cfg_router_config_wdata[LOCAL_ID_WIDTH +: GROUP_ID_WIDTH];
                        bp_score_mode <= 1'b1;
                    end
                    default: ;
                endcase
            end
        end
    end

    //=========================================================================
    // Status Readback
    //=========================================================================
    reg [31:0] total_neuron_spikes;

    // Parametric spike summation: returns 0 for out-of-range group indices
    function [31:0] safe_spike_count;
        input integer idx;
        begin
            if (idx < NUM_GROUPS)
                safe_spike_count = grp_spike_count[32*idx +: 32];
            else
                safe_spike_count = 32'd0;
        end
    endfunction

    // Sum spike counts from all groups (pipelined for timing)
    reg [31:0] spike_sum_stage1 [0:3];  // 4 partial sums of up to 4 groups each
    integer si;
    always @(posedge clk_100mhz) begin
        if (!rst_n_sync || hls_snn_reset) begin
            for (si = 0; si < 4; si = si + 1)
                spike_sum_stage1[si] <= 0;
            total_neuron_spikes <= 0;
        end else begin
            // Stage 1: Sum up to 4 groups each (safe for NUM_GROUPS < 16)
            spike_sum_stage1[0] <= safe_spike_count(0)  + safe_spike_count(1)  +
                                   safe_spike_count(2)  + safe_spike_count(3);
            spike_sum_stage1[1] <= safe_spike_count(4)  + safe_spike_count(5)  +
                                   safe_spike_count(6)  + safe_spike_count(7);
            spike_sum_stage1[2] <= safe_spike_count(8)  + safe_spike_count(9)  +
                                   safe_spike_count(10) + safe_spike_count(11);
            spike_sum_stage1[3] <= safe_spike_count(12) + safe_spike_count(13) +
                                   safe_spike_count(14) + safe_spike_count(15);
            // Stage 2: Final sum
            total_neuron_spikes <= spike_sum_stage1[0] + spike_sum_stage1[1] +
                                   spike_sum_stage1[2] + spike_sum_stage1[3];
        end
    end

    assign cfg_router_config_rdata = (cfg_cmd == 4'h2) ? routed_spike_count :
                                     (cfg_cmd == 4'h3) ? total_neuron_spikes :
                                     32'hDEAD_BEEF;

    //=========================================================================
    // Block Design Instantiation (PS + HLS IP + AXI + Config Regs)
    //=========================================================================

    design_1_wrapper u_block_design (
        // DDR Interface
        .DDR_addr           (DDR_addr),
        .DDR_ba             (DDR_ba),
        .DDR_cas_n          (DDR_cas_n),
        .DDR_ck_n           (DDR_ck_n),
        .DDR_ck_p           (DDR_ck_p),
        .DDR_cke            (DDR_cke),
        .DDR_cs_n           (DDR_cs_n),
        .DDR_dm             (DDR_dm),
        .DDR_dq             (DDR_dq),
        .DDR_dqs_n          (DDR_dqs_n),
        .DDR_dqs_p          (DDR_dqs_p),
        .DDR_odt            (DDR_odt),
        .DDR_ras_n          (DDR_ras_n),
        .DDR_reset_n        (DDR_reset_n),
        .DDR_we_n           (DDR_we_n),

        // Fixed IO
        .FIXED_IO_ddr_vrn   (FIXED_IO_ddr_vrn),
        .FIXED_IO_ddr_vrp   (FIXED_IO_ddr_vrp),
        .FIXED_IO_mio       (FIXED_IO_mio),
        .FIXED_IO_ps_clk    (FIXED_IO_ps_clk),
        .FIXED_IO_ps_porb   (FIXED_IO_ps_porb),
        .FIXED_IO_ps_srstb  (FIXED_IO_ps_srstb),

        // PL Clock/Reset
        .clk_100mhz         (clk_100mhz),
        .rst_n_sync          (rst_n_sync),

        // Debug
        .debug_learning_active (debug_learning_active),

        // HLS → RTL Spike Interface
        .spike_in_valid          (hls_spike_out_valid),
        .spike_in_neuron_id      (hls_spike_out_neuron_id),
        .spike_in_weight         (hls_spike_out_weight),
        .spike_in_ready          (rtl_spike_in_ready),

        // RTL → HLS Spike Interface
        .spike_out_valid         (rtl_spike_out_valid),
        .spike_out_neuron_id     (rtl_spike_out_neuron_id),
        .spike_out_weight        (rtl_spike_out_weight),
        .spike_out_ready         (hls_spike_in_ready),

        // SNN Control
        .snn_enable              (hls_snn_enable),
        .snn_reset               (hls_snn_reset),
        .snn_ready               (rtl_snn_ready),
        .snn_busy                (rtl_snn_busy),

        // HLS Neuron Parameters
        .threshold_out           (hls_threshold_out),
        .leak_rate_out           (hls_leak_rate_out),

        // Config Registers
        .cfg_router_config_we    (cfg_router_config_we),
        .cfg_router_config_addr  (cfg_router_config_addr),
        .cfg_router_config_wdata (cfg_router_config_wdata),
        .cfg_router_config_rdata (cfg_router_config_rdata),
        .cfg_neuron_config_we    (cfg_neuron_config_we),
        .cfg_neuron_config_addr  (cfg_neuron_config_addr),
        .cfg_neuron_config_wdata (cfg_neuron_config_wdata),
        .cfg_global_threshold    (cfg_global_threshold),
        .cfg_global_leak_rate    (cfg_global_leak_rate),
        .cfg_global_refrac_period(cfg_global_refrac_period),

        // Status
        .cfg_router_spike_count  (routed_spike_count),
        .cfg_neuron_spike_count  (total_neuron_spikes),
        .cfg_fifo_overflow       (1'b0),  // TODO: aggregate from groups
        .cfg_active_neurons      (8'd0),  // TODO: aggregate
        .cfg_throughput_counter  (first_spike_latency_snapshot),
        .cfg_service_cycles_counter(service_cycles_snapshot),
        .cfg_router_busy        (router_busy),
        .cfg_any_core_group_busy(|grp_busy),
        .cfg_snn_ready          (rtl_snn_ready),
        .cfg_profile_active     (profile_active),
        .cfg_profile_done       (profile_done),
        .cfg_profile_start      (cfg_profile_start),
        .cfg_profile_stop       (cfg_profile_stop),
        .cfg_profile_expected_count(cfg_profile_expected_count),
        .cfg_profile_index      (cfg_profile_index),
        .cfg_profile_data       (cfg_profile_data),
        .cfg_profile_info       (cfg_profile_info)
    );

    //=========================================================================
    // HLS ↔ Event Router Bridge
    //=========================================================================

    // HLS spike output → Event Router external input
    // Convert HLS neuron ID to global format {group_id[3:0], local_id[6:0]}
    wire [GLOBAL_ID_WIDTH-1:0]   hls_global_id;
    wire [WEIGHT_WIDTH-1:0]      hls_weight_truncated;

    assign hls_global_id       = hls_spike_out_neuron_id[GLOBAL_ID_WIDTH-1:0];
    assign hls_weight_truncated = hls_spike_out_weight[WEIGHT_WIDTH-1:0];

    always @(posedge clk_100mhz) begin
        if (!rst_n_sync || hls_snn_reset)
            hls_spike_out_toggle_d <= 1'b0;
        else
            hls_spike_out_toggle_d <= hls_spike_out_valid;
    end

    assign hls_spike_out_pulse = hls_spike_out_valid ^ hls_spike_out_toggle_d;

    assign ext_input_fifo_empty = (ext_input_fifo_count == {(EXT_INPUT_FIFO_AW+1){1'b0}});
    assign ext_input_fifo_full  = (ext_input_fifo_count == EXT_INPUT_FIFO_DEPTH_COUNT);
    assign ext_input_fifo_push  = hls_spike_out_pulse && !ext_input_fifo_full;
    assign ext_input_fifo_pop   = !ext_input_fifo_empty && router_ext_spike_ready;

    always @(posedge clk_100mhz) begin
        if (!rst_n_sync || hls_snn_reset) begin
            ext_input_fifo_wr_ptr <= {EXT_INPUT_FIFO_AW{1'b0}};
            ext_input_fifo_rd_ptr <= {EXT_INPUT_FIFO_AW{1'b0}};
            ext_input_fifo_count  <= {(EXT_INPUT_FIFO_AW+1){1'b0}};
            for (ext_fifo_i = 0; ext_fifo_i < EXT_INPUT_FIFO_DEPTH; ext_fifo_i = ext_fifo_i + 1) begin
                ext_input_fifo_id[ext_fifo_i]     <= {GLOBAL_ID_WIDTH{1'b0}};
                ext_input_fifo_weight[ext_fifo_i] <= {WEIGHT_WIDTH{1'b0}};
            end
        end else begin
            if (ext_input_fifo_push) begin
                ext_input_fifo_id[ext_input_fifo_wr_ptr]     <= hls_global_id;
                ext_input_fifo_weight[ext_input_fifo_wr_ptr] <= hls_weight_truncated;
                ext_input_fifo_wr_ptr <= ext_input_fifo_wr_ptr + 1'b1;
            end

            if (ext_input_fifo_pop)
                ext_input_fifo_rd_ptr <= ext_input_fifo_rd_ptr + 1'b1;

            case ({ext_input_fifo_push, ext_input_fifo_pop})
                2'b10: ext_input_fifo_count <= ext_input_fifo_count + 1'b1;
                2'b01: ext_input_fifo_count <= ext_input_fifo_count - 1'b1;
                default: begin
                end
            endcase
        end
    end

    assign hls_ext_pending        = !ext_input_fifo_empty;
    assign hls_ext_pending_id     = ext_input_fifo_id[ext_input_fifo_rd_ptr];
    assign hls_ext_pending_weight = ext_input_fifo_weight[ext_input_fifo_rd_ptr];

    // HLS ready/busy
    assign rtl_spike_in_ready = !ext_input_fifo_full;

    // Event Router / first-spike tap → HLS observation
    // The tap has priority only while it holds the first group output event.
    assign rtl_spike_out_valid     = first_spike_tap_valid ? 1'b1 : learn_spike_valid;
    assign rtl_spike_out_neuron_id = first_spike_tap_valid ? first_spike_tap_id : learn_spike_src_id;
    assign rtl_learn_weight_ready  = LEARN_WEIGHT_BRIDGE_ENABLE ? learn_weight_ready_br : 1'b0;

    // Weight bridge: zero-extend if WEIGHT_WIDTH < HLS_WEIGHT_WIDTH, else direct
    generate
        if (HLS_WEIGHT_WIDTH > WEIGHT_WIDTH)
            assign rtl_spike_out_weight = first_spike_tap_valid ? first_spike_tap_weight :
                                          {{(HLS_WEIGHT_WIDTH-WEIGHT_WIDTH){1'b0}},
                                           ct_result_weight};
        else
            assign rtl_spike_out_weight = first_spike_tap_valid ? first_spike_tap_weight :
                                          ct_result_weight[HLS_WEIGHT_WIDTH-1:0];
    endgenerate

    assign learn_spike_ready       = first_spike_tap_valid ? 1'b0 : hls_spike_in_ready;

    // A sample is drained only after the HLS-to-router staging FIFO is empty as
    // well as the router and every core group. Omitting hls_ext_pending lets a
    // tail event from the previous image arrive after its state-clear sweep.
    assign rtl_snn_ready      = !hls_ext_pending && !router_busy &&
                                (grp_busy == {NUM_GROUPS{1'b0}});
    assign rtl_snn_busy       = hls_ext_pending || router_busy ||
                                (grp_busy != {NUM_GROUPS{1'b0}});

    //=========================================================================
    // Core Group Instantiations
    //=========================================================================
    // Each group has its own NEURONS_PER_GROUP from SNN_GROUP_SIZE_x defines.
    // All groups share the same LOCAL_ID_WIDTH (= clog2(max(group_sizes)))
    // for uniform interconnect bus widths.
    //
    // Default: 16 × 128 = 2048 neurons (all groups identical).
    // Variable: configure group_sizes in snn_params.yaml for mixed sizes.
    //=========================================================================

    // Mux write enable: AXI config writes OR event router weight updates
    wire [NUM_GROUPS-1:0]     combined_weight_we;
    wire [LOCAL_ID_WIDTH-1:0] combined_weight_src [0:NUM_GROUPS-1];
    wire [LOCAL_ID_WIDTH-1:0] combined_weight_dst [0:NUM_GROUPS-1];
    wire [FANOUT_IDX_WIDTH-1:0] combined_weight_fanout_idx [0:NUM_GROUPS-1];
    wire [WEIGHT_WIDTH-1:0]   combined_weight_data[0:NUM_GROUPS-1];
    wire                      combined_weight_exc [0:NUM_GROUPS-1];

    genvar g;
    generate
        for (g = 0; g < NUM_GROUPS; g = g + 1) begin : gen_weight_mux
            // AXI config has priority over router weight updates
            assign combined_weight_we[g]   = intra_weight_we_reg[g] | grp_weight_we[g];
            assign combined_weight_src[g]  = intra_weight_we_reg[g] ?
                                             intra_weight_src_reg : grp_weight_src;
            assign combined_weight_dst[g]  = intra_weight_we_reg[g] ?
                                             intra_weight_dst_reg : grp_weight_dst;
            assign combined_weight_fanout_idx[g] = intra_weight_we_reg[g] ?
                                             intra_weight_fanout_idx_reg :
                                             grp_weight_dst[FANOUT_IDX_WIDTH-1:0];
            assign combined_weight_data[g] = intra_weight_we_reg[g] ?
                                             intra_weight_data_reg : grp_weight_data;
            assign combined_weight_exc[g]  = intra_weight_we_reg[g] ?
                                             intra_weight_exc_reg : grp_weight_exc;
        end
    endgenerate

    generate
        for (g = 0; g < NUM_GROUPS; g = g + 1) begin : gen_core_groups

            // Per-group neuron count: maps genvar g → SNN_GROUP_SIZE_x define.
            // All groups use uniform LOCAL_ID_WIDTH for bus compatibility.
            localparam THIS_NPG =
                (g ==  0) ? `SNN_GROUP_SIZE_0  :
                (g ==  1) ? `SNN_GROUP_SIZE_1  :
                (g ==  2) ? `SNN_GROUP_SIZE_2  :
                (g ==  3) ? `SNN_GROUP_SIZE_3  :
                (g ==  4) ? `SNN_GROUP_SIZE_4  :
                (g ==  5) ? `SNN_GROUP_SIZE_5  :
                (g ==  6) ? `SNN_GROUP_SIZE_6  :
                (g ==  7) ? `SNN_GROUP_SIZE_7  :
                (g ==  8) ? `SNN_GROUP_SIZE_8  :
                (g ==  9) ? `SNN_GROUP_SIZE_9  :
                (g == 10) ? `SNN_GROUP_SIZE_10 :
                (g == 11) ? `SNN_GROUP_SIZE_11 :
                (g == 12) ? `SNN_GROUP_SIZE_12 :
                (g == 13) ? `SNN_GROUP_SIZE_13 :
                (g == 14) ? `SNN_GROUP_SIZE_14 :
                            `SNN_GROUP_SIZE_15 ;

            core_group #(
                .GROUP_ID           (g),
                .NEURONS_PER_GROUP  (THIS_NPG),
                .LOCAL_ID_WIDTH     (LOCAL_ID_WIDTH),
                .DATA_WIDTH         (DATA_WIDTH),
                .WEIGHT_WIDTH       (WEIGHT_WIDTH),
                .THRESHOLD_WIDTH    (THRESHOLD_WIDTH),
                .LEAK_WIDTH         (LEAK_WIDTH),
                .REFRAC_WIDTH       (REFRAC_WIDTH),
                .SPIKE_BUFFER_DEPTH (SPIKE_BUFFER_DEPTH),
                .INTRA_MAX_FANOUT   (INTRA_MAX_FANOUT),
                .ENABLE_INTRA_SPARSE(ENABLE_INTRA_SPARSE),
                .ENABLE_INTRA_DENSE (ENABLE_INTRA_DENSE)
            ) u_core_group (
                .clk                (clk_100mhz),
                .rst_n              (rst_n_sync & ~hls_snn_reset),
                .enable             (hls_snn_enable),

                // External spike input (from event router)
                .ext_spike_valid    (grp_in_valid[g]),
                .ext_spike_dest_id  (grp_in_dest_id[g*LOCAL_ID_WIDTH +: LOCAL_ID_WIDTH]),
                .ext_spike_weight   (grp_in_weight[g*WEIGHT_WIDTH +: WEIGHT_WIDTH]),
                .ext_spike_exc_inh  (grp_in_exc[g]),
                .ext_spike_ready    (grp_in_ready[g]),

                // Output spike (to event router)
                .out_spike_valid    (grp_spike_valid[g]),
                .out_spike_neuron_id(grp_spike_neuron_id[g*LOCAL_ID_WIDTH +: LOCAL_ID_WIDTH]),
                .out_spike_ready    (grp_spike_ready[g]),

                // Global neuron parameters
                .global_threshold   (cfg_global_threshold),
                .global_leak_rate   (cfg_global_leak_rate),
                .global_refrac_period(cfg_global_refrac_period),

                // Weight load (combined from AXI config + learning engine)
                .weight_we          (combined_weight_we[g]),
                .weight_src_id      (combined_weight_src[g]),
                .weight_dst_id      (combined_weight_dst[g]),
                .weight_fanout_idx  (combined_weight_fanout_idx[g]),
                .weight_data        (combined_weight_data[g]),
                .weight_exc         (combined_weight_exc[g]),

                // Status
                .spike_count        (grp_spike_count[g*32 +: 32]),
                .group_busy         (grp_busy[g]),
                .profile_active     (profile_active),
                .profile_start      (cfg_profile_start),
                .profile_stop       (profile_stop_local),
                .sample_clear       (profile_stop_local),
                .sample_clear_done  (grp_sample_clear_done[g]),
                .score_cfg_we       (score_cfg_we_reg[g]),
                .score_cfg_class    (score_cfg_class_reg),
                .score_cfg_local_id (score_cfg_local_id_reg),
                .score_cfg_valid    (score_cfg_valid_reg),
                .score_live_snapshot(grp_score_live[g*NUM_CLASSES*32 +: NUM_CLASSES*32]),
                .profile_snapshot   (grp_profile_snapshot[g*PROFILE_CORE_METRIC_COUNT*32 +: PROFILE_CORE_METRIC_COUNT*32])
            );
        end
    endgenerate

    //=========================================================================
    // Synaptic Connectivity Table (Inter-Group Connections)
    //=========================================================================

    // Mux: AXI config writes OR event router config writes
    wire                          ct_cfg_we_mux;
    wire [GROUP_ID_WIDTH-1:0]     ct_cfg_src_group_mux;
    wire [LOCAL_ID_WIDTH-1:0]     ct_cfg_src_neuron_mux;
    wire [FANOUT_IDX_WIDTH-1:0]   ct_cfg_fanout_idx_mux;
    wire                          ct_cfg_valid_mux;
    wire [GROUP_ID_WIDTH-1:0]     ct_cfg_dst_group_mux;
    wire [LOCAL_ID_WIDTH-1:0]     ct_cfg_dst_neuron_mux;
    wire [WEIGHT_WIDTH-1:0]       ct_cfg_weight_mux;
    wire                          ct_cfg_exc_inh_mux;

    // AXI config has priority
    assign ct_cfg_we_mux         = ct_cfg_we_reg | ct_cfg_we;
    assign ct_cfg_src_group_mux  = ct_cfg_we_reg ? ct_cfg_src_group_reg  : ct_cfg_src_group;
    assign ct_cfg_src_neuron_mux = ct_cfg_we_reg ? ct_cfg_src_neuron_reg : ct_cfg_src_neuron;
    assign ct_cfg_fanout_idx_mux = ct_cfg_we_reg ? ct_cfg_fanout_idx_reg : ct_cfg_fanout_idx;
    assign ct_cfg_valid_mux      = ct_cfg_we_reg ? ct_cfg_valid_reg      : ct_cfg_valid_bit;
    assign ct_cfg_dst_group_mux  = ct_cfg_we_reg ? ct_cfg_dst_group_reg  : ct_cfg_dst_group;
    assign ct_cfg_dst_neuron_mux = ct_cfg_we_reg ? ct_cfg_dst_neuron_reg : ct_cfg_dst_neuron;
    assign ct_cfg_weight_mux     = ct_cfg_we_reg ? ct_cfg_weight_reg     : ct_cfg_weight;
    assign ct_cfg_exc_inh_mux    = ct_cfg_we_reg ? ct_cfg_exc_inh_reg    : ct_cfg_exc_inh;

    synaptic_connectivity_table #(
        .NUM_GROUPS         (NUM_GROUPS),
        .NEURONS_PER_GROUP  (NEURONS_PER_GROUP),
        .WEIGHT_WIDTH       (WEIGHT_WIDTH),
        .MAX_FANOUT_INTER   (MAX_FANOUT_INTER)
    ) u_connectivity_table (
        .clk                (clk_100mhz),
        .rst_n              (rst_n_sync & ~hls_snn_reset),

        // Write port (config)
        .cfg_we             (ct_cfg_we_mux),
        .cfg_src_group      (ct_cfg_src_group_mux),
        .cfg_src_neuron     (ct_cfg_src_neuron_mux),
        .cfg_fanout_idx     (ct_cfg_fanout_idx_mux),
        .cfg_valid          (ct_cfg_valid_mux),
        .cfg_dst_group      (ct_cfg_dst_group_mux),
        .cfg_dst_neuron     (ct_cfg_dst_neuron_mux),
        .cfg_weight         (ct_cfg_weight_mux),
        .cfg_exc_inh        (ct_cfg_exc_inh_mux),

        // Lookup port (from event router)
        .lookup_en          (ct_lookup_en),
        .lookup_src_group   (ct_lookup_src_group),
        .lookup_src_neuron  (ct_lookup_src_neuron),
        .lookup_fanout_idx  (ct_lookup_fanout_idx),

        // Lookup result
        .result_valid       (ct_result_valid),
        .result_dst_group   (ct_result_dst_group),
        .result_dst_neuron  (ct_result_dst_neuron),
        .result_weight      (ct_result_weight),
        .result_exc_inh     (ct_result_exc_inh),
        .result_entry_valid (ct_result_entry_valid)
    );

    //=========================================================================
    // Event Router (Next-Gen) - Central Spike Routing Hub
    //=========================================================================

    event_router_ng #(
        .NUM_GROUPS         (NUM_GROUPS),
        .NEURONS_PER_GROUP  (NEURONS_PER_GROUP),
        .WEIGHT_WIDTH       (WEIGHT_WIDTH),
        .MAX_FANOUT_INTER   (MAX_FANOUT_INTER),
        .CLASSIFIER_NEURONS (CLASSIFIER_NEURONS),
        .INPUT_SOURCE_NEURONS(INPUT_SOURCE_NEURONS),
        .NUM_CLASSES        (NUM_CLASSES),
        .FPS_PER_CLASS      (FPS_PER_CLASS)
    ) u_event_router (
        .clk                (clk_100mhz),
        .rst_n              (rst_n_sync & ~hls_snn_reset),
        .enable             (hls_snn_enable),

        // Core group spike outputs (FROM groups)
        .grp_spike_valid    (grp_spike_valid),
        .grp_spike_neuron_id(grp_spike_neuron_id),
        .grp_spike_ready    (grp_spike_ready),

        // Core group spike inputs (TO groups)
        .grp_in_valid       (grp_in_valid),
        .grp_in_dest_id     (grp_in_dest_id),
        .grp_in_weight      (grp_in_weight),
        .grp_in_exc         (grp_in_exc),
        .grp_in_ready       (grp_in_ready),

        // External spike input (from HLS)
        .ext_spike_valid    (hls_ext_pending),
        .ext_spike_neuron_id(hls_ext_pending_id),
        .ext_spike_weight   (hls_ext_pending_weight),
        .ext_spike_exc      (1'b1),  // HLS spikes default excitatory
        .ext_spike_ready    (router_ext_spike_ready),

        // Learning engine observation
        .learn_spike_valid  (learn_spike_valid),
        .learn_spike_src_id (learn_spike_src_id),
        .learn_spike_ready  (learn_spike_ready),

        // Learning weight update (HLS -> Event Router bridge)
        .learn_weight_valid     (learn_weight_valid_br),
        .learn_weight_group     (learn_weight_group_br),
        .learn_weight_src       (learn_weight_src_br),
        .learn_weight_dst       (learn_weight_dst_br),
        .learn_weight_data      (learn_weight_data_br),
        .learn_weight_exc       (learn_weight_exc_br),
        .learn_weight_is_inter  (learn_weight_is_inter_br),
        .learn_weight_dst_group (learn_weight_dst_group_br),
        .learn_weight_fanout_idx(learn_weight_fanout_idx_br),
        .learn_weight_ready     (learn_weight_ready_br),

        // Connectivity table interface
        .ct_lookup_en       (ct_lookup_en),
        .ct_lookup_src_group(ct_lookup_src_group),
        .ct_lookup_src_neuron(ct_lookup_src_neuron),
        .ct_lookup_fanout_idx(ct_lookup_fanout_idx),

        .ct_result_valid    (ct_result_valid),
        .ct_result_dst_group(ct_result_dst_group),
        .ct_result_dst_neuron(ct_result_dst_neuron),
        .ct_result_weight   (ct_result_weight),
        .ct_result_exc_inh  (ct_result_exc_inh),
        .ct_result_entry_valid(ct_result_entry_valid),

        // Weight config passthrough
        .grp_weight_we      (grp_weight_we),
        .grp_weight_src     (grp_weight_src),
        .grp_weight_dst     (grp_weight_dst),
        .grp_weight_data    (grp_weight_data),
        .grp_weight_exc     (grp_weight_exc),

        // CT config passthrough
        .ct_cfg_we          (ct_cfg_we),
        .ct_cfg_src_group   (ct_cfg_src_group),
        .ct_cfg_src_neuron  (ct_cfg_src_neuron),
        .ct_cfg_fanout_idx  (ct_cfg_fanout_idx),
        .ct_cfg_valid       (ct_cfg_valid_bit),
        .ct_cfg_dst_group   (ct_cfg_dst_group),
        .ct_cfg_dst_neuron  (ct_cfg_dst_neuron),
        .ct_cfg_weight      (ct_cfg_weight),
        .ct_cfg_exc_inh     (ct_cfg_exc_inh),

        // Status
        .routed_spike_count (routed_spike_count),
        .router_busy        (router_busy),
        .profile_fanout_valid     (router_profile_fanout_valid),
        .profile_fanout_src_group (router_profile_fanout_src_group),
        .profile_fanout_src_neuron(router_profile_fanout_src_neuron),
        .profile_fanout_dst_group (router_profile_fanout_dst_group),
        .profile_fanout_dst_neuron(router_profile_fanout_dst_neuron),
        .profile_fanout_weight    (router_profile_fanout_weight),
        .profile_class_valid      (router_profile_class_valid),
        .profile_class_id         (router_profile_class_id),
        .profile_class_weight     (router_profile_class_weight),
        .profile_active     (profile_active),
        .profile_start      (cfg_profile_start),
        .profile_stop       (profile_stop_local),
        .profile_snapshot   (router_profile_snapshot)
    );

endmodule
