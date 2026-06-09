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
//     [4:0]   = reserved
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
    parameter HLS_WEIGHT_WIDTH      = `SNN_HLS_WEIGHT_WIDTH
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
    reg                             hls_ext_pending;
    reg  [GLOBAL_ID_WIDTH-1:0]      hls_ext_pending_id;
    reg  [WEIGHT_WIDTH-1:0]         hls_ext_pending_weight;

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

    // Weight config from event router → core groups
    wire [NUM_GROUPS-1:0]                       grp_weight_we;
    wire [LOCAL_ID_WIDTH-1:0]                   grp_weight_src;
    wire [LOCAL_ID_WIDTH-1:0]                   grp_weight_dst;
    wire [WEIGHT_WIDTH-1:0]                     grp_weight_data;
    wire                                        grp_weight_exc;

    // Core group status
    wire [NUM_GROUPS*32-1:0]                    grp_spike_count;
    wire [NUM_GROUPS-1:0]                       grp_busy;
    wire [NUM_GROUPS*5*32-1:0]                  grp_profile_snapshot;

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
    wire [(11+2*NUM_GROUPS)*32-1:0] router_profile_snapshot;

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
    integer                      first_spike_tap_i;

    always @(*) begin
        first_spike_tap_seen = 1'b0;
        first_spike_tap_candidate = {GLOBAL_ID_WIDTH{1'b0}};
        for (first_spike_tap_i = 0; first_spike_tap_i < NUM_GROUPS; first_spike_tap_i = first_spike_tap_i + 1) begin
            if (!first_spike_tap_seen && grp_spike_valid[first_spike_tap_i]) begin
                first_spike_tap_seen = 1'b1;
                first_spike_tap_candidate = {
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
    localparam PROFILE_ROUTER_SCALAR_COUNT = 11;
    localparam PROFILE_ROUTER_TOTAL_COUNT  = 11 + 2*NUM_GROUPS;
    localparam PROFILE_CORE_BASE           = PROFILE_ROUTER_TOTAL_COUNT;
    localparam PROFILE_TOTAL_COUNT         = PROFILE_CORE_BASE + 5*NUM_GROUPS;
    localparam [7:0] PROFILE_NUM_GROUPS_INFO = NUM_GROUPS;
    localparam [15:0] PROFILE_TOTAL_COUNT_INFO = PROFILE_TOTAL_COUNT;

    reg        profile_active;
    reg        profile_done;
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
    integer profile_sel;

    assign cfg_profile_info = {8'h01, PROFILE_NUM_GROUPS_INFO, PROFILE_TOTAL_COUNT_INFO};

    always @(posedge clk_100mhz) begin
        if (!rst_n_sync || hls_snn_reset) begin
            profile_active          <= 1'b0;
            profile_done            <= 1'b0;
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
        end else begin
            if (cfg_profile_start) begin
                profile_active               <= 1'b1;
                profile_done                 <= 1'b0;
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
                end
                if (sample_seen_input && !sample_seen_output && rtl_spike_out_valid) begin
                    sample_seen_output <= 1'b1;
                    first_spike_latency_snapshot <= first_spike_latency_live;
                end
                if (sample_seen_input && !sample_service_done && sample_input_done &&
                    !hls_ext_pending && !router_busy && (grp_busy == {NUM_GROUPS{1'b0}})) begin
                    sample_service_done <= 1'b1;
                    service_cycles_snapshot <= service_cycles_live;
                end
            end
            if (cfg_profile_stop) begin
                profile_active         <= 1'b0;
                profile_done           <= 1'b1;
                total_latency_snapshot <= total_latency_live;
                if (!sample_service_done)
                    service_cycles_snapshot <= service_cycles_live;
            end
        end
    end

    always @(*) begin
        cfg_profile_data = 32'd0;
        profile_sel = cfg_profile_index;
        if (profile_sel == 10)
            cfg_profile_data = total_latency_snapshot;
        else if (profile_sel < PROFILE_ROUTER_TOTAL_COUNT)
            cfg_profile_data = router_profile_snapshot[profile_sel*32 +: 32];
        else if (profile_sel < PROFILE_TOTAL_COUNT)
            cfg_profile_data = grp_profile_snapshot[(profile_sel-PROFILE_CORE_BASE)*32 +: 32];
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

    always @(posedge clk_100mhz) begin
        if (!rst_n_sync) begin
            ct_cfg_we_reg       <= 0;
            intra_weight_we_reg <= {NUM_GROUPS{1'b0}};
        end else begin
            ct_cfg_we_reg       <= 0;
            intra_weight_we_reg <= {NUM_GROUPS{1'b0}};

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
                        intra_weight_we_reg[cfg_router_config_wdata[8:5]] <= 1'b1;
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

    always @(posedge clk_100mhz) begin
        if (!rst_n_sync || hls_snn_reset) begin
            hls_ext_pending        <= 1'b0;
            hls_ext_pending_id     <= {GLOBAL_ID_WIDTH{1'b0}};
            hls_ext_pending_weight <= {WEIGHT_WIDTH{1'b0}};
        end else begin
            if (hls_ext_pending && router_ext_spike_ready)
                hls_ext_pending <= 1'b0;

            if (hls_spike_out_pulse) begin
                hls_ext_pending        <= 1'b1;
                hls_ext_pending_id     <= hls_global_id;
                hls_ext_pending_weight <= hls_weight_truncated;
            end
        end
    end

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

    // HLS ready/busy
    assign rtl_spike_in_ready = !hls_ext_pending;
    assign rtl_snn_ready      = !router_busy & (grp_busy == {NUM_GROUPS{1'b0}});
    assign rtl_snn_busy       = router_busy | (grp_busy != {NUM_GROUPS{1'b0}});

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
                .SPIKE_BUFFER_DEPTH (SPIKE_BUFFER_DEPTH)
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
                .weight_data        (combined_weight_data[g]),
                .weight_exc         (combined_weight_exc[g]),

                // Status
                .spike_count        (grp_spike_count[g*32 +: 32]),
                .group_busy         (grp_busy[g]),
                .profile_active     (profile_active),
                .profile_start      (cfg_profile_start),
                .profile_stop       (cfg_profile_stop),
                .profile_snapshot   (grp_profile_snapshot[g*5*32 +: 5*32])
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
        .MAX_FANOUT_INTER   (MAX_FANOUT_INTER)
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
        .profile_active     (profile_active),
        .profile_start      (cfg_profile_start),
        .profile_stop       (cfg_profile_stop),
        .profile_snapshot   (router_profile_snapshot)
    );

endmodule
