//-----------------------------------------------------------------------------
// Title         : Event Router (Next-Gen) - Central Spike Router for Core Groups
// Project       : PYNQ-Z2 SNN Accelerator
// File          : event_router_ng.v
// Author        : Jiwoon Lee (@metr0jw)
// Organization  : Kwangwoon University, Seoul, South Korea
// Contact       : jwlee@linux.com
// Description   : Central event routing hub connecting:
//                 1. Core groups (8 bidirectional ports)
//                 2. Synaptic connectivity table (inter-group lookup)
//                 3. Learning engine / HLS IP (spike observation + weight updates)
//                 4. Host PC (AXI4-Lite control)
//                 5. External sensors (AXI4-Stream input)
//
// Operation:
//   When a core group outputs a spike:
//   1. Arbiter selects one source (round-robin among groups + external)
//   2. Connectivity table lookup: iterate fanout_idx for inter-group connections
//   3. Route spikes to destination core groups
//   4. Forward spike to learning engine (HLS) for trace/eligibility updates
//
// Resource Budget:
//   - Arbiter: ~200 LUT
//   - Connectivity table FSM: ~300 LUT
//   - AXI interface: ~500 LUT
//   - FIFOs: ~200 LUT (LUTRAM)
//   Total: ~1,200 LUT, ~400 FF
//-----------------------------------------------------------------------------

`timescale 1ns / 1ps
`include "snn_params.vh"

module event_router_ng #(
    parameter NUM_GROUPS        = `SNN_NUM_GROUPS,
    parameter NEURONS_PER_GROUP = `SNN_NEURONS_PER_GROUP,
    parameter WEIGHT_WIDTH      = `SNN_WEIGHT_WIDTH,
    parameter MAX_FANOUT_INTER  = `SNN_MAX_FANOUT_INTER,
    parameter GROUP_ID_WIDTH    = $clog2(NUM_GROUPS),
    parameter LOCAL_ID_WIDTH    = $clog2(NEURONS_PER_GROUP),
    parameter GLOBAL_ID_WIDTH   = GROUP_ID_WIDTH + LOCAL_ID_WIDTH,
    parameter FANOUT_IDX_WIDTH  = $clog2(MAX_FANOUT_INTER),
    parameter CLASSIFIER_NEURONS = 150,
    parameter INPUT_SOURCE_NEURONS = 784,
    parameter NUM_CLASSES        = 10,
    parameter FPS_PER_CLASS      = 15
)(
    input  wire                         clk,
    input  wire                         rst_n,
    input  wire                         enable,

    // --- Core Group Spike Output Ports (spikes FROM groups) ---
    input  wire [NUM_GROUPS-1:0]        grp_spike_valid,
    input  wire [NUM_GROUPS*LOCAL_ID_WIDTH-1:0] grp_spike_neuron_id,
    output reg  [NUM_GROUPS-1:0]        grp_spike_ready,

    // --- Core Group Spike Input Ports (spikes TO groups) ---
    output reg  [NUM_GROUPS-1:0]        grp_in_valid,
    output reg  [NUM_GROUPS*LOCAL_ID_WIDTH-1:0]  grp_in_dest_id,
    output reg  [NUM_GROUPS*WEIGHT_WIDTH-1:0]    grp_in_weight,
    output reg  [NUM_GROUPS-1:0]        grp_in_exc,
    input  wire [NUM_GROUPS-1:0]        grp_in_ready,

    // --- External Spike Input (Sensors / Host PC) ---
    input  wire                         ext_spike_valid,
    input  wire [GLOBAL_ID_WIDTH-1:0]   ext_spike_neuron_id,
    input  wire [WEIGHT_WIDTH-1:0]      ext_spike_weight,
    input  wire                         ext_spike_exc,
    output wire                         ext_spike_ready,

    // --- Learning Engine / HLS Observation Port ---
    output reg                          learn_spike_valid,
    output reg  [GLOBAL_ID_WIDTH-1:0]   learn_spike_src_id,
    input  wire                         learn_spike_ready,

    // --- Learning Engine Weight Update Port ---
    input  wire                         learn_weight_valid,
    input  wire [GROUP_ID_WIDTH-1:0]    learn_weight_group,
    input  wire [LOCAL_ID_WIDTH-1:0]    learn_weight_src,
    input  wire [LOCAL_ID_WIDTH-1:0]    learn_weight_dst,
    input  wire [WEIGHT_WIDTH-1:0]      learn_weight_data,
    input  wire                         learn_weight_exc,
    input  wire                         learn_weight_is_inter,   // 0=intra-group, 1=inter-group
    input  wire [GROUP_ID_WIDTH-1:0]    learn_weight_dst_group,  // for inter-group
    input  wire [FANOUT_IDX_WIDTH-1:0]  learn_weight_fanout_idx, // for inter-group
    output wire                         learn_weight_ready,

    // --- Connectivity Table Interface ---
    output reg                          ct_lookup_en,
    output reg  [GROUP_ID_WIDTH-1:0]    ct_lookup_src_group,
    output reg  [LOCAL_ID_WIDTH-1:0]    ct_lookup_src_neuron,
    output reg  [FANOUT_IDX_WIDTH-1:0]  ct_lookup_fanout_idx,

    input  wire                         ct_result_valid,
    input  wire [GROUP_ID_WIDTH-1:0]    ct_result_dst_group,
    input  wire [LOCAL_ID_WIDTH-1:0]    ct_result_dst_neuron,
    input  wire [WEIGHT_WIDTH-1:0]      ct_result_weight,
    input  wire                         ct_result_exc_inh,
    input  wire                         ct_result_entry_valid,

    // --- Weight Config Passthrough to Core Groups ---
    output reg  [NUM_GROUPS-1:0]        grp_weight_we,
    output reg  [LOCAL_ID_WIDTH-1:0]    grp_weight_src,
    output reg  [LOCAL_ID_WIDTH-1:0]    grp_weight_dst,
    output reg  [WEIGHT_WIDTH-1:0]      grp_weight_data,
    output reg                          grp_weight_exc,

    // --- Connectivity Table Config Passthrough ---
    output reg                          ct_cfg_we,
    output reg  [GROUP_ID_WIDTH-1:0]    ct_cfg_src_group,
    output reg  [LOCAL_ID_WIDTH-1:0]    ct_cfg_src_neuron,
    output reg  [FANOUT_IDX_WIDTH-1:0]  ct_cfg_fanout_idx,
    output reg                          ct_cfg_valid,
    output reg  [GROUP_ID_WIDTH-1:0]    ct_cfg_dst_group,
    output reg  [LOCAL_ID_WIDTH-1:0]    ct_cfg_dst_neuron,
    output reg  [WEIGHT_WIDTH-1:0]      ct_cfg_weight,
    output reg                          ct_cfg_exc_inh,

    // --- Status ---
    output wire [31:0]                  routed_spike_count,
    output wire                         router_busy,

    // Registered pulse for a CT valid fanout that was actually delivered.
    output reg                          profile_fanout_valid,
    output reg  [GROUP_ID_WIDTH-1:0]    profile_fanout_src_group,
    output reg  [LOCAL_ID_WIDTH-1:0]    profile_fanout_src_neuron,
    output reg  [GROUP_ID_WIDTH-1:0]    profile_fanout_dst_group,
    output reg  [LOCAL_ID_WIDTH-1:0]    profile_fanout_dst_neuron,
    output reg  [WEIGHT_WIDTH-1:0]      profile_fanout_weight,
    output reg                          profile_class_valid,
    output reg  [3:0]                   profile_class_id,
    output reg  [WEIGHT_WIDTH-1:0]      profile_class_weight,

    // --- Per-sample profiling ---
    input  wire                         profile_active,
    input  wire                         profile_start,
    input  wire                         profile_stop,
    output wire [(11+2*NUM_GROUPS+2*NUM_CLASSES)*32-1:0] profile_snapshot
);

    //=========================================================================
    // FSM States
    //=========================================================================
    localparam [3:0]
        ST_IDLE         = 4'd0,
        ST_ARB_SELECT   = 4'd1,     // Select next source via round-robin
        ST_EXT_ROUTE    = 4'd2,     // Route external spike directly
        ST_CT_LOOKUP    = 4'd3,     // Issue connectivity table lookup
        ST_CT_WAIT1     = 4'd4,     // Wait for CT BRAM read (cycle 1)
        ST_CT_WAIT2     = 4'd5,     // Wait for CT data unpack (cycle 2)
        ST_CT_DELIVER   = 4'd6,     // Deliver result to destination group
        ST_CT_NEXT      = 4'd7,     // Advance fanout index
        ST_LEARN_NOTIFY = 4'd8,     // Notify learning engine
        ST_WEIGHT_FWD   = 4'd9;     // Forward weight update

    reg [3:0] state;

    //=========================================================================
    // Round-Robin Arbiter
    //=========================================================================
    reg [GROUP_ID_WIDTH-1:0] rr_priority;       // Current priority pointer
    reg [GROUP_ID_WIDTH-1:0] selected_group;    // Which group won arbitration
    reg [LOCAL_ID_WIDTH-1:0] selected_neuron;   // Neuron ID from winning group
    reg                      ext_selected;      // External source selected
    reg [31:0]               spike_counter;
    reg [GROUP_ID_WIDTH-1:0] ext_route_group;
    reg [LOCAL_ID_WIDTH-1:0] ext_route_neuron;
    reg [WEIGHT_WIDTH-1:0]   ext_route_weight;
    reg                      ext_route_exc;

    assign routed_spike_count = spike_counter;
    assign router_busy = (state != ST_IDLE);
    assign ext_spike_ready = (state == ST_IDLE);
    assign learn_weight_ready = (state == ST_IDLE);

    //=========================================================================
    // Fanout iteration
    //=========================================================================
    reg [FANOUT_IDX_WIDTH-1:0] fanout_idx;

    // CT result is a single-cycle pulse. Keep it pending while the destination
    // group applies backpressure so a valid fanout cannot be lost.
    reg                         ct_pending_valid;
    reg                         ct_pending_entry_valid;
    reg [GROUP_ID_WIDTH-1:0]    ct_pending_dst_group;
    reg [LOCAL_ID_WIDTH-1:0]    ct_pending_dst_neuron;
    reg [WEIGHT_WIDTH-1:0]      ct_pending_weight;
    reg                         ct_pending_exc_inh;

    wire                        ct_cur_valid       = ct_pending_valid || ct_result_valid;
    wire                        ct_cur_entry_valid = ct_pending_valid ? ct_pending_entry_valid : ct_result_entry_valid;
    wire [GROUP_ID_WIDTH-1:0]   ct_cur_dst_group   = ct_pending_valid ? ct_pending_dst_group : ct_result_dst_group;
    wire [LOCAL_ID_WIDTH-1:0]   ct_cur_dst_neuron  = ct_pending_valid ? ct_pending_dst_neuron : ct_result_dst_neuron;
    wire [WEIGHT_WIDTH-1:0]     ct_cur_weight      = ct_pending_valid ? ct_pending_weight : ct_result_weight;
    wire                        ct_cur_exc_inh     = ct_pending_valid ? ct_pending_exc_inh : ct_result_exc_inh;

    //=========================================================================
    // Per-sample profiling
    //=========================================================================
    localparam PROFILE_ROUTER_TOTAL_COUNT = 11 + 2*NUM_GROUPS + 2*NUM_CLASSES;
    localparam PROFILE_CLASS_SCORE_BASE   = 11 + 2*NUM_GROUPS;
    localparam PROFILE_CLASS_EVENT_BASE   = PROFILE_CLASS_SCORE_BASE + NUM_CLASSES;

    reg [31:0] input_spike_count_live;
    reg [31:0] output_spike_count_live;
    reg [31:0] ct_lookup_count_live;
    reg [31:0] ct_valid_entry_count_live;
    reg [31:0] ct_invalid_entry_count_live;
    reg [31:0] router_busy_cycles_live;
    reg [31:0] router_idle_cycles_live;
    reg [31:0] router_stall_cycles_live;
    reg [31:0] cross_group_event_count_live;
    reg [31:0] same_group_event_count_live;
    reg [31:0] total_latency_placeholder_live;
    reg [31:0] input_spike_count_snap;
    reg [31:0] output_spike_count_snap;
    reg [31:0] ct_lookup_count_snap;
    reg [31:0] ct_valid_entry_count_snap;
    reg [31:0] ct_invalid_entry_count_snap;
    reg [31:0] router_busy_cycles_snap;
    reg [31:0] router_idle_cycles_snap;
    reg [31:0] router_stall_cycles_snap;
    reg [31:0] cross_group_event_count_snap;
    reg [31:0] same_group_event_count_snap;
    reg [31:0] total_latency_placeholder_snap;
    reg [31:0] per_group_in_event_live [0:NUM_GROUPS-1];
    reg [31:0] per_group_stall_cycles_live [0:NUM_GROUPS-1];
    reg [31:0] per_group_in_event_snap [0:NUM_GROUPS-1];
    reg [31:0] per_group_stall_cycles_snap [0:NUM_GROUPS-1];
    reg [31:0] router_class_score_live [0:NUM_CLASSES-1];
    reg [31:0] router_class_event_live [0:NUM_CLASSES-1];
    reg [31:0] router_class_score_snap [0:NUM_CLASSES-1];
    reg [31:0] router_class_event_snap [0:NUM_CLASSES-1];
    reg        profile_ext_input_accept_d;
    reg        profile_group_output_accept_d;
    reg        profile_ct_lookup_en_d;
    reg        profile_ct_valid_consume_d;
    reg        profile_ct_invalid_consume_d;
    reg        profile_fanout_stall_d;
    reg        profile_ext_delivery_stall_d;
    reg        profile_ext_delivery_accept_d;
    reg [GROUP_ID_WIDTH-1:0] profile_ext_route_group_d;
    integer pi;

    function is_classifier_global_id;
        input [GROUP_ID_WIDTH-1:0] group_id;
        input [LOCAL_ID_WIDTH-1:0] local_id;
        integer logical_id;
        begin
            logical_id = local_id * NUM_GROUPS + group_id;
            is_classifier_global_id = (logical_id < CLASSIFIER_NEURONS);
        end
    endfunction

    function [3:0] classifier_class_id;
        input [GROUP_ID_WIDTH-1:0] group_id;
        input [LOCAL_ID_WIDTH-1:0] local_id;
        integer logical_id;
        begin
            logical_id = local_id * NUM_GROUPS + group_id;
            classifier_class_id = logical_id / FPS_PER_CLASS;
        end
    endfunction

    wire ext_input_accept = (state == ST_IDLE) && enable &&
                            ext_spike_valid && !learn_weight_valid;
    wire group_output_accept = |(grp_spike_valid & grp_spike_ready);
    wire ct_result_consume = (state == ST_CT_DELIVER) && ct_cur_valid &&
                             (!ct_cur_entry_valid ||
                              (ct_cur_dst_group >= NUM_GROUPS) ||
                              grp_in_ready[ct_cur_dst_group]);
    wire ct_valid_consume = ct_result_consume && ct_cur_entry_valid;
    wire ct_invalid_consume = ct_result_consume && !ct_cur_entry_valid;
    wire fanout_delivery_accept = (state == ST_CT_DELIVER) && ct_cur_valid &&
                                  ct_cur_entry_valid &&
                                  (ct_cur_dst_group < NUM_GROUPS) &&
                                  grp_in_ready[ct_cur_dst_group];
    wire fanout_stall = (state == ST_CT_DELIVER) && ct_cur_valid &&
                        ct_cur_entry_valid &&
                        (ct_cur_dst_group < NUM_GROUPS) &&
                        !grp_in_ready[ct_cur_dst_group];
    wire ext_delivery_accept = (state == ST_EXT_ROUTE) &&
                               (ext_route_group < NUM_GROUPS) &&
                               grp_in_ready[ext_route_group];
    wire ext_delivery_stall = (state == ST_EXT_ROUTE) &&
                              (ext_route_group < NUM_GROUPS) &&
                              !grp_in_ready[ext_route_group];
    wire fanout_delivery_classifier = fanout_delivery_accept &&
                                       is_classifier_global_id(ct_cur_dst_group,
                                                               ct_cur_dst_neuron);
    wire [3:0] fanout_delivery_class_id = classifier_class_id(ct_cur_dst_group,
                                                              ct_cur_dst_neuron);

    generate
        genvar pg;
        for (pg = 0; pg < PROFILE_ROUTER_TOTAL_COUNT; pg = pg + 1) begin : gen_profile_snapshot
            if (pg == 0) begin : gen_input_spike_count
                assign profile_snapshot[pg*32 +: 32] = input_spike_count_snap;
            end else if (pg == 1) begin : gen_output_spike_count
                assign profile_snapshot[pg*32 +: 32] = output_spike_count_snap;
            end else if (pg == 2) begin : gen_ct_lookup_count
                assign profile_snapshot[pg*32 +: 32] = ct_lookup_count_snap;
            end else if (pg == 3) begin : gen_ct_valid_entry_count
                assign profile_snapshot[pg*32 +: 32] = ct_valid_entry_count_snap;
            end else if (pg == 4) begin : gen_ct_invalid_entry_count
                assign profile_snapshot[pg*32 +: 32] = ct_invalid_entry_count_snap;
            end else if (pg == 5) begin : gen_router_busy_cycles
                assign profile_snapshot[pg*32 +: 32] = router_busy_cycles_snap;
            end else if (pg == 6) begin : gen_router_idle_cycles
                assign profile_snapshot[pg*32 +: 32] = router_idle_cycles_snap;
            end else if (pg == 7) begin : gen_router_stall_cycles
                assign profile_snapshot[pg*32 +: 32] = router_stall_cycles_snap;
            end else if (pg == 8) begin : gen_cross_group_event_count
                assign profile_snapshot[pg*32 +: 32] = cross_group_event_count_snap;
            end else if (pg == 9) begin : gen_same_group_event_count
                assign profile_snapshot[pg*32 +: 32] = same_group_event_count_snap;
            end else if (pg == 10) begin : gen_total_latency_placeholder
                assign profile_snapshot[pg*32 +: 32] = total_latency_placeholder_snap;
            end else if (pg >= 11 && pg < 11+NUM_GROUPS) begin : gen_group_in_snapshot
                assign profile_snapshot[pg*32 +: 32] =
                    per_group_in_event_snap[pg-11];
            end else if (pg >= 11+NUM_GROUPS && pg < PROFILE_CLASS_SCORE_BASE) begin : gen_group_stall_snapshot
                assign profile_snapshot[pg*32 +: 32] =
                    per_group_stall_cycles_snap[pg-(11+NUM_GROUPS)];
            end else if (pg >= PROFILE_CLASS_SCORE_BASE && pg < PROFILE_CLASS_EVENT_BASE) begin : gen_class_score_snapshot
                assign profile_snapshot[pg*32 +: 32] =
                    router_class_score_snap[pg-PROFILE_CLASS_SCORE_BASE];
            end else if (pg >= PROFILE_CLASS_EVENT_BASE) begin : gen_class_event_snapshot
                assign profile_snapshot[pg*32 +: 32] =
                    router_class_event_snap[pg-PROFILE_CLASS_EVENT_BASE];
            end else begin : gen_base_snapshot
                assign profile_snapshot[pg*32 +: 32] = 32'd0;
            end
        end
    endgenerate

    always @(posedge clk) begin
        if (!rst_n) begin
            input_spike_count_live <= 32'd0;
            output_spike_count_live <= 32'd0;
            ct_lookup_count_live <= 32'd0;
            ct_valid_entry_count_live <= 32'd0;
            ct_invalid_entry_count_live <= 32'd0;
            router_busy_cycles_live <= 32'd0;
            router_idle_cycles_live <= 32'd0;
            router_stall_cycles_live <= 32'd0;
            cross_group_event_count_live <= 32'd0;
            same_group_event_count_live <= 32'd0;
            total_latency_placeholder_live <= 32'd0;
            input_spike_count_snap <= 32'd0;
            output_spike_count_snap <= 32'd0;
            ct_lookup_count_snap <= 32'd0;
            ct_valid_entry_count_snap <= 32'd0;
            ct_invalid_entry_count_snap <= 32'd0;
            router_busy_cycles_snap <= 32'd0;
            router_idle_cycles_snap <= 32'd0;
            router_stall_cycles_snap <= 32'd0;
            cross_group_event_count_snap <= 32'd0;
            same_group_event_count_snap <= 32'd0;
            total_latency_placeholder_snap <= 32'd0;
            for (pi = 0; pi < NUM_GROUPS; pi = pi + 1) begin
                per_group_in_event_live[pi] <= 32'd0;
                per_group_stall_cycles_live[pi] <= 32'd0;
                per_group_in_event_snap[pi] <= 32'd0;
                per_group_stall_cycles_snap[pi] <= 32'd0;
            end
            for (pi = 0; pi < NUM_CLASSES; pi = pi + 1) begin
                router_class_score_live[pi] <= 32'd0;
                router_class_event_live[pi] <= 32'd0;
                router_class_score_snap[pi] <= 32'd0;
                router_class_event_snap[pi] <= 32'd0;
            end
            profile_ext_input_accept_d <= 1'b0;
            profile_group_output_accept_d <= 1'b0;
            profile_ct_lookup_en_d <= 1'b0;
            profile_ct_valid_consume_d <= 1'b0;
            profile_ct_invalid_consume_d <= 1'b0;
            profile_fanout_stall_d <= 1'b0;
            profile_ext_delivery_stall_d <= 1'b0;
            profile_ext_delivery_accept_d <= 1'b0;
            profile_ext_route_group_d <= {GROUP_ID_WIDTH{1'b0}};
        end else begin
            if (profile_start) begin
                input_spike_count_live <= 32'd0;
                output_spike_count_live <= 32'd0;
                ct_lookup_count_live <= 32'd0;
                ct_valid_entry_count_live <= 32'd0;
                ct_invalid_entry_count_live <= 32'd0;
                router_busy_cycles_live <= 32'd0;
                router_idle_cycles_live <= 32'd0;
                router_stall_cycles_live <= 32'd0;
                cross_group_event_count_live <= 32'd0;
                same_group_event_count_live <= 32'd0;
                total_latency_placeholder_live <= 32'd0;
                for (pi = 0; pi < NUM_GROUPS; pi = pi + 1) begin
                    per_group_in_event_live[pi] <= 32'd0;
                    per_group_stall_cycles_live[pi] <= 32'd0;
                end
                for (pi = 0; pi < NUM_CLASSES; pi = pi + 1) begin
                    router_class_score_live[pi] <= 32'd0;
                    router_class_event_live[pi] <= 32'd0;
                end
                profile_ext_input_accept_d <= 1'b0;
                profile_group_output_accept_d <= 1'b0;
                profile_ct_lookup_en_d <= 1'b0;
                profile_ct_valid_consume_d <= 1'b0;
                profile_ct_invalid_consume_d <= 1'b0;
                profile_fanout_stall_d <= 1'b0;
                profile_ext_delivery_stall_d <= 1'b0;
                profile_ext_delivery_accept_d <= 1'b0;
                profile_ext_route_group_d <= {GROUP_ID_WIDTH{1'b0}};
            end else if (profile_active) begin
                if (profile_ext_input_accept_d)
                    input_spike_count_live <= input_spike_count_live + 1'b1;
                if (profile_group_output_accept_d)
                    output_spike_count_live <= output_spike_count_live + 1'b1;
                if (profile_ct_lookup_en_d)
                    ct_lookup_count_live <= ct_lookup_count_live + 1'b1;
                if (profile_ct_valid_consume_d)
                    ct_valid_entry_count_live <= ct_valid_entry_count_live + 1'b1;
                if (profile_ct_invalid_consume_d)
                    ct_invalid_entry_count_live <= ct_invalid_entry_count_live + 1'b1;
                if (state != ST_IDLE)       router_busy_cycles_live <= router_busy_cycles_live + 1'b1;
                else                        router_idle_cycles_live <= router_idle_cycles_live + 1'b1;
                if (profile_fanout_stall_d || profile_ext_delivery_stall_d)
                    router_stall_cycles_live <= router_stall_cycles_live + 1'b1;
                if (profile_fanout_valid && profile_fanout_src_group != profile_fanout_dst_group)
                    cross_group_event_count_live <= cross_group_event_count_live + 1'b1;
                if (profile_fanout_valid && profile_fanout_src_group == profile_fanout_dst_group)
                    same_group_event_count_live <= same_group_event_count_live + 1'b1;
                for (pi = 0; pi < NUM_GROUPS; pi = pi + 1) begin
                    if ((profile_fanout_valid && profile_fanout_dst_group == pi) ||
                        (profile_ext_delivery_accept_d && profile_ext_route_group_d == pi))
                        per_group_in_event_live[pi] <= per_group_in_event_live[pi] + 1'b1;
                    if ((profile_fanout_stall_d && ct_cur_dst_group == pi) ||
                        (profile_ext_delivery_stall_d && profile_ext_route_group_d == pi))
                        per_group_stall_cycles_live[pi] <= per_group_stall_cycles_live[pi] + 1'b1;
                end
                if (profile_class_valid) begin
                    case (profile_class_id)
                        4'd0: begin
                            router_class_score_live[0] <= router_class_score_live[0] + profile_class_weight;
                            router_class_event_live[0] <= router_class_event_live[0] + 1'b1;
                        end
                        4'd1: begin
                            router_class_score_live[1] <= router_class_score_live[1] + profile_class_weight;
                            router_class_event_live[1] <= router_class_event_live[1] + 1'b1;
                        end
                        4'd2: begin
                            router_class_score_live[2] <= router_class_score_live[2] + profile_class_weight;
                            router_class_event_live[2] <= router_class_event_live[2] + 1'b1;
                        end
                        4'd3: begin
                            router_class_score_live[3] <= router_class_score_live[3] + profile_class_weight;
                            router_class_event_live[3] <= router_class_event_live[3] + 1'b1;
                        end
                        4'd4: begin
                            router_class_score_live[4] <= router_class_score_live[4] + profile_class_weight;
                            router_class_event_live[4] <= router_class_event_live[4] + 1'b1;
                        end
                        4'd5: begin
                            router_class_score_live[5] <= router_class_score_live[5] + profile_class_weight;
                            router_class_event_live[5] <= router_class_event_live[5] + 1'b1;
                        end
                        4'd6: begin
                            router_class_score_live[6] <= router_class_score_live[6] + profile_class_weight;
                            router_class_event_live[6] <= router_class_event_live[6] + 1'b1;
                        end
                        4'd7: begin
                            router_class_score_live[7] <= router_class_score_live[7] + profile_class_weight;
                            router_class_event_live[7] <= router_class_event_live[7] + 1'b1;
                        end
                        4'd8: begin
                            router_class_score_live[8] <= router_class_score_live[8] + profile_class_weight;
                            router_class_event_live[8] <= router_class_event_live[8] + 1'b1;
                        end
                        4'd9: begin
                            router_class_score_live[9] <= router_class_score_live[9] + profile_class_weight;
                            router_class_event_live[9] <= router_class_event_live[9] + 1'b1;
                        end
                        default: begin
                        end
                    endcase
                end

                profile_ext_input_accept_d <= ext_input_accept;
                profile_group_output_accept_d <= group_output_accept;
                profile_ct_lookup_en_d <= ct_lookup_en;
                profile_ct_valid_consume_d <= ct_valid_consume;
                profile_ct_invalid_consume_d <= ct_invalid_consume;
                profile_fanout_stall_d <= fanout_stall;
                profile_ext_delivery_stall_d <= ext_delivery_stall;
                profile_ext_delivery_accept_d <= ext_delivery_accept;
                profile_ext_route_group_d <= ext_route_group;
            end
            if (profile_stop) begin
                input_spike_count_snap <= input_spike_count_live;
                output_spike_count_snap <= output_spike_count_live;
                ct_lookup_count_snap <= ct_lookup_count_live;
                ct_valid_entry_count_snap <= ct_valid_entry_count_live;
                ct_invalid_entry_count_snap <= ct_invalid_entry_count_live;
                router_busy_cycles_snap <= router_busy_cycles_live;
                router_idle_cycles_snap <= router_idle_cycles_live;
                router_stall_cycles_snap <= router_stall_cycles_live;
                cross_group_event_count_snap <= cross_group_event_count_live;
                same_group_event_count_snap <= same_group_event_count_live;
                total_latency_placeholder_snap <= total_latency_placeholder_live;
                for (pi = 0; pi < NUM_GROUPS; pi = pi + 1) begin
                    per_group_in_event_snap[pi] <= per_group_in_event_live[pi];
                    per_group_stall_cycles_snap[pi] <= per_group_stall_cycles_live[pi];
                end
                for (pi = 0; pi < NUM_CLASSES; pi = pi + 1) begin
                    router_class_score_snap[pi] <= router_class_score_live[pi];
                    router_class_event_snap[pi] <= router_class_event_live[pi];
                end
            end
        end
    end

    //=========================================================================
    // Main Router FSM
    //=========================================================================
    integer gi;

    always @(posedge clk) begin
        if (!rst_n) begin
            state           <= ST_IDLE;
            rr_priority     <= 0;
            selected_group  <= 0;
            selected_neuron <= 0;
            ext_selected    <= 0;
            spike_counter   <= 0;
            ext_route_group <= 0;
            ext_route_neuron <= 0;
            ext_route_weight <= 0;
            ext_route_exc <= 0;
            fanout_idx      <= 0;
            ct_lookup_en    <= 0;
            ct_pending_valid <= 1'b0;
            ct_pending_entry_valid <= 1'b0;
            ct_pending_dst_group <= {GROUP_ID_WIDTH{1'b0}};
            ct_pending_dst_neuron <= {LOCAL_ID_WIDTH{1'b0}};
            ct_pending_weight <= {WEIGHT_WIDTH{1'b0}};
            ct_pending_exc_inh <= 1'b0;
            profile_fanout_valid <= 1'b0;
            profile_fanout_src_group <= {GROUP_ID_WIDTH{1'b0}};
            profile_fanout_src_neuron <= {LOCAL_ID_WIDTH{1'b0}};
            profile_fanout_dst_group <= {GROUP_ID_WIDTH{1'b0}};
            profile_fanout_dst_neuron <= {LOCAL_ID_WIDTH{1'b0}};
            profile_fanout_weight <= {WEIGHT_WIDTH{1'b0}};
            profile_class_valid <= 1'b0;
            profile_class_id <= 4'd0;
            profile_class_weight <= {WEIGHT_WIDTH{1'b0}};

            grp_spike_ready   <= {NUM_GROUPS{1'b0}};
            grp_in_valid      <= {NUM_GROUPS{1'b0}};
            grp_in_dest_id    <= {(NUM_GROUPS*LOCAL_ID_WIDTH){1'b0}};
            grp_in_weight     <= {(NUM_GROUPS*WEIGHT_WIDTH){1'b0}};
            grp_in_exc        <= {NUM_GROUPS{1'b0}};
            learn_spike_valid <= 0;
            learn_spike_src_id <= 0;

            grp_weight_we   <= {NUM_GROUPS{1'b0}};
            grp_weight_src  <= 0;
            grp_weight_dst  <= 0;
            grp_weight_data <= 0;
            grp_weight_exc  <= 0;

            ct_cfg_we       <= 0;
        end else begin
            // Default deasserts
            grp_spike_ready   <= {NUM_GROUPS{1'b0}};
            grp_in_valid      <= {NUM_GROUPS{1'b0}};
            learn_spike_valid <= 0;
            ct_lookup_en      <= 0;
            grp_weight_we     <= {NUM_GROUPS{1'b0}};
            ct_cfg_we         <= 0;
            profile_fanout_valid <= 1'b0;
            profile_class_valid <= 1'b0;

            if (enable || state != ST_IDLE) begin
                case (state)
                    //----------------------------------------------------------
                    ST_IDLE: begin
                        // Handle weight update forwarding (highest priority)
                        if (learn_weight_valid) begin
                            if (learn_weight_is_inter) begin
                                // Forward to connectivity table
                                ct_cfg_we         <= 1;
                                ct_cfg_src_group  <= learn_weight_group;
                                ct_cfg_src_neuron <= learn_weight_src;
                                ct_cfg_fanout_idx <= learn_weight_fanout_idx;
                                ct_cfg_valid      <= 1;
                                ct_cfg_dst_group  <= learn_weight_dst_group;
                                ct_cfg_dst_neuron <= learn_weight_dst;
                                ct_cfg_weight     <= learn_weight_data;
                                ct_cfg_exc_inh    <= learn_weight_exc;
                            end else begin
                                // Forward to appropriate core group
                                grp_weight_we[learn_weight_group] <= 1;
                                grp_weight_src  <= learn_weight_src;
                                grp_weight_dst  <= learn_weight_dst;
                                grp_weight_data <= learn_weight_data;
                                grp_weight_exc  <= learn_weight_exc;
                            end
                        end
                        // Check for external spike
                        else if (ext_spike_valid) begin
                            ext_selected  <= 1;
                            ext_route_group  <= ext_spike_neuron_id[GLOBAL_ID_WIDTH-1:LOCAL_ID_WIDTH];
                            ext_route_neuron <= ext_spike_neuron_id[LOCAL_ID_WIDTH-1:0];
                            ext_route_weight <= ext_spike_weight;
                            ext_route_exc    <= ext_spike_exc;
                            state         <= ST_EXT_ROUTE;
                        end
                        // Check group spikes via round-robin. Stay truly idle
                        // when there is no work so the HLS input path sees
                        // continuous ready instead of a periodic busy pulse.
                        else if (|grp_spike_valid) begin
                            state <= ST_ARB_SELECT;
                        end else begin
                            state <= ST_IDLE;
                        end
                    end

                    //----------------------------------------------------------
                    ST_ARB_SELECT: begin
                        // Round-robin scan starting from rr_priority
                        ext_selected <= 0;
                        begin : arb_scan
                            reg found;
                            reg [GROUP_ID_WIDTH-1:0] idx;
                            reg [GROUP_ID_WIDTH-1:0] found_idx;
                            reg [LOCAL_ID_WIDTH-1:0] found_neuron;
                            found = 0;
                            found_idx = 0;
                            found_neuron = {LOCAL_ID_WIDTH{1'b0}};
                            for (gi = 0; gi < NUM_GROUPS; gi = gi + 1) begin
                                idx = (rr_priority + gi[GROUP_ID_WIDTH-1:0]) % NUM_GROUPS;
                                if (!found && grp_spike_valid[idx]) begin
                                    selected_group  <= idx;
                                    found_neuron = grp_spike_neuron_id[idx*LOCAL_ID_WIDTH +: LOCAL_ID_WIDTH];
                                    selected_neuron <= found_neuron;
                                    grp_spike_ready[idx] <= 1;
                                    found = 1;
                                    found_idx = idx;
                                end
                            end
                            if (found) begin
                                rr_priority <= (found_idx + 1) % NUM_GROUPS;
                                fanout_idx  <= 0;
                                state <= ST_CT_LOOKUP;
                            end else begin
                                state <= ST_IDLE;  // No spikes pending
                            end
                        end
                    end

                    //----------------------------------------------------------
                    ST_EXT_ROUTE: begin
                        // Route external spike directly to target group
                        begin : ext_route_body
                            reg [GROUP_ID_WIDTH-1:0] tgt_grp;
                            reg [LOCAL_ID_WIDTH-1:0] tgt_neuron;
                            tgt_grp    = ext_route_group;
                            tgt_neuron = ext_route_neuron;

                            if (tgt_grp < NUM_GROUPS && grp_in_ready[tgt_grp]) begin
                                grp_in_valid[tgt_grp] <= 1;
                                grp_in_dest_id[tgt_grp*LOCAL_ID_WIDTH +: LOCAL_ID_WIDTH] <= tgt_neuron;
                                grp_in_weight[tgt_grp*WEIGHT_WIDTH +: WEIGHT_WIDTH]      <= ext_route_weight;
                                grp_in_exc[tgt_grp]   <= ext_route_exc;
                                spike_counter <= spike_counter + 1;
                                state         <= ST_IDLE;
                            end else if (tgt_grp >= NUM_GROUPS) begin
                                state <= ST_IDLE;  // Drop invalid group index
                            end
                            // else: wait for ready (backpressure)
                        end
                    end

                    //----------------------------------------------------------
                    ST_CT_LOOKUP: begin
                        // Issue connectivity table read
                        ct_lookup_en         <= 1;
                        ct_lookup_src_group  <= selected_group;
                        ct_lookup_src_neuron <= selected_neuron;
                        ct_lookup_fanout_idx <= fanout_idx;
                        ct_pending_valid     <= 1'b0;
                        state                <= ST_CT_WAIT1;
                    end

                    ST_CT_WAIT1: begin
                        // Wait cycle 1: BRAM read latency
                        state <= ST_CT_WAIT2;
                    end

                    ST_CT_WAIT2: begin
                        // Wait cycle 2: data unpack latency
                        state <= ST_CT_DELIVER;
                    end

                    ST_CT_DELIVER: begin
                        if (!ct_pending_valid && ct_result_valid && ct_result_entry_valid &&
                            (ct_result_dst_group < NUM_GROUPS) &&
                            !grp_in_ready[ct_result_dst_group]) begin
                            ct_pending_valid       <= 1'b1;
                            ct_pending_entry_valid <= ct_result_entry_valid;
                            ct_pending_dst_group   <= ct_result_dst_group;
                            ct_pending_dst_neuron  <= ct_result_dst_neuron;
                            ct_pending_weight      <= ct_result_weight;
                            ct_pending_exc_inh     <= ct_result_exc_inh;
                        end

                        if (ct_cur_valid && ct_cur_entry_valid) begin
                            // Deliver spike to destination core group
                            if (ct_cur_dst_group >= NUM_GROUPS) begin
                                // Invalid destination group — skip to next fanout
                                ct_pending_valid <= 1'b0;
                                state <= ST_CT_NEXT;
                            end else if (grp_in_ready[ct_cur_dst_group]) begin
                                grp_in_valid[ct_cur_dst_group] <= 1;
                                grp_in_dest_id[ct_cur_dst_group*LOCAL_ID_WIDTH +: LOCAL_ID_WIDTH]
                                    <= ct_cur_dst_neuron;
                                grp_in_weight[ct_cur_dst_group*WEIGHT_WIDTH +: WEIGHT_WIDTH]
                                    <= ct_cur_weight;
                                grp_in_exc[ct_cur_dst_group] <= ct_cur_exc_inh;
                                spike_counter <= spike_counter + 1;
                                profile_fanout_valid      <= 1'b1;
                                profile_fanout_src_group  <= selected_group;
                                profile_fanout_src_neuron <= selected_neuron;
                                profile_fanout_dst_group  <= ct_cur_dst_group;
                                profile_fanout_dst_neuron <= ct_cur_dst_neuron;
                                profile_fanout_weight     <= ct_cur_weight;
                                if (fanout_delivery_classifier) begin
                                    profile_class_valid  <= 1'b1;
                                    profile_class_id     <= fanout_delivery_class_id;
                                    profile_class_weight <= ct_cur_weight;
                                end
                                ct_pending_valid <= 1'b0;
                                state         <= ST_CT_NEXT;
                            end
                            // else: wait for group ready (backpressure)
                        end else if (ct_cur_valid) begin
                            // No more valid connections — notify learning engine
                            ct_pending_valid <= 1'b0;
                            state <= ST_LEARN_NOTIFY;
                        end
                    end

                    ST_CT_NEXT: begin
                        if (fanout_idx + 1 >= MAX_FANOUT_INTER) begin
                            state <= ST_LEARN_NOTIFY;
                        end else begin
                            fanout_idx <= fanout_idx + 1;
                            state      <= ST_CT_LOOKUP;
                        end
                    end

                    //----------------------------------------------------------
                    ST_LEARN_NOTIFY: begin
                        // Notify learning engine of the spike event
                        if (learn_spike_ready || !learn_spike_valid) begin
                            learn_spike_valid  <= 1;
                            learn_spike_src_id <= {selected_group, selected_neuron};
                            state              <= ST_IDLE;
                        end
                    end

                    default: state <= ST_IDLE;
                endcase
            end
        end
    end

endmodule
