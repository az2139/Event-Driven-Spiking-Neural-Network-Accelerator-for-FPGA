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
    parameter FANOUT_IDX_WIDTH  = $clog2(MAX_FANOUT_INTER)
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

    // --- Per-sample profiling ---
    input  wire                         profile_active,
    input  wire                         profile_start,
    input  wire                         profile_stop,
    output wire [(11+2*NUM_GROUPS)*32-1:0] profile_snapshot
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

    //=========================================================================
    // Per-sample profiling
    //=========================================================================
    reg [31:0] profile_live [0:10+2*NUM_GROUPS];
    reg [31:0] profile_snap [0:10+2*NUM_GROUPS];
    integer pi;

    wire ext_input_accept = (state == ST_IDLE) && enable &&
                            ext_spike_valid && !learn_weight_valid;
    wire group_output_accept = |(grp_spike_valid & grp_spike_ready);
    wire ct_result_consume = (state == ST_CT_DELIVER) && ct_result_valid &&
                             (!ct_result_entry_valid ||
                              (ct_result_dst_group >= NUM_GROUPS) ||
                              grp_in_ready[ct_result_dst_group]);
    wire ct_valid_consume = ct_result_consume && ct_result_entry_valid;
    wire ct_invalid_consume = ct_result_consume && !ct_result_entry_valid;
    wire fanout_delivery_accept = (state == ST_CT_DELIVER) && ct_result_valid &&
                                  ct_result_entry_valid &&
                                  (ct_result_dst_group < NUM_GROUPS) &&
                                  grp_in_ready[ct_result_dst_group];
    wire fanout_stall = (state == ST_CT_DELIVER) && ct_result_valid &&
                        ct_result_entry_valid &&
                        (ct_result_dst_group < NUM_GROUPS) &&
                        !grp_in_ready[ct_result_dst_group];
    wire ext_delivery_accept = (state == ST_EXT_ROUTE) &&
                               (ext_route_group < NUM_GROUPS) &&
                               grp_in_ready[ext_route_group];
    wire ext_delivery_stall = (state == ST_EXT_ROUTE) &&
                              (ext_route_group < NUM_GROUPS) &&
                              !grp_in_ready[ext_route_group];

    generate
        genvar pg;
        for (pg = 0; pg < 11+2*NUM_GROUPS; pg = pg + 1) begin : gen_profile_snapshot
            assign profile_snapshot[pg*32 +: 32] = profile_snap[pg];
        end
    endgenerate

    always @(posedge clk) begin
        if (!rst_n) begin
            for (pi = 0; pi < 11+2*NUM_GROUPS; pi = pi + 1) begin
                profile_live[pi] <= 32'd0;
                profile_snap[pi] <= 32'd0;
            end
        end else begin
            if (profile_start) begin
                for (pi = 0; pi < 11+2*NUM_GROUPS; pi = pi + 1)
                    profile_live[pi] <= 32'd0;
            end else if (profile_active) begin
                if (ext_input_accept)       profile_live[0] <= profile_live[0] + 1'b1;
                if (group_output_accept)    profile_live[1] <= profile_live[1] + 1'b1;
                if (ct_lookup_en)           profile_live[2] <= profile_live[2] + 1'b1;
                if (ct_valid_consume)       profile_live[3] <= profile_live[3] + 1'b1;
                if (ct_invalid_consume)     profile_live[4] <= profile_live[4] + 1'b1;
                if (state != ST_IDLE)       profile_live[5] <= profile_live[5] + 1'b1;
                else                        profile_live[6] <= profile_live[6] + 1'b1;
                if (fanout_stall || ext_delivery_stall)
                    profile_live[7] <= profile_live[7] + 1'b1;
                if (fanout_delivery_accept && selected_group != ct_result_dst_group)
                    profile_live[8] <= profile_live[8] + 1'b1;
                if (fanout_delivery_accept && selected_group == ct_result_dst_group)
                    profile_live[9] <= profile_live[9] + 1'b1;
                for (pi = 0; pi < NUM_GROUPS; pi = pi + 1) begin
                    if ((fanout_delivery_accept && ct_result_dst_group == pi) ||
                        (ext_delivery_accept && ext_route_group == pi))
                        profile_live[11+pi] <= profile_live[11+pi] + 1'b1;
                    if ((fanout_stall && ct_result_dst_group == pi) ||
                        (ext_delivery_stall && ext_route_group == pi))
                        profile_live[11+NUM_GROUPS+pi] <= profile_live[11+NUM_GROUPS+pi] + 1'b1;
                end
            end
            if (profile_stop) begin
                for (pi = 0; pi < 11+2*NUM_GROUPS; pi = pi + 1)
                    profile_snap[pi] <= profile_live[pi];
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
                            found = 0;
                            found_idx = 0;
                            for (gi = 0; gi < NUM_GROUPS; gi = gi + 1) begin
                                idx = (rr_priority + gi[GROUP_ID_WIDTH-1:0]) % NUM_GROUPS;
                                if (!found && grp_spike_valid[idx]) begin
                                    selected_group  <= idx;
                                    selected_neuron <= grp_spike_neuron_id[idx*LOCAL_ID_WIDTH +: LOCAL_ID_WIDTH];
                                    grp_spike_ready[idx] <= 1;
                                    found = 1;
                                    found_idx = idx;
                                end
                            end
                            if (found) begin
                                rr_priority <= (found_idx + 1) % NUM_GROUPS;
                                fanout_idx  <= 0;
                                state       <= ST_CT_LOOKUP;
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
                        if (ct_result_valid && ct_result_entry_valid) begin
                            // Deliver spike to destination core group
                            if (ct_result_dst_group >= NUM_GROUPS) begin
                                // Invalid destination group — skip to next fanout
                                state <= ST_CT_NEXT;
                            end else if (grp_in_ready[ct_result_dst_group]) begin
                                grp_in_valid[ct_result_dst_group] <= 1;
                                grp_in_dest_id[ct_result_dst_group*LOCAL_ID_WIDTH +: LOCAL_ID_WIDTH]
                                    <= ct_result_dst_neuron;
                                grp_in_weight[ct_result_dst_group*WEIGHT_WIDTH +: WEIGHT_WIDTH]
                                    <= ct_result_weight;
                                grp_in_exc[ct_result_dst_group] <= ct_result_exc_inh;
                                spike_counter <= spike_counter + 1;
                                state         <= ST_CT_NEXT;
                            end
                            // else: wait for group ready (backpressure)
                        end else begin
                            // No more valid connections — notify learning engine
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
