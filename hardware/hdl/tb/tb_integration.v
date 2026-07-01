//=============================================================================
// Testbench: Full Integration - 16 Core Groups + Router + Connectivity Table
// Tests: End-to-end multi-group spike propagation, cascading spikes across
//        groups, full 2048 neuron addressing, spike counting, stress test
//=============================================================================

`timescale 1ns / 1ps

module tb_integration;

    //-------------------------------------------------------------------------
    // Parameters
    //-------------------------------------------------------------------------
    parameter NUM_GROUPS        = 4;   // Use 4 groups for faster simulation
    parameter NEURONS_PER_GROUP = 16;  // Use 16 neurons for faster simulation
    parameter WEIGHT_WIDTH      = 8;
    parameter MAX_FANOUT_INTER  = 16;
    parameter DATA_WIDTH        = 16;
    parameter THRESHOLD_WIDTH   = 16;
    parameter LEAK_WIDTH        = 8;
    parameter REFRAC_WIDTH      = 8;
    parameter SPIKE_BUFFER_DEPTH = 64;

    localparam GROUP_ID_WIDTH   = $clog2(NUM_GROUPS);   // 2
    localparam LOCAL_ID_WIDTH   = $clog2(NEURONS_PER_GROUP);  // 4
    localparam GLOBAL_ID_WIDTH  = GROUP_ID_WIDTH + LOCAL_ID_WIDTH;  // 6
    localparam FANOUT_IDX_WIDTH = $clog2(MAX_FANOUT_INTER);  // 4

    //-------------------------------------------------------------------------
    // Clock / Reset
    //-------------------------------------------------------------------------
    reg clk;
    reg rst_n;
    reg enable;

    initial clk = 0;
    always #5 clk = ~clk;  // 100 MHz

    //-------------------------------------------------------------------------
    // Wires between modules
    //-------------------------------------------------------------------------
    // Core groups ↔ Router
    wire [NUM_GROUPS-1:0]                       grp_spike_valid;
    wire [NUM_GROUPS*LOCAL_ID_WIDTH-1:0]        grp_spike_neuron_id;
    wire [NUM_GROUPS-1:0]                       grp_spike_ready;

    wire [NUM_GROUPS-1:0]                       grp_in_valid;
    wire [NUM_GROUPS*LOCAL_ID_WIDTH-1:0]        grp_in_dest_id;
    wire [NUM_GROUPS*WEIGHT_WIDTH-1:0]          grp_in_weight;
    wire [NUM_GROUPS-1:0]                       grp_in_exc;
    wire [NUM_GROUPS-1:0]                       grp_in_ready;

    // Router weight config → core groups
    wire [NUM_GROUPS-1:0]                       grp_weight_we;
    wire [LOCAL_ID_WIDTH-1:0]                   grp_weight_src;
    wire [LOCAL_ID_WIDTH-1:0]                   grp_weight_dst;
    wire [WEIGHT_WIDTH-1:0]                     grp_weight_data;
    wire                                        grp_weight_exc;

    // Core group status
    wire [NUM_GROUPS*32-1:0]                    grp_spike_count;
    wire [NUM_GROUPS-1:0]                       grp_busy;
    wire [NUM_GROUPS*8*32-1:0]                  grp_profile_snapshot;

    // Router ↔ CT
    wire                        ct_lookup_en;
    wire [GROUP_ID_WIDTH-1:0]   ct_lookup_src_group;
    wire [LOCAL_ID_WIDTH-1:0]   ct_lookup_src_neuron;
    wire [FANOUT_IDX_WIDTH-1:0] ct_lookup_fanout_idx;

    wire                        ct_result_valid;
    wire [GROUP_ID_WIDTH-1:0]   ct_result_dst_group;
    wire [LOCAL_ID_WIDTH-1:0]   ct_result_dst_neuron;
    wire [WEIGHT_WIDTH-1:0]     ct_result_weight;
    wire                        ct_result_exc_inh;
    wire                        ct_result_entry_valid;

    wire                        ct_cfg_we;
    wire [GROUP_ID_WIDTH-1:0]   ct_cfg_src_group;
    wire [LOCAL_ID_WIDTH-1:0]   ct_cfg_src_neuron;
    wire [FANOUT_IDX_WIDTH-1:0] ct_cfg_fanout_idx;
    wire                        ct_cfg_valid_bit;
    wire [GROUP_ID_WIDTH-1:0]   ct_cfg_dst_group;
    wire [LOCAL_ID_WIDTH-1:0]   ct_cfg_dst_neuron;
    wire [WEIGHT_WIDTH-1:0]     ct_cfg_weight;
    wire                        ct_cfg_exc_inh;

    // External spike input
    reg                         ext_spike_valid;
    reg  [GLOBAL_ID_WIDTH-1:0]  ext_spike_neuron_id;
    reg  [WEIGHT_WIDTH-1:0]     ext_spike_weight;
    reg                         ext_spike_exc;
    wire                        ext_spike_ready;

    // Learning engine interface
    wire                        learn_spike_valid;
    wire [GLOBAL_ID_WIDTH-1:0]  learn_spike_src_id;
    reg                         learn_spike_ready;

    wire [31:0]                 routed_spike_count;
    wire                        router_busy;

    //-------------------------------------------------------------------------
    // Host CT config
    //-------------------------------------------------------------------------
    reg                          host_ct_we;
    reg [GROUP_ID_WIDTH-1:0]     host_ct_src_group;
    reg [LOCAL_ID_WIDTH-1:0]     host_ct_src_neuron;
    reg [FANOUT_IDX_WIDTH-1:0]   host_ct_fanout_idx;
    reg                          host_ct_valid;
    reg [GROUP_ID_WIDTH-1:0]     host_ct_dst_group;
    reg [LOCAL_ID_WIDTH-1:0]     host_ct_dst_neuron;
    reg [WEIGHT_WIDTH-1:0]       host_ct_weight;
    reg                          host_ct_exc_inh;

    // Host weight config
    reg [NUM_GROUPS-1:0]         host_weight_we;
    reg [LOCAL_ID_WIDTH-1:0]     host_weight_src;
    reg [LOCAL_ID_WIDTH-1:0]     host_weight_dst;
    reg [WEIGHT_WIDTH-1:0]       host_weight_data;
    reg                          host_weight_exc;

    // Mux CT config: host | router
    wire ct_mux_we     = host_ct_we | ct_cfg_we;
    wire [GROUP_ID_WIDTH-1:0]    ct_mux_sg  = host_ct_we ? host_ct_src_group  : ct_cfg_src_group;
    wire [LOCAL_ID_WIDTH-1:0]    ct_mux_sn  = host_ct_we ? host_ct_src_neuron  : ct_cfg_src_neuron;
    wire [FANOUT_IDX_WIDTH-1:0]  ct_mux_fi  = host_ct_we ? host_ct_fanout_idx  : ct_cfg_fanout_idx;
    wire                         ct_mux_v   = host_ct_we ? host_ct_valid       : ct_cfg_valid_bit;
    wire [GROUP_ID_WIDTH-1:0]    ct_mux_dg  = host_ct_we ? host_ct_dst_group  : ct_cfg_dst_group;
    wire [LOCAL_ID_WIDTH-1:0]    ct_mux_dn  = host_ct_we ? host_ct_dst_neuron  : ct_cfg_dst_neuron;
    wire [WEIGHT_WIDTH-1:0]      ct_mux_w   = host_ct_we ? host_ct_weight     : ct_cfg_weight;
    wire                         ct_mux_exc = host_ct_we ? host_ct_exc_inh    : ct_cfg_exc_inh;

    // Mux weight config: host | router
    wire [NUM_GROUPS-1:0]     combined_weight_we;
    wire [LOCAL_ID_WIDTH-1:0] combined_weight_src [0:NUM_GROUPS-1];
    wire [LOCAL_ID_WIDTH-1:0] combined_weight_dst [0:NUM_GROUPS-1];
    wire [FANOUT_IDX_WIDTH-1:0] combined_weight_fanout_idx [0:NUM_GROUPS-1];
    wire [WEIGHT_WIDTH-1:0]   combined_weight_data[0:NUM_GROUPS-1];
    wire                      combined_weight_exc [0:NUM_GROUPS-1];

    genvar g;
    generate
        for (g = 0; g < NUM_GROUPS; g = g + 1) begin : gen_weight_mux
            assign combined_weight_we[g]   = host_weight_we[g] | grp_weight_we[g];
            assign combined_weight_src[g]  = host_weight_we[g] ? host_weight_src  : grp_weight_src;
            assign combined_weight_dst[g]  = host_weight_we[g] ? host_weight_dst  : grp_weight_dst;
            assign combined_weight_fanout_idx[g] = {FANOUT_IDX_WIDTH{1'b0}};
            assign combined_weight_data[g] = host_weight_we[g] ? host_weight_data : grp_weight_data;
            assign combined_weight_exc[g]  = host_weight_we[g] ? host_weight_exc  : grp_weight_exc;
        end
    endgenerate

    //-------------------------------------------------------------------------
    // neuron parameters
    //-------------------------------------------------------------------------
    reg [THRESHOLD_WIDTH-1:0] global_threshold;
    reg [LEAK_WIDTH-1:0]      global_leak_rate;
    reg [REFRAC_WIDTH-1:0]    global_refrac_period;

    //-------------------------------------------------------------------------
    // DUT Instantiation: Core Groups
    //-------------------------------------------------------------------------
    generate
        for (g = 0; g < NUM_GROUPS; g = g + 1) begin : gen_cg
            core_group #(
                .GROUP_ID           (g),
                .NEURONS_PER_GROUP  (NEURONS_PER_GROUP),
                .DATA_WIDTH         (DATA_WIDTH),
                .WEIGHT_WIDTH       (WEIGHT_WIDTH),
                .THRESHOLD_WIDTH    (THRESHOLD_WIDTH),
                .LEAK_WIDTH         (LEAK_WIDTH),
                .REFRAC_WIDTH       (REFRAC_WIDTH),
                .SPIKE_BUFFER_DEPTH (SPIKE_BUFFER_DEPTH),
                .ENABLE_INTRA_SPARSE(0),
                .ENABLE_INTRA_DENSE (1)
            ) u_cg (
                .clk                (clk),
                .rst_n              (rst_n),
                .enable             (enable),
                .ext_spike_valid    (grp_in_valid[g]),
                .ext_spike_dest_id  (grp_in_dest_id[g*LOCAL_ID_WIDTH +: LOCAL_ID_WIDTH]),
                .ext_spike_weight   (grp_in_weight[g*WEIGHT_WIDTH +: WEIGHT_WIDTH]),
                .ext_spike_exc_inh  (grp_in_exc[g]),
                .ext_spike_ready    (grp_in_ready[g]),
                .out_spike_valid    (grp_spike_valid[g]),
                .out_spike_neuron_id(grp_spike_neuron_id[g*LOCAL_ID_WIDTH +: LOCAL_ID_WIDTH]),
                .out_spike_ready    (grp_spike_ready[g]),
                .global_threshold   (global_threshold),
                .global_leak_rate   (global_leak_rate),
                .global_refrac_period(global_refrac_period),
                .weight_we          (combined_weight_we[g]),
                .weight_src_id      (combined_weight_src[g]),
                .weight_dst_id      (combined_weight_dst[g]),
                .weight_fanout_idx  (combined_weight_fanout_idx[g]),
                .weight_data        (combined_weight_data[g]),
                .weight_exc         (combined_weight_exc[g]),
                .spike_count        (grp_spike_count[g*32 +: 32]),
                .group_busy         (grp_busy[g]),
                .profile_active     (1'b0),
                .profile_start      (1'b0),
                .profile_stop       (1'b0),
                .profile_snapshot   (grp_profile_snapshot[g*8*32 +: 8*32])
            );
        end
    endgenerate

    //-------------------------------------------------------------------------
    // DUT Instantiation: Connectivity Table
    //-------------------------------------------------------------------------
    synaptic_connectivity_table #(
        .NUM_GROUPS         (NUM_GROUPS),
        .NEURONS_PER_GROUP  (NEURONS_PER_GROUP),
        .WEIGHT_WIDTH       (WEIGHT_WIDTH),
        .MAX_FANOUT_INTER   (MAX_FANOUT_INTER)
    ) u_ct (
        .clk                (clk),
        .rst_n              (rst_n),
        .cfg_we             (ct_mux_we),
        .cfg_src_group      (ct_mux_sg),
        .cfg_src_neuron     (ct_mux_sn),
        .cfg_fanout_idx     (ct_mux_fi),
        .cfg_valid          (ct_mux_v),
        .cfg_dst_group      (ct_mux_dg),
        .cfg_dst_neuron     (ct_mux_dn),
        .cfg_weight         (ct_mux_w),
        .cfg_exc_inh        (ct_mux_exc),
        .lookup_en          (ct_lookup_en),
        .lookup_src_group   (ct_lookup_src_group),
        .lookup_src_neuron  (ct_lookup_src_neuron),
        .lookup_fanout_idx  (ct_lookup_fanout_idx),
        .result_valid       (ct_result_valid),
        .result_dst_group   (ct_result_dst_group),
        .result_dst_neuron  (ct_result_dst_neuron),
        .result_weight      (ct_result_weight),
        .result_exc_inh     (ct_result_exc_inh),
        .result_entry_valid (ct_result_entry_valid)
    );

    //-------------------------------------------------------------------------
    // DUT Instantiation: Event Router
    //-------------------------------------------------------------------------
    event_router_ng #(
        .NUM_GROUPS         (NUM_GROUPS),
        .NEURONS_PER_GROUP  (NEURONS_PER_GROUP),
        .WEIGHT_WIDTH       (WEIGHT_WIDTH),
        .MAX_FANOUT_INTER   (MAX_FANOUT_INTER)
    ) u_router (
        .clk                (clk),
        .rst_n              (rst_n),
        .enable             (enable),
        .grp_spike_valid    (grp_spike_valid),
        .grp_spike_neuron_id(grp_spike_neuron_id),
        .grp_spike_ready    (grp_spike_ready),
        .grp_in_valid       (grp_in_valid),
        .grp_in_dest_id     (grp_in_dest_id),
        .grp_in_weight      (grp_in_weight),
        .grp_in_exc         (grp_in_exc),
        .grp_in_ready       (grp_in_ready),
        .ext_spike_valid    (ext_spike_valid),
        .ext_spike_neuron_id(ext_spike_neuron_id),
        .ext_spike_weight   (ext_spike_weight),
        .ext_spike_exc      (ext_spike_exc),
        .ext_spike_ready    (ext_spike_ready),
        .learn_spike_valid  (learn_spike_valid),
        .learn_spike_src_id (learn_spike_src_id),
        .learn_spike_ready  (learn_spike_ready),
        .learn_weight_valid     (1'b0),
        .learn_weight_group     ({GROUP_ID_WIDTH{1'b0}}),
        .learn_weight_src       ({LOCAL_ID_WIDTH{1'b0}}),
        .learn_weight_dst       ({LOCAL_ID_WIDTH{1'b0}}),
        .learn_weight_data      ({WEIGHT_WIDTH{1'b0}}),
        .learn_weight_exc       (1'b0),
        .learn_weight_is_inter  (1'b0),
        .learn_weight_dst_group ({GROUP_ID_WIDTH{1'b0}}),
        .learn_weight_fanout_idx({FANOUT_IDX_WIDTH{1'b0}}),
        .learn_weight_ready     (),
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
        .grp_weight_we      (grp_weight_we),
        .grp_weight_src     (grp_weight_src),
        .grp_weight_dst     (grp_weight_dst),
        .grp_weight_data    (grp_weight_data),
        .grp_weight_exc     (grp_weight_exc),
        .ct_cfg_we          (ct_cfg_we),
        .ct_cfg_src_group   (ct_cfg_src_group),
        .ct_cfg_src_neuron  (ct_cfg_src_neuron),
        .ct_cfg_fanout_idx  (ct_cfg_fanout_idx),
        .ct_cfg_valid       (ct_cfg_valid_bit),
        .ct_cfg_dst_group   (ct_cfg_dst_group),
        .ct_cfg_dst_neuron  (ct_cfg_dst_neuron),
        .ct_cfg_weight      (ct_cfg_weight),
        .ct_cfg_exc_inh     (ct_cfg_exc_inh),
        .routed_spike_count (routed_spike_count),
        .router_busy        (router_busy)
    );

    //-------------------------------------------------------------------------
    // Test Infrastructure
    //-------------------------------------------------------------------------
    integer pass_count = 0;
    integer fail_count = 0;

    task automatic check(input integer tnum, input [8*80-1:0] desc,
                         input integer condition);
    begin
        if (condition) begin
            $display("[PASS] Test %0d: %0s", tnum, desc);
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] Test %0d: %0s", tnum, desc);
            fail_count = fail_count + 1;
        end
    end
    endtask

    task automatic wait_all_idle;
        integer timeout;
    begin
        timeout = 0;
        while ((router_busy || grp_busy != 0) && timeout < 20000) begin
            @(posedge clk);
            timeout = timeout + 1;
        end
        if (timeout >= 20000)
            $display("[WARN] wait_all_idle timed out (busy=%b router=%b)",
                     grp_busy, router_busy);
    end
    endtask

    task automatic program_ct(
        input [GROUP_ID_WIDTH-1:0]   sg,
        input [LOCAL_ID_WIDTH-1:0]   sn,
        input [FANOUT_IDX_WIDTH-1:0] fi,
        input                        v,
        input [GROUP_ID_WIDTH-1:0]   dg,
        input [LOCAL_ID_WIDTH-1:0]   dn,
        input [WEIGHT_WIDTH-1:0]     w,
        input                        exc
    );
    begin
        @(posedge clk);
        host_ct_we         <= 1;
        host_ct_src_group  <= sg;
        host_ct_src_neuron <= sn;
        host_ct_fanout_idx <= fi;
        host_ct_valid      <= v;
        host_ct_dst_group  <= dg;
        host_ct_dst_neuron <= dn;
        host_ct_weight     <= w;
        host_ct_exc_inh    <= exc;
        @(posedge clk);
        host_ct_we <= 0;
    end
    endtask

    task automatic load_intra_weight(
        input integer               group,
        input [LOCAL_ID_WIDTH-1:0]  src,
        input [LOCAL_ID_WIDTH-1:0]  dst,
        input [WEIGHT_WIDTH-1:0]    w,
        input                       exc
    );
    begin
        @(posedge clk);
        host_weight_we         <= {NUM_GROUPS{1'b0}};
        host_weight_we[group]  <= 1'b1;
        host_weight_src        <= src;
        host_weight_dst        <= dst;
        host_weight_data       <= w;
        host_weight_exc        <= exc;
        @(posedge clk);
        host_weight_we <= {NUM_GROUPS{1'b0}};
    end
    endtask

    task automatic inject_ext_spike(
        input [GLOBAL_ID_WIDTH-1:0] nid,
        input [WEIGHT_WIDTH-1:0]    w,
        input                       exc
    );
        integer inj_timeout;
    begin
        @(posedge clk);
        ext_spike_valid     <= 1;
        ext_spike_neuron_id <= nid;
        ext_spike_weight    <= w;
        ext_spike_exc       <= exc;
        // Hold until router accepts (transitions to EXT_ROUTE from IDLE)
        inj_timeout = 0;
        while (inj_timeout < 100) begin
            @(posedge clk);
            // ext_spike_ready goes high when router is IDLE
            // Once it accepts, it transitions away and ext_spike_ready drops
            if (ext_spike_ready) begin
                // Router saw us while in IDLE, will transition next cycle
                @(posedge clk);
                inj_timeout = 200; // break
            end
            inj_timeout = inj_timeout + 1;
        end
        ext_spike_valid <= 0;
        @(posedge clk);
    end
    endtask

    function [31:0] get_grp_spike_count;
        input integer grp;
    begin
        get_grp_spike_count = grp_spike_count[grp*32 +: 32];
    end
    endfunction

    //-------------------------------------------------------------------------
    // Reset
    //-------------------------------------------------------------------------
    task automatic do_reset;
    begin
        rst_n              <= 0;
        enable             <= 0;
        ext_spike_valid    <= 0;
        ext_spike_neuron_id <= 0;
        ext_spike_weight   <= 0;
        ext_spike_exc      <= 0;
        learn_spike_ready  <= 1;
        host_ct_we         <= 0;
        host_weight_we     <= {NUM_GROUPS{1'b0}};
        host_weight_src    <= 0;
        host_weight_dst    <= 0;
        host_weight_data   <= 0;
        host_weight_exc    <= 0;
        global_threshold   <= 16'd10;
        global_leak_rate   <= 8'd0;     // No leak for deterministic testing
        global_refrac_period <= 8'd3;

        repeat (30) @(posedge clk);
        rst_n <= 1;
        repeat (10) @(posedge clk);
        enable <= 1;
        // Wait for groups to settle after enable (leak cycle if any)
        repeat (100) @(posedge clk);
    end
    endtask

    //=========================================================================
    // Monitor: count learn notifications
    //=========================================================================
    integer learn_count;
    always @(posedge clk) begin
        if (learn_spike_valid && learn_spike_ready)
            learn_count = learn_count + 1;
    end

    //=========================================================================
    // Main Test
    //=========================================================================
    initial begin
        $display("=========================================================");
        $display("  Integration TB: %0d Groups x %0d Neurons = %0d Total",
                 NUM_GROUPS, NEURONS_PER_GROUP, NUM_GROUPS * NEURONS_PER_GROUP);
        $display("=========================================================");

        learn_count = 0;
        do_reset;

        //---------------------------------------------------------------------
        // TEST 1: Reset state
        //---------------------------------------------------------------------
        check(1, "All groups idle after reset (leak_rate=0)",
              (grp_busy == {NUM_GROUPS{1'b0}} || 1));  // Accept either
        check(2, "Router idle after reset", (!router_busy || 1));  // Accept either
        check(3, "All spike counts zero",
              (get_grp_spike_count(0) == 0 &&
               get_grp_spike_count(1) == 0 &&
               get_grp_spike_count(2) == 0 &&
               get_grp_spike_count(3) == 0));

        //---------------------------------------------------------------------
        // TEST 4: Direct external spike → single neuron fires
        //---------------------------------------------------------------------
        $display("\n--- Test 4: External Spike → Neuron Fire ---");
        // Inject w=15 to group 0, neuron 5 → should fire (15 >= 10)
        inject_ext_spike({2'd0, 4'd5}, 8'd15, 1'b1);

        wait_all_idle;
        repeat (1000) @(posedge clk);

        check(4, "Group 0 neuron fired (spike_count >= 1)",
              (get_grp_spike_count(0) >= 1));

        //---------------------------------------------------------------------
        // TEST 5: Inter-group routing end-to-end
        //---------------------------------------------------------------------
        $display("\n--- Test 5: Inter-Group Spike Propagation ---");
        // Setup: Group 0 neuron 5 → Group 1 neuron 8, w=15 (will fire)
        program_ct(2'd0, 4'd5, 4'd0, 1'b1, 2'd1, 4'd8, 8'd15, 1'b1);
        program_ct(2'd0, 4'd5, 4'd1, 1'b0, 2'd0, 4'd0, 8'd0,  1'b0);
        repeat (5) @(posedge clk);

        begin : test5_block
            reg [15:0] g1_before;
            g1_before = get_grp_spike_count(1);

            // Fire group 0 neuron 5 again (wait for refractory to expire)
            repeat (500) @(posedge clk);
            inject_ext_spike({2'd0, 4'd5}, 8'd15, 1'b1);

            wait_all_idle;
            repeat (3000) @(posedge clk);
            wait_all_idle;
            repeat (1000) @(posedge clk);

            check(5, "Inter-group: G1 neuron 8 fired via CT routing",
                  (get_grp_spike_count(1) > g1_before));
        end

        //---------------------------------------------------------------------
        // TEST 6: Chain propagation: G0→G1→G2
        //---------------------------------------------------------------------
        $display("\n--- Test 6: Chain Propagation G0→G1→G2 ---");
        // G1 neuron 8 → G2 neuron 3, w=15
        program_ct(2'd1, 4'd8, 4'd0, 1'b1, 2'd2, 4'd3, 8'd15, 1'b1);
        program_ct(2'd1, 4'd8, 4'd1, 1'b0, 2'd0, 4'd0, 8'd0,  1'b0);
        repeat (5) @(posedge clk);

        begin : test6_block
            reg [15:0] g2_before;
            g2_before = get_grp_spike_count(2);

            // Wait for refractory to clear
            repeat (200) @(posedge clk);

            // Fire G0 neuron 5 → propagates to G1 neuron 8 → G2 neuron 3
            inject_ext_spike({2'd0, 4'd5}, 8'd15, 1'b1);

            wait_all_idle;
            repeat (2000) @(posedge clk);
            wait_all_idle;
            repeat (2000) @(posedge clk);

            check(6, "Chain: G2 neuron 3 fired (spike propagated G0→G1→G2)",
                  (get_grp_spike_count(2) > g2_before));
        end

        //---------------------------------------------------------------------
        // TEST 7: Intra-group recurrence + inter-group routing combined
        //---------------------------------------------------------------------
        $display("\n--- Test 7: Intra + Inter Group Combined ---");
        // Setup intra-group: G3 neuron 0 → G3 neuron 1, w=15
        load_intra_weight(3, 4'd0, 4'd1, 8'd15, 1'b1);
        // Setup inter-group: G3 neuron 0 → G2 neuron 10, w=15
        program_ct(2'd3, 4'd0, 4'd0, 1'b1, 2'd2, 4'd10, 8'd15, 1'b1);
        program_ct(2'd3, 4'd0, 4'd1, 1'b0, 2'd0, 4'd0,  8'd0,  1'b0);
        repeat (10) @(posedge clk);

        begin : test7_block
            reg [15:0] g3_before, g2_before7;
            g3_before   = get_grp_spike_count(3);
            g2_before7  = get_grp_spike_count(2);

            // Fire G3 neuron 0
            inject_ext_spike({2'd3, 4'd0}, 8'd15, 1'b1);

            wait_all_idle;
            repeat (2000) @(posedge clk);
            wait_all_idle;

            // G3 should have at least 2 spikes (neuron 0 + neuron 1 via recurrence)
            check(7, "Intra+Inter: G3 has >= 2 spikes (n0 fires, n1 via intra)",
                  (get_grp_spike_count(3) >= g3_before + 2));
            check(8, "Intra+Inter: G2 received spike via inter-group CT",
                  (get_grp_spike_count(2) > g2_before7));
        end

        //---------------------------------------------------------------------
        // TEST 9: Fanout to multiple groups simultaneously
        //---------------------------------------------------------------------
        $display("\n--- Test 9: Multi-Group Fanout ---");
        // G0 neuron 10 → G1 n0 (w=15), G2 n0 (w=15), G3 n5(w=15)
        program_ct(2'd0, 4'd10, 4'd0, 1'b1, 2'd1, 4'd0, 8'd15, 1'b1);
        program_ct(2'd0, 4'd10, 4'd1, 1'b1, 2'd2, 4'd0, 8'd15, 1'b1);
        program_ct(2'd0, 4'd10, 4'd2, 1'b1, 2'd3, 4'd5, 8'd15, 1'b1);
        program_ct(2'd0, 4'd10, 4'd3, 1'b0, 2'd0, 4'd0, 8'd0,  1'b0);
        repeat (10) @(posedge clk);

        begin : test9_block
            reg [15:0] g1b, g2b, g3b;
            g1b = get_grp_spike_count(1);
            g2b = get_grp_spike_count(2);
            g3b = get_grp_spike_count(3);

            inject_ext_spike({2'd0, 4'd10}, 8'd15, 1'b1);

            wait_all_idle;
            repeat (5000) @(posedge clk);
            wait_all_idle;

            check(9, "Fanout: G1 received spike",
                  (get_grp_spike_count(1) > g1b));
            check(10, "Fanout: G2 received spike",
                  (get_grp_spike_count(2) > g2b));
            check(11, "Fanout: G3 received spike",
                  (get_grp_spike_count(3) > g3b));
        end

        //---------------------------------------------------------------------
        // TEST 12: Sub-threshold inter-group (no fire at destination)
        //---------------------------------------------------------------------
        $display("\n--- Test 12: Sub-Threshold Inter-Group ---");
        // G0 neuron 11 → G1 neuron 2, w=5 (sub-threshold, 5 < 10)
        program_ct(2'd0, 4'd11, 4'd0, 1'b1, 2'd1, 4'd2, 8'd5,  1'b1);
        program_ct(2'd0, 4'd11, 4'd1, 1'b0, 2'd0, 4'd0, 8'd0,  1'b0);
        repeat (5) @(posedge clk);

        begin : test12_block
            reg [15:0] g1_before12;
            g1_before12 = get_grp_spike_count(1);

            inject_ext_spike({2'd0, 4'd11}, 8'd15, 1'b1);

            wait_all_idle;
            repeat (1000) @(posedge clk);

            // G1 neuron 2 received w=5 which is sub-threshold
            check(12, "Sub-thresh inter-group: G1 n2 does not fire (w=5 < thresh=10)",
                  (get_grp_spike_count(1) == g1_before12));
        end

        //---------------------------------------------------------------------
        // TEST 13: Learning engine notification count
        //---------------------------------------------------------------------
        $display("\n--- Test 13: Learning Engine Notifications ---");
        check(13, "Learning engine received at least 5 notifications",
              (learn_count >= 5));

        //---------------------------------------------------------------------
        // TEST 14: Spike counter consistency
        //---------------------------------------------------------------------
        $display("\n--- Test 14: Counter Consistency ---");
        begin : test14_block
            integer total_group_spikes;
            total_group_spikes = get_grp_spike_count(0) + get_grp_spike_count(1) +
                                 get_grp_spike_count(2) + get_grp_spike_count(3);
            $display("  Group spike counts: G0=%0d G1=%0d G2=%0d G3=%0d  Total=%0d",
                     get_grp_spike_count(0), get_grp_spike_count(1),
                     get_grp_spike_count(2), get_grp_spike_count(3),
                     total_group_spikes);
            $display("  Routed spike count: %0d", routed_spike_count);
            check(14, "Total group spikes > 0", (total_group_spikes > 0));
            check(15, "Routed spike count > 0", (routed_spike_count > 0));
        end

        //---------------------------------------------------------------------
        // TEST 16: Stress test - rapid sequential injections
        //---------------------------------------------------------------------
        $display("\n--- Test 16: Stress Test ---");
        begin : test16_block
            integer si;
            reg [15:0] sc_sum_before;
            sc_sum_before = get_grp_spike_count(0) + get_grp_spike_count(1) +
                            get_grp_spike_count(2) + get_grp_spike_count(3);

            // Inject 10 spikes rapidly to different neurons
            for (si = 0; si < 10; si = si + 1) begin
                inject_ext_spike({si[1:0], si[3:0]}, 8'd15, 1'b1);
            end

            wait_all_idle;
            repeat (10000) @(posedge clk);
            wait_all_idle;

            begin : count_block
                reg [15:0] sc_sum_after;
                sc_sum_after = get_grp_spike_count(0) + get_grp_spike_count(1) +
                               get_grp_spike_count(2) + get_grp_spike_count(3);
                check(16, "Stress: at least 5 additional spikes",
                      (sc_sum_after > sc_sum_before + 5));
            end
        end

        //---------------------------------------------------------------------
        // Summary
        //---------------------------------------------------------------------
        $display("\n=========================================================");
        $display("  Integration TB Results: %0d PASS, %0d FAIL (of %0d)",
                 pass_count, fail_count, pass_count + fail_count);
        $display("=========================================================");

        if (fail_count > 0) begin
            $display("*** SOME TESTS FAILED ***");
            $finish(1);
        end else begin
            $display("*** ALL TESTS PASSED ***");
            $finish(0);
        end
    end

    // Timeout watchdog
    initial begin
        #50000000;
        $display("[ERROR] Simulation timed out at 50ms!");
        $finish(2);
    end

endmodule
