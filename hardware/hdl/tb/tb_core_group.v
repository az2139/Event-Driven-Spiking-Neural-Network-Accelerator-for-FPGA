`timescale 1ns / 1ps
`include "snn_params.vh"

module tb_core_group;
    parameter NEURONS_PER_GROUP  = `SNN_NEURONS_PER_GROUP;
    parameter DATA_WIDTH         = `SNN_DATA_WIDTH;
    parameter WEIGHT_WIDTH       = `SNN_WEIGHT_WIDTH;
    parameter THRESHOLD_WIDTH    = `SNN_THRESHOLD_WIDTH;
    parameter LEAK_WIDTH         = `SNN_LEAK_WIDTH;
    parameter REFRAC_WIDTH       = `SNN_REFRAC_WIDTH;
    parameter SPIKE_BUFFER_DEPTH = `SNN_SPIKE_BUFFER_DEPTH;
    parameter LOCAL_ID_WIDTH     = `SNN_LOCAL_ID_WIDTH;

    reg clk = 0;
    always #5 clk = ~clk;
    reg rst_n, enable;
    reg ext_spike_valid;
    reg [LOCAL_ID_WIDTH-1:0] ext_spike_dest_id;
    reg [WEIGHT_WIDTH-1:0] ext_spike_weight;
    reg ext_spike_exc_inh;
    wire ext_spike_ready;
    wire out_spike_valid;
    wire [LOCAL_ID_WIDTH-1:0] out_spike_neuron_id;
    reg out_spike_ready;
    reg [THRESHOLD_WIDTH-1:0] global_threshold;
    reg [LEAK_WIDTH-1:0] global_leak_rate;
    reg [REFRAC_WIDTH-1:0] global_refrac_period;
    reg weight_we;
    reg [LOCAL_ID_WIDTH-1:0] weight_src_id, weight_dst_id;
    reg [WEIGHT_WIDTH-1:0] weight_data;
    reg weight_exc;
    wire [15:0] spike_count;
    wire group_busy;

    core_group #(
        .GROUP_ID(0), .NEURONS_PER_GROUP(NEURONS_PER_GROUP),
        .DATA_WIDTH(DATA_WIDTH), .WEIGHT_WIDTH(WEIGHT_WIDTH),
        .THRESHOLD_WIDTH(THRESHOLD_WIDTH), .LEAK_WIDTH(LEAK_WIDTH),
        .REFRAC_WIDTH(REFRAC_WIDTH), .SPIKE_BUFFER_DEPTH(SPIKE_BUFFER_DEPTH)
    ) u_dut (
        .clk(clk), .rst_n(rst_n), .enable(enable),
        .ext_spike_valid(ext_spike_valid), .ext_spike_dest_id(ext_spike_dest_id),
        .ext_spike_weight(ext_spike_weight), .ext_spike_exc_inh(ext_spike_exc_inh),
        .ext_spike_ready(ext_spike_ready),
        .out_spike_valid(out_spike_valid), .out_spike_neuron_id(out_spike_neuron_id),
        .out_spike_ready(out_spike_ready),
        .global_threshold(global_threshold), .global_leak_rate(global_leak_rate),
        .global_refrac_period(global_refrac_period),
        .weight_we(weight_we), .weight_src_id(weight_src_id),
        .weight_dst_id(weight_dst_id), .weight_data(weight_data),
        .weight_exc(weight_exc),
        .spike_count(spike_count), .group_busy(group_busy)
    );

    integer pass_count = 0;
    integer fail_count = 0;

    task automatic check;
        input integer tnum;
        input [8*80-1:0] desc;
        input integer cond;
    begin
        if (cond) begin
            $display("[PASS] Test %0d: %0s", tnum, desc);
            pass_count = pass_count + 1;
        end else begin
            $display("[FAIL] Test %0d: %0s", tnum, desc);
            fail_count = fail_count + 1;
        end
    end
    endtask

    task automatic wait_done;
        integer t;
    begin
        t = 0;
        while (t < 300000) begin
            @(posedge clk);
            t = t + 1;
            if (u_dut.fifo_count == 0 && u_dut.state <= 4'd3)
                t = 300000;
        end
        repeat(5) @(posedge clk);
    end
    endtask

    task automatic wait_spikes;
        input integer expected;
        integer t;
    begin
        t = 0;
        while (spike_count < expected && t < 300000) begin
            @(posedge clk);
            t = t + 1;
        end
        repeat(5) @(posedge clk);
    end
    endtask

    task automatic inject_spike;
        input [LOCAL_ID_WIDTH-1:0] dest;
        input [WEIGHT_WIDTH-1:0] w;
        input exc;
    begin
        @(posedge clk);
        ext_spike_valid   <= 1;
        ext_spike_dest_id <= dest;
        ext_spike_weight  <= w;
        ext_spike_exc_inh <= exc;
        @(posedge clk);
        while (!ext_spike_ready) @(posedge clk);
        ext_spike_valid   <= 0;
    end
    endtask

    task automatic load_weight;
        input [LOCAL_ID_WIDTH-1:0] src;
        input [LOCAL_ID_WIDTH-1:0] dst;
        input [WEIGHT_WIDTH-1:0] w;
        input exc;
    begin
        @(posedge clk);
        weight_we <= 1; weight_src_id <= src; weight_dst_id <= dst;
        weight_data <= w; weight_exc <= exc;
        @(posedge clk);
        weight_we <= 0;
    end
    endtask

    task automatic do_reset;
    begin
        rst_n <= 0; enable <= 0;
        ext_spike_valid <= 0; out_spike_ready <= 1;
        global_threshold <= 16'd10; global_leak_rate <= 8'd0;
        global_refrac_period <= 8'd5;
        weight_we <= 0; weight_src_id <= 0; weight_dst_id <= 0;
        weight_data <= 0; weight_exc <= 0;
        repeat(20) @(posedge clk);
        rst_n <= 1;
        repeat(5) @(posedge clk);
        enable <= 1;
        repeat(5) @(posedge clk);
    end
    endtask

    initial begin
        $display("=========================================================");
        $display("  Core Group Testbench - 128 LIF Neurons");
        $display("=========================================================");
        do_reset;

        // TEST 1: Reset state
        check(1, "spike_count=0 after reset", spike_count == 0);

        // TEST 2: Weight load
        $display("\n--- Test 2: Weight Load ---");
        load_weight(7'd0, 7'd1, 8'd15, 1'b1);
        repeat(5) @(posedge clk);
        // White-box verification: read back weight_mem entry written by load_weight
        if (u_dut.weight_mem[1][WEIGHT_WIDTH-1:0] == 15 &&
            u_dut.weight_mem[1][WEIGHT_WIDTH] == 1'b1)
            check(2, "Weight loaded without error", 1);
        else
            check(2, "Weight loaded without error", 0);

        // TEST 3: Sub-threshold (w=5 < threshold=10)
        $display("\n--- Test 3: Sub-threshold ---");
        inject_spike(7'd5, 8'd5, 1'b1);
        wait_done;
        check(3, "Sub-threshold: spike_count stays 0", spike_count == 0);

        // TEST 4: Supra-threshold (w=15 >= threshold=10)
        $display("\n--- Test 4: Supra-threshold ---");
        inject_spike(7'd10, 8'd15, 1'b1);
        wait_spikes(1);
        check(4, "Supra-threshold: spike_count=1", spike_count == 1);

        // TEST 5: Output spike detected
        $display("\n--- Test 5: Output Spike Detection ---");
        begin : t5
            reg seen;
            integer i;
            seen = 0;
            inject_spike(7'd20, 8'd15, 1'b1);
            for (i = 0; i < 5000; i = i + 1) begin
                @(posedge clk);
                if (out_spike_valid && out_spike_neuron_id == 7'd20) seen = 1;
            end
            check(5, "Output spike for neuron 20 seen", seen);
        end

        // TEST 6: Refractory blocks second fire
        $display("\n--- Test 6: Refractory ---");
        begin : t6
            reg [15:0] sc;
            global_refrac_period <= 8'd200;
            repeat(5) @(posedge clk);
            sc = spike_count;
            inject_spike(7'd15, 8'd15, 1'b1);
            wait_spikes(sc + 1);
            sc = spike_count;
            inject_spike(7'd15, 8'd15, 1'b1);
            wait_done;
            check(6, "Refractory blocks second fire", spike_count == sc);
            global_refrac_period <= 8'd5;
        end

        // TEST 7: Accumulation (3x4=12 > threshold=10)
        $display("\n--- Test 7: Accumulation ---");
        begin : t7
            reg [15:0] sc;
            sc = spike_count;
            inject_spike(7'd30, 8'd4, 1'b1);
            wait_done;
            inject_spike(7'd30, 8'd4, 1'b1);
            wait_done;
            inject_spike(7'd30, 8'd4, 1'b1);
            wait_spikes(sc + 1);
            check(7, "3x4=12>10 fires", spike_count > sc);
        end

        // TEST 8: Inhibitory (no fire)
        $display("\n--- Test 8: Inhibitory ---");
        begin : t8
            reg [15:0] sc;
            sc = spike_count;
            inject_spike(7'd40, 8'd10, 1'b0);
            wait_done;
            check(8, "Inhibitory: no fire", spike_count == sc);
        end

        // TEST 9: Intra-group recurrence (50->51)
        $display("\n--- Test 9: Intra-Group Recurrence ---");
        begin : t9
            reg [15:0] sc;
            load_weight(7'd50, 7'd51, 8'd15, 1'b1);
            repeat(5) @(posedge clk);
            sc = spike_count;
            inject_spike(7'd50, 8'd15, 1'b1);
            wait_spikes(sc + 2);
            check(9, "Recurrence: both 50 and 51 fire", spike_count >= sc + 2);
        end

        // TEST 10: Backpressure
        $display("\n--- Test 10: Backpressure ---");
        begin : t10
            reg [15:0] sc;
            out_spike_ready <= 0;
            sc = spike_count;
            inject_spike(7'd60, 8'd15, 1'b1);
            wait_spikes(sc + 1);
            check(10, "Fires even with backpressure", spike_count > sc);
            out_spike_ready <= 1;
            repeat(100) @(posedge clk);
        end

        // TEST 11: Zero weight
        $display("\n--- Test 11: Zero Weight ---");
        begin : t11
            reg [15:0] sc;
            sc = spike_count;
            inject_spike(7'd70, 8'd0, 1'b1);
            wait_done;
            check(11, "Zero weight: no fire", spike_count == sc);
        end

        // TEST 12: Burst (10 neurons w=15)
        $display("\n--- Test 12: Burst ---");
        begin : t12
            integer bi;
            reg [15:0] sc;
            sc = spike_count;
            for (bi = 0; bi < 10; bi = bi + 1) begin
                inject_spike(80 + bi[6:0], 8'd15, 1'b1);
            end
            wait_spikes(sc + 10);
            check(12, "Burst: 10 spikes all fire", spike_count >= sc + 10);
        end

        // TEST 13: Exact threshold (w=10 == threshold)
        $display("\n--- Test 13: Exact Threshold ---");
        begin : t13
            reg [15:0] sc;
            sc = spike_count;
            inject_spike(7'd100, 8'd10, 1'b1);
            wait_spikes(sc + 1);
            check(13, "Exact threshold fires", spike_count > sc);
        end

        // TEST 14: Multi-neuron diverse weights
        $display("\n--- Test 14: Multi-Neuron Diverse ---");
        begin : t14
            reg [15:0] sc;
            sc = spike_count;
            inject_spike(7'd110, 8'd3, 1'b1);
            wait_done;
            inject_spike(7'd111, 8'd12, 1'b1);
            wait_spikes(sc + 1);
            inject_spike(7'd112, 8'd1, 1'b1);
            wait_done;
            check(14, "Only w=12 neuron fires", spike_count == sc + 1);
        end

        // TEST 15: Leak decay
        $display("\n--- Test 15: Leak Decay ---");
        begin : t15
            reg [15:0] sc;
            global_leak_rate <= 8'd1;
            repeat(5) @(posedge clk);
            sc = spike_count;
            inject_spike(7'd120, 8'd8, 1'b1);
            wait_done;
            repeat(5000) @(posedge clk);
            inject_spike(7'd120, 8'd3, 1'b1);
            wait_done;
            check(15, "After leak decay sub-threshold stays sub-thresh", spike_count == sc);
            global_leak_rate <= 8'd0;
        end

        $display("\n=========================================================");
        $display("  Core Group TB: %0d PASS, %0d FAIL (of %0d)",
                 pass_count, fail_count, pass_count + fail_count);
        $display("=========================================================");
        if (fail_count > 0) $finish(1);
        else $finish(0);
    end

    initial begin
        #100000000;
        $display("[ERROR] Global timeout!");
        $finish(2);
    end
endmodule
