//-----------------------------------------------------------------------------
// Title         : Integrated SNN Accelerator Top (BD Wrapper + RTL)
// Project       : PYNQ-Z2 SNN Accelerator
// File          : snn_integrated_top.v
// Author        : Jiwoon Lee (@metr0jw)
// Organization  : Kwangwoon University, Seoul, South Korea
// Contact       : jwlee@linux.com
// Description   : Top-level wrapper connecting Block Design (PS + HLS + DMA +
//                 Config Regs) to RTL spike_router and lif_neuron_array.
//                 DDR/FIXED_IO are internal to the BD.
//
//                 Key fixes:
//                 1. HLS ap_none "new spike" detector on spike_in_valid +
//                    payload change (prevents duplication and dropped events
//                    when valid stays high across packets).
//                 2. FIFO bridge on RTL->HLS neuron output so HLS ready
//                    stalls do not drop post-spike bursts.
//
//                 NeuronGroup Note (Phase 6.5):
//                 The NeuronGroup connection topology (Brian2-style) is handled
//                 entirely in HLS weight_memory (flat buffer) and Python host.
//                 The RTL spike_router uses a generic config-based conn_memory
//                 that is NeuronGroup-agnostic. The Python host generates the
//                 routing table entries based on NeuronGroup connections.
//-----------------------------------------------------------------------------

`timescale 1ns / 1ps

module snn_integrated_top #(
    parameter NUM_NEURONS           = 1024,
    parameter NUM_AXONS             = 1024,
    parameter NUM_PARALLEL_UNITS    = 4,
    parameter SPIKE_BUFFER_DEPTH    = 64,
    parameter HLS_NEURON_ID_WIDTH   = 11,
    parameter HLS_MAX_NEURONS       = NUM_NEURONS,
    parameter NEURON_ID_WIDTH       = 10,
    parameter AXON_ID_WIDTH         = 10,
    parameter DATA_WIDTH            = 16,
    parameter WEIGHT_WIDTH          = 8,
    parameter LEAK_WIDTH            = 8,
    parameter THRESHOLD_WIDTH       = 16,
    parameter REFRAC_WIDTH          = 8,
    parameter ROUTER_BUFFER_DEPTH   = 512
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
    // BD <-> RTL Interconnect Signals
    //=========================================================================
    wire        clk;
    wire [0:0]  rst_n_vec;
    wire        rst_n;

    // HLS -> RTL spike (ap_none outputs from BD)
    wire [0:0]  bd_spike_in_valid;
    wire [HLS_NEURON_ID_WIDTH-1:0]  bd_spike_in_neuron_id;
    wire [7:0]  bd_spike_in_weight;
    wire        bd_spike_in_ready;

    // RTL -> HLS spike (ap_none inputs to BD)
    wire        bd_spike_out_valid;
    wire [HLS_NEURON_ID_WIDTH-1:0]  bd_spike_out_neuron_id;
    wire [7:0]  bd_spike_out_weight;
    wire [0:0]  bd_spike_out_ready;

    // SNN Control
    wire [0:0]  bd_snn_enable;
    wire [0:0]  bd_snn_reset;
    wire        bd_snn_ready;
    wire        bd_snn_busy;
    wire        bd_learn_weight_valid;
    wire [3:0]  bd_learn_weight_group;
    wire [6:0]  bd_learn_weight_src;
    wire [6:0]  bd_learn_weight_dst;
    wire [7:0]  bd_learn_weight_data;
    wire        bd_learn_weight_exc;
    wire        bd_learn_weight_is_inter;
    wire [3:0]  bd_learn_weight_dst_group;
    wire [3:0]  bd_learn_weight_fanout_idx;
    wire        bd_learn_weight_ready;

    // Config
    wire        cfg_router_config_we;
    wire [31:0] cfg_router_config_addr;
    wire [31:0] cfg_router_config_wdata;
    wire [31:0] cfg_router_config_rdata;
    wire        cfg_neuron_config_we;
    wire [9:0]  cfg_neuron_config_addr;
    wire [31:0] cfg_neuron_config_wdata;
    wire [15:0] cfg_global_threshold;
    wire [7:0]  cfg_global_leak_rate;
    wire [7:0]  cfg_global_refrac_period;

    // Status
    wire [31:0] router_spike_count;
    wire [31:0] neuron_spike_count;
    wire        fifo_overflow;
    wire [7:0]  active_neurons;
    wire [31:0] cfg_throughput_counter;
    wire [31:0] cfg_service_cycles_counter;
    wire [31:0] neuron_throughput_counter;
    wire [15:0] threshold_out;
    wire [15:0] leak_rate_out;
    wire [0:0]  debug_learning_active;

    // Router config command queue
    localparam ROUTER_CFG_CMD_FIFO_DEPTH = 16;
    wire        router_cfg_cmd_fifo_full;
    wire        router_cfg_cmd_fifo_empty;
    wire        router_cfg_cmd_fifo_almost_full;
    wire        router_cfg_cmd_fifo_almost_empty;
    wire [64-1:0] router_cfg_cmd_fifo_rd_data;
    wire [$clog2(ROUTER_CFG_CMD_FIFO_DEPTH):0] router_cfg_cmd_fifo_count;
    wire        router_cfg_cmd_fifo_overflow;
    wire        router_cfg_cmd_fifo_underflow;
    wire        router_cfg_cmd_fifo_rd_en;
    reg         router_cfg_cmd_pop_pending;
    reg         router_cfg_cmd_valid;
    reg  [31:0] router_cfg_cmd_addr;
    reg  [31:0] router_cfg_cmd_data;

    // Router internal
    wire                       router_spike_valid;
    wire [NEURON_ID_WIDTH-1:0] router_spike_dest_id;
    wire [WEIGHT_WIDTH-1:0]    router_spike_weight;
    wire                       router_spike_exc_inh;
    wire                       router_spike_ready;
    wire                       router_input_valid;
    wire [NEURON_ID_WIDTH-1:0] router_input_neuron_id;
    wire                       router_input_ready;
    wire                       spike_in_in_router_range;

    // Neuron output
    wire                       neuron_spike_valid;
    wire [NEURON_ID_WIDTH-1:0] neuron_spike_id;
    wire                       neuron_spike_ready_wire;
    wire                       router_busy;
    wire                       neuron_array_busy;

    assign rst_n = rst_n_vec[0];
    assign bd_learn_weight_ready = 1'b1;

    //=========================================================================
    // Block Design Instantiation
    //=========================================================================
    design_1_wrapper u_block_design (
        // DDR Interface
        .DDR_addr            (DDR_addr),
        .DDR_ba              (DDR_ba),
        .DDR_cas_n           (DDR_cas_n),
        .DDR_ck_n            (DDR_ck_n),
        .DDR_ck_p            (DDR_ck_p),
        .DDR_cke             (DDR_cke),
        .DDR_cs_n            (DDR_cs_n),
        .DDR_dm              (DDR_dm),
        .DDR_dq              (DDR_dq),
        .DDR_dqs_n           (DDR_dqs_n),
        .DDR_dqs_p           (DDR_dqs_p),
        .DDR_odt             (DDR_odt),
        .DDR_ras_n           (DDR_ras_n),
        .DDR_reset_n         (DDR_reset_n),
        .DDR_we_n            (DDR_we_n),
        // Fixed IO
        .FIXED_IO_ddr_vrn    (FIXED_IO_ddr_vrn),
        .FIXED_IO_ddr_vrp    (FIXED_IO_ddr_vrp),
        .FIXED_IO_mio        (FIXED_IO_mio),
        .FIXED_IO_ps_clk     (FIXED_IO_ps_clk),
        .FIXED_IO_ps_porb    (FIXED_IO_ps_porb),
        .FIXED_IO_ps_srstb   (FIXED_IO_ps_srstb),
        // Clock/Reset
        .clk_100mhz             (clk),
        .rst_n_sync              (rst_n_vec),
        .debug_learning_active   (debug_learning_active),

        // HLS -> RTL spike (BD port names match HLS ap_none ports)
        .spike_in_valid          (bd_spike_in_valid),
        .spike_in_neuron_id      (bd_spike_in_neuron_id),
        .spike_in_weight         (bd_spike_in_weight),
        .spike_in_ready          (bd_spike_in_ready),

        // RTL -> HLS spike
        .spike_out_valid         (bd_spike_out_valid),
        .spike_out_neuron_id     (bd_spike_out_neuron_id),
        .spike_out_weight        (bd_spike_out_weight),
        .spike_out_ready         (bd_spike_out_ready),
        // HLS learned-weight bridge (kept connected for synthesis consistency)
        .learn_weight_valid      (bd_learn_weight_valid),
        .learn_weight_group      (bd_learn_weight_group),
        .learn_weight_src        (bd_learn_weight_src),
        .learn_weight_dst        (bd_learn_weight_dst),
        .learn_weight_data       (bd_learn_weight_data),
        .learn_weight_exc        (bd_learn_weight_exc),
        .learn_weight_is_inter   (bd_learn_weight_is_inter),
        .learn_weight_dst_group  (bd_learn_weight_dst_group),
        .learn_weight_fanout_idx (bd_learn_weight_fanout_idx),
        .learn_weight_ready      (bd_learn_weight_ready),

        // SNN Control
        .snn_enable              (bd_snn_enable),
        .snn_reset               (bd_snn_reset),
        .snn_ready               (bd_snn_ready),
        .snn_busy                (bd_snn_busy),

        // Monitoring
        .threshold_out           (threshold_out),
        .leak_rate_out           (leak_rate_out),

        // Config
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

        // Status feedback
        .cfg_router_spike_count  (router_spike_count),
        .cfg_neuron_spike_count  (neuron_spike_count),
        .cfg_fifo_overflow       (fifo_overflow),
        .cfg_active_neurons      (active_neurons),
        .cfg_throughput_counter  (cfg_throughput_counter),
        .cfg_service_cycles_counter(cfg_service_cycles_counter),
        .cfg_router_busy         (router_busy),
        .cfg_any_core_group_busy (neuron_array_busy),
        .cfg_snn_ready           (bd_snn_ready),
        .cfg_profile_active      (1'b0),
        .cfg_profile_done        (1'b0),
        .cfg_profile_start       (),
        .cfg_profile_stop        (),
        .cfg_profile_index       (),
        .cfg_profile_data        (32'd0),
        .cfg_profile_info        (32'd0)
    );

    //=========================================================================
    // FIX 1: HLS ap_none "new spike" detector
    // HLS can keep spike_in_valid HIGH across multiple packets.
    // Generate one pulse when:
    //   (a) valid rises, or
    //   (b) valid is HIGH and payload (neuron_id/weight) changes.
    // This avoids both FIFO duplication (same payload held) and
    // packet loss (multiple packets while valid stays HIGH).
    //=========================================================================
    reg                              hls_spike_valid_d;
    reg  [HLS_NEURON_ID_WIDTH-1:0]   hls_spike_nid_d;
    reg  [WEIGHT_WIDTH-1:0]          hls_spike_wt_d;
    wire                             hls_spike_payload_changed;
    wire                             hls_spike_event;

    always @(posedge clk) begin
        if (!rst_n || bd_snn_reset[0] || !bd_snn_enable[0]) begin
            hls_spike_valid_d <= 1'b0;
            hls_spike_nid_d   <= {HLS_NEURON_ID_WIDTH{1'b0}};
            hls_spike_wt_d    <= {WEIGHT_WIDTH{1'b0}};
        end else begin
            hls_spike_valid_d <= bd_spike_in_valid[0];
            if (bd_spike_in_valid[0]) begin
                hls_spike_nid_d <= bd_spike_in_neuron_id;
                hls_spike_wt_d  <= bd_spike_in_weight;
            end
        end
    end

    assign hls_spike_payload_changed =
        (bd_spike_in_neuron_id != hls_spike_nid_d) ||
        (bd_spike_in_weight    != hls_spike_wt_d);

    assign hls_spike_event = bd_spike_in_valid[0] &
                             (~hls_spike_valid_d | hls_spike_payload_changed);

    //=========================================================================
    // Spike Input Mux  (HLS priority > recurrent)
    //=========================================================================
    assign spike_in_in_router_range = (bd_spike_in_neuron_id < NUM_NEURONS);
    assign router_input_valid     = (hls_spike_event & spike_in_in_router_range) | neuron_spike_valid;
    assign router_input_neuron_id = (hls_spike_event & spike_in_in_router_range)
                                    ? bd_spike_in_neuron_id[NEURON_ID_WIDTH-1:0]
                                    : neuron_spike_id;
    assign bd_spike_in_ready      = router_input_ready;
    assign neuron_spike_ready_wire = router_input_ready & ~hls_spike_event;

    //=========================================================================
    // FIX 2: FIFO Bridge  RTL -> HLS
    // Neuron pulse can arrive while HLS ready is deasserted. A single hold
    // register drops bursts in that case, so buffer post-spikes in a small
    // local FIFO and stream them to HLS with ready/valid handshaking.
    //=========================================================================
    localparam [HLS_NEURON_ID_WIDTH-1:0] HLS_MAX_NEURON_ID = HLS_MAX_NEURONS[HLS_NEURON_ID_WIDTH-1:0];
    wire [HLS_NEURON_ID_WIDTH-1:0] neuron_spike_id_hls =
        {{(HLS_NEURON_ID_WIDTH-NEURON_ID_WIDTH){1'b0}}, neuron_spike_id};
    // lif_neuron_array exports only spike ID on post-spike path.
    // Keep HLS port width stable by tagging each post-spike with unit weight.
    wire [WEIGHT_WIDTH-1:0] neuron_spike_weight_hls = {{(WEIGHT_WIDTH-1){1'b0}}, 1'b1};
    wire neuron_in_hls_range = (neuron_spike_id_hls < HLS_MAX_NEURON_ID);
    localparam SPIKE_OUT_FIFO_DEPTH = 32;
    localparam SPIKE_OUT_FIFO_AW    = $clog2(SPIKE_OUT_FIFO_DEPTH);
    localparam SPIKE_OUT_FIFO_DW    = HLS_NEURON_ID_WIDTH + WEIGHT_WIDTH;
    localparam [SPIKE_OUT_FIFO_AW-1:0] SPIKE_OUT_FIFO_LAST = {SPIKE_OUT_FIFO_AW{1'b1}};

    reg [SPIKE_OUT_FIFO_DW-1:0] spike_out_fifo_mem [0:SPIKE_OUT_FIFO_DEPTH-1];
    reg [SPIKE_OUT_FIFO_AW-1:0] spike_out_fifo_wr_ptr;
    reg [SPIKE_OUT_FIFO_AW-1:0] spike_out_fifo_rd_ptr;
    reg [SPIKE_OUT_FIFO_AW:0]   spike_out_fifo_count;
    reg                         spike_out_ready_d;
    reg                         neuron_spike_valid_d;
    reg [NEURON_ID_WIDTH-1:0]   neuron_spike_id_d;

    wire spike_out_fifo_empty = (spike_out_fifo_count == 0);
    wire spike_out_fifo_full  = (spike_out_fifo_count == SPIKE_OUT_FIFO_DEPTH);
    // Capture post-spikes with edge/payload-change event detection so they are
    // not dropped when input-path arbitration temporarily deasserts ready.
    wire neuron_spike_payload_changed = (neuron_spike_id != neuron_spike_id_d);
    wire neuron_spike_event = neuron_spike_valid &&
                              (~neuron_spike_valid_d || neuron_spike_payload_changed);
    wire spike_out_fifo_push  = neuron_spike_event && neuron_in_hls_range;
    // HLS drives spike_out_ready as a consume-ack toggle token; pop once per toggle.
    wire spike_out_ready_toggle = (bd_spike_out_ready[0] ^ spike_out_ready_d);
    wire spike_out_fifo_pop   = spike_out_ready_toggle && !spike_out_fifo_empty;
    wire [SPIKE_OUT_FIFO_DW-1:0] spike_out_fifo_head =
        spike_out_fifo_mem[spike_out_fifo_rd_ptr];
    wire [SPIKE_OUT_FIFO_AW-1:0] spike_out_fifo_wr_ptr_next =
        (spike_out_fifo_wr_ptr == SPIKE_OUT_FIFO_LAST) ? {SPIKE_OUT_FIFO_AW{1'b0}} : (spike_out_fifo_wr_ptr + 1'b1);
    wire [SPIKE_OUT_FIFO_AW-1:0] spike_out_fifo_rd_ptr_next =
        (spike_out_fifo_rd_ptr == SPIKE_OUT_FIFO_LAST) ? {SPIKE_OUT_FIFO_AW{1'b0}} : (spike_out_fifo_rd_ptr + 1'b1);

    always @(posedge clk) begin
        if (!rst_n || bd_snn_reset[0] || !bd_snn_enable[0]) begin
            spike_out_fifo_wr_ptr <= {SPIKE_OUT_FIFO_AW{1'b0}};
            spike_out_fifo_rd_ptr <= {SPIKE_OUT_FIFO_AW{1'b0}};
            spike_out_fifo_count  <= {(SPIKE_OUT_FIFO_AW+1){1'b0}};
            spike_out_ready_d     <= 1'b0;
            neuron_spike_valid_d  <= 1'b0;
            neuron_spike_id_d     <= {NEURON_ID_WIDTH{1'b0}};
        end else begin
            spike_out_ready_d <= bd_spike_out_ready[0];
            neuron_spike_valid_d <= neuron_spike_valid;
            if (neuron_spike_valid) begin
                neuron_spike_id_d <= neuron_spike_id;
            end

            if (spike_out_fifo_push && !spike_out_fifo_full) begin
                spike_out_fifo_mem[spike_out_fifo_wr_ptr] <= {neuron_spike_id_hls, neuron_spike_weight_hls};
                spike_out_fifo_wr_ptr <= spike_out_fifo_wr_ptr_next;
            end

            if (spike_out_fifo_pop) begin
                spike_out_fifo_rd_ptr <= spike_out_fifo_rd_ptr_next;
            end

            case ({(spike_out_fifo_push && !spike_out_fifo_full), spike_out_fifo_pop})
                2'b10: spike_out_fifo_count <= spike_out_fifo_count + 1'b1;
                2'b01: spike_out_fifo_count <= spike_out_fifo_count - 1'b1;
                default: spike_out_fifo_count <= spike_out_fifo_count;
            endcase
        end
    end

    assign bd_spike_out_valid     = !spike_out_fifo_empty;
    assign bd_spike_out_neuron_id = spike_out_fifo_head[SPIKE_OUT_FIFO_DW-1:WEIGHT_WIDTH];
    assign bd_spike_out_weight    = spike_out_fifo_head[WEIGHT_WIDTH-1:0];

    //=========================================================================
    // PL-only latency counter (cycles)
    // Measures: first accepted HLS input spike -> first neuron output spike.
    // Exported through cfg_throughput_counter (0x24) for host-side conversion:
    //   latency_ms = cycles / f_clk_hz * 1000
    //=========================================================================
    reg [31:0] pl_latency_cycles_cur;
    reg [31:0] pl_latency_cycles_latched;
    reg        pl_latency_active;
    reg        pl_latency_done;
    reg [31:0] pl_service_cycles_cur;
    reg [31:0] pl_service_cycles_latched;
    reg        pl_service_active;
    reg        pl_service_done;
    reg        pl_service_seen_busy;

    wire hls_input_accept_event = hls_spike_event & router_input_ready;

    always @(posedge clk) begin
        if (!rst_n || bd_snn_reset[0]) begin
            pl_latency_cycles_cur    <= 32'd0;
            pl_latency_cycles_latched<= 32'd0;
            pl_latency_active        <= 1'b0;
            pl_latency_done          <= 1'b0;
            pl_service_cycles_cur    <= 32'd0;
            pl_service_cycles_latched<= 32'd0;
            pl_service_active        <= 1'b0;
            pl_service_done          <= 1'b0;
            pl_service_seen_busy     <= 1'b0;
        end else if (!bd_snn_enable[0]) begin
            // Inter-image idle window: keep last latched value readable.
            pl_latency_cycles_cur <= 32'd0;
            pl_latency_active     <= 1'b0;
            pl_latency_done       <= 1'b0;
            pl_service_cycles_cur <= 32'd0;
            pl_service_active     <= 1'b0;
            pl_service_done       <= 1'b0;
            pl_service_seen_busy  <= 1'b0;
        end else begin
            if (!pl_latency_active && !pl_latency_done) begin
                if (hls_input_accept_event) begin
                    pl_latency_cycles_cur     <= 32'd0;
                    pl_latency_cycles_latched <= 32'd0;
                    pl_latency_active         <= 1'b1;
                end
            end else if (pl_latency_active) begin
                pl_latency_cycles_cur <= pl_latency_cycles_cur + 1'b1;
                if (neuron_spike_event) begin
                    pl_latency_cycles_latched <= pl_latency_cycles_cur + 1'b1;
                    pl_latency_active         <= 1'b0;
                    pl_latency_done           <= 1'b1;
                end
            end

            // Service-time cycles:
            // first accepted input spike -> return to idle after busy period.
            if (!pl_service_active && !pl_service_done) begin
                if (hls_input_accept_event) begin
                    pl_service_cycles_cur     <= 32'd0;
                    pl_service_cycles_latched <= 32'd0;
                    pl_service_active         <= 1'b1;
                    pl_service_seen_busy      <= bd_snn_busy;
                end
            end else if (pl_service_active) begin
                pl_service_cycles_cur <= pl_service_cycles_cur + 1'b1;
                if (bd_snn_busy) begin
                    pl_service_seen_busy <= 1'b1;
                end
                if ((pl_service_seen_busy || bd_snn_busy) && !bd_snn_busy) begin
                    pl_service_cycles_latched <= pl_service_cycles_cur + 1'b1;
                    pl_service_active         <= 1'b0;
                    pl_service_done           <= 1'b1;
                end
            end
        end
    end

    assign cfg_throughput_counter = pl_latency_cycles_latched;
    assign cfg_service_cycles_counter = pl_service_cycles_latched;

    //=========================================================================
    // SNN Status
    //=========================================================================
    assign bd_snn_ready = ~router_busy & ~neuron_array_busy;
    assign bd_snn_busy  = router_busy | neuron_array_busy;

    //=========================================================================
    // Router Config Queue (Loihi-style decoupled control path)
    // Decouple AXI register fanout from router BRAM write path.
    //=========================================================================
    fifo #(
        .DATA_WIDTH(64),
        .DEPTH(ROUTER_CFG_CMD_FIFO_DEPTH)
    ) u_router_cfg_cmd_fifo (
        .clk          (clk),
        .rst_n        (rst_n & ~bd_snn_reset[0]),
        .wr_en        (cfg_router_config_we),
        .wr_data      ({cfg_router_config_addr, cfg_router_config_wdata}),
        .full         (router_cfg_cmd_fifo_full),
        .almost_full  (router_cfg_cmd_fifo_almost_full),
        .rd_en        (router_cfg_cmd_fifo_rd_en),
        .rd_data      (router_cfg_cmd_fifo_rd_data),
        .empty        (router_cfg_cmd_fifo_empty),
        .almost_empty (router_cfg_cmd_fifo_almost_empty),
        .count        (router_cfg_cmd_fifo_count),
        .overflow     (router_cfg_cmd_fifo_overflow),
        .underflow    (router_cfg_cmd_fifo_underflow)
    );

    assign router_cfg_cmd_fifo_rd_en = !router_cfg_cmd_pop_pending && !router_cfg_cmd_fifo_empty;

    always @(posedge clk) begin
        if (!rst_n || bd_snn_reset[0]) begin
            router_cfg_cmd_pop_pending <= 1'b0;
            router_cfg_cmd_valid       <= 1'b0;
            router_cfg_cmd_addr        <= 32'd0;
            router_cfg_cmd_data        <= 32'd0;
        end else begin
            router_cfg_cmd_valid <= 1'b0;

            if (router_cfg_cmd_pop_pending) begin
                router_cfg_cmd_addr        <= router_cfg_cmd_fifo_rd_data[63:32];
                router_cfg_cmd_data        <= router_cfg_cmd_fifo_rd_data[31:0];
                router_cfg_cmd_valid       <= 1'b1;
                router_cfg_cmd_pop_pending <= 1'b0;
            end else if (!router_cfg_cmd_fifo_empty) begin
                router_cfg_cmd_pop_pending <= 1'b1;
            end
        end
    end

    //=========================================================================
    // Spike Router
    //=========================================================================
    spike_router #(
        .NUM_NEURONS     (NUM_NEURONS),
        .MAX_FANOUT      (32),
        .WEIGHT_WIDTH    (WEIGHT_WIDTH),
        .NEURON_ID_WIDTH (NEURON_ID_WIDTH),
        .DELAY_WIDTH     (8),
        .FIFO_DEPTH      (ROUTER_BUFFER_DEPTH)
    ) u_spike_router (
        .clk             (clk),
        .rst_n           (rst_n & ~bd_snn_reset[0]),
        .s_spike_valid   (router_input_valid),
        .s_spike_neuron_id(router_input_neuron_id),
        .s_spike_ready   (router_input_ready),
        .m_spike_valid   (router_spike_valid),
        .m_spike_dest_id (router_spike_dest_id),
        .m_spike_weight  (router_spike_weight),
        .m_spike_exc_inh (router_spike_exc_inh),
        .m_spike_ready   (router_spike_ready),
        .config_we       (1'b0),
        .config_addr     (cfg_router_config_addr),
        .config_data     (32'd0),
        .config_cmd_valid(router_cfg_cmd_valid),
        .config_cmd_addr (router_cfg_cmd_addr),
        .config_cmd_data (router_cfg_cmd_data),
        .config_readdata (cfg_router_config_rdata),
        .routed_spike_count(router_spike_count),
        .router_busy     (router_busy),
        .fifo_overflow   (fifo_overflow)
    );

    //=========================================================================
    // LIF Neuron Array
    //=========================================================================
    lif_neuron_array #(
        .NUM_NEURONS        (NUM_NEURONS),
        .NUM_AXONS          (NUM_AXONS),
        .DATA_WIDTH         (DATA_WIDTH),
        .WEIGHT_WIDTH       (WEIGHT_WIDTH),
        .THRESHOLD_WIDTH    (THRESHOLD_WIDTH),
        .LEAK_WIDTH         (LEAK_WIDTH),
        .REFRAC_WIDTH       (REFRAC_WIDTH),
        .NUM_PARALLEL_UNITS (NUM_PARALLEL_UNITS),
        .SPIKE_BUFFER_DEPTH (SPIKE_BUFFER_DEPTH),
        .USE_BRAM           (1),
        .USE_DSP            (1)
    ) u_neuron_array (
        .clk                    (clk),
        .rst_n                  (rst_n & ~bd_snn_reset[0]),
        .enable                 (bd_snn_enable[0]),
        .s_axis_spike_valid     (router_spike_valid),
        .s_axis_spike_dest_id   (router_spike_dest_id),
        .s_axis_spike_weight    (router_spike_weight),
        .s_axis_spike_exc_inh   (router_spike_exc_inh),
        .s_axis_spike_ready     (router_spike_ready),
        .m_axis_spike_valid     (neuron_spike_valid),
        .m_axis_spike_neuron_id (neuron_spike_id),
        .m_axis_spike_ready     (neuron_spike_ready_wire),
        .config_we              (cfg_neuron_config_we),
        .config_addr            (cfg_neuron_config_addr),
        .config_data            (cfg_neuron_config_wdata),
        .global_threshold       (cfg_global_threshold),
        .global_leak_rate       (cfg_global_leak_rate),
        .global_refrac_period   (cfg_global_refrac_period),
        .spike_count            (neuron_spike_count),
        .array_busy             (neuron_array_busy),
        .throughput_counter     (neuron_throughput_counter),
        .active_neurons         (active_neurons)
    );

endmodule
