//-----------------------------------------------------------------------------
// Title         : SNN Configuration Register File (AXI4-Lite Slave)
// Project       : PYNQ-Z2 SNN Accelerator
// File          : snn_config_regs.v
// Author        : Jiwoon Lee (@metr0jw)
// Organization  : Kwangwoon University, Seoul, South Korea
// Contact       : jwlee@linux.com
// Description   : AXI4-Lite slave register file for runtime configuration
//                 of the SNN accelerator's RTL modules (spike router and
//                 LIF neuron array). Provides PS-accessible registers for:
//                 - Router/neuron connectivity programming
//                 - Global neuron parameter tuning
//                 - Performance monitoring and status readback
//
// Register Map (32-bit registers, byte-addressed):
//   0x00  CONFIG_CTRL     [W]  [1:0] config_target (0=router, 1=neuron)
//   0x04  CONFIG_ADDR     [W]  [31:0] config address for target module
//   0x08  CONFIG_WDATA    [W]  [31:0] config write data (triggers config_we)
//   0x0C  CONFIG_RDATA    [R]  [31:0] config read data from router
//   0x10  NEURON_THRESHOLD[RW] [15:0] global firing threshold
//   0x14  NEURON_PARAMS   [RW] [7:0] leak_rate, [15:8] refrac_period
//   0x18  ROUTER_SPIKE_CNT[R]  [31:0] routed spike count
//   0x1C  NEURON_SPIKE_CNT[R]  [31:0] neuron spike count
//   0x20  STATUS          [R]  [0] fifo_overflow, [1] router_busy,
//                              [2] any_core_group_busy, [3] snn_ready,
//                              [4] profile_active, [5] profile_done,
//                              [13:6] active_neurons
//   0x24  THROUGHPUT      [R]  [31:0] first-spike latency counter
//   0x28  VERSION         [R]  [31:0] = 0x534E4E01 ("SNN" + v1)
//   0x2C  SERVICE_CYCLES  [R]  [31:0] service-time counter
//   0x30  PROFILE_CTRL    [RW] [0] start/clear pulse, [1] stop/latch pulse
//   0x34  PROFILE_INDEX   [RW] [7:0] snapshot counter index
//   0x38  PROFILE_DATA    [R]  [31:0] selected snapshot counter
//   0x3C  PROFILE_INFO    [R]  [31:0] profile metadata
//-----------------------------------------------------------------------------

`timescale 1ns / 1ps

module snn_config_regs #(
    parameter C_S_AXI_ADDR_WIDTH = 6,
    parameter C_S_AXI_DATA_WIDTH = 32
)(
    //=========================================================================
    // AXI4-Lite Slave Interface
    //=========================================================================
    (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 s_axi_aclk CLK" *)
    (* X_INTERFACE_PARAMETER = "ASSOCIATED_BUSIF s_axi, ASSOCIATED_RESET s_axi_aresetn" *)
    input  wire                              s_axi_aclk,

    (* X_INTERFACE_INFO = "xilinx.com:signal:reset:1.0 s_axi_aresetn RST" *)
    (* X_INTERFACE_PARAMETER = "POLARITY ACTIVE_LOW" *)
    input  wire                              s_axi_aresetn,

    // Write Address Channel
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi AWADDR" *)
    input  wire [C_S_AXI_ADDR_WIDTH-1:0]     s_axi_awaddr,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi AWPROT" *)
    input  wire [2:0]                        s_axi_awprot,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi AWVALID" *)
    input  wire                              s_axi_awvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi AWREADY" *)
    output wire                              s_axi_awready,

    // Write Data Channel
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi WDATA" *)
    input  wire [C_S_AXI_DATA_WIDTH-1:0]     s_axi_wdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi WSTRB" *)
    input  wire [C_S_AXI_DATA_WIDTH/8-1:0]   s_axi_wstrb,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi WVALID" *)
    input  wire                              s_axi_wvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi WREADY" *)
    output wire                              s_axi_wready,

    // Write Response Channel
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi BRESP" *)
    output wire [1:0]                        s_axi_bresp,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi BVALID" *)
    output wire                              s_axi_bvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi BREADY" *)
    input  wire                              s_axi_bready,

    // Read Address Channel
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi ARADDR" *)
    input  wire [C_S_AXI_ADDR_WIDTH-1:0]     s_axi_araddr,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi ARPROT" *)
    input  wire [2:0]                        s_axi_arprot,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi ARVALID" *)
    input  wire                              s_axi_arvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi ARREADY" *)
    output wire                              s_axi_arready,

    // Read Data Channel
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi RDATA" *)
    output wire [C_S_AXI_DATA_WIDTH-1:0]     s_axi_rdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi RRESP" *)
    output wire [1:0]                        s_axi_rresp,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi RVALID" *)
    output wire                              s_axi_rvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 s_axi RREADY" *)
    input  wire                              s_axi_rready,

    //=========================================================================
    // Configuration Output Ports (to RTL modules)
    //=========================================================================
    output wire                              router_config_we,
    output wire [31:0]                       router_config_addr,
    output wire [31:0]                       router_config_wdata,
    input  wire [31:0]                       router_config_rdata,

    output wire                              neuron_config_we,
    output wire [9:0]                        neuron_config_addr,
    output wire [31:0]                       neuron_config_wdata,

    output wire [15:0]                       global_threshold,
    output wire [7:0]                        global_leak_rate,
    output wire [7:0]                        global_refrac_period,

    //=========================================================================
    // Status Input Ports (from RTL modules)
    //=========================================================================
    input  wire [31:0]                       router_spike_count,
    input  wire [31:0]                       neuron_spike_count,
    input  wire                              fifo_overflow,
    input  wire [7:0]                        active_neurons,
    input  wire [31:0]                       throughput_counter,
    input  wire [31:0]                       service_cycles_counter,
    input  wire                              router_busy,
    input  wire                              any_core_group_busy,
    input  wire                              snn_ready,
    input  wire                              profile_active,
    input  wire                              profile_done,
    output wire                              profile_start,
    output wire                              profile_stop,
    output wire [15:0]                       profile_expected_count,
    output wire [7:0]                        profile_index,
    input  wire [31:0]                       profile_data,
    input  wire [31:0]                       profile_info
);

    // AXI4-Lite interface parameters
    (* X_INTERFACE_PARAMETER = "PROTOCOL AXI4LITE, DATA_WIDTH 32, ADDR_WIDTH 6" *)

    //=========================================================================
    // Register Address Decode (word-aligned, [5:2] selects register)
    //=========================================================================
    localparam ADDR_CONFIG_CTRL      = 4'h0;   // 0x00
    localparam ADDR_CONFIG_ADDR      = 4'h1;   // 0x04
    localparam ADDR_CONFIG_WDATA     = 4'h2;   // 0x08
    localparam ADDR_CONFIG_RDATA     = 4'h3;   // 0x0C
    localparam ADDR_THRESHOLD        = 4'h4;   // 0x10
    localparam ADDR_NEURON_PARAMS    = 4'h5;   // 0x14
    localparam ADDR_ROUTER_SPIKE_CNT = 4'h6;   // 0x18
    localparam ADDR_NEURON_SPIKE_CNT = 4'h7;   // 0x1C
    localparam ADDR_STATUS           = 4'h8;   // 0x20
    localparam ADDR_THROUGHPUT       = 4'h9;   // 0x24
    localparam ADDR_VERSION          = 4'hA;   // 0x28
    localparam ADDR_SERVICE_CYCLES   = 4'hB;   // 0x2C
    localparam ADDR_PROFILE_CTRL     = 4'hC;   // 0x30
    localparam ADDR_PROFILE_INDEX    = 4'hD;   // 0x34
    localparam ADDR_PROFILE_DATA     = 4'hE;   // 0x38
    localparam ADDR_PROFILE_INFO     = 4'hF;   // 0x3C

    //=========================================================================
    // AXI4-Lite State Machine
    //=========================================================================
    reg  aw_ready;
    reg  w_ready;
    reg  [1:0] b_resp;
    reg  b_valid;
    reg  ar_ready;
    reg  [C_S_AXI_DATA_WIDTH-1:0] r_data;
    reg  [1:0] r_resp;
    reg  r_valid;

    reg  [C_S_AXI_ADDR_WIDTH-1:0] aw_addr;
    reg  [C_S_AXI_ADDR_WIDTH-1:0] ar_addr;
    reg  aw_en;

    assign s_axi_awready = aw_ready;
    assign s_axi_wready  = w_ready;
    assign s_axi_bresp   = b_resp;
    assign s_axi_bvalid  = b_valid;
    assign s_axi_arready = ar_ready;
    assign s_axi_rdata   = r_data;
    assign s_axi_rresp   = r_resp;
    assign s_axi_rvalid  = r_valid;

    //=========================================================================
    // Configuration Registers
    //=========================================================================
    reg  [31:0] reg_config_ctrl;      // [1:0] = target (0=router, 1=neuron)
    reg  [31:0] reg_config_addr;
    reg  [31:0] reg_config_wdata;
    reg  [15:0] reg_threshold;
    reg  [7:0]  reg_leak_rate;
    reg  [7:0]  reg_refrac_period;
    reg  [7:0]  reg_profile_index;
    reg  [15:0] reg_profile_expected_count;
    reg         profile_start_pulse;
    reg         profile_stop_pulse;

    // Config write enable pulse (one-cycle pulse on CONFIG_WDATA write)
    reg         config_we_pulse;
    reg  [1:0]  config_target;

    //=========================================================================
    // Output Assignments
    //=========================================================================
    assign router_config_we    = config_we_pulse & (config_target == 2'd0);
    assign router_config_addr  = reg_config_addr;
    assign router_config_wdata = reg_config_wdata;

    assign neuron_config_we    = config_we_pulse & (config_target == 2'd1);
    assign neuron_config_addr  = reg_config_addr[9:0];
    assign neuron_config_wdata = reg_config_wdata;

    assign global_threshold    = reg_threshold;
    assign global_leak_rate    = reg_leak_rate;
    assign global_refrac_period = reg_refrac_period;
    assign profile_start = profile_start_pulse;
    assign profile_stop  = profile_stop_pulse;
    assign profile_expected_count = reg_profile_expected_count;
    assign profile_index = reg_profile_index;

    //=========================================================================
    // AXI Write Address Channel
    //=========================================================================
    always @(posedge s_axi_aclk) begin
        if (!s_axi_aresetn) begin
            aw_ready <= 1'b0;
            aw_en    <= 1'b1;
            aw_addr  <= {C_S_AXI_ADDR_WIDTH{1'b0}};
        end else begin
            if (~aw_ready && s_axi_awvalid && s_axi_wvalid && aw_en) begin
                aw_ready <= 1'b1;
                aw_addr  <= s_axi_awaddr;
                aw_en    <= 1'b0;
            end else if (s_axi_bready && b_valid) begin
                aw_en    <= 1'b1;
                aw_ready <= 1'b0;
            end else begin
                aw_ready <= 1'b0;
            end
        end
    end

    //=========================================================================
    // AXI Write Data Channel
    //=========================================================================
    always @(posedge s_axi_aclk) begin
        if (!s_axi_aresetn) begin
            w_ready <= 1'b0;
        end else begin
            if (~w_ready && s_axi_wvalid && s_axi_awvalid && aw_en) begin
                w_ready <= 1'b1;
            end else begin
                w_ready <= 1'b0;
            end
        end
    end

    //=========================================================================
    // Register Write Logic
    //=========================================================================
    wire write_en = aw_ready && s_axi_awvalid && w_ready && s_axi_wvalid;
    wire [3:0] write_addr = aw_addr[C_S_AXI_ADDR_WIDTH-1:2];  // Word address

    always @(posedge s_axi_aclk) begin
        if (!s_axi_aresetn) begin
            reg_config_ctrl  <= 32'd0;
            reg_config_addr  <= 32'd0;
            reg_config_wdata <= 32'd0;
            reg_threshold    <= 16'd100;        // Default: 100
            reg_leak_rate    <= 8'h03;          // Default: shift1=3 (tau≈0.875)
            reg_refrac_period <= 8'd10;         // Default: 10 cycles
            config_we_pulse  <= 1'b0;
            config_target    <= 2'd0;
            reg_profile_index <= 8'd0;
            reg_profile_expected_count <= 16'd0;
            profile_start_pulse <= 1'b0;
            profile_stop_pulse  <= 1'b0;
        end else begin
            // Default: clear config_we pulse after one cycle
            config_we_pulse <= 1'b0;
            profile_start_pulse <= 1'b0;
            profile_stop_pulse  <= 1'b0;

            if (write_en) begin
                case (write_addr)
                    ADDR_CONFIG_CTRL: begin
                        if (s_axi_wstrb[0]) begin
                            reg_config_ctrl[7:0]   <= s_axi_wdata[7:0];
                            config_target          <= s_axi_wdata[1:0];
                        end
                    end

                    ADDR_CONFIG_ADDR: begin
                        if (s_axi_wstrb[0]) reg_config_addr[7:0]   <= s_axi_wdata[7:0];
                        if (s_axi_wstrb[1]) reg_config_addr[15:8]  <= s_axi_wdata[15:8];
                        if (s_axi_wstrb[2]) reg_config_addr[23:16] <= s_axi_wdata[23:16];
                        if (s_axi_wstrb[3]) reg_config_addr[31:24] <= s_axi_wdata[31:24];
                    end

                    ADDR_CONFIG_WDATA: begin
                        if (s_axi_wstrb[0]) reg_config_wdata[7:0]   <= s_axi_wdata[7:0];
                        if (s_axi_wstrb[1]) reg_config_wdata[15:8]  <= s_axi_wdata[15:8];
                        if (s_axi_wstrb[2]) reg_config_wdata[23:16] <= s_axi_wdata[23:16];
                        if (s_axi_wstrb[3]) reg_config_wdata[31:24] <= s_axi_wdata[31:24];
                        // Auto-trigger config_we on WDATA write
                        config_we_pulse <= 1'b1;
                    end

                    ADDR_THRESHOLD: begin
                        if (s_axi_wstrb[0]) reg_threshold[7:0]  <= s_axi_wdata[7:0];
                        if (s_axi_wstrb[1]) reg_threshold[15:8] <= s_axi_wdata[15:8];
                    end

                    ADDR_NEURON_PARAMS: begin
                        if (s_axi_wstrb[0]) reg_leak_rate      <= s_axi_wdata[7:0];
                        if (s_axi_wstrb[1]) reg_refrac_period   <= s_axi_wdata[15:8];
                    end

                    ADDR_PROFILE_CTRL: begin
                        if (s_axi_wstrb[0]) begin
                            profile_start_pulse <= s_axi_wdata[0];
                            profile_stop_pulse  <= s_axi_wdata[1];
                        end
                        if (s_axi_wstrb[2]) reg_profile_expected_count[7:0]  <= s_axi_wdata[23:16];
                        if (s_axi_wstrb[3]) reg_profile_expected_count[15:8] <= s_axi_wdata[31:24];
                    end

                    ADDR_PROFILE_INDEX: begin
                        if (s_axi_wstrb[0])
                            reg_profile_index <= s_axi_wdata[7:0];
                    end

                    default: ; // Read-only or reserved registers
                endcase
            end
        end
    end

    //=========================================================================
    // AXI Write Response
    //=========================================================================
    always @(posedge s_axi_aclk) begin
        if (!s_axi_aresetn) begin
            b_valid <= 1'b0;
            b_resp  <= 2'b00;
        end else begin
            if (write_en && ~b_valid) begin
                b_valid <= 1'b1;
                b_resp  <= 2'b00;   // OKAY
            end else if (s_axi_bready && b_valid) begin
                b_valid <= 1'b0;
            end
        end
    end

    //=========================================================================
    // AXI Read Address Channel
    //=========================================================================
    always @(posedge s_axi_aclk) begin
        if (!s_axi_aresetn) begin
            ar_ready <= 1'b0;
            ar_addr  <= {C_S_AXI_ADDR_WIDTH{1'b0}};
        end else begin
            if (~ar_ready && s_axi_arvalid) begin
                ar_ready <= 1'b1;
                ar_addr  <= s_axi_araddr;
            end else begin
                ar_ready <= 1'b0;
            end
        end
    end

    //=========================================================================
    // Register Read Logic
    //=========================================================================
    wire [3:0] read_addr = ar_addr[C_S_AXI_ADDR_WIDTH-1:2];

    always @(posedge s_axi_aclk) begin
        if (!s_axi_aresetn) begin
            r_data  <= 32'd0;
            r_valid <= 1'b0;
            r_resp  <= 2'b00;
        end else begin
            if (ar_ready && s_axi_arvalid && ~r_valid) begin
                r_valid <= 1'b1;
                r_resp  <= 2'b00;   // OKAY
                case (read_addr)
                    ADDR_CONFIG_CTRL:       r_data <= reg_config_ctrl;
                    ADDR_CONFIG_ADDR:       r_data <= reg_config_addr;
                    ADDR_CONFIG_WDATA:      r_data <= reg_config_wdata;
                    ADDR_CONFIG_RDATA:      r_data <= router_config_rdata;
                    ADDR_THRESHOLD:         r_data <= {16'd0, reg_threshold};
                    ADDR_NEURON_PARAMS:     r_data <= {16'd0, reg_refrac_period, reg_leak_rate};
                    ADDR_ROUTER_SPIKE_CNT:  r_data <= router_spike_count;
                    ADDR_NEURON_SPIKE_CNT:  r_data <= neuron_spike_count;
                    ADDR_STATUS:            r_data <= {18'd0, active_neurons,
                                                       profile_done, profile_active,
                                                       snn_ready, any_core_group_busy,
                                                       router_busy, fifo_overflow};
                    ADDR_THROUGHPUT:        r_data <= throughput_counter;
                    ADDR_VERSION:           r_data <= 32'h534E4E01;  // "SNN" + v1
                    ADDR_SERVICE_CYCLES:    r_data <= service_cycles_counter;
                    ADDR_PROFILE_CTRL:      r_data <= {30'd0, profile_done, profile_active};
                    ADDR_PROFILE_INDEX:     r_data <= {24'd0, reg_profile_index};
                    ADDR_PROFILE_DATA:      r_data <= profile_data;
                    ADDR_PROFILE_INFO:      r_data <= profile_info;
                    default:                r_data <= 32'hDEADBEEF;
                endcase
            end else if (r_valid && s_axi_rready) begin
                r_valid <= 1'b0;
            end
        end
    end

    // Suppress unused port warnings
    wire _unused = &{s_axi_awprot, s_axi_arprot, 1'b0};

endmodule
