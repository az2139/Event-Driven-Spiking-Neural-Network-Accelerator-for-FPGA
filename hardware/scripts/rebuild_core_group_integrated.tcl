#-----------------------------------------------------------------------------
# Rebuild the hierarchical core-group SNN bitstream with profile support.
# Top: snn_core_group_top
#
# Usage: vivado -mode batch -source hardware/scripts/rebuild_core_group_integrated.tcl
#-----------------------------------------------------------------------------

set script_dir [file dirname [file normalize [info script]]]
set project_dir [file normalize [file join $script_dir ../..]]
set build_dir   "${project_dir}/hardware/build/snn_core_group_profile"
set rtl_dir     "${project_dir}/hardware/hdl/rtl"
set ip_repo     "${project_dir}/hardware/ip_repo"
set output_dir  "${project_dir}/outputs"
set part        "xc7z020clg400-2"
# Integrated default clock for generated bitstream.
# Override with SNN_PL_CLK_MHZ when sweeping lower/higher operating points.
set pl_clk_mhz 100
if {[info exists ::env(SNN_PL_CLK_MHZ)] && $::env(SNN_PL_CLK_MHZ) ne ""} {
    set pl_clk_mhz [expr {int($::env(SNN_PL_CLK_MHZ))}]
}
puts "Using processing_system7_0/FCLK_CLK0 frequency: ${pl_clk_mhz} MHz"

# Clean previous build
file delete -force $build_dir

# Create project
create_project snn_core_group_profile $build_dir -part $part -force
set_property target_language Verilog [current_project]
# Use automatic source management so module-reference BD cells resolve correctly.
set_property source_mgmt_mode All [current_project]
# Keep latest config-reg RTL in project so BD can fall back to module reference
# when a stale packaged IP is present in ip_repo.
add_files -norecurse "${rtl_dir}/common/snn_config_regs.v"
update_compile_order -fileset sources_1
# Add the lightweight inference/profile HLS IP repository.
set hls_ip_repo_generated "${project_dir}/hardware/hls/hls_inference_profile_output/hls/impl/ip"
set hls_ip_repo_cached "${ip_repo}/snn_inference_profile_hls_1_0"
if {[file exists $hls_ip_repo_generated]} {
    set hls_ip_repo $hls_ip_repo_generated
} elseif {[file exists $hls_ip_repo_cached]} {
    set hls_ip_repo $hls_ip_repo_cached
} else {
    puts "ERROR: Inference/profile HLS IP repository not found in:"
    puts "  - $hls_ip_repo_generated"
    puts "  - $hls_ip_repo_cached"
    puts "Build it first: hardware/hls/scripts/build_inference_profile_hls.sh --clean"
    exit 1
}
puts "Using HLS IP repository: $hls_ip_repo"
set_property ip_repo_paths [list \
    $hls_ip_repo \
] [current_project]
update_ip_catalog

# =============================================================================
# Create Block Design
# =============================================================================
create_bd_design "design_1"

# --- Zynq PS ---
create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 processing_system7_0

# Apply PYNQ-Z2 preset
set_property -dict [list \
    CONFIG.PCW_USE_S_AXI_HP0 {1} \
    CONFIG.PCW_USE_FABRIC_INTERRUPT {1} \
    CONFIG.PCW_IRQ_F2P_INTR {1} \
    CONFIG.PCW_FPGA0_PERIPHERAL_FREQMHZ $pl_clk_mhz \
    CONFIG.PCW_USE_M_AXI_GP0 {1} \
    CONFIG.PCW_QSPI_PERIPHERAL_ENABLE {1} \
    CONFIG.PCW_ENET0_PERIPHERAL_ENABLE {1} \
    CONFIG.PCW_SD0_PERIPHERAL_ENABLE {1} \
    CONFIG.PCW_UART0_PERIPHERAL_ENABLE {1} \
    CONFIG.PCW_USB0_PERIPHERAL_ENABLE {1} \
    CONFIG.PCW_GPIO_MIO_GPIO_ENABLE {1} \
    CONFIG.PCW_TTC0_PERIPHERAL_ENABLE {0} \
] [get_bd_cells processing_system7_0]

# --- Make DDR and FIXED_IO external (required for Zynq PS) ---
create_bd_intf_port -mode Master -vlnv xilinx.com:interface:ddrx_rtl:1.0 DDR
connect_bd_intf_net [get_bd_intf_pins processing_system7_0/DDR] [get_bd_intf_ports DDR]

create_bd_intf_port -mode Master -vlnv xilinx.com:display_processing_system7:fixedio_rtl:1.0 FIXED_IO
connect_bd_intf_net [get_bd_intf_pins processing_system7_0/FIXED_IO] [get_bd_intf_ports FIXED_IO]

# --- Processor System Reset ---
create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 proc_sys_reset_0

# --- AXI DMA (spike stream) ---
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 axi_dma_0
set_property -dict [list \
    CONFIG.c_include_sg {0} \
    CONFIG.c_sg_include_stscntrl_strm {0} \
    CONFIG.c_sg_length_width {23} \
    CONFIG.c_mm2s_burst_size {16} \
    CONFIG.c_s2mm_burst_size {16} \
    CONFIG.c_m_axi_mm2s_data_width {32} \
    CONFIG.c_m_axi_s2mm_data_width {32} \
    CONFIG.c_m_axis_mm2s_tdata_width {32} \
    CONFIG.c_s_axis_s2mm_tdata_width {32} \
] [get_bd_cells axi_dma_0]

# --- Lightweight inference/profile HLS SNN IP ---
create_bd_cell -type ip -vlnv xilinx.com:hls:snn_inference_profile_hls:1.0 snn_top_hls_0

# --- Config Regs ---
# Always instantiate from latest RTL source to avoid stale packaged-IP drift.
create_bd_cell -type module -reference snn_config_regs snn_config_regs_0
if {[llength [get_bd_pins -quiet snn_config_regs_0/service_cycles_counter]] == 0} {
    error "snn_config_regs_0/service_cycles_counter pin missing. Check hardware/hdl/rtl/common/snn_config_regs.v"
}
foreach required_pin {
    profile_start
    profile_stop
    profile_expected_count
    profile_index
    profile_data
    profile_info
    router_busy
    any_core_group_busy
    snn_ready
} {
    if {[llength [get_bd_pins -quiet snn_config_regs_0/$required_pin]] == 0} {
        error "snn_config_regs_0/$required_pin pin missing. Check hardware/hdl/rtl/common/snn_config_regs.v"
    }
}

# --- AXI Interconnect for GP0 (3 slaves: HLS, Config, spike DMA) ---
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_interconnect_0
set_property CONFIG.NUM_MI {3} [get_bd_cells axi_interconnect_0]

# --- AXI Interconnect for HP0 (spike DMA MM2S/S2MM) ---
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_interconnect_hp0
set_property -dict [list CONFIG.NUM_SI {2} CONFIG.NUM_MI {1}] [get_bd_cells axi_interconnect_hp0]

# --- Constants ---
create_bd_cell -type ip -vlnv xilinx.com:ip:xlconstant:1.1 const_zero_1bit
set_property CONFIG.CONST_VAL {0} [get_bd_cells const_zero_1bit]

# =============================================================================
# Clock and Reset Connections
# =============================================================================
connect_bd_net [get_bd_pins processing_system7_0/FCLK_CLK0] \
    [get_bd_pins proc_sys_reset_0/slowest_sync_clk] \
    [get_bd_pins axi_dma_0/s_axi_lite_aclk] \
    [get_bd_pins axi_dma_0/m_axi_mm2s_aclk] \
    [get_bd_pins axi_dma_0/m_axi_s2mm_aclk] \
    [get_bd_pins snn_top_hls_0/ap_clk] \
    [get_bd_pins snn_config_regs_0/s_axi_aclk] \
    [get_bd_pins axi_interconnect_0/ACLK] \
    [get_bd_pins axi_interconnect_0/S00_ACLK] \
    [get_bd_pins axi_interconnect_0/M00_ACLK] \
    [get_bd_pins axi_interconnect_0/M01_ACLK] \
    [get_bd_pins axi_interconnect_0/M02_ACLK] \
    [get_bd_pins axi_interconnect_hp0/ACLK] \
    [get_bd_pins axi_interconnect_hp0/S00_ACLK] \
    [get_bd_pins axi_interconnect_hp0/S01_ACLK] \
    [get_bd_pins axi_interconnect_hp0/M00_ACLK] \
    [get_bd_pins processing_system7_0/M_AXI_GP0_ACLK] \
    [get_bd_pins processing_system7_0/S_AXI_HP0_ACLK]

connect_bd_net [get_bd_pins processing_system7_0/FCLK_RESET0_N] \
    [get_bd_pins proc_sys_reset_0/ext_reset_in]

connect_bd_net [get_bd_pins proc_sys_reset_0/peripheral_aresetn] \
    [get_bd_pins snn_top_hls_0/ap_rst_n] \
    [get_bd_pins snn_config_regs_0/s_axi_aresetn] \
    [get_bd_pins axi_dma_0/axi_resetn] \
    [get_bd_pins axi_interconnect_0/ARESETN] \
    [get_bd_pins axi_interconnect_0/S00_ARESETN] \
    [get_bd_pins axi_interconnect_0/M00_ARESETN] \
    [get_bd_pins axi_interconnect_0/M01_ARESETN] \
    [get_bd_pins axi_interconnect_0/M02_ARESETN] \
    [get_bd_pins axi_interconnect_hp0/ARESETN] \
    [get_bd_pins axi_interconnect_hp0/S00_ARESETN] \
    [get_bd_pins axi_interconnect_hp0/S01_ARESETN] \
    [get_bd_pins axi_interconnect_hp0/M00_ARESETN]

# =============================================================================
# AXI GP0 Interconnect (PS → Slaves)
# =============================================================================
connect_bd_intf_net [get_bd_intf_pins processing_system7_0/M_AXI_GP0] \
    [get_bd_intf_pins axi_interconnect_0/S00_AXI]

# M00 → snn_top_hls_0/s_axi_ctrl
connect_bd_intf_net [get_bd_intf_pins axi_interconnect_0/M00_AXI] \
    [get_bd_intf_pins snn_top_hls_0/s_axi_ctrl]

# M01 → snn_config_regs_0/S_AXI
connect_bd_intf_net [get_bd_intf_pins axi_interconnect_0/M01_AXI] \
    [get_bd_intf_pins snn_config_regs_0/s_axi]

# M02 → axi_dma_0/S_AXI_LITE
connect_bd_intf_net [get_bd_intf_pins axi_interconnect_0/M02_AXI] \
    [get_bd_intf_pins axi_dma_0/S_AXI_LITE]

# =============================================================================
# AXI HP0 Interconnect (DMA → DDR)
# =============================================================================
connect_bd_intf_net [get_bd_intf_pins axi_dma_0/M_AXI_MM2S] \
    [get_bd_intf_pins axi_interconnect_hp0/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins axi_dma_0/M_AXI_S2MM] \
    [get_bd_intf_pins axi_interconnect_hp0/S01_AXI]
connect_bd_intf_net [get_bd_intf_pins axi_interconnect_hp0/M00_AXI] \
    [get_bd_intf_pins processing_system7_0/S_AXI_HP0]

# =============================================================================
# DMA ↔ HLS AXI-Stream Connections
# =============================================================================
# MM2S → HLS s_axis_spikes
connect_bd_intf_net [get_bd_intf_pins axi_dma_0/M_AXIS_MM2S] \
    [get_bd_intf_pins snn_top_hls_0/s_axis_spikes]

# HLS m_axis_spikes → S2MM
connect_bd_intf_net [get_bd_intf_pins snn_top_hls_0/m_axis_spikes] \
    [get_bd_intf_pins axi_dma_0/S_AXIS_S2MM]

# =============================================================================
# HLS ↔ RTL External Ports  (connected in snn_core_group_top.v wrapper)
# =============================================================================
# Make these external so the top wrapper can connect them

# Derive neuron-id bus widths from HLS IP pins to avoid stale hardcoded widths.
set spike_in_id_pin  [get_bd_pins snn_top_hls_0/spike_in_neuron_id]
set spike_out_id_pin [get_bd_pins snn_top_hls_0/spike_out_neuron_id]
set spike_in_id_left   [get_property LEFT  $spike_in_id_pin]
set spike_in_id_right  [get_property RIGHT $spike_in_id_pin]
set spike_out_id_left  [get_property LEFT  $spike_out_id_pin]
set spike_out_id_right [get_property RIGHT $spike_out_id_pin]
puts "Using HLS spike_in_neuron_id width  : [expr {$spike_in_id_left - $spike_in_id_right + 1}] bits (${spike_in_id_left}:${spike_in_id_right})"
puts "Using HLS spike_out_neuron_id width : [expr {$spike_out_id_left - $spike_out_id_right + 1}] bits (${spike_out_id_left}:${spike_out_id_right})"

# HLS → RTL: spike output to router
create_bd_port -dir O spike_in_valid -type data
create_bd_port -dir O -from $spike_in_id_left -to $spike_in_id_right spike_in_neuron_id -type data
create_bd_port -dir O -from 7 -to 0 spike_in_weight -type data
create_bd_port -dir I spike_in_ready -type data

connect_bd_net [get_bd_pins snn_top_hls_0/spike_in_valid]       [get_bd_ports spike_in_valid]
connect_bd_net [get_bd_pins snn_top_hls_0/spike_in_neuron_id]   [get_bd_ports spike_in_neuron_id]
connect_bd_net [get_bd_pins snn_top_hls_0/spike_in_weight]      [get_bd_ports spike_in_weight]
connect_bd_net [get_bd_ports spike_in_ready]                     [get_bd_pins snn_top_hls_0/spike_in_ready]

# RTL → HLS: spike from neurons
create_bd_port -dir I spike_out_valid -type data
create_bd_port -dir I -from $spike_out_id_left -to $spike_out_id_right spike_out_neuron_id -type data
create_bd_port -dir I -from 7 -to 0 spike_out_weight -type data
create_bd_port -dir O spike_out_ready -type data

connect_bd_net [get_bd_ports spike_out_valid]                    [get_bd_pins snn_top_hls_0/spike_out_valid]
connect_bd_net [get_bd_ports spike_out_neuron_id]               [get_bd_pins snn_top_hls_0/spike_out_neuron_id]
connect_bd_net [get_bd_ports spike_out_weight]                   [get_bd_pins snn_top_hls_0/spike_out_weight]
connect_bd_net [get_bd_pins snn_top_hls_0/spike_out_ready]      [get_bd_ports spike_out_ready]

# SNN Control
create_bd_port -dir O snn_enable -type data
create_bd_port -dir O snn_reset -type data
create_bd_port -dir I snn_ready -type data
create_bd_port -dir I snn_busy -type data
create_bd_port -dir O -from 15 -to 0 threshold_out -type data
create_bd_port -dir O -from 15 -to 0 leak_rate_out -type data

connect_bd_net [get_bd_pins snn_top_hls_0/snn_enable]           [get_bd_ports snn_enable]
connect_bd_net [get_bd_pins snn_top_hls_0/snn_reset]            [get_bd_ports snn_reset]
connect_bd_net [get_bd_ports snn_ready]                          [get_bd_pins snn_top_hls_0/snn_ready]
connect_bd_net [get_bd_ports snn_busy]                           [get_bd_pins snn_top_hls_0/snn_busy]
connect_bd_net [get_bd_pins snn_top_hls_0/threshold_out]        [get_bd_ports threshold_out]
connect_bd_net [get_bd_pins snn_top_hls_0/leak_rate_out]        [get_bd_ports leak_rate_out]

# Config regs ↔ External ports for router/neuron config
create_bd_port -dir O cfg_router_config_we -type data
create_bd_port -dir O -from 31 -to 0 cfg_router_config_addr -type data
create_bd_port -dir O -from 31 -to 0 cfg_router_config_wdata -type data
create_bd_port -dir I -from 31 -to 0 cfg_router_config_rdata -type data
create_bd_port -dir O cfg_neuron_config_we -type data
create_bd_port -dir O -from 9 -to 0 cfg_neuron_config_addr -type data
create_bd_port -dir O -from 31 -to 0 cfg_neuron_config_wdata -type data
create_bd_port -dir O -from 15 -to 0 cfg_global_threshold -type data
create_bd_port -dir O -from 7 -to 0 cfg_global_leak_rate -type data
create_bd_port -dir O -from 7 -to 0 cfg_global_refrac_period -type data
create_bd_port -dir I -from 31 -to 0 cfg_router_spike_count -type data
create_bd_port -dir I -from 31 -to 0 cfg_neuron_spike_count -type data
create_bd_port -dir I cfg_fifo_overflow -type data
create_bd_port -dir I -from 7 -to 0 cfg_active_neurons -type data
create_bd_port -dir I -from 31 -to 0 cfg_throughput_counter -type data
create_bd_port -dir I -from 31 -to 0 cfg_service_cycles_counter -type data
create_bd_port -dir I cfg_router_busy -type data
create_bd_port -dir I cfg_any_core_group_busy -type data
create_bd_port -dir I cfg_snn_ready -type data
create_bd_port -dir I cfg_profile_active -type data
create_bd_port -dir I cfg_profile_done -type data
create_bd_port -dir O cfg_profile_start -type data
create_bd_port -dir O cfg_profile_stop -type data
create_bd_port -dir O -from 15 -to 0 cfg_profile_expected_count -type data
create_bd_port -dir O -from 7 -to 0 cfg_profile_index -type data
create_bd_port -dir I -from 31 -to 0 cfg_profile_data -type data
create_bd_port -dir I -from 31 -to 0 cfg_profile_info -type data

# Connect config regs to external ports
connect_bd_net [get_bd_pins snn_config_regs_0/router_config_we]     [get_bd_ports cfg_router_config_we]
connect_bd_net [get_bd_pins snn_config_regs_0/router_config_addr]   [get_bd_ports cfg_router_config_addr]
connect_bd_net [get_bd_pins snn_config_regs_0/router_config_wdata]  [get_bd_ports cfg_router_config_wdata]
connect_bd_net [get_bd_ports cfg_router_config_rdata]                [get_bd_pins snn_config_regs_0/router_config_rdata]
connect_bd_net [get_bd_pins snn_config_regs_0/neuron_config_we]     [get_bd_ports cfg_neuron_config_we]
connect_bd_net [get_bd_pins snn_config_regs_0/neuron_config_addr]   [get_bd_ports cfg_neuron_config_addr]
connect_bd_net [get_bd_pins snn_config_regs_0/neuron_config_wdata]  [get_bd_ports cfg_neuron_config_wdata]
connect_bd_net [get_bd_pins snn_config_regs_0/global_threshold]     [get_bd_ports cfg_global_threshold]
connect_bd_net [get_bd_pins snn_config_regs_0/global_leak_rate]     [get_bd_ports cfg_global_leak_rate]
connect_bd_net [get_bd_pins snn_config_regs_0/global_refrac_period] [get_bd_ports cfg_global_refrac_period]
connect_bd_net [get_bd_ports cfg_router_spike_count]                 [get_bd_pins snn_config_regs_0/router_spike_count]
connect_bd_net [get_bd_ports cfg_neuron_spike_count]                 [get_bd_pins snn_config_regs_0/neuron_spike_count]
connect_bd_net [get_bd_ports cfg_fifo_overflow]                      [get_bd_pins snn_config_regs_0/fifo_overflow]
connect_bd_net [get_bd_ports cfg_active_neurons]                     [get_bd_pins snn_config_regs_0/active_neurons]
connect_bd_net [get_bd_ports cfg_throughput_counter]                  [get_bd_pins snn_config_regs_0/throughput_counter]
connect_bd_net [get_bd_ports cfg_service_cycles_counter]              [get_bd_pins snn_config_regs_0/service_cycles_counter]
connect_bd_net [get_bd_ports cfg_router_busy]                         [get_bd_pins snn_config_regs_0/router_busy]
connect_bd_net [get_bd_ports cfg_any_core_group_busy]                 [get_bd_pins snn_config_regs_0/any_core_group_busy]
connect_bd_net [get_bd_ports cfg_snn_ready]                           [get_bd_pins snn_config_regs_0/snn_ready]
connect_bd_net [get_bd_ports cfg_profile_active]                      [get_bd_pins snn_config_regs_0/profile_active]
connect_bd_net [get_bd_ports cfg_profile_done]                        [get_bd_pins snn_config_regs_0/profile_done]
connect_bd_net [get_bd_pins snn_config_regs_0/profile_start]          [get_bd_ports cfg_profile_start]
connect_bd_net [get_bd_pins snn_config_regs_0/profile_stop]           [get_bd_ports cfg_profile_stop]
connect_bd_net [get_bd_pins snn_config_regs_0/profile_expected_count] [get_bd_ports cfg_profile_expected_count]
connect_bd_net [get_bd_pins snn_config_regs_0/profile_index]          [get_bd_ports cfg_profile_index]
connect_bd_net [get_bd_ports cfg_profile_data]                        [get_bd_pins snn_config_regs_0/profile_data]
connect_bd_net [get_bd_ports cfg_profile_info]                        [get_bd_pins snn_config_regs_0/profile_info]

# Clock/reset external ports
# NOTE: Port name is kept as clk_100mhz for backward compatibility with
# existing top-level wrapper wiring regardless of configured FCLK frequency.
create_bd_port -dir O clk_100mhz -type clk
create_bd_port -dir O rst_n_sync -type rst
create_bd_port -dir O debug_learning_active -type data

connect_bd_net [get_bd_pins processing_system7_0/FCLK_CLK0] [get_bd_ports clk_100mhz]
connect_bd_net [get_bd_pins proc_sys_reset_0/peripheral_aresetn] [get_bd_ports rst_n_sync]
connect_bd_net [get_bd_pins const_zero_1bit/dout] [get_bd_ports debug_learning_active]

# HLS threshold/leak connected via snn_core_group_top.v wrapper, not via config_regs
# (config_regs doesn't have hls_snn_enable/hls_threshold/hls_leak_rate ports)

# Interrupt
connect_bd_net [get_bd_pins axi_dma_0/mm2s_introut] [get_bd_pins processing_system7_0/IRQ_F2P]

# =============================================================================
# Address Map
# =============================================================================
assign_bd_address

# Set specific addresses to match original design
# Use catch to handle address segment name variations
foreach seg [get_bd_addr_segs -of_objects [get_bd_addr_spaces processing_system7_0/Data]] {
    set seg_name [get_property NAME $seg]
    puts "Found address segment: $seg_name"
    
    if {[string match "*snn_top_hls*" $seg_name]} {
        set_property offset 0x43C00000 $seg
        # Expose full HLS AXI-Lite register map (up to 0x9C in current IP).
        # Keeping this at 128B hides version/reward registers above 0x7F and
        # makes on-board IP/version parity checks impossible.
        set_property range 4K $seg
        puts "  -> HLS at 0x43C00000 (range 4K)"
    } elseif {[string match "*config_regs*" $seg_name]} {
        set_property offset 0x43C10000 $seg
        set_property range 4K $seg
        puts "  -> Config at 0x43C10000"
    } elseif {[string match "*axi_dma_0*" $seg_name]} {
        set_property offset 0x41E00000 $seg
        set_property range 64K $seg
        puts "  -> Spike DMA at 0x41E00000"
    }
}

# =============================================================================
# Validate and Save Block Design
# =============================================================================
foreach required_port {
    cfg_profile_start
    cfg_profile_stop
    cfg_profile_expected_count
    cfg_profile_index
    cfg_profile_data
    cfg_profile_info
    cfg_router_busy
    cfg_any_core_group_busy
    cfg_snn_ready
} {
    if {[llength [get_bd_ports -quiet $required_port]] == 0} {
        error "Required BD port missing: $required_port"
    }
}
validate_bd_design
save_bd_design

# Generate BD wrapper
make_wrapper -files [get_files ${build_dir}/snn_core_group_profile.srcs/sources_1/bd/design_1/design_1.bd] -top
add_files -norecurse ${build_dir}/snn_core_group_profile.gen/sources_1/bd/design_1/hdl/design_1_wrapper.v

# =============================================================================
# Add RTL Sources (hierarchical core-group architecture)
# =============================================================================
add_files -norecurse ${project_dir}/config/generated/snn_params.vh
add_files -norecurse ${rtl_dir}/core/core_group.v
add_files -norecurse ${rtl_dir}/core/event_router_ng.v
add_files -norecurse ${rtl_dir}/core/synaptic_connectivity_table.v
add_files -norecurse ${rtl_dir}/top/snn_core_group_top.v
set_property include_dirs [list ${project_dir}/config/generated] [get_filesets sources_1]

# Set snn_core_group_top as the real top module
set_property top snn_core_group_top [get_filesets sources_1]
update_compile_order -fileset sources_1

# Add PYNQ-Z2 constraints
if {[file exists ${project_dir}/hardware/constraints/pynq_z2.xdc]} {
    add_files -fileset constrs_1 -norecurse ${project_dir}/hardware/constraints/pynq_z2.xdc
}

# =============================================================================
# Synthesis
# =============================================================================
puts "===== Starting Synthesis ====="
launch_runs synth_1 -jobs 4
wait_on_run synth_1

if {[get_property PROGRESS [get_runs synth_1]] != "100%"} {
    puts "ERROR: Synthesis failed!"
    exit 1
}
puts "===== Synthesis Complete ====="

# Confirm that the synthesized hierarchy is the intended core-group route.
open_run synth_1
set event_router_cells [get_cells -hier -quiet -filter {REF_NAME == event_router_ng}]
set core_group_cells   [get_cells -hier -quiet -filter {REF_NAME == core_group}]
set ct_cells           [get_cells -hier -quiet -filter {REF_NAME == synaptic_connectivity_table}]
set legacy_router_cells [get_cells -hier -quiet -filter {REF_NAME == spike_router}]

if {[llength $event_router_cells] == 0} {
    error "event_router_ng was not synthesized"
}
if {[llength $core_group_cells] == 0} {
    error "core_group instances were not synthesized"
}
if {[llength $ct_cells] == 0} {
    error "synaptic_connectivity_table was not synthesized"
}
if {[llength $legacy_router_cells] != 0} {
    error "Unexpected legacy spike_router found in core-group build"
}
puts "Verified hierarchy: [llength $event_router_cells] event_router_ng, [llength $core_group_cells] core_group, [llength $ct_cells] connectivity table"
report_utilization -file ${output_dir}/snn_core_group_profile_utilization_synth.rpt

# Stop before implementation when the synthesized design cannot fit the target.
set used_ramb36 [llength [get_cells -hier -quiet -filter {REF_NAME == RAMB36E1}]]
set used_ramb18 [llength [get_cells -hier -quiet -filter {REF_NAME == RAMB18E1}]]
set avail_bram_tiles [get_property BLOCK_RAMS [get_parts $part]]
set used_bram_tiles [expr {$used_ramb36 + int(ceil($used_ramb18 / 2.0))}]
puts "BRAM usage: RAMB36=${used_ramb36}, RAMB18=${used_ramb18}, tiles=${used_bram_tiles}/${avail_bram_tiles}"
if {$used_bram_tiles > $avail_bram_tiles} {
    error "Core-group design exceeds target BRAM capacity; reduce group sizes/count or CT fanout before implementation"
}
close_design

# =============================================================================
# Implementation (Place & Route)
# =============================================================================
puts "===== Starting Implementation ====="
launch_runs impl_1 -to_step write_bitstream -jobs 4
wait_on_run impl_1

if {[get_property PROGRESS [get_runs impl_1]] != "100%"} {
    puts "ERROR: Implementation failed!"
    exit 1
}
puts "===== Implementation Complete ====="

# =============================================================================
# Copy Outputs
# =============================================================================
set bit_file [glob -nocomplain ${build_dir}/snn_core_group_profile.runs/impl_1/*.bit]
set hwh_file [glob -nocomplain ${build_dir}/snn_core_group_profile.gen/sources_1/bd/design_1/hw_handoff/*.hwh]

if {$bit_file ne ""} {
    file copy -force $bit_file ${output_dir}/snn_core_group_profile.bit
    puts "Bitstream: ${output_dir}/snn_core_group_profile.bit"
}
if {$hwh_file ne ""} {
    file copy -force $hwh_file ${output_dir}/snn_core_group_profile.hwh
    puts "HWH: ${output_dir}/snn_core_group_profile.hwh"
}

# Reports
if {[catch {open_run impl_1} open_err]} {
    puts "WARNING: Could not open impl_1 for report generation: $open_err"
} else {
    report_utilization -file ${output_dir}/snn_core_group_profile_utilization.rpt
    report_timing_summary -file ${output_dir}/snn_core_group_profile_timing.rpt
    report_power -file ${output_dir}/snn_core_group_profile_power.rpt
}

puts "===== ALL DONE ====="
