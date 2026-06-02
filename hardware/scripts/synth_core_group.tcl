# Vivado Synthesis Script for Core Group Architecture
# Checks resource utilization without full implementation

# Set part
set part xc7z020clg400-2

set script_dir [file dirname [file normalize [info script]]]
set project_root [file normalize [file join $script_dir ../..]]
set rtl_dir "$project_root/hardware/hdl/rtl"
set generated_dir "$project_root/config/generated"
set output_dir "$project_root/outputs"

# Create in-memory project
create_project -in_memory -part $part
set_property include_dirs [list $generated_dir] [get_filesets sources_1]
add_files -norecurse "$generated_dir/snn_params.vh"

# Add RTL sources
# Core group modules
read_verilog "$rtl_dir/core/core_group.v"
read_verilog "$rtl_dir/core/synaptic_connectivity_table.v"
read_verilog "$rtl_dir/core/event_router_ng.v"

# Synthesize core_group standalone first to check per-group resource usage
puts "===== Synthesizing core_group (single instance) ====="
synth_design -top core_group -part $part -mode out_of_context
report_utilization -file "$output_dir/core_group_utilization.rpt"
puts "===== core_group synthesis complete ====="

# Close and reopen for connectivity table
close_design
create_project -in_memory -part $part
set_property include_dirs [list $generated_dir] [get_filesets sources_1]
add_files -norecurse "$generated_dir/snn_params.vh"
read_verilog "$rtl_dir/core/synaptic_connectivity_table.v"

puts "===== Synthesizing synaptic_connectivity_table ====="
synth_design -top synaptic_connectivity_table -part $part -mode out_of_context
report_utilization -file "$output_dir/connectivity_table_utilization.rpt"
puts "===== connectivity_table synthesis complete ====="

# Close and reopen for event router (16 groups)
close_design
create_project -in_memory -part $part
set_property include_dirs [list $generated_dir] [get_filesets sources_1]
add_files -norecurse "$generated_dir/snn_params.vh"
read_verilog "$rtl_dir/core/event_router_ng.v"

puts "===== Synthesizing event_router_ng (NUM_GROUPS=16) ====="
synth_design -top event_router_ng -part $part -mode out_of_context \
    -generic {NUM_GROUPS=16}
report_utilization -file "$output_dir/event_router_ng_utilization.rpt"
puts "===== event_router_ng synthesis complete ====="

# Close and reopen for connectivity table (16 groups)
close_design
create_project -in_memory -part $part
set_property include_dirs [list $generated_dir] [get_filesets sources_1]
add_files -norecurse "$generated_dir/snn_params.vh"
read_verilog "$rtl_dir/core/synaptic_connectivity_table.v"

puts "===== Synthesizing synaptic_connectivity_table (NUM_GROUPS=16) ====="
synth_design -top synaptic_connectivity_table -part $part -mode out_of_context \
    -generic {NUM_GROUPS=16}
report_utilization -file "$output_dir/connectivity_table_16g_utilization.rpt"
puts "===== connectivity_table (16g) synthesis complete ====="

puts "===== ALL SYNTHESIS COMPLETE ====="
