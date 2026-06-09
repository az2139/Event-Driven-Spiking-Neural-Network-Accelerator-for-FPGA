# Report hierarchical utilization from an existing core-group synthesis run.

set script_dir [file dirname [file normalize [info script]]]
set project_dir [file normalize [file join $script_dir ../..]]
set dcp_file "${project_dir}/hardware/build/snn_core_group_profile/snn_core_group_profile.runs/synth_1/snn_core_group_top.dcp"
set output_file "${project_dir}/outputs/snn_core_group_profile_utilization_hierarchical_synth.rpt"

if {![file exists $dcp_file]} {
    error "Synthesis checkpoint not found: $dcp_file"
}

open_checkpoint $dcp_file
report_utilization -hierarchical -hierarchical_depth 3 -file $output_file
puts "Hierarchical utilization report: $output_file"
