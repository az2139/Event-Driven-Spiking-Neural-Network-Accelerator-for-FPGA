#!/bin/bash
# hardware/scripts/run_xpa_final.sh
# Runs XPA using the routed DCP and the generated VCD

set -e

source /xilinx/2025.2/Vivado/settings64.sh

PROJ_ROOT="/mnt/workspace/Event-Driven-Spiking-Neural-Network-Accelerator-for-FPGA"
DCP="$PROJ_ROOT/hardware/build/snn_integrated_v2/snn_integrated_v2.runs/impl_1/snn_integrated_top_routed.dcp"
VCD="$PROJ_ROOT/hardware/sim_work_power/power_sweep.saif"
XPA_TCL="$PROJ_ROOT/hardware/scripts/run_xpa.tcl"

if [ ! -f "$DCP" ]; then
    echo "ERROR: Routed DCP not found at $DCP"
    exit 1
fi

if [ ! -f "$VCD" ]; then
    echo "ERROR: VCD not found at $VCD"
    exit 1
fi

echo "--- Starting XPA Power Analysis ---"
# We need to specify the -strip_path in the TCL or here.
# In the sim, the top was 'tb_snn_power_sweep'. The DUTs were 'router_inst' and 'neuron_array_inst'.
# In the integrated design, 'spike_router' is usually at 'spike_router_inst' or similar.
# However, for a quick check, we can just read the VCD and let Vivado match by name.

OUT_DIR="$PROJ_ROOT/outputs"
mkdir -p "$OUT_DIR"

vivado -mode batch -source "$XPA_TCL" -tclargs "$DCP" "$VCD" "$OUT_DIR"

echo "--- XPA Analysis Complete. Check $OUT_DIR/integrated_power_final.rpt ---"
