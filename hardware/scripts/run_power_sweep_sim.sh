#!/bin/bash
# hardware/scripts/run_power_sweep_sim.sh
# Compiles and runs the power sweep testbench to generate VCD for XPA

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RTL_DIR="$PROJ_ROOT/hardware/hdl/rtl"
SIM_DIR="$PROJ_ROOT/hardware/sim"
WORK_DIR="$PROJ_ROOT/hardware/sim_work_power"

source /xilinx/2025.2/Vivado/settings64.sh

mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

# Sources
SRC_FILES=(
    "$RTL_DIR/router/spike_router.v"
    "$RTL_DIR/neurons/lif_neuron_array.v"
    "$RTL_DIR/common/fifo.v"
    "$SIM_DIR/tb_snn_power_sweep.v"
)

echo "--- Compiling Power Sweep Testbench ---"
xvlog "${SRC_FILES[@]}"

echo "--- Elaborating ---"
xelab -debug typical tb_snn_power_sweep -s power_sim

echo "--- Running Simulation (SAIF generation) ---"
xsim power_sim -tclbatch "$SCRIPT_DIR/gen_saif.tcl"

echo "--- SAIF generation complete: $WORK_DIR/power_sweep.saif ---"
