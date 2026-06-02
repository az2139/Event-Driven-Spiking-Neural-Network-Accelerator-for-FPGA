#!/bin/bash
# hardware/scripts/impl_and_xpa.sh
# 1. Hardware Implementation (Synthesis & Route)
# 2. XPA Power Analysis

set -e

source /xilinx/2025.2/Vivado/settings64.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
REBUILD_TCL="$PROJ_ROOT/hardware/scripts/rebuild_integrated.tcl"
XPA_SH="$PROJ_ROOT/hardware/scripts/run_xpa.sh"

echo "================================================================"
echo "STep 1: Running Hardware Implementation (v2)..."
echo "================================================================"
vivado -mode batch -source "$REBUILD_TCL"

echo "================================================================"
echo "Step 2: Running XPA Power Analysis..."
echo "================================================================"
bash "$XPA_SH"

echo "================================================================"
echo "All Tasks Complete."
echo "Check outputs/integrated_power_final.rpt for results."
echo "================================================================"
