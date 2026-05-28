#!/bin/bash
#=============================================================================
# Run all SNN Core Group testbenches using Vivado xsim
# Usage: bash run_testbenches.sh
#=============================================================================
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RTL_DIR="$PROJ_ROOT/hardware/hdl/rtl"
TB_DIR="$PROJ_ROOT/hardware/hdl/tb"
WORK_DIR="$PROJ_ROOT/hardware/sim_work"
INC_DIR="$PROJ_ROOT/config/generated"

source /xilinx/2025.2/Vivado/settings64.sh

mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

PASS_TOTAL=0
FAIL_TOTAL=0

run_test() {
    local TESTNAME=$1
    local TB_FILE=$2
    shift 2
    local SRC_FILES=("$@")

    echo ""
    echo "============================================================"
    echo "  Running: $TESTNAME"
    echo "============================================================"

    # Clean
    rm -rf xsim.dir *.log *.jou *.pb *.wdb webtalk* .Xil 2>/dev/null || true

    # Compile
    echo "  [1/3] Compiling..."
    if ! xvlog -nolog -i "$INC_DIR" "${SRC_FILES[@]}" "$TB_FILE" > compile_${TESTNAME}.log 2>&1; then
        echo "  [COMPILE ERROR] $TESTNAME"
        cat compile_${TESTNAME}.log
        FAIL_TOTAL=$((FAIL_TOTAL + 1))
        return
    fi

    # Elaborate
    echo "  [2/3] Elaborating..."
    local TOP=$(basename "$TB_FILE" .v)
    if ! xelab -nolog -debug off "$TOP" -s "${TESTNAME}_sim" > elab_${TESTNAME}.log 2>&1; then
        echo "  [ELABORATE ERROR] $TESTNAME"
        cat elab_${TESTNAME}.log
        FAIL_TOTAL=$((FAIL_TOTAL + 1))
        return
    fi

    # Simulate
    echo "  [3/3] Simulating..."
    if ! xsim -nolog "${TESTNAME}_sim" -runall > sim_${TESTNAME}.log 2>&1; then
        echo "  [SIM ERROR] $TESTNAME"
        cat sim_${TESTNAME}.log
        FAIL_TOTAL=$((FAIL_TOTAL + 1))
        return
    fi

    # Print results
    echo ""
    grep -aE "\[PASS\]|\[FAIL\]|Results:|ALL TESTS|SOME TESTS|ERROR" sim_${TESTNAME}.log || true

    # Count
    local P F
    P=$(grep -ac "\[PASS\]" sim_${TESTNAME}.log 2>/dev/null || true)
    F=$(grep -ac "\[FAIL\]" sim_${TESTNAME}.log 2>/dev/null || true)
    P=${P:-0}
    F=${F:-0}
    PASS_TOTAL=$((PASS_TOTAL + P))
    FAIL_TOTAL=$((FAIL_TOTAL + F))
}

#-----------------------------------------------------------------------------
# Test 1: Core Group
#-----------------------------------------------------------------------------
run_test "core_group" \
    "$TB_DIR/tb_core_group.v" \
    "$RTL_DIR/core/core_group.v"

#-----------------------------------------------------------------------------
# Test 2: Router + Connectivity Table
#-----------------------------------------------------------------------------
run_test "router_ct" \
    "$TB_DIR/tb_router_ct.v" \
    "$RTL_DIR/core/event_router_ng.v" \
    "$RTL_DIR/core/synaptic_connectivity_table.v"

#-----------------------------------------------------------------------------
# Test 3: Full Integration
#-----------------------------------------------------------------------------
run_test "integration" \
    "$TB_DIR/tb_integration.v" \
    "$RTL_DIR/core/core_group.v" \
    "$RTL_DIR/core/event_router_ng.v" \
    "$RTL_DIR/core/synaptic_connectivity_table.v"

#-----------------------------------------------------------------------------
# Summary
#-----------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  OVERALL RESULTS: $PASS_TOTAL PASS, $FAIL_TOTAL FAIL"
echo "============================================================"

if [ $FAIL_TOTAL -gt 0 ]; then
    echo "*** SOME TESTS FAILED ***"
    exit 1
else
    echo "*** ALL TESTS PASSED ***"
    exit 0
fi
