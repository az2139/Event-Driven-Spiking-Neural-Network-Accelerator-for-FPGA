#!/bin/bash
#
# Quick Bitstream Build Verification Script
#
# This script performs a quick verification that the build environment
# is correctly configured and all necessary files are present.

set -e

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}✓${NC} $1"
}

log_error() {
    echo -e "${RED}✗${NC} $1"
}

echo "SNN Accelerator Build Environment Verification"
echo "=============================================="
echo ""

# Check source files
echo "Checking source files..."

if [ -f "hardware/hls/src/snn_top_hls.cpp" ]; then
    log_info "HLS kernel source found"
else
    log_error "HLS kernel source NOT found"
    exit 1
fi

if [ -f "hardware/hls/include/snn_top_hls.h" ]; then
    log_info "HLS kernel header found"
else
    log_error "HLS kernel header NOT found"
    exit 1
fi

if [ -f "hardware/hls/test/test_snn_top_hls.cpp" ]; then
    log_info "HLS testbench found"
else
    log_error "HLS testbench NOT found"
    exit 1
fi

echo ""
echo "Checking scripts..."

if [ -f "hardware/scripts/synth_core_group.tcl" ]; then
    log_info "Synthesis script found"
else
    log_error "Synthesis script NOT found"
    exit 1
fi

if [ -f "hardware/scripts/run_testbenches.sh" ]; then
    log_info "Testbench runner script found"
else
    log_error "Testbench runner script NOT found"
    exit 1
fi

if [ -f "hardware/scripts/pynq_z2_ps_config.tcl" ]; then
    log_info "PYNQ-Z2 PS config found"
else
    log_error "PYNQ-Z2 PS config NOT found"
    exit 1
fi

echo ""
echo "Checking constraints..."

if [ -f "hardware/constraints/pynq_z2_pins.xdc" ]; then
    log_info "Pin constraints found"
else
    log_error "Pin constraints NOT found"
    exit 1
fi

if [ -f "hardware/constraints/timing.xdc" ]; then
    log_info "Timing constraints found"
else
    log_error "Timing constraints NOT found"
    exit 1
fi

echo ""
echo "Checking tools..."

# Check v++ (replaces deprecated vitis_hls)
if command -v v++ &> /dev/null; then
    VPP_VER=$(v++ --version 2>&1 | head -n 1)
    log_info "v++ compiler found: ${VPP_VER}"
else
    log_error "v++ compiler NOT found"
    echo "  Run: source /xilinx/2025.2/Vitis/settings64.sh"
    exit 1
fi

# Check Vivado
if command -v vivado &> /dev/null; then
    VIVADO_VER=$(vivado -version | head -n 1)
    log_info "Vivado found: ${VIVADO_VER}"
else
    log_error "Vivado NOT found"
    echo "  Run: source /xilinx/2025.2/Vivado/settings64.sh"
    exit 1
fi

echo ""
echo "=============================================="
log_info "Build environment ready!"
echo ""
echo "To run RTL testbenches:"
echo "  cd hardware/scripts"
echo "  ./run_testbenches.sh"
echo ""
echo "To run synthesis:"
echo "  cd hardware/scripts"
echo "  vivado -mode batch -source synth_core_group.tcl"
echo ""
echo "Estimated build time: 20-40 minutes"
