#!/bin/bash
# Build the lightweight inference/profile HLS IP without learning or weights.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
HLS_DIR="$(dirname "$SCRIPT_DIR")"

cd "$HLS_DIR"

if [[ "${1:-}" == "--clean" ]]; then
    rm -rf hls_inference_profile_output
fi

v++ -c --mode hls \
    --part xc7z020clg400-2 \
    --config scripts/snn_inference_profile_hls_vpp.cfg \
    --work_dir hls_inference_profile_output

echo "IP package: $HLS_DIR/hls_inference_profile_output/hls/impl/ip"
