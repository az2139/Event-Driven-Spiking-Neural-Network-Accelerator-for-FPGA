# Developer Guide

This is the main public developer-facing guide for build, deployment, and the
maintained native execution paths.

## Documentation Map

- `README.md`: project overview and quick start
- `docs/api_reference.md`: Python/runtime API details
- `docs/architecture.md`: hardware/software architecture overview
- `docs/user_guide.md`: end-user setup and examples

## Setup

```bash
# Clone and install
git clone https://github.com/metr0jw/Event-Driven-Spiking-Neural-Network-Accelerator-for-FPGA.git
cd Event-Driven-Spiking-Neural-Network-Accelerator-for-FPGA

# Python dev install
python3 -m venv venv
source venv/bin/activate
cd software/python
pip install -e .
pip install pytest pytest-cov black flake8 mypy

# Vivado tools
source /xilinx/2025.2/Vivado/settings64.sh
export LC_ALL=en_US.UTF-8
```

## Project Structure

```
hardware/
├── hdl/rtl/            # Verilog RTL
│   ├── core/           # Core group, event router, connectivity table
│   └── top/            # Top-level integration (snn_core_group_top)
├── hdl/tb/             # Testbenches (3 active)
├── hls/                # Vitis HLS
│   ├── src/            # Learning HLS and lightweight inference/profile HLS
│   ├── include/        # Headers
│   ├── test/           # HLS testbenches
│   └── scripts/        # HLS build scripts
├── constraints/        # Timing and pin constraints
└── scripts/            # Build & simulation scripts

software/python/        # Python package
examples/               # Usage examples
docs/                   # Documentation
```

## Building

### RTL Simulation

```bash
cd hardware/scripts
./run_testbenches.sh  # Run all 3 core group testbenches (55 checks)
```

### HLS Build

Build the original learning-capable HLS IP:

```bash
cd hardware/hls
./scripts/build_hls.sh --clean
```

Build the lightweight HLS IP used by `snn_core_group_top`:

```bash
cd hardware/hls
./scripts/build_inference_profile_hls.sh --clean
```

`snn_inference_profile_hls` keeps the spike stream, control/status registers,
and RTL spike handshake. It intentionally omits HLS weight memory, STDP,
R-STDP, traces, eligibility, encoder state, weight streams, and checkpoint
support. The RTL core groups and connectivity table are the only synaptic
storage in this build.

### Vivado Synthesis Check

```bash
cd hardware/scripts
source /xilinx/2025.2/Vivado/settings64.sh
export LC_ALL=en_US.UTF-8
vivado -mode batch -source synth_core_group.tcl
```

Output: utilization reports in `outputs/` such as:

- `core_group_utilization.rpt`
- `connectivity_table_utilization.rpt`
- `event_router_ng_utilization.rpt`
- `connectivity_table_16g_utilization.rpt`

For the legacy `snn_integrated_top` bitstream, run:

```bash
vivado -mode batch -source rebuild_integrated.tcl
```

For the hierarchical `snn_core_group_top` bitstream with communication profile
counters, build the lightweight HLS IP first and then run:

```bash
cd hardware/hls
./scripts/build_inference_profile_hls.sh --clean
cd ../scripts
vivado -mode batch -source rebuild_core_group_integrated.tcl
```

The core-group build writes `outputs/snn_core_group_profile.bit` and
`outputs/snn_core_group_profile.hwh`. It also verifies that `event_router_ng`,
`core_group`, and `synaptic_connectivity_table` are present and that the legacy
`spike_router` is absent.

### Convert a pruned BP model for the core-group RTL

After alternating BP pruning and mapping, convert the saved model and a matching
mapping snapshot into the hardware deployment package:

```bash
python3 tests/prepare_bp_coregroup_deployment.py \
  --model data/cache/bp_prune_model_1000h_10c.npz \
  --mapping data/cache/bp_coregroup_mapping_best.npz \
  --output data/cache/bp_coregroup_deployment.npz
```

With alternating mapping/pruning enabled, training writes the restored best
deployable checkpoint and its exact matching ``*_best.npz`` mapping together.
It also prints a full-test-set INT8 software reference after training.

The output contains the logical-to-hardware neuron IDs, quantized signed-edge
magnitudes, input/hidden/output ID ranges, and ordered `cfg_addr`/`cfg_wdata`
writes for the local sparse fanout tables and inter-group connectivity table.
The adjacent JSON file records dimensions, fanout use, and quantization values.

The current RTL has one global threshold and an 8-bit unsigned weight magnitude
plus an excitatory/inhibitory bit. The converter therefore rejects unequal
hidden/output thresholds and fanout overflow. New pruning runs default to zero
software leak, one input presentation, and deterministic `pixel > 0.3` encoding
to match the current deployment configuration. A nonzero leak saved by an older
model is still reported because the trainer's subtractive leak is not bit-exact
with the RTL's shift-based leak.

At the end of each profiled image, the board waits for DMA/HLS input completion,
an idle router, and idle core groups before issuing `PROFILE_STOP`. This command
also starts a parallel 128-entry state-memory clear in every core group. The RTL
delays `profile_done` until all membrane-potential and refractory entries are
zero, so the next image cannot inherit neuron state from the preceding image.
Weights, local fanout tables, the inter-group connectivity table, and cumulative
hardware counters are preserved.

The standard core-group entry point auto-detects the mapped two-layer BP package:

The event-aware BP route requires a freshly trained model, deployment format v2,
and a rebuilt core-group bitstream exposing profile version 13. The RTL configures
the ten mapped output IDs as non-spiking signed score accumulators, snapshots all
ten scores at sample stop, and exposes them through the profile window. The host
computes argmax from those ten signed integer scores. Older model/deployment/
bitstream combinations are rejected.

```bash
python3 tests/fpga_10class_coregroup_inference.py \
  --data /home/xilinx/snn \
  --weights /home/xilinx/snn/bp_coregroup_deployment.npz \
  --dataset /home/xilinx/snn/mnist_10class_deployment_100n.npz \
  --n 100
```

This path programs the pre-encoded local-fanout and CT writes, injects events
through the deployment's 784 mapped input IDs, and classifies only spikes from
its ten mapped output IDs. It reports two software references: an INT8
layer-synchronous `T=1` forward pass and an event-serial hardware-semantics
model. Use `--allow-repeat-fire-reference` only with an RTL build that does not
enforce one spike per neuron per image. Models trained with the older stochastic
25-step input path must be retrained and reconverted before this comparison.

## Supported Workflow Policy

- Native library-first path is the maintained route.
- Removed from supported path: `SpikingJelly auto-conversion`.
- Recommended scenarios:
  1. GPU train (surrogate/STDP) -> native export -> FPGA inference
  2. FPGA STDP train + inference with parity tooling

## Maintained Native Workflows

### Scenario 1: GPU Train -> FPGA Inference

```bash
./scripts/run_scenario1_native_fpga_infer.sh \
  --deployment /home/xilinx/snn/mnist_10class_deployment.npz \
  --output /home/xilinx/snn/mnist_10class_results_scenario1.json
```

### Scenario 2: FPGA STDP Train + FPGA Inference

```bash
./scripts/run_scenario2_fpga_stdp_train_infer.sh \
  --stdp-steps 100 \
  --infer-output /home/xilinx/snn/mnist_10class_results_scenario2.json
```

## RTL Development

### Module Template

```verilog
module my_module #(
    parameter DATA_WIDTH = 16
) (
    input wire clk,
    input wire rst_n,
    input wire [DATA_WIDTH-1:0] data_in,
    output reg [DATA_WIDTH-1:0] data_out
);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            data_out <= 0;
        end else begin
            data_out <= data_in;
        end
    end

endmodule
```

### Coding Style

- **Naming**: `snake_case` for modules/signals, `UPPER_SNAKE_CASE` for parameters
- **Reset**: Active-low async reset (`rst_n`)
- **Assignments**: Non-blocking (`<=`) in sequential, blocking (`=`) in combinational
- **Clock**: Single domain unless noted

### Testbench

```verilog
`timescale 1ns/1ps

module tb_my_module;
    reg clk, rst_n;
    reg [15:0] data_in;
    wire [15:0] data_out;

    my_module #(.DATA_WIDTH(16)) dut (
        .clk(clk), .rst_n(rst_n),
        .data_in(data_in), .data_out(data_out)
    );

    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end

    initial begin
        $dumpfile("work/my_module.vcd");
        $dumpvars(0, tb_my_module);
        
        rst_n = 0; data_in = 0;
        #20 rst_n = 1;
        #10 data_in = 16'hABCD;
        #10;
        
        if (data_out == 16'hABCD) $display("PASS");
        else $display("FAIL");
        
        $finish;
    end
endmodule
```

Run: `iverilog -o work/test tb_my_module.v my_module.v && vvp work/test`

## HLS Development

### Function Template

```cpp
#include "ap_int.h"
#include "hls_stream.h"

void my_function(
    hls::stream<ap_uint<32>>& input,
    hls::stream<ap_uint<32>>& output,
    ap_uint<8> config
) {
    #pragma HLS INTERFACE axis port=input
    #pragma HLS INTERFACE axis port=output
    #pragma HLS INTERFACE s_axilite port=config
    
    for (int i = 0; i < 100; i++) {
        #pragma HLS PIPELINE II=1
        ap_uint<32> data = input.read();
        output.write(data + config);
    }
}
```

### Build

```bash
cd hardware/hls
v++ -c --mode hls \
    --part xc7z020clg400-1 \
    --kernel my_function \
    --hls.clock 10 \
    --config config.ini \
    src/my_function.cpp
```

### Optimization

The learning-capable `snn_top_hls` contains the weight memory and learning
loops. The hierarchical inference/profile route uses
`snn_inference_profile_hls`, which contains no synaptic memory or learning
state.

Learning HLS optimization features include:

- **Pipeline**: `#pragma HLS PIPELINE II=1` — all major loops (LTD, LTP, WEIGHT_SUM) run at II=1
- **Loop unroll**: `#pragma HLS UNROLL factor=4` — used on LTD_LOOP, RSTDP_INNER, DECAY loops
- **Array partition**: Weight memory uses 8 banks (cyclic factor=2 on dim=1, factor=4 on dim=2). Trace arrays use cyclic factor=4.
- **Dataflow**: `#pragma HLS DATAFLOW` for parallelism

Avoid DSP usage: Use shifts instead of multiplies when possible.

**Key constants** (in `snn_top_hls.h`):
- `MAX_NEURONS = 720`, `MAX_SYNAPSES = 518400`
- `NEURON_ID_WIDTH = 10` (10-bit neuron IDs via `neuron_id_t`)
- `WEIGHT_WIDTH = 4`, `MAX_INPUT_CHANNELS = 784`

## Python Development

### Package Structure

```
software/python/snn_fpga_accelerator/
├── __init__.py
├── accelerator.py          # Main API
├── cli.py                  # Command-line interface
├── deploy.py               # Deployment utilities
├── encoder.py              # Delta-sigma encoder
├── exceptions.py           # Custom exceptions
├── fpga_controller.py      # FPGA control interface
├── hw_accurate_simulator.py  # Bit-accurate sim (LIF, STDP)
├── layer.py                # SNN layer abstraction
├── learning.py             # STDP/R-STDP
├── neuron.py               # HW-accurate core group sim
├── pytorch_interface.py    # PyTorch integration
├── pytorch_snn_layers.py   # Custom PyTorch layers
├── rtl_simulator.py        # RTL simulation driver
├── spike_encoding.py       # Spike encoders (Poisson, Temporal, Phase)
├── spyketorch_compat.py    # SpykeTorch compatibility
├── training.py             # Training loop utilities
├── utils.py                # Utilities (tau conversion, visualization)
└── xrt_backend.py          # XRT/Vitis backend
```

### Testing

```bash
cd software/python
pytest tests/
pytest --cov=snn_fpga_accelerator
```

### Code Style

```bash
black .
flake8 .
mypy .
```

## Adding a New Feature

Example: Add a new spike encoder

1. **Define interface** (spike_encoding.py):

```python
class MyEncoder:
    def __init__(self, num_neurons, duration, my_param):
        self.num_neurons = num_neurons
        self.duration = duration
        self.my_param = my_param
    
    def encode(self, input_data):
        # Convert input to spike times
        spike_times = []
        for i, val in enumerate(input_data):
            if val > 0.5:
                spike_times.append((i, val * self.duration))
        return spike_times
```

2. **Add tests** (tests/test_encoders.py):

```python
def test_my_encoder():
    encoder = MyEncoder(10, 0.1, 1.0)
    data = np.random.rand(10)
    spikes = encoder.encode(data)
    assert len(spikes) > 0
```

3. **Document** (docs/api_reference.md)

4. **Add example** (examples/)

## Debugging

### RTL

```bash
# Simulate with waveforms
cd hardware/hdl/sim
iverilog -o work/test tb_module.v module.v
vvp work/test
gtkwave work/waves.vcd
```

### HLS

Check synthesis report: `hls_output/hls/syn/report/csynth.rpt`

### Python

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Performance Profiling

### RTL Timing

Check Vivado timing report: `outputs/integrated_timing.rpt`

Key metrics:
- WNS (Worst Negative Slack): Must be ≥ 0
- TNS (Total Negative Slack): Should be 0

### Python

```python
import time
start = time.time()
output = accel.infer(spikes)
print(f"Inference time: {time.time() - start:.3f}s")
```

## Troubleshooting

**Vivado synthesis fails**: Check for syntax errors with `iverilog -t null -Wall file.v`

**HLS build fails**: Check C++ syntax, add `#include`s

**Python import error**: Run `pip install -e .` in dev mode

**Timing violations**: Reduce clock frequency or add pipeline stages

**Resource overflow**: Reduce network size or optimize modules

## Contributing

See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

## References

- [User Guide](user_guide.md) - Usage and examples
- [API Reference](api_reference.md) - Python API
- [Architecture](architecture.md) - System design
