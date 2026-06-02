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
│   ├── src/            # HLS source (snn_top_hls)
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

```bash
cd hardware/hls
./scripts/build_hls.sh --clean
```

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

For a full bitstream, run `hardware/scripts/rebuild_integrated.tcl`, which performs synthesis, implementation, and `write_bitstream`.

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

Current HLS design targets 720 neurons with aggressive pipelining:

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
