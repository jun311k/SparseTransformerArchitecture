# BF16 Multiplier Simulation Environment

This directory contains the simulation environment for the BF16 multiplier implementation using Icarus Verilog and GTKWave.

## Directory Structure
```
mac_array_sim/
├── bf16_multiplier_tb.v    # Testbench for BF16 multiplier
├── run_sim.sh             # Simulation run script
└── README.md             # This file
```

## Simulation Setup

### Prerequisites
- Icarus Verilog (iverilog)
- GTKWave
- Bash shell

### Installation

#### macOS (using Homebrew):
```bash
brew install icarus-verilog
brew install gtkwave
```

#### Ubuntu/Debian:
```bash
sudo apt-get install iverilog gtkwave
```

#### Windows:
1. Install Icarus Verilog from: http://bleyer.org/icarus/
2. Install GTKWave from: http://gtkwave.sourceforge.net/

### Running Simulation

1. Make the run script executable:
```bash
chmod +x run_sim.sh
```

2. Run the simulation:
```bash
./run_sim.sh
```

This will:
- Compile the design and testbench
- Run the simulation
- Generate a VCD file
- Open GTKWave to view the waveforms

### Manual Steps

If you prefer to run the simulation manually:

1. Compile the design:
```bash
iverilog -o sim.out ../mac_array/bf16_multiplier.v bf16_multiplier_tb.v
```

2. Run the simulation:
```bash
vvp sim.out
```

3. View waveforms:
```bash
gtkwave dump.vcd
```

## Test Cases

The testbench includes the following test cases:
1. Normal numbers (1.0 * 2.0 = 2.0)
2. Negative numbers (-1.0 * 2.0 = -2.0)
3. Small numbers (0.5 * 0.5 = 0.25)
4. Zero multiplication (0.0 * 1.0 = 0.0)
5. Infinity handling (inf * 1.0 = inf)
6. NaN handling (NaN * 1.0 = NaN)

## Waveform Viewing in GTKWave

1. Open the VCD file in GTKWave
2. In the left panel, expand the testbench module
3. Select signals to view
4. Use the toolbar to:
   - Zoom in/out
   - Pan left/right
   - Search for specific times
   - Save/load signal configurations

## Debugging Tips

1. Check the terminal output for simulation messages
2. Use GTKWave to inspect signal values
3. Use the search function in GTKWave to find specific events
4. Save your signal configuration in GTKWave for future use
5. Check the test summary at the end of simulation 