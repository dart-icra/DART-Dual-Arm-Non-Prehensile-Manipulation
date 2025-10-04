# Dual Arm Non-Prehensile Task (DualArmNPT)

A comprehensive dual-arm robotics control system implementing Model Predictive Control (MPC) for non-prehensile manipulation tasks using MuJoCo physics simulation.

## Overview

This repository provides three different implementations for controlling dual xArm7 robots to perform non-prehensile manipulation tasks (e.g., moving objects on a tray by tilting):

1. **`main.py`** - Simple, direct control implementation
2. **`main_parallel.py`** - Parallel MPC computation with multiprocessing
3. **`main_parallel_enhanced.py`** - Full-featured version with logging, video recording, and comprehensive data collection

## Features

- **Dual-arm coordination** using impedance control and MPC
- **Object-specific MPC parameter tuning** (cube, cylinder, sphere)
- **Real-time physics simulation** with MuJoCo
- **Multiprocessing support** for parallel MPC computation
- **Comprehensive data logging** and analysis
- **Video recording** of experiments

## System Architecture

### Controllers Used

1. **DACTL (Dual Arm Control)**: Impedance-based dual-arm controller for coordinated motion
2. **PMPC (Predictive Model Predictive Control)**: 3D MPC controller for non-prehensile manipulation

### MPC Formulation

The MPC controller optimizes tray tilting angles to move objects to target positions:
- **State**: `[px, vx, py, vy, pz, vz]` (position and velocity in 3D)
- **Control**: `[θx, θy]` (tilt angles about x and y axes)
- **Dynamics**: Includes gravity, friction, and constraint forces

## Installation

### Prerequisites

```bash
# Install required packages
pip install mujoco numpy scipy casadi matplotlib opencv-python icecream
```

### Repository Structure

```
DualArmNPT/
├── main.py                      # Simple control implementation
├── main_parallel.py             # Parallel MPC implementation
├── main_parallel_enhanced.py    # Full-featured implementation
├── src/
│   ├── __init__.py
│   ├── dualctl.py              # Dual arm controller
│   ├── logger.py               # Async data logging
│   └── controller/
│       ├── __init__.py
│       ├── arm.py              # Arm control utilities
│       └── mpc_3d.py           # 3D MPC implementation
├── models/
│   └── xarm7/                  # MuJoCo model files
└── video_results/              # Experiment output directory
```

## Usage

### 1. Simple Control (`main.py`)

Basic implementation without multiprocessing - ideal for development and testing.

```bash
# Basic usage
python main.py

# Custom target and object
python main.py --target 0.1 0.0 0.05 0.0 0.0 0.0 --object_name cube --friction 0.3

# Different world file
python main.py --world_path ./models/xarm7/world_cube_m=1.0_mu=0.1.xml
```

#### Arguments:
- `--target`: Target state `[x, vx, y, vy, z, vz]` (default: all zeros)
- `--friction`: Friction coefficient (default: 0.4)
- `--object_name`: Object to manipulate - "cube", "cylinder", "sphere" (default: "cube")
- `--world_path`: Path to MuJoCo XML file (default: "./models/xarm7/world_cube_m=0.5_mu=0.05.xml")

### 2. Parallel Control (`main_parallel.py`)

Optimized version with parallel MPC computation for better real-time performance.

```bash
# Basic parallel execution
python main_parallel.py

# Extended runtime with cylinder object
python main_parallel.py --object_name cylinder --runtime 60.0 --mass 2.0

# High precision with custom tolerance
python main_parallel.py --tolerance 0.005 --friction 0.2

# Disable parameter tuning (use general parameters)
python main_parallel.py --no_tune
```

#### Additional Arguments:
- `--mass`: Object mass in kg (default: 0.5)
- `--runtime`: Maximum simulation time in seconds (default: 30.0)
- `--tolerance`: Position tolerance for target achievement (default: 0.01)
- `--no_tune`: Use general MPC parameters instead of object-specific tuning

### 3. Enhanced Control (`main_parallel_enhanced.py`)

Full-featured implementation with comprehensive logging and video recording.

```bash
# Full experiment with data collection
python main_parallel_enhanced.py --runtime 30.0 --object_name cube

# Headless execution (no GUI)
python main_parallel_enhanced.py --no_display --runtime 45.0

# Custom experiment configuration
python main_parallel_enhanced.py \
    --target 0.15 0.0 -0.1 0.0 0.0 0.0 \
    --object_name sphere \
    --mass 1.5 \
    --friction 0.3 \
    --runtime 40.0 \
    --tolerance 0.008
```

#### Additional Arguments:
- `--no_display`: Run without GUI visualization (headless mode)

## Experiment Data Collection

### Data Format (Enhanced Version)

The enhanced version automatically saves comprehensive experiment data:

```python
experiment_data = {
    # Time series data (N×1)
    "t": [0.0, 0.002, 0.004, ...],                    # Time stamps
    "loss": [1.25, 1.18, 1.05, ...],                  # MPC cost values
    "solve_time": [0.012, 0.011, 0.013, ...],         # MPC solve times
    
    # Object state and control (N×6, N×2)
    "X": [[px, vx, py, vy, pz, vz], ...],             # Object states
    "X_target": [[px_t, vx_t, py_t, vy_t, pz_t, vz_t], ...],  # Target states
    "U_cmd": [[θx, θy], ...],                         # Control commands
    "quat_tray": [[qw, qx, qy, qz], ...],             # Tray orientations
    
    # Robot joint data (N×7 each)
    "L_qpos": [[q1, q2, ..., q7], ...],               # Left arm joint positions
    "R_qpos": [[q1, q2, ..., q7], ...],               # Right arm joint positions
    "L_qvel": [[qd1, qd2, ..., qd7], ...],            # Left arm joint velocities
    "R_qvel": [[qd1, qd2, ..., qd7], ...],            # Right arm joint velocities
    "L_torques": [[τ1, τ2, ..., τ7], ...],           # Left arm joint torques
    "R_torques": [[τ1, τ2, ..., τ7], ...],           # Right arm joint torques
    
    # End-effector data (N×3, N×6)
    "L_ee_pos": [[x, y, z], ...],                     # Left EE positions
    "R_ee_pos": [[x, y, z], ...],                     # Right EE positions
    "L_ee_vel": [[vx, vy, vz, ωx, ωy, ωz], ...],     # Left EE velocities
    "R_ee_vel": [[vx, vy, vz, ωx, ωy, ωz], ...],     # Right EE velocities
}

# Performance metrics
metrics = {
    "steady_state_error": 0.0023,        # Final position error (m)
    "convergence_time": 12.5,            # Time to reach target (s)
    "control_effort": 45.2,              # Total control effort
    "average_solve_time": 0.0125         # Average MPC solve time (s)
}
```

### Output Files

Results are automatically saved in `video_results/mpc_target_{target_string}/`:

- `experiment_data.npz` - Complete numerical data
- `experiment_metrics.json` - Performance metrics
- `experiment_info.json` - Experiment configuration
- `mujoco_simulation_{object_name}.mp4` - Recorded video
- `*_error.png` - Position error plots
- `*_control_effort.png` - Control input plots
- `*_joint_torques.png` - Robot torque plots
- `*_ee_positions.png` - End-effector trajectory plots

## MPC Parameter Tuning

Object-specific parameters are automatically selected:

### Cube Parameters
```python
mpc_params_cube = {
    "Qp": 600,    # Position error weight
    "Qv": 5,      # Velocity error weight  
    "R": 0.1,     # Control effort weight
    "N": 15       # Prediction horizon
}
```

### Cylinder Parameters
```python
mpc_params_cylinder = {
    "Qp": 400,    # Lower position weight
    "Qv": 2.5,    # Moderate velocity weight
    "R": 0.2,     # Higher control penalty
    "N": 15
}
```

### Sphere Parameters
```python
mpc_params_sphere = {
    "Qp": 200,    # Lowest position weight
    "Qv": 2,      # Low velocity weight
    "R": 0.2,     # Higher control penalty
    "N": 15
}
```

## Example Experiments

### Experiment 1: Cube Positioning
```bash
python main_parallel_enhanced.py \
    --target 0.1 0.0 0.0 0.0 0.0 0.0 \
    --object_name cube \
    --mass 0.5 \
    --friction 0.4 \
    --runtime 25.0
```

### Experiment 2: Cylinder with High Friction
```bash
python main_parallel_enhanced.py \
    --target 0.05 0.0 0.1 0.0 0.0 0.0 \
    --object_name cylinder \
    --mass 1.0 \
    --friction 0.6 \
    --runtime 35.0
```

### Experiment 3: Sphere Fast Motion
```bash
python main_parallel_enhanced.py \
    --target 0.15 0.0 -0.05 0.0 0.0 0.0 \
    --object_name sphere \
    --mass 0.3 \
    --friction 0.2 \
    --runtime 20.0 \
    --tolerance 0.015
```

## Performance Analysis

### Key Metrics
- **Steady-state error**: Final position accuracy
- **Convergence time**: Time to reach target within tolerance
- **Control effort**: Total energy expenditure
- **MPC solve time**: Real-time performance indicator

### Typical Performance
- **Position accuracy**: 1-5mm steady-state error
- **Convergence time**: 10-30 seconds depending on object type , object params and target distance
- **MPC frequency**: ~80-100 Hz with parallel implementation
- **Success rate**: >95% for targets within ±15cm

## Troubleshooting

### Common Issues

1. **MPC solver convergence**: Reduce prediction horizon or adjust weights
2. **Real-time performance**: Use parallel version or reduce MPC frequency
3. **Stability issues**: Check friction coefficients and object mass parameters
4. **Target unreachable**: Ensure target is within tray workspace limits

