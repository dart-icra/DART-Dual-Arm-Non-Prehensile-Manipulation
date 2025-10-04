"""
Configuration examples for DualArmNPT experiments.

This file demonstrates different parameter configurations for various scenarios.
Copy and modify these examples for your specific use case.
"""

# Example 1: High precision cube manipulation
CUBE_HIGH_PRECISION = {
    "target": [0.08, 0.0, 0.05, 0.0, 0.0, 0.0],
    "object_name": "cube",
    "mass": 0.5,
    "friction": 0.4,
    "runtime": 35.0,
    "tolerance": 0.003,
    "world_path": "./models/xarm7/world_cube_m=0.5_mu=0.2.xml"
}

# Example 2: Fast cylinder positioning
CYLINDER_FAST = {
    "target": [0.12, 0.0, -0.08, 0.0, 0.0, 0.0],
    "object_name": "cylinder", 
    "mass": 1.0,
    "friction": 0.3,
    "runtime": 25.0,
    "tolerance": 0.008,
    "world_path": "./models/xarm7/world_cylinder_m=1.0_mu=0.1.xml"
}

# Example 3: Heavy sphere with high friction
SPHERE_HEAVY = {
    "target": [0.06, 0.0, 0.10, 0.0, 0.0, 0.0],
    "object_name": "sphere",
    "mass": 2.0, 
    "friction": 0.6,
    "runtime": 45.0,
    "tolerance": 0.012,
    "world_path": "./models/xarm7/world_sphere_m=2.0_mu=0.2.xml"
}

# Example 4: Light object, low friction (challenging)
LIGHT_LOW_FRICTION = {
    "target": [0.15, 0.0, 0.0, 0.0, 0.0, 0.0],
    "object_name": "cube",
    "mass": 0.2,
    "friction": 0.05,
    "runtime": 40.0,
    "tolerance": 0.015,
    "world_path": "./models/xarm7/world_cube_m=0.5_mu=0.05.xml"
}

# Custom MPC parameters (advanced users)
CUSTOM_MPC_PARAMS = {
    "Ts": 0.002,        # Timestep
    "nx": 6,            # State dimension
    "nu": 2,            # Control dimension  
    "N": 20,            # Prediction horizon
    "Qp": 800,          # Position error weight
    "Qv": 8,            # Velocity error weight
    "R": 0.05,          # Control effort weight
    "u_bounds": (-0.8, 0.8),  # Control limits
    "mu": 0.4           # Friction coefficient
}

# Robot controller parameters (advanced users)
CUSTOM_ARM_PARAMS = {
    "Wimp": [10.0, 10.0, 10.0, 1.0, 1.0, 1.0],  # Impedance weights
    "Wpos": 0.1,        # Position tracking weight
    "Wsmooth": 0.0,     # Smoothing weight
    "K": 5000.0,        # Stiffness gain
    "K_null": 1.0       # Null space stiffness
}

def get_bash_command(config_name, config):
    """Generate bash command for a given configuration."""
    cmd = f"python main_parallel_enhanced.py"
    cmd += f" --target {' '.join(map(str, config['target']))}"
    cmd += f" --object_name {config['object_name']}"
    cmd += f" --mass {config['mass']}"
    cmd += f" --friction {config['friction']}"
    cmd += f" --runtime {config['runtime']}"
    cmd += f" --tolerance {config['tolerance']}"
    cmd += f" --world_path {config['world_path']}"
    
    return f"# {config_name}\n{cmd}\n"

if __name__ == "__main__":
    print("Example bash commands:\n")
    
    configs = {
        "High Precision Cube": CUBE_HIGH_PRECISION,
        "Fast Cylinder": CYLINDER_FAST, 
        "Heavy Sphere": SPHERE_HEAVY,
        "Light Low Friction": LIGHT_LOW_FRICTION
    }
    
    for name, config in configs.items():
        print(get_bash_command(name, config))