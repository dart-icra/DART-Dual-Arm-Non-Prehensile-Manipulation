import numpy as np
import multiprocessing as mp
import os
import time
import json
from datetime import datetime
import matplotlib.pyplot as plt


class AsyncLogger:
    """
    Logger class that runs in a separate process to avoid slowing down the main simulation.
    Collects and saves various system metrics during robot arm control experiments.
    """
    def __init__(self, log_dir, experiment_name, dt, max_steps, object_name, mass, friction):
        """
        Initialize the logger with experiment parameters.
        
        Args:
            log_dir (str): Directory to save logs
            experiment_name (str): Name of the experiment
            dt (float): Timestep for simulation
            max_steps (int): Maximum number of steps to log
            object_name (str): Name of the object being manipulated
            mass (float): Mass of the object
            friction (float): Friction coefficient
        """
        self.log_queue = mp.Queue()
        self.command_queue = mp.Queue()
        self.process = None
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        self.dt = dt
        self.max_steps = max_steps
        self.object_name = object_name
        self.mass = mass
        self.friction = friction
        
    def start(self):
        """Start the logger process."""
        self.process = mp.Process(
            target=self._logger_process, 
            args=(
                self.log_queue, 
                self.command_queue, 
                self.log_dir, 
                self.experiment_name,
                self.dt,
                self.max_steps,
                self.object_name,
                self.mass,
                self.friction
            )
        )
        self.process.daemon = True  # Process will exit when main program exits
        self.process.start()
        
    def log_data(self, data_dict):
        """
        Log data to the async process.
        
        Args:
            data_dict (dict): Dictionary containing data to log
        """
        self.log_queue.put(data_dict)
        
    def stop(self):
        """Stop the logger process and save all data."""
        self.command_queue.put("STOP")
        self.process.join()#type: ignore
    
    @staticmethod
    def _logger_process(log_queue, command_queue, log_dir, experiment_name, dt, max_steps, 
                        object_name, mass, friction):
        """
        Logger process main function. Runs in a separate process.
        
        Args:
            log_queue (mp.Queue): Queue for receiving log data
            command_queue (mp.Queue): Queue for receiving commands
            log_dir (str): Directory to save logs
            experiment_name (str): Name of the experiment
            dt (float): Timestep for simulation
            max_steps (int): Maximum number of steps to log
            object_name (str): Name of the object being manipulated
            mass (float): Mass of the object
            friction (float): Friction coefficient
        """
        # Initialize log storage
        logs = {
            # Standard logs
            "t": np.zeros(max_steps),
            "X": np.zeros((max_steps, 6)),  # Object state
            "X_target": np.zeros((max_steps, 6)),  # Target state
            "U_cmd": np.zeros((max_steps, 2)),  # Control commands
            "quat_tray": np.zeros((max_steps, 4)),  # Tray orientation
            "loss": np.zeros(max_steps),  # MPC loss
            "solve_time": np.zeros(max_steps),  # MPC solve time
            
            # Additional logs
            "L_torques": np.zeros((max_steps, 7)),  # Left arm torques
            "R_torques": np.zeros((max_steps, 7)),  # Right arm torques
            "L_qpos": np.zeros((max_steps, 7)),  # Left arm joint positions
            "R_qpos": np.zeros((max_steps, 7)),  # Right arm joint positions
            "L_qvel": np.zeros((max_steps, 7)),  # Left arm joint velocities
            "R_qvel": np.zeros((max_steps, 7)),  # Right arm joint velocities
            "L_ee_pos": np.zeros((max_steps, 3)),  # Left end effector position
            "R_ee_pos": np.zeros((max_steps, 3)),  # Right end effector position
            "L_ee_vel": np.zeros((max_steps, 6)),  # Left end effector velocity (linear+angular)
            "R_ee_vel": np.zeros((max_steps, 6)),  # Right end effector velocity (linear+angular)
        }
        
        # Track log index
        log_idx = 0
        running = True
        
        while running:
            # Check for commands
            try:
                cmd = command_queue.get_nowait()
                if cmd == "STOP":
                    running = False
            except:
                pass
            
            # Process log data
            try:
                data = log_queue.get(timeout=0.01)  # Small timeout to allow checking commands
                
                # Update logs with received data
                for key, value in data.items():
                    if key in logs:
                        logs[key][log_idx] = value
                
                log_idx += 1
                
                # Check if we've reached max capacity
                if log_idx >= max_steps:
                    # Double the size of all arrays
                    for key in logs:
                        if logs[key].ndim == 1:
                            logs[key] = np.concatenate([logs[key], np.zeros_like(logs[key])])
                        else:
                            logs[key] = np.concatenate([logs[key], np.zeros_like(logs[key])], axis=0)
                    max_steps *= 2
            except:
                # No data available, continue
                pass
        
        # Trim logs to actual size
        for key in logs:
            logs[key] = logs[key][:log_idx]
            
        # Calculate metrics
        if log_idx > 0:
            steady_state_error = np.linalg.norm(logs["X"][-1,[0,2]] - logs["X_target"][-1,[0,2]])
            
            # Calculate convergence time - when error drops below a threshold
            error_threshold = 0.01  # 1cm
            errors = np.linalg.norm(logs["X"][:,[0,2]] - logs["X_target"][:,[0,2]], axis=1)
            convergence_indices = np.where(errors < error_threshold)[0]
            
            if len(convergence_indices) > 0:
                convergence_time = logs["t"][convergence_indices[0]]
            else:
                convergence_time = logs["t"][-1]  # Never converged
                
            # Calculate control effort (integral of control input magnitudes)
            control_effort = np.sum(np.linalg.norm(logs["U_cmd"], axis=1)) * dt
                
            # Add metrics to logs
            metrics = {
                "steady_state_error": steady_state_error,
                "convergence_time": convergence_time,
                "control_effort": control_effort
            }
            
            # Create save directory
            save_dir = f"{log_dir}/{object_name}/mass={mass}_friction={friction}"
            os.makedirs(save_dir, exist_ok=True)
            
            # Save data
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"{save_dir}/{experiment_name}_{timestamp}"
            
            # Save raw data
            np.savez(f"{save_path}.npz", **logs, **metrics)
            
            # Generate and save plots
            # AsyncLogger._generate_plots(logs, metrics, save_path)
            
            print(f"Results saved to {save_path}.npz")
    
    @staticmethod
    def _generate_plots(logs, metrics, save_path):
        """Generate standard plots for the experiment results."""
        # Position error plot
        plt.figure(figsize=(10, 6))
        plt.plot(logs["t"], np.linalg.norm(logs["X"][:,[0,2]] - logs["X_target"][:,[0,2]], axis=1), 
                label="XY Error (m)")
        plt.xlabel("Time (s)")
        plt.ylabel("Error (m)")
        plt.title(f"XY Position Error Over Time\nSteady State Error: {metrics['steady_state_error']:.4f}m")
        plt.grid(True)
        plt.legend()
        plt.savefig(f"{save_path}_position_error.png")
        # plt.close()
        plt.show()
        
        # Control input plot
        plt.figure(figsize=(10, 6))
        plt.plot(logs["t"], logs["U_cmd"][:,0], label="u_x")
        plt.plot(logs["t"], logs["U_cmd"][:,1], label="u_y")
        plt.xlabel("Time (s)")
        plt.ylabel("Control Inputs")
        plt.title(f"Control Inputs Over Time\nTotal Control Effort: {metrics['control_effort']:.4f}")
        plt.grid(True)
        plt.legend()
        plt.savefig(f"{save_path}_control_effort.png")
        # plt.close()
        plt.show()
        
        # Joint torques plot
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        for i in range(7):
            plt.plot(logs["t"], logs["L_torques"][:,i], label=f"L_joint{i+1}")
        plt.xlabel("Time (s)")
        plt.ylabel("Torque (Nm)")
        plt.title("Left Arm Joint Torques")
        plt.grid(True)
        plt.legend()
        
        plt.subplot(2, 1, 2)
        for i in range(7):
            plt.plot(logs["t"], logs["R_torques"][:,i], label=f"R_joint{i+1}")
        plt.xlabel("Time (s)")
        plt.ylabel("Torque (Nm)")
        plt.title("Right Arm Joint Torques")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{save_path}_joint_torques.png")
        # plt.close()
        plt.show()
        
        # End effector position plot
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        plt.plot(logs["t"], logs["L_ee_pos"][:,0], label="X")
        plt.plot(logs["t"], logs["L_ee_pos"][:,1], label="Y")
        plt.plot(logs["t"], logs["L_ee_pos"][:,2], label="Z")
        plt.xlabel("Time (s)")
        plt.ylabel("Position (m)")
        plt.title("Left End Effector Position")
        plt.grid(True)
        plt.legend()
        
        plt.subplot(2, 1, 2)
        plt.plot(logs["t"], logs["R_ee_pos"][:,0], label="X")
        plt.plot(logs["t"], logs["R_ee_pos"][:,1], label="Y")
        plt.plot(logs["t"], logs["R_ee_pos"][:,2], label="Z")
        plt.xlabel("Time (s)")
        plt.ylabel("Position (m)")
        plt.title("Right End Effector Position")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{save_path}_ee_position.png")
        # plt.close()
        plt.show()
