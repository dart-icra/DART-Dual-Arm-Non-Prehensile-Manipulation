import mujoco
import mujoco.viewer as viewer
from scipy.spatial.transform import Rotation as Rot
import argparse
import numpy as np
from src import PMPC
from src import DACTL
from src.logger import AsyncLogger
from icecream import ic

import time
import multiprocessing as mp
import matplotlib.pyplot as plt
import os
import signal
import sys
import cv2
import threading
import queue
from multiprocessing import Process, Queue

def mpc_worker(model_path, target_body, params, state_queue, control_queue):
    """
    Worker process for MPC computations.
    
    Args:
        model_path (str): Path to the MuJoCo XML model file
        target_body (str): Name of the target body in the model
        params (dict): MPC parameters
        state_queue (mp.Queue): Queue for receiving state updates
        control_queue (mp.Queue): Queue for sending control commands
    """
    import mujoco
    from src import PMPC
    import numpy as np
    import time

    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    ctrl = PMPC(model, data, **params)
    ctrl.target_body = target_body

    while True:
        item = state_queue.get()
        if item == "STOP":
            break
        state, target = item
        data.body(ctrl.target_body).xpos[:] = [state[0], state[2], state[4]]
        data.body(ctrl.target_body).cvel[3:6] = [state[1], state[3], state[5]]

        t0 = time.time()
        u_cmd, loss = ctrl.solve(target)
        solve_time = time.time() - t0

        control_queue.put((u_cmd, loss, solve_time))


class VideoWriterThread:
    def __init__(self, filename, fourcc, fps, frame_size):
        self.queue = queue.Queue()
        self.writer = cv2.VideoWriter(filename, fourcc, fps, frame_size)
        self.running = True
        self.thread = threading.Thread(target=self._write_frames)
        self.thread.start()

    def _write_frames(self):
        while self.running or not self.queue.empty():
            try:
                frame = self.queue.get(timeout=0.1)
                self.writer.write(frame)
            except queue.Empty:
                continue

    def write(self, frame):
        if self.running:
            self.queue.put(frame)

    def stop(self):
        self.running = False
        self.thread.join()
        self.writer.release()

class VideoWriterProcess:
    def __init__(self, filename, fourcc, fps, frame_size):
        self.queue = Queue()
        self.process = Process(target=self._write_frames, args=(filename, fourcc, fps, frame_size))
        self.process.start()

    def _write_frames(self, filename, fourcc, fps, frame_size):
        writer = cv2.VideoWriter(filename, fourcc, fps, frame_size)
        while True:
            frame = self.queue.get()
            if frame is None:  # Stop signal
                break
            writer.write(frame)
        writer.release()

    def write(self, frame):
        self.queue.put(frame)

    def stop(self):
        self.queue.put(None)  # Send stop signal
        self.process.join()

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", nargs=6, type=float, default=[0.0, 0.0, 0.0 , 0.0 , 0.0 , 0.0],
                        help="Target location for the cube in format [x, vx, y, vy, z, vz]")
    parser.add_argument("--friction", type=float, default=0.4, help="Friction coefficient between cube and tray")
    parser.add_argument("--object_name", type=str, default="cube", help="Name of the object to be manipulated")
    parser.add_argument("--mass", type=float, default=0.5, help="Mass of the object to be manipulated")
    parser.add_argument("--runtime", type=float, default=20.0, help="Max runtime in seconds")
    parser.add_argument("--world_path", type=str, default="./models/xarm7/world_cube_m=0.5_mu=0.05.xml", 
                        help="Path to the world XML file")
    parser.add_argument("--no_display", action="store_true", help="Run without visualization")
    parser.add_argument("--no_tune", action="store_true", help="Use default MPC parameters without tuning")
    parser.add_argument("--tolerance", type=float, default=0.01, help="Tolerance for settling time.")
    args = parser.parse_args()
    flag_settled = False
    path = args.world_path
    model = mujoco.MjModel.from_xml_path(path)
    data = mujoco.MjData(model)
    sid_target_bar_h = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "cross_bar_h")
    sid_target_bar_v = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "cross_bar_v")
    site_tolerance_ring = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tolerance_ring")
    # Launch viewer if display is enabled
    if not args.no_display:
        viewer_obj = viewer.launch_passive(model, data)
    else:
        viewer_obj = None
        class DummyViewer:
            def __init__(self):
                self.is_running = lambda: True
            def sync(self):
                pass
        viewer_obj = DummyViewer()

    R_params = {
        "joint_names" : [f"R_joint{i}" for i in range(1,8)],
        "actuator_names" : [f"R_act{i}" for i in range(1,8)],
        "body_name" : "xarm_R_gripper_base_link",
        "mocap_name" : "R_mocap_marker",
        "offset" : 0.125,
        "Wimp": np.diag([10.0, 10.0, 10.0, 1.0, 1.0, 1.0]),
        "Wpos": np.eye(7) * 0.1,
        "Wsmooth": np.eye(7) * 0,
        "Qmin": np.array([-6.28319, -2.059, -6.28319, -0.19198, -6.28319, -1.69297, -6.28319]),
        "Qmax": np.array([6.28319, 2.0944, 6.28319, 3.927, 6.28319, 3.14159, 6.28319]),
        "Qdotmin": np.ones(7) * -20.0,
        "Qdotmax": np.ones(7) * 20.0,
        "taumin": np.array([-50, -50, -30, -30, -30, -20, -20]),
        "taumax": np.array([50, 50, 30, 30, 30, 20, 20]),
        "K": np.diag([5000.0, 5000.0, 5000.0, 50.0, 50.0, 50.0])*0.1*10,
        "K_null": np.diag([1] * 7),
        "dt": model.opt.timestep,
        "a": 1,
    }

    L_params = R_params.copy()
    L_params["joint_names"] = [f"L_joint{i}" for i in range(1,8)]
    L_params["actuator_names"] = [f"L_act{i}" for i in range(1,8)]
    L_params["body_name"] = "xarm_L_gripper_base_link"
    L_params["mocap_name"] = "L_mocap_marker"

    gripper_actuator_ids = np.array([model.actuator(name).id for name in ["L_gripper", "R_gripper"]])

    controller = DACTL(model, data, L_params, R_params)

    # Define MPC parameters for different object types
    mpc_params_cube = dict(
        Ts=model.opt.timestep, nx=6, nu=2, N=15, Qp=600, Qv=5, R=0.1, u_bounds=(-0.6, 0.6), mu=args.friction
    )
    mpc_params_cylinder = dict(
        Ts=model.opt.timestep, nx=6, nu=2, N=15, Qp=400, Qv=2.5, R=0.2, u_bounds=(-0.6, 0.6), mu=args.friction
    )
    mpc_params_sphere = dict(
        Ts=model.opt.timestep, nx=6, nu=2, N=15, Qp=200, Qv=2, R=0.2, u_bounds=(-0.6, 0.6), mu=args.friction
    )

    mpc_params_general = dict(
        Ts=model.opt.timestep, nx=6, nu=2, N=15, Qp=300, Qv=2.0, R=0.2, u_bounds=(-0.6, 0.6), mu=args.friction
    )
    if(args.no_tune):
        mpc_params = mpc_params_general
    else:
        if args.object_name == "cube":
            mpc_params = mpc_params_cube
        elif args.object_name == "cylinder":
            mpc_params = mpc_params_cylinder
        elif args.object_name == "sphere":
            mpc_params = mpc_params_sphere
        else:
            mpc_params = mpc_params_general
            print(f"Warning: Using general MPC parameters for unknown object '{args.object_name}'")
            # raise ValueError(f"Unknown object_name: {args.object_name}")
    
    mpc_target_body = args.object_name

    state_queue = mp.Queue()
    control_queue = mp.Queue()

    mpc_proc = mp.Process(
        target=mpc_worker,
        args=(path, mpc_target_body, mpc_params, state_queue, control_queue)
    )
    mpc_proc.start()

    target_3d = np.array(args.target)
    pos = data.body("tray").xpos.copy()
    quat = data.body("tray").xquat.copy()

    # First run the simulation for 2 seconds without MPC to let physics settle
    print("Running initial settling phase (2 seconds)...")
    settling_start = time.time()
    while time.time() - settling_start < 2.0:
        mujoco.mj_step(model, data)
        viewer_obj.sync()
        if not args.no_display:
            time.sleep(model.opt.timestep)
    
    # Reset to keyframe position after settling
    print("Resetting to keyframe position...")
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    viewer_obj.sync()
    
    # Run MPC for 0.5 seconds before data collection to stabilize
    print("Running stabilization phase (0.5 seconds)...")
    stabilization_start = time.time()
    
    u_cmd = np.zeros(2)
    loss = 0
    solve_time = 0
    data_collection_started = False
    
    # Actual simulation start time after initialization phases
    TIME_start = time.time() + 0.5  # Add 0.5 seconds to delay data collection start
    
    # === Preallocate logs ===
    max_steps = int(args.runtime / model.opt.timestep) + 5
    
    # Initialize AsyncLogger
    target_str = "_".join([f"{val:.2f}" for val in args.target])
    experiment_name = f"mpc_target_{target_str}"
    logger = AsyncLogger(
        log_dir="video_results",
        experiment_name=experiment_name,
        dt=model.opt.timestep,
        max_steps=max_steps,
        object_name=args.object_name,
        mass=args.mass,
        friction=args.friction
    )
    logger.start()
    
    # Set up signal handling for graceful shutdown
    def signal_handler(sig, frame):
        print("\nReceived interrupt signal. Shutting down gracefully...")
        # Stop the logger
        logger.stop()
        # Stop the MPC worker
        state_queue.put("STOP")
        mpc_proc.join(timeout=5)  # Wait up to 5 seconds for process to join
        sys.exit(0)
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Setup joint and actuator indices for easy access
    L_joint_ids = [model.joint(name).id for name in L_params["joint_names"]]
    R_joint_ids = [model.joint(name).id for name in R_params["joint_names"]]
    L_actuator_ids = [model.actuator(name).id for name in L_params["actuator_names"]]
    R_actuator_ids = [model.actuator(name).id for name in R_params["actuator_names"]]
    
    # Get body IDs for left and right end effectors
    L_ee_body_id = model.body(L_params["body_name"]).id
    R_ee_body_id = model.body(R_params["body_name"]).id

    # Video Capture Setup
    width, height = 1280, 720
    fps = 30
    output_filename = f'mujoco_simulation_{args.object_name}.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = VideoWriterProcess(output_filename, fourcc, fps, (width, height))
    renderer = mujoco.Renderer(model, height=height, width=width)

    try:
        while viewer_obj.is_running():
            current_wall_time = time.time()
            elapsed_time = current_wall_time - stabilization_start
            
            # Check if we're still in the stabilization phase (first 0.5 seconds)
            if not data_collection_started and elapsed_time >= 0.5:
                data_collection_started = True
                TIME_start = current_wall_time
                print("Starting data collection...")
            model.site_pos[sid_target_bar_h, 0] = target_3d[0]
            model.site_pos[sid_target_bar_h, 1] = target_3d[2]
            model.site_pos[sid_target_bar_v, 0] = target_3d[0]
            model.site_pos[sid_target_bar_v, 1] = target_3d[2]
            model.site_pos[site_tolerance_ring, 0] = target_3d[0]
            model.site_pos[site_tolerance_ring, 1] = target_3d[2]
            # Use the proper time reference depending on which phase we're in
            if data_collection_started:
                current_time = current_wall_time - TIME_start
                if current_time > args.runtime:
                    print("Reached runtime limit")
                    viewer_obj.close() #type: ignore
                    break
            else:
                current_time = 0  # No data collection during stabilization
            
            target_position = target_3d + np.array([
                data.body("tray").xpos[0], 0.0,
                data.body("tray").xpos[1], 0.0,
                data.body("tray").xpos[2], 0.0
            ])
            data.mocap_pos[model.body("mocap_marker").mocapid[0]] = target_position[[0, 2, 4]]
            pos_cube = data.body(mpc_target_body).xpos
            vel_cube = data.body(mpc_target_body).cvel[3:6]
            state = np.array([pos_cube[0], vel_cube[0], pos_cube[1], vel_cube[1], pos_cube[2], vel_cube[2]])

            state_queue.put((state, target_position))

            try:
                while True:
                    u_cmd, loss, solve_time = control_queue.get_nowait()
            except Exception:
                pass

            if elapsed_time > 3.0:
                # Create a rotation from Euler angles
                angles = [u_cmd[1], -u_cmd[0], 0.0]
                
                # Manual quaternion creation from Euler angles (xyz order)
                # This avoids dependency on scipy's changing API
                cx, cy, cz = np.cos(np.array(angles) / 2.0)
                sx, sy, sz = np.sin(np.array(angles) / 2.0)
                
                # Quaternion in w, x, y, z format (MuJoCo convention)
                quat = np.array([
                    cx * cy * cz + sx * sy * sz,
                    sx * cy * cz - cx * sy * sz,
                    cx * sy * cz + sx * cy * sz,
                    cx * cy * sz - sx * sy * cz
                ])

            torques, _ = controller.control(pos, quat)
            data.ctrl[controller.actuator_ids] = np.hstack(torques)
            data.ctrl[gripper_actuator_ids] = [255, 255]

            mujoco.mj_step(model, data)
            viewer_obj.sync()

            # --- Video Capture ---
            renderer.update_scene(data, camera=0)  # Use camera 0 or set your camera index
            frame = renderer.render()
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video_writer.write(frame_bgr)  # Asynchronous write

            # Extract additional data
            L_torques = data.actuator_force[L_actuator_ids].copy()
            R_torques = data.actuator_force[R_actuator_ids].copy()
            
            # Get joint positions and velocities by accessing each joint individually
            L_qpos = np.array([data.joint(joint_id).qpos[0] for joint_id in L_joint_ids])
            R_qpos = np.array([data.joint(joint_id).qpos[0] for joint_id in R_joint_ids])
            
            L_qvel = np.array([data.joint(joint_id).qvel[0] for joint_id in L_joint_ids])
            R_qvel = np.array([data.joint(joint_id).qvel[0] for joint_id in R_joint_ids])
            
            # End effector data
            L_ee_pos = data.body(L_ee_body_id).xpos.copy()
            R_ee_pos = data.body(R_ee_body_id).xpos.copy()
            
            L_ee_vel = np.concatenate([
                data.body(L_ee_body_id).cvel[3:6].copy(),  # Linear velocity
                data.body(L_ee_body_id).cvel[0:3].copy()   # Angular velocity
            ])
            
            R_ee_vel = np.concatenate([
                data.body(R_ee_body_id).cvel[3:6].copy(),  # Linear velocity
                data.body(R_ee_body_id).cvel[0:3].copy()   # Angular velocity
            ])
            
            # Only log data after the stabilization phase
            if data_collection_started:
                # Log data asynchronously
                logger.log_data({
                    "t": current_time,
                    "X": state,
                    "X_target": target_position,
                    "error": np.linalg.norm(state[[0,2]] - target_3d[[0,2]]),
                    "U_cmd": u_cmd,
                    "quat_tray": quat,
                    "loss": loss,
                    "solve_time": solve_time,
                    "L_torques": L_torques,
                    "R_torques": R_torques,
                    "L_qpos": L_qpos,
                    "R_qpos": R_qpos,
                    "L_qvel": L_qvel,
                    "R_qvel": R_qvel,
                    "L_ee_pos": L_ee_pos,
                    "R_ee_pos": R_ee_pos,
                    "L_ee_vel": L_ee_vel,
                    "R_ee_vel": R_ee_vel
                })
            
            # if not args.no_display:
            #     time.sleep(model.opt.timestep)
            print("Error:", np.linalg.norm(state[[0,2]] - target_3d[[0,2]]), end="\r")
            if data_collection_started and np.linalg.norm(state[[0,2]] - target_3d[[0,2]]) < args.tolerance and not flag_settled:
                print(f"Target reached within tolerance of {args.tolerance}. Stopping simulation.")
                print(f"Final state: {state}, Target: {target_3d}")
                print(f"Total time: {current_time:.2f} seconds")
                flag_settled = True
    except KeyboardInterrupt:
        print("Experiment interrupted by user")
    finally:
        # Stop the logger
        logger.stop()
        
        # Stop the MPC worker
        state_queue.put("STOP")
        mpc_proc.join()
        
        # Stop the video writer
        video_writer.stop()
        print(f"Video saved to {output_filename}")
        print("Experiment completed successfully")
