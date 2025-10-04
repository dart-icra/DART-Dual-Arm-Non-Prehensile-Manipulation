import mujoco
import mujoco.viewer as viewer
import argparse
import numpy as np
from src import PMPC
from src import DACTL
import time
import multiprocessing as mp

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


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", nargs=6, type=float, 
                        default=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        help="Target location for the cube in format [x, vx, y, vy, z, vz]")
    parser.add_argument("--friction", type=float, default=0.4, 
                        help="Friction coefficient between cube and tray")
    parser.add_argument("--object_name", type=str, default="cube", 
                        help="Name of the object to be manipulated")
    parser.add_argument("--mass", type=float, default=0.5, 
                        help="Mass of the object to be manipulated")
    parser.add_argument("--runtime", type=float, default=30.0, 
                        help="Max runtime in seconds")
    parser.add_argument("--world_path", type=str, default="./models/xarm7/world_cube_m=0.5_mu=0.05.xml", 
                        help="Path to the world XML file")
    parser.add_argument("--no_tune", action="store_true", 
                        help="Use default MPC parameters without tuning")
    parser.add_argument("--tolerance", type=float, default=0.01, 
                        help="Tolerance for settling time")
    args = parser.parse_args()
    
    # Load model and create data
    model = mujoco.MjModel.from_xml_path(args.world_path)
    data = mujoco.MjData(model)
    viewer_obj = viewer.launch_passive(model, data)

    # Robot controller parameters
    R_params = {
        "joint_names": [f"R_joint{i}" for i in range(1, 8)],
        "actuator_names": [f"R_act{i}" for i in range(1, 8)],
        "body_name": "xarm_R_gripper_base_link",
        "mocap_name": "R_mocap_marker",
        "offset": 0.125,
        "Wimp": np.diag([10.0, 10.0, 10.0, 1.0, 1.0, 1.0]),
        "Wpos": np.eye(7) * 0.1,
        "Wsmooth": np.eye(7) * 0,
        "Qmin": np.array([-6.28319, -2.059, -6.28319, -0.19198, -6.28319, -1.69297, -6.28319]),
        "Qmax": np.array([6.28319, 2.0944, 6.28319, 3.927, 6.28319, 3.14159, 6.28319]),
        "Qdotmin": np.ones(7) * -20.0,
        "Qdotmax": np.ones(7) * 20.0,
        "taumin": np.array([-50, -50, -30, -30, -30, -20, -20]),
        "taumax": np.array([50, 50, 30, 30, 30, 20, 20]),
        "K": np.diag([5000.0, 5000.0, 5000.0, 50.0, 50.0, 50.0]) * 0.1 * 10,
        "K_null": np.diag([1] * 7),
        "dt": model.opt.timestep,
        "a": 1,
    }

    L_params = R_params.copy()
    L_params["joint_names"] = [f"L_joint{i}" for i in range(1, 8)]
    L_params["actuator_names"] = [f"L_act{i}" for i in range(1, 8)]
    L_params["body_name"] = "xarm_L_gripper_base_link"
    L_params["mocap_name"] = "L_mocap_marker"

    # Initialize controllers
    controller = DACTL(model, data, L_params, R_params)
    gripper_actuator_ids = np.array([model.actuator(name).id 
                                    for name in ["L_gripper", "R_gripper"]])

    # Define MPC parameters for different object types
    mpc_params_cube = {
        "Ts": model.opt.timestep, "nx": 6, "nu": 2, "N": 15, 
        "Qp": 600, "Qv": 5, "R": 0.1, "u_bounds": (-0.6, 0.6), "mu": args.friction
    }
    mpc_params_cylinder = {
        "Ts": model.opt.timestep, "nx": 6, "nu": 2, "N": 15, 
        "Qp": 400, "Qv": 2.5, "R": 0.2, "u_bounds": (-0.6, 0.6), "mu": args.friction
    }
    mpc_params_sphere = {
        "Ts": model.opt.timestep, "nx": 6, "nu": 2, "N": 15, 
        "Qp": 200, "Qv": 2, "R": 0.2, "u_bounds": (-0.6, 0.6), "mu": args.friction
    }
    mpc_params_general = {
        "Ts": model.opt.timestep, "nx": 6, "nu": 2, "N": 15, 
        "Qp": 300, "Qv": 2.0, "R": 0.2, "u_bounds": (-0.6, 0.6), "mu": args.friction
    }

    # Select MPC parameters based on object type
    if args.no_tune:
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

    # Setup multiprocessing for MPC
    state_queue = mp.Queue()
    control_queue = mp.Queue()

    mpc_proc = mp.Process(
        target=mpc_worker,
        args=(args.world_path, args.object_name, mpc_params, state_queue, control_queue)
    )
    mpc_proc.start()

    # Initialize simulation
    target_3d = np.array(args.target)
    pos = data.body("tray").xpos.copy()
    quat = data.body("tray").xquat.copy()

    # Reset and initialize
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    viewer_obj.sync()

    # Initial settling phase
    print("Running initial settling phase (2 seconds)...")
    settling_start = time.time()
    while time.time() - settling_start < 2.0:
        mujoco.mj_step(model, data)
        viewer_obj.sync()

    # Reset after settling
    print("Resetting to keyframe position...")
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    viewer_obj.sync()

    TIME_start = time.time()
    u_cmd = np.zeros(2)
    loss = 0
    solve_time = 0
    flag_settled = False

    try:
        while viewer_obj.is_running():
            current_time = time.time() - TIME_start

            # Check runtime limit
            if current_time > args.runtime:
                print("Reached runtime limit")
                break

            # Calculate target position relative to tray
            target_position = target_3d + np.array([
                data.body("tray").xpos[0], 0.0,
                data.body("tray").xpos[1], 0.0,
                data.body("tray").xpos[2], 0.0
            ])

            # Get cube state
            pos_cube = data.body(args.object_name).xpos
            vel_cube = data.body(args.object_name).cvel[3:6]
            state = np.array([pos_cube[0], vel_cube[0], pos_cube[1], vel_cube[1], pos_cube[2], vel_cube[2]])

            # Send state to MPC worker
            state_queue.put((state, target_position))

            # Get latest control command from MPC worker
            try:
                while True:
                    u_cmd, loss, solve_time = control_queue.get_nowait()
            except:
                pass

            # Apply control after stabilization period
            if current_time > 3.0:
                # Convert control to quaternion rotation
                angles = [u_cmd[1], -u_cmd[0], 0.0]
                cx, cy, cz = np.cos(np.array(angles) / 2.0)
                sx, sy, sz = np.sin(np.array(angles) / 2.0)
                
                quat = np.array([
                    cx * cy * cz + sx * sy * sz,
                    sx * cy * cz - cx * sy * sz,
                    cx * sy * cz + sx * cy * sz,
                    cx * cy * sz - sx * sy * cz
                ])

            # Control dual arm
            torques, _ = controller.control(pos, quat)
            data.ctrl[controller.actuator_ids] = np.hstack(torques)
            data.ctrl[gripper_actuator_ids] = [255, 255]

            # Step simulation
            mujoco.mj_step(model, data)
            viewer_obj.sync()

            # Check if target reached
            error = np.linalg.norm(state[[0, 2]] - target_3d[[0, 2]])
            print(f"Error: {error:.4f}", end="\r")
            
            if error < args.tolerance and not flag_settled:
                print(f"\nTarget reached within tolerance of {args.tolerance}. Stopping simulation.")
                print(f"Final state: {state}, Target: {target_3d}")
                print(f"Total time: {current_time:.2f} seconds")
                flag_settled = True

    except KeyboardInterrupt:
        print("\nExperiment interrupted by user")
    finally:
        # Stop the MPC worker
        state_queue.put("STOP")
        mpc_proc.join()
        print("Experiment completed successfully")