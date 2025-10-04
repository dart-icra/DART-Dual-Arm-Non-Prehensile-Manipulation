import mujoco
import mujoco.viewer as viewer
import numpy as np
from src import PMPC
from src import DACTL
import argparse
import time

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", nargs=6, type=float, 
                        default=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        help="Target location for the cube in format [x, vx, y, vy, z, vz]")
    parser.add_argument("--friction", type=float, default=0.4, 
                        help="Friction coefficient between cube and tray")
    parser.add_argument("--object_name", type=str, default="cube", 
                        help="Name of the object to be manipulated")
    parser.add_argument("--world_path", type=str, default="./models/xarm7/world_cube_m=0.5_mu=0.05.xml", 
                        help="Path to the world XML file")
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
    
    # MPC parameters
    mpc_params = {
        "Ts": model.opt.timestep, 
        "nx": 6, 
        "nu": 2, 
        "N": 15, 
        "Qp": 400, 
        "Qv": 2, 
        "R": 0.2, 
        "u_bounds": (-0.6, 0.6), 
        "mu": args.friction
    }
    
    mpc_controller = PMPC(model, data, **mpc_params)
    mpc_controller.target_body = args.object_name

    # Gripper actuators
    gripper_actuator_ids = np.array([model.actuator(name).id 
                                    for name in ["L_gripper", "R_gripper"]])

    # Target configuration
    target_3d = np.array(args.target)
    pos = data.body("tray").xpos.copy()
    quat = data.body("tray").xquat.copy()

    # Reset and initialize
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    viewer_obj.sync()

    u_cmd = np.zeros(2)
    
    while viewer_obj.is_running():
        # Calculate target position relative to tray
        target_position = target_3d + np.array([
            data.body("tray").xpos[0], 0.0,
            data.body("tray").xpos[1], 0.0,
            data.body("tray").xpos[2], 0.0
        ])

        # Get cube state
        pos_cube = data.body(mpc_controller.target_body).xpos
        vel_cube = data.body(mpc_controller.target_body).cvel[3:6]
        state = np.array([pos_cube[0], vel_cube[0], pos_cube[1], vel_cube[1], pos_cube[2], vel_cube[2]])

        # Solve MPC (no multiprocessing - direct call)
        u_cmd, loss = mpc_controller.solve(target_position)

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
    


