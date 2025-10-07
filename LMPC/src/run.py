import argparse as ap
import os
import time

import mujoco
import mujoco.viewer as viewer
import numpy as np
from analyitics import Logger
from controller.rlmpc2 import RLMPC, gen_targ
from dualctl import DACTL
from scipy.spatial.transform import Rotation as Rot

def gen_radial_targ(model,data,MIN_DIST=0.08, MAX_DIST=0.12):
    return np.array([
        0.05,0,0.05,0,0,0,0,0
    ])

arg_parser = ap.ArgumentParser()
arg_parser.add_argument("--headless", action="store_true")
arg_parser.add_argument("--env", type=str, default="general")
arg_parser.add_argument("--tag", type=str, default="")
arg_parser.add_argument("--test", action="store_true")
arg_parser.add_argument("--train", action="store_true")
arg_parser.add_argument("--logdir",type=str,default="")
args = arg_parser.parse_args()

assert not (args.test and args.train), "Choose either test or train mode"
training = args.train or not args.test

model_path = (
    "./models/xarm7/world_general.xml"
    if training
    else f"./models/xarm7/world_{args.env}.xml"
)


def find_geom(spec, name):
    for g in spec.geoms:
        if g.name == name:
            return g
    return None


def ensure_cpkt(dir_path):
    if not os.path.exists(dir_path):
        raise Exception(f"Checkpoint directory {dir_path} does not exist.")
    return dir_path


if __name__ == "__main__":
    path = model_path
    # model = mujoco.MjModel.from_xml_path(path)
    spec = mujoco.MjSpec.from_file(path)
    model = spec.compile()
    data = mujoco.MjData(model)

    if not args.headless:
        viewer_obj = viewer.launch_passive(model, data,show_left_ui=False,show_right_ui=False)
    # viewer_obj.opt.frame = mujoco.mjtFrame.mjFRAME_SITE

    logger = Logger("./data/logs")
    logger.set_fpath(f"{args.logdir}_test/{args.env}.npy")

    masses = [3, 2, 1]
    frictions = [0.2, 0.05, 0.1]
    objects = ["cube", "cylinder", "sphere"]

    R_params = {
        "joint_names": [f"R_joint{i}" for i in range(1, 8)],
        "actuator_names": [f"R_act{i}" for i in range(1, 8)],
        "body_name": "xarm_R_gripper_base_link",
        "mocap_name": "R_mocap_marker",
        "offset": 0.125,
        "Wimp": np.diag([10.0, 10.0, 10.0, 1.0, 1.0, 1.0]),
        "Wpos": np.eye(7) * 0.1,
        "Wsmooth": np.eye(7) * 0,  # Smoothing weights
        "Qmin": np.array(
            [-6.28319, -2.059, -6.28319, -0.19198, -6.28319, -1.69297, -6.28319]
        ),
        "Qmax": np.array([6.28319, 2.0944, 6.28319, 3.927, 6.28319, 3.14159, 6.28319]),
        "Qdotmin": np.ones(7) * -20.0,
        "Qdotmax": np.ones(7) * 20.0,
        "taumin": np.array([-50, -50, -30, -30, -30, -20, -20]),
        "taumax": np.array([50, 50, 30, 30, 30, 20, 20]),
        "K": np.diag([5000.0, 5000.0, 5000.0, 50.0, 50.0, 50.0])
        * 0.1
        * 10,  # 1D array for stiffness
        "K_null": np.diag([1] * 7),
        "dt": model.opt.timestep,
        "a": 1,
    }

    L_params = {
        "joint_names": [f"L_joint{i}" for i in range(1, 8)],
        "actuator_names": [f"L_act{i}" for i in range(1, 8)],
        "body_name": "xarm_L_gripper_base_link",
        "mocap_name": "L_mocap_marker",
        "offset": 0.125,
        "Wimp": np.diag([10.0, 10.0, 10.0, 1.0, 1.0, 1.0]),
        "Wpos": np.eye(7) * 0.1,
        "Wsmooth": np.eye(7) * 0,  # Smoothing weights
        "Qmin": np.array(
            [-6.28319, -2.059, -6.28319, -0.19198, -6.28319, -1.69297, -6.28319]
        ),
        "Qmax": np.array([6.28319, 2.0944, 6.28319, 3.927, 6.28319, 3.14159, 6.28319]),
        "Qdotmin": np.ones(7) * -20.0,
        "Qdotmax": np.ones(7) * 20.0,
        "taumin": np.array([-50, -50, -30, -30, -30, -20, -20]),
        "taumax": np.array([50, 50, 30, 30, 30, 20, 20]),
        "K": np.diag([5000.0, 5000.0, 5000.0, 50.0, 50.0, 50.0])
        * 0.1
        * 10,  # 1D array for stiffness
        "K_null": np.diag([1] * 7),
        "dt": model.opt.timestep,
        "a": 1,
    }

    mpc_params = {
        "Ts": model.opt.timestep,
        "nx": 8,
        "nu": 2,
        "N": 20,
        "Q": [200.0, 2.0, 200.0, 2.0, 0.0, 0.0, 0.0, 0.0],
        "Qt": [200.0, 2.0, 200.0, 2.0, 0.0, 0.0, 0.0, 0.0],
        "R": [0.1, 0.1, 1.0, 1.0],
        "u_bounds": (-0.4, 0.4),
        "body_name": "cube2",
        "body_id": None,  # Will be set automatically if None
        "g": 9.81,
        # "policy_hidden": [128, 128],
        # "value_hidden": [128],
        "policy_std_init": 0.1,
        "clip_eps": 0.2,
        "epochs": 16,
        "rollout_len": 2048,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "obs_dim": 8,
        "max_param_abs": 2.0,
        "max_delta_abs": 0.02,
        "vf_coef": 0.5,
        "ent_coef": 0.01,
        "w_pos": 40.0,
        "w_ctrl": 10.0,
        "w_d_ctrl": 10.0,
        "max_episode_steps": 20000,
        "checkpoint_dir": ensure_cpkt(f"./src/checkpoints/general{args.tag}/")
        if not training
        else f"./src/checkpoints/{args.env}{args.tag}/",
        "train": training,
    }

    gripper_actuator_ids = np.array(
        [model.actuator(name).id for name in ["L_gripper", "R_gripper"]]
    )

    controller = DACTL(model, data, L_params, R_params)

    ctlr = RLMPC(model, data, mpc_params)

    idx = 0
    # target = gen_targ(model, data, MAX_DIST=0.125) if training else gen_radial_targ(model, data, MIN_DIST=0.08, MAX_DIST=0.12)
    target = np.array([-0.1,0,-0.1,0,0,0,0,0]) #! for video submission only
    Rel_target = target.copy()
    print(target)

    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    data.mocap_pos[model.body("targ").mocapid[0]][:2] = [target[0], target[2]]

    pos = data.body("tray").xpos.copy()
    quat = data.body("tray").xquat.copy()

    if not args.headless:
        viewer_obj.sync()

    running = True
    num_episodes = 0

    start_time = time.time()
    max_episodes = 1500 if training else 5

    try:
        while running:
            if not args.headless:
                running = viewer_obj.is_running()

            if num_episodes >= max_episodes:
                running = False

            tray_pos = data.body("tray").xpos.copy()
            tray_quat = data.body("tray").xquat.copy()

            local_offset = np.array([Rel_target[0], Rel_target[2], 0.022])

            world_offset = Rot.from_quat(tray_quat, scalar_first=True).apply(
                local_offset
            )

            data.mocap_pos[model.body("targ").mocapid[0]] = tray_pos + world_offset

            target[[0, 2]] = data.mocap_pos[model.body("targ").mocapid[0]][:2]

            if ctlr.events["reset"].is_set():
                kf = 0
                if training:
                    geom = find_geom(spec, "cube_geom")
                    assert geom is not None, "Geometry 'cube2' not found in the model."
                    objtype = np.random.choice(objects)
                    if objtype == "cube":
                        geom.type = mujoco.mjtGeom.mjGEOM_BOX
                        kf = 0
                    elif objtype == "cylinder":
                        geom.type = mujoco.mjtGeom.mjGEOM_CYLINDER
                        kf = 1
                    else:
                        geom.type = mujoco.mjtGeom.mjGEOM_SPHERE
                        kf = 0
                    mass = np.random.choice(masses)
                    geom.mass = mass
                    geom.friction = np.random.choice(frictions) * np.array(
                        [1.0, 1.0, 0.01]
                    )

                    model, data = spec.recompile(model, data)

                    print("===================== new Params ======================")
                    print("friction ", model.geom("cube_geom").friction)
                    print("mass ", mass)
                    print("=======================================================")

                    if viewer_obj is not None:
                        viewer_obj.close()
                        viewer_obj = mujoco.viewer.launch_passive(model, data, show_left_ui=False,show_right_ui=False)

                    mujoco.mj_resetDataKeyframe(model, data, kf)
                    mujoco.mj_forward(model, data)

                    controller.rebind(model, data)
                    ctlr.rebind(model, data)

                    #! uptill rebinding only durting training rest during eval and train

                mujoco.mj_resetDataKeyframe(model, data, kf)
                mujoco.mj_forward(model, data)

                # Rel_target = gen_targ(model, data, MAX_DIST=0.125) if training else gen_radial_targ(model, data, MIN_DIST=0.08, MAX_DIST=0.12)
                Rel_target = np.array([-0.1,0,-0.1,0,0,0,0,0]) #! for video submission only
                print(f"[RLMPC] New target: {target}")

                logger.save() if not training else None
                start_time = time.time()
                num_episodes += 1
                ctlr.events["reset"].clear()

            u_cmd, loss = ctlr.solve(target)
            u_cmd *= -1

            quat = Rot.from_euler("xyz", [u_cmd[1], -u_cmd[0], 0.0]).as_quat(
                scalar_first=True
            )

            torques, loss = controller.control(pos, quat)

            data.ctrl[controller.actuator_ids] = np.hstack(torques)
            data.ctrl[gripper_actuator_ids] = [255, 255]

            mujoco.mj_step(model, data)

            sys_info = ctlr.measure()

            state = sys_info["state"]

            error = np.linalg.norm(state[[0, 2]] - target[[0, 2]])

            #! =======================================================================
            #! CHANGE THE PATH BELOW IN THE FORMAT <METRIC>_<OBJECT>_<MASS>_<FRICTION>
            #! ALSO GO THROUGH WORLD.XML, TO UPDATE FRICTION IN 2 DIFFERENT PLACES
            #! =======================================================================

            if not training and num_episodes >= 1:
                logger.log("pos_error", error)
                logger.log("u_cmd", u_cmd)
                logger.log("timestep", time.time() - start_time)
                logger.log("arm_torque", np.hstack(torques))
                logger.log("state",ctlr.get_state())

            ctlr.views["state_next"][:] = ctlr.get_state()

            if not args.headless:
                viewer_obj.sync()

    finally:
        ctlr.close()
        viewer_obj.close()
