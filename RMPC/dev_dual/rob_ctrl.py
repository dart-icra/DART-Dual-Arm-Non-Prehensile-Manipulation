from __future__ import annotations

import argparse as ap
import json
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict
import xml.etree.ElementTree as ET

import cv2
import mujoco
import mujoco.viewer as viewer
import numpy as np
from scipy.spatial.transform import Rotation as Rot

from dualctl import DACTL
from controller.np_mpc_adaptive_with_linear_regressor import AdaptiveNPMPCSmooth, RLS

# ---------- CLI ----------
parser = ap.ArgumentParser()
parser.add_argument("--model_path", type=str, default="./models_dual/xarm7/world_general.xml")
parser.add_argument("--headless", action="store_true")

# Object selection and physical params
parser.add_argument("--object", type=str, choices=["sphere", "cube", "cylinder"], required=True)
parser.add_argument("--mass", type=float, required=True, help="Object mass in kg")
parser.add_argument("--mu", type=float, nargs=3, metavar=("mu_t", "mu_tors", "mu_roll"),
                    default=[0.2, 0.2, 0.002], help="Friction coefficients (tangential, torsional, rolling)")

# Dimensions (meters). Sphere: --radius; Cube: --edge; Cylinder: --radius --height
parser.add_argument("--radius", type=float, help="Sphere or cylinder radius (m)")
parser.add_argument("--edge", type=float, help="Cube edge length (m)")
parser.add_argument("--height", type=float, help="Cylinder height (m)")

# Initial object pose (world). If not provided, place above tray center with small z offset
parser.add_argument("--obj_pos", type=float, nargs=3, metavar=("x", "y", "z"),help="Initial object Position (xyz)")
parser.add_argument("--obj_quat", type=float, nargs=4, metavar=("w", "x", "y", "z"),
                    help="Initial object orientation (wxyz)")

# Tray-relative target offsets (meters)
parser.add_argument("--tx", type=float, required=True, help="Target x offset relative to tray center (m)")
parser.add_argument("--ty", type=float, required=True, help="Target y offset relative to tray center (m)")

# Save Results
parser.add_argument('--save', action='store_true', help='Enable saving of results.')
 

args = parser.parse_args()

# ---------- Helpers ----------
def to_jsonable(obj):
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    return obj

def add_episode(store: Dict[str, Dict[str, Any]],
                ep_name: str,
                pos_err: Any,
                error_norm: Any,
                u_cmd: Any,
                torque: Any,
                timestep: Any) -> None:
    store[ep_name] = {
        "pos_err": pos_err,
        "pos_err_norm": error_norm,
        "u_cmd": u_cmd,
        "torque": torque,
        "timestep": timestep,
    }

def save_episodes_json(path: str | Path,
                       episodes: Dict[str, Dict[str, Any]],
                       pretty: bool = True) -> None:
    path = Path(path)
    indent = 2 if pretty else None
    with path.open("w", encoding="utf-8") as f:
        json.dump({"data": episodes}, f, indent=indent, ensure_ascii=False, allow_nan=False, default=to_jsonable)

def build_object_geom_elem(obj_type: str, mass: float, mu: np.ndarray, dims: Dict[str, float]) -> ET.Element:
    # Build a single <geom> with requested attributes; contype/conaffinity ensure contacts
    geom = ET.Element("geom")
    geom.set("material", "mat_obj")
    geom.set("mass", f"{mass:.9g}")
    geom.set("friction", f"{mu[0]:.9g} {mu[1]:.9g} {mu[2]:.9g}")
    geom.set("contype", "1")
    geom.set("conaffinity", "1")
    if obj_type == "sphere":
        if "radius" not in dims or dims["radius"] is None:
            raise ValueError("Sphere requires --radius")
        geom.set("name", "sphere_geom")
        geom.set("type", "sphere")
        geom.set("size", f"{dims['radius']:.9g}")
        geom.set("rgba", "0.9 0.1 0.1 1")
    elif obj_type == "cube":
        if "edge" not in dims or dims["edge"] is None:
            raise ValueError("Cube requires --edge")
        h = 0.5 * dims["edge"]
        geom.set("name", "cube_geom")
        geom.set("type", "box")
        geom.set("size", f"{h:.9g} {h:.9g} {h:.9g}")
        geom.set("rgba", "0.1 0.9 0.1 1")
    elif obj_type == "cylinder":
        if "radius" not in dims or "height" not in dims or dims["radius"] is None or dims["height"] is None:
            raise ValueError("Cylinder requires --radius and --height")
        half_h = 0.5 * dims["height"]
        geom.set("name", "cylinder_geom")
        geom.set("type", "cylinder")
        geom.set("size", f"{dims['radius']:.9g} {half_h:.9g}")
        geom.set("rgba", "0.2039 0.0824 0.2235 1")
    else:
        raise ValueError(f"Unsupported object type: {obj_type}")
    return geom

def rewrite_xml_object_block(xml_path: str,
                             obj_type: str,
                             mass: float,
                             mu: np.ndarray,
                             dims: Dict[str, float],
                             init_pos: np.ndarray | None,
                             init_quat: np.ndarray | None) -> Path:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # default_elm = root.find("default")
    # default_geom = None
    # for child in default_elm:
    #     if child.tag == "geom":
    #         default_geom = child
    #         break
    # default_geom.set("friction", f"{mu[0]:.9g} {mu[1]:.9g} {mu[2]:.9g}")
    for default in root.findall("default"):
        geom = default.find("geom")
        if geom is not None and ("class" not in default.attrib) and ("class" not in geom.attrib):
            geom.set("friction", f"{mu[0]:.9g} {mu[1]:.9g} {mu[2]:.9g}")

    # Find worldbody/object body
    ns = ""  # MJCF has no XML namespace
    worldbody = root.find("worldbody")
    assert worldbody is not None, "worldbody not found in XML"
    body_obj = None
    for b in worldbody.findall("body"):
        if b.get("name") == "object":
            body_obj = b
            break
    if body_obj is None:
        raise RuntimeError("No body named 'object' found in XML")

    # Remove existing geoms inside object body
    for g in list(body_obj.findall("geom")):
        body_obj.remove(g)

    # Insert the chosen geom
    geom = build_object_geom_elem(obj_type, mass, mu, dims)
    body_obj.append(geom)
    keyframe = root.find("keyframe")
    for k in keyframe.findall("key"):
        if k.get("name") == "home":
            key_home = k
    pos_home=key_home.get("qpos")
    pos_obj="".join(f"{init_pos[0]:.9g} {init_pos[1]:.9g} {init_pos[2]:.9g} {init_quat[0]:.9g} {init_quat[1]:.9g} {init_quat[2]:.9g} {init_quat[3]:.9g}" )
    # Set initial pose if provided (world frame)
    # if init_pos is not None:
    #     body_obj.set("pos", f"{init_pos[0]:.9g} {init_pos[1]:.9g} {init_pos[2]:.9g}")
    # if init_quat is not None:
    #     body_obj.set("quat", f"{init_quat[0]:.9g} {init_quat[1]:.9g} {init_quat[2]:.9g} {init_quat[3]:.9g}")
    key_home.set("qpos", pos_home+pos_obj)
    # Write to a temp path next to original
    out_path = Path(xml_path).with_name("world_general_auto.xml")
    tree.write(out_path, encoding="utf-8", xml_declaration=True)
    return out_path

# ---------- Prepare model from parametric XML ----------
def main():
    dims = {"radius": args.radius, "edge": args.edge, "height": args.height}
    mu = np.array(args.mu, dtype=float)

    # Initial object pose defaults to above tray; we use a placeholder here and update after compile if needed
    init_pos = [0, 0, 0.43] if args.obj_pos is None else np.array(args.obj_pos, dtype=float)
    init_quat = [0.707,0.707,0,0] if args.obj_quat is None else np.array(args.obj_quat, dtype=float)

    auto_xml = rewrite_xml_object_block(args.model_path, args.object, args.mass, mu, dims, init_pos, init_quat)

    model = mujoco.MjModel.from_xml_path(str(auto_xml))
    data = mujoco.MjData(model)

    # If initial pose not provided, place object above tray center with small z offset after compile
    if init_pos is None:
        tray_pos = data.body("tray").xpos.copy()
        # Use current object z + offset to avoid penetration
        obj_pos = data.body("object").xpos.copy()
        new_xyz = np.array([tray_pos[0], tray_pos[1], max(obj_pos[2], tray_pos[2] + 0.01)], dtype=float)
        body_id = model.body("object").id
        jnt_id = model.body_jntadr[body_id]
        qpos_adr = model.jnt_qposadr[jnt_id]
        data.qpos[qpos_adr:qpos_adr+3] = new_xyz
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    # ---------- Viewer/Renderer ----------
    camera_name = "result_cam"
    camera_id = model.camera(camera_name).id

    viewer_obj = None
    if not args.headless:
        viewer_obj = viewer.launch_passive(model, data, show_right_ui=False, show_left_ui=False)
        # viewer_obj.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        # viewer_obj.cam.fixedcamid = camera_id
        viewer_obj.sync()

    width, height, fps = 640, 480, 30
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    # Build descriptive filenames
    tx, ty = args.tx, args.ty
    mu_str = f"{mu[0]:.3f}-{mu[1]:.3f}-{mu[2]:.3f}"
    tag = f"{args.object}_m{args.mass:.3f}_mu{mu_str}_tx{tx:.3f}_ty{ty:.3f}"
    video_name = f"tray_{tag}.mp4"
    json_name = f"{tag}.json"
    if args.save:
        out = cv2.VideoWriter(video_name, fourcc, fps, (width, height))
        renderer = mujoco.Renderer(model, height=height, width=width)

    # ---------- Controller setup (unchanged structure, body names updated) ----------
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
        "dt": float(model.opt.timestep),
        "a": 1,
    }
    L_params = {
        "joint_names" : [f"L_joint{i}" for i in range(1,8)],
        "actuator_names" : [f"L_act{i}" for i in range(1,8)],
        "body_name" : "xarm_L_gripper_base_link",
        "mocap_name" : "L_mocap_marker",
        "offset" : 0.125,
        
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
        "K": np.diag([5000.0, 5000.0, 5000.0, 50.0, 50.0, 50.0])*0.1
        * 10,  # 1D array for stiffness
        "K_null": np.diag([1] * 7),
        "dt": model.opt.timestep,
        "a": 1,
    }

    gripper_actuator_ids = np.array([model.actuator(name).id for name in ["L_gripper", "R_gripper"]])

    controller = DACTL(model, data, L_params, R_params)
    Ts = float(model.opt.timestep)
    ctlr = AdaptiveNPMPCSmooth(model, data, Ts=Ts, nx=4, nu=2, N=20,
                            Qp=80.0, Qv=2.0, Ru=0.02, Rdu=1.0,
                            u_bounds=(-0.6, 0.6), du_bounds=(-0.06, 0.06),
                            vmax=0.2, v_eps=0.1, target_body="object")

    rls_x = RLS(p=7, theta0=np.zeros(7), P0=1e3, lam=0.995)
    rls_y = RLS(p=7, theta0=np.zeros(7), P0=1e3, lam=0.995)
    theta_hat = np.zeros(14)

    sid_target_bar_h = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "cross_bar_h")
    sid_target_bar_v = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "cross_bar_v")
    site_tolerance_ring = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tolerance_ring")

    # Set tray-relative target markers (XY only) and propagate
    model.site_pos[sid_target_bar_h, 0] = args.tx
    model.site_pos[sid_target_bar_h, 1] = args.ty
    model.site_pos[sid_target_bar_v, 0] = args.tx
    model.site_pos[sid_target_bar_v, 1] = args.ty
    model.site_pos[site_tolerance_ring, 0] = args.tx
    model.site_pos[site_tolerance_ring, 1] = args.ty
    # mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    # ---------- Logs ----------
    episodes: Dict[str, Dict[str, Any]] = {}
    time_log, err_pos_log, u_cmd_log, x_state_log = [], [], [], []
    loss_log_mpc, loss_log_ctrl, tray_state_log, torques_log, error_norm_log = [], [], [], [], []

    # ---------- Main loop ----------
    if not args.headless and viewer_obj is not None:
        viewer_obj.opt.frame = mujoco.mjtFrame.mjFRAME_WORLD
        viewer_obj.sync()

    target_offsets = np.array([args.tx, 0.0, args.ty, 0.0], dtype=float)
    tray_xy = np.array([data.body("tray").xpos[0], 0.0, data.body("tray").xpos[1], 0.0], dtype=float)
    target = tray_xy + target_offsets

    pos = data.body("tray").xpos.copy()
    quat = data.body("tray").xquat.copy()

    r_v = tray_xy.copy()
    dr_max, alpha_rg = 0.01, 0.5
    pos_tol = 0.01

    xk = ctlr.get_state()
    u_prev = np.zeros(2)
    prev_state = xk.copy()

    start_time = time.time()
    running = True
    while running:
        running = viewer_obj.is_running() if (viewer_obj is not None) else True

        # RLS update based on previous state
        xk = ctlr.get_state()
        ax_meas = (xk[1] - prev_state[1]) / Ts
        ay_meas = (xk[3] - prev_state[3]) / Ts
        phi_prev = np.array([prev_state[0], prev_state[1], prev_state[2], prev_state[3],
                            np.tanh(prev_state[1]/ctlr.v_eps), np.tanh(prev_state[3]/ctlr.v_eps), 1.0], dtype=float)
        rls_x.update(phi_prev, ax_meas)
        rls_y.update(phi_prev, ay_meas)
        theta_hat[:7] = rls_x.get()
        theta_hat[7:] = rls_y.get()

        # Reference governor toward target
        err_pos = np.array([target[0]-r_v[0], 0.0, target[2]-r_v[2], 0.0])
        step_pos = np.array([np.clip(err_pos[0], -dr_max, dr_max), 0.0, np.clip(err_pos[2], -dr_max, dr_max), 0.0])
        r_v = r_v + alpha_rg * step_pos

        # MPC
        Rref_flat = ctlr.build_ref_traj(xk, r_v, target, ctlr.N, ctlr.nx, step_fraction=0.2)
        u_cmd, loss_mpc = ctlr.solve(xk, u_prev, theta_hat, Rref_flat)

        # Tray orientation from MPC command
        quat = Rot.from_euler('xyz', [u_cmd[1], -u_cmd[0], 0.0]).as_quat(scalar_first=True)
        
        torques, loss_ctrl = controller.control(pos, quat)

        # Apply controls
        data.ctrl[controller.actuator_ids] = np.hstack(torques)
        data.ctrl[gripper_actuator_ids] = [255, 255]

        # Step simulation
        mujoco.mj_step(model, data)

        # Render offscreen and write video
        if args.save:
            renderer.update_scene(data, camera="result_cam")
            frame = renderer.render()
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)

        # Viewer interaction
        if not args.headless and viewer_obj is not None:
            viewer_obj.sync()

        # Logging
        
        err_vec = np.array([xk[0]-target[0], xk[2]-target[2]])
        err_norm = float(np.linalg.norm(err_vec))
        error_norm_log.append(err_norm)
        time_log.append(time.time() - start_time)
        err_pos_log.append([target[0]-xk[0], target[2]-xk[2]])
        u_cmd_log.append(u_cmd.copy())
        x_state_log.append(xk.copy())
        loss_log_mpc.append(loss_mpc)
        loss_log_ctrl.append(loss_ctrl)
        tray_state_log.append(quat.copy())
        torques_log.append(torques)

        if err_norm < pos_tol:
            end_time = time.time()
            converged_time = end_time - start_time
            print(f"[MPC] Converged at t={converged_time:.3f}s, pos_err={err_norm:.4f} m")
            add_episode(episodes, "ep1",
                        pos_err=err_pos_log,
                        error_norm=error_norm_log,
                        u_cmd=u_cmd_log,
                        torque=torques_log,
                        timestep=time_log)
            start_time=time.time()
            time_log = []
            err_pos_log = []   # [ex, ey]
            # err_vel_log = []   # [evx, evy]
            u_cmd_log = []     # raw MPC output
            # u_apply_log = []   # post-gate applied input
            x_state_log = []   # xk
            target_log = []    # target snapshot
            loss_log_mpc = []
            loss_log_ctrl = []
            tray_state_log = []
            torques_log = []
            error_norm_log=[]
            break
        prev_state = xk.copy()
        u_prev = u_cmd.copy()

    # Cleanup and save logs
    
    if args.save:
        renderer.close()
        out.release()
        save_episodes_json(json_name, episodes, pretty=True)
        print(f"Saved video to {video_name}")
        print(f"Saved JSON to {json_name}")
if __name__ == "__main__":
    main()