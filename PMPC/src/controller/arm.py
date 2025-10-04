"""
multiproc_sharedmem_convimp.py
MuJoCo (main) + CasADi solver (worker) using multiprocessing.shared_memory
"""

import multiprocessing as mp
from multiprocessing import shared_memory

import casadi as ca#type: ignore
import mujoco#type: ignore
import mujoco.viewer as viewer#type: ignore
import numpy as np#type: ignore
from scipy.spatial.transform import Rotation as Rot#type: ignore
import time


class ARMCONTROL:
    """
    A parallel controller class that computes torques using a separate process.
    The main process handles simulation stepping and viewer, while this class
    manages the optimization process for torque computation.
    """

    def __init__(self, model, data, params):
        """
        Initialize the parallel controller.

        Args:
            model: MuJoCo model (used only for initialization, not stored)
            params: Dictionary containing controller parameters
        """
        mp.set_start_method("spawn", force=True)

        self.data = data
        self.model = model

        self.joint_names = params.get("joint_names",None)
        assert self.joint_names is not None, "joint_names must be specified in params"
        self.actuator_names = params.get("actuator_names",None)
        assert self.actuator_names is not None, "actuator_names must be specified in params"
        
        self.dof_ids = np.array([model.joint(name).id for name in self.joint_names])
        self.actuator_ids = np.array(
            [model.actuator(name).id for name in self.actuator_names]
        )
        
        self.mocap = params.get("mocap_name", None)  
        assert self.mocap is not None, "mocap_name must be specified in params"

        try:
            self.mocap_id = model.body(self.mocap).mocapid[0]
        except:
            self.mocap_id = None

        self.nq = len(self.dof_ids)
        self.params = params

        self.body_name = params.get("body_name", None)  
        assert self.body_name is not None, "body_name must be specified in params"
        
        self.offset = params.get("offset", 0.0)  
        
        self.ee_xpos = np.zeros(3) # exposing to class
        self.ee_xquat = np.array([1,0,0,0])

        # Create shared memory buffers and gather their names
        self.shapes = {
            "q": (self.nq,),
            "qd": (self.nq,),
            "qdd_prev": (self.nq,),
            "mocap_pos": (3,),
            "mocap_quat": (4,),
            "jac": (6, self.nq),
            "jacDot": (6, self.nq),
            "M": (self.nq, self.nq),
            "h": (self.nq,),
            "Mx_inv": (6, 6),
            "ee_pos": (3,),
            "ee_quat": (4,),
            "rotvec": (3,),
            "torque_out": (self.nq,),
            "loss_out": (1,),
        }
        self.shms = {}
        self.views = {}
        for key, shape in self.shapes.items():
            shm, arr = self.create_shm_array(shape)
            self.shms[key] = shm
            self.views[key] = arr

        # pack names to send to solver:q
        self.shm_names = {k: v.name for k, v in self.shms.items()}

        # Events
        self.events = {
            "state_ready": mp.Event(),
            "ctrl_ready": mp.Event(),
            "terminate": mp.Event(),
        }

        # Launch solver process
        self.solver_proc = mp.Process(
            target=ARMCONTROL.solver_worker,
            args=(self.shm_names, self.events, self.params, self.shapes, self.nq),
            daemon=True,
        )
        self.solver_proc.start()

        print(f"[ARMCONTROL] Initialized with {self.nq} DOFs")

    def compute_dynamics(self):
        """
        Compute all dynamics quantities needed for the controller.
        This should be called by the main process with current model/data.

        Returns:
            dict: Dictionary containing all computed dynamics quantities
        """

        # Compute jacobian
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacBody(
            self.model, self.data, jacp, jacr, self.model.body(self.body_name).id
        )
        jac6 = np.vstack([jacp[:, self.dof_ids], jacr[:, self.dof_ids]])

        # Mass matrix
        M_full = np.zeros((self.model.nv, self.model.nv))
        mujoco.mj_fullM(self.model, M_full, self.data.qM)
        M_arm = M_full[np.ix_(self.dof_ids, self.dof_ids)]

        # Compute Minv_all via mj_solveM on identity
        I_nv = np.eye(self.model.nv)
        buf = np.zeros((self.model.nv, self.model.nv))
        mujoco.mj_solveM(self.model, self.data, buf, I_nv)
        Minv_all = np.array(buf).reshape(self.model.nv, self.model.nv)
        Minv_arm = Minv_all[np.ix_(self.dof_ids, self.dof_ids)]
        Mx_inv = jac6 @ Minv_arm @ jac6.T

        # jacDot
        jacDotp = np.zeros((3, self.model.nv))
        jacDotr = np.zeros((3, self.model.nv))
        mujoco.mj_jacDot(
            self.model,
            self.data,
            jacDotp,
            jacDotr,
            np.array([0,0,self.offset],dtype=np.float64),
            self.model.body(self.body_name).id,
        )
        jacDot6 = np.vstack([jacDotp[:, self.dof_ids], jacDotr[:, self.dof_ids]])

        # Bias forces
        h = self.data.qfrc_bias[self.dof_ids].copy()

        # End-effector position and quaternion
        self.ee_xpos = self.data.body(self.model.body(self.body_name).id).xpos.copy()
        self.ee_quat = self.data.body(self.model.body(self.body_name).id).xquat.copy()
        
        
        if self.offset != 0.0:
            ee_xmat = self.data.body(self.model.body(self.body_name).id).xmat.reshape(3, 3)
            ee_z_axis = ee_xmat[:, 2]
            self.ee_xpos = self.ee_xpos + self.offset * ee_z_axis

        # Mocap position/quaternion
        if self.mocap_id is not None:
            mocap_pos = self.data.mocap_pos[self.mocap_id].copy()
            mocap_quat = self.data.mocap_quat[self.mocap_id].copy()
        else:
            mocap_pos = np.zeros(3)
            mocap_quat = np.array([1.0, 0.0, 0.0, 0.0])

        # Quaternion error -> rotation vector
        quat_conj = np.zeros(4)
        mujoco.mju_negQuat(quat_conj, self.ee_quat)
        error_quat = np.zeros(4)
        mujoco.mju_mulQuat(error_quat, mocap_quat, quat_conj)
        rot = Rot.from_quat(
            [error_quat[1], error_quat[2], error_quat[3], error_quat[0]]
        )
        rotvec = rot.as_rotvec()

        return {
            "q": self.data.qpos[self.dof_ids].copy(),
            "qd": self.data.qvel[self.dof_ids].copy(),
            "qdd_prev": self.views["qdd_prev"].copy(),
            "jac": jac6,
            "jacDot": jacDot6,
            "M": M_arm,
            "h": h,
            "Mx_inv": Mx_inv,
            "ee_pos": self.ee_xpos,
            "ee_quat": self.ee_xquat,
            "mocap_pos": mocap_pos,
            "mocap_quat": mocap_quat,
            "rotvec": rotvec,
        }

    def compute_torque(self):
        """
        Compute torques for the current state.
        This is the main method to be called by the main process.

        Args:
            model: MuJoCo model
            data: MuJoCo data

        Returns:
            tuple: (torque_cmd, loss) where torque_cmd is numpy array and loss is float
        """
        # Compute dynamics
        dynamics = self.compute_dynamics()

        # Write to shared memory
        for key, value in dynamics.items():
            self.views[key][:] = value

        # Signal solver that new state is ready
        self.events["state_ready"].set()

        # Wait for solver to compute torques
        self.events["ctrl_ready"].wait(timeout=0.005)
        self.events["ctrl_ready"].clear()

        # Read results
        torque_cmd = self.views["torque_out"].copy()
        loss = float(self.views["loss_out"][0])

        return torque_cmd, loss

    @staticmethod
    def safe_matrix_sqrt(matrix):
        # Eigenvalue decomposition
        eigvals, eigvecs = np.linalg.eigh(matrix)

        # Take square root of the absolute values of the eigenvalues
        sqrt_eigvals = np.sqrt(np.abs(eigvals))

        # Reconstruct the square root of the matrix
        sqrt_matrix = eigvecs @ np.diag(sqrt_eigvals) @ eigvecs.T

        return sqrt_matrix

    @staticmethod
    def compute_damping_matrices(Mx, K):
        # Compute the square roots using the safe square root function
        sqrt_Mx = ARMCONTROL.safe_matrix_sqrt(Mx)
        sqrt_K = np.sqrt(K)  # K has no negative eigenvalues

        # Compute the damping matrices
        D = sqrt_Mx @ sqrt_K + sqrt_K @ sqrt_Mx
        return D

    def create_shm_array(self, shape, dtype=np.float64, name_prefix="shm"):
        """Create SharedMemory and return (SharedMemory, numpy.ndarray)"""
        dtype = np.dtype(dtype)
        size = int(np.prod(shape)) * dtype.itemsize
        shm = shared_memory.SharedMemory(create=True, size=size)
        arr = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
        arr[:] = 0  
        return shm, arr

    @staticmethod
    def solver_worker(shm_names, events, params, shapes, nq):
        """
        Attach to shared memory by names and run solver loop.
        Warm-starts CasADi/Ipopt using last primal & dual solutions.
        """
        # Attach shared memory
        views, shms = {}, {}
        for key, shape in shapes.items():
            shm = shared_memory.SharedMemory(name=shm_names[key])
            shms[key] = shm
            views[key] = np.ndarray(shape, dtype=np.float64, buffer=shm.buf)

        # Unpack params
        Wimp = params["Wimp"]
        Wpos = params["Wpos"]
        Wsmooth = params["Wsmooth"]
        Qmin = params["Qmin"]
        Qmax = params["Qmax"]
        Qdotmin = params["Qdotmin"]
        Qdotmax = params["Qdotmax"]
        taumin = params["taumin"]
        taumax = params["taumax"]
        K = params["K"]
        K_null = params["K_null"]
        dt = params["dt"]

        # Convert constant weights once
        Wimp_DM = ca.DM(Wimp)
        Wpos_DM = ca.DM(Wpos)
        Wsmooth_DM = ca.DM(Wsmooth)

        # Persisted warm-start state (in worker only)
        prev_qdd = None
        prev_lam_x = None
        prev_lam_g = None

        # IPOPT warm-start friendly options
        solver_opts = {
            "ipopt.print_level": 0,
            "print_time": False,
            "ipopt.sb": "yes",
            # --- warm start knobs ---
            "ipopt.warm_start_init_point": "yes",
            "ipopt.mu_init": 0.1,  # often helps
            "ipopt.warm_start_bound_push": 1e-6,
            "ipopt.warm_start_mult_bound_push": 1e-6,
            "ipopt.warm_start_slack_bound_push": 1e-6,
            # "ipopt.max_iter": 100,                # keep latency in check
        }

        print("[solver] started (warm-start enabled)")

        try:
            while True:
                events["state_ready"].wait(timeout=0.01)
                if events["terminate"].is_set():
                    break
                events["state_ready"].clear()

                # Read latest snapshot
                q = views["q"].copy()
                qd = views["qd"].copy()
                qdd_prev_sh = views["qdd_prev"].copy()  # shared copy of previous qdd
                mocap_pos = views["mocap_pos"].copy()
                jac = views["jac"].copy()
                jacDot = views["jacDot"].copy()
                M = views["M"].copy()
                h = views["h"].copy()
                Mx_inv = views["Mx_inv"].copy()
                ee_pos = views["ee_pos"].copy()
                rotvec = views["rotvec"].copy()

                n = nq
                qdd = ca.MX.sym("qdd", n)

                dx = mocap_pos - ee_pos
                twist = np.zeros(6)
                twist[:3] = dx
                twist[3:] = rotvec

                # Dynamics terms
                try:
                    Minv = np.linalg.pinv(M, rcond=1e-6)
                except Exception:
                    Minv = np.linalg.pinv(M)

                try:
                    if abs(np.linalg.det(Mx_inv)) > 1e-8:
                        Mx = np.linalg.inv(Mx_inv)
                    else:
                        Mx = np.linalg.pinv(Mx_inv, rcond=1e-3)
                except Exception:
                    Mx = np.linalg.pinv(Mx_inv, rcond=1e-3)

                # mu and damping
                mu_np = Mx @ (jac @ (Minv @ h) + jacDot @ qd)

                def safe_matrix_sqrt(matrix):
                    eigvals, eigvecs = np.linalg.eigh(matrix)
                    sqrt_eigvals = np.sqrt(np.abs(eigvals))
                    return eigvecs @ np.diag(sqrt_eigvals) @ eigvecs.T

                D_np = safe_matrix_sqrt(Mx) @ np.sqrt(K) + np.sqrt(
                    K
                ) @ safe_matrix_sqrt(Mx)

                # CasADi constants
                jac_ca = ca.DM(jac)
                jacDot_ca = ca.DM(jacDot)
                Mx_inv_ca = ca.DM(Mx_inv)
                D_ca = ca.DM(D_np)
                mu_ca = ca.DM(mu_np)
                K_ca = ca.DM(K)
                qd_ca = ca.DM(qd)
                twist_ca = ca.DM(twist)
                qdd_prev_ca = ca.DM(qdd_prev_sh)

                # Cost terms
                F_ca = -D_ca @ (jac_ca @ qd_ca) + K_ca @ twist_ca + mu_ca
                Eimp = jac_ca @ qdd + jacDot_ca @ qd_ca - Mx_inv_ca @ F_ca

                beta_np = 2.0 * np.sqrt(np.diag(K_null)) * (-qd) + (K_null @ (-q))
                Epos = qdd - ca.DM(beta_np)

                qddd = (qdd - qdd_prev_ca) / dt

                cost = (
                    ca.mtimes([Eimp.T, Wimp_DM, Eimp])
                    + ca.mtimes([Epos.T, Wpos_DM, Epos])
                    + ca.mtimes([-qddd.T, Wsmooth_DM, -qddd])
                )

                # Constraints
                g1 = 0.5 * qdd * dt**2 + qd_ca * dt + ca.DM(q)
                g2 = qdd * dt + qd_ca
                tau_expr = ca.DM(M) @ qdd + ca.DM(h)
                g = ca.vertcat(g1, g2, tau_expr)

                lb_g = np.concatenate([Qmin, Qdotmin, taumin])
                ub_g = np.concatenate([Qmax, Qdotmax, taumax])

                nlp = {"x": qdd, "f": cost, "g": g}
                solver = ca.nlpsol("solver", "ipopt", nlp, solver_opts)

                # ---- Warm-start guesses ----
                # Prefer the last *optimal* qdd (prev_qdd), otherwise fall back to shared qdd_prev, else zeros.
                if prev_qdd is not None:
                    x0 = prev_qdd
                elif qdd_prev_sh is not None and qdd_prev_sh.shape[0] == n:
                    x0 = qdd_prev_sh
                else:
                    x0 = np.zeros(n)

                # Dual guesses (shape must match)
                arg = dict(x0=x0, lbg=lb_g, ubg=ub_g, lbx=-ca.inf, ubx=ca.inf)
                if prev_lam_x is not None:
                    arg["lam_x0"] = prev_lam_x
                if prev_lam_g is not None:
                    arg["lam_g0"] = prev_lam_g

                # Solve
                try:
                    sol = solver(**arg)
                    x_sol = sol["x"].full().flatten()
                    lam_x = sol.get("lam_x", None)
                    lam_g = sol.get("lam_g", None)
                    tau = M @ x_sol + h
                    loss = float(sol["f"])
                    # Save warm-start data for next cycle
                    prev_qdd = x_sol
                    prev_lam_x = lam_x if lam_x is not None else prev_lam_x
                    prev_lam_g = lam_g if lam_g is not None else prev_lam_g
                except Exception:
                    # Keep previous warm-start to try again next cycle
                    x_sol = prev_qdd if prev_qdd is not None else np.zeros(n)
                    tau = np.zeros(n)
                    loss = -3.0

                # Publish solution & store for next warm start (shared)
                views["qdd_prev"][:] = x_sol
                views["torque_out"][:] = tau
                views["loss_out"][0] = loss

                events["ctrl_ready"].set()

        finally:
            for shm in shms.values():
                try:
                    shm.close()
                except:
                    pass
            print("[solver] exiting (warm-start worker)")

    def close(self):
        """
        Clean up resources and terminate the solver process.
        """
        self.events["terminate"].set()
        self.events["state_ready"].set()
        self.solver_proc.join(timeout=2.0)

        for shm in self.shms.values():
            try:
                shm.close()
                shm.unlink()
            except Exception:
                pass

        print("[ARMCONTROL] Closed")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


import matplotlib.pyplot as plt#type: ignore

torques = []

if __name__ == "__main__":

    path = "/home/autrio/linx/college/RRC/DualArmNPT/models_dual/xarm7/world.xml"
    model = mujoco.MjModel.from_xml_path(path)
    data = mujoco.MjData(model)

    viewer_obj = viewer.launch_passive(model, data)
    viewer_obj.opt.frame = mujoco.mjtFrame.mjFRAME_SITE

    R_params = {
        "joint_names" : [f"R_joint{i}" for i in range(1,8)],
        "actuator_names" : [f"R_act{i}" for i in range(1,8)],
        "body_name" : "xarm_R_gripper_base_link",
        "mocap_name" : "R_mocap_marker",
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
        "K": np.diag([1000.0, 1000.0, 1000.0, 50.0, 50.0, 50.0])
        * 10,  # 1D array for stiffness
        "K_null": np.diag([1] * 7),
        "dt": model.opt.timestep,
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
        "K": np.diag([1000.0, 1000.0, 1000.0, 50.0, 50.0, 50.0])
        * 10,  # 1D array for stiffness
        "K_null": np.diag([1] * 7),
        "dt": model.opt.timestep,
        "a": 1,
    }


    R_controller = ARMCONTROL(model, data, R_params)
    L_controller = ARMCONTROL(model, data, L_params)
    

    for i in range(model.njnt):
        jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        jtype = model.jnt_type[i] 
        limits = model.jnt_range[i]  
        print(f"Joint {i}: {jname}, type={jtype}, limits={limits}")

    print("[Main] Starting simulation loop")
    time.sleep(0.5)  # to account for warm starting

    gripper_actuator_ids = np.array([model.actuator(name).id for name in ["L_gripper", "R_gripper"]]
        )

    try:
        while viewer_obj.is_running():

            L_torque_cmd, L_loss = L_controller.compute_torque()
            torques.append(L_torque_cmd.copy())

            R_torque_cmd, R_loss = R_controller.compute_torque()
            torques.append(R_torque_cmd.copy())

            data.ctrl[L_controller.actuator_ids] = L_torque_cmd
            data.ctrl[R_controller.actuator_ids] = R_torque_cmd
            
            data.ctrl[gripper_actuator_ids] = [140, 140]

            mujoco.mj_step(model, data)

            viewer_obj.sync()

        if len(torques) > 1:
            torque_derivatives = np.diff(torques, axis=0)
            plt.figure(figsize=(12, 8))
            plt.subplot(2, 1, 1)
            plt.plot(torques)
            plt.title("Joint Torques Over Time")
            plt.xlabel("Time Step")
            plt.ylabel("Torque (N⋅m)")
            plt.legend([f"Joint {i + 1}" for i in range(len(torques[0]))])
            plt.grid(True)

            plt.subplot(2, 1, 2)
            plt.plot(torque_derivatives)
            plt.title("Joint Torque Derivatives Over Time")
            plt.xlabel("Time Step")
            plt.ylabel("Torque Derivative (N⋅m/step)")
            plt.legend([f"Joint {i + 1}" for i in range(len(torque_derivatives[0]))])
            plt.grid(True)

            plt.tight_layout()
            plt.show()

    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        L_controller.close()
        R_controller.close()
        viewer_obj.close()
        print("Main exiting")
