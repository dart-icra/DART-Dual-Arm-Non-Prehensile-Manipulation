import mujoco
import mujoco.viewer as viewer
import numpy as np

class OPSPACE:
    def __init__(self, model, data, params):
        self.model = model
        self.data = data
        self.params = params
        self.damping_ratio = params.get("damping_ratio", 1)

        self.K = params["K"]
        self.K_null = params["K_null"]
        self.gravity_compensation = params.get("gravity_compensation", True)

        self.dt = self.model.opt.timestep
        self.D = 2 * self.damping_ratio * np.sqrt(self.K)

        # Body/site names
        self.body_id = self.model.body("link7").id
        self.site_name = "attachment_site"
        self.site_id = self.model.site(self.site_name).id
        self.mocap_name = "test_mocap_marker"
        self.mocap_id = self.model.body(self.mocap_name).mocapid[0]

        # Joint/actuator indexing
        joint_names = params.get("joint_names", None)
        assert joint_names is not None, "joint_names must be specified in params"
        actuator_names = [f"act{i}" for i in range(1, 8)]
        self.dof_ids = np.array([self.model.joint(name).id for name in joint_names])
        self.actuator_ids = np.array([self.model.actuator(name).id for name in actuator_names])

        # Home configuration
        key_name = "home"
        self.key_id = self.model.key(key_name).id
        self.q0 = self.model.key(key_name).qpos

        # Buffers
        self.jacp = np.zeros((3, self.model.nv))
        self.jacr = np.zeros((3, self.model.nv))
        self.jacDotp = np.zeros((3, self.model.nv))
        self.jacDotr = np.zeros((3, self.model.nv))
        self.twist = np.zeros(6)
        self.site_quat_conj = np.zeros(4)
        self.error_quat = np.zeros(4)
        self.M_all = np.zeros((self.model.nv, self.model.nv))
        self.Mx = np.zeros((6, 6))
        
        self.prev_tau = np.zeros(len(self.actuator_ids))  # for low-pass filter
        self.alpha = params.get("lowpass_alpha", 0.001)   

    def reset(self):
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.key_id)

    def ctrl(self):
        # Compute position error
        dx = self.data.mocap_pos[self.mocap_id] - self.data.body(self.body_id).xpos
        self.twist[:3] = dx

        # Compute orientation error
        mujoco.mju_negQuat(self.site_quat_conj, self.data.body(self.body_id).xquat)
        mujoco.mju_mulQuat(self.error_quat, self.data.mocap_quat[self.mocap_id], self.site_quat_conj)
        mujoco.mju_quat2Vel(self.twist[3:], self.error_quat, 1.0)

        # Jacobian and its derivative
        mujoco.mj_jacBody(self.model, self.data, self.jacp, self.jacr, self.body_id)
        mujoco.mj_jacDot(self.model, self.data, self.jacDotp, self.jacDotr, np.zeros(3), self.body_id)
        
        self.jac = np.vstack([self.jacp[:, self.dof_ids], self.jacr[:, self.dof_ids]])
        self.jacDot = np.vstack((self.jacDotp[:, self.dof_ids], self.jacDotr[:, self.dof_ids]))

        # Inertia matrices
        mujoco.mj_solveM(self.model, self.data, self.M_all, np.eye(self.model.nv))
        
        self.M_inv = self.M_all[np.ix_(self.dof_ids, self.dof_ids)]
        
        Mx_inv = self.jac @ self.M_inv @ self.jac.T
        if abs(np.linalg.det(Mx_inv)) >= 1e-2:
            self.Mx = np.linalg.inv(Mx_inv)
        else:
            self.Mx = np.linalg.pinv(Mx_inv, rcond=1e-2)

        # Bias forces and Coriolis
        h = self.data.qfrc_bias[self.dof_ids]
        mu = self.Mx @ (self.jac @ self.M_inv @ h - self.jacDot @ self.data.qvel[self.dof_ids])

        # Task space torque
        tau = self.jac.T @ self.Mx @ (self.K * self.twist - self.D * (self.jac @ self.data.qvel[self.dof_ids]) + mu)

        # Null-space torque
        Jbar = self.M_inv @ self.jac.T @ self.Mx
        ddq = self.K_null * (self.q0[self.dof_ids] - self.data.qpos[self.dof_ids]) - 2 * self.damping_ratio * np.sqrt(self.K_null) * self.data.qvel[self.dof_ids]
        tau += (np.eye(len(self.dof_ids)) - self.jac.T @ Jbar.T) @ ddq

        # Gravity compensation
        if self.gravity_compensation:
            tau += self.data.qfrc_bias[self.dof_ids]

        # Clip to actuator limits
        np.clip(tau, *self.model.actuator_ctrlrange.T, out=tau)
        tau_filtered = self.alpha * tau[self.actuator_ids] + (1 - self.alpha) * self.prev_tau 
        self.prev_tau = tau_filtered

        # Apply filtered torque to actuators
        self.data.ctrl[self.actuator_ids] = tau_filtered

        return tau_filtered, np.linalg.norm(self.twist)


if __name__ == "__main__":
    # path = "/home/autrio/linx/college/RRC/DualArmNPT/models/panda/scene.xml"
    path = "/home/autrio/linx/college/RRC/DualArmNPT/models/xarm7/scene.xml"
    
    model = mujoco.MjModel.from_xml_path(path)
    data = mujoco.MjData(model)
    # model.opt.timestep = 0.0002 
    viewer = viewer.launch_passive(model, data)
    viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE
    tauRange_lower = np.array([-50, -50, -30, -30, -30, -20, -20])
    tauRange_upper = np.array([ 50,  50,  30,  30,  30,  20,  20])
    Qdotrange_lower = np.array([-2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0])*10  # rad/s
    Qdotrange_upper = np.array([ 2.0,  2.0,  2.0,  2.0,  2.0,  2.0,  2.0])*10
    Qrange_lower = np.array([-6.28319, -2.059, -6.28319, -0.19198, -6.28319, -1.69297, -6.28319])
    Qrange_upper = np.array([ 6.28319,  2.0944,  6.28319,  3.927,    6.28319,  3.14159,  6.28319])


    params = {
        "Kpos": 0.95,
        "Kori": 0.95,
        "K" : np.array([50.0, 50.0, 50.0, 5.0, 5.0, 5.0])*200,
        "K_null": np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])*100,
        "Wimp":np.diag([10.0, 10.0, 10.0, 1.0, 1.0, 1.0])*1 ,
        "Wpos": np.diag([1, 1, 1, 1, 1, 1,1])*0.01,
        "Qrange": np.vstack([Qrange_lower, Qrange_upper]),
        "Qdotrange": np.vstack([Qdotrange_lower, Qdotrange_upper]),
        "tauRange": np.vstack([tauRange_lower, tauRange_upper]),
        "gravity_compensation": True,
        "drive_type": "torque",
        "damping_ratio": 1,
        "lowpass_alpha": 0.001
    }
    
    controller = OPSPACE(model, data, params)
    while viewer.is_running():
        controller.ctrl()
        mujoco.mj_step(model, data)
        viewer.sync()