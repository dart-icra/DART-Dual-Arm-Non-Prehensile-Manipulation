import mujoco
import multiprocessing as mp
from multiprocessing import shared_memory

import casadi as ca
import mujoco
import mujoco.viewer as viewer
import numpy as np
from scipy.spatial.transform import Rotation as Rot
import time
from icecream import ic

from controller.parallel import ARMCONTROL

class DACTL:
    def __init__(self,model,data,L_params,R_params):
        
        self.data = data
        self.model = model
        
        self.L_arm_control = ARMCONTROL(model,data,L_params)
        self.R_arm_control = ARMCONTROL(model,data,R_params)

        self.L_mocap_id = model.body("L_mocap_marker").mocapid[0]
        self.R_mocap_id = model.body("R_mocap_marker").mocapid[0]

        self.dof_ids = np.hstack([self.L_arm_control.dof_ids, self.R_arm_control.dof_ids])
        self.actuator_ids = np.hstack([self.L_arm_control.actuator_ids, self.R_arm_control.actuator_ids])

    def _resolve_ee_pos(self, desired_obj_pos, desired_obj_quat):
        """
        object quaternion to ee poses
        """

        T_wo_R = Rot.from_quat(desired_obj_quat, scalar_first=True).as_matrix()
        T_wo_t = np.array(desired_obj_pos)

        # --- Precomputed grasp transforms ---
        # These must be set at grasp time:
        self.L_grasp_pos, self.L_grasp_quat = np.array([-0.175, 0, 0]),np.array([0.5, 0.5, 0.5, 0.5])  
        self.R_grasp_pos, self.R_grasp_quat = np.array([ 0.175, 0, 0]),np.array([0.5, -0.5, -0.5, 0.5])

        # Left grasp transform
        T_ogL_R = Rot.from_quat(self.L_grasp_quat, scalar_first=True).as_matrix()
        T_ogL_t = self.L_grasp_pos

        # Right grasp transform
        T_ogR_R = Rot.from_quat(self.R_grasp_quat, scalar_first=True).as_matrix()
        T_ogR_t = self.R_grasp_pos

        # --- Compute EE world poses ---
        # EE^L = Obj * (EE relative to Obj)
        L_ee_pos = T_wo_t + T_wo_R @ T_ogL_t
        L_ee_rot = Rot.from_matrix(T_wo_R @ T_ogL_R)

        R_ee_pos = T_wo_t + T_wo_R @ T_ogR_t
        R_ee_rot = Rot.from_matrix(T_wo_R @ T_ogR_R)

        # --- Set mocap targets ---
        self.data.mocap_pos[self.L_mocap_id] = L_ee_pos
        self.data.mocap_pos[self.R_mocap_id] = R_ee_pos

        self.data.mocap_quat[self.L_mocap_id] = L_ee_rot.as_quat(scalar_first=True)
        self.data.mocap_quat[self.R_mocap_id] = R_ee_rot.as_quat(scalar_first=True)


    def control(self, pos, quat):

        self._resolve_ee_pos(pos, quat)

        L_torque_cmd, L_loss = self.L_arm_control.compute_torque()
        R_torque_cmd, R_loss = self.R_arm_control.compute_torque()

        return (L_torque_cmd, R_torque_cmd), (L_loss, R_loss)
    
