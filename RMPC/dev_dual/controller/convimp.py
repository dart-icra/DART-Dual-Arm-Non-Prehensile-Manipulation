import mujoco
import mujoco.viewer as viewer
import numpy as np
import cvxpy as cp
import casadi as ca
from scipy.spatial.transform import Rotation as Rot
import time
from icecream import ic

class CONVIMP:
    def __init__(self,model,data, params:dict):
        self.model = model
        self.data = data
        self.K = params["K"]
        self.Kpos = params["Kpos"]
        self.Kori = params["Kori"]
        self.K_null = params["K_null"]
        self.gravity_compensation = params["gravity_compensation"]
        self.dt = self.model.opt.timestep
        self.Wimp = params["Wimp"]
        self.Wpos = params["Wpos"]
        self.Qrange = params["Qrange"]
        self.Qdotrange = params["Qdotrange"]
        self.tauRange = params["tauRange"]
        self.drive_type = params.get("drive_type", "direct")  # Default to torque control if not specified
        
        self.K = np.diag(self.K)
        self.K_null = np.diag(self.K_null)
        
        self.joint_names = [self.model.jnt(name).name for name in range(self.model.njnt) if "joint" in self.model.jnt(name).name]
        self.actuator_names = [self.model.actuator(name).name for name in range(self.model.nu)]
        self.dof_ids = np.array([self.model.joint(name).id for name in self.joint_names])
        # print("Dof IDs:", self.dof_ids)
        self.actuator_ids = np.array([self.model.actuator(name).id for name in self.actuator_names])
        self.mocap_id = self.model.body("test_mocap_marker").mocapid[0]
        # self.mocap_id = self.model.body("target").mocapid[0]  # Use the mocap marker for the target
        # print("Mocap ID:", self.mocap_id)
        self.site_id = self.model.site("attachment_site").id
        self.jacp = np.zeros((3, self.model.nv), dtype=np.float64)
        self.jacr = np.zeros((3, self.model.nv), dtype=np.float64)
        self.jacDotp = np.zeros((3, self.model.nv), dtype=np.float64)
        self.jacDotr = np.zeros((3, self.model.nv), dtype=np.float64)

        # self.jacPrev = np.zeros((6,7),dtype=np.float64) #prev values of jac for finite difference jdot

    
        self.M_all = np.zeros((self.model.nv, self.model.nv))

        self.Mx = np.zeros((6, 6))

        self.eye = np.eye(7)

        self.twist = np.zeros(6)

        self.quat_conj = np.zeros(4)

        self.error_quat = np.zeros(4)
        
        self.ee_body_id = "link7"

        self.PosturalBias = self.data.qpos[self.dof_ids].copy()
        self.velocityBias = self.data.qvel[self.dof_ids].copy()


    def safe_matrix_sqrt(self,matrix):
        # Eigenvalue decomposition
        eigvals, eigvecs = np.linalg.eigh(matrix)
        
        # Take square root of the absolute values of the eigenvalues
        sqrt_eigvals = np.sqrt(np.abs(eigvals))
        
        # Reconstruct the square root of the matrix
        sqrt_matrix = eigvecs @ np.diag(sqrt_eigvals) @ eigvecs.T
        
        return sqrt_matrix

    def compute_damping_matrices(self, Mx, K):
        # Compute the square roots using the safe square root function
        sqrt_Mx = self.safe_matrix_sqrt(Mx)
        sqrt_K = np.sqrt(K)  # K has no negative eigenvalues
        
        # Compute the damping matrices
        D = sqrt_Mx @ sqrt_K + sqrt_K @ sqrt_Mx

        return D

    def ctrl(self):
        # End-effector position and orientation
        self.x = self.data.body(self.ee_body_id).xpos
        self.dx = self.data.mocap_pos[self.mocap_id] - self.x
        self.twist[:3] = self.Kpos * self.dx 
        

        mujoco.mju_negQuat(self.quat_conj, self.data.body(self.ee_body_id).xquat)
        mujoco.mju_mulQuat(self.error_quat, self.data.mocap_quat[self.mocap_id], self.quat_conj)
        mujoco.mju_quat2Vel(self.twist[3:], self.error_quat, 1.0)
        self.twist[3:] *= self.Kori 

        self.ro = Rot.from_quat([
            self.error_quat[1],  # x
            self.error_quat[2],  # y
            self.error_quat[3],  # z
            self.error_quat[0]   # w
        ]).as_rotvec()

        self.error = np.concatenate((self.dx, self.ro))

        # Jacobian
        mujoco.mj_jacBody(self.model, self.data, self.jacp, self.jacr, self.model.body(self.ee_body_id).id)
        self.jac = np.vstack([self.jacp[:,self.dof_ids], self.jacr[:,self.dof_ids]])  # Only take the first 7 rows for the arm

        # Mass matrix
        self.M = np.zeros((self.model.nv, self.model.nv))
        mujoco.mj_fullM(self.model, self.M, self.data.qM)
        self.M = self.M[np.ix_(self.dof_ids, self.dof_ids)]


        # Task-space inertia
        mujoco.mj_solveM(self.model, self.data, self.M_all, np.eye(self.model.nv))
        self.M_inv = self.M_all[np.ix_(self.dof_ids, self.dof_ids)]
        
        
        
        self.Mx_inv = self.jac @ self.M_inv @ self.jac.T
        if abs(np.linalg.det(self.Mx_inv)) >= 1e-2:
            self.Mx = np.linalg.inv(self.Mx_inv)
        else:
            self.Mx = np.linalg.pinv(self.Mx_inv, rcond=1e-2)

        mujoco.mj_jacDot(self.model, self.data, self.jacDotp, self.jacDotr, np.zeros(3,dtype=np.float64), self.model.body(self.ee_body_id).id)
        
        self.jacDot = np.vstack([self.jacDotp[:,self.dof_ids], self.jacDotr[:,self.dof_ids]])  # Only take the first 7 rows for the arm

        self.h = self.data.qfrc_bias[self.dof_ids]
        self.mu = self.Mx @ (self.jac @ self.M_inv @ self.h + self.jacDot @ self.data.qvel[self.dof_ids])
        self.D = self.compute_damping_matrices(self.Mx, self.K)
        self.q = self.data.qpos[self.dof_ids]
        self.qd = self.data.qvel[self.dof_ids]
        
        self.beta = 2*np.sqrt(self.K_null)@(self.velocityBias - self.qd) + self.K_null @ (self.PosturalBias - self.q)  
        
        self.qdd = cp.Variable(7) # 0-8 for left arm 9-17 for right arm

        self.F = - self.D @ (self.jac @ self.qd) + self.K @ self.twist + self.mu

        self.Eimp = self.jac @ self.qdd + self.jacDot @ self.qd - self.Mx_inv @ self.F

        self.Epos = self.qdd - self.beta

        # ic(self.F1)

        self.objective = cp.Minimize( cp.quad_form(self.Eimp , self.Wimp) + cp.quad_form(self.Epos , self.Wpos))

        self.constraints = [0.5 * self.qdd * self.dt**2 + self.qd * self.dt + self.q <= self.Qrange[1],
                            0.5 * self.qdd * self.dt**2 + self.qd * self.dt + self.q >= self.Qrange[0],

                            self.qdd*self.dt + self.qd <= self.Qdotrange[1],
                            self.qdd*self.dt + self.qd >= self.Qdotrange[0],

                            self.M @ self.qdd + self.h <= self.tauRange[1],
                            self.M @ self.qdd + self.h >= self.tauRange[0],
                            ]
        
        self.problem = cp.Problem(self.objective,self.constraints)
        
        try:
            self.loss = self.problem.solve(verbose=False)
            # print("============LOSS===========: ",self.loss)
        except:
            print("----------------------infeasible-------------------------------------")

        if self.qdd.value is None :
            self.tau = np.zeros(7,)
            self.loss = -2
            return self.tau, self.loss
        if(self.qdd.value.all() != None):
            self.tau = self.M @ self.qdd.value + self.h
            if self.drive_type == "direct":
                self.data.ctrl[self.actuator_ids] = self.qdd.value
            elif self.drive_type == "torque":
                self.data.ctrl[self.actuator_ids] = self.tau  
            
        print(self.loss)

        return self.tau, self.loss

class CONVIMP2:
    def __init__(self,model,data, params:dict):
            self.model = model
            self.data = data
            self.K = params["K"]
            self.Kpos = params["Kpos"]
            self.Kori = params["Kori"]
            self.K_null = params["K_null"]
            self.gravity_compensation = params["gravity_compensation"]
            self.dt = self.model.opt.timestep
            self.Wimp = params["Wimp"]
            self.Wpos = params["Wpos"]
            self.Qrange = params["Qrange"]
            self.Qdotrange = params["Qdotrange"]
            self.tauRange = params["tauRange"]
            self.drive_type = params.get("drive_type", "direct")  # Default to torque control if not specified
            
            self.K = np.diag(self.K)
            self.K_null = np.diag(self.K_null)
            
            self.joint_names = [self.model.jnt(name).name for name in range(self.model.njnt) if "joint" in self.model.jnt(name).name]
            self.actuator_names = [self.model.actuator(name).name for name in range(self.model.nu)]
            self.dof_ids = np.array([self.model.joint(name).id for name in self.joint_names])
            # print("Dof IDs:", self.dof_ids)
            self.actuator_ids = np.array([self.model.actuator(name).id for name in self.actuator_names])
            self.mocap_id = self.model.body("test_mocap_marker").mocapid[0]
            # self.mocap_id = self.model.body("target").mocapid[0]  # Use the mocap marker for the target
            # print("Mocap ID:", self.mocap_id)
            self.site_id = self.model.site("attachment_site").id
            self.jacp = np.zeros((3, self.model.nv), dtype=np.float64)
            self.jacr = np.zeros((3, self.model.nv), dtype=np.float64)
            self.jacDotp = np.zeros((3, self.model.nv), dtype=np.float64)
            self.jacDotr = np.zeros((3, self.model.nv), dtype=np.float64)

            # self.jacPrev = np.zeros((6,7),dtype=np.float64) #prev values of jac for finite difference jdot

        
            self.M_all = np.zeros((self.model.nv, self.model.nv))

            self.Mx = np.zeros((6, 6))

            self.eye = np.eye(7)

            self.twist = np.zeros(6)

            self.quat_conj = np.zeros(4)

            self.error_quat = np.zeros(4)
            
            self.ee_body_id = "link7"
            # self.ee_body_id = "A_tray"

            self.PosturalBias = self.data.qpos[self.dof_ids]
            self.velocityBias = self.data.qvel[self.dof_ids]


    def safe_matrix_sqrt(self,matrix):
        # Eigenvalue decomposition
        eigvals, eigvecs = np.linalg.eigh(matrix)
        
        # Take square root of the absolute values of the eigenvalues
        sqrt_eigvals = np.sqrt(np.abs(eigvals))
        
        # Reconstruct the square root of the matrix
        sqrt_matrix = eigvecs @ np.diag(sqrt_eigvals) @ eigvecs.T
        
        return sqrt_matrix

    def compute_damping_matrices(self, Mx, K):
        # Compute the square roots using the safe square root function
        sqrt_Mx = self.safe_matrix_sqrt(Mx)
        sqrt_K = np.sqrt(K)  # K has no negative eigenvalues
        
        # Compute the damping matrices
        D = sqrt_Mx @ sqrt_K + sqrt_K @ sqrt_Mx

        return D
    
    def ctrl(self):
        # End-effector position and orientation
        self.x = self.data.body(self.ee_body_id).xpos
        self.dx = self.data.mocap_pos[self.mocap_id] - self.x
        self.twist[:3] = self.dx 

        mujoco.mju_negQuat(self.quat_conj, self.data.body(self.ee_body_id).xquat)
        mujoco.mju_mulQuat(self.error_quat, self.data.mocap_quat[self.mocap_id], self.quat_conj)
        mujoco.mju_quat2Vel(self.twist[3:], self.error_quat, 1.0)
        # self.twist[3:] *= self.Kori * 1 / 1.0

        self.ro = Rot.from_quat([
            self.error_quat[1],
            self.error_quat[2],
            self.error_quat[3],
            self.error_quat[0]
        ]).as_rotvec()

        self.error = np.concatenate((self.dx, self.ro))

        # Jacobian
        mujoco.mj_jacBody(self.model, self.data, self.jacp, self.jacr, self.model.body(self.ee_body_id).id)
        self.jac = np.vstack([self.jacp[:, self.dof_ids], self.jacr[:, self.dof_ids]])

        # Mass matrix
        self.M = np.zeros((self.model.nv, self.model.nv))
        mujoco.mj_fullM(self.model, self.M, self.data.qM)
        self.M = self.M[np.ix_(self.dof_ids, self.dof_ids)]

        # Task-space inertia
        mujoco.mj_solveM(self.model, self.data, self.M_all, np.eye(self.model.nv))
        self.M_inv = self.M_all[np.ix_(self.dof_ids, self.dof_ids)]

        self.Mx_inv = self.jac @ self.M_inv @ self.jac.T
        if abs(np.linalg.det(self.Mx_inv)) >= 1e-2:
            self.Mx = np.linalg.inv(self.Mx_inv)
        else:
            self.Mx = np.linalg.pinv(self.Mx_inv, rcond=1e-2)

        mujoco.mj_jacDot(self.model, self.data, self.jacDotp, self.jacDotr, np.zeros(3, dtype=np.float64), self.model.body(self.ee_body_id).id)
        self.jacDot = np.vstack([self.jacDotp[:, self.dof_ids], self.jacDotr[:, self.dof_ids]])

        self.h = self.data.qfrc_bias[self.dof_ids]
        self.mu = self.Mx @ (self.jac @ self.M_inv @ self.h + self.jacDot @ self.data.qvel[self.dof_ids])
        self.D = self.compute_damping_matrices(self.Mx, self.K)

        self.q = self.data.qpos[self.dof_ids]
        self.qd = self.data.qvel[self.dof_ids]
        self.beta = 2*np.sqrt(self.K_null)@(self.velocityBias - self.qd) + self.K_null @ (self.PosturalBias - self.q)

        # CasADi variables
        qdd = ca.MX.sym('qdd', 7)

        self.F = - self.D @ (self.jac @ self.qd) + self.K @ self.twist + self.mu

        Eimp = self.jac @ qdd + self.jacDot @ self.qd - self.Mx_inv @ self.F
        Epos = qdd - self.beta

        # Quadratic cost
        cost = ca.mtimes([Eimp.T, self.Wimp, Eimp]) + ca.mtimes([Epos.T, self.Wpos, Epos])
        
        # vel_error = qdd * self.dt + self.qd
        # cost += ca.mtimes([vel_error.T, np.eye(7), vel_error])


        # Constraints
        g = []

        # Joint position limits
        g.append(0.5 * qdd * self.dt**2 + self.qd * self.dt + self.q)
        lb_g1 = self.Qrange[0]
        ub_g1 = self.Qrange[1]

        # Joint velocity limits
        g.append(qdd * self.dt + self.qd)
        lb_g2 = self.Qdotrange[0]
        ub_g2 = self.Qdotrange[1]

        # Torque limits
        g.append(self.M @ qdd + self.h)
        lb_g3 = self.tauRange[0]
        ub_g3 = self.tauRange[1]

        g = ca.vertcat(*g)
        lb_g = np.concatenate([lb_g1, lb_g2, lb_g3])
        ub_g = np.concatenate([ub_g1, ub_g2, ub_g3])

        # Create solver
        nlp = {'x': qdd, 'f': cost, 'g': g}
        
        opts = {
            "ipopt.print_level": 0,        # No IPOPT console output
            "print_time": False,           # Don't print total time
            "ipopt.sb": "yes"              # Suppress banner
        }
        
        solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

        # Solve
        sol = solver(lbx=-ca.inf, ubx=ca.inf, lbg=lb_g, ubg=ub_g)

        if sol['x'] is not None:
            qdd_sol = np.array(sol['x']).flatten()
            self.tau = self.M @ qdd_sol + self.h
            if self.drive_type == "direct":
                # self.data.ctrl[self.actuator_ids] = qdd_sol
                return qdd_sol, float(sol['f'])
            elif self.drive_type == "torque":
                # self.data.ctrl[self.actuator_ids] = self.tau
                return self.tau, float(sol['f'])
            return self.tau, float(sol['f'])
        else:
            self.tau = np.zeros(7,)
            return self.tau, -2
        
import threading

if __name__ == "__main__":
    path = "/home/autrio/linx/college/RRC/DualArmNPT/models/xarm7/scene.xml"
    model = mujoco.MjModel.from_xml_path(path)
    data = mujoco.MjData(model)
    viewer = viewer.launch_passive(model, data)
    viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE
    tauRange_lower = np.array([-50, -50, -30, -30, -30, -20, -20])
    tauRange_upper = np.array([ 50,  50,  30,  30,  30,  20,  20])
    Qdotrange_lower = np.array([-2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0])*1  # rad/s
    Qdotrange_upper = np.array([ 2.0,  2.0,  2.0,  2.0,  2.0,  2.0,  2.0])*1
    Qrange_lower = np.array([-6.28319, -2.059, -6.28319, -0.19198, -6.28319, -1.69297, -6.28319])
    Qrange_upper = np.array([ 6.28319,  2.0944,  6.28319,  3.927,    6.28319,  3.14159,  6.28319])

    # model.opt.timestep = 0.00002 
    
    solve_result = None
    solve_thread = None

    params = {
        "Kpos": 0.95,
        "Kori": 0.95,
        "K" : np.array([500.0, 500.0, 500.0, 5.0, 5.0, 5.0])*10,
        "K_null": np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])*1,
        "Wimp":np.diag([10.0, 10.0, 10.0, 1.0, 1.0, 1.0])*1 ,
        "Wpos": np.diag([1, 1, 1, 1, 1, 1,1])*0.01,
        "Qrange": np.vstack([Qrange_lower, Qrange_upper]),
        "Qdotrange": np.vstack([Qdotrange_lower, Qdotrange_upper]),
        "tauRange": np.vstack([tauRange_lower, tauRange_upper]),
        "gravity_compensation": True,
        "drive_type": "torque"
    }
    
    controller = CONVIMP2(model, data, params)
    
    def solve_target():
        global solve_result
        solve_result = controller.ctrl()
        time.sleep(1)  # Small sleep to allow the viewer to update
        
    u_cmd = np.zeros(7)
    prev_u_cmd = np.zeros(7)

    while viewer.is_running():
        if solve_thread is None or not solve_thread.is_alive():
        # If we have a result from a previous solve, use it
            if solve_result is not None:
                u_cmd, loss = solve_result
                prev_u_cmd = u_cmd
                solve_result = None
            else:
                u_cmd, loss = np.zeros(7), -1

        # Start a new thread for the next iteration
            solve_thread = threading.Thread(target=solve_target)
            solve_thread.daemon = True
            solve_thread.start()
            ic(u_cmd)
            
        # data.ctrl = u_cmd
            
            
        mujoco.mj_step(model, data)
        viewer.sync()