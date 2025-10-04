import casadi as ca
import mujoco
import mujoco.viewer as viewer
import numpy as np
import time
from scipy.spatial.transform import Rotation as Rot
from icecream import ic
import multiprocessing as mp
import matplotlib.pyplot as plt

class PMPC:
    def __init__(self, model, data, Ts, nx=6, nu=2, N=20, Qp=100, Qv=0, R=0.1, mu=0.4, u_bounds=(-0.5, 0.5)):
        self.model = model
        self.data = data
        self.Ts = Ts
        self.nx = nx
        self.nu = nu
        self.N = N
        self.Qp = Qp
        self.Qv = Qv
        self.R = R
        self.mu = mu
        self.g = model.opt.gravity[2]
        self.h_cube = 0.1
        self.u_bounds = u_bounds
        self.target_body = "cube"

        x_sym = ca.SX.sym('x', nx)
        u_sym = ca.SX.sym('u', nu)
        self.f = ca.Function("f", [x_sym, u_sym], [self._rk4_step(x_sym, u_sym, Ts)])

        X = ca.SX.sym('X', nx, N+1)
        U = ca.SX.sym('U', nu, N)
        P = ca.SX.sym('P', nx+nx)

        obj = 0
        g_constr = [X[:,0]-P[:nx]]

        for k in range(N):
            xk = X[:,k]
            uk = U[:,k]
            ref = P[nx:]

            pos_err = ca.vertcat(xk[0]-ref[0], xk[2]-ref[2])
            vel_err = ca.vertcat(xk[1]-ref[1], xk[3]-ref[3])
            obj += Qp*ca.sumsqr(pos_err) + Qv*ca.sumsqr(vel_err) + R*ca.sumsqr(uk)

            g_constr.append(X[:,k+1] - self.f(xk, uk))

            # vx, vy = xk[1], xk[3]
            # ax = self.g*ca.sin(uk[0]) - self.mu*vx
            # ay = self.g*ca.sin(uk[1]) - self.mu*vy
            # # tray tilt rates (angular velocity)
            # omega_x = (uk[0] - U[:,k-1][0])/Ts if k>0 else 0
            # omega_y = (uk[1] - U[:,k-1][1])/Ts if k>0 else 0

            # v_perp = xk[5] - (uk[0]*vx + uk[1]*vy)
            # a_perp = ( (self.f(xk, uk)[5] - xk[5]) / self.Ts ) - (uk[0]*ax + uk[1]*ay + omega_x*vx + omega_y*vy)
            # g_constr.append(v_perp)
            # g_constr.append(a_perp)

        # Terminal cost
        ref = P[nx:]
        pos_err_terminal = ca.vertcat(X[0,N]-ref[0], X[2,N]-ref[2])
        vel_err_terminal = ca.vertcat(X[1,N]-ref[1], X[3,N]-ref[3])
        obj += Qp*ca.sumsqr(pos_err_terminal) + Qv*ca.sumsqr(vel_err_terminal)

        g_constr = ca.vertcat(*g_constr)
        opt_vars = ca.vertcat(ca.reshape(X,-1,1), ca.reshape(U,-1,1))

        # Bounds
        lbx = []
        ubx = []
        for _ in range(N+1):
            lbx.extend([-ca.inf]*nx)
            ubx.extend([ ca.inf]*nx)
        for _ in range(N):
            lbx.extend([u_bounds[0], u_bounds[0]])
            ubx.extend([u_bounds[1], u_bounds[1]])

        self.nlp = {'x':opt_vars,'f':obj,'g':g_constr,'p':P}
        self.solver = ca.nlpsol('solver','ipopt',self.nlp,{'ipopt.print_level':0,'print_time':0})
        self.lbx = lbx
        self.ubx = ubx
        self.w0 = np.zeros(opt_vars.shape[0])

    def _dynamics(self, x, u):
        px, vx, py, vy, pz, vz = ca.vertsplit(x)
        theta_x, theta_y = ca.vertsplit(u)

        ax = self.g * ca.sin(theta_x) - self.mu * vx
        ay = self.g * ca.sin(theta_y) - self.mu * vy
        vz_new = -self.g*(theta_x**2 + theta_y**2) # approximation for sin

        az = (vz_new - vz) / self.Ts

        return ca.vertcat(vx, ax, vy, ay, vz_new, az) 

    def _rk4_step(self,x,u,Ts):
        k1 = self._dynamics(x,u)
        k2 = self._dynamics(x + Ts/2*k1, u)
        k3 = self._dynamics(x + Ts/2*k2, u)
        k4 = self._dynamics(x + Ts*k3, u)
        return x + Ts/6*(k1 + 2*k2 + 2*k3 + k4)

    def get_state(self):
        """
        Extracts the current state from Mujoco data.
        Returns: np.array([px, vx, py, vy, pz, vz])
        """
        pos = self.data.body(self.target_body).xpos
        vel = self.data.body(self.target_body).cvel[3:6]
        return np.array([pos[0], vel[0], pos[1], vel[1], pos[2], vel[2]])

    def solve(self, target):
        """
        Solve the MPC problem for given target.
        Returns:
            u_cmd (np.ndarray): First control input in optimal sequence.
            loss (float): Objective value.
        """
        state = self.get_state()
        init_guess = np.concatenate((np.tile(state, (self.N+1,)), np.zeros(self.nu*self.N)))
        sol = self.solver(
            x0=init_guess,
            p=np.concatenate([state, target]),
            # lbg=np.ones(self.nx*(self.N+1) + 2*self.N)*1e-4,
            # ubg=np.ones(self.nx*(self.N+1) + 2*self.N)*1e-4,
            lbg=0, ubg=0,
            lbx=self.lbx,
            ubx=self.ubx
        )
        w_opt = sol['x'].full().flatten()
        loss = sol['f'].full().flatten()
        self.w0 = w_opt

        U_opt = w_opt[self.nx*(self.N+1):].reshape(self.N, self.nu)
        return U_opt[0], loss

def mpc_worker(model_path, state_queue, control_queue, params):
    # Each process must load its own Mujoco model/data and CasADi objects
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    mpc = PMPC(model, data, **params)
    while True:
        item = state_queue.get()
        if item == "STOP":
            break
        state, target = item
        # Set the state in the mpc's data object for get_state()
        # (You may need to update data.qpos/qvel if get_state uses them)
        # For now, we just pass state directly to solve
        # Optionally, you can modify PMPC.solve to accept state as argument
        mpc_state = np.array(state)
        mpc.data.body(mpc.target_body).xpos[:] = [mpc_state[0], mpc_state[2], mpc_state[4]]
        mpc.data.body(mpc.target_body).cvel[3:6] = [mpc_state[1], mpc_state[3], mpc_state[5]]
        u_cmd, loss = mpc.solve(target)
        control_queue.put((u_cmd, loss))

if __name__ == "__main__":
    mp.set_start_method('spawn')  # safer for mujoco/casadi
    path = "models/free_tray_2dof.xml"
    model = mujoco.MjModel.from_xml_path(path)
    data = mujoco.MjData(model)
    viewer = viewer.launch_passive(model, data)

    # params = dict(
        # Ts=model.opt.timestep, nx=6, nu=2, N=20, Qp=100, Qv=1, R=0.1, u_bounds=(-0.6, 0.6)
    # )
    
    params = dict(
    Ts=model.opt.timestep,
    nx=6,
    nu=2,
    N=15,          # shorter horizon → less aggressive
    Qp=100,        # position error still important
    Qv=5,          # give velocity some weight (was 0)
    R=2.0,         # <<<< raise tilt penalty (was 0.1)
    u_bounds=(-0.25, 0.25)  # <<<< halve max tilt
)

    state_queue = mp.Queue()
    control_queue = mp.Queue()

    # Start MPC process
    mpc_proc = mp.Process(target=mpc_worker, args=(path, state_queue, control_queue, params))
    mpc_proc.start()

    targets = np.array([[0.0, 0, 0.0, 0, 0, 0]])
    idx = 0
    counter = 0

    u_cmd = np.zeros(2)
    loss = 0

    while viewer.is_running():
        mujoco.mj_forward(model, data)
        tray_pos = data.body("tray").xpos
        target = targets[idx] + np.array([tray_pos[0], 0, tray_pos[1], 0, tray_pos[2], 0])

        # Get current state
        pos = data.body("cube").xpos
        vel = data.body("cube").cvel[3:6]
        state = np.array([pos[0], vel[0], pos[1], vel[1], pos[2], vel[2]])

        # Send state and target to MPC process
        state_queue.put((state, target))

        # Non-blocking: try to get latest control from MPC process
        try:
            while True:
                u_cmd, loss = control_queue.get_nowait()
        except Exception:
            pass  # No new control yet

        ic(u_cmd)

        # Apply u_cmd as tilts about the x and y axes using the mocap marker orientation
        quat = Rot.from_euler('xyz', [u_cmd[1], -u_cmd[0], 0.0]).as_quat()
        quat_mj = np.array([quat[3], quat[0], quat[1], quat[2]])
        data.mocap_quat[0] = quat_mj

        mujoco.mj_step(model, data)

        if counter > 1000:
            idx = (idx + 1) % len(targets)
            counter = 0

        viewer.sync()
        counter += 1
        time.sleep(model.opt.timestep)

    # Clean up
    state_queue.put("STOP")
    mpc_proc.join()