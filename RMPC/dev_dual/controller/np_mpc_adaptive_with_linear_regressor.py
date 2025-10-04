import casadi as ca
import mujoco
import mujoco.viewer as viewer
import numpy as np
from scipy.spatial.transform import Rotation as Rot
# from stick_slide_gate import StickSlideGate


# ---------- Recursive Least Squares ----------
class RLS:
    def __init__(self, p, theta0=None, P0=1e3, lam=0.995):
        self.p = p
        self.theta = np.zeros(p) if theta0 is None else theta0.astype(float)
        self.P = np.eye(p) * float(P0)
        self.lam = float(lam)

    def update(self, phi, y):
        # Ensure correct shapes/types
        phi = np.asarray(phi, dtype=float).reshape(-1)   # (p,)
        y = float(np.asarray(y).reshape(()))             # scalar

        denom = self.lam + phi @ self.P @ phi           # scalar
        K = (self.P @ phi) / denom                      # (p,)
        err = y - (phi @ self.theta)                    # scalar

        self.theta = self.theta + K * err               # (p,)
        self.P = (self.P - np.outer(K, phi) @ self.P) / self.lam  # (p,p)

    def get(self):
        return self.theta.copy()


# ---------- Adaptive NMPC with smooth tracking ----------
class AdaptiveNPMPCSmooth:
    def __init__(self, model, data, Ts, nx=4, nu=2, N=20,
                 Qp=100.0, Qv=1.0, Ru=0.05, Rdu=1.0,
                 u_bounds=(-0.4, 0.4), du_bounds=(-0.05, 0.05),
                 vmax=0.25, v_eps=0.1, target_body="cube"):
        self.model = model
        self.data = data
        self.Ts = float(Ts)
        self.nx = nx
        self.nu = nu
        self.N = N
        self.Qp = Qp
        self.Qv = Qv
        self.Ru = Ru
        self.Rdu = Rdu
        self.u_bounds = u_bounds
        self.du_bounds = du_bounds
        self.vmax = vmax
        self.v_eps = v_eps

        self.target_body = target_body
        self.body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, self.target_body)
        self.gz = float(self.model.opt.gravity[2])  # negative

        # Expanded friction/features per axis:
        # [px, vx, py, vy, tanh(vx/veps), tanh(vy/veps), 1] => 7 params per axis
        self.px = 7
        self.py = 7
        self.p_total = self.px + self.py  # 14

        # CasADi symbols
        x_sym  = ca.SX.sym('x', nx)                 # [px, vx, py, vy]
        u_sym  = ca.SX.sym('u', nu)                 # [alpha, beta]
        th_sym = ca.SX.sym('th', self.p_total)      # theta_hat (14,)

        self.f_disc = ca.Function(
            "f_disc",
            [x_sym, u_sym, th_sym],
            [self._rk4_step_regressor(x_sym, u_sym, th_sym, self.Ts)]
        )

        # Decision variables
        X = ca.SX.sym('X', nx, N+1)
        U = ca.SX.sym('U', nu, N)

        # Parameters: x0 (nx), u_prev (nu), theta_hat (p_total), ref_traj ((N+1)*nx)
        P = ca.SX.sym('P', nx + nu + self.p_total + (N+1)*nx)

        obj = 0
        g_list = []
        glb = []
        gub = []

        # Initial state equality
        x0 = P[0:nx]
        g_list.append(X[:, 0] - x0)
        glb += [0.0]*nx
        gub += [0.0]*nx

        u_prev = P[nx:nx+nu]
        theta_hat = P[nx+nu:nx+nu+self.p_total]
        Rref_flat = P[nx+nu+self.p_total:]

        # Helper to extract per-stage reference
        def ref_k(idx):
            base = idx*nx
            return Rref_flat[base:base+nx]

        # Build constraints and cost
        for k in range(N):
            xk = X[:, k]
            uk = U[:, k]
            rk = ref_k(k)

            # Dynamics
            x_next = self.f_disc(xk, uk, theta_hat)
            g_list.append(X[:, k+1] - x_next)
            glb += [0.0]*nx
            gub += [0.0]*nx

            # Tilt rate limits with u_prev at k=0
            if k == 0:
                du = U[:, 0] - u_prev
            else:
                du = U[:, k] - U[:, k-1]
            g_list.append(du)
            glb += [self.du_bounds[0]]*self.nu
            gub += [self.du_bounds[1]]*self.nu

            # Velocity caps: |vx|<=vmax, |vy|<=vmax
            g_list.append(ca.vertcat(xk[1] - self.vmax, -xk[1] - self.vmax,
                                     xk[3] - self.vmax, -xk[3] - self.vmax))
            glb += [-ca.inf, -ca.inf, -ca.inf, -ca.inf]
            gub += [0.0, 0.0, 0.0, 0.0]

            # Costs: position/velocity tracking to staged reference + move cost + move suppression
            pos_err = ca.vertcat(xk[0] - rk[0], xk[2] - rk[2])
            vel_err = ca.vertcat(xk[1] - rk[1], xk[3] - rk[3])
            obj += self.Qp * ca.sumsqr(pos_err) + self.Qv * ca.sumsqr(vel_err) \
                   + self.Ru * ca.sumsqr(uk) + self.Rdu * ca.sumsqr(du)

        # Terminal tracking to final staged reference
        rT = ref_k(self.N)
        xT = X[:, self.N]
        pos_err_T = ca.vertcat(xT[0] - rT[0], xT[2] - rT[2])
        vel_err_T = ca.vertcat(xT[1] - rT[1], xT[3] - rT[3])
        obj += self.Qp * ca.sumsqr(pos_err_T) + self.Qv * ca.sumsqr(vel_err_T)

        # Stack constraints and variables
        g_constr = ca.vertcat(*g_list)
        opt_vars = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))

        # Variable bounds
        lbx = []
        ubx = []
        for _ in range(N+1):
            lbx.extend([-ca.inf]*nx)
            ubx.extend([ ca.inf]*nx)
        for _ in range(N):
            lbx.extend([self.u_bounds[0]]*self.nu)
            ubx.extend([self.u_bounds[1]]*self.nu)

        # Solver
        self.nlp = {'x': opt_vars, 'f': obj, 'g': g_constr, 'p': P}
        self.solver = ca.nlpsol('solver', 'ipopt', self.nlp,
                                {'ipopt.print_level': 0,
                                 'print_time': 0,
                                 'ipopt.sb': 'yes',
                                 'ipopt.max_iter': 200})

        self.lbx = np.array(lbx, dtype=float)
        self.ubx = np.array(ubx, dtype=float)
        self.lbg = np.array(glb, dtype=float)
        self.ubg = np.array(gub, dtype=float)
        self.w0  = np.zeros(opt_vars.shape[0])

    # New unified 7D feature
    def _phi(self, x):
        px, vx, py, vy = x[0], x[1], x[2], x[3]
        return ca.vertcat(px, vx, py, vy,
                          ca.tanh(vx / self.v_eps),
                          ca.tanh(vy / self.v_eps),
                          1.0)

    def _dyn_regressor(self, x, u, th):
        px, vx, py, vy = ca.vertsplit(x)
        alpha, beta = ca.vertsplit(u)
        thx = th[0:self.px]                 # 0:7
        thy = th[self.px:self.px+self.py]   # 7:14
        phi = self._phi(x)
        ax = self.gz * ca.sin(alpha) + ca.dot(phi, thx)
        ay = self.gz * ca.sin(beta)  + ca.dot(phi, thy)
        return ca.vertcat(vx, ax, vy, ay)

    def _rk4_step_regressor(self, x, u, th, Ts):
        k1 = self._dyn_regressor(x, u, th)
        k2 = self._dyn_regressor(x + Ts/2 * k1, u, th)
        k3 = self._dyn_regressor(x + Ts/2 * k2, u, th)
        k4 = self._dyn_regressor(x + Ts * k3, u, th)
        return x + Ts/6*(k1 + 2*k2 + 2*k3 + k4)

    def get_state(self):
        pos = self.data.body(self.target_body).xpos[:2]
        vxy = self.data.body(self.target_body).cvel[3:5]  # linear velocity components
        return np.array([pos[0], vxy[0], pos[1], vxy[1]], dtype=float)

    # Build staged reference over horizon from current virtual reference r_v and terminal target
    @staticmethod
    def build_ref_traj(x_now, r_v, target, N, nx, step_fraction=0.2):
        # Progress exponentially from r_v toward target over the horizon
        R = np.zeros(((N+1), nx), dtype=float)
        for i in range(N+1):
            w = 1.0 - (1.0 - step_fraction)**(i+1)
            r_i = r_v + w * (target - r_v)
            # Track zero velocity by default
            R[i, :] = np.array([r_i[0], 0.0, r_i[2], 0.0])
        return R.reshape(-1)

    def solve(self, x0, u_prev, theta_hat, Rref_flat):
        Pval = np.concatenate([x0, u_prev, theta_hat, Rref_flat]).astype(float)
        sol = self.solver(x0=self.w0, p=Pval, lbx=self.lbx, ubx=self.ubx,
                          lbg=self.lbg, ubg=self.ubg)
        w_opt = sol['x'].full().flatten()
        self.w0 = w_opt
        nX = self.nx*(self.N+1)
        nU = self.nu*self.N
        U_opt = w_opt[nX:nX+nU].reshape(self.N, self.nu)
        loss = sol['f'].full().flatten()
        return U_opt[0], loss


# # -------------- Main with reference governor --------------
# if __name__ == "__main__":
#     path = "models/free_tray_2dof.xml"
#     model = mujoco.MjModel.from_xml_path(path)
#     data = mujoco.MjData(model)
#     viewer = viewer.launch_passive(model, data)
#     sid_target = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "target_site")

#     pos_tol = 0.01    # 1 cm position tolerance
#     vel_tol = 0.02    # 2 cm/s velocity tolerance
#     dwell_time = 0.5  # must remain inside tolerance for 0.5 s
#     dwell_accum = 0.0
#     converged = False
#     converged_time = None
#     start_time = data.time  # MuJoCo sim time origin

#     Ts = float(model.opt.timestep)
#     ctrl = AdaptiveNPMPCSmooth(model, data, Ts=Ts, nx=4, nu=2, N=20,
#                                Qp=80.0, Qv=2.0, Ru=0.02, Rdu=1.0,
#                                u_bounds=(-0.4, 0.4), du_bounds=(-0.04, 0.04),
#                                vmax=0.2, v_eps=0.1, target_body="cube")

#     # RLS estimators for friction params (7 per axis now)
#     rls_x = RLS(p=7, theta0=np.zeros(7), P0=1e3, lam=0.995)
#     rls_y = RLS(p=7, theta0=np.zeros(7), P0=1e3, lam=0.995)
#     theta_hat = np.zeros(14)

#     # Initial states and buffers
#     xk = ctrl.get_state()
#     u_prev = np.zeros(2)
#     prev_state = xk.copy()

#     # Desired target (relative to tray center)
#     target_offset = np.array([0.0, 0.0, -0.1, 0.0])  # 10 cm x, -10 cm y
#     # Reference governor state: virtual ref r_v on positions only; keep velocity ref 0
#     tray_xy = np.array([data.body("tray").xpos[0], data.body("tray").xpos[1]])
#     r_v = np.array([tray_xy[0], 0.0, tray_xy[1], 0.0], dtype=float)

#     # RG tuning: max move per control step (meters) and a first-order filter blend
#     dr_max = 0.01  # 1 cm per step toward target
#     alpha_rg = 0.5 # blending toward capped step

#     while viewer.is_running():
#         tray_xy = np.array([data.body("tray").xpos[0], data.body("tray").xpos[1]])
#         target = np.array([tray_xy[0] + target_offset[0], 0.0,
#                            tray_xy[1] + target_offset[2], 0.0], dtype=float)
#         model.site_pos[sid_target, 0] = np.clip(target[0], -0.2*0.9, 0.2*0.9)
#         model.site_pos[sid_target, 1] = np.clip(target[2], -0.2*0.9, 0.2*0.9)

#         # RLS update using previous input/state (subtract known gravity part)
#         xk = ctrl.get_state()
#         ax_meas = (xk[1] - prev_state[1]) / Ts
#         ay_meas = (xk[3] - prev_state[3]) / Ts

#         # unified 7D feature from previous state
#         phi_full_prev = np.array([
#             prev_state[0], prev_state[1], prev_state[2], prev_state[3],
#             np.tanh(prev_state[1]/ctrl.v_eps),
#             np.tanh(prev_state[3]/ctrl.v_eps),
#             1.0
#         ], dtype=float)

#         # rls_x.update(phi_full_prev, ax_meas - ctrl.gz*np.sin(u_prev[0]))
#         # rls_y.update(phi_full_prev, ay_meas - ctrl.gz*np.sin(u_prev[1]))

#         rls_x.update(phi_full_prev, ax_meas )
#         rls_y.update(phi_full_prev, ay_meas )
#         theta_hat[:7]  = rls_x.get()
#         theta_hat[7:]  = rls_y.get()

#         # Reference governor: cap per-step move of reference toward target
#         err_pos = np.array([target[0]-r_v[0], 0.0, target[2]-r_v[2], 0.0])
#         step_pos = np.array([
#             np.clip(err_pos[0], -dr_max, dr_max),
#             0.0,
#             np.clip(err_pos[2], -dr_max, dr_max),
#             0.0
#         ])
#         r_v = r_v + alpha_rg * step_pos  # blend for extra smoothness

#         # Build staged reference trajectory over the MPC horizon from r_v toward target
#         Rref_flat = ctrl.build_ref_traj(xk, r_v, target, ctrl.N, ctrl.nx, step_fraction=0.2)

#         # gate = StickSlideGate(mu_s_hat_x=0.5, mu_s_hat_y=0.5, breakaway_margin=0.05, v_thresh=0.01,
#         #                       pos_tol=1e-3, stall_time=0.2, pulse_time=0.15, dither_amp=0.02, dither_hz=8.0,
#         #                       u_bounds=ctrl.u_bounds, du_bounds=ctrl.du_bounds, Ts=Ts)

#         # Solve MPC
#         u_cmd, loss = ctrl.solve(xk, u_prev, theta_hat, Rref_flat)

#         # u_apply, info = gate.apply(u_cmd, u_prev, xk, target, data.time)

#         # Apply tilt (axis convention: x->beta, y->-alpha)
#         quat = Rot.from_euler('xyz', [u_cmd[1], -u_cmd[0], 0.0]).as_quat()  # [x,y,z,w]
#         quat_mj = np.array([quat[3], quat[0], quat[1], quat[2]])
#         data.mocap_quat[0] = quat_mj

#         mujoco.mj_step(model, data)
#         viewer.sync()

#         # shift buffers
#         prev_state = xk.copy()
#         u_prev = u_cmd.copy()
