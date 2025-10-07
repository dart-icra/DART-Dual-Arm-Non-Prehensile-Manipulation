import multiprocessing as mp
from multiprocessing import Event, shared_memory
import os
import casadi as ca
from cvxpy import pos
import mujoco
import mujoco.viewer as viewer
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.spatial.transform import Rotation as Rot
from torch.distributions import Normal
from collections import deque
import time
import torch.nn.functional as F


def gen_targ(model,data,MAX_DIST=0.1):
        x,y = np.random.uniform(-MAX_DIST, MAX_DIST, size=2)
        target = np.array([x, 0, y, 0, 0, 0, 0, 0]) + np.array([data.body("tray").xpos[0], 0 , data.body("tray").xpos[1],0, 0, 0, 0, 0])  
        return target


def gen_radial_targ(model,data,MIN_DIST=0.08,MAX_DIST=0.12):
        r = np.random.uniform(MIN_DIST, MAX_DIST)
        theta = np.random.uniform(0, 2*np.pi)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        target = np.array([x, 0, y, 0, 0, 0, 0, 0]) + np.array([data.body("tray").xpos[0], 0 , data.body("tray").xpos[1],0, 0, 0, 0, 0])  
        return target

class Policy(nn.Module):
    def __init__(self, obs_dim, act_dim, packet):
        super().__init__()
        hidden_size = packet.get("hidden_size", 64)
        hidden_layers = packet.get("hidden_layers", 2)

        actor_layers = []
        in_dim = obs_dim
        for _ in range(hidden_layers):
            actor_layers.append(nn.Linear(in_dim, hidden_size))
            actor_layers.append(nn.Tanh())
            in_dim = hidden_size
        actor_layers.append(nn.Linear(hidden_size, act_dim))
        self.mean_net = nn.Sequential(*actor_layers)

        critic_layers = []
        in_dim = obs_dim
        for _ in range(hidden_layers):
            critic_layers.append(nn.Linear(in_dim, hidden_size))
            critic_layers.append(nn.Tanh())
            in_dim = hidden_size
        critic_layers.append(nn.Linear(hidden_size, 1))
        self.value_net = nn.Sequential(*critic_layers)

        init_std = packet.get("policy_std_init", 0.1)
        self.log_std = nn.Parameter(torch.ones(act_dim) * np.log(init_std))

        self.min_log_std = np.log(packet.get("policy_std_min", 1e-2))  
        self.max_log_std = np.log(packet.get("policy_std_max", 2.0))   
        self._init_weights()

    def _init_weights(self):
        """Orthogonal init for stability"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)

    def forward(self, obs: torch.Tensor):
        """
        obs: [batch, obs_dim]
        returns: mean [batch, act_dim], std [act_dim], value [batch]
        """
        mean = self.mean_net(obs)
        log_std = torch.clamp(self.log_std, self.min_log_std, self.max_log_std)
        std = torch.exp(log_std)
        value = self.value_net(obs).squeeze(-1)
        return mean, std, value



class RolloutBuffer:
    def __init__(self, size, obs_dim, act_dim):
        self.obs = []
        self.actions = []
        self.logps = []
        self.rewards = []
        self.values = []
        self.dones = []

    def add(self, o, a, logp, r, v, done):
        self.obs.append(o)
        self.actions.append(a)
        self.logps.append(logp)
        self.rewards.append(r)
        self.values.append(v)
        self.dones.append(done)

    def clear(self):
        self.obs.clear()
        self.actions.clear()
        self.logps.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()


class RLMPC:
    def __init__(self, model, data, params):
        
        self.setup(model, data, params)
        
        self.shapes = {
            "state": (int(params["nx"]),),
            "state_next": (int(params["nx"]),),  
            "target": (int(params["nx"]),),
            "w_opt": (
                int(params["nx"] * (params["N"] + 1) + params["nu"] * params["N"]),
            ),
            "loss": (1,),
            "control": (int(params["nu"]),),
            "model_params": (int(self.params_len),),
            "state_deriv": (int(params["nx"]),),
            "in_contact": (1,),
            "RLstatus": (1,),
        }

        self.shms = {}
        self.views = {}
        for key, shape in self.shapes.items():
            shm, arr = self.create_shm_array(shape)
            self.shms[key] = shm
            self.views[key] = arr

        self.shm_names = {k: v.name for k, v in self.shms.items()}

        self.views["control"][:] = np.array([0.0, 0.0])
        self.views["model_params"][:] = np.random.uniform(0,params["max_param_abs"]/2,size=(self.params_len,))
        self.last_control = np.array([0.0, 0.0])

        self.events = {
            "state_ready": Event(),
            "ctrl_ready": Event(),
            "data_ready": Event(),
            "terminate": Event(),
            "reset": Event(),
        }

        self.solver_proc = mp.Process(
            target=RLMPC._solver_worker,
            args=(self.shm_names, self.events, self.cs_packet, self.shapes),
            daemon=True,
        )

        self.rl_proc = mp.Process(
            target=RLMPC._rl_worker,
            args=(self.shm_names, self.events, self.rl_packet, self.shapes),
            daemon=True,
        )

        self.solver_proc.start()
        self.rl_proc.start()
        
    def rebind(self, model, data):
        self.setup(model, data, self.params)
        
    def setup(self, model, data, params):
        self.model = model
        self.data = data
        self.params = params

        self.mu = params.get("mu", 0.2)
        self.body_name = params.get("body_name", "cube2")
        self.body_id = params.get("body_id", self.model.body(self.body_name).id)

        self.params_len = 34
        
        FILE_PATH = os.path.dirname(os.path.abspath(__file__))
        self.checkpoint_dir = params.get("checkpoint_dir", FILE_PATH+"/../checkpoints/unnamed")
        
        print(f"[RLMPC] Checkpoint dir: {self.checkpoint_dir}")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.cs_packet = {
            "Ts": params["Ts"],
            "nx": params["nx"],
            "nu": params["nu"],
            "N": params["N"],
            "Q": params["Q"],
            "Qt": params["Qt"],
            "R": params["R"],
            "u_bounds": params["u_bounds"],
            "mu": self.mu,
            "body_name": self.body_name,
            "body_id": self.body_id,
            "g": params["g"],
            "params_len": self.params_len,
        }

        self.rl_packet = {
            "nx": params["nx"],
            "nu": params["nu"],
            "policy_hidden": params.get("policy_hidden", [256, 256]),
            "value_hidden": params.get("value_hidden", [256]),
            "lr": params.get("lr", params.get("learning_rate", 3e-4)),
            "policy_std_init": params.get("policy_std_init", 0.1),
            "clip_eps": params.get("clip_eps", 0.2),
            "epochs": params.get("epochs", 8),
            "mini_batch_size": params.get("mini_batch_size", 64),
            "rollout_len": params.get("rollout_len", 2048),
            "gamma": params.get("gamma", 0.99),
            "gae_lambda": params.get("gae_lambda", 0.95),
            "obs_dim": params.get("obs_dim", params["nx"] * 2),
            "max_param_abs": params.get("max_param_abs", .5),
            "max_delta_abs": params.get("max_delta_abs", 0.1),
            "vf_coef": params.get("vf_coef", 0.25),  # Reduced from 0.5
            "ent_coef": params.get("ent_coef", 0.01),  # Increased from 0.0 for exploration
            "w_pos": params.get("w_pos", 1.0),
            "w_vel": params.get("w_vel", 0.1),
            "w_ctrl": params.get("w_ctrl", 1e-4),
            "max_episode_steps": params.get("max_episode_steps", 10000),
            "checkpoint_dir": self.checkpoint_dir,
            "train" : params.get("train", True),
        }

    @staticmethod
    def _solver_worker(shm_names, events, packet, shapes):
        views, shms = {}, {}
        for key, shape in shapes.items():
            shm = shared_memory.SharedMemory(name=shm_names[key])
            shms[key] = shm
            views[key] = np.ndarray(shape, dtype=np.float64, buffer=shm.buf)

        x_sym = ca.SX.sym("x", packet["nx"])
        u_sym = ca.SX.sym("u", packet["nu"])

        X = ca.SX.sym("X", packet["nx"], packet["N"] + 1)
        U = ca.SX.sym("U", packet["nu"], packet["N"])
        P = ca.SX.sym(
            "P",
            packet["nx"]
            + packet["nu"]
            + packet["params_len"]
            #! + (packet["N"] + 1) * packet["nx"],
            + packet["nx"],
        )  # initial state + reference + model parameters +

        obj = 0
        g_constr = [X[:, 0] - P[: packet["nx"]]]
        u_prev = P[packet["nx"] : packet["nx"] + packet["nu"]]
        pvec = P[
            packet["nx"] + packet["nu"] : packet["nx"]
            + packet["nu"]
            + packet["params_len"]
        ]
        traj = P[packet["nx"] + packet["nu"] + packet["params_len"] :]
        
        def safe_dynamics(x, u, pvec):
            """
            Extended dynamics including rolling + toppling/rotation.

            State x:
                [px, vx, py, vy,
                theta_x, omega_x, theta_y, omega_y]
                (theta_x: rotation about x-axis (pitch/topple around x),
                theta_y: rotation about y-axis (roll/topple around y))

            Control u:
                [a, b]  # plate tilt angles as before

            pvec indices (expanded):
                0..5   : m_x, m_y, c_x, c_y, k_x, k_y      (squashed)
                6..7   : alpha_x, alpha_y                  (tilt scaling - kept raw)
                8..17  : translational Stribeck params (unchanged indices from your code)
                        F_s_x(8), F_c_x(9), B_x(10), v_s_x(11), eps_x(12),
                        F_s_y(13), F_c_y(14), B_y(15), v_s_y(16), eps_y(17)
                18..   : new params for rotation/rolling
                        I_x (18), I_y (19),
                        r_x (20), r_y (21),
                        c_rot_x (22), c_rot_y (23),    # rotational viscous damping
                        F_s_rot_x (24), F_c_rot_x (25), B_rot_x (26), v_s_rot_x (27), eps_rot_x (28),
                        F_s_rot_y (29), F_c_rot_y (30), B_rot_y (31), v_s_rot_y (32), eps_rot_y (33),
                        h_com_x (34), h_com_y (35)      # COM height for toppling torque (can be small)
            """
            def squash_param(p_raw, p_min, p_max):
                # return p_min + 0.5 * (p_max - p_min) * (ca.tanh(p_raw) + 1)z
                return ca.fabs(p_raw) + 1e-6
            px, vx, py, vy = x[0], x[1], x[2], x[3] # translational states

            # --- unpack rotational state ---
            theta_x = x[4]   # pitch/topple about x
            omega_x = x[5]
            theta_y = x[6]   # roll/topple about y
            omega_y = x[7]

            a, b = u[0], u[1]

            # --- squash translational params (existing) ---
            m_x = squash_param(pvec[0], 0.0, 10.0)   #! >=0 
            m_y = squash_param(pvec[1], 0.0, 10.0)   #! >=0
            c_x = squash_param(pvec[2], -10.0, 10.0)
            c_y = squash_param(pvec[3], -10.0, 10.0)
            k_x = squash_param(pvec[4], 0.0, 100.0)  #! >=0
            k_y = squash_param(pvec[5], 0.0, 100.0)  #! >=0

            # translational Stribeck (indices unchanged)
            F_s_x, F_c_x, B_x = pvec[6], pvec[7], pvec[8]   #! >=0
            v_s_x = squash_param(pvec[9], 1e-4, 1.0)        #! >0
            eps_x = squash_param(pvec[10], 1e-6, 1e-2)       #! >0
            F_s_y, F_c_y, B_y = pvec[11], pvec[12], pvec[13]  #! >=0
            v_s_y = squash_param(pvec[14], 1e-4, 1.0)        #! >0
            eps_y = squash_param(pvec[15], 1e-6, 1e-2)       #! >0

            # --- rotational / rolling params (new) ---
            I_x = squash_param(pvec[16], 1e-6, 50.0)   # rotational inertia about x  #!>=0
            I_y = squash_param(pvec[17], 1e-6, 50.0)   # rotational inertia about y  #!>=0
            r_x = squash_param(pvec[18], 1e-4, 1.0)    # contact radius (for x-translation <-> omega_y) #!>0
            r_y = squash_param(pvec[19], 1e-4, 1.0)    # contact radius (for y-translation <-> omega_x) #!>0
            c_rot_x = squash_param(pvec[20], 0.0, 50.0)
            c_rot_y = squash_param(pvec[21], 0.0, 50.0)

            # rotational Stribeck params (torque-like) for roll about x and y
            F_s_rot_x, F_c_rot_x, B_rot_x = pvec[22], pvec[23], pvec[24]  #! >=0
            v_s_rot_x = squash_param(pvec[25], 1e-6, 10.0)   # Fixed index from 26 to 25
            eps_rot_x = squash_param(pvec[26], 1e-8, 1e-2)   # Fixed index from 27 to 26
            F_s_rot_y, F_c_rot_y, B_rot_y = pvec[27], pvec[28], pvec[29] # Fixed indices
            v_s_rot_y = squash_param(pvec[30], 1e-6, 10.0)   # Fixed index from 31 to 30
            eps_rot_y = squash_param(pvec[31], 1e-8, 1e-2)   # Fixed index from 32 to 31

            # COM height for toppling gravity torque (small positive)  
            h_com_x = squash_param(pvec[32], 1e-4, 0.5)  #! >0
            h_com_y = squash_param(pvec[33], 1e-4, 0.5)  #! >0

            # --- matrices (translational) ---
            M = ca.diag(ca.vertcat(m_x, m_y))
            C = ca.diag(ca.vertcat(c_x, c_y))
            K = ca.diag(ca.vertcat(k_x, k_y))

            # gravity/tilt forcing (translational)
            g = 9.81
            Gvec = ca.vertcat(
                m_x * (g * ca.sin(a)),
                m_y * (g * ca.sin(b))
            )

            qdot = ca.vertcat(vx, vy)
            q = ca.vertcat(px, py)

            # --- helper functions (as before) ---
            def smooth_sign(v, eps):
                return ca.tanh(v / eps)

            def stribeck_fric(v, F_s, F_c, B, v_s, eps):
                abs_v = ca.fabs(v)
                exp_term = ca.exp(-abs_v / (v_s + 1e-12))
                static_to_coulomb = F_c + (F_s - F_c) * exp_term
                return smooth_sign(v, eps) * static_to_coulomb + B * v

            # translational sliding friction (existing)
            Ff_x = stribeck_fric(vx, F_s_x, F_c_x, B_x, v_s_x, eps_x)
            Ff_y = stribeck_fric(vy, F_s_y, F_c_y, B_y, v_s_y, eps_y)
            F_fric = ca.vertcat(Ff_x, Ff_y)

            # --- rolling-slip: compute slip velocities between surface contact rotation and translation ---
            # Mapping convention: rotation about y-axis (omega_y) produces motion in x-direction: v_x ≈ r_x * omega_y
            #                    rotation about x-axis (omega_x) produces motion in y-direction: v_y ≈ - r_y * omega_x
            # (sign choices are conventional; adjust to match your body frames)
            v_roll_equiv_x = r_x * omega_y
            v_roll_equiv_y = - r_y * omega_x

            v_slip_x = vx - v_roll_equiv_x
            v_slip_y = vy - v_roll_equiv_y

            # frictional force due to rolling slip (treat like translational friction from slip velocity)
            # These oppose the slip translationally.
            F_roll_x = stribeck_fric(v_slip_x, F_s_x, F_c_x, B_x, v_s_x, eps_x)   # reuse translational params or define separate
            F_roll_y = stribeck_fric(v_slip_y, F_s_y, F_c_y, B_y, v_s_y, eps_y)

            F_roll = ca.vertcat(F_roll_x, F_roll_y)

            # --- rotational torques from rolling friction (contact reaction) ---
            # Torque from translational contact force = - r * F_roll (sign chosen so resistive)
            tau_slip_x = - r_y * F_roll_y    # rotation about x affected by y-direction contact force
            tau_slip_y = - r_x * F_roll_x    # rotation about y affected by x-direction contact force

            # rotational Stribeck (torque resisting angular motion)
            T_noslip_x = stribeck_fric(omega_x, F_s_rot_x, F_c_rot_x, B_rot_x, v_s_rot_x, eps_rot_x)
            T_noslip_y = stribeck_fric(omega_y, F_s_rot_y, F_c_rot_y, B_rot_y, v_s_rot_y, eps_rot_y)

            # rotational damping viscous term
            T_damp_x = c_rot_x * omega_x
            T_damp_y = c_rot_y * omega_y

            # gravity/topple restoring torque approximation: tau ≈ - m*g*h_com*sin(theta)
            tau_topple_x = - m_y * g * h_com_x * ca.sin(theta_x)   # sign/assignment heuristic: use appropriate mass coupling
            tau_topple_y = - m_x * g * h_com_y * ca.sin(theta_y)

            # total torque on each rotational DOF
            tau_x = tau_slip_x - T_noslip_x - T_damp_x + tau_topple_x
            tau_y = tau_slip_y - T_noslip_y - T_damp_y + tau_topple_y

            # rotational accelerations
            alpha_rot_x = tau_x / (I_x + 1e-12)
            alpha_rot_y = tau_y / (I_y + 1e-12)

            # --- translational equation: include rolling reaction forces in translational rhs ---
            # Note: F_roll already resists slip translationally; combine with sliding friction and other forces.
            rhs = Gvec - C @ qdot - K @ q - F_fric - F_roll

            qddot = ca.inv(M) @ rhs  # translational accelerations

            # assemble full state derivative
            # ordering: px_dot = vx, vx_dot = qddot_x, py_dot = vy, vy_dot = qddot_y,
            #           theta_x_dot = omega_x, omega_x_dot = alpha_rot_x,
            #           theta_y_dot = omega_y, omega_y_dot = alpha_rot_y
            xd = ca.vertcat(
                qdot[0],
                qddot[0],
                qdot[1],
                qddot[1],
                omega_x,
                alpha_rot_x,
                omega_y,
                alpha_rot_y
            )

            return xd

        def _rk4(x, u, pvec):
            k1 = safe_dynamics(x, u, pvec)
            k2 = safe_dynamics(x + 0.5 * packet["Ts"] * k1, u, pvec)
            k3 = safe_dynamics(x + 0.5 * packet["Ts"] * k2, u, pvec)
            k4 = safe_dynamics(x + packet["Ts"] * k3, u, pvec)
            return x + packet["Ts"] * (k1 + 2 * k2 + 2 * k3 + k4) / 6

        f = ca.Function("f", [x_sym, u_sym, pvec], [_rk4(x_sym, u_sym, pvec)])

        def _ref(k):
            ptr = k * packet["nx"]
            return traj[ptr : ptr + packet["nx"]]

        for k in range(packet["N"]):
            xk = X[:, k]
            uk = U[:, k]
            #! ref = _ref(k)
            ref  = traj
            duk = uk - U[:, k - 1] if k > 0 else uk - u_prev

            state_error = xk - ref
            ctrl_error = ca.vertcat(uk[0], uk[1], duk[0], duk[1])

            obj += ( ca.mtimes([state_error.T, ca.diag(packet["Q"]), state_error]) +
                     ca.mtimes([ctrl_error.T, ca.diag(packet["R"]), ctrl_error])
            )
            g_constr.append(X[:, k + 1] - f(xk, uk, pvec))

        #! r_term = _ref(packet["N"])
        r_term = traj
        
        x_term = X[:, packet["N"]]
        term_state_err = x_term - r_term
        obj += ca.mtimes([term_state_err.T, ca.diag(packet["Qt"]), term_state_err])

        g_constr = ca.vertcat(*g_constr)
        opt_vars = ca.vertcat(ca.reshape(X, (-1, 1)), ca.reshape(U, (-1, 1)))

        # Bounds
        lbx = []
        ubx = []
        for _ in range(packet["N"] + 1):
            lbx.extend([-ca.inf] * packet["nx"])
            ubx.extend([ca.inf] * packet["nx"])
        for _ in range(packet["N"]):
            lbx.extend([packet["u_bounds"][0], packet["u_bounds"][0]])
            ubx.extend([packet["u_bounds"][1], packet["u_bounds"][1]])

        nlp = {"x": opt_vars, "f": obj, "g": g_constr, "p": P}
        opts = {
            "ipopt.print_level": 0,
            "print_time": 0,
            "ipopt.sb": "yes",
            "ipopt.max_iter": 50,
            "ipopt.max_cpu_time": 0.05,
            "ipopt.tol": 1e-4,
            "ipopt.acceptable_tol": 1e-3,
            "ipopt.acceptable_iter": 5,
        }

        solver = ca.nlpsol("solver", "ipopt", nlp, opts)
        w0 = np.zeros(opt_vars.shape[0])

        try:
            while True:
                events["state_ready"].wait(timeout=0.01)
                if events["terminate"].is_set():
                    break
                events["state_ready"].clear()
                state = views["state"].copy()
                target = views["target"].copy()
                control = views["control"].copy()
                pvec = views["model_params"].copy()
                # traj = RLMPC.gen_Trajectory(
                #     state, target, packet["N"], packet["nx"], packet["Ts"]
                # )

                sol = solver(
                    x0=w0,
                    p=np.concatenate([state, control, pvec, target]),
                    lbg=0,
                    ubg=0,
                    lbx=lbx,
                    ubx=ubx,
                )

                w_opt = sol["x"].full().flatten()
                loss = sol["f"].full().flatten()
                w0 = w_opt

                views["w_opt"][:] = w_opt
                views["loss"][:] = loss

                events["ctrl_ready"].set()
        except KeyboardInterrupt:
            pass
        finally:
            for shm in shms.values():
                try:
                    shm.close()
                except:
                    pass
            print("[solver] exiting (warm-start worker)")


    @staticmethod
    def _rl_worker(shm_names, events, packet, shapes):

        views, shms = {}, {}
        for key, shape in shapes.items():
            shm = shared_memory.SharedMemory(name=shm_names[key])
            shms[key] = shm
            views[key] = np.ndarray(shape, dtype=np.float64, buffer=shm.buf)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        history_len = packet.get("history_len", 10)
        base_obs_dim = int(np.prod(shapes["state"])) + int(np.prod(shapes["target"])) + int(np.prod(shapes["control"]))
        act_dim = int(np.prod(shapes["model_params"]))
        base_obs_dim += act_dim
        obs_dim = history_len * base_obs_dim

        # --- Welford online normalization vars (use float64 for stability) ---
        obs_mean = np.zeros(base_obs_dim, dtype=np.float64)
        obs_M2 = np.zeros(base_obs_dim, dtype=np.float64)
        obs_count = 0

        padding = np.zeros(base_obs_dim, dtype=np.float32)
        history = deque([padding.copy() for _ in range(history_len)], maxlen=history_len)

        policy = Policy(obs_dim, act_dim, packet).to(device)
        optimizer = optim.Adam(policy.parameters(), lr=packet["lr"], weight_decay=packet.get("weight_decay", 1e-5))

        training = packet.get("train", True)
        prev_cmd = views["control"].copy()

        if training:
            policy.train()
        else:
            policy.eval()
            if packet.get("checkpoint_dir", None) is not None:
                checkpoint_path = os.path.join(packet["checkpoint_dir"], "best_agent.pth")
                if os.path.exists(checkpoint_path):
                    policy.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=False)["model"])
                    print(f"[RL WORKER] Loaded policy from {checkpoint_path}")
                else:
                    print(f"[RL WORKER] No checkpoint found at {checkpoint_path}, starting fresh.")
                    training = True
                    policy.train()

        buf = RolloutBuffer(packet["rollout_len"], obs_dim, act_dim)
        glob_buf = RolloutBuffer(packet["rollout_len"], obs_dim, act_dim)
        max_delta = packet.get("max_delta_abs", 0.1)
        k_max = packet.get("max_param_abs", 0.4)

        # New safety & init hyperparams (exposed via packet)
        min_k = packet.get("min_k", 1e-2)                # enforced lower bound (practical > 0)
        init_frac = packet.get("init_k_frac", 0.5)       # initial fraction of k_max
        init_jitter = packet.get("init_k_jitter", 0.05)  # ± jitter relative to k_max
        k_ceiling_margin = packet.get("k_ceiling_margin", max(1e-3, 0.05 * k_max))  # leave margin under k_max
        action_scale = packet.get("action_scale", 1.0)   # scale on raw action -> delta_z

        def compute_gae(rewards, values, dones, last_value, gamma, lam):
            adv, gae = [], 0.0
            values = values + [last_value]
            for step in reversed(range(len(rewards))):
                delta = rewards[step] + gamma * values[step + 1] * (1.0 - dones[step]) - values[step]
                gae = delta + gamma * lam * (1.0 - dones[step]) * gae
                adv.insert(0, gae)
            return adv

        def prox_reward(pos_err, vel_err, sigma_pos=0.05, sigma_vel=0.05, w_pos=40, w_vel=20):
            pos_term = np.exp(-(pos_err**2) / (2 * sigma_pos**2))
            vel_term = np.exp(-(vel_err**2) / (2 * sigma_vel**2))
            return w_pos * pos_term + w_vel * pos_term * vel_term

        def write_params_to_shm(k_numpy):
            k_numpy = np.asarray(k_numpy, dtype=np.float64)
            prev = views["model_params"].copy()
            alpha = packet.get("shm_smooth_alpha", 0.5)
            smoothed = alpha * k_numpy + (1 - alpha) * prev
            def smooth_clip(x, min_v, max_v, margin=1e-3):
                center = (max_v + min_v) / 2
                scale = (max_v - min_v) / 2 - margin
                return center + scale * np.tanh((x - center) / scale)
            clipped = smooth_clip(smoothed, min_k, k_max - k_ceiling_margin)
            views["model_params"][:] = clipped

        # --- Initialize current_k at mid-range with small jitter (avoid zero init) ---
        rng = np.random.default_rng(seed=packet.get("seed", None))
        base_init = init_frac * k_max
        jitter = rng.uniform(-init_jitter, init_jitter, size=act_dim) * k_max
        current_k = np.clip(np.full(act_dim, base_init, dtype=np.float64) + jitter, min_k, k_max - k_ceiling_margin)
        write_params_to_shm(current_k)

        timestep = 0
        rollout_len = packet["rollout_len"]
        gamma, lam = packet["gamma"], packet["gae_lambda"]
        time_penalty = 0.0
        episode_step, episode_return, moving_return, episode_count = 0, 0.0, 0.0, 0
        max_episode_steps = packet.get("max_episode_steps", 1000)
        best_return = -float("inf")
        diag_interval = packet.get("diag_interval", 500)

        while True:
            if events["terminate"].is_set():
                break
            events["state_ready"].wait(timeout=0.01)
            if events["terminate"].is_set():
                break

            state = views["state"].copy()
            target = views["target"].copy()
            control = views["control"].copy()

            base_vec = np.concatenate([
                state.astype(np.float32),
                target.astype(np.float32),
                control.astype(np.float32),
                current_k.astype(np.float32),
            ]).astype(np.float64)

            # --- Welford update (numerically stable) ---
            obs_count += 1
            delta = base_vec - obs_mean
            obs_mean += delta / obs_count
            delta2 = base_vec - obs_mean
            obs_M2 += delta * delta2
            if obs_count > 1:
                obs_var = obs_M2 / (obs_count - 1)
            else:
                obs_var = np.ones_like(obs_M2) * 1e-6
            obs_std = np.sqrt(np.maximum(obs_var, 1e-12)).astype(np.float32)

            # Normalize observation
            normalized_vec = ((base_vec.astype(np.float32) - obs_mean.astype(np.float32)) / (obs_std + 1e-8)).astype(np.float32)
            history.append(normalized_vec)
            obs = np.concatenate(list(history)).astype(np.float32)
            obs_t = torch.from_numpy(obs).to(device).unsqueeze(0)

            with torch.no_grad():
                mean, std, value = policy(obs_t)
                # Ensure std is strictly positive and stable
                if (std <= 0).any().item():
                    std = F.softplus(std) + 1e-6
                else:
                    std = torch.clamp(std, min=1e-6)
                dist = Normal(mean, std)
                raw_action = dist.rsample()  # latent action in z-space
                # scale action delta
                delta_z = raw_action * (max_delta * action_scale)
                # compute numeric diagnostics
                delta_z_np = delta_z.detach().cpu().numpy().reshape(-1)
                delta_z_norm = np.linalg.norm(delta_z_np)
                per_dim_rms = delta_z_norm / np.sqrt(act_dim)

                # diag print
                if timestep % diag_interval == 0:
                    print(f"[Diag] delta_z_norm={delta_z_norm:.6f}, per_dim_rms={per_dim_rms:.6f}, act_dim={act_dim}")

                # auto-damp
                MAX_PER_DIM_RMS = packet.get("max_per_dim_rms", 0.5)
                if per_dim_rms > MAX_PER_DIM_RMS:
                    damp = MAX_PER_DIM_RMS / (per_dim_rms + 1e-12)
                    delta_z = delta_z * float(damp)
                    if timestep % diag_interval == 0:
                        print(f"[Diag] damped delta_z by factor {damp:.4f} to enforce max_per_dim_rms={MAX_PER_DIM_RMS}")

                
                logp = dist.log_prob(raw_action).sum(dim=-1)
                action_np = raw_action.cpu().numpy().squeeze(0).astype(np.float32)
                value_np = value.cpu().numpy().squeeze(0)

            pos = np.array([state[0], state[2]])
            vel = np.array([state[1], state[3]])
            tpos = np.array([target[0], target[2]])
            tvel = np.array([0, 0])
            pos_err = np.linalg.norm(np.abs(tpos - pos))
            vel_err = np.linalg.norm(np.abs(tvel - vel))
            change_cost = np.linalg.norm(delta_z.cpu().numpy())
            ctrl_rate_penalty = np.sum(np.abs(control - prev_cmd))
            prev_cmd = control
            proximity_reward = prox_reward(pos_err, vel_err, sigma_pos=0.02, sigma_vel=0.02, w_pos=packet.get('w_pos', 60), w_vel=packet.get('w_vel', 30))
            change_penalty = packet.get("w_change", 1e-3) * change_cost
            control_rate_penalty = packet.get("w_d_ctrl", 5.0) * ctrl_rate_penalty

            # More balanced reward composition
            reward = proximity_reward - change_penalty - control_rate_penalty - time_penalty

            # Success bonus (less extreme)
            if pos_err < 0.01 and vel_err < 0.01:
                reward += 20.0

            done = False
            episode_step += 1

            tray_limit = packet.get("tray_limit", [0.2, 0.15])
            if abs(state[0]) > tray_limit[0] or abs(state[2]) > tray_limit[1]:
                reward -= 20.0
                done = True
                status = "out_of_bounds"
            else:
                status = ""

            # Reduced contact penalty
            if views["in_contact"].copy() == 0.0:
                reward -= 10.0

            if episode_step >= max_episode_steps:
                done = True
                status = "max_steps_reached"

            if timestep % 8 == 0:  # increased frequency
                raw_action_tensor = torch.from_numpy(action_np).to(device).unsqueeze(0)
                buf.add(obs, action_np, float(logp.item()), float(value_np.item()) if np.ndim(value_np) == 0 else float(value_np[0]), reward, float(done))

                cur_k_shm = views["model_params"].copy()
                cur_k_t = torch.tensor(cur_k_shm, dtype=raw_action_tensor.dtype, device=device)

                # clamp fractions using min_k/k_max not a tiny 1e-6
                min_frac = float(min_k / k_max)
                cur_k_frac = torch.clamp(cur_k_t / k_max, min_frac, 1.0 - 1e-6)
                z_prev = torch.logit(cur_k_frac)

                # update in logit-space (existing approach) but with scaled action
                z_new = z_prev + (raw_action_tensor * max_delta * action_scale).squeeze(0)

                k_new_t = k_max * torch.sigmoid(z_new)
                k_new = k_new_t.cpu().numpy()
                write_params_to_shm(k_new)

                # diagnostics for action magnitude / param movement
                if timestep % diag_interval == 0:
                    cur = views["model_params"].copy()
                    frac = cur / k_max
                    print(f"[Diagnostics t={timestep}] k min/max: {cur.min():.6f}/{cur.max():.6f}, frac min/max: {frac.min():.6e}/{frac.max():.6e}")
                    with torch.no_grad():
                        # mean/std might be tensors
                        print(f"[Diagnostics] action mean range: {mean.min().cpu().item():.6f}/{mean.max().cpu().item():.6f}, std mean: {std.mean().cpu().item():.6f}")
                        print(f"[Diagnostics] delta_z norm: {delta_z.norm().cpu().item():.6f}")

            timestep += 1
            time_penalty += 1e-4
            episode_return += reward

            if len(buf.rewards) >= rollout_len:
                obs_last = torch.from_numpy(obs.astype(np.float32)).to(device).unsqueeze(0)
                with torch.no_grad():
                    _, _, last_val = policy(obs_last)
                last_val = last_val.cpu().item()

                advantages = compute_gae(buf.rewards, buf.values, buf.dones, last_val, gamma, lam)
                returns = np.array(advantages) + np.array(buf.values)
                returns = (returns - returns.mean()) / (returns.std() + 1e-8)

                obs_b = torch.tensor(np.array(buf.obs), dtype=torch.float32, device=device)
                acts_b = torch.tensor(np.array(buf.actions), dtype=torch.float32, device=device)
                old_logps_b = torch.tensor(np.array(buf.logps), dtype=torch.float32, device=device)
                adv_b = torch.tensor(np.array(advantages), dtype=torch.float32, device=device)
                ret_b = torch.tensor(returns, dtype=torch.float32, device=device)
                adv_b = (adv_b - adv_b.mean()) / (adv_b.std() + 1e-8)

                batch_size = packet["mini_batch_size"]
                num_samples = obs_b.shape[0]
                for _e in range(packet["epochs"]):
                    idxs = np.random.permutation(num_samples)
                    for start in range(0, num_samples, batch_size):
                        mb_idx = idxs[start:start + batch_size]
                        o_mb, a_mb = obs_b[mb_idx], acts_b[mb_idx]
                        old_logp_mb, adv_mb, ret_mb = old_logps_b[mb_idx], adv_b[mb_idx], ret_b[mb_idx]
                        mean_mb, std_mb, val_mb = policy(o_mb)
                        # ensure std positive
                        if (std_mb <= 0).any().item():
                            std_mb = F.softplus(std_mb) + 1e-6
                        else:
                            std_mb = torch.clamp(std_mb, min=1e-6)
                        dist = Normal(mean_mb, std_mb)
                        raw_action_mb = a_mb
                        logp_mb = dist.log_prob(raw_action_mb).sum(axis=-1)
                        ratio = torch.exp(logp_mb - old_logp_mb)
                        surr1 = ratio * adv_mb
                        surr2 = torch.clamp(ratio, 1.0 - packet["clip_eps"], 1.0 + packet["clip_eps"]) * adv_mb
                        policy_loss = -torch.min(surr1, surr2).mean()
                        value_loss = nn.MSELoss()(val_mb, ret_mb)
                        entropy = dist.entropy().sum(axis=-1).mean()
                        loss = policy_loss + packet.get("vf_coef", 0.25) * value_loss - packet.get("ent_coef", 0.01) * entropy
                        optimizer.zero_grad()
                        loss.backward()
                        nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
                        optimizer.step()

                print(f"[LOCAL PPO UPDATE] Episode {episode_count}, Cumulative Reward: {episode_return:.3f}, Cartesian Error: {pos_err:.4f}")

                sample_size = max(1, len(buf.obs) // 4)
                if len(buf.obs) >= sample_size:
                    idxs = np.random.choice(len(buf.obs), sample_size, replace=False)
                    for i in idxs:
                        glob_buf.add(buf.obs[i], buf.actions[i], buf.logps[i], buf.rewards[i], buf.values[i], buf.dones[i])

                if len(glob_buf.rewards) >= rollout_len:
                    glob_obs_last = torch.from_numpy(glob_buf.obs[-1].astype(np.float32)).to(device).unsqueeze(0)
                    with torch.no_grad():
                        _, _, glob_last_val = policy(glob_obs_last)
                    glob_last_val = glob_last_val.cpu().item()
                    glob_advantages = compute_gae(glob_buf.rewards, glob_buf.values, glob_buf.dones, glob_last_val, gamma, lam)
                    glob_returns = np.array(glob_advantages) + np.array(glob_buf.values)
                    glob_returns = (glob_returns - glob_returns.mean()) / (glob_returns.std() + 1e-8)

                    glob_obs_b = torch.tensor(np.array(glob_buf.obs), dtype=torch.float32, device=device)
                    glob_acts_b = torch.tensor(np.array(glob_buf.actions), dtype=torch.float32, device=device)
                    glob_old_logps_b = torch.tensor(np.array(glob_buf.logps), dtype=torch.float32, device=device)
                    glob_adv_b = torch.tensor(np.array(glob_advantages), dtype=torch.float32, device=device)
                    glob_ret_b = torch.tensor(glob_returns, dtype=torch.float32, device=device)
                    glob_adv_b = (glob_adv_b - glob_adv_b.mean()) / (glob_adv_b.std() + 1e-8)

                    glob_batch_size = packet["mini_batch_size"]
                    glob_num_samples = glob_obs_b.shape[0]
                    for _e in range(packet["epochs"]):
                        glob_idxs = np.random.permutation(glob_num_samples)
                        for glob_start in range(0, glob_num_samples, glob_batch_size):
                            glob_mb_idx = glob_idxs[glob_start:glob_start + glob_batch_size]
                            glob_o_mb, glob_a_mb = glob_obs_b[glob_mb_idx], glob_acts_b[glob_mb_idx]
                            glob_old_logp_mb, glob_adv_mb, glob_ret_mb = glob_old_logps_b[glob_mb_idx], glob_adv_b[glob_mb_idx], glob_ret_b[glob_mb_idx]
                            glob_mean_mb, glob_std_mb, glob_val_mb = policy(glob_o_mb)
                            if (glob_std_mb <= 0).any().item():
                                glob_std_mb = F.softplus(glob_std_mb) + 1e-6
                            else:
                                glob_std_mb = torch.clamp(glob_std_mb, min=1e-6)
                            glob_dist = Normal(glob_mean_mb, glob_std_mb)
                            glob_raw_action_mb = glob_a_mb
                            glob_logp_mb = glob_dist.log_prob(glob_raw_action_mb).sum(axis=-1)
                            glob_ratio = torch.exp(glob_logp_mb - glob_old_logp_mb)
                            glob_surr1 = glob_ratio * glob_adv_mb
                            glob_surr2 = torch.clamp(glob_ratio, 1.0 - packet["clip_eps"], 1.0 + packet["clip_eps"]) * glob_adv_mb
                            glob_policy_loss = -torch.min(glob_surr1, glob_surr2).mean()
                            glob_value_loss = nn.MSELoss()(glob_val_mb, glob_ret_mb)
                            glob_entropy = glob_dist.entropy().sum(axis=-1).mean()
                            glob_loss = glob_policy_loss + packet.get("vf_coef", 0.25) * glob_value_loss - packet.get("ent_coef", 0.01) * glob_entropy
                            optimizer.zero_grad()
                            glob_loss.backward()
                            nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
                            optimizer.step()

                    print(f"[GLOBAL PPO UPDATE] Episode {episode_count}, Cumulative Reward: {episode_return:.3f}, Cartesian Error: {pos_err:.4f}")
                    glob_buf.clear()

                # final mean-based update to SHM (periodic)
                with torch.no_grad():
                    cur_k_shm = views["model_params"].copy()
                    obs_for_mean = np.concatenate(list(history)).astype(np.float32)
                    obs_for_mean_t = torch.from_numpy(obs_for_mean).to(device).unsqueeze(0)
                    mean_final, std_final, _ = policy(obs_for_mean_t)
                    # enforce std positive
                    if (std_final <= 0).any().item():
                        std_final = F.softplus(std_final) + 1e-6
                    else:
                        std_final = torch.clamp(std_final, min=1e-6)
                    raw_action_final = mean_final
                    cur_k_t = torch.tensor(cur_k_shm, dtype=raw_action_final.dtype, device=device)
                    min_frac = float(min_k / k_max)
                    cur_k_frac = torch.clamp(cur_k_t / k_max, min_frac, 1.0 - 1e-6)
                    z_prev = torch.logit(cur_k_frac)
                    z_new = z_prev + (raw_action_final * max_delta * action_scale).squeeze(0)
                    k_final_t = k_max * torch.sigmoid(z_new)
                    k_final = k_final_t.cpu().numpy()
                    write_params_to_shm(k_final)
                    current_k = views["model_params"].copy()

                buf.clear()

            if done:
                if training:
                    episode_count += 1 if status != "out_of_bounds" else 0
                    moving_return = 0.9 * moving_return + 0.1 * episode_return
                    avg_per_step = episode_return / max(1, episode_step)
                    print(f"[Episode End] Return={episode_return:.3f}, Steps={episode_step}, AvgPerStep={avg_per_step:.4f}, MovingReturn={moving_return:.3f}, Episodes={episode_count}")
                    print(f"[Episode End] Status={status} num_steps={episode_step} pos_err={pos_err:.4f} vel_err={vel_err:.4f}")

                    # Log some training diagnostics
                    if episode_count % 10 == 0:
                        current_params = views["model_params"].copy()
                        print(f"[Training Diagnostics] Current params range: [{current_params.min():.4f}, {current_params.max():.4f}]")
                        # show a few obs_mean/std stats
                        print(f"[Training Diagnostics] Obs mean range: [{obs_mean.min():.4f}, {obs_mean.max():.4f}]")
                        approx_std = np.sqrt(np.maximum(obs_M2 / np.maximum(obs_count - 1, 1), 1e-12))
                        print(f"[Training Diagnostics] Obs std range: [{approx_std.min():.4f}, {approx_std.max():.4f}]")

                    assert packet.get("checkpoint_dir", None) is not None, "checkpoint_dir must be specified for training"
                    if episode_return > best_return:
                        best_return = episode_return
                        print(f"[BEST MODEL SAVED] New best reward: {episode_return:.3f} (previous best: {best_return:.3f}) - Episode {episode_count}")
                        torch.save({"model": policy.state_dict(), "optimizer": optimizer.state_dict(), "episode": episode_count, "return": episode_return, "episode number": episode_count}, os.path.join(packet["checkpoint_dir"], "best_agent.pth"))
                    torch.save({"model": policy.state_dict(), "optimizer": optimizer.state_dict(), "episode": episode_count, "return": episode_return, "episode number": episode_count}, os.path.join(packet["checkpoint_dir"], "latest_agent.pth"))
                    events["reset"].set()
                    episode_return, episode_step, time_penalty = 0.0, 0, 0.0
                    while events["reset"].is_set():
                        time.sleep(0.002)
                else:
                    episode_count += 1 if status != "out_of_bounds" else 0
                    if status != "out_of_bounds":
                        views["RLstatus"][:] = 2.0
                    moving_return = 0.9 * moving_return + 0.1 * episode_return
                    avg_per_step = episode_return / max(1, episode_step)
                    print(f"[Eval Episode End] Return={episode_return:.3f}, Steps={episode_step}, Status={status}")
                    events["reset"].set()
                    episode_return, episode_step, time_penalty = 0.0, 0, 0.0
                    while events["reset"].is_set():
                        time.sleep(0.002)

        for shm in shms.values():
            try:
                shm.close()
            except:
                pass



    @staticmethod
    def gen_Trajectory(state, target, N, nx, Ts):
        R = np.zeros((N + 1, nx), dtype=float)

        p0 = np.array([state[0], state[2]])
        v0 = np.array([state[1], state[3]])
        a0 = np.array([0.0, 0.0])
        pf = np.array([target[0], target[2]])
        vf = np.array([0.0, 0.0])
        af = np.array([0.0, 0.0])

        T = N * Ts
        t_mat = np.array(
            [
                [0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 2, 0, 0],
                [T**5, T**4, T**3, T**2, T, 1],
                [5 * T**4, 4 * T**3, 3 * T**2, 2 * T, 1, 0],
                [20 * T**3, 12 * T**2, 6 * T, 2, 0, 0],
            ],
            dtype=float,
        )

        R_coeffs = []
        for i in range(2):
            b = np.array([p0[i], v0[i], a0[i], pf[i], vf[i], af[i]], dtype=float)
            coeffs = np.linalg.solve(t_mat, b)
            R_coeffs.append(coeffs)

        for k in range(N + 1):
            t = k * Ts
            R[k, 0] = np.polyval(R_coeffs[0], t)
            R[k, 2] = np.polyval(R_coeffs[1], t)
            # R[k, 1] = np.polyval(np.polyder(R_coeffs[0]), t)
            # R[k, 3] = np.polyval(np.polyder(R_coeffs[1]), t)

        return R.reshape(-1)

    def solve(self, target):
        state = self.get_state()
        self.views["state"][:] = state
        self.views["target"][:] = target

        # push new state to solver without blocking
        self.events["state_ready"].set()

        # non-blocking check for fresh solution
        if self.events["ctrl_ready"].is_set():
            self.events["ctrl_ready"].clear()
            w_opt = self.views["w_opt"].copy()
            self.loss = self.views["loss"].copy()

            X_opt = w_opt[: self.params["nx"] * (self.params["N"] + 1)].reshape(
                self.params["N"] + 1, self.params["nx"]
            )
            U_opt = w_opt[self.params["nx"] * (self.params["N"] + 1) :].reshape(
                self.params["N"], self.params["nu"]
            )

            # store full trajectory for shift reuse
            self.X_plan = X_opt
            self.U_plan = U_opt
            if U_opt.size > 0:
                self.last_control = U_opt[0].astype(np.float64)

        else:
            # no new solution → shift old plan if possible
            if hasattr(self, "U_plan") and self.U_plan.shape[0] > 1:
                self.last_control = self.U_plan[1]
                self.U_plan = self.U_plan[1:]
            # else: hold last_control

        self.views["control"][:] = self.last_control
        return self.last_control.copy(), getattr(self, "loss", 0.0)



    def get_state_derivative(self):
        vel = self.data.body(self.params["body_name"]).cvel[3:5]
        acc = self.data.body(self.params["body_name"]).cacc[3:5]
        omega = self.data.body(self.params["body_name"]).cvel[:2]
        alpha = self.data.body(self.params["body_name"]).cacc[:2]

        state_deriv = np.array([vel[0], acc[0], vel[1], acc[1], omega[0], alpha[0], omega[1], alpha[1]])
        return state_deriv

    def get_state(self):
        pos = self.data.body(self.params["body_name"]).xpos[:2]
        vel = self.data.body(self.params["body_name"]).cvel[3:5]
        
        rmat = self.data.body(self.params["body_name"]).xmat
        theta = Rot.from_matrix(rmat.reshape(3, 3)).as_euler("xyz", degrees=False)[:2] # only roll and pitch
        omega = self.data.body(self.params["body_name"]).cvel[:2]

        return np.array([pos[0], vel[0], pos[1], vel[1], theta[0], omega[0], theta[1], omega[1]])
    
    def check_contact(self, model, data):
        num_contacts = data.ncon
        for i in range(num_contacts):
            con = data.contact[i]
            if (
                model.geom(con.geom1).name == "cube_geom"
                or model.geom(con.geom2).name == "cube_geom"
            ):
                return 1.0
        return 0.0
    
    def measure(self):
        state = self.get_state()
        state_deriv = self.get_state_derivative()

        contact = self.check_contact(self.model, self.data)

        self.views["state"][:] = state
        self.views["state_deriv"][:] = state_deriv
        self.views["in_contact"][:] = contact

        return {"state": state, "state_deriv": state_deriv, "contact": contact}

    def create_shm_array(self, shape, dtype=np.float64, name_prefix="shm"):
        """Create SharedMemory and return (SharedMemory, numpy.ndarray)"""
        dtype = np.dtype(dtype)
        size = int(np.prod(shape)) * dtype.itemsize
        shm = shared_memory.SharedMemory(create=True, size=size)
        arr = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
        arr[:] = 0
        return shm, arr

    def close(self):
        """
        Clean up resources and terminate the solver process.
        """
        self.events["terminate"].set()
        self.events["state_ready"].set()
        self.solver_proc.join(timeout=2.0)
        self.rl_proc.join(timeout=2.0)

        for shm in self.shms.values():
            try:
                shm.close()
                shm.unlink()
            except Exception:
                pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


if __name__ == "__main__":
    path = "free.xml"
    model = mujoco.MjModel.from_xml_path(path)
    data = mujoco.MjData(model)
    viewer = mujoco.viewer.launch_passive(model, data)

    # Define controller parameters
    params = {
        "Ts": model.opt.timestep,
        "nx": 8,
        "nu": 2,
        "N": 20,
        "Q": [200.0, 2.0, 200.0, 2.0 , 0.0 , 0.0, 0.0, 0.0],
        "Qt": [200.0, 2.0, 200.0, 2.0 , 0.0 , 0.0, 0.0, 0.0],
        "R": [0.1,0.1,1.0,1.0],
        "u_bounds": (-0.4, 0.4),
        "body_name": "cube2",
        "body_id": None,  # Will be set automatically if None
        "g": 9.81,
        "policy_hidden": [128, 128],
        "value_hidden": [128],
        "policy_std_init": 0.1,
        "clip_eps": 0.2,
        "epochs": 16,
        "rollout_len": 2048,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "obs_dim": 8,
        "max_param_abs": .4,
        "max_delta_abs": .1,
        "vf_coef": 0.5,
        "ent_coef": 0.1,
        "w_pos": 1.0,
        "w_vel": 0.1,
        "w_ctrl": 0.001,
        "max_episode_steps": 5000,
    }

    ctlr = RLMPC(model, data, params)
    

    target = np.array([0.0, 0.06, 0.0, 0.06, 0.0, 0.0, 0.0, 0.0])
    data.mocap_pos[model.body("targ").mocapid[0]][:2] = [target[0], target[2]]
    
    try:
        while viewer.is_running():
            
            if ctlr.events["reset"].is_set():
                mujoco.mj_resetData(model, data)
                mujoco.mj_resetDataKeyframe(model, data, 0)
                target = gen_targ(model,data)
                print(f"[RLMPC] New target: {target}")
                data.mocap_pos[model.body("targ").mocapid[0]][:2] = [target[0], target[2]]
                mujoco.mj_forward(model, data)  
                ctlr.events["reset"].clear()

            u_cmd, loss = ctlr.solve(target)
            u_cmd *= -1
            # print(f"Command: {u_cmd}, Loss: {loss}")

            quat = Rot.from_euler("xyz", [u_cmd[1], -u_cmd[0], 0.0]).as_quat(
                scalar_first=True
            )
            data.mocap_quat[0] = quat

            mujoco.mj_step(model, data)

            sys_info = ctlr.measure()
        
            state = sys_info["state"]

            error = np.linalg.norm(state[[0,2]] - target[[0,2]])

            viewer.sync()

            # print(f"Current Position: {np.round(current_pos, 4)}, Target: {np.round(target, 4)}, Error: {error:.4f}")

    finally:
        ctlr.close()
        viewer.close()

