# DART: Learning-Enhanced Model Predictive Control for Dual-Arm Non-Prehensile Manipulation

This repository contains the **official implementation** of the paper  
**“DART: Learning-Enhanced Model Predictive Control for Dual-Arm Non-Prehensile Manipulation”**,  
submitted to **IEEE International Conference on Robotics and Automation (ICRA) 2026**.

> 📄 [Project Website](https://dart-icra.github.io/dart/)

---

## 🧩 Overview

**DART** introduces a **dual-arm control framework** for **non-prehensile manipulation**, enabling robots to move objects on a tray without grasping.  
It integrates **nonlinear Model Predictive Control (MPC)** with an **optimization-based impedance controller** to coordinate two robotic arms manipulating a shared tray.

The framework is validated in **MuJoCo simulation** for objects with varying **mass**, **geometry**, and **friction coefficients**, demonstrating accurate and generalizable control.

---

## ⚙️ Framework Summary

DART operates in two hierarchical layers:

1. **High-Level Controller (Nonlinear MPC)**  
   Computes optimal tray-tilt commands that drive the object from its current state to a target position.

2. **Low-Level Controller (Dual-Arm Impedance Controller)**  
   Converts tilt commands into joint torques for both manipulators, ensuring compliant and stable tray motion.

The MPC’s internal state-transition model is realized using three complementary strategies — each implemented as a separate method.

---

## 🔬 Implemented Methods

We propose and implement **three complementary dynamics modeling strategies**, each corresponding to a subfolder in this repository:

| Method | Description | Folder |
|--------|--------------|--------|
| **PMPC** | Physics-Based MPC using analytical tray–object dynamics | `PMPC/` |
| **RMPC** | Regression-Based MPC with online Recursive Least Squares (RLS) parameter adaptation | `RMPC/` |
| **LMPC** | Learning-Based MPC using PPO-trained dynamics model for generalization | `LMPC/` |

Each folder contains its own code, configuration files, and specific instructions to run the respective method.

---
## 📁 Repository Structure
```
DART-Dual-Arm-Non-Prehensile-Manipulation/
├── LMPC/ # Reinforcement Learning–based MPC
├── RMPC/ # Regression-Based MPC
└── PMPC/ # Physics-Based MPC
```

Each method folder includes:
- Python scripts for simulation, training, and evaluation  
- Configuration files (e.g., `config.yaml`)  
- A dedicated README with instructions to run that method  

---

## 🧰 Installation

### 1. Clone the repository
```bash
git clone git@github.com:dart-icra/DART-Dual-Arm-Non-Prehensile-Manipulation.git
cd DART-Dual-Arm-Non-Prehensile-Manipulation
```

### 2.Create and activate the environment
```
conda create --name dart python=3.10 -y
conda activate dart
```

### 3. Install dependencies

Each method folder includes its own requirements.txt file:
```
cd LMPC
pip install -r requirements.txt
cd ../RMPC
pip install -r requirements.txt
cd ../PMPC
pip install -r requirements.txt
```

## 🧠 Simulation Environment

All simulations are conducted in the MuJoCo physics engine.
Please install MuJoCo (v2.3 or newer) following the official installation guide.

The codebase uses:

- CasADi + IPOPT for solving the nonlinear MPC
- PyTorch for the learning-based dynamics model (LMPC)
- Python multiprocessing for real-time coordination of MPC, controller, and simulator processes.

## 📊 Evaluation Setup

All methods are evaluated on 18 distinct object configurations, varying by:

- Shape ∈ {cube, cylinder, sphere}
- Mass ∈ {1 kg, 2 kg}
- Friction coefficient ∈ {0.05, 0.10, 0.20}
  

Metrics used for comparison:

- Settling Time: time to reach steady state
- Steady-State Error: mean position error in final 20% of samples
- Control Effort: energy and smoothness of applied torques

## 🧩 Results Summary

| Metric                 | PMPC             | RMPC            | LMPC                            |
| :--------------------- | :--------------- | :-------------- | :------------------------------ |
| **Settling Time**      | Moderate         | High            | **Lowest**                      |
| **Steady-State Error** | Low (fine-tuned) | Moderate        | **Low (generalized)**           |
| **Control Effort**     | Low–Moderate     | **Lowest**      | Moderate–High                   |
| **Generalization**     | Object-specific  | Online adaptive | **Cross-object generalization** |

- LMPC achieves the fastest convergence and best generalization,
- RMPC provides stable online adaptation, and
- PMPC delivers interpretable, physics-based control.

## 🧾 Citation

If you use this work in your research, please cite:

---
### 💬 Contact

For any questions, please visit the project website or open an issue on this repository.


