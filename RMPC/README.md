# Overview

The **rob_ctrl.py** launches a MuJoCo dual-arm scene with a free tray and a parameterized object (sphere, cube, or cylinder), runs adaptive Recursive Least Square (RLS) based MPC + dual-arm control to move the tray to converge the object to a target position over the tray, and records a video and JSON logs keyed by the object parameters.

# Quick start

Change Directory:
```bash
cd ~/<Path to DART-Dual...-Manipulation>/RMPC
```

Sphere, radius 0.025 m, mass 2.0 kg, friction [0.2, 0.2, 0.002], target at (+0.05, +0.05) m relative to tray center:
```bash
python dev_dual/rob_ctrl.py --object sphere --radius 0.025 --mass 2.0 --mu 0.2 0.2 0.002 --tx 0.05 --ty 0.05
```

Cube, edge 0.04 m, mass 1.5 kg, friction [0.3, 0.1, 0.001], target at (-0.06, +0.02) m:

```bash
python dev_dual/rob_ctrl.py --object cube --edge 0.04 --mass 1.5 --mu 0.3 0.1 0.001 --tx -0.06 --ty 0.02
```
Cylinder, radius 0.03 m, height 0.06 m, mass 1.0 kg, friction [0.25 0.15 0.002], target at (+0.02,-0.03) m:
```bash
python dev_dual/rob_ctrl.py --object cylinder --radius 0.03 --height 0.06 --mass 1.0 --mu 0.25 0.15 0.002 --tx 0.02 --ty -0.03
```

# CLI arguments

* **Model and mode**:

    * **--model_path**: Path to world_general.xml (defaults to ./models_dual/xarm7/world_general.xml)

    * **--headless**: Disable the interactive viewer

* **Object selection**:

    * **--object**: sphere | cube | cylinder

    * **--mass**: Object mass in kg

    * **--mu**: Three friction coefficients “tangential, torsional, rolling” (e.g., 0.2 0.2 0.002)

* **Dimensions:**

    * sphere: **--radius**

    * cube: **--edge** (edge length)

    * cylinder: **--radius --height**

* **Initial object pose (world frame, optional):**

    * **--obj_pos**: Position; if omitted, object is placed above tray center with a small offset

    * **--obj_quat**: Orientation quaternion w x y z (optional)

* **Tray-relative target:**

    * **--tx --ty**: Offsets in meters relative to the tray center; the script sets tray sites accordingly
      
* **Save Output**
   * **--save**: Save the Video and JSON logs.


# Outputs

* **Video:** tray_{object}_m{mass}_mu{mu_t}-{mu_tors}-{mu_roll}_tx{tx}_ty{ty}.mp4

* **JSON logs:** {object}_m{mass}_mu{mu_t}-{mu_tors}-{mu_roll}_tx{tx}_ty{ty}.json
   * **JSON includes:**

      * pos_err, pos_err_norm, u_cmd, torque, timestep arrays

