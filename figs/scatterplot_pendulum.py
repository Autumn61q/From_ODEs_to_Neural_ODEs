#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Use a 6th-order symplectic (SV6) integrator to generate data for
H(q,p) = 1/2 p^2 + cos(q), and plot scatter points in phase space.

- If your project module `learner.integrator.hamiltonian.SV` is available,
  it will be used with order=6.
- Otherwise, a built-in Yoshida 6th-order composition of velocity-Verlet
  is used as a fallback.

Output:
  - PNG figure: sv6_pendulum_scatter.png
  - NPZ dataset: sv6_pendulum_dataset.npz  (pairs (x, y) with one-step map)
"""

import numpy as np
import matplotlib.pyplot as plt


plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'axes.facecolor': 'white',
    'figure.facecolor': 'white',
    'font.size': 14
})

# -----------------------------
# Problem setup: H = 1/2 p^2 + cos(q)
# dq/dt =  p
# dp/dt =  sin(q)
# -----------------------------
def dq_dt(q, p):
    return p

def dp_dt(q, p):
    return np.sin(q)


# ========== Try to use your project's SV(order=6) ==========
def try_project_sv6():
    """
    Try to import and build your project's SV6 integrator.
    Expected signature (from your snippet):
      SV(None, lambda p,q: (p, np.sin(q)), iterations=1, order=6, N=100)
    Returns:
      flow_func(x0: (2,), h: float, num: int) -> ndarray of shape (num+1, 2)
    """
    try:
        from learner.integrator.hamiltonian import SV
        import torch

        def flow_project(x0, h, num):
            """Use your SV(order=6) to integrate one trajectory."""
            # your SV takes a callable returning (T'(p), -V'(q)) which here is (p, sin(q))
            true_solver = SV(None, lambda p, q: (p, np.sin(q)),
                             iterations=1, order=6, N=100)
            X = true_solver.flow(torch.tensor(x0, dtype=torch.double), h, num)
            return X.detach().cpu().numpy()

        return flow_project
    except Exception as e:
        print("[Info] Could not import your SV integrator, using built-in SV6 instead.")
        print("       Reason:", repr(e))
        return None


# ========== Built-in SV6 (Yoshida 6th-order via composition) ==========
def make_builtin_sv6():
    """
    Build a 6th-order symplectic integrator by composing velocity-Verlet (second-order).
    We use a standard symmetric 6th-order set of coefficients (sum to 1):

    c = [ 0.392256805238780,  0.510043411918458, -0.471053385409757,
          0.068753168252520,  0.068753168252520, -0.471053385409757,
          0.510043411918458,  0.392256805238780 ]

    Each stage applies one velocity-Verlet step with timestep (c_i * h).

    Reference: widely-used Yoshida-type 6th-order symmetric composition
    (values chosen so that sum(c_i) ≈ 1 and odd error terms cancel).
    """

    coeffs = np.array([
        0.392256805238780,
        0.510043411918458,
       -0.471053385409757,
        0.068753168252520,
        0.068753168252520,
       -0.471053385409757,
        0.510043411918458,
        0.392256805238780,
    ], dtype=float)

    def vv_step(q, p, h):
        """One velocity-Verlet step for separable H = T(p) + V(q)."""
        # Kick (half)
        p = p + 0.5 * h * dp_dt(q, p)   # here dp/dt = sin(q)
        # Drift
        q = q + h * dq_dt(q, p)         # dq/dt = p
        # Kick (half)
        p = p + 0.5 * h * dp_dt(q, p)
        return q, p

    def flow_builtin(x0, h, num):
        """Integrate trajectory using SV6 composition."""
        q, p = float(x0[0]), float(x0[1])
        traj = np.empty((num + 1, 2), dtype=float)
        traj[0] = [q, p]
        for n in range(1, num + 1):
            # One global step of size h made of composed VV substeps
            qq, pp = q, p
            for c in coeffs:
                qq, pp = vv_step(qq, pp, c * h)
            q, p = qq, pp
            traj[n] = [q, p]
        return traj

    return flow_builtin


# ==================== Main: generate data & plot ====================
def main():
    # Integration parameters
    h   = 0.05      # step size
    num = 600       # steps per trajectory (total time num*h)

    # Initial conditions grid (same style as你的示例)
    q_init = np.linspace(-2*np.pi, 2*np.pi, 6)
    p_init = np.linspace(-3.0, 3.0, 5)
    initials = np.array([[q0, p0] for q0 in q_init for p0 in p_init], dtype=float)

    # Choose integrator
    flow = try_project_sv6()
    if flow is None:
        flow = make_builtin_sv6()

    # Collect data (x -> y one-step pairs) and all points for scatter
    X_list, Y_list, cloud = [], [], []
    for x0 in initials:
        traj = flow(x0, h, num)   # (num+1, 2)
        # accumulate all points for scatter
        cloud.append(traj)
        # build supervised pairs (x_t, x_{t+1})
        X_list.append(traj[:-1])
        Y_list.append(traj[1:])

    cloud = np.vstack(cloud)      # all visited states
    X = np.vstack(X_list)         # inputs
    Y = np.vstack(Y_list)         # targets

    # Save dataset
    np.savez("sv6_pendulum_dataset.npz", x=X, y=Y, h=h)
    print(f"[Saved] sv6_pendulum_dataset.npz  (x shape {X.shape}, y shape {Y.shape}, h={h})")

    # Plot scatter in phase space
    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    k = 15   # 每10个点取1个，点立即变稀疏
    ax.scatter(cloud[::k, 0], cloud[::k, 1],
           s=20, c='royalblue', alpha=0.6, edgecolors='none')

    ax.set_xlim(-2*np.pi, 2*np.pi)
    ax.set_ylim(-3.5, 3.5)
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.6)
   
    # ax.set_title(r'Phase space (SV6) for $H(q,p)=\frac{1}{2}p^2+\cos q$')
    # ax.set_xlabel(r'Angle $q$')
    ax.set_xlabel(r'Angle q', fontsize=18)
    # ax.set_ylabel(r'Momentum $p$')
    ax.set_ylabel(r'Momentum p', fontsize=18)


    fig.tight_layout()
    fig.savefig("sv6_pendulum_scatter.png", dpi=300, bbox_inches='tight')
    print("[Saved] sv6_pendulum_scatter.png")

if __name__ == "__main__":
    main()