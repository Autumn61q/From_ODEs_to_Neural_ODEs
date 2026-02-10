import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def pendulum_ode(state, t):
    q, p = state
    return [p, np.sin(q)]

fig, ax = plt.subplots(figsize=(7, 5), dpi=150)

q_init = np.linspace(-2*np.pi, 2*np.pi, 6)
p_init = np.linspace(-3.0, 3.0, 5)

for q0 in q_init:
    for p0 in p_init:
        t = np.linspace(0, 15, 800)
        trajectory = odeint(pendulum_ode, [q0, p0], t)
        q_traj = trajectory[:, 0]
        p_traj = trajectory[:, 1]

        # 画轨迹
        ax.plot(q_traj, p_traj, color='royalblue', alpha=0.8, linewidth=1.4)

        # ---- 在轨迹上加箭头 ----
        # 选几个点作为箭头位置
        arrow_indices = np.linspace(50, len(t)-50, 4).astype(int)

        for idx in arrow_indices:
            dq = p_traj[idx]
            dp = np.sin(q_traj[idx])
            norm = np.sqrt(dq**2 + dp**2)
            dq /= norm
            dp /= norm

            ax.arrow(
                q_traj[idx], p_traj[idx],
                0.2 * dq, 0.2 * dp,     # 箭头长度
                head_width=0.25,
                head_length=0.25,
                fc='royalblue',
                ec='royalblue',
                alpha=0.8,
                length_includes_head=True
            )

ax.set_xlabel('Angle q', fontsize=18)
ax.set_ylabel('Momentum p', fontsize=18)
ax.tick_params(axis='both', labelsize=14)

ax.set_xlim(-2*np.pi, 2*np.pi)
ax.set_ylim(-3.5, 3.5)

ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
ax.set_facecolor('white')
fig.patch.set_facecolor('white')

plt.tight_layout()
plt.savefig('pendulum_flow_map.png', dpi=300, bbox_inches='tight', facecolor='white')
print("Saved phase portrait with arrows on trajectories.")