import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Conservative pendulum
def conservative_pendulum(state, t):
    q, p = state
    return [p, np.sin(q)]

# Damped pendulum
def damped_pendulum(state, t, gamma=0.3):
    q, p = state
    return [p, np.sin(q) - gamma*p]

# Arrow helper
def add_arrows(ax, q_traj, p_traj, num_arrows=4, size=0.22):
    arrow_indices = np.linspace(50, len(q_traj)-50, num_arrows).astype(int)
    for idx in arrow_indices:
        dq = p_traj[idx]
        dp = np.sin(q_traj[idx])
        norm = np.sqrt(dq**2 + dp**2)
        dq /= norm
        dp /= norm
        ax.arrow(
            q_traj[idx], p_traj[idx],
            size * dq, size * dp,
            head_width=0.22,
            head_length=0.22,
            fc='royalblue',
            ec='royalblue',
            alpha=0.85,
            length_includes_head=True
        )

# Create figure (VERTICAL layout)
fig, axes = plt.subplots(2, 1, figsize=(7, 8), dpi=150)

# Initial conditions
q_init = np.linspace(-2*np.pi, 2*np.pi, 6)
p_init = np.linspace(-3.0, 3.0, 5)
t = np.linspace(0, 20, 2000)

# ================= TOP: CONSERVATIVE =================
ax = axes[0]
for q0 in q_init:
    for p0 in p_init:
        traj = odeint(conservative_pendulum, [q0, p0], t)
        q_traj, p_traj = traj[:,0], traj[:,1]
        ax.plot(q_traj, p_traj, color='royalblue', alpha=0.8, linewidth=1.4)
        add_arrows(ax, q_traj, p_traj)

ax.set_title("Conservative Pendulum", fontsize=16, fontweight='bold')
ax.set_xlabel("Angle q", fontsize=18)
ax.set_ylabel("Momentum p", fontsize=18)
ax.set_xlim(-2*np.pi, 2*np.pi)
ax.set_ylim(-3.5, 3.5)
ax.grid(True, alpha=0.2, linestyle='--')

# ================= BOTTOM: DAMPED =================
ax = axes[1]
for q0 in q_init:
    for p0 in p_init:
        traj = odeint(damped_pendulum, [q0, p0], t)
        q_traj, p_traj = traj[:,0], traj[:,1]
        ax.plot(q_traj, p_traj, color='royalblue', alpha=0.8, linewidth=1.4)
        add_arrows(ax, q_traj, p_traj)

ax.set_title("Damped Pendulum", fontsize=16, fontweight='bold')
ax.set_xlabel("Angle q", fontsize=18)
ax.set_ylabel("Momentum p", fontsize=18)
ax.set_xlim(-2*np.pi, 2*np.pi)
ax.set_ylim(-3.5, 3.5)
ax.grid(True, alpha=0.2, linestyle='--')

plt.tight_layout()
plt.savefig('energy_comparison.png', dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
print("Comparison figure saved as 'energy_comparison.png'")

# plt.show()
