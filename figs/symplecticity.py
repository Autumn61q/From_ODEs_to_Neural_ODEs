import numpy as np
import matplotlib.pyplot as plt

# Style consistency with stability_domains.py
plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'axes.facecolor': 'white',
    'figure.facecolor': 'white',
    'font.size': 14
})

# Hamiltonian system: H(q,p) = 0.5 * p^2 - cos(q)
# Equations: q' = p, p' = -sin(q)

def symplectic_step(q, p, dt):
    """One step of Velocity-Verlet (symplectic integrator)."""
    p_half = p + 0.5 * dt * (-np.sin(q))
    q_new = q + dt * p_half
    p_new = p_half + 0.5 * dt * (-np.sin(q_new))
    return q_new, p_new

def flow_map(q, p, T=1.0, dt=1e-2):
    """Advance (q,p) by time T using symplectic steps."""
    n = int(np.round(T / dt))
    rem = T - n * dt
    for _ in range(n):
        q, p = symplectic_step(q, p, dt)
    if abs(rem) > 1e-12:
        q, p = symplectic_step(q, p, rem)
    return q, p

def polygon_area(x, y):
    """Compute polygon area via shoelace formula."""
    return 0.5 * np.abs(np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))

# Initial condition: blob (disc) in phase space
q_center, p_center = np.pi / 3, 0.0
radius = 0.45
M = 600  # number of boundary points

angles = np.linspace(0, 2 * np.pi, M, endpoint=False)
q0 = q_center + radius * np.cos(angles)
p0 = p_center + radius * np.sin(angles)

# Snapshots over time
K = 5
T_unit = 1.3

fig, ax = plt.subplots(figsize=(7, 5))

# Use a colormap similar to the royalblue theme
colors = plt.cm.Blues(np.linspace(0.4, 0.9, K))
areas = []

qk, pk = q0.copy(), p0.copy()
for k in range(K):
    # Plot outline
    ax.plot(qk, pk, color=colors[k], linewidth=2.0, label=f't = {k * T_unit:.1f}')
    # Fill with low alpha for visibility, consistent with stability regions style
    ax.fill(qk, pk, color=colors[k], alpha=0.15)
    
    areas.append(polygon_area(qk, pk))
    qk, pk = flow_map(qk, pk, T=T_unit, dt=1e-2)

ax.set_xlabel('Position q', fontsize=18)
ax.set_ylabel('Momentum p', fontsize=18)
ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_aspect('equal', adjustable='box')
ax.legend(loc='upper right', framealpha=0.95, fontsize=10)
# ax.set_title('Phase-Space Volume Preservation', 
#              fontsize=16, fontweight='bold')

# Style specific additions matching stability_domains.py
ax.axhline(0, color='gray', linewidth=0.8, alpha=0.6)
ax.axvline(0, color='gray', linewidth=0.8, alpha=0.6)
ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)

plt.tight_layout()
plt.savefig('symplecticity.png', dpi=300, bbox_inches='tight')
print('Saved: symplecticity.png')

# Print area check
print('\nPhase-space volume (should be nearly constant):')
for k, area in enumerate(areas):
    print(f't = {k * T_unit:.1f}: Area = {area:.6f}')