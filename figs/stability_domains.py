import numpy as np
import matplotlib.pyplot as plt

# Style: white background, grid, royalblue; font size 14; English labels only
plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'axes.facecolor': 'white',
    'figure.facecolor': 'white',
    'font.size': 18
})

# Complex plane grid z = x + i y
xmin, xmax = -4.0, 2.0
ymin, ymax = -3.5, 3.5
Nx, Ny = 800, 700
x = np.linspace(xmin, xmax, Nx)
y = np.linspace(ymin, ymax, Ny)
X, Y = np.meshgrid(x, y)
Z = X + 1j*Y

# Stability functions
R_fe = 1 + Z                               # Forward Euler
R_im = (1 + Z/2) / (1 - Z/2)               # Implicit Midpoint

stable_fe = np.abs(R_fe) <= 1.0
stable_im = np.abs(R_im) <= 1.0

# Figure with two subplots stacked vertically
fig, axes = plt.subplots(2, 1, figsize=(7, 8), sharex=True)

# Top: Forward Euler
ax = axes[0]
ax.contourf(
    X, Y, stable_fe,
    levels=[-0.5, 0.5, 1.5],
    colors=[(0,0,0,0), (65/255,105/255,225/255,0.15)],
)
ax.contour(X, Y, np.abs(R_fe), levels=[1.0], colors=['royalblue'], linewidths=2.0)
# Reference circle: center -1, radius 1
theta = np.linspace(0, 2*np.pi, 800)
ax.plot(-1 + np.cos(theta), np.sin(theta), color='royalblue', alpha=0.6, linestyle='--', linewidth=1.2)
ax.axhline(0, color='gray', linewidth=0.8, alpha=0.6)
ax.axvline(0, color='gray', linewidth=0.8, alpha=0.6)
ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
ax.set_ylabel('Im(z)')
ax.set_title('Forward Euler', fontsize=16, fontweight='bold')
ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)

# Bottom: Implicit Midpoint
ax = axes[1]
ax.contourf(
    X, Y, stable_im,
    levels=[-0.5, 0.5, 1.5],
    colors=[(0,0,0,0), (65/255,105/255,225/255,0.15)],
)
ax.contour(X, Y, np.abs(R_im), levels=[1.0], colors=['royalblue'], linewidths=2.0)
ax.axvline(0, color='royalblue', linewidth=1.2, linestyle='--', alpha=0.6)
ax.axhline(0, color='gray', linewidth=0.8, alpha=0.6)
ax.axvline(0, color='gray', linewidth=0.8, alpha=0.6)
ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
ax.set_xlabel('Re(z)')
ax.set_ylabel('Im(z)')
ax.set_title('Implicit Midpoint', fontsize=16, fontweight='bold')
ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)

plt.tight_layout()
fig.savefig('stability_domains.png', bbox_inches='tight')
print('Saved: stability_domains.png')