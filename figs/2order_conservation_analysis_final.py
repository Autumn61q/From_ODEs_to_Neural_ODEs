import numpy as np
import torch
import sys
import os
import matplotlib.pyplot as plt

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Add parent directory to Python path to import learner module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import learner as ln
from learner.integrator.hamiltonian import SV

# Style consistency with reference files
plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'axes.facecolor': 'white',
    'figure.facecolor': 'white',
    'font.size': 14
})

def polygon_area(x, y):
    """Compute polygon area via shoelace formula."""
    return 0.5 * np.abs(np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))

def get_energy(traj):
    """
    Calculate energy for trajectory. 
    traj shape: [..., 2] where [..., 0] is momentum p, [..., 1] is position q.
    Hamiltonian: H = 0.5 * p^2 - cos(q)
    """
    p = traj[..., 0]
    q = traj[..., 1]
    return 0.5 * p**2 - np.cos(q)

def main():
    device = 'cpu'
    # HNN size based on defaults or common project structure
    H_size = [2, 30, 30, 1]
    h_pred = 0.08
    steps = 400
    base_model_path = r'../sympnets/models/2nd'
    
    # --- 1. Load 5 models ---
    models = []
    print("Loading 5 models...")
    for run in range(1, 2):
    # for run in range(1, 6):
        model_path = os.path.join(base_model_path, f'h0p08_run{run}')
        model_file = os.path.join(model_path, 'model_best.pkl')
        if os.path.exists(model_file):
            try:
                # Load with weights_only=False as the saved files are pkl objects
                model = torch.load(model_file, map_location=device, weights_only=False)
                model.eval()
                model.to(torch.float64)
                models.append(model)
                print(f"  Loaded run {run}")
            except Exception as e:
                print(f"  Error loading run {run}: {e}")
        else:
            print(f"  Warning: {model_file} not found.")

    if not models:
        print("No models loaded. Exiting.")
        return

    # --- 2. Energy Conservation (Single Initial Condition) ---
    x0_energy = np.array([0.0, 1]) # [p, q]
    print(f"Generating energy trajectory for x0={x0_energy}...")
    
    # Ground Truth: SV6
    # Note: p' = -sin(q), q' = p
    true_solver = SV(None, lambda p,q: (p, np.sin(q)), iterations=1, order=6, N=100)
    
    gt_traj = true_solver.flow(torch.tensor(x0_energy, dtype=torch.double), h_pred, steps).numpy()
    E_gt = get_energy(gt_traj)
    
    # Model predictions
    model_e_trajs = []
    for i, model in enumerate(models):
        pred = model.predict(x0_energy, h_pred, steps=steps, keepinitx=True, returnnp=True)
        model_e_trajs.append(get_energy(pred))
    
    model_e_trajs = np.array(model_e_trajs) # [5, steps+1]
    E_mean = np.mean(model_e_trajs, axis=0)
    E_std = np.std(model_e_trajs, axis=0)
    times = np.arange(steps + 1) * h_pred

    # --- 3. Phase Space Volume (Circle of Initial Conditions) ---
    print("Generating Phase Space Volume over time...")
    M = 100 # number of boundary points
    q_center, p_center = 0.0, 1 # matching x0_energy
    radius = 0.2
    angles = np.linspace(0, 2 * np.pi, M, endpoint=False)
    q0_blob = q_center + radius * np.cos(angles)
    p0_blob = p_center + radius * np.sin(angles)
    x0_blob = np.stack([p0_blob, q0_blob], axis=1) # [M, 2] -> [p, q]
    
    # Store all trajectories for all models to calculate area at each step
    all_models_blob_trajs = []
    for i, model in enumerate(models):
        print(f"  Evolving blob with model {i+1}...")
        model_trajs = []
        for j in range(M):
            traj = model.predict(x0_blob[j], h_pred, steps=steps, keepinitx=True, returnnp=True)
            model_trajs.append(traj)
        all_models_blob_trajs.append(np.array(model_trajs))
    
    all_models_blob_trajs = np.array(all_models_blob_trajs) # [5, M, steps+1, 2]
    avg_blob_trajs = np.mean(all_models_blob_trajs, axis=0) # [M, steps+1, 2]
    
    areas = []
    for k in range(steps + 1):
        points = avg_blob_trajs[:, k, :] # [M, 2]
        areas.append(polygon_area(points[:, 1], points[:, 0])) # (q, p)
    areas = np.array(areas)

    # --- 4. Plotting (Vertical Layout) ---
    fig, axes = plt.subplots(2, 1, figsize=(7, 7))
    
    # Subplot 1: Phase Space Volume Conservation
    ax = axes[0]
    ax.plot(times, areas, color='royalblue', linewidth=1.5, label='Area of Prediction')
    ax.axhline(y=np.pi * radius**2, color='k', linestyle='--', alpha=0.7, label='Ground Truth')
        
    ax.set_title("Phase-Space Volume Preservation", fontsize=16, fontweight='bold')
    ax.set_xlabel("Time t", fontsize=18)
    ax.set_ylabel("Area", fontsize=18)
    ax.legend(loc='best')
    ax.set_ylim(0.02, 0.2)
    ax.grid(True, alpha=0.2, linestyle='--')
    # Subplot 2: Energy Conservation
    ax = axes[1]
    # Energy deviation E(t) - E(0)
    de_mean = E_mean - E_mean[0]
    de_gt = E_gt - E_gt[0]
    
    ax.plot(times, de_gt, linestyle='--', linewidth=1.4, label='Ground Truth')
    ax.plot(times, de_mean, color='royalblue', linewidth=1.4, label='Energy of Prediction')
    ax.fill_between(times, de_mean - E_std, de_mean + E_std, color='royalblue', alpha=0.2)
    
    ax.set_title("Energy Conservation", fontsize=16, fontweight='bold')
    ax.set_xlabel("Time t", fontsize=18)
    ax.set_ylabel(r"$\Delta$ Energy", fontsize=18)
    
    ax.set_ylim(-0.3, 0.7) 
    
    axins = ax.inset_axes([0.55, 0.60, 0.45, 0.45]) 
    axins.plot(times, de_gt, linestyle='--', linewidth=1.2)
    axins.plot(times, de_mean, color='royalblue', linewidth=1.2)
    axins.fill_between(times, de_mean - E_std, de_mean + E_std, color='royalblue', alpha=0.2)

    x1, x2, y1, y2 = 5, 20, -0.006, 0.01
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)

    axins.set_xticklabels([])
    axins.set_yticklabels([])

    ax.indicate_inset_zoom(axins, edgecolor="red")

    ax.grid(True, alpha=0.2, linestyle='--')
    ax.legend(loc='upper left')

    plt.tight_layout()
    save_name = '2order_conservation_analysis_final.png'
    plt.savefig(save_name, dpi=300, bbox_inches='tight')
    print(f"Successfully saved conservation analysis to {save_name}")
    
    # Print areas for console check
    print("\nPhase-space area conservation check:")
    snapshot_indices = np.linspace(0, steps, 5).astype(int)
    for idx in snapshot_indices:
        t_val = idx * h_pred
        print(f"  t = {t_val:.1f}: Area = {areas[idx]:.6f}")

if __name__ == '__main__':
    main()
