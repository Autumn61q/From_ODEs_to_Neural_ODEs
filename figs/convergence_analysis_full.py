import numpy as np
import matplotlib.pyplot as plt

# Style consistency matching e:/ODEresearch/From_ODEs_to_Neural_ODEs/figs
plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'axes.facecolor': 'white',
    'figure.facecolor': 'white',
    'font.size': 14
})

def parse_result_file(file_path):
    h_list = []
    te_mae_list = []
    zero_mae_list = []
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        line = line.strip()
        # Skip header/footer decorators
        if not line or line.startswith('=') or line.startswith('-') or line.startswith('h '):
            continue
            
        parts = [p.strip() for p in line.split('|')]
        if len(parts) < 4:
            continue
            
        h_str = parts[0]
        # Ignore lines starting with # as requested
        if h_str.startswith('#'):
            continue
            
        try:
            h = float(h_str)
            te_mae = float(parts[2])
            zero_mae = float(parts[3])
            
            h_list.append(h)
            te_mae_list.append(te_mae)
            zero_mae_list.append(zero_mae)
        except ValueError:
            continue
            
    return np.array(h_list), np.array(te_mae_list), np.array(zero_mae_list)

# Paths
res2_path = r'../sympnets/2nd_result.txt'
res1_path = r'../sympnets/1st_result.txt'

# Load data
h2, te2, zero2 = parse_result_file(res2_path)
h1, te1, zero1 = parse_result_file(res1_path)

fig, axes = plt.subplots(2, 1, figsize=(7, 10), dpi=150)

# ================= TOP: 1-step Full-shot MAE =================
ax = axes[0]

ax.loglog(h1, te1, 'o-', color='crimson', linewidth=2, markersize=8, label='1st-order')
ax.loglog(h2, te2, 'o-', color='royalblue', linewidth=2, markersize=8, label='2nd-order')

h_ref = np.array([min(np.concatenate([h2, h1])), max(np.concatenate([h2, h1]))])
ax.loglog(h_ref, 10 ** (0.35) * (h_ref**1.7), 'k:', alpha=0.3, linewidth=1.5, label='Slope 1.7')
ax.loglog(h_ref, 10 ** (-0.35) * (h_ref**3), 'k--', alpha=0.3, linewidth=1.5, label='Slope 3')

ax.set_title("1-step Prediction Error (Full-shot)", fontsize=16, fontweight='bold')
ax.set_ylabel('MAE', fontsize=18)
ax.grid(True, which="both", alpha=0.2, linestyle='--')
ax.legend(loc='best', fontsize=12)

# ================= BOTTOM: Long-term rollout error =================
ax2 = axes[1]


ax2.loglog(h1, zero1, 'o-', color='crimson', linewidth=2, markersize=8, label='1st-order')
ax2.loglog(h2, zero2, 'o-', color='royalblue', linewidth=2, markersize=8, label='2nd-order')

ax2.loglog(h_ref, 10 ** (0.11) * (h_ref**1), 'k:', alpha=0.3, linewidth=1.5, label='Slope 1')
ax2.loglog(h_ref, 10 ** (-0.53) * (h_ref**2), 'k--', alpha=0.3, linewidth=1.5, label='Slope 2')

ax2.set_title("Long-term Rollout Error", fontsize=16, fontweight='bold')
ax2.set_xlabel('Step size $h$', fontsize=18)
ax2.set_ylabel('MAE', fontsize=18)
ax2.grid(True, which="both", alpha=0.2, linestyle='--')
ax2.legend(loc='best', fontsize=12)

plt.tight_layout()
save_path = r'convergence_analysis_full.png'
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f'Saved convergence analysis plot to {save_path}')

