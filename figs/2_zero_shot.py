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
    zero_mae_list = []
    lo_zero_mae_list = []
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        line = line.strip()
        # Skip header/footer decorators
        if not line or line.startswith('=') or line.startswith('-') or line.startswith('h '):
            continue
            
        parts = [p.strip() for p in line.split('|')]
        if len(parts) < 6:
            continue
            
        h_str = parts[0]
        # Ignore lines starting with # as requested
        if h_str.startswith('#'):
            continue
            
        try:
            h = float(h_str)
            zero_mae = float(parts[3])
            lo_zero_mae = float(parts[5])
            
            h_list.append(h)
            zero_mae_list.append(zero_mae)
            lo_zero_mae_list.append(lo_zero_mae)
        except ValueError:
            continue
            
    return np.array(h_list), np.array(zero_mae_list), np.array(lo_zero_mae_list)

# Paths
res2_path = r'../sympnets/2nd_result.txt'

# Load data
h, zero_mae, lo_zero_mae = parse_result_file(res2_path)

# Plotting (Vertical layout matching energy_comparsion.py)
fig, axes = plt.subplots(2, 1, figsize=(7, 10), dpi=150)

# ================= TOP: Zero-shot 1-step MAE =================
ax = axes[0]
ax.loglog(h, zero_mae, 'o-', color='royalblue', linewidth=2, markersize=8, label='2nd-order curve')

# Reference slope lines
h_ref = np.array([min(h), max(h)])
# ax.loglog(h_ref, 10 ** (-1.7) * (h_ref**2), 'k--', alpha=0.3, linewidth=1.5, label='Slope 2')
ax.loglog(h_ref, 10 ** (-0.4) * (h_ref**2.7), 'k:', alpha=0.3, linewidth=1.5, label='Slope 2.7')

ax.set_title("Zero-shot 1-step Prediction Error", fontsize=16, fontweight='bold')
ax.set_ylabel('MAE', fontsize=18)
ax.grid(True, which="both", alpha=0.2, linestyle='--')
ax.legend(loc='best', fontsize=14)

# ================= BOTTOM: Zero-shot Long-term MAE =================
ax = axes[1]
ax.loglog(h, lo_zero_mae, 's-', color='crimson', linewidth=2, markersize=8, label='2nd-order curve')

ax.loglog(h_ref, 10 ** (-0.8) * (h_ref**2.4), 'k--', alpha=0.3, linewidth=1.5, label='Slope 2.4')
# ax.loglog(h_ref, 10 ** (-1.2) * (h_ref**3), 'k:', alpha=0.3, linewidth=1.5, label='Slope 3')

ax.set_title("Zero-shot Long-term Prediction Error", fontsize=16, fontweight='bold')
ax.set_xlabel('Step size $h$', fontsize=18)
ax.set_ylabel('MAE', fontsize=18)
ax.grid(True, which="both", alpha=0.2, linestyle='--')
ax.legend(loc='best', fontsize=14)

plt.tight_layout()
save_path = r'2_zero_shot.png'
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f'Saved zero-shot analysis plot to {save_path}')
# plt.show()
