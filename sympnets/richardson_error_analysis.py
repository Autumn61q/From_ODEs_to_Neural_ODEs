import numpy as np
import matplotlib.pyplot as plt

def parse_result_file(filename):
    """Parse result txt file and extract h, Tr Net, Tr Int, Te Net, Te Int"""
    h_values = []
    tr_net = []
    tr_int = []
    te_net = []
    te_int = []
    
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Skip header lines (=== and column names)
    data_lines = [line.strip() for line in lines if line.strip() and not line.startswith('=') and not line.startswith('-')]
    data_lines = data_lines[1:]  # Skip column header
    
    for line in data_lines:
        if not line or line[0] == "#":
            continue
        # Split by '|' and extract columns
        parts = [p.strip() for p in line.split('|')]
        if len(parts) >= 11:  # Need at least h + 10 columns
            h_values.append(float(parts[0]))
            tr_net.append(float(parts[7]))   # Column 8: Tr Net
            tr_int.append(float(parts[8]))   # Column 9: Tr Int
            te_net.append(float(parts[9]))   # Column 10: Te Net
            te_int.append(float(parts[10]))  # Column 11: Te Int
    
    return np.array(h_values), np.array(tr_net), np.array(tr_int), np.array(te_net), np.array(te_int)

def fit_slope(h, err):
    # 只用非零点
    mask = (h > 0) & (err > 0)
    # 过滤掉过小的步长，只用较大步长计算斜率（例如 h >= 0.2），避免机器精度/分离失效干扰
    # 对于本实验数据，0.5, 0.4, 0.3, 0.2是比较可靠的渐近区间
    valid_indices = np.where(h[mask] >= 0.2)
    
    if len(valid_indices[0]) < 2:
        # 如果点太少，就用所有点
        h_fit = h[mask]
        e_fit = err[mask]
    else:
        h_fit = h[mask][valid_indices]
        e_fit = err[mask][valid_indices]

    logh = np.log(h_fit)
    loge = np.log(e_fit)
    slope, _ = np.polyfit(logh, loge, 1)
    return slope

# Read data from both files
h2, tr_net2, tr_int2, te_net2, te_int2 = parse_result_file('2nd_result.txt')
h3, tr_net3, tr_int3, te_net3, te_int3 = parse_result_file('3rd_result.txt')

# 计算所有曲线斜率
slope_tr_net2 = fit_slope(h2, tr_net2)
slope_tr_net3 = fit_slope(h3, tr_net3)
slope_tr_int2 = fit_slope(h2, tr_int2)
slope_tr_int3 = fit_slope(h3, tr_int3)
slope_te_net2 = fit_slope(h2, te_net2)
slope_te_net3 = fit_slope(h3, te_net3)
slope_te_int2 = fit_slope(h2, te_int2)
slope_te_int3 = fit_slope(h3, te_int3)

# Create figure with 2x2 subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=150)

# Row 1: Train (Net and Int)
# Left: Train Net
ax = axes[0, 0]
ax.loglog(h2, tr_net2, 'o-', label=f'2nd Order (slope={slope_tr_net2:.2f})', linewidth=2, markersize=8)
ax.loglog(h3, tr_net3, 's-', label=f'3rd Order (slope={slope_tr_net3:.2f})', linewidth=2, markersize=8)
ax.set_ylabel('Error', fontsize=12, fontweight='bold')
ax.set_title('Train Net Error', fontsize=13, fontweight='bold')
ax.grid(True, which='both', alpha=0.3)
ax.legend(fontsize=11)

# Right: Train Int
ax = axes[0, 1]
ax.loglog(h2, tr_int2, 'o-', label=f'2nd Order (slope={slope_tr_int2:.2f})', linewidth=2, markersize=8)
ax.loglog(h3, tr_int3, 's-', label=f'3rd Order (slope={slope_tr_int3:.2f})', linewidth=2, markersize=8)
ax.set_ylabel('Error', fontsize=12, fontweight='bold')
ax.set_title('Train Int Error', fontsize=13, fontweight='bold')
ax.grid(True, which='both', alpha=0.3)
ax.legend(fontsize=11)

# Row 2: Test (Net and Int)
# Left: Test Net


ax = axes[1, 0]
ax.loglog(h2, te_net2, 'o-', label=f'2nd Order (slope={slope_te_net2:.2f})', linewidth=2, markersize=8)
ax.loglog(h3, te_net3, 's-', label=f'3rd Order (slope={slope_te_net3:.2f})', linewidth=2, markersize=8)
ax.set_xlabel('Step size h', fontsize=12, fontweight='bold')
ax.set_ylabel('Error', fontsize=12, fontweight='bold')
ax.set_title('Test Net Error', fontsize=13, fontweight='bold')
ax.grid(True, which='both', alpha=0.3)
ax.legend(fontsize=11)

# Right: Test Int
ax = axes[1, 1]
ax.loglog(h2, te_int2, 'o-', label=f'2nd Order (slope={slope_te_int2:.2f})', linewidth=2, markersize=8)
ax.loglog(h3, te_int3, 's-', label=f'3rd Order (slope={slope_te_int3:.2f})', linewidth=2, markersize=8)
ax.set_xlabel('Step size h', fontsize=12, fontweight='bold')
ax.set_ylabel('Error', fontsize=12, fontweight='bold')
ax.set_title('Test Int Error', fontsize=13, fontweight='bold')
ax.grid(True, which='both', alpha=0.3)
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig('richardson_error_analysis.png', dpi=300, bbox_inches='tight')
print("Saved as: richardson_error_analysis.png'")
# plt.show()

print("斜率统计：")
print(f"Train Net: 2nd={slope_tr_net2:.3f}, 3rd={slope_tr_net3:.3f}")
print(f"Train Int: 2nd={slope_tr_int2:.3f}, 3rd={slope_tr_int3:.3f}")
print(f"Test Net:  2nd={slope_te_net2:.3f}, 3rd={slope_te_net3:.3f}")
print(f"Test Int:  2nd={slope_te_int2:.3f}, 3rd={slope_te_int3:.3f}")
