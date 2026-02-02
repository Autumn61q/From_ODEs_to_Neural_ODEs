import numpy as np
import torch
import learner as ln
from learner.integrator.hamiltonian import SV
from pendulum import PDData
import matplotlib.pyplot as plt
import os

# 环境配置
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 1. 数据类：支持指定步长 h 生成轨迹
class FlexiblePDData(ln.Data):
    def __init__(self, x0, h, train_num, test_num, add_h=True):
        super(FlexiblePDData, self).__init__()
        # ground truth Hamiltonian: H = 0.5 * p^2 - cos(q)
        self.H = lambda p, q: 0.5 * p**2 - np.cos(q)
        self.x0 = x0
        self.h = h
        self.train_num = train_num
        self.test_num = test_num
        self.add_h = add_h
        self.__init_data()
        
    @property
    def dim(self):
        return 2
    
    def __generate_flow(self, x0, h, num):
        # # 用二阶中点法（RK2）生成轨迹
        # def dH(p, q):
        #     return p, np.sin(q)
        # x = np.array(x0).reshape(1, -1)
        # traj = [x[0]]
        # for i in range(num):
        #     p, q = traj[-1][0], traj[-1][1]
        #     dp, dq = dH(p, q)
        #     # Midpoint method (RK2): second-order Runge-Kutta
        #     p_mid = p + h / 2 * dp
        #     q_mid = q + h / 2 * dq
        #     dp_mid, dq_mid = dH(p_mid, q_mid)
        #     p_new = p + h * dp_mid
        #     q_new = q + h * dq_mid
        #     traj.append([p_new, q_new])
        # X = np.array(traj)
        # x, y = X[:-1], X[1:]
        # if self.add_h:
        #     x = [x, self.h * np.ones([x.shape[0], 1])]
        # return x, y
        # 改用高精度积分器（如 SV6）生成真值数据，这样 RK3 才会体现出阶数误差
        from learner.integrator.hamiltonian import SV
        true_solver = SV(None, lambda p,q: (p, np.sin(q)), iterations=1, order=6, N=100)
        X = true_solver.flow(torch.tensor(x0, dtype=torch.double), h, num).numpy()
        
        x, y = X[:-1], X[1:]
        if self.add_h:
            x = [x, self.h * np.ones([x.shape[0], 1])]
        return x, y
    
    def __init_data(self):
        self.X_train, self.y_train = self.__generate_flow(self.x0, self.h, self.train_num)
        self.X_test, self.y_test = self.__generate_flow(self.y_train[-1], self.h, self.test_num)

def calculate_metrics(y_true, y_pred):
    mse = np.mean((y_true - y_pred)**2)
    mae = np.mean(np.abs(y_true - y_pred))
    return mse, mae

def analyze_error_separation(best_model, x0, h_base, target_time, true_solver):
    """
    使用 Richardson 外推分离网络误差和积分器误差（长时预测）
    
    返回:
        - y_h: 步长为 h 的预测结果
        - y_network: Richardson 外推得到的网络主导结果 (消除积分器误差)
        - y_true: 高精度真值
        - network_error: 网络学习误差 (y_network - y_true)
        - integrator_error_h: h 步长下的积分器误差估计 (y_h - y_network)
    """
    # Richardson 外推：用三个不同步长计算
    hs = [h_base, h_base/2, h_base/4]
    predictions = []
    
    for h in hs:
        steps = int(target_time / h)
        pred = best_model.predict(x0, h, steps=steps, keepinitx=True, returnnp=True)
        predictions.append(pred[-1])  # 取终点
    
    y_h, y_h2, y_h4 = predictions
    
    # 2点 Richardson 外推：消除 O(h^2) 的积分器误差（2阶方法）
    # 对于2阶方法，全局误差是 O((Δh)^2)，误差比例为 1:1/4:1/16
    # Richardson公式：y_network = (4 * y_h4 - y_h2) / 3
    y_network = (4 * y_h4 - y_h2) / 3
    
    # 计算高精度真值
    steps_true = int(target_time / (h_base / 10))
    y_true = true_solver.flow(x0, h_base / 10, steps_true).cpu().detach().numpy() if isinstance(true_solver.flow(x0, h_base / 10, steps_true), torch.Tensor) else true_solver.flow(x0, h_base / 10, steps_true)
    y_true = y_true[-1] if isinstance(y_true, np.ndarray) and y_true.ndim == 2 else y_true
    
    # 计算各类误差
    network_error = np.linalg.norm(y_network - y_true)
    integrator_error_h = np.linalg.norm(y_h - y_network)
    total_error = np.linalg.norm(y_h - y_true)
    
    return {
        'y_h': y_h,
        'y_network': y_network,
        'y_true': y_true,
        'network_error': network_error,
        'integrator_error': integrator_error_h,
        'total_error': total_error
    }

def analyze_short_term_error_separation(best_model, x0_data, h, y_true_data, true_solver):
    """
    使用 Richardson 外推分离短时预测（1-step）的网络误差和积分器误差
    
    参数:
        - best_model: 训练好的模型
        - x0_data: 初始条件数据 (形状: [N, 2])
        - h: 步长
        - y_true_data: 真值数据 (形状: [N, 2])
        - true_solver: 真值求解器
    """
    # Richardson 外推：保持终点时间相同 (= h)，但用不同步长
    # h步长预测1步，h/2步长预测2步，h/4步长预测4步
    hs_configs = [
        (h, 1),      # h, 1 step
        (h/2, 2),    # h/2, 2 steps
        (h/4, 4)     # h/4, 4 steps
    ]
    
    predictions = []
    for h_step, num_steps in hs_configs:
        pred = best_model.predict(x0_data, h_step, steps=num_steps, returnnp=True)
        # 只取最后一步的结果
        if pred.ndim == 3:  # 形状为 [N, steps, dim] 或 [steps, dim]
            predictions.append(pred[..., -1, :])  # 取最后一步
        else:
            predictions.append(pred)
    
    y_h, y_h2, y_h4 = predictions
    
    # 2点 Richardson 外推：消除 O(h^2) 的积分器误差（2阶方法）
    # 对于2阶方法，全局误差是 O((Δh)^2)，误差比例为 1:1/4:1/16
    # Richardson公式：y_network = (4 * y_h4 - y_h2) / 3
    y_network = (4 * y_h4 - y_h2) / 3
    
    # 计算各类误差
    network_error = np.mean(np.abs(y_network - y_true_data))     # 平均绝对误差
    integrator_error_h = np.mean(np.abs(y_h - y_network))        # 积分器误差
    total_error = np.mean(np.abs(y_h - y_true_data))             # 总误差
    
    return {
        'y_h': y_h,
        'y_network': y_network,
        'y_true': y_true_data,
        'network_error': network_error,
        'integrator_error': integrator_error_h,
        'total_error': total_error
    }

def run_diff_h_experiment():
    device = 'gpu' if torch.cuda.is_available() else 'cpu'
    # 转换 device 为 PyTorch 能识别的格式
    pytorch_device = 'cuda' if device == 'gpu' else 'cpu'
    print(f"Using device: {pytorch_device}")
    
    # 固定参数
    x0 = np.array([0.0, 1.0])
    train_num = 20
    test_num = 100
    num_runs = 5
    iterations = 30000
    H_size = [2, 30, 30, 1]
    
    # 变化的步长 h (训练和推理保持一致)
    # hs = [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01]
    hs = [0.5, 0.4, 0.3, 0.2, 0.1]
    
    all_summary = []

    for h in hs:
        print(f"\n>>>> Testing h = {h} (Consistency Mode)")
        # 保持训练和测试的总时长恒定，消除采样范围带来的机制转换
        # T_train = 5.0, T_test = 10.0 (以 h=0.1, num=50 为基准)
        current_train_num = int(max(10.0 / h, 1))
        current_test_num = int(max(10.0 / h, 1))
        
        run_stats = []
        
        # 物理真值分析 (用于长时对比)
        T_eval = 0.5
        steps_eval = int(max(T_eval / h, 1))
        # 确保真值生成的步数能覆盖到 T_eval
        true_h_solver = SV(None, lambda p,q: (p, np.sin(q)), iterations=1, order=6, N=100)
        flow_ref = true_h_solver.flow(x0, h, steps_eval)

        for run in range(num_runs):
            print(f"  Run {run+1}/{num_runs}...", end=' ', flush=True)
            
            # 数据生成
            data = FlexiblePDData(x0, h, current_train_num, current_test_num, add_h=True)
            
            # 使用标准的 2 阶 Midpoint HNN
            net = ln.nn.HNN(H_size, activation='tanh')
            
            model_save_dir = f'../models/2nd/h{h:.2f}_run{run+1}'.replace('.', 'p')
            args = {
                'data': data, 'net': net, 'criterion': None, 'optimizer': 'adam',
                'lr': 0.001, 'iterations': iterations, 'print_every': iterations,
                'save': 'best_only', 'callback': None, 'dtype': 'double', 'device': device,
                'model_save_path': model_save_dir
            }
            
            ln.Brain.Init(**args)
            ln.Brain.Run()
            ln.Brain.Restore()
            best_model = ln.Brain.Best_model()

            # 评估 1: 训练集 1-step
            y_tr_pred = best_model.predict(data.X_train[0], data.h, steps=1, returnnp=True)
            tr_mse, tr_mae = calculate_metrics(data.y_train_np if hasattr(data, 'y_train_np') else data.y_train, y_tr_pred)
            
            # Richardson 外推分离训练集短时预测的误差
            true_h_solver = SV(None, lambda p,q: (p, np.sin(q)), iterations=1, order=6, N=100)
            tr_error_sep = analyze_short_term_error_separation(best_model, data.X_train[0], h, 
                                                               data.y_train_np if hasattr(data, 'y_train_np') else data.y_train, 
                                                               true_h_solver)
            
            # 评估 2: 测试集 1-step
            y_te_pred = best_model.predict(data.X_test[0], data.h, steps=1, returnnp=True)
            te_mse, te_mae = calculate_metrics(data.y_test_np if hasattr(data, 'y_test_np') else data.y_test, y_te_pred)
            
            # Richardson 外推分离测试集短时预测的误差
            te_error_sep = analyze_short_term_error_separation(best_model, data.X_test[0], h,
                                                               data.y_test_np if hasattr(data, 'y_test_np') else data.y_test,
                                                               true_h_solver)

            # 评估 3: 长时预测 (h 保持完全一致，不使用 predict 内部的 N 分步)
            # 使用 order=2 的 SV 求解器，且 N=1，确保求解器步长就是 h
            custom_solver = SV(best_model.ms['H'], None, iterations=1, order=2, N=1)
            x0_tensor = torch.tensor(x0.reshape(1, -1), dtype=torch.float64, device=pytorch_device)
            flow_pred = custom_solver.flow(x0_tensor, h, steps_eval).cpu().detach().numpy().reshape(-1, 2)
            
            lo_mse, lo_mae = calculate_metrics(flow_ref, flow_pred)
            
            # 评估 4: Richardson 外推分离网络误差和积分器误差
            true_h_solver = SV(None, lambda p,q: (p, np.sin(q)), iterations=1, order=6, N=100)
            error_sep = analyze_error_separation(best_model, x0, h, 0.5, true_h_solver)
            
            run_stats.append([tr_mse, tr_mae, te_mse, te_mae, lo_mse, lo_mae,
                            tr_error_sep['network_error'], tr_error_sep['integrator_error'],
                            te_error_sep['network_error'], te_error_sep['integrator_error'],
                            error_sep['network_error'], error_sep['integrator_error']])
            print(f"Tr(Net:{tr_error_sep['network_error']:.2e}, Int:{tr_error_sep['integrator_error']:.2e}) | "
                  f"Te(Net:{te_error_sep['network_error']:.2e}, Int:{te_error_sep['integrator_error']:.2e}) | "
                  f"Lo(MSE:{lo_mse:.2e}, Net:{error_sep['network_error']:.2e}, Int:{error_sep['integrator_error']:.2e})")

        stats_array = np.array(run_stats)
        means = np.mean(stats_array, axis=0)
        stds = np.std(stats_array, axis=0)
        all_summary.append({'h': h, 'means': means, 'stds': stds})

    # 输出汇总表格（包含Richardson分析结果）
    table_lines = []
    table_lines.append("="*170)
    table_lines.append(f"{'h':<6} | {'Tr MSE':<10} | {'Tr MAE':<10} | {'Te MSE':<10} | {'Te MAE':<10} | {'Lo MSE':<10} | {'Lo MAE':<10} | "
          f"{'Tr Net':<10} | {'Tr Int':<10} | {'Te Net':<10} | {'Te Int':<10} | {'Lo Net':<10} | {'Lo Int':<10}")
    table_lines.append("-" * 170)
    for res in all_summary:
        m = res['means']
        line = (f"{res['h']:<6.2f} | {m[0]:.2e} | {m[1]:.2e} | {m[2]:.2e} | {m[3]:.2e} | {m[4]:.2e} | {m[5]:.2e} | "
              f"{m[6]:.2e} | {m[7]:.2e} | {m[8]:.2e} | {m[9]:.2e} | {m[10]:.2e} | {m[11]:.2e}")
        table_lines.append(line)
    table_lines.append("="*170)
    
    # 打印到控制台
    print("\n")
    for line in table_lines:
        print(line)
    
    # 保存到txt文件
    with open('2nd_result.txt', 'w') as f:
        for line in table_lines:
            f.write(line + '\n')
    print("\nResults saved to 2nd_result.txt")

    # 绘图：h 与各类 Error 的 Log-Log 图 (3x3 子图)
    hs_plot = [r['h'] for r in all_summary]
    
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    
    plot_configs = [
        # (行, 列, 指标索引, 标题)
        (0, 0, 0, 'Train MSE'),
        (0, 1, 1, 'Train MAE'),
        (0, 2, None, 'Train Error Separation'),
        (1, 0, 2, 'Test MSE'),
        (1, 1, 3, 'Test MAE'),
        (1, 2, None, 'Test Error Separation'),
        (2, 0, 4, 'Long-term MSE'),
        (2, 1, 5, 'Long-term MAE'),
        (2, 2, None, 'Long-term Error Separation'),
    ]
    
    for row, col, idx, title in plot_configs:
        ax = axes[row, col]
        
        if idx is not None:
            # 标准误差指标 (MSE/MAE)
            data_plot = [r['means'][idx] for r in all_summary]
            ax.loglog(hs_plot, data_plot, 'ro-', label=title, linewidth=2, markersize=8)
            
            # 添加参考线
            h_ref = np.array([min(hs_plot), max(hs_plot)])
            slope = 4 if 'MSE' in title else 2
            ref_val = (h_ref**slope) * (data_plot[-1] / (hs_plot[-1]**slope))
            ax.loglog(h_ref, ref_val, 'k--', label=f'Ref Slope = {slope}', alpha=0.5)
            
            ax.set_xlabel('Step size h')
            ax.set_ylabel('Error')
            ax.set_title(title)
            ax.grid(True, which="both", ls="-", alpha=0.2)
            ax.legend()
        else:
            # 误差分解子图 (总误差、网络误差、积分器误差)
            if row == 0:  # 训练集
                net_idx, int_idx = 6, 7
                label_net, label_int = 'Train Network', 'Train Integrator'
            elif row == 1:  # 测试集
                net_idx, int_idx = 8, 9
                label_net, label_int = 'Test Network', 'Test Integrator'
            else:  # 长时
                net_idx, int_idx = 10, 11
                label_net, label_int = 'Long Network', 'Long Integrator'
            
            net_errors = [r['means'][net_idx] for r in all_summary]
            int_errors = [r['means'][int_idx] for r in all_summary]
            total_errors = [net + inte for net, inte in zip(net_errors, int_errors)]
            
            ax.loglog(hs_plot, total_errors, 'ko-', label='Total Error', linewidth=2, markersize=8)
            ax.loglog(hs_plot, net_errors, 'b^--', label=label_net, linewidth=2, markersize=7)
            ax.loglog(hs_plot, int_errors, 'gs--', label=label_int, linewidth=2, markersize=7)
            
            ax.set_xlabel('Step size h')
            ax.set_ylabel('Error')
            ax.set_title(title)
            ax.grid(True, which="both", ls="-", alpha=0.2)
            ax.legend()

    plt.tight_layout()
    plt.savefig('2_diff_h_convergence_analysis_3x3.png', dpi=150)
    print("\nLog-Log plot saved to 2_diff_h_convergence_analysis_3x3.png")

if __name__ == '__main__':
    run_diff_h_experiment()
