import numpy as np
import torch
import learner as ln
from learner.integrator.hamiltonian import SV
from learner.utils import grad, lazy_property
from pendulum import PDData
import matplotlib.pyplot as plt
import os

# 环境配置
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class FourthOrderHNN(ln.nn.Algorithm):
    def __init__(self, layers, activation='tanh', initializer='orthogonal'):
        super(FourthOrderHNN, self).__init__()
        self.H_size = layers
        self.activation = activation
        self.initializer = initializer
        self.ms = self.__init_modules()

    @lazy_property
    def J(self):
        d = int(self.H_size[0] / 2)
        res = np.eye(self.H_size[0], k=d) - np.eye(self.H_size[0], k=-d)
        return torch.tensor(res, dtype=self.dtype, device=self.device)

    def __init_modules(self):
        modules = torch.nn.ModuleDict()
        modules['H'] = ln.nn.FNN(self.H_size, self.activation, self.initializer)
        return modules

    def criterion(self, x, y):
        x_in = x[0]
        h = x[1]
        x_in = x_in.requires_grad_(True)  # 确保可以计算梯度
        # 使用 SV4 积分器计算 y_pred
        solver = SV(self.ms['H'], None, iterations=10, order=4, N=1)
        y_pred = solver.solve(x_in, h)
        return torch.nn.functional.mse_loss(y_pred, y)

    def predict(self, x0, h, steps=1, keepinitx=False, returnnp=False):
        x0 = self._to_tensor(x0)
        # 使用 4 阶辛积分器 SV4，N=1 以保证实验的步长就是 h
        solver = SV(self.ms['H'], None, iterations=10, order=4, N=1)
        res = solver.flow(x0, h, steps) if keepinitx else solver.flow(x0, h, steps)[..., 1:, :].squeeze()
        return res.cpu().detach().numpy() if returnnp else res

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

def run_diff_h_experiment():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # 固定参数
    x0 = np.array([0.0, 1.0])
    train_num = 20
    test_num = 100
    num_runs = 15
    iterations = 30000
    H_size = [2, 30, 30, 1]
    
    # 变化的步长 h (训练和推理保持一致)
    # hs = [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01]
    hs = [0.5, 0.4, 0.3, 0.2, 0.1]
    
    all_summary = []

    for h in hs:
        print(f"\n>>>> Testing h = {h} (Consistency Mode)")
        # 保持训练和测试的总时长恒定
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
            # 自动清理过往模型缓存
            if os.path.exists('model'):
                import shutil
                shutil.rmtree('model')
                
            print(f"  Run {run+1}/{num_runs}...", end=' ', flush=True)
            
            # 数据生成
            data = FlexiblePDData(x0, h, current_train_num, current_test_num, add_h=True)
            
            # 使用自定义的 4 阶 RK4 HNN
            net = FourthOrderHNN(H_size, activation='tanh')
            
            args = {
                'data': data, 'net': net, 'criterion': net.criterion, 'optimizer': 'adam',
                'lr': 0.001, 'iterations': iterations, 'print_every': iterations,
                'save': 'best_only', 'device': device, 'dtype': 'double'
            }
            
            ln.Brain.Init(**args)
            ln.Brain.Run()
            
            # L-BFGS 强力优化阶段：将误差压制到 10^-8 以下
            print("  Fine-tuning with L-BFGS...")
            ln.Brain.Run(optimizer='LBFGS', lr=1.0, iterations=1000, print_every=200)
            
            ln.Brain.Restore()
            best_model = ln.Brain.Best_model()

            # 评估 1: 训练集 1-step
            y_tr_pred = best_model.predict(data.X_train[0], data.h, steps=1, returnnp=True)
            tr_mse, tr_mae = calculate_metrics(data.y_train_np if hasattr(data, 'y_train_np') else data.y_train, y_tr_pred)
            
            # 评估 2: 测试集 1-step
            y_te_pred = best_model.predict(data.X_test[0], data.h, steps=1, returnnp=True)
            te_mse, te_mae = calculate_metrics(data.y_test_np if hasattr(data, 'y_test_np') else data.y_test, y_te_pred)

            # 评估 3: 长时预测 (h 保持完全一致)
            # 使用 best_model.predict，它内部已封装了积分器逻辑，与 hnn.py 风格对齐
            flow_pred = best_model.predict(x0, h, steps=steps_eval, keepinitx=True, returnnp=True)
            
            lo_mse, lo_mae = calculate_metrics(flow_ref, flow_pred)
            
            run_stats.append([tr_mse, tr_mae, te_mse, te_mae, lo_mse, lo_mae])
            print(f"Long MSE: {lo_mse:.2e}")

        stats_array = np.array(run_stats)
        means = np.mean(stats_array, axis=0)
        stds = np.std(stats_array, axis=0)
        all_summary.append({'h': h, 'means': means, 'stds': stds})

    # 输出汇总表格
    print("\n" + "="*110)
    print(f"{'Step size h':<12} | {'Tr MSE':<10} | {'Tr MAE':<10} | {'Te MSE':<10} | {'Te MAE':<10} | {'Lo MSE':<10} | {'Lo MAE'}")
    print("-" * 110)
    for res in all_summary:
        m = res['means']
        print(f"{res['h']:<12.3f} | {m[0]:.2e} | {m[1]:.2e} | {m[2]:.2e} | {m[3]:.2e} | {m[4]:.2e} | {m[5]:.2e}")
    print("="*110)

    # 绘图：h 与各类 Error 的 Log-Log 图 (3x2 子图)
    hs_plot = [r['h'] for r in all_summary]
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 18))
    metrics_names = ['Train MSE', 'Train MAE', 'Test MSE', 'Test MAE', 'Long-term MSE', 'Long-term MAE']
    
    for i in range(3):
        for j in range(2):
            ax = axes[i, j]
            idx = i * 2 + j
            data_plot = [r['means'][idx] for r in all_summary]
            
            ax.loglog(hs_plot, data_plot, 'ro-', label=metrics_names[idx])
            
            # 添加参考线
            h_ref = np.array([min(hs_plot), max(hs_plot)])
            # 估算阶数：MSE ~ h^8, MAE ~ h^4 (对于 4 阶方法)
            slope = 8 if 'MSE' in metrics_names[idx] else 4
            ref_val = (h_ref**slope) * (data_plot[-1] / (hs_plot[-1]**slope))
            ax.loglog(h_ref, ref_val, 'k--', label=f'Ref Slope = {slope}')
            
            ax.set_xlabel('Step size h')
            ax.set_ylabel('Error')
            ax.set_title(f'{metrics_names[idx]} vs Step Size')
            ax.grid(True, which="both", ls="-", alpha=0.2)
            ax.legend()

    plt.tight_layout()
    plt.savefig('4_diff_h_consistency_analysis.png')
    print("\nLog-Log plot saved to 4_diff_h_consistency_analysis.png")

if __name__ == '__main__':
    run_diff_h_experiment()
