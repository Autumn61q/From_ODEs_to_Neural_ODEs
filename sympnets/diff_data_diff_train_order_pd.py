import numpy as np
import torch
import learner as ln
from pendulum import PDData
import os

# 环境变量解决 OpenMP 冲突
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 1. 各阶损失函数定义 (与之前脚本保持一致)
class HNN_Order1(ln.nn.HNN):
    def criterion(self, x0h, x1):
        x0, h = x0h[0].detach().requires_grad_(True), x0h[1]
        gradH = ln.utils.grad(self.ms['H'](x0), x0)
        return torch.nn.MSELoss()((x1 - x0) / h, gradH @ self.J)

class HNN_GL4(ln.nn.HNN):
    def criterion(self, x0h, x1):
        x0, h = x0h[0], x0h[1]
        c = [0.5 - np.sqrt(3)/6, 0.5 + np.sqrt(3)/6]
        v_gl = 0
        for ci in c:
            node = ((1 - ci) * x0 + ci * x1).detach().requires_grad_(True)
            v_gl += 0.5 * (ln.utils.grad(self.ms['H'](node), node) @ self.J)
        return torch.nn.MSELoss()((x1 - x0) / h, v_gl)

class HNN_GL6(ln.nn.HNN):
    def criterion(self, x0h, x1):
        x0, h = x0h[0], x0h[1]
        c = [0.5 - np.sqrt(15)/10, 0.5, 0.5 + np.sqrt(15)/10]
        w = [5/18, 8/18, 5/18]
        v_gl = 0
        for i in range(3):
            node = ((1 - c[i]) * x0 + c[i] * x1).detach().requires_grad_(True)
            v_gl += w[i] * (ln.utils.grad(self.ms['H'](node), node) @ self.J)
        return torch.nn.MSELoss()((x1 - x0) / h, v_gl)

# 2. 支持自定义阶数的数据生成类
class FlexiblePDData(PDData):
    def __init__(self, x0, h, train_num, test_num, order=4, **kwargs):
        self.order = order
        super().__init__(x0, h, train_num, test_num, **kwargs)
    
    def solver_init(self):
        # 覆写父类方法，根据 order 选择积分器
        return ln.integrator.hamiltonian.SV(None, self.dH, iterations=10, order=self.order, N=1)

def calculate_metrics(y_true, y_pred):
    mse = np.mean((y_true - y_pred)**2)
    mae = np.mean(np.abs(y_true - y_pred))
    return mse, mae

def run_combined_experiment():
    import shutil
    if os.path.exists('model'):
        shutil.rmtree('model')
        
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    x0, h = [0, 1], 0.1
    train_num, test_num = 40, 100
    num_runs = 5
    
    # 网络参数
    H_size = [2, 30, 30, 1]
    
    # 实验配置：(数据阶数, 模型类)
    configs = [
        (1, HNN_Order1),
        (2, ln.nn.HNN),
        (4, HNN_GL4),
        (6, HNN_GL6)
    ]
    
    all_results = []
    
    # 建立一个最高精度真值生成器 (Order 6 + N=50)
    data_ref = PDData(x0, h, 1, 1) 
    ref_solver = ln.integrator.hamiltonian.SV(None, data_ref.dH, iterations=10, order=6, N=50)

    for order, model_class in configs:
        print(f"\n>>>> Starting Experiment: Order {order} Data -> Order {order} Model")
        run_stats = []
        
        # 生成对应阶数的训练/短时测试数据
        data = FlexiblePDData(x0, h, train_num, test_num, order=order, add_h=True)

        for run in range(num_runs):
            print(f"  Run {run+1}/{num_runs}...", end='\r')
            net = model_class(H_size, activation='tanh').to(device)
            ln.Brain.Init(data=data, net=net, iterations=40000, lr=0.001, print_every=4000, save='best_only', device=device)
            ln.Brain.Run()
            ln.Brain.Restore()
            best_model = ln.Brain.Best_model()

            # 评估 Train (1-step)
            y_tr_pred = best_model.predict(data.X_train[0], data.h, steps=1, returnnp=True)
            tr_mse, tr_mae = calculate_metrics(data.y_train.detach().cpu().numpy(), y_tr_pred)
            
            # 评估 Test (1-step)
            y_te_pred = best_model.predict(data.X_test[0], data.h, steps=1, returnnp=True)
            te_mse, te_mae = calculate_metrics(data.y_test.detach().cpu().numpy(), y_te_pred)
            
            # 评估 Long-term (1000-step) 统一对比 Order 6 真值
            flow_true = ref_solver.flow(data.X_test_np[0][0], data.h, 1000)
            # 显式指定 predict 在 GPU 上运行
            flow_pred = best_model.predict(data.X_test[0][0], data.h, 1000, keepinitx=True, returnnp=True)
            lo_mse, lo_mae = calculate_metrics(flow_true, flow_pred)
            
            run_stats.append([tr_mse, tr_mae, te_mse, te_mae, lo_mse, lo_mae])
            
        means = np.mean(run_stats, axis=0)
        all_results.append({'order': order, 'means': means})

    # 打印最终汇总表格
    print("\n\n" + "="*110)
    print(f"{'Order':<6} | {'Tr MSE':<9} | {'Tr MAE':<9} | {'Te MSE':<9} | {'Te MAE':<9} | {'Long MSE':<9} | {'Long MAE':<9}")
    print("-" * 110)
    for r in all_results:
        m = r['means']
        print(f"Ord {r['order']:<2} | {m[0]:.2e} | {m[1]:.2e} | {m[2]:.2e} | {m[3]:.2e} | {m[4]:.2e} | {m[5]:.2e}")
    print("="*110)

if __name__ == '__main__':
    run_combined_experiment()