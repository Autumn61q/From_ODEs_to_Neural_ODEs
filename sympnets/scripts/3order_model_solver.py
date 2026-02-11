import numpy as np
import torch
import sys
import os
import matplotlib.pyplot as plt

# Add parent directory to Python path to import learner module BEFORE importing
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import learner as ln
from learner.integrator.hamiltonian import SV
from learner.utils import grad, lazy_property

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# --- Ruth's 3rd-order Symplectic Integrator Coefficients ---
# Reference: Ruth, R. D. (1983). A Canonical Integration Technique.
# IEEE Trans. Nucl. Sci., 30(4), 2669–2671.
RUTH_C = [7.0/24.0, 3.0/4.0, -1.0/24.0]
RUTH_D = [2.0/3.0, -2.0/3.0, 1.0]

class Config:
    """Configuration parameters for 3rd-order model solver experiment."""
    def __init__(self):
        # Random seed for reproducibility
        self.random_seed = 42
        
        # Number of initial conditions for zero-shot evaluation
        self.num_train_test_ics = 10  # ICs for full-shot (train + test1)
        self.num_zero_shot_ics = 10   # ICs for zero-shot (test2)
        
        self.num_runs = 5
        self.iterations = 30000
        
        # Neural network architecture
        self.H_size = [2, 30, 30, 1]
        
        # Training configuration
        self.step_sizes = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3,]
        self.lr = 0.001
        self.criterion = None
        self.optimizer = 'adam'
        
        # Evaluation configuration
        self.T_eval = 50  # Evaluation time window
        self.add_h_feature = True  # Add h as feature to network input

# --- Ruth's 3rd-order Symplectic Integrator ---
class SymplecticSplitting3:
    """Ruth's 3rd-order symplectic integrator (1983).
    
    This is a non-symmetric 3rd-order symplectic method derived by Ronald Ruth.
    It works for separable Hamiltonians H(q,p) = T(p) + V(q).

    Coefficients:
    c1 = 7/24, c2 = 3/4, c3 = -1/24
    d1 = 2/3,  d2 = -2/3, d3 = 1

    Algorithm (Drift-Kick):
    For i in 1..3:
        q <- q + c_i * h * p   (Drift)
        p <- p - d_i * h * ∇V  (Kick)
    """
    def __init__(self, N=1):
        self.N = N
        # Use global Ruth coefficients
        self.c = RUTH_C
        self.d = RUTH_D
        
    def _kinetic_step(self, z, h):
        """Kinetic T-flow (Drift): q' = q + h*p, p' = p"""
        d = z.shape[-1] // 2
        q, p = z[..., :d], z[..., d:]
        q_new = q + h * p
        return torch.cat([q_new, p], dim=-1)
    
    def _potential_step(self, z, h):
        """Potential V-flow (Kick): q' = q, p' = p - h*∇V
           For H = T + V, equations are:
           dq/dt = dH/dp = p
           dp/dt = -dH/dq = -dV/dq
           Here V(q) = -cos(q), so -dV/dq = -sin(q)
        """
        d = z.shape[-1] // 2
        q, p = z[..., :d], z[..., d:]
        # V(q) = -cos(q) => -V'(q) = -sin(q)
        # So p' = p + h * (-sin(q))
        dp = -torch.sin(q) 
        p_new = p + h * dp
        return torch.cat([q, p_new], dim=-1)
    
    def solve(self, z, h):
        """Single step Ruth 3rd-order integrator."""
        h_step = h / self.N
        curr = z.clone() if hasattr(z, 'clone') else z
        
        for _ in range(self.N):
            # 3 stages
            for i in range(3):
                # Drift (c_i * h)
                if self.c[i] != 0:
                    curr = self._kinetic_step(curr, self.c[i] * h_step)
                # Kick (d_i * h)
                if self.d[i] != 0:
                    curr = self._potential_step(curr, self.d[i] * h_step)
        
        return curr

    def flow(self, x, h, steps):
        """Multi-step integration, returns trajectory."""
        X = [x]
        for _ in range(steps):
            X.append(self.solve(X[-1], h))
        dim = x.shape[-1]
        size = len(x.shape)
        shape = [steps + 1, dim] if size == 1 else [-1, steps + 1, dim]
        return torch.cat(X, dim=-1).view(shape)

class ThirdOrderHNN(ln.nn.Algorithm):
    """Hamiltonian Neural Network using Ruth's 3rd-order symplectic integrator.
    
    The criterion uses the true derived coefficients from Ruth (1983) to 
    compute 1-step predictions with O(h^4) local / O(h^3) global error.
    """
    def __init__(self, layers, activation='tanh', initializer='orthogonal'):
        super(ThirdOrderHNN, self).__init__()
        self.H_size = layers
        self.activation = activation
        self.initializer = initializer
        self.ms = self.__init_modules()
        
        # Use global Ruth coefficients
        self.c = RUTH_C
        self.d = RUTH_D

    def __init_modules(self):
        modules = torch.nn.ModuleDict()
        modules['H'] = ln.nn.FNN(self.H_size, self.activation, self.initializer)
        return modules

    def criterion(self, x, y):
        """Use Ruth's 3rd order splitting for 1-step prediction."""
        x_in = x[0]  # shape: [N, 2]
        h = x[1]     # shape: [N, 1]
        h_scalar = h[0, 0]  # extract scalar h
        
        curr = x_in
        
        def _kinetic_step(z, h_val):
            """T-flow for learned Hamiltonian."""
            d = z.shape[-1] // 2
            q, p = z[..., :d], z[..., d:]
            q_new = q + h_val * p
            return torch.cat([q_new, p], dim=-1)
        
        def _potential_step(z, h_val):
            """V-flow: p' = p - h*dH/dq"""
            d = z.shape[-1] // 2
            q, p = z[..., :d], z[..., d:]
            
            # Compute dH/dq for learned Hamiltonian
            z_req = z.detach().requires_grad_(True)
            H_val = self.ms['H'](z_req)
            dH = torch.autograd.grad(H_val.sum(), z_req, create_graph=True)[0]
            # dH = [dH/dq, dH/dp]
            dH_dq = dH[..., :d]
            
            # Hamilton's eq: dp/dt = -dH/dq
            p_new = p - h_val * dH_dq
            return torch.cat([q, p_new], dim=-1)
        
        # Ruth 3-stage composition
        for i in range(3):
            # c_i step (Drift)
            curr = _kinetic_step(curr, self.c[i] * h_scalar)
            # d_i step (Kick)
            curr = _potential_step(curr, self.d[i] * h_scalar)
        
        return torch.nn.functional.mse_loss(curr, y)

    def predict(self, x0, h, steps=1, keepinitx=False, returnnp=False):
        """Predict trajectory using Ruth's 3rd-order symplectic splitting."""
        x0 = self._to_tensor(x0)
        
        # Build learned H-based integrator
        z = x0
        Z = [z] if keepinitx else []
        
        # Helper functions similar to criterion but detached graph where possible for efficiency?
        # Note: predict typically doesn't need graph unless we differentiate through it. 
        # But we need graph for H gradients (dH/dq).
        
        d = z.shape[-1] // 2
        
        def _kinetic_step(z_val, h_val):
            q, p = z_val[..., :d], z_val[..., d:]
            q_new = q + h_val * p
            return torch.cat([q_new, p], dim=-1)
        
        def _potential_step(z_val, h_val):
            q, p = z_val[..., :d], z_val[..., d:]
            z_req = z_val.detach().requires_grad_(True)
            H_val = self.ms['H'](z_req)
            dH = torch.autograd.grad(H_val.sum(), z_req, create_graph=False)[0]
            dH_dq = dH[..., :d]
            p_new = p - h_val * dH_dq
            return torch.cat([q, p_new], dim=-1)
            
        for _ in range(steps):
             # Ruth 3-stage composition
            for i in range(3):
                z = _kinetic_step(z, self.c[i] * h)
                z = _potential_step(z, self.d[i] * h)
            
            Z.append(z)
        
        res = torch.stack(Z, dim=-2) if keepinitx and len(Z) > 1 else (Z[-1] if len(Z) == 1 else torch.stack(Z, dim=-2))
        return res.cpu().detach().numpy() if returnnp else res


class FlexiblePDData(ln.Data):
    """Data class supporting multiple initial conditions for full-shot evaluation."""
    def __init__(self, x0_list, h, train_num, test_num, add_h=True):
        """Initialize with a list of initial conditions.
        
        Args:
            x0_list: List of initial condition arrays, each shape (2,)
            h: Step size
            train_num: Number of steps per trajectory for training
            test_num: Number of steps per trajectory for testing
            add_h: Whether to add h as feature
        """
        super(FlexiblePDData, self).__init__()
        # ground truth Hamiltonian: H = 0.5 * p^2 - cos(q)
        self.H = lambda p, q: 0.5 * p**2 - np.cos(q)
        self.x0_list = x0_list
        self.h = h
        self.train_num = train_num
        self.test_num = test_num
        self.add_h = add_h
        self.__init_data()
        
    @property
    def dim(self):
        return 2
    
    def __generate_flow(self, x0, h, num):
        """Generate flow from a single initial condition."""
        true_solver = SV(None, lambda p, q: (p, np.sin(q)), iterations=1, order=6, N=100)
        X = true_solver.flow(torch.tensor(x0, dtype=torch.double), h, num).numpy()
        
        x, y = X[:-1], X[1:]
        if self.add_h:
            x = [x, self.h * np.ones([x.shape[0], 1])]
        return x, y
    
    def __init_data(self):
        """Concatenate data from all initial conditions."""
        # Scale the trajectory length for each IC
        steps_per_ic_train = max(1, int(self.train_num / len(self.x0_list)))
        steps_per_ic_test = max(1, int(self.test_num / len(self.x0_list)))
        
        X_trains = []
        y_trains = []
        X_tests = []
        y_tests = []
        
        for x0 in self.x0_list:
            X_train, y_train = self.__generate_flow(x0, self.h, steps_per_ic_train)
            if isinstance(X_train, list):
                X_trains.append(X_train[0])
            else:
                X_trains.append(X_train)
            y_trains.append(y_train)
            
            # For test, start from the last point of training trajectory
            last_point = y_train[-1]
            X_test, y_test = self.__generate_flow(last_point, self.h, steps_per_ic_test)
            if isinstance(X_test, list):
                X_tests.append(X_test[0])
            else:
                X_tests.append(X_test)
            y_tests.append(y_test)
        
        # Concatenate all data
        X_train_data = np.vstack(X_trains)
        y_train_data = np.vstack(y_trains)
        X_test_data = np.vstack(X_tests)
        y_test_data = np.vstack(y_tests)
        
        # Add h feature if needed
        if self.add_h:
            h_feature_train = self.h * np.ones([X_train_data.shape[0], 1])
            h_feature_test = self.h * np.ones([X_test_data.shape[0], 1])
            self.X_train = [X_train_data, h_feature_train]
            self.X_test = [X_test_data, h_feature_test]
        else:
            self.X_train = X_train_data
            self.X_test = X_test_data
        
        self.y_train = y_train_data
        self.y_test = y_test_data

class DataHelper:
    """Helper class to uniformly access training and test data."""
    @staticmethod
    def get_y_train(data):
        """Get training target data."""
        return data.y_train_np if hasattr(data, 'y_train_np') else data.y_train
    
    @staticmethod
    def get_y_test(data):
        """Get test target data."""
        return data.y_test_np if hasattr(data, 'y_test_np') else data.y_test
    
    @staticmethod
    def get_X_train(data):
        """Get training input data."""
        return data.X_train[0] if isinstance(data.X_train, list) else data.X_train
    
    @staticmethod
    def get_X_test(data):
        """Get test input data."""
        return data.X_test[0] if isinstance(data.X_test, list) else data.X_test

def calculate_metrics(y_true, y_pred):
    mse = np.mean((y_true - y_pred)**2)
    mae = np.mean(np.abs(y_true - y_pred))
    return mse, mae


def run_diff_h_experiment():
    config = Config()
    device = 'gpu' if torch.cuda.is_available() else 'cpu'
    pytorch_device = 'cuda' if device == 'gpu' else 'cpu'
    print(f"Using device: {pytorch_device}")
    
    # Generate initial conditions with fixed random seed
    np.random.seed(config.random_seed)
    num_total_ics = config.num_train_test_ics + config.num_zero_shot_ics
    
    # Generate random initial conditions
    # Each IC is [q, p] with q in [-π, π] and p in [-2, 2]
    x0_list_full = []
    for i in range(num_total_ics):
        q0 = np.random.uniform(-np.pi, np.pi)
        p0 = np.random.uniform(-2.0, 2.0)
        x0_list_full.append(np.array([q0, p0]))
    
    # Split into full-shot and zero-shot
    x0_list_fullshot = x0_list_full[:config.num_train_test_ics]
    x0_list_zeroshot = x0_list_full[config.num_train_test_ics:]
    
    print(f"Generated {len(x0_list_fullshot)} full-shot ICs and {len(x0_list_zeroshot)} zero-shot ICs")
    
    all_summary = []

    for h in config.step_sizes:
        print(f"\n>>>> Testing h = {h}")
        # Keep total training and testing durations constant
        current_train_num = int(max(10.0 / h, 1))
        current_test_num = int(max(10.0 / h, 1))
        
        run_stats = []
        
        # Physical ground-truth analysis (for long-term comparison)
        steps_eval = int(max(config.T_eval / h, 1))
        true_h_solver = SV(None, lambda p, q: (p, np.sin(q)), iterations=1, order=6, N=100)

        for run in range(config.num_runs):
            print(f"  Run {run+1}/{config.num_runs}...", end=' ', flush=True)
            
            # Generate full-shot data from full-shot ICs
            data_fullshot = FlexiblePDData(x0_list_fullshot, h, current_train_num, current_test_num, 
                                           add_h=config.add_h_feature)
            
            # Generate zero-shot data from zero-shot ICs
            data_zeroshot = FlexiblePDData(x0_list_zeroshot, h, current_train_num, current_test_num,
                                          add_h=config.add_h_feature)
            
            # Use the 3rd-order symplectic HNN
            net = ThirdOrderHNN(config.H_size, activation='tanh')
            
            model_save_dir = f'../models/3rd/h{h:.2f}_run{run+1}'.replace('.', 'p')
            args = {
                'data': data_fullshot, 'net': net, 'criterion': net.criterion, 
                'optimizer': config.optimizer, 'lr': config.lr, 
                'iterations': config.iterations, 'print_every': config.iterations // 10,
                'save': 'best_only', 'callback': None, 'dtype': 'double', 'device': device,
                'model_save_path': model_save_dir
            }
            
            ln.Brain.Init(**args)
            ln.Brain.Run()
            
            # L-BFGS Strong Optimization Stage: Reduce error to 10^-8 below
            print(" Fine-tuning with L-BFGS...")
            ln.Brain.Run(optimizer='LBFGS', lr=1.0, iterations=1000, print_every=200)
            
            ln.Brain.Restore()
            best_model = ln.Brain.Best_model()

            # Evaluation 1: training set 1-step (full-shot)
            X_train = DataHelper.get_X_train(data_fullshot)
            y_train = DataHelper.get_y_train(data_fullshot)
            y_tr_pred = best_model.predict(X_train, data_fullshot.h, steps=1, returnnp=True)
            tr_mse, tr_mae = calculate_metrics(y_train, y_tr_pred)
            
            # Evaluation 2: test set 1-step (full-shot)
            X_test = DataHelper.get_X_test(data_fullshot)
            y_test = DataHelper.get_y_test(data_fullshot)
            y_te_pred = best_model.predict(X_test, data_fullshot.h, steps=1, returnnp=True)
            te_mse, te_mae = calculate_metrics(y_test, y_te_pred)
            
            # Evaluation 3: zero-shot 1-step (unseen ICs)
            X_zeroshot = DataHelper.get_X_train(data_zeroshot)
            y_zeroshot = DataHelper.get_y_train(data_zeroshot)
            y_zero_pred = best_model.predict(X_zeroshot, data_zeroshot.h, steps=1, returnnp=True)
            zero_mse, zero_mae = calculate_metrics(y_zeroshot, y_zero_pred)
            
            # Evaluation 4: long-term prediction on first full-shot IC
            # Use reference physics solver
            x0_long = torch.tensor(x0_list_fullshot[0].reshape(1, -1), dtype=torch.float64, device=pytorch_device)
            
            # Predict using best_model
            flow_pred = best_model.predict(x0_list_fullshot[0], h, steps=steps_eval, keepinitx=True, returnnp=True)
            
            # Ground truth for long-term
            flow_ref = true_h_solver.flow(torch.tensor(x0_list_fullshot[0], dtype=torch.double), h, steps_eval)
            flow_ref_np = flow_ref.cpu().detach().numpy()
            if flow_ref_np.ndim > 2:
                flow_ref_np = flow_ref_np.reshape(-1, 2)
            
            lo_mse, lo_mae = calculate_metrics(flow_ref_np, flow_pred)
            
            # Evaluation 5: long-term prediction on first zero-shot IC
            flow_zero_pred = best_model.predict(x0_list_zeroshot[0], h, steps=steps_eval, keepinitx=True, returnnp=True)
            
            # Ground truth for zero-shot long-term
            flow_zero_ref = true_h_solver.flow(torch.tensor(x0_list_zeroshot[0], dtype=torch.double), h, steps_eval)
            flow_zero_ref_np = flow_zero_ref.cpu().detach().numpy()
            if flow_zero_ref_np.ndim > 2:
                flow_zero_ref_np = flow_zero_ref_np.reshape(-1, 2)
            
            lo_zero_mse, lo_zero_mae = calculate_metrics(flow_zero_ref_np, flow_zero_pred)
            
            # Record only MAE (no MSE)
            run_stats.append([tr_mae, te_mae, zero_mae, lo_mae, lo_zero_mae])
            print(f"Full: TR[{tr_mae:.2e}] TE[{te_mae:.2e}] | Zero: 1-step[{zero_mae:.2e}] Long[{lo_zero_mae:.2e}]")

        stats_array = np.array(run_stats)
        means = np.mean(stats_array, axis=0)
        stds = np.std(stats_array, axis=0)
        all_summary.append({'h': h, 'means': means, 'stds': stds})

    # output summary table (MAE only)
    table_lines = []
    table_lines.append(f"{'h':<6} | {'Tr MAE':<12} | {'Te MAE':<12} | {'Zero MAE':<12} | {'Lo MAE':<12} | {'Lo Zero MAE':<12}")
    table_lines.append("-" * 75)
    for res in all_summary:
        m = res['means']
        line = (f"{res['h']:<6.2f} | {m[0]:.2e} | {m[1]:.2e} | {m[2]:.2e} | {m[3]:.2e} | {m[4]:.2e}")
        table_lines.append(line)
    
    # print to console
    print("\n")
    for line in table_lines:
        print(line)
    
    # save to txt file
    with open('3rd_result.txt', 'w') as f:
        for line in table_lines:
            f.write(line + '\n')
    print("\nResults saved to 3rd_result.txt")

    # plotting: log-log plots of MAE only (2x3 subplots)
    hs_plot = [r['h'] for r in all_summary]
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()
    
    plot_configs = [
        # (idx, title, label)
        (0, 'Train MAE (Full-shot)', 'Train MAE'),
        (1, 'Test MAE (Full-shot)', 'Test MAE'),
        (2, 'Zero-shot 1-step MAE', 'Zero-shot MAE'),
        (3, 'Long-term MAE (Full-shot)', 'Long-term MAE'),
        (4, 'Long-term MAE (Zero-shot)', 'Zero-shot Long MAE'),
    ]
    

    for plot_idx, (idx, title, label) in enumerate(plot_configs):
        ax = axes[plot_idx]
        data_plot = [r['means'][idx] for r in all_summary]
        ax.loglog(hs_plot, data_plot, 'ro-', label=label, linewidth=2, markersize=8)
        
        # add reference line (slope = 3 for 3rd-order MAE convergence)
        h_ref = np.array([min(hs_plot), max(hs_plot)])
        ref_val = (h_ref**3) * (data_plot[-1] / (hs_plot[-1]**3))
        ax.loglog(h_ref, ref_val, 'k--', label='Ref Slope = 3', alpha=0.5)
        
        ax.set_xlabel('Step size h')
        ax.set_ylabel('MAE')
        ax.set_title(title)
        ax.grid(True, which="both", ls="-", alpha=0.2)
        ax.legend()
    
    # Hide last subplot
    axes[5].set_visible(False)

    plt.tight_layout()
    plt.savefig('3_fullshot_zeroshot_mae_convergence.png', dpi=150)
    print("\nLog-Log MAE plot saved to 3_fullshot_zeroshot_mae_convergence.png")

if __name__ == '__main__':
    run_diff_h_experiment()
