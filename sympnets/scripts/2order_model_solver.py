import numpy as np
import torch
import sys
import os
import matplotlib.pyplot as plt

# Add parent directory to Python path to import learner module BEFORE importing
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import learner as ln
from learner.integrator.hamiltonian import SV

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class Config:
    """Configuration parameters for 2nd-order model solver experiment."""
    def __init__(self):
        # Initial condition and simulation
        self.x0 = np.array([0.0, 1.0])
        self.num_runs = 5
        self.iterations = 30000
        
        # Neural network architecture
        self.H_size = [2, 30, 30, 1]
        
        # Training configuration
        # self.step_sizes = [0.2, 0.1,]
        self.step_sizes = [0.5, 0.4, 0.3, 0.2, 0.1]
        self.lr = 0.001
        self.criterion = None
        self.optimizer = 'adam'
        
        # Evaluation configuration
        self.T_eval = 0.5  # Evaluation time window
        self.add_h_feature = True  # Add h as feature to network input

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
    
    # generate the ground-truth data using a 6th-order symplectic integrator (SV6)
    def __generate_flow(self, x0, h, num):
        true_solver = SV(None, lambda p,q: (p, np.sin(q)), iterations=1, order=6, N=100)
        X = true_solver.flow(torch.tensor(x0, dtype=torch.double), h, num).numpy()
        
        x, y = X[:-1], X[1:]
        if self.add_h:
            x = [x, self.h * np.ones([x.shape[0], 1])]
        return x, y
    
    def __init_data(self):
        self.X_train, self.y_train = self.__generate_flow(self.x0, self.h, self.train_num)
        self.X_test, self.y_test = self.__generate_flow(self.y_train[-1], self.h, self.test_num)

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

def analyze_error_separation(best_model, x0, h, y_true_data=None, true_solver=None, 
                              target_time=None, use_richardson=True, mode='short_term'):
    """
    Use Richardson extrapolation to separate network and integrator errors.
    
    Parameters:
        - best_model: trained model
        - x0: initial condition (single point or batch)
        - h: step size
        - y_true_data: ground-truth data for short-term analysis
        - true_solver: high-precision reference solver
        - target_time: total evaluation time (for long-term analysis)
        - use_richardson: whether to use Richardson extrapolation
        - mode: 'short_term' (1-step prediction) or 'long_term' (multi-step evolution)
    
    Returns:
        dict with keys: y_h, y_network, y_true, network_error, integrator_error, total_error
    """
    if mode == 'long_term' and (target_time is None or true_solver is None):
        raise ValueError("target_time and true_solver required for long_term mode")
    
    # Richardson extrapolation: keep the same final time (= h) but use different step sizes
    # predict 1 step with h, 2 steps with h/2, and 4 steps with h/4
    hs_configs = [
        (h, 1),      # h, 1 step
        (h/2, 2),    # h/2, 2 steps
        (h/4, 4)     # h/4, 4 steps
    ]
    
    predictions = []
    for h_step, num_steps in hs_configs:
        pred = best_model.predict(x0, h_step, steps=num_steps, returnnp=True)
        # Only take the final step's result (last row)
        # Ensure we always get a 1D array for a single point prediction
        if isinstance(pred, np.ndarray) and pred.ndim >= 2:
            predictions.append(pred[-1])  # Take last row
        else:
            predictions.append(pred)
    
    y_h, y_h2, y_h4 = predictions
    
    # 2-point Richardson extrapolation: remove O(h^2) integrator error (2nd-order method)
    # For a 2nd-order method the global error scales as O(h^2), with ratios 1:1/4:1/16
    # Richardson formula: y_network = (4 * y_h4 - y_h2) / 3
    y_network = (4 * y_h4 - y_h2) / 3
    
    # calculate errors
    network_error = np.mean(np.abs(y_network - y_true_data))
    integrator_error_h = np.mean(np.abs(y_h - y_network))
    total_error = np.mean(np.abs(y_h - y_true_data))
    
    return {
        'y_h': y_h,
        'y_network': y_network,
        'y_true': y_true_data,
        'network_error': network_error,
        'integrator_error': integrator_error_h,
        'total_error': total_error
    }

def run_diff_h_experiment():
    config = Config()
    device = 'gpu' if torch.cuda.is_available() else 'cpu'
    pytorch_device = 'cuda' if device == 'gpu' else 'cpu'
    print(f"Using device: {pytorch_device}")
    
    all_summary = []

    for h in config.step_sizes:
        print(f"\n>>>> Testing h = {h} (Consistency Mode)")
        # Keep total training and testing durations constant to avoid mechanism shifts
        current_train_num = int(max(10.0 / h, 1))
        current_test_num = int(max(10.0 / h, 1))
        
        run_stats = []
        
        # Physical ground-truth analysis (for long-term comparison)
        steps_eval = int(max(config.T_eval / h, 1))
        true_h_solver = SV(None, lambda p, q: (p, np.sin(q)), iterations=1, order=6, N=100)
        flow_ref = true_h_solver.flow(torch.tensor(config.x0, dtype=torch.double), h, steps_eval)

        for run in range(config.num_runs):
            print(f"  Run {run+1}/{config.num_runs}...", end=' ', flush=True)
            
            # Generate data
            data = FlexiblePDData(config.x0, h, current_train_num, current_test_num, 
                                  add_h=config.add_h_feature)
            
            # Use the standard 2nd-order midpoint HNN
            net = ln.nn.HNN(config.H_size, activation='tanh')
            
            model_save_dir = f'../models/2nd/h{h:.2f}_run{run+1}'.replace('.', 'p')
            args = {
                'data': data, 'net': net, 'criterion': config.criterion, 
                'optimizer': config.optimizer, 'lr': config.lr, 
                'iterations': config.iterations, 'print_every': config.iterations,
                'save': 'best_only', 'callback': None, 'dtype': 'double', 'device': device,
                'model_save_path': model_save_dir
            }
            
            ln.Brain.Init(**args)
            ln.Brain.Run()
            ln.Brain.Restore()
            best_model = ln.Brain.Best_model()

            # Evaluation 1: training set 1-step
            X_train = DataHelper.get_X_train(data)
            y_train = DataHelper.get_y_train(data)
            y_tr_pred = best_model.predict(X_train, data.h, steps=1, returnnp=True)
            tr_mse, tr_mae = calculate_metrics(y_train, y_tr_pred)
            
            # Richardson extrapolation to separate short-term errors on training set
            # true_h_solver = SV(None, lambda p, q: (p, np.sin(q)), iterations=1, order=6, N=100)
            # tr_error_sep = analyze_error_separation(best_model, X_train, h, 
            #                                         y_true_data=y_train, 
            #                                         true_solver=true_h_solver, mode='short_term')
            tr_error_sep = {'network_error': 0.0, 'integrator_error': 0.0}
            
            # Evaluation 2: test set 1-step
            X_test = DataHelper.get_X_test(data)
            y_test = DataHelper.get_y_test(data)
            y_te_pred = best_model.predict(X_test, data.h, steps=1, returnnp=True)
            te_mse, te_mae = calculate_metrics(y_test, y_te_pred)
            
            # Richardson extrapolation to separate short-term errors on test set
            # te_error_sep = analyze_error_separation(best_model, X_test, h,
            #                                         y_true_data=y_test,
            #                                         true_solver=true_h_solver, mode='short_term')
            te_error_sep = {'network_error': 0.0, 'integrator_error': 0.0}

            # Evaluation 3: long-term prediction (h is kept consistent; do not use predict's internal N subdivisions)
            # use SV solver with order=2 and N=1 so the solver step equals h
            custom_solver = SV(best_model.ms['H'], None, iterations=1, order=2, N=1)
            x0_tensor = torch.tensor(config.x0.reshape(1, -1), dtype=torch.float64, device=pytorch_device)
            flow_pred = custom_solver.flow(x0_tensor, h, steps_eval).cpu().detach().numpy().reshape(-1, 2)
            
            lo_mse, lo_mae = calculate_metrics(flow_ref.cpu().detach().numpy().reshape(-1, 2), flow_pred)
            
            # Evaluation 4: Richardson extrapolation to separate network and integrator errors on long-term
            # true_h_solver = SV(None, lambda p, q: (p, np.sin(q)), iterations=1, order=6, N=100)
            # # For long-term analysis: use the final step of the long-term evolution as true reference
            # error_sep = analyze_error_separation(best_model, config.x0, h, 
            #                                     y_true_data=flow_ref.cpu().detach().numpy()[-1].reshape(-1),
            #                                     true_solver=true_h_solver, mode='short_term')
            error_sep = {'network_error': 0.0, 'integrator_error': 0.0}
            
            run_stats.append([tr_mse, tr_mae, te_mse, te_mae, lo_mse, lo_mae])
            print(f"Tr(MSE:{tr_mse:.2e}, MAE:{tr_mae:.2e}) | "
                  f"Te(MSE:{te_mse:.2e}, MAE:{te_mae:.2e}) | "
                  f"Lo(MSE:{lo_mse:.2e}, MAE:{lo_mae:.2e})")

        stats_array = np.array(run_stats)
        means = np.mean(stats_array, axis=0)
        stds = np.std(stats_array, axis=0)
        all_summary.append({'h': h, 'means': means, 'stds': stds})

    # output summary table
    table_lines = []
    table_lines.append(f"{'h':<6} | {'Tr MSE':<10} | {'Tr MAE':<10} | {'Te MSE':<10} | {'Te MAE':<10} | {'Lo MSE':<10} | {'Lo MAE':<10}")
    table_lines.append("-" * 84)
    for res in all_summary:
        m = res['means']
        line = (f"{res['h']:<6.2f} | {m[0]:.2e} | {m[1]:.2e} | {m[2]:.2e} | {m[3]:.2e} | {m[4]:.2e} | {m[5]:.2e}")
        table_lines.append(line)
    
    # print to console
    print("\n")
    for line in table_lines:
        print(line)
    
    # save to txt file
    with open('2nd_result.txt', 'w') as f:
        for line in table_lines:
            f.write(line + '\n')
    print("\nResults saved to 2nd_result.txt")

    # plotting: log-log plots of h vs various errors (2x3 subplots)
    hs_plot = [r['h'] for r in all_summary]
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    
    plot_configs = [
        # (row, col, idx, title)
        (0, 0, 0, 'Train MSE'),
        (0, 1, 1, 'Train MAE'),
        (0, 2, 2, 'Test MSE'),
        (1, 0, 3, 'Test MAE'),
        (1, 1, 4, 'Long-term MSE'),
        (1, 2, 5, 'Long-term MAE'),
    ]
    
    for row, col, idx, title in plot_configs:
        ax = axes[row, col]
        
        # standard error metrics (MSE/MAE)
        data_plot = [r['means'][idx] for r in all_summary]
        ax.loglog(hs_plot, data_plot, 'ro-', label=title, linewidth=2, markersize=8)
        
        # add reference line
        h_ref = np.array([min(hs_plot), max(hs_plot)])
        slope = 4 if 'MSE' in title else 2
        ref_val = (h_ref**slope) * (data_plot[-1] / (hs_plot[-1]**slope))
        ax.loglog(h_ref, ref_val, 'k--', label=f'Ref Slope = {slope}', alpha=0.5)
        
        ax.set_xlabel('Step size h')
        ax.set_ylabel('Error')
        ax.set_title(title)
        ax.grid(True, which="both", ls="-", alpha=0.2)
        ax.legend()

    plt.tight_layout()
    plt.savefig('2_diff_h_convergence_analysis_2x3.png', dpi=150)
    print("\nLog-Log plot saved to 2_diff_h_convergence_analysis_2x3.png")

if __name__ == '__main__':
    run_diff_h_experiment()
