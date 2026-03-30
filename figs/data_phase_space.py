import numpy as np
import torch
import sys
import os
import matplotlib.pyplot as plt

figs_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(figs_dir)
sympnets_dir = os.path.join(project_root, 'sympnets')
sys.path.insert(0, sympnets_dir)

import learner as ln
from learner.integrator.hamiltonian import SV

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Style consistency with project reference files
plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'axes.facecolor': 'white',
    'figure.facecolor': 'white',
    'font.size': 14
})


class Config:
    """Configuration parameters for training data visualization."""
    def __init__(self):
        # Random seed for reproducibility
        self.random_seed = 42
        
        # Number of initial conditions
        self.num_train_test_ics = 10   # ICs for full-shot (train + test)
        self.num_zero_shot_ics = 10    # ICs for zero-shot (test only, unseen during training)
        
        # Training configuration
        self.h = 0.3  # Single step size
        self.add_h_feature = True  # Add h as feature to network input


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
    
    # generate the ground-truth data using a 6th-order symplectic integrator (SV6)
    def __generate_flow(self, x0, h, num):
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


def visualize_training_data():
    """Visualize training, test, and zero-shot data for phase space and time series."""
    config = Config()
    
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
    
    # ============= Phase Space Plot =============
    h = config.h
    current_train_num = int(max(10.0 / h, 1))
    current_test_num = int(max(10.0 / h, 1))
    
    # Full-shot data (train + test)
    data_fullshot = FlexiblePDData(x0_list_fullshot, h, current_train_num, current_test_num, 
                                  add_h=config.add_h_feature)
    
    # Zero-shot data (unseen ICs)
    data_zeroshot = FlexiblePDData(x0_list_zeroshot, h, current_train_num, current_test_num,
                                  add_h=config.add_h_feature)
    
    # Get data
    X_train = data_fullshot.X_train[0] if isinstance(data_fullshot.X_train, list) else data_fullshot.X_train
    X_test = data_fullshot.X_test[0] if isinstance(data_fullshot.X_test, list) else data_fullshot.X_test
    X_zeroshot = data_zeroshot.X_train[0] if isinstance(data_zeroshot.X_train, list) else data_zeroshot.X_train
    
    # Plot the phase space trajectory
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(X_train[:, 0], X_train[:, 1], c='blue', s=50, alpha=0.8, label='Train (Full-shot)', linewidth=0)
    ax.scatter(X_test[:, 0], X_test[:, 1], c='red', s=50, alpha=0.8, label='Test (Full-shot)', linewidth=0)
    ax.scatter(X_zeroshot[:, 0], X_zeroshot[:, 1], c='green', s=50, alpha=0.8, label='Zero-shot', linewidth=0)
    
    ax.set_xlabel('Position (q)', fontsize=18)
    ax.set_ylabel('Momentum (p)', fontsize=18)
    ax.set_title(f'Phase Space Trajectory (h={h})', fontsize=18, fontweight='bold')
    ax.grid(True, alpha=0.2, linestyle='--')
    ax.legend(fontsize=16, frameon=True, framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig('data_phase_space.png', bbox_inches='tight')
    print("Phase space plot saved to data_phase_space.png")
    plt.close()
    
    # ============= Time Series Plot =============
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Generate time arrays for each data segment
    time_train = np.arange(len(X_train)) * h
    time_test = (np.arange(len(X_test)) + len(X_train)) * h
    time_zeroshot = (np.arange(len(X_zeroshot)) + len(X_train) + len(X_test)) * h
    
    ax.plot(time_train, X_train[:, 0], 'o-', color='royalblue', label='Train (Full-shot)', linewidth=2, markersize=5)
    ax.plot(time_test, X_test[:, 0], '^-', color='darkorange', label='Test (Full-shot)', linewidth=2, markersize=5)
    ax.plot(time_zeroshot, X_zeroshot[:, 0], 's-', color='crimson', label='Zero-shot', linewidth=2, markersize=5)
    
    ax.set_xlabel('Time', fontsize=18)
    ax.set_ylabel('Position (q)', fontsize=18)
    ax.set_title(f'Time Series of Position (h={h})', fontsize=18, fontweight='bold')
    ax.grid(True, alpha=0.2, linestyle='--')
    ax.legend(fontsize=14, frameon=True, framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig('data_time_series.png', bbox_inches='tight')
    print("Time series plot saved to data_time_series.png")
    plt.close()


if __name__ == '__main__':
    visualize_training_data()