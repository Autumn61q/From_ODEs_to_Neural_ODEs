import numpy as np
import matplotlib.pyplot as plt

# Style consistency with project reference files
plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'axes.facecolor': 'white',
    'figure.facecolor': 'white',
    'font.size': 14
})

def main():
    # 1. Generate synthetic data
    t = np.linspace(0, 10, 500)
    t0 = 6.5
    
    # A complex wave for demonstration (Sine + Cosine + minor noise)
    y = np.sin(t) + 0.5 * np.cos(2 * t) + 0.05 * np.random.randn(len(t))
    
    # Split data at t0
    mask_obs = t <= t0
    mask_pred = t >= t0
    
    # 2. Plotting
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Observed data: Blue solid line
    ax.plot(t[mask_obs], y[mask_obs], color='royalblue', linewidth=2, label='Observed (History)')
    
    # Forecasted data: Red dashed line
    ax.plot(t[mask_pred], y[mask_pred], color='crimson', linewidth=2, linestyle='--', label='Forecast (Continuous ODE)')
    
    # Vertical line indicating t0 (Reference point)
    ax.axvline(x=t0, color='gray', linestyle=':', alpha=0.7, label='Prediction Start ($t_0$)')
    
    # Labels and Title
    ax.set_xlabel('Time', fontsize=18)
    ax.set_ylabel('Value', fontsize=18)
    ax.set_title('Neural ODE for Time Series Forecasting', fontsize=18, fontweight='bold')
    
    # Legend and Grid
    ax.legend(loc='lower left', fontsize=14, frameon=True, framealpha=0.9)
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
    
    # Clean up formatting
    plt.tight_layout()
    
    save_name = 'time_series_forecasting.png'
    plt.savefig(save_name, bbox_inches='tight')
    print(f"Successfully saved forecasting plot to {save_name}")
    plt.show()

if __name__ == "__main__":
    main()
