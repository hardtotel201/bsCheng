import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib import font_manager, rcParams

# Set font properties - Times New Roman for English, SimSun (宋体) for Chinese
plt.rcParams['font.family'] = ['Times New Roman', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False  # Ensure minus sign displays correctly

def create_performance_plot(csv_file, title, output_filename_prefix, y_label=r'PR$_U$', y_lim=(0.2, 1.01)):
    """
    Create separate visualizations of method performance.
    
    Parameters:
    -----------
    csv_file : str
        Path to the CSV file containing the performance data
    title : str
        Title prefix for the plots
    output_filename_prefix : str
        Filename prefix to save the plots
    y_label : str
        Label for y-axis
    y_lim : tuple
        Y-axis limits (min, max)
    """
    # Load the data
    df = pd.read_csv(csv_file)
    
    # Create separate figures
    
    # Figure 1: Line chart
    fig1 = plt.figure(figsize=(16, 7.5))
    ax1 = fig1.add_subplot(111)
    
    for method in df.columns:
        ax1.plot(range(len(df)), df[method], label=method, linewidth=1.5)
    
    ax1.set_title(f"{title} - 时间序列", fontsize=23)
    ax1.set_xlabel('流量矩阵测试集', fontsize=21)
    ax1.set_ylabel(y_label, fontsize=21)
    ax1.legend(loc='lower right', fontsize=21)
    ax1.set_ylim(y_lim)
    ax1.tick_params(axis='both', labelsize=21)
    
    line_plot_filename = f"{output_filename_prefix}_line.png"
    plt.tight_layout()
    plt.savefig(line_plot_filename, dpi=600)
    print(f"Line plot created and saved as '{line_plot_filename}'")
    
    # Calculate statistics for the box plot
    method_stats = {method: [np.mean(df[method]), np.std(df[method])] 
                    for method in df.columns}
    
    # Figure 2: Box plot
    fig2 = plt.figure(figsize=(16, 7.5))
    ax2 = fig2.add_subplot(111)
    
    sns.boxplot(data=df, ax=ax2, showmeans=True, meanprops={"marker":"o", 
                                                         "markerfacecolor":"white", 
                                                         "markeredgecolor":"black",
                                                         "markersize":"8"})
    ax2.set_title(f"{title} - 统计分布", fontsize=23)
    ax2.set_ylabel(y_label, fontsize=21)
    ax2.set_ylim(y_lim)
    ax2.tick_params(axis='both', labelsize=21)
    
    # Add text annotations with mean and std
    for i, method in enumerate(df.columns):
        mean, std = method_stats[method]
        ax2.text(i, y_lim[0] + 0.15, f"Mean: {mean:.4f}\nStd: {std:.4f}",
                ha='center', va='center', fontsize=21, 
                bbox=dict(facecolor='white', alpha=0.8))
    
    box_plot_filename = f"{output_filename_prefix}_box.png"
    plt.tight_layout()
    plt.savefig(box_plot_filename, dpi=600)
    print(f"Box plot created and saved as '{box_plot_filename}'")
    
    return fig1, fig2

if __name__ == "__main__":
    # Generate plots for Abilene data
    abilene_line_fig, abilene_box_fig = create_performance_plot(
        'abilene_methods_data.csv',
        'Abilene网络',
        'Abilene_PRU'
    )
    
    # Generate plots for Mininet data
    mininet_line_fig, mininet_box_fig = create_performance_plot(
        'mininet_methods_data.csv',
        'Mininet网络',
        'Mininet_PRU'
    )
    
    plt.show()
