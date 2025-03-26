import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib import font_manager, rcParams
from matplotlib.ticker import MultipleLocator

# Set font properties - Times New Roman for English, SimSun (宋体) for Chinese
plt.rcParams['font.family'] = ['Times New Roman', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False  # Ensure minus sign displays correctly

# 定义一个统一的颜色映射，保证所有图中相同方法使用相同颜色
METHOD_COLORS = {
    'CFR-RL': '#1f77b4',      # 蓝色
    'Top-k critical': '#ff7f0e', # 橙色
    'Top-k': '#2ca02c',     # 绿色
    'ECMP': '#d62728'       # 红色
}

def create_performance_plot(csv_file, title, output_filename_prefix, y_label='$\it{PR_U}$', y_lim=(0.2, 1.01)):
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
    fig1 = plt.figure(figsize=(11, 7.5))
    ax1 = fig1.add_subplot(111)
    
    for method in df.columns:
        ax1.plot(range(len(df)), df[method], label=method, linewidth=1.5)
    
    ax1.set_title(f"{title}", fontsize=23)
    ax1.set_xlabel('流量矩阵测试集', fontsize=21)
    ax1.set_ylabel(y_label, fontsize=21)
    ax1.legend(loc='lower right', fontsize=21, frameon=True, edgecolor='black', framealpha=1, fancybox=False)
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
    fig2 = plt.figure(figsize=(11, 7.5))
    ax2 = fig2.add_subplot(111)
    
    sns.boxplot(data=df, ax=ax2, showmeans=True, showfliers=False, meanprops={"marker":"o", 
                                                         "markerfacecolor":"white", 
                                                         "markeredgecolor":"black",
                                                         "markersize":"8"})
    ax2.set_title(f"{title}", fontsize=23)
    ax2.set_ylabel(y_label, fontsize=21)
    ax2.set_ylim(y_lim)
    ax2.tick_params(axis='both', labelsize=21)
    
    # Add text annotations with mean and std
    # for i, method in enumerate(df.columns):
    #     mean, std = method_stats[method]
    #     ax2.text(i, y_lim[0] + 0.15, f"Mean: {mean:.4f}\nStd: {std:.4f}",
    #             ha='center', va='center', fontsize=21, 
    #             bbox=dict(facecolor='white', alpha=0.8))

    
    box_plot_filename = f"{output_filename_prefix}_box.png"
    plt.tight_layout()
    plt.savefig(box_plot_filename, dpi=600)
    print(f"Box plot created and saved as '{box_plot_filename}'")
    
    return fig1, fig2

def create_day_performance_plot(csv_file, title, output_filename_prefix, y_label='$\it{PR_U}$', y_lim=(0.2, 1.01)):
    """
    Create bar chart visualization of method performance grouped by days.
    
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
    
    # Create bar plot showing mean values with error bars
    fig = plt.figure(figsize=(12, 7.5))
    ax = fig.add_subplot(111)
    
    # Create pivot for easier plotting
    pivot_df = df.pivot(index='Day', columns='Method', values='Value')
    
    # 指定方法顺序
    method_order = ['CFR-RL', 'Top-k critical', 'Top-k', 'ECMP']
    pivot_df = pivot_df[method_order]
    
    # 同样对误差数据应用相同的顺序
    std_pivot = df.pivot(index='Day', columns='Method', values='Std')
    std_pivot = std_pivot[method_order]
    
    # 准备颜色列表，按方法顺序
    colors = [METHOD_COLORS[method] for method in method_order]
    
    # Create bar plot with specified colors
    pivot_df.plot(kind='bar', yerr=std_pivot, 
                ax=ax, capsize=4, width=0.7, figsize=(12, 7.5),
                color=colors)
    
    # 移除标题和X轴标签
    # ax.set_title(f"{title}", fontsize=23)
    ax.set_ylabel(y_label, fontsize=21)
    ax.set_xlabel('测试集时间', fontsize=21)
    ax.set_ylim(y_lim)
    ax.tick_params(axis='both', labelsize=21)
    
    # 设置y轴刻度间隔为0.2，范围是（0，1）
    # Set y-axis range from 0 to 1 with steps of 0.2
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    
    # 将x轴标签水平显示，而不是竖直显示
    plt.setp(ax.get_xticklabels(), rotation=0, ha='center')
    
    ax.legend(fontsize=21, frameon=True, edgecolor='black', framealpha=1, fancybox=False)
    
    bar_plot_filename = f"{output_filename_prefix}_bar.png"
    plt.tight_layout()
    plt.savefig(bar_plot_filename, dpi=600)
    print(f"Day bar plot created and saved as '{bar_plot_filename}'")
    
    return fig

if __name__ == "__main__":
    # Generate plots for Abilene pru data
    abilene_pru_line_fig, abilene_pru_box_fig = create_performance_plot(
        'abilene_methods_pru_data.csv',
        'Abilene网络',
        'Abilene_PRU'
    )
    
    # Generate plots for Abilene promega data
    abilene_promega_line_fig, abilene_promega_box_fig = create_performance_plot(
        'abilene_methods_promega_data.csv',
        'Abilene网络',
        'Abilene_Promega',
        y_label='$\it{PR_\omega}$'
    )

    # Generate plots for Mininet pru data
    mininet_pru_line_fig, mininet_pru_box_fig = create_performance_plot(
        'mininet_methods_pru_data.csv',
        'Mininet仿真网络',
        'Mininet_PRU'
    )

    # Generate plots for Mininet promega data
    mininet_promega_line_fig, mininet_promega_box_fig = create_performance_plot(
        'mininet_methods_promega_data.csv',
        'Mininet仿真网络',
        'Mininet_Promega',
        y_label='$\it{PR_\Omega}$'
    )
    
    # Define a function to create combined box plots
    def create_combined_box_plot(csv_files, dataset_names, title, output_filename, y_label='$\it{PR_U}$', y_lim=(0.2, 1.01)):
        """
        Create a combined box plot showing metrics from two different datasets.
        """
        # Load and prepare data from both datasets
        combined_data = []
        
        for csv_file, dataset_name in zip(csv_files, dataset_names):
            df = pd.read_csv(csv_file)
            
            # Reshape the dataframe
            melted_df = df.melt(var_name='Method', value_name='Value')
            melted_df['Dataset'] = dataset_name
            combined_data.append(melted_df)
        
        # Concatenate the dataframes
        combined_df = pd.concat(combined_data, ignore_index=True)
        
        # Create the figure
        fig = plt.figure(figsize=(11, 7.5))
        ax = fig.add_subplot(111)
        
        # Create box plot with dataset on x-axis and method as hue
        sns.boxplot(data=combined_df, x='Dataset', y='Value', hue='Method', ax=ax, 
                    showmeans=True, showfliers=False, meanprops={"marker":"o", 
                                                              "markerfacecolor":"white", 
                                                              "markeredgecolor":"black",
                                                              "markersize":"8"})
        
        # ax.set_title(title, fontsize=23)
        ax.set_ylabel(y_label, fontsize=21)
        ax.set_xlabel('网络数据集', fontsize=21)
        ax.set_ylim(y_lim)
        ax.tick_params(axis='both', labelsize=21)
        
        # Adjust legend
        ax.legend(fontsize=21, frameon=True, edgecolor='black', framealpha=1, fancybox=False)
        
        plt.tight_layout()
        plt.savefig(output_filename, dpi=600)
        print(f"Combined box plot created and saved as '{output_filename}'")
        
        return fig
    
    # Generate combined box plots for PR_U
    combined_pru_box_fig = create_combined_box_plot(
        ['abilene_methods_pru_data.csv', 'mininet_methods_pru_data.csv'],
        ['Abilene网络', 'Mininet仿真网络'],
        '网络PR_U性能对比',
        'Combined_PRU_box.png',
        y_label='$\it{PR_U}$'
    )
    
    # Generate combined box plots for PR_Omega
    combined_promega_box_fig = create_combined_box_plot(
        ['abilene_methods_promega_data.csv', 'mininet_methods_promega_data.csv'],
        ['Abilene网络', 'Mininet仿真网络'],
        '网络PR_Ω性能对比',
        'Combined_Promega_box.png',
        y_label='$\it{PR_\Omega}$'
    )
    
    # Generate plots for daily PR_U performance - 只生成bar图并修改前缀
    day_pru_bar_fig = create_day_performance_plot(
        'fh_good_pr_u_performance_data.csv',
        'PR_U性能按天对比',
        'fh_PRU',
        y_label='$\it{PR_U}$'
    )
    
    # Generate plots for daily PR_Omega performance - 只生成bar图并修改前缀
    day_promega_bar_fig = create_day_performance_plot(
        'fh_good_pr_omega_performance_data.csv',
        'PR_Ω性能按天对比',
        'fh_Promega',
        y_label='$\it{PR_\Omega}$'
    )
    
    plt.show()
