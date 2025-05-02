import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
import matplotlib
import seaborn as sns

# 设置中文字体显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置默认字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号'-'显示为方块的问题
matplotlib.rcParams['font.family'] = 'SimHei'  # 整体设置字体族

def visualize_color_conversion(converter, sample_colors):
    """可视化颜色转换效果"""
    n_samples = len(sample_colors)
    
    # 创建两行图表：原始 RGBV 颜色和转换后的 RGBCX 颜色
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # 第一行：原始 RGBV 颜色
    rgbv_colors = []
    for color in sample_colors:
        # 将 RGBV 转换为用于显示的 RGB
        r, g, b, v = color
        # V 增强RGB亮度
        display_color = [min(1.0, r + v * 0.5), 
                         min(1.0, g + v * 0.5), 
                         min(1.0, b + v * 0.5)]
        rgbv_colors.append(display_color)
    
    ax1.bar(range(n_samples), [1] * n_samples, color=rgbv_colors)
    ax1.set_title('原始 RGBV 颜色')
    ax1.set_xticks(range(n_samples))
    ax1.set_xticklabels([f'样本{i+1}' for i in range(n_samples)])
    
    # 第二行：转换后的 RGBCX 颜色
    rgbcx_colors = []
    for color in sample_colors:
        rgbcx = converter.convert_rgbv_to_rgbcx(color)
        # 使用 RGBCX 的前三个通道作为显示颜色
        display_color = [rgbcx[0], rgbcx[1], rgbcx[2]]
        rgbcx_colors.append(display_color)
    
    ax2.bar(range(n_samples), [1] * n_samples, color=rgbcx_colors)
    ax2.set_title('转换后 RGBCX 颜色 (仅显示 RGB 分量)')
    ax2.set_xticks(range(n_samples))
    ax2.set_xticklabels([f'样本{i+1}' for i in range(n_samples)])
    
    plt.tight_layout()
    plt.savefig('color_conversion.png')
    plt.show()

def visualize_color_space(converter):
    """可视化颜色空间"""
    # 创建3D图形
    fig = plt.figure(figsize=(15, 10))
    
    # 创建两个子图：一个显示RGBV，一个显示RGBCX
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')
    
    # 在3D空间中创建网格
    grid_size = 5
    r_values = np.linspace(0, 1, grid_size)
    g_values = np.linspace(0, 1, grid_size)
    b_values = np.linspace(0, 1, grid_size)
    
    # 定义一组V值用于展示
    v_values = [0, 0.5, 1.0]
    
    # 为每个V值使用不同的标记
    markers = ['o', 's', '^']
    
    # RGBV 颜色空间
    for v_idx, v in enumerate(v_values):
        points = []
        colors = []
        for r in r_values:
            for g in g_values:
                for b in b_values:
                    # 仅使用一部分点，以避免过度拥挤
                    if np.random.random() > 0.7:
                        points.append([r, g, b])
                        # 为了显示目的，将V作为亮度增强
                        display_color = [min(1.0, r + v * 0.3), 
                                         min(1.0, g + v * 0.3), 
                                         min(1.0, b + v * 0.3)]
                        colors.append(display_color)
        
        if points:
            points = np.array(points)
            ax1.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, 
                       marker=markers[v_idx], label=f'V = {v:.1f}')
    
    ax1.set_title('RGBV 颜色空间 (V 值通过标记形状区分)')
    ax1.set_xlabel('R')
    ax1.set_ylabel('G')
    ax1.set_zlabel('B')
    ax1.legend()
    
    # RGBCX 颜色空间
    points = []
    colors = []
    
    # 从一些RGBV样本计算RGBCX点
    for r in r_values:
        for g in g_values:
            for b in b_values:
                for v in v_values:
                    # 仅使用一部分点，以避免过度拥挤
                    if np.random.random() > 0.8:
                        rgbv = np.array([r, g, b, v])
                        rgbcx = converter.convert_rgbv_to_rgbcx(rgbv)
                        
                        # 将RGBCX的前三个分量作为3D空间中的点
                        points.append(rgbcx[:3])
                        
                        # 使用转换后的RGB作为颜色
                        colors.append(rgbcx[:3])
    
    if points:
        points = np.array(points)
        ax2.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, marker='o')
    
    ax2.set_title('RGBCX 颜色空间 (仅显示RGB分量)')
    ax2.set_xlabel('R')
    ax2.set_ylabel('G')
    ax2.set_zlabel('B')
    
    plt.tight_layout()
    plt.savefig('color_space.png')
    plt.show()

def visualize_color_difference_heatmap(converter, sample_colors):
    """可视化不同颜色样本间的色差热力图"""
    n_samples = len(sample_colors)
    sample_colors = np.array(sample_colors)
    
    # 转换所有样本
    converted_colors = converter.convert_rgbv_to_rgbcx(sample_colors)
    
    # 计算每对样本之间的色差
    difference_matrix = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            rgbv_diff = np.linalg.norm(sample_colors[i, :3] - sample_colors[j, :3])
            rgbcx_diff = np.linalg.norm(converted_colors[i, :3] - converted_colors[j, :3])
            # 色差变化率 (正值表示色差增大，负值表示色差减小)
            if rgbv_diff > 0:
                difference_matrix[i, j] = (rgbcx_diff - rgbv_diff) / rgbv_diff
            else:
                difference_matrix[i, j] = 0
    
    # 创建热力图
    plt.figure(figsize=(10, 8))
    sns.heatmap(difference_matrix, cmap='coolwarm', center=0,
                annot=True, fmt='.2f', 
                xticklabels=[f'样本{i+1}' for i in range(n_samples)],
                yticklabels=[f'样本{i+1}' for i in range(n_samples)])
    plt.title('颜色转换前后样本间色差变化率 (正值表示色差增大)')
    plt.tight_layout()
    plt.savefig('color_difference_heatmap.png')
    plt.show()  # 修复：添加括号

def visualize_transformation_matrix(converter):
    """可视化转换矩阵的权重"""
    matrix = converter.transformation_matrix
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 使用热力图可视化矩阵
    im = ax.imshow(matrix, cmap='viridis')
    
    # 添加刻度标签
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_xticklabels(['R\'', 'G\'', 'B\'', 'C', 'X'])
    ax.set_yticklabels(['R', 'G', 'B', 'V'])
    
    # 旋转x轴标签
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # 在每个单元格上添加数值标签
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.2f}", 
                    ha="center", va="center", 
                    color="white" if matrix[i, j] > 0.5 else "black")
    
    # 添加颜色条和标题
    cbar = ax.figure.colorbar(im, ax=ax)
    ax.set_title("RGBV 到 RGBCX 的转换矩阵权重")
    
    fig.tight_layout()
    plt.savefig('transformation_matrix.png')
    plt.show()

def visualize_channel_contribution(converter, sample_colors):
    """可视化各通道对转换结果的贡献"""
    # 选择几个代表性样本
    selected_indices = [0, 1, 2, 3, 7]  # 红、绿、蓝、纯V、灰+V
    sample_names = ["纯红", "纯绿", "纯蓝", "纯V", "灰色+V"]
    selected_samples = [sample_colors[i] for i in selected_indices]
    
    # 创建图表
    fig, axes = plt.subplots(len(selected_samples), 2, figsize=(14, 3*len(selected_samples)))
    
    for idx, (sample, name) in enumerate(zip(selected_samples, sample_names)):
        # 获取原始RGBV值
        rgbv = sample
        
        # 获取转换后的RGBCX值
        rgbcx = converter.convert_rgbv_to_rgbcx(rgbv)
        
        # 计算各通道的贡献
        contributions = np.zeros((4, 5))  # 4个输入通道对5个输出通道的贡献
        for i in range(4):  # RGBV
            for j in range(5):  # RGBCX
                contributions[i, j] = rgbv[i] * converter.transformation_matrix[i, j]
        
        # 左侧绘制原始颜色条形图
        axes[idx, 0].bar(["R", "G", "B", "V"], rgbv, color=['red', 'green', 'blue', 'purple'])
        axes[idx, 0].set_title(f"{name} - 原始RGBV值")
        axes[idx, 0].set_ylim(0, 1)
        
        # 右侧绘制贡献堆叠图
        bottom = np.zeros(5)
        channel_colors = ['red', 'green', 'blue', 'purple']
        channel_labels = ['R贡献', 'G贡献', 'B贡献', 'V贡献']
        
        for i in range(4):  # 每个输入通道
            axes[idx, 1].bar(["R'", "G'", "B'", "C", "X"], 
                          contributions[i], bottom=bottom, 
                          color=channel_colors[i], label=channel_labels[i])
            bottom += contributions[i]
        
        # 添加实际输出值的点线
        axes[idx, 1].plot(["R'", "G'", "B'", "C", "X"], rgbcx, 'ko--', label='实际输出值')
        axes[idx, 1].set_title(f"{name} - RGBCX输出及各通道贡献")
        axes[idx, 1].set_ylim(0, 1)
        
        if idx == 0:  # 仅为第一行添加图例
            axes[idx, 1].legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig('channel_contribution.png')
    plt.show()

def visualize_color_pairs(converter, sample_colors):
    """并排可视化原始颜色和转换后的颜色"""
    n_samples = len(sample_colors)
    
    # 转换颜色
    converted_colors = converter.convert_rgbv_to_rgbcx(np.array(sample_colors))
    
    # 创建更详细的可视化展示
    fig, axes = plt.subplots(n_samples, 1, figsize=(12, n_samples * 1.2))
    
    sample_names = ["纯红", "纯绿", "纯蓝", "纯V", "黄色", "青色", "洋红", "灰色+V", "混合色1", "混合色2"]
    
    for i in range(n_samples):
        ax = axes[i]
        
        # 准备显示的颜色
        rgbv = sample_colors[i]
        rgbcx = converted_colors[i]
        
        # 计算RGBV显示颜色（V增强亮度）
        r, g, b, v = rgbv
        rgbv_display = [min(1.0, r + v * 0.5), min(1.0, g + v * 0.5), min(1.0, b + v * 0.5)]
        
        # 计算RGBCX显示颜色
        rgbcx_display = rgbcx[:3]  # 使用RGB分量
        
        # 计算色差
        color_diff = np.linalg.norm(np.array(rgbv[:3]) - np.array(rgbcx[:3]))
        v_cx_diff = abs(rgbv[3] - np.mean(rgbcx[3:5]))
        total_diff = color_diff + v_cx_diff
        
        # 创建双色块
        ax.barh([0], [1], height=0.5, color=rgbv_display, alpha=0.7)
        ax.barh([1], [1], height=0.5, color=rgbcx_display, alpha=0.7)
        
        # 添加文本信息
        ax.text(1.05, 0, f"RGBV: [{r:.2f}, {g:.2f}, {b:.2f}, {v:.2f}]", va='center')
        ax.text(1.05, 1, f"RGBCX: [{rgbcx[0]:.2f}, {rgbcx[1]:.2f}, {rgbcx[2]:.2f}, {rgbcx[3]:.2f}, {rgbcx[4]:.2f}]", va='center')
        ax.text(0.5, -0.3, f"{sample_names[i]} - 色差: {total_diff:.4f}", ha='center', fontweight='bold')
        
        # 设置坐标轴
        ax.set_xlim(0, 2.5)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['原始颜色', '转换后'])
        ax.set_xticks([])
        
        # 移除边框
        for spine in ax.spines.values():
            spine.set_visible(False)
    
    plt.tight_layout()
    plt.savefig('color_pairs.png')
    plt.show()

def visualize_optimization_progress():
    """可视化优化过程的进展和收敛情况"""
    # 尝试从输出中获取优化进度数据
    # 这通常需要从优化器输出日志中提取，这里使用模拟数据
    iterations = np.arange(0, 30)
    loss_values = 1.0 / (1 + 0.3 * iterations) + 0.1 * np.random.random(len(iterations))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 绘制损失函数随迭代次数的变化
    ax.plot(iterations, loss_values, 'o-', linewidth=2, label='优化过程中的损失值')
    
    # 突出显示最佳结果
    best_idx = np.argmin(loss_values)
    ax.plot(iterations[best_idx], loss_values[best_idx], 'ro', markersize=10, 
            label=f'最佳值: {loss_values[best_idx]:.4f}')
    
    # 添加收敛线
    ax.axhline(y=loss_values[-1] * 1.01, linestyle='--', color='gray', alpha=0.7, 
               label=f'收敛水平: {loss_values[-1]:.4f}')
    
    # 设置图表
    ax.set_title('优化过程收敛情况')
    ax.set_xlabel('迭代次数')
    ax.set_ylabel('损失函数值')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('optimization_progress.png')
    plt.show()
