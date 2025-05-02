import numpy as np
from color_converter import ColorSpaceConverter
from visualization import (
    visualize_color_conversion, 
    visualize_color_space, 
    visualize_color_difference_heatmap,
    visualize_transformation_matrix,
    visualize_channel_contribution,
    visualize_color_pairs,
    visualize_optimization_progress
)


def main():
    # 创建颜色空间转换器
    converter = ColorSpaceConverter()

    # 生成RGBV颜色样本
    sample_colors = [
        np.array([1.0, 0.0, 0.0, 0.0]),  # 纯红
        np.array([0.0, 1.0, 0.0, 0.0]),  # 纯绿
        np.array([0.0, 0.0, 1.0, 0.0]),  # 纯蓝
        np.array([0.0, 0.0, 0.0, 1.0]),  # 纯V
        np.array([0.5, 0.5, 0.0, 0.0]),  # 黄色
        np.array([0.0, 0.5, 0.5, 0.0]),  # 青色
        np.array([0.5, 0.0, 0.5, 0.0]),  # 洋红
        np.array([0.3, 0.3, 0.3, 0.3]),  # 灰色加V
        np.array([0.7, 0.2, 0.4, 0.3]),  # 混合色1
        np.array([0.2, 0.8, 0.1, 0.5])  # 混合色2
    ]

    # 查看初始矩阵特性
    print("\n初始转换矩阵特性:")
    matrix_analysis = converter.get_matrix_analysis()
    print("输入通道总影响: ", end="")
    for channel, value in matrix_analysis['input_influence'].items():
        print(f"{channel}: {value:.2f}  ", end="")
    print("\n")

    # 计算优化前的色差
    initial_converted = converter.convert_rgbv_to_rgbcx(np.array(sample_colors))
    initial_loss = converter.calculate_color_loss(sample_colors, initial_converted)
    
    print(f"优化前平均色差: {initial_loss/len(sample_colors):.6f}")
    
    # 优化转换矩阵
    print("\n正在优化转换矩阵...")
    optimized_matrix = converter.optimize_transformation(sample_colors)
    
    # 计算优化后的色差
    converted_colors = converter.convert_rgbv_to_rgbcx(np.array(sample_colors))
    optimized_loss = converter.calculate_color_loss(sample_colors, converted_colors)
    
    print(f"优化后平均色差: {optimized_loss/len(sample_colors):.6f}")
    print(f"色差优化改善: {((initial_loss - optimized_loss)/initial_loss)*100:.2f}%")
    
    # 查看优化后矩阵特性
    print("\n优化后转换矩阵特性:")
    matrix_analysis = converter.get_matrix_analysis()
    print("输入通道总影响: ", end="")
    for channel, value in matrix_analysis['input_influence'].items():
        print(f"{channel}: {value:.2f}  ", end="")
        
    print("\n各输出通道的输入组成百分比:")
    for output_channel, composition in matrix_analysis['output_composition'].items():
        print(f"  {output_channel}: ", end="")
        for input_channel, percentage in composition.items():
            if percentage > 1.0:  # 只显示有意义的贡献
                print(f"{input_channel}: {percentage:.1f}%  ", end="")
        print()
    
    # 显示最关键的几个颜色样本的优化效果
    print("\n关键样本优化效果:")
    key_samples = [0, 1, 2, 7]  # 选择几个代表性样本：红、绿、蓝和灰色+V
    
    for i in key_samples:
        rgbv = sample_colors[i]
        rgbcx = converted_colors[i]
        sample_name = ["纯红", "纯绿", "纯蓝", "纯V", "黄色", "青色", "洋红", "灰色+V", "混合色1", "混合色2"][i]
        individual_loss = converter.calculate_individual_color_loss(rgbv, rgbcx)
        print(f"{sample_name}: 色差 = {individual_loss:.6f}")

    # 生成所有可视化图表
    print("\n正在生成可视化图表...")
    
    # 基本颜色转换可视化
    print("生成基本颜色转换可视化...")
    visualize_color_conversion(converter, sample_colors)
    
    # 颜色空间可视化
    print("生成颜色空间可视化...")
    visualize_color_space(converter)
    
    # 转换矩阵可视化
    print("生成转换矩阵可视化...")
    visualize_transformation_matrix(converter)
    
    # 色差热力图
    print("生成色差热力图...")
    visualize_color_difference_heatmap(converter, sample_colors)
    
    # 通道贡献可视化
    print("生成通道贡献可视化...")
    visualize_channel_contribution(converter, sample_colors)
    
    # 颜色对比可视化
    print("生成颜色对比可视化...")
    visualize_color_pairs(converter, sample_colors)
    
    # 新增：优化过程可视化
    print("生成优化过程可视化...")
    visualize_optimization_progress()
    
    print("\n所有可视化图表已保存")


if __name__ == "__main__":
    main()
