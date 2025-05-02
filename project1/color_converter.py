import numpy as np
from scipy.optimize import minimize


class ColorSpaceConverter:
    def __init__(self):
        # 修改初始转换矩阵，引入交叉影响
        self.transformation_matrix = np.array([
            [0.9, 0.1, 0.0, 0.1, 0.0],  # R影响
            [0.1, 0.9, 0.1, 0.0, 0.0],  # G影响
            [0.0, 0.1, 0.9, 0.0, 0.1],  # B影响
            [0.1, 0.1, 0.1, 0.8, 0.8],  # V影响
        ])
        # 缓存已计算的颜色
        self._color_cache = {}
    
    def convert_rgbv_to_rgbcx(self, rgbv):
        """将 RGBV 颜色转换为 RGBCX 颜色"""
        # 检查输入是否为批量处理
        if isinstance(rgbv, np.ndarray) and rgbv.ndim > 1:
            # 批量处理多个颜色
            results = []
            for color in rgbv:
                results.append(self.convert_rgbv_to_rgbcx(color))
            return np.array(results)
        
        # 转换单个颜色
        # 使用缓存避免重复计算
        rgbv_tuple = tuple(rgbv)
        if rgbv_tuple in self._color_cache:
            return self._color_cache[rgbv_tuple]
        
        # 应用转换矩阵
        rgbcx = np.dot(rgbv, self.transformation_matrix)
        
        # 确保值在合理范围内 (0到1)
        rgbcx = np.clip(rgbcx, 0, 1)
        
        # 缓存结果
        self._color_cache[rgbv_tuple] = rgbcx
        
        return rgbcx
    
    def calculate_color_loss(self, original_colors, converted_colors):
        """计算原始颜色与转换后颜色之间的色差总和"""
        total_loss = 0.0
        
        for i in range(len(original_colors)):
            rgbv = original_colors[i]
            rgbcx = converted_colors[i]
            total_loss += self.calculate_individual_color_loss(rgbv, rgbcx)
        
        return total_loss
    
    def calculate_individual_color_loss(self, original_color, converted_color):
        """计算单个颜色样本的色差 - 修改以匹配显示方式"""
        # 使用显示时的处理方法计算色差
        r, g, b, v = original_color
        rgbv_display = [min(1.0, r + v * 0.5), min(1.0, g + v * 0.5), min(1.0, b + v * 0.5)]
        rgb_diff = np.linalg.norm(np.array(rgbv_display) - converted_color[:3])
        
        # V与CX色差
        v_cx_diff = np.abs(v - np.mean(converted_color[3:5]))
        
        # 总色差
        total_loss = rgb_diff + v_cx_diff
        return total_loss
    
    def optimization_function(self, params):
        """优化函数，用于最小化色差并提高表现力"""
        # 重塑参数为转换矩阵
        matrix = params.reshape(4, 5)
        
        # 临时保存原始矩阵
        original_matrix = self.transformation_matrix.copy()
        self.transformation_matrix = matrix
        
        # 清除缓存，确保使用新矩阵重新计算
        self._color_cache.clear()
        
        # 验证样本颜色
        sample_colors = self._optimization_samples
        converted_colors = self.convert_rgbv_to_rgbcx(np.array(sample_colors))
        
        # 计算色差
        loss = self.calculate_color_loss(sample_colors, converted_colors)
        
        # 增加表现力评估（确保通道间有足够差异）
        channel_variance = np.sum(np.var(matrix, axis=0))  # 各输出通道权重的方差之和
        expressiveness = -0.15 * channel_variance  # 负号使方差最大化
        
        # 增加一个惩罚项，避免矩阵中的极端值
        extreme_penalty = 0.1 * np.sum((matrix > 0.95) | (matrix < 0.05))
        
        # 恢复原始矩阵
        self.transformation_matrix = original_matrix
        self._color_cache.clear()
        
        # 返回综合损失
        return loss + expressiveness + extreme_penalty
    
    def optimize_transformation(self, sample_colors):
        """优化转换矩阵以最小化色差"""
        # 存储样本颜色供优化使用
        self._optimization_samples = sample_colors
        
        # 初始参数为当前转换矩阵
        initial_params = self.transformation_matrix.flatten()
        
        # 设置约束条件，确保矩阵元素在0到1之间
        bounds = [(0, 1) for _ in range(len(initial_params))]
        
        # 执行优化
        print("开始优化转换矩阵...(可能需要几分钟)")
        result = minimize(
            self.optimization_function, 
            initial_params, 
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200, 'disp': True}
        )
        
        # 更新转换矩阵为优化结果
        optimized_matrix = result.x.reshape(4, 5)
        self.transformation_matrix = optimized_matrix
        
        # 清除缓存，确保使用新矩阵
        self._color_cache.clear()
        
        return optimized_matrix
        
    def get_matrix_analysis(self):
        """分析当前转换矩阵的特性"""
        matrix = self.transformation_matrix
        
        # 计算每个输入通道对各输出通道的影响
        input_influence = {
            'R': np.sum(matrix[0]),
            'G': np.sum(matrix[1]), 
            'B': np.sum(matrix[2]),
            'V': np.sum(matrix[3])
        }
        
        # 计算每个输出通道受各输入通道的影响程度
        output_composition = {
            "R'": {
                'R': matrix[0,0]/np.sum(matrix[:,0])*100 if np.sum(matrix[:,0]) > 0 else 0,
                'G': matrix[1,0]/np.sum(matrix[:,0])*100 if np.sum(matrix[:,0]) > 0 else 0,
                'B': matrix[2,0]/np.sum(matrix[:,0])*100 if np.sum(matrix[:,0]) > 0 else 0,
                'V': matrix[3,0]/np.sum(matrix[:,0])*100 if np.sum(matrix[:,0]) > 0 else 0
            },
            "G'": {
                'R': matrix[0,1]/np.sum(matrix[:,1])*100 if np.sum(matrix[:,1]) > 0 else 0,
                'G': matrix[1,1]/np.sum(matrix[:,1])*100 if np.sum(matrix[:,1]) > 0 else 0,
                'B': matrix[2,1]/np.sum(matrix[:,1])*100 if np.sum(matrix[:,1]) > 0 else 0,
                'V': matrix[3,1]/np.sum(matrix[:,1])*100 if np.sum(matrix[:,1]) > 0 else 0
            },
            "B'": {
                'R': matrix[0,2]/np.sum(matrix[:,2])*100 if np.sum(matrix[:,2]) > 0 else 0,
                'G': matrix[1,2]/np.sum(matrix[:,2])*100 if np.sum(matrix[:,2]) > 0 else 0,
                'B': matrix[2,2]/np.sum(matrix[:,2])*100 if np.sum(matrix[:,2]) > 0 else 0,
                'V': matrix[3,2]/np.sum(matrix[:,2])*100 if np.sum(matrix[:,2]) > 0 else 0
            },
            "C": {
                'R': matrix[0,3]/np.sum(matrix[:,3])*100 if np.sum(matrix[:,3]) > 0 else 0,
                'G': matrix[1,3]/np.sum(matrix[:,3])*100 if np.sum(matrix[:,3]) > 0 else 0,
                'B': matrix[2,3]/np.sum(matrix[:,3])*100 if np.sum(matrix[:,3]) > 0 else 0,
                'V': matrix[3,3]/np.sum(matrix[:,3])*100 if np.sum(matrix[:,3]) > 0 else 0
            },
            "X": {
                'R': matrix[0,4]/np.sum(matrix[:,4])*100 if np.sum(matrix[:,4]) > 0 else 0,
                'G': matrix[1,4]/np.sum(matrix[:,4])*100 if np.sum(matrix[:,4]) > 0 else 0,
                'B': matrix[2,4]/np.sum(matrix[:,4])*100 if np.sum(matrix[:,4]) > 0 else 0,
                'V': matrix[3,4]/np.sum(matrix[:,4])*100 if np.sum(matrix[:,4]) > 0 else 0
            }
        }
        
        return {
            'input_influence': input_influence,
            'output_composition': output_composition
        }
