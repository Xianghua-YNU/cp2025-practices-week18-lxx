import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.gridspec as gridspec

class AdvancedChainReactionSimulator:
    def __init__(self, p_fission=0.5, k=2.5, p_absorb=0.3, p_escape=0.2, 
                 initial_neutrons=1, max_generations=20, spatial_size=10):
        """
        增强版链式反应模拟器，增加空间维度
        
        参数:
        p_fission: 裂变概率
        k: 每次裂变产生的中子数
        p_absorb: 吸收概率
        p_escape: 逃逸概率
        initial_neutrons: 初始中子数
        max_generations: 最大模拟代数
        spatial_size: 空间网格大小
        """
        self.p_fission = p_fission
        self.k = k
        self.p_absorb = p_absorb
        self.p_escape = p_escape
        self.initial_neutrons = initial_neutrons
        self.max_generations = max_generations
        self.spatial_size = spatial_size
        
        # 验证概率总和为1
        assert np.isclose(p_fission + p_absorb + p_escape, 1.0), "概率之和必须为1"
    
    def simulate_spatial(self):
        """带空间维度的链式反应模拟"""
        # 初始化空间网格和中子计数
        neutron_grid = np.zeros((self.max_generations, self.spatial_size, self.spatial_size))
        neutron_counts = np.zeros(self.max_generations)
        
        # 在中心位置放置初始中子
        center = self.spatial_size // 2
        neutron_grid[0, center, center] = self.initial_neutrons
        neutron_counts[0] = self.initial_neutrons
        
        for gen in range(1, self.max_generations):
            # 复制上一代的中子分布
            prev_grid = neutron_grid[gen-1].copy()
            new_grid = np.zeros((self.spatial_size, self.spatial_size))
            new_neutrons = 0
            
            # 遍历空间中的每个位置
            for i in range(self.spatial_size):
                for j in range(self.spatial_size):
                    neutrons_here = prev_grid[i, j]
                    if neutrons_here == 0:
                        continue
                        
                    # 对每个中子进行处理
                    for _ in range(int(neutrons_here)):
                        fate = np.random.choice(
                            ['fission', 'absorb', 'escape'],
                            p=[self.p_fission, self.p_absorb, self.p_escape]
                        )
                        
                        if fate == 'fission':
                            # 裂变产生k个新中子，随机分布在周围
                            for _ in range(int(self.k)):
                                # 随机选择相邻位置
                                di, dj = np.random.randint(-1, 2, size=2)
                                ni, nj = i + di, j + dj
                                
                                # 检查边界
                                if 0 <= ni < self.spatial_size and 0 <= nj < self.spatial_size:
                                    new_grid[ni, nj] += 1
                                    new_neutrons += 1
                                else:
                                    # 超出边界视为逃逸
                                    pass
                        
                        # 吸收和逃逸不产生新中子
                        elif fate == 'absorb':
                            pass
                        elif fate == 'escape':
                            # 随机方向逃逸
                            if np.random.rand() < 0.5:  # 50%概率逃出系统
                                pass
                            else:
                                # 可能逃到网格外
                                di, dj = np.random.randint(-1, 2, size=2)
                                ni, nj = i + di, j + dj
                                if 0 <= ni < self.spatial_size and 0 <= nj < self.spatial_size:
                                    new_grid[ni, nj] += 0  # 逃逸不增加中子
                                    
            neutron_grid[gen] = new_grid
            neutron_counts[gen] = new_neutrons
            
            # 如果中子数为0，反应停止
            if new_neutrons == 0:
                break
                
        return neutron_counts, neutron_grid
    
    def run_multiple_trials(self, num_trials=100, spatial=False):
        """运行多次试验取平均结果"""
        all_results = []
        all_grids = []
        
        for _ in tqdm(range(num_trials)):
            if spatial:
                result, grid = self.simulate_spatial()
            else:
                result = self.simulate()
                grid = None
                
            padded_result = np.zeros(self.max_generations)
            padded_result[:len(result)] = result
            all_results.append(padded_result)
            all_grids.append(grid)
            
        return np.mean(all_results, axis=0), all_grids
    
    def simulate(self):
        """无空间维度的简单模拟"""
        neutron_counts = np.zeros(self.max_generations)
        neutron_counts[0] = self.initial_neutrons
        
        for gen in range(1, self.max_generations):
            prev_neutrons = neutron_counts[gen-1]
            new_neutrons = 0
            
            for _ in range(int(prev_neutrons)):
                fate = np.random.choice(
                    ['fission', 'absorb', 'escape'],
                    p=[self.p_fission, self.p_absorb, self.p_escape]
                )
                
                if fate == 'fission':
                    new_neutrons += self.k
            
            neutron_counts[gen] = new_neutrons
            
            if new_neutrons == 0:
                break
                
        return neutron_counts
    
    def plot_comparison(self, param_sets):
        """比较不同参数设置下的链式反应"""
        plt.figure(figsize=(12, 8))
        
        for params in param_sets:
            # 临时修改参数
            original_params = {
                'p_fission': self.p_fission,
                'k': self.k,
                'p_absorb': self.p_absorb,
                'p_escape': self.p_escape
            }
            
            self.p_fission = params['p_fission']
            self.k = params['k']
            self.p_absorb = params['p_absorb']
            self.p_escape = params['p_escape']
            
            # 计算k_effective
            k_eff = self.k * self.p_fission
            
            # 运行模拟
            result = self.run_multiple_trials(num_trials=500)[0]
            
            # 绘制结果
            label = (f"p_f={self.p_fission}, k={self.k}, k_eff={k_eff:.2f}\n"
                    f"abs={self.p_absorb}, esc={self.p_escape}")
            plt.plot(result, 'o-', linewidth=2, markersize=6, label=label)
            
            # 恢复原始参数
            self.p_fission = original_params['p_fission']
            self.k = original_params['k']
            self.p_absorb = original_params['p_absorb']
            self.p_escape = original_params['p_escape']
        
        plt.xlabel('Generation', fontsize=12)
        plt.ylabel('Neutron Count (average)', fontsize=12)
        plt.title('Chain Reaction Behavior Under Different Parameters', fontsize=14)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_spatial_evolution(self, grid, max_generations=5):
        """可视化空间演化的前几代"""
        if grid is None:
            print("No spatial data available")
            return
        
        actual_generations = min(max_generations, len(grid))
        
        plt.figure(figsize=(15, 3*actual_generations))
        gs = gridspec.GridSpec(actual_generations, 1)
        
        for gen in range(actual_generations):
            ax = plt.subplot(gs[gen])
            im = ax.imshow(grid[gen], cmap='hot', interpolation='nearest')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title(f'Generation {gen}', fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
        
        plt.suptitle('Spatial Distribution of Neutrons Over Generations', fontsize=14)
        plt.tight_layout()
        plt.show()

# 主程序
if __name__ == "__main__":
    # 创建模拟器实例
    simulator = AdvancedChainReactionSimulator(
        p_fission=0.5, 
        k=2.5, 
        p_absorb=0.3, 
        p_escape=0.2,
        max_generations=20,
        spatial_size=15
    )
    
    print("Running basic chain reaction simulations with different parameters...")
    
    # 定义不同参数组合
    param_sets = [
        {'p_fission': 0.4, 'k': 2.0, 'p_absorb': 0.4, 'p_escape': 0.2},  # 亚临界
        {'p_fission': 0.5, 'k': 2.0, 'p_absorb': 0.3, 'p_escape': 0.2},  # 临界
        {'p_fission': 0.6, 'k': 2.5, 'p_absorb': 0.1, 'p_escape': 0.3},  # 超临界
        {'p_fission': 0.3, 'k': 4.0, 'p_absorb': 0.4, 'p_escape': 0.3}   # 高倍增因子
    ]
    
    # 比较不同参数下的链式反应行为
    simulator.plot_comparison(param_sets)
    
    print("\nRunning spatial simulation...")
    
    # 运行空间模拟
    spatial_result, spatial_grids = simulator.run_multiple_trials(num_trials=1, spatial=True)
    
    # 可视化空间演化
    simulator.plot_spatial_evolution(spatial_grids[0], max_generations=5)
    
    # 绘制平均结果
    plt.figure(figsize=(10, 6))
    plt.plot(spatial_result, 'o-', linewidth=2, markersize=8)
    plt.xlabel('Generation', fontsize=12)
    plt.ylabel('Neutron Count', fontsize=12)
    plt.title('Average Chain Reaction with Spatial Simulation', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.show()
