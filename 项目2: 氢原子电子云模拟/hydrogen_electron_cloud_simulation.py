import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import rv_continuous
from matplotlib import cm

# 物理常数和参数设置
a = 5.29e-2  # 玻尔半径 (nm)
D_max = 1.1   # 概率密度最大值
r0 = 0.25     # 收敛半径 (nm)
num_points = 50000  # 模拟点数

# 1. 定义概率密度函数类
class ElectronDistribution(rv_continuous):
    """氢原子基态电子概率密度分布"""
    def _pdf(self, r):
        return (4 * r**2 / a**3) * np.exp(-2 * r / a)

# 2. 创建概率分布对象并生成随机样本
electron_dist = ElectronDistribution(a=0, b=r0, name='electron_dist')
r_samples = electron_dist.rvs(size=num_points)

# 生成随机角度分布 (球坐标)
theta_samples = np.random.uniform(0, np.pi, num_points)
phi_samples = np.random.uniform(0, 2*np.pi, num_points)

# 3. 转换为笛卡尔坐标
x = r_samples * np.sin(theta_samples) * np.cos(phi_samples)
y = r_samples * np.sin(theta_samples) * np.sin(phi_samples)
z = r_samples * np.cos(theta_samples)

# 4. 计算概率密度值用于颜色映射
density_values = (4 * r_samples**2 / a**3) * np.exp(-2 * r_samples / a)
norm_density = density_values / D_max  # 归一化

# 5. 创建3D可视化
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# 绘制电子云 (使用密度值进行颜色映射)
scatter = ax.scatter(
    x, y, z, 
    c=norm_density, 
    cmap='viridis',
    s=1, 
    alpha=0.15,
    vmin=0, 
    vmax=1
)

# 添加颜色条
cbar = fig.colorbar(scatter, ax=ax, shrink=0.6)
cbar.set_label('Normalized Probability Density')

# 绘制原子核
ax.scatter([0], [0], [0], s=150, c='red', marker='o', label='Nucleus')

# 设置图形参数
ax.set_xlim([-r0, r0])
ax.set_ylim([-r0, r0])
ax.set_zlim([-r0, r0])
ax.set_xlabel('X (nm)')
ax.set_ylabel('Y (nm)')
ax.set_zlabel('Z (nm)')
ax.set_title('Hydrogen Atom Electron Cloud Simulation (n=1, l=0, m=0)', pad=20)
ax.legend()

# 6. 添加2D径向概率密度分布图
plt.figure(figsize=(8, 6))
r = np.linspace(0, r0, 500)
D_r = (4 * r**2 / a**3) * np.exp(-2 * r / a)
plt.plot(r, D_r, 'b-', linewidth=2, label='Theoretical')
plt.xlabel('Distance from nucleus (nm)')
plt.ylabel('Probability Density')
plt.title('Radial Probability Density Distribution')
plt.axvline(a, color='r', linestyle='--', label=f'Bohr radius (a={a:.3f} nm)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
