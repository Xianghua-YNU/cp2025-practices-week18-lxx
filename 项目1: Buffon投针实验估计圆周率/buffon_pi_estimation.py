import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def buffon_needle_experiment(num_trials, d=2, l=1):
    """
    执行Buffon投针实验
    :param num_trials: 实验次数
    :param d: 平行线间距 (默认2)
    :param l: 针的长度 (默认1)
    :return: π的估计值
    """
    hits = 0
    
    for _ in range(num_trials):
        # 随机生成针的位置和角度
        y = np.random.uniform(0, d/2)  # 针中点到最近平行线的距离
        theta = np.random.uniform(0, np.pi/2)  # 针的角度
        
        # 判断是否相交 (y ≤ (l/2)*sinθ)
        if y <= (l/2) * np.sin(theta):
            hits += 1
    
    # 避免除以零的情况
    if hits == 0:
        return 0
    return (2 * l * num_trials) / (hits * d)

def run_experiments(trial_counts):
    """
    运行不同次数的实验
    :param trial_counts: 实验次数列表
    :return: π估计值列表
    """
    estimates = []
    for n in trial_counts:
        pi_estimate = buffon_needle_experiment(n)
        estimates.append(pi_estimate)
    return estimates

def analyze_results(trial_counts, pi_estimates):
    """
    分析结果并生成表格和图表
    :param trial_counts: 实验次数列表
    :param pi_estimates: π估计值列表
    """
    # 创建结果表格
    results = pd.DataFrame({
        'Trials': trial_counts,
        'Estimated π': pi_estimates,
        'Relative Error (%)': abs((np.array(pi_estimates) - np.pi)/np.pi)*100
    })
    
    print("实验结果表格:")
    print(results.to_string(index=False))
    
    # 绘制结果图表
    plt.figure(figsize=(12, 6))
    
    # π估计值随实验次数的变化
    plt.subplot(1, 2, 1)
    plt.plot(trial_counts, pi_estimates, 'o-', label='Estimated π')
    plt.axhline(y=np.pi, color='r', linestyle='--', label='True π')
    plt.xscale('log')
    plt.xlabel('Number of trials (log scale)')
    plt.ylabel('Estimated π value')
    plt.title('π Estimation vs. Number of Trials')
    plt.legend()
    plt.grid(True)
    
    # 相对误差随实验次数的变化
    plt.subplot(1, 2, 2)
    plt.plot(trial_counts, results['Relative Error (%)'], 's-', color='green')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Number of trials (log scale)')
    plt.ylabel('Relative Error (%) (log scale)')
    plt.title('Estimation Error vs. Number of Trials')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def main():
    # 设置实验次数
    trial_counts = [100, 1000, 10000, 100000, 1000000]
    
    print("正在运行Buffon投针实验...")
    pi_estimates = run_experiments(trial_counts)
    
    # 分析并展示结果
    analyze_results(trial_counts, pi_estimates)

if __name__ == "__main__":
    main()
