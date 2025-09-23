import numpy as np

# 常数
S_in = 283.531  # W/m² (更正后的太阳输入)
sigma = 5.67e-8  # Stefan-Boltzmann constant
alpha = 0.300  # 反照率
epsilon_CO2 = 0.722  # CO2倍增后的发射率

# 首先计算初始温度（无CO2倍增，ε=0.700）
epsilon_initial = 0.700
T_s4_initial = S_in / (sigma * (1 - epsilon_initial/2))
T_initial = T_s4_initial ** 0.25
print(f"初始地表温度 T_initial = {T_initial:.3f} K")

# 计算仅CO2倍增的温度（ε=0.722，无水汽反馈）
T_s4_CO2 = S_in / (sigma * (1 - epsilon_CO2/2))
T_CO2_only = T_s4_CO2 ** 0.25
delta_T_CO2 = T_CO2_only - T_initial
print(f"仅CO2倍增后温度 = {T_CO2_only:.3f} K")
print(f"仅CO2倍增的温升 ΔT = {delta_T_CO2:.3f} K\n")

# 迭代求解包含水汽反馈的温度
print("开始迭代（包含水汽反馈）：")
print("-" * 50)

delta_T = delta_T_CO2  # 从CO2倍增的温升开始
tolerance = 1e-6
max_iter = 30

for i in range(max_iter):
    # 水汽反馈：ε增加0.01每度温升
    epsilon = epsilon_CO2 + 0.01 * delta_T
    
    # 计算新的平衡温度
    T_s4 = S_in / (sigma * (1 - epsilon/2))
    T_s = T_s4 ** 0.25
    
    # 计算新的温升
    delta_T_new = T_s - T_initial
    
    # 打印迭代信息
    print(f"迭代 {i+1:2d}: ε = {epsilon:.4f}, T_s = {T_s:.3f} K, ΔT = {delta_T_new:.3f} K")
    
    # 检查收敛
    if abs(delta_T_new - delta_T) < tolerance:
        print("-" * 50)
        print(f"收敛于第 {i+1} 次迭代")
        break
    
    delta_T = delta_T_new

# 最终结果
print("\n" + "=" * 50)
print("最终结果：")
print(f"最终发射率 ε = {epsilon:.4f}")
print(f"最终地表温度 T_s = {T_s:.3f} K")
print(f"最终温升 ΔT = {delta_T:.3f} K")
print(f"反馈放大因子 = {delta_T/delta_T_CO2:.2f}")