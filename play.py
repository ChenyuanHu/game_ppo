import gym
import torch
import numpy as np
from ppo import PPO
from grpo import GRPO
import time

# 创建环境
env = gym.make('LunarLander-v2', render_mode='human')
state_dim = env.observation_space.shape[0]  # 8维状态空间
action_dim = env.action_space.n  # 4维动作空间

# 初始化代理
# agent = PPO(state_dim, action_dim)
agent = GRPO(state_dim, action_dim)

# 加载训练好的模型（尝试加载最佳模型，如果失败则加载最终模型）
try:
    agent.policy.load_state_dict(torch.load('best_policy.pth'))
    print("已加载最佳模型")
except FileNotFoundError:
    try:
        agent.policy.load_state_dict(torch.load('final_policy.pth'))
        print("已加载最终模型")
    except FileNotFoundError:
        print("未找到任何模型文件，请先运行训练脚本")
        exit(1)

# 设置为评估模式
agent.policy.eval()

# 游戏参数
n_episodes = 5  # 玩5局游戏
max_steps = 1000  # 每局最多1000步

# 统计数据
episode_rewards = []

print("\n开始测试 LunarLander-v2 环境...")
print(f"状态空间维度: {state_dim}")
print(f"动作空间维度: {action_dim}")
print("动作含义: 0=不动, 1=主引擎, 2=左引擎, 3=右引擎")
print("----------------------------------------")

for episode in range(n_episodes):
    state, _ = env.reset()
    episode_reward = 0
    
    print(f"\n开始第 {episode + 1} 局游戏...")
    
    for step in range(max_steps):
        # 渲染环境
        env.render()
        
        # 控制游戏速度
        time.sleep(0.02)  # 稍微放慢速度以便观察
        
        # 获取动作
        with torch.no_grad():
            action, _ = agent.get_action(state)
        
        # 执行动作
        next_state, reward, done, _, _ = env.step(action)
        
        episode_reward += reward
        state = next_state
        
        if done:
            break
    
    episode_rewards.append(episode_reward)
    print(f"第 {episode + 1} 局游戏结束，得分: {episode_reward:.2f}")

# 打印统计信息
print("\n游戏统计:")
print(f"平均得分: {np.mean(episode_rewards):.2f}")
print(f"最高得分: {np.max(episode_rewards):.2f}")
print(f"最低得分: {np.min(episode_rewards):.2f}")

env.close() 