import gym
import torch
import numpy as np
from ppo import PPO
import time

# 创建环境
env = gym.make('CartPole-v1', render_mode='human')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# 初始化代理
agent = PPO(state_dim, action_dim)

# 加载训练好的模型
agent.policy.load_state_dict(torch.load('ppo_policy.pth'))

# 设置为评估模式
agent.policy.eval()

# 游戏参数
n_episodes = 5  # 玩5局游戏
max_steps = 500  # 每局最多500步

# 统计数据
episode_rewards = []

for episode in range(n_episodes):
    state, _ = env.reset()
    episode_reward = 0
    
    print(f"\n开始第 {episode + 1} 局游戏...")
    
    for step in range(max_steps):
        # 渲染环境
        env.render()
        
        # 控制游戏速度
        time.sleep(0.01)  # 添加小延迟使游戏更容易观察
        
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
    print(f"第 {episode + 1} 局游戏结束，得分: {episode_reward}")

# 打印统计信息
print("\n游戏统计:")
print(f"平均得分: {np.mean(episode_rewards):.2f}")
print(f"最高得分: {np.max(episode_rewards):.2f}")
print(f"最低得分: {np.min(episode_rewards):.2f}")

env.close() 