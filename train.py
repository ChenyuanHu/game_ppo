import gym
import numpy as np
import torch
from ppo import PPO
import matplotlib.pyplot as plt

# 创建环境
env = gym.make('LunarLander-v2')
state_dim = env.observation_space.shape[0]  # 8维状态空间
action_dim = env.action_space.n  # 4维动作空间

# 初始化PPO代理
agent = PPO(state_dim, action_dim)

# 训练参数
episodes = 10000  # 增加训练回合数
max_steps = 1000
batch_size = 32

# 用于存储训练数据
episode_rewards = []
best_reward = -float('inf')

# 训练循环
for episode in range(episodes):
    state, _ = env.reset()
    episode_reward = 0
    
    states = []
    actions = []
    rewards = []
    log_probs = []
    next_states = []
    dones = []
    
    for step in range(max_steps):
        # 收集轨迹
        action, log_prob = agent.get_action(state)
        next_state, reward, done, _, _ = env.step(action)
        
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        log_probs.append(log_prob)
        next_states.append(next_state)
        dones.append(done)
        
        state = next_state
        episode_reward += reward
        
        if done:
            break
    
    # 更新策略
    agent.update(states, actions, rewards, log_probs, next_states, dones)
    
    episode_rewards.append(episode_reward)
    
    # 保存最佳模型
    if episode_reward > best_reward:
        best_reward = episode_reward
        torch.save(agent.policy.state_dict(), 'best_policy.pth')
        torch.save(agent.value.state_dict(), 'best_value.pth')
    
    if (episode + 1) % 10 == 0:
        avg_reward = np.mean(episode_rewards[-10:])
        print(f"Episode {episode + 1}, Average Reward: {avg_reward:.2f}, Best Reward: {best_reward:.2f}")

# 绘制训练曲线
plt.figure(figsize=(10, 5))
plt.plot(episode_rewards)
plt.title('Training Progress - LunarLander-v2')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.savefig('training_curve.png')
plt.close()

# 保存最终模型
torch.save(agent.policy.state_dict(), 'final_policy.pth')
torch.save(agent.value.state_dict(), 'final_value.pth') 