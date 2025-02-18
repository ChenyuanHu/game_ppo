import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical

class GRPO:
    def __init__(self, state_dim, action_dim, lr=0.002, gamma=0.99, epsilon=0.2, epochs=4, group_size=100):
        self.policy = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )
        
        self.optimizer = optim.Adam(list(self.policy.parameters()), lr=lr)
        
        self.gamma = gamma
        self.epsilon = epsilon
        self.epochs = epochs

        self.group_size = group_size
        self.returns_group = []
        
    def add_return(self, episode_return):
        """添加一局游戏的Return到滑动窗口中"""
        self.returns_group.append(episode_return)
        if len(self.returns_group) > self.group_size:
            self.returns_group.pop(0)
            
    def get_average_return(self):
        """获取滑动窗口中Return的平均值"""
        return np.mean(self.returns_group)
        
    def get_action(self, state):
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state)
        else:
            state = torch.FloatTensor(np.array(state))
        probs = self.policy(state)
        dist = Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def get_return(self, rewards, dones):
        # 将输入数据转换为张量
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)

        R = rewards[-1]
        
        for r, done in zip(reversed(rewards.tolist()), reversed(dones.tolist())):
            R += r

        return R
    
    def update(self, states, actions, rewards, log_probs, next_states, dones):
        # 将输入数据转换为张量
        states = torch.FloatTensor(np.array(states))
        next_states = torch.FloatTensor(np.array(next_states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        log_probs = torch.FloatTensor(log_probs)
        dones = torch.FloatTensor(dones)
        
        R = self.get_return(rewards, dones)
        self.add_return(R)
        avg_R = self.get_average_return()
        
        advantages = []
        for r, done in zip(reversed(rewards.tolist()), reversed(dones.tolist())):
            advantage = R - avg_R
            advantages.insert(0, advantage)
            
        advantages = torch.FloatTensor(advantages)
        
        # PPO更新
        for _ in range(self.epochs):
            current_probs = self.policy(states)
            dist = Categorical(current_probs)
            current_log_probs = dist.log_prob(actions)
            
            ratios = torch.exp(current_log_probs - log_probs)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.epsilon, 1+self.epsilon) * advantages
            
            actor_loss = -torch.min(surr1, surr2).mean()
            
            loss = actor_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step() 