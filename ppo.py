import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical

class PPO:
    def __init__(self, state_dim, action_dim, lr=0.002, gamma=0.99, epsilon=0.2, epochs=4):
        self.policy = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )
        
        self.value = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
        self.optimizer = optim.Adam(list(self.policy.parameters()) + list(self.value.parameters()), lr=lr)
        
        self.gamma = gamma
        self.epsilon = epsilon
        self.epochs = epochs
        
    def get_action(self, state):
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state)
        else:
            state = torch.FloatTensor(np.array(state))
        probs = self.policy(state)
        dist = Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)
    
    def update(self, states, actions, rewards, log_probs, next_states, dones):
        # 将输入数据转换为张量
        states = torch.FloatTensor(np.array(states))
        next_states = torch.FloatTensor(np.array(next_states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        log_probs = torch.FloatTensor(log_probs)
        dones = torch.FloatTensor(dones)
        
        # 计算优势函数
        values = self.value(states).squeeze()
        next_values = self.value(next_states).squeeze()
        
        returns = []
        advantages = []
        R = next_values[-1] * (1 - dones[-1])
        
        for r, v, done in zip(reversed(rewards.tolist()), reversed(values.detach().tolist()), reversed(dones.tolist())):
            R = r + self.gamma * R * (1 - done)
            advantage = R - v
            returns.insert(0, R)
            advantages.insert(0, advantage)
            
        returns = torch.FloatTensor(returns)
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
            critic_loss = 0.5 * ((returns - self.value(states).squeeze())**2).mean()
            
            loss = actor_loss + critic_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step() 