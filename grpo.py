import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical

class GRPO:
    def __init__(self, state_dim, action_dim, lr=0.002, epsilon=0.2, epochs=8, group_size=256):
        self.policy = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )
        
        self.optimizer = optim.Adam(list(self.policy.parameters()), lr=lr)
        
        self.epsilon = epsilon
        self.epochs = epochs
        self.group_size = group_size
        self.sample_buffer = []
        
    def get_action(self, state):
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state)
        else:
            state = torch.FloatTensor(np.array(state))
        probs = self.policy(state)
        dist = Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def get_return(self, rewards):
        R = 0
        for r in rewards.tolist():
            R += r

        return R

    def update_policy(self):
        returns = []
        for sample in self.sample_buffer:
            rewards = sample['rewards']
            R = self.get_return(rewards)
            returns.append(R)
        
        returns = torch.FloatTensor(returns)
        avg_return = returns.mean()
        std_return = returns.std()

        returns = (returns - avg_return) / (std_return + 1e-8)

        for i, sample in enumerate(self.sample_buffer):
            states = sample['states']
            actions = sample['actions']
            rewards = sample['rewards']
            log_probs = sample['log_probs']

            self.train_sample(states, actions, rewards, log_probs, returns[i])

    def train_sample(self, states, actions, rewards, log_probs, R):
        advantages = [R] * len(rewards)
        advantages = torch.FloatTensor(advantages)
        
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

    def update(self, states, actions, rewards, log_probs, next_states, dones):
        # 将输入数据转换为张量
        states = torch.FloatTensor(np.array(states))
        next_states = torch.FloatTensor(np.array(next_states))
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        log_probs = torch.FloatTensor(log_probs)
        dones = torch.FloatTensor(dones)

        sample = {
            'states':states, 
            'actions':actions, 
            'rewards':rewards, 
            'log_probs':log_probs, 
            'next_states':next_states, 
            'dones':dones
        }
        self.sample_buffer.append(sample)

        if len(self.sample_buffer) >= self.group_size:
            for _ in range(self.epochs):
                self.update_policy()
            self.sample_buffer = []
