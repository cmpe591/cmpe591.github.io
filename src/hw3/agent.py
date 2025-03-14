import torch
from torch import optim

from model import VPG
import torch.nn.functional as F

gamma = 0.99


class Agent():
    def __init__(self):
        # edit as needed
        self.model = VPG()
        self.rewards = []
        
    def decide_action(self, state):
        # edit as needed
        action_mean, act_std = self.model(state).chunk(2, dim=-1)
        action_std = F.softplus(act_std) + 5e-2  # increase variance to stimulate exploration

    
    def update_model(self):
        # crucial part of the code
        pass

    def add_reward(self, reward):
        self.rewards.append(reward)
        
