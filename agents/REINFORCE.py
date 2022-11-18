import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from agents import Agent


class REINFORCE(Agent):

    def __init__(self, num_features: int, num_actions: int,
                 num_layers: int, hidden_sizes: int, learning_rate: float, discount_factor: float,
                 name="REINFORCE"):

        super(REINFORCE, self).__init__(name)
        self.trainable = True

        self.network = PolicyNetwork(num_features, num_actions, num_layers, hidden_sizes, learning_rate)
        self.gamma = discount_factor

        self.rewards_buffer = []
        self.action_logits_buffer = []
        self.last_policy = None

    def action(self, observation):
        x = torch.from_numpy(observation).float().unsqueeze(0)
        actor_output = self.network(x)
        self.last_policy = Categorical(actor_output)
        action = self.last_policy.sample()
        return action.item()

    def step(self, timestep):

        if self.trainable:

            self.action_logits_buffer.append(self.last_policy.log_prob(torch.tensor(timestep.action)))
            self.rewards_buffer.append(timestep.reward)

            if timestep.terminal:
                returns = self.compute_returns()
                self.network.update(returns, self.action_logits_buffer)
                self.rewards_buffer.clear()
                self.action_logits_buffer.clear()

    def compute_returns(self, nan_preventing_eps=np.finfo(np.float32).eps.item()):
        returns = []
        current_return = 0
        for reward in reversed(self.rewards_buffer):
            current_return = reward + self.gamma * current_return
            returns = [current_return] + returns
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + nan_preventing_eps)
        return returns

    def train(self):
        self.trainable = True

    def eval(self):
        self.trainable = False

class PolicyNetwork(nn.Module):

    """
        Implements the policy for the REINFORCE algorithm
    """

    def __init__(self, num_features: int, num_actions: int, num_layers: int, hidden_sizes: int, learning_rate: float):
        super(PolicyNetwork, self).__init__()
        hidden_layers = [nn.Linear(num_features, hidden_sizes), nn.ReLU()]
        for _ in range(num_layers - 1):
            hidden_layers.append(nn.Linear(hidden_sizes, hidden_sizes))
            hidden_layers.append(nn.ReLU())
        self.hidden_layers = nn.Sequential(*hidden_layers)
        self.actor_layer = nn.Linear(hidden_sizes, num_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, X):
        X = self.hidden_layers(X)
        action_logits = self.actor_layer(X)
        policy = nn.functional.softmax(action_logits, dim=1)
        return policy

    def update(self, returns, action_logits):
        self.optimizer.zero_grad()
        loss = self.loss_fn(returns, action_logits)
        loss.backward()
        self.optimizer.step()

    @staticmethod
    def loss_fn(returns, action_logits):
        batch_size = len(returns)
        policy_losses = [-action_logits[b] * returns[b] for b in range(batch_size)]
        loss = torch.cat(policy_losses).sum() / batch_size
        return loss
