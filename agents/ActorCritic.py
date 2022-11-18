import pathlib
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from agents import Agent, Timestep


class ActorCritic(Agent):

    """
        Actor-Critic Agent
        Implemented in PyTorch
        (see ActorCriticNetwork class below)
        (see https://proceedings.neurips.cc/paper/1999/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf for more info)
    """

    def __init__(self,
                 num_features: int, num_actions: int,
                 num_layers: int, hidden_sizes: int,
                 learning_rate: float, discount_factor: float,
                 name="Actor Critic"):

        super(ActorCritic, self).__init__(name)

        self.network = PolicyValueNetwork(num_features, num_actions, num_layers, hidden_sizes, learning_rate)
        self.gamma = discount_factor

        self.rewards_buffer = []
        self.values_buffer = []
        self.action_logits_buffer = []

        self.last_policy = None
        self.last_value = None

        self.trainable = True

    # ############### #
    # Agent Interface #
    # ############### #

    def action(self, observation: np.ndarray):
        x = torch.from_numpy(observation).float()
        self.last_policy, self.last_value = self.network(x)
        self.last_policy = Categorical(self.last_policy)
        action = self.last_policy.sample()
        return action.item()

    def step(self, timestep: Timestep):
        if self.trainable:
            self.action_logits_buffer.append(self.last_policy.log_prob(torch.tensor(timestep.action)))
            self.values_buffer.append(self.last_value)
            self.rewards_buffer.append(timestep.reward)
            if timestep.terminal:
                self.learning_step()

    # ######### #
    # Auxiliary #
    # ######### #

    def learning_step(self):
        returns = self.compute_returns()
        self.network.update(returns, self.action_logits_buffer, self.values_buffer)
        self.rewards_buffer.clear()
        self.values_buffer.clear()
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


class PolicyValueNetwork(nn.Module):

    def __init__(self, num_features: int, num_actions: int, num_layers: int, hidden_sizes: int, learning_rate: float):
        super(PolicyValueNetwork, self).__init__()
        hidden_layers = [nn.Linear(num_features, hidden_sizes), nn.ReLU()]
        for _ in range(num_layers - 1):
            hidden_layers.append(nn.Linear(hidden_sizes, hidden_sizes))
            hidden_layers.append(nn.ReLU())
        self.hidden_layers = nn.Sequential(*hidden_layers)
        self.actor_layer = nn.Linear(hidden_sizes, num_actions)
        self.critic_layer = nn.Linear(hidden_sizes, out_features=1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, X):
        X = self.hidden_layers(X)
        action_logits = self.actor_layer(X)
        values = self.critic_layer(X)
        policy = nn.functional.softmax(action_logits, dim=-1)
        return policy, values

    def update(self, returns, action_logits, values):
        self.optimizer.zero_grad()
        loss = self.loss_fn(returns, action_logits, values)
        loss.backward()
        self.optimizer.step()

    @staticmethod
    def loss_fn(returns, action_logits, values):

        batch_size = len(returns)

        actor_losses = []
        critic_losses = []
        for b in range(batch_size):

            advantage = returns[b] - values[b].item()

            actor_loss = - action_logits[b] * advantage
            critic_loss = nn.functional.smooth_l1_loss(values[b], torch.tensor([returns[b]]))

            actor_losses.append(actor_loss)
            critic_losses.append(critic_loss)

        loss = (torch.stack(actor_losses).sum() + 0.5 * torch.stack(critic_losses).sum()) / batch_size

        return loss

    def save(self, directory):
        pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), f"{directory}/model.pt")
        torch.save(self.optimizer.state_dict(), f"{directory}/optimizer.pt")

    def load(self, directory):
        model_state_dict = torch.load(f"{directory}/model.pt")
        optimizer_state_dict = torch.load(f"{directory}/optimizer.pt")
        self.load_state_dict(model_state_dict)
        self.optimizer.load_state_dict(optimizer_state_dict)
