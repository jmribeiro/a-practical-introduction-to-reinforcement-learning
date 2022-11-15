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

        self.network = ...  # FIXME
        self.gamma = discount_factor

        self.rewards_buffer = []
        self.action_logits_buffer = []
        self.last_policy = None

    def action(self, observation):
        # TODO
        raise NotImplementedError()

    def step(self, timestep):
        # TODO
        if self.trainable:
            raise NotImplementedError()

    def train(self):
        self.trainable = True

    def eval(self):
        self.trainable = False
