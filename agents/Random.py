import numpy as np

from random import randrange

from agents import Agent, Timestep


class Random(Agent):

    """
        Random Agent
        (Selects actions at random)
    """

    def __init__(self, num_actions: int, name="Random"):
        super().__init__(name)
        self.num_actions = num_actions

    def action(self, observation: np.ndarray) -> int:
        return randrange(self.num_actions)

    def step(self, timestep: Timestep) -> dict:
        pass
