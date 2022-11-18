import numpy as np
from abc import ABC, abstractmethod

from agents import Timestep


class Agent(ABC):

    """
        Base Agent Class
        (Abstract)
    """

    def __init__(self, name):
        self.name = name

    @abstractmethod
    def action(self, observation: np.ndarray):
        pass

    @abstractmethod
    def step(self, timestep: Timestep):
        pass
