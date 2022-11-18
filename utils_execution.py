import math

import numpy as np
import yaml
from tqdm import tqdm
from typing import Optional

import gym
from gym import Env

from agents import Timestep, Agent, Random, ActorCritic, REINFORCE


def run_episode(agent: Agent, environment: Env, max_timesteps, render):

    episode_reward = 0.0
    observation = environment.reset()

    if render:
        environment.render()

    step = 0
    terminal = False
    while not terminal and step < max_timesteps:

        action = agent.action(observation)
        next_observation, reward, terminal, info = environment.step(action)
        agent.step(Timestep(observation, action, reward, next_observation, terminal, info))

        observation = next_observation

        if render:
            environment.render()

        episode_reward += reward
        step += 1

    if render:
        environment.close()

    return episode_reward


def agent_factory(agent_name: str, environment: Env, hyperparameters: Optional[dict] = None):

    if agent_name == "Random":
        num_actions = environment.action_space.n
        return Random(num_actions)

    elif agent_name == "ActorCritic":

        num_features = environment.observation_space.shape[0]
        num_actions = environment.action_space.n

        hidden_sizes = hyperparameters["hidden sizes"]
        num_layers = hyperparameters["num layers"]

        learning_rate = float(hyperparameters["learning rate"])
        discount_factor = hyperparameters["discount factor"]

        return ActorCritic(num_features, num_actions, num_layers, hidden_sizes, learning_rate, discount_factor)

    elif agent_name == "REINFORCE":

        num_features = environment.observation_space.shape[0]
        num_actions = environment.action_space.n

        hidden_sizes = hyperparameters["hidden sizes"]
        num_layers = hyperparameters["num layers"]

        learning_rate = float(hyperparameters["learning rate"])
        discount_factor = hyperparameters["discount factor"]

        return REINFORCE(num_features, num_actions, num_layers, hidden_sizes, learning_rate, discount_factor)

    else:
        raise ValueError(f"{agent_name} unknown on agent_factory.")


def requires_training(agent_name: str):
    # If more Deep RL agents were developed, their name would be added here
    return agent_name in ["ActorCritic", "REINFORCE"]


def train_agent(agent_name: str, environment_name: str, resources: str):

    with open(f"{resources}/hyperparameters/{agent_name}-{environment_name}.yaml", "r") as file:
        hyperparameters = yaml.load(file, Loader=yaml.Loader)

    training_episodes = hyperparameters["training episodes"]

    environment = gym.make(environment_name)
    agent = agent_factory(agent_name, environment, hyperparameters)
    agent.train()
    reward_learning_curve = np.array([
        run_episode(agent, environment, max_timesteps=math.inf, render=False)
        for _ in tqdm(range(training_episodes), desc="\t\tProgress")
    ])
    agent.eval()
    return agent, reward_learning_curve
