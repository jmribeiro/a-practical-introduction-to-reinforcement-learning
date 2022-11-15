import argparse
import math
import random

import numpy as np

import gym
import torch

from agents import ActorCritic
from utils_execution import run_episode
from utils_plotting import plot_learning_curves

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--environment",            type=str,   default="LunarLander-v2")
    parser.add_argument("--num-training-episodes",  type=int,   default=1000)
    parser.add_argument("--num-demo-episodes",      type=int,   default=3)
    parser.add_argument("--seed",                   type=int,   default=42)

    opt = parser.parse_args()

    env = gym.make(opt.environment)

    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    env.seed(opt.seed)

    agent = ActorCritic(
        num_features=env.observation_space.shape[0],
        num_actions=env.action_space.n,
        num_layers=1,
        hidden_sizes=512,
        learning_rate=0.01,
        discount_factor=0.99
    )

    print(f"Demo for untrained agent ({opt.num_demo_episodes} episodes)")
    agent.eval()
    for episode in range(opt.num_demo_episodes):
        reward = run_episode(agent, env, max_timesteps=math.inf, render=True)
        print(f"Demo Episode {episode+1}/{opt.num_demo_episodes}: Reward={round(reward, 3)}")

    print(f"\n\nTraining for {opt.num_training_episodes} episodes")
    agent.train()
    rewards = []
    for episode in range(opt.num_training_episodes):
        reward = run_episode(agent, env, max_timesteps=math.inf, render=False)
        rewards.append(reward)
        print(f"Training Episode {episode+1}/{opt.num_training_episodes}: Reward = {round(reward, 3)}", end="\r")

    print(f"\nPlotting Learning Curves to Plot-Demo.png")
    plot_learning_curves("Demo", "MountainCar-v0", {"ActorCritic": {opt.seed: np.array(rewards)}})

    print(f"\n\nDemo for trained agent ({opt.num_demo_episodes} episodes)")
    agent.eval()
    for episode in range(opt.num_demo_episodes):
        reward = run_episode(agent, env, max_timesteps=math.inf, render=True)
        print(f"Demo Episode {episode + 1}/{opt.num_demo_episodes}: Reward={round(reward, 3)}")
