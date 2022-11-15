import argparse
import math
from collections import defaultdict
from datetime import datetime
from typing import List

import gym
import numpy as np

from utils_execution import requires_training, train_agent, agent_factory, run_episode
from utils_plotting import plot_learning_curves, plot_rewards


def compare_agents(
        agent_names: List[str], environment_name: str, num_train_trials: int,
        episodes: int, max_timesteps: int, render_last_three: bool, resources: str):

    environment = gym.make(environment_name)

    rewards = defaultdict(lambda: {})
    reward_learning_curves = defaultdict(lambda: {})

    for trial_number in range(num_train_trials):

        print(f"Running on {environment_name} (Trial {trial_number+1}/{num_train_trials})")

        for agent_name in agent_names:

            # ##### #
            # Train #
            # ##### #

            if requires_training(agent_name):
                print(f"\t- {agent_name} (Training)")
                agent, reward_learning_curves[agent_name][trial_number] = train_agent(
                    agent_name, environment_name, resources
                )
            else:
                agent = agent_factory(agent_name, environment)

            # ######## #
            # Evaluate #
            # ######## #

            print(f"\t- {agent_name} (Evaluating {episodes} episodes)")
            rewards[agent_name][trial_number] = [
                run_episode(agent, environment, max_timesteps, (episode >= episodes-3 and render_last_three))
                for episode in range(episodes)
            ]

            trial = np.array(rewards[agent_name][trial_number])
            print(f"\t\tAvg. Accumulated Ep. Reward: {round(trial.mean(), 2)} (+-{round(trial.std(), 2)})")

    # ############# #
    # Plot Training #
    # ############# #

    stamp = datetime.now().strftime("%d-%m-%Y--%H:%M:%S")

    print(f"\nPlotting to Learning Curves ({stamp})", end="... ")
    plot_learning_curves(stamp, environment_name, reward_learning_curves)
    print("Done!")

    # ############### #
    # Plot Evaluation #
    # ############### #

    print(f"\nPlotting Evaluation ({stamp})", end="... ")
    plot_rewards(stamp, environment_name, agent_names, rewards)
    print("Done!")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--environment", type=str, default="CartPole-v0")
    parser.add_argument('--agents', nargs='+', default=[
        "ActorCritic",
        "Random",
    ])

    parser.add_argument("--num-train-trials", default=3, type=int)
    parser.add_argument("--num-eval-episodes", default=8, type=int)
    parser.add_argument("--render-last-three", action="store_true")
    parser.add_argument("--max-timesteps", default=math.inf, type=int)
    parser.add_argument("--resources", type=str, default="resources")

    opt = parser.parse_args()
    opt.agents = [agent.replace("_", " ") for agent in opt.agents]

    compare_agents(
        opt.agents, opt.environment,
        opt.num_train_trials, opt.num_eval_episodes, opt.max_timesteps, opt.render_last_three, opt.resources
    )
