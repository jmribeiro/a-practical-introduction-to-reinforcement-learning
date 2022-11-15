import gym
import numpy as np
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


class PolicyNetwork(nn.Module):

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


def compute_action(network, state):
    x = torch.from_numpy(state).float().unsqueeze(0)
    actor_output = network(x)
    last_policy = Categorical(actor_output)
    action = last_policy.sample()
    return action.item(), last_policy


def reinforce(network, action, reward, terminal, last_policy, action_logits_buffer, rewards_buffer, gamma):
    action_logits_buffer.append(last_policy.log_prob(torch.tensor(action)))
    rewards_buffer.append(reward)
    if terminal:
        returns = []
        current_return = 0
        for reward in reversed(rewards_buffer):
            current_return = reward + gamma * current_return
            returns = [current_return] + returns
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + np.finfo(np.float32).eps.item())
        network.update(returns, action_logits_buffer)
        rewards_buffer.clear()
        action_logits_buffer.clear()


def run_episode(env, network, rewards_buffer, action_logits_buffer, gamma):

    """
        Runs a full episode on the environment
    """

    ep_reward = 0
    ep_steps = 0

    observation = env.reset()
    terminal = False
    while not terminal:

        action, last_policy = compute_action(network, observation)
        observation, reward, terminal, _ = env.step(action)

        reinforce(network, action, reward, terminal, last_policy, action_logits_buffer, rewards_buffer, gamma)

        ep_reward += reward
        ep_steps += 1

    return ep_reward, ep_steps


if __name__ == '__main__':

    env = gym.make("CartPole-v0")
    num_features = env.observation_space.shape[0]
    num_actions = env.action_space.n
    num_layers = 1
    hidden_sizes = 128
    learning_rate = 0.01
    gamma = 0.99
    
    network = PolicyNetwork(num_features, num_actions, num_layers, hidden_sizes, learning_rate)
    rewards_buffer = []
    action_logits_buffer = []

    running_reward = 10

    # Run infinitely many episodes
    for episode in count(1):

        ep_reward, ep_steps = run_episode(env, network, rewards_buffer, action_logits_buffer, gamma)

        # update cumulative reward
        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward

        # Log results
        if episode % 10 == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                episode, ep_reward, running_reward))

        # Check if we have "solved" the problem
        if running_reward > env.spec.reward_threshold:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, ep_steps))
            break
