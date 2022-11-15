import math
from collections import defaultdict
from typing import List, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


def z_table(confidence):
    return {
        0.99: 2.576,
        0.95: 1.96,
        0.90: 1.645
    }[confidence]


def confidence_interval(mean: float, num_samples: int, confidence: float):
    return z_table(confidence) * (mean / math.sqrt(num_samples))


def standard_error(std_dev: float, num_samples: int, confidence: float):
    return z_table(confidence) * (std_dev / math.sqrt(num_samples))


def plot_confidence_bar(stamp: str, names: List[str], means: List[float], std_devs: List[float], N: List[int],
                        title: str, y_label: str, confidence: float=0.95, colors: Optional[List[str]]=None):

    names = [name.replace(" ", "\n") for name in names]
    errors = [standard_error(std_devs[i], N[i], confidence) for i in range(len(means))]
    fig, ax = plt.subplots()
    x_pos = np.arange(len(names))
    ax.bar(x_pos, means,
           yerr=errors, align='center', alpha=0.5, color=colors if colors is not None else "gray",
           ecolor='black', capsize=10)
    ax.set_ylabel(y_label)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names)
    ax.set_title(title)
    ax.yaxis.grid(True)
    plt.tight_layout()
    plt.savefig(f"Plot-{stamp}-evaluation.png")


class RunStats:

    def __init__(self, confidence=0.95):
        self.runs = []
        self.confidence = confidence
        self.N = 0

    def add(self, run):
        if len(self.runs) == 0:
            self.num_measurements = run.size
            self.means = np.zeros((self.num_measurements,))
            self.std_devs = np.zeros((self.num_measurements,))
            self.errors = np.zeros((self.num_measurements,))
        self.runs.append(run)
        self.recompute_errors(self.confidence)
        self.N = len(self.runs)

    def recompute_errors(self, confidence, max_n=math.inf):
        num_runs = len(self.runs)
        num_runs = min(max_n, num_runs)
        runs = np.array(self.runs[:num_runs])
        for measure in range(self.num_measurements):
            if runs[:,].shape[1] > measure:
                column = runs[:, measure]
                self.means[measure] = column.mean()
                self.std_devs[measure] = column.std()
                self.errors[measure] = standard_error(self.std_devs[measure], num_runs, confidence)
                self.N = num_runs


class LinePlot:

    def __init__(self, title,
                 x_label, y_label,
                 x_tick_step=1,
                 confidence=0.95,
                 ymin=None, ymax=None, colors=None):

        self._title = title

        self._stats = defaultdict(lambda: RunStats(confidence))
        self._colors = {} if colors is None else colors
        self._markers = {}

        self._legend = {}

        self._x_label = x_label
        self._y_label = y_label

        self._x_tick_step = x_tick_step

        self._current_figure = None
        self._highest_y_value = 1

        self._ymin = ymin
        self._ymax = ymax

    @property
    def has_runs(self):
        return self.num_runs() > 0

    def num_runs(self, agent_name=None):
        if agent_name is not None:
            stats = self._stats[agent_name]
            return stats.N
        else:
            total_runs = 0
            for agent in self._stats:
                total_runs += self.num_runs(agent)
            return total_runs

    def add_run(self, agent_name, run, color=None, add_to_legend=True, marker="o"):

        if color is not None:
            self._colors[agent_name] = color

        self._markers[agent_name] = marker
        self._legend[agent_name] = add_to_legend

        # First measurement added for agent, create color
        if agent_name not in self._colors:
            color = self._random_color(list(self._colors.values()))
            self._colors[agent_name] = color

        stats = self._stats[agent_name]
        stats.add(run)

        self._highest_y_value = max(int(run.max()) + 1, self._highest_y_value)

    def show(self, error_fill=True, error_fill_transparency=0.25):
        if len(self._stats) > 0:
            if self._current_figure is not None: plt.close(self._current_figure)
            self._make_fig(error_fill, error_fill_transparency)
            self._current_figure.show()
            plt.close(self._current_figure)
            del self._current_figure
            self._current_figure = None

    def savefig(self, filename=None, error_fill=True, error_fill_transparency=0.25):
        if len(self._stats) > 0:
            if filename is None: filename = self._title
            if self._current_figure is not None: plt.close(self._current_figure)
            self._make_fig(error_fill, error_fill_transparency)
            self._current_figure.savefig(filename)
            plt.close(self._current_figure)
            del self._current_figure
            self._current_figure = None

    def _make_fig(self, error_fill=True, error_fill_transparency=0.25, show_legend=True):

        self._current_figure, ax = plt.subplots(1)

        for agent_name, stats in self._stats.items():

            num_runs = stats.N
            means = stats.means
            errors = stats.errors

            color = self._colors[agent_name]
            marker = self._markers[agent_name]
            x_ticks = (np.arange(means.size) + 1) * self._x_tick_step
            x_ticks = np.array(x_ticks, dtype=int)
            if self._legend[agent_name]:
                ax.plot(x_ticks, means, lw=2, label=f"{agent_name} (N={num_runs})", color=color, marker=marker)
            else:
                ax.plot(x_ticks, means, lw=2, color=color, marker=marker)

            if error_fill:
                ax.fill_between(x_ticks, means + errors, means - errors, facecolor=color, alpha=error_fill_transparency)

        ax.set_title(self._title)
        ax.set_xlabel(self._x_label)
        ax.set_ylabel(self._y_label)

        if self._ymin is not None:
            ax.set_ylim(bottom=self._ymin)
        else:
            ax.set_ylim(top=self._highest_y_value)

        if self._ymax is not None:
            ax.set_ylim(top=self._ymax)

        if show_legend:
            ax.legend()

        ax.grid()

    @staticmethod
    def _random_color(excluded_colors):
        excluded_colors = excluded_colors or []
        color_map = plt.get_cmap('gist_rainbow')
        if len(excluded_colors) == 0:
            color = color_map(np.random.uniform())
        else:
            color = excluded_colors[0]
            while color in excluded_colors:
                color = color_map(np.random.uniform())
        return color


def compute_avg(run):
    avg_run = np.zeros_like(run)
    for i in range(run.size):
        avg_run[i] = run[:i+1].mean()
    return avg_run


def plot_learning_curves(stamp: str, environment_name: str, learning_curves: dict, confidence: float = 0.95, window=-1):

    runs = defaultdict(lambda: [])

    for agent_name in learning_curves:
        for seed_number, seed_learning_curve in learning_curves[agent_name].items():
            seed_learning_curve_avg = compute_avg(seed_learning_curve)
            runs[agent_name].append(seed_learning_curve_avg)

    ylabel = "Avg. Acc Reward" if window == -1 else f"Avg. Last {window} Eps Acc Reward"
    plot = LinePlot(f"Agents learning on {environment_name}", "Training Episode", ylabel, 1, confidence=confidence)
    for agent_name, agent_runs in runs.items():
        for run in agent_runs:
            plot.add_run(agent_name, run, marker=",")
    plot.savefig(filename=f"Plot-{stamp}-learning-curves.png")


def plot_rewards(stamp: str, environment_name: str, agent_names: List[str], rewards: dict):
    means, std_devs, N = [], [], []
    for agent_name in agent_names:
        agent_rewards = []
        for seed_number, seed_rewards in rewards[agent_name].items():
            agent_rewards.extend(seed_rewards)
        agent_rewards = np.array(agent_rewards)
        means.append(agent_rewards.mean())
        std_devs.append(agent_rewards.std())
        N.append(agent_rewards.size)
    title = f"Agents on {environment_name}"
    plot_confidence_bar(stamp, agent_names, means, std_devs, N, title, "Average Accumulated Ep. Reward")

