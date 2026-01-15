# experiments/run_value_iteration.py
import os
import numpy as np
import matplotlib.pyplot as plt

from env.maze_env import MazeEnv
from agents.value_iteration import ValueIterationAgent
from utils.config import (
    GRID_SIZE,
    MAX_EPISODES,
    MAX_STEPS_PER_EPISODE,
    GAMMA,
)


def moving_average(x, window=50):
    x = np.asarray(x, dtype=np.float32)
    if len(x) < window:
        return x
    return np.convolve(x, np.ones(window, dtype=np.float32) / window, mode="valid")


def run_episode(agent, env, render=False):
    """Run one episode using the agent's computed policy."""
    obs, _ = env.reset()
    total_r = 0.0
    steps = 0

    done = False
    while not done and steps < MAX_STEPS_PER_EPISODE:
        if render:
            env.render()

        a = agent.act(obs)
        obs, r, terminated, truncated, _ = env.step(a)
        done = terminated or truncated

        total_r += float(r)
        steps += 1

    success = 1 if done else 0
    return total_r, steps, success


def train_one_seed(seed: int):
    """
    Value Iteration is not episodic learning, but to match the output/plots
    format of Q-learning/DQN, we:
      1) compute policy once (fit)
      2) "evaluate" it for MAX_EPISODES episodes
      3) log and plot per-episode metrics
    """
    env = MazeEnv(grid_size=GRID_SIZE)

    # Create and solve (plan) once
    agent = ValueIterationAgent(
        grid_size=GRID_SIZE,
        n_actions=env.action_space.n,
        gamma=GAMMA,
        theta=1e-6,
    )
    agent.fit()

    rewards = []
    steps_list = []
    success = []

    for ep in range(MAX_EPISODES):
        render_first = (ep == 0)

        total_r, steps, succ = run_episode(agent, env, render=render_first)

        rewards.append(total_r)
        steps_list.append(steps)
        success.append(succ)

        if ep % 50 == 0 or ep == MAX_EPISODES - 1:
            print(
                f"[seed {seed} | ep {ep}] "
                f"reward={total_r:.1f} steps={steps} success={bool(succ)} eps=0.000"
            )

    return {
        "rewards": np.array(rewards, dtype=np.float32),
        "steps": np.array(steps_list, dtype=np.int32),
        "success": np.array(success, dtype=np.int32),
    }


if __name__ == "__main__":
    seeds = [0, 1, 2]
    window = 50

    plot_dir = os.path.join("plots", "value_iteration")
    os.makedirs(plot_dir, exist_ok=True)

    all_rewards, all_steps, all_success = [], [], []

    print("\nTRAIN VALUE ITERATION")
    for seed in seeds:
        data = train_one_seed(seed)
        all_rewards.append(data["rewards"])
        all_steps.append(data["steps"])
        all_success.append(data["success"])

    all_rewards = np.array(all_rewards)
    all_steps = np.array(all_steps)
    all_success = np.array(all_success)

    last = 100
    print("\nFINAL TRAIN STATS (last 100 episodes)")
    print(f"Avg reward: {all_rewards[:, -last:].mean():.2f} ± {all_rewards[:, -last:].std():.2f}")
    print(f"Success rate: {all_success[:, -last:].mean():.2f}")
    print(f"Avg steps: {all_steps[:, -last:].mean():.2f} ± {all_steps[:, -last:].std():.2f}")

    episodes = np.arange(MAX_EPISODES)

    # reward.png
    plt.figure()
    for i, seed in enumerate(seeds):
        ma = moving_average(all_rewards[i], window)
        plt.plot(episodes[-len(ma):], ma, label=f"seed {seed}")
    plt.title("Value Iteration: Episode return")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "reward.png"), dpi=150)
    plt.show()

    # success.png
    plt.figure()
    for i, seed in enumerate(seeds):
        ma = moving_average(all_success[i], window)
        plt.plot(episodes[-len(ma):], ma, label=f"seed {seed}")
    plt.title("Value Iteration: Success rate")
    plt.xlabel("Episode")
    plt.ylabel("Success (0/1)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "success.png"), dpi=150)
    plt.show()

    # steps.png
    plt.figure()
    for i, seed in enumerate(seeds):
        ma = moving_average(all_steps[i], window)
        plt.plot(episodes[-len(ma):], ma, label=f"seed {seed}")
    plt.title("Value Iteration: Episode length")
    plt.xlabel("Episode")
    plt.ylabel("Steps")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "steps.png"), dpi=150)
    plt.show()
