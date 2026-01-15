import os
import numpy as np
import matplotlib.pyplot as plt

from env.maze_env import MazeEnv
from agents.q_learning import QLearningAgent
from utils.config import (
    GRID_SIZE,
    MAX_EPISODES,
    MAX_STEPS_PER_EPISODE,
    GAMMA,
    LEARNING_RATE,
    EPSILON_START,
    EPSILON_END,
    EPSILON_DECAY,
)


def moving_average(x, window=50):
    if len(x) < window:
        return np.array(x)
    return np.convolve(x, np.ones(window) / window, mode="valid")


def train_one_seed(seed):
    env = MazeEnv(grid_size=GRID_SIZE)
    agent = QLearningAgent(
        grid_size=GRID_SIZE,
        n_actions=env.action_space.n,
        learning_rate=LEARNING_RATE,
        gamma=GAMMA,
        epsilon=EPSILON_START,
        epsilon_end=EPSILON_END,
        epsilon_decay=EPSILON_DECAY,
        seed=seed,
    )

    rewards = []
    steps_list = []
    success = []

    for ep in range(MAX_EPISODES):
        state, _ = env.reset(seed=seed)
        total_reward = 0.0
        done = False
        steps = 0

        for _ in range(MAX_STEPS_PER_EPISODE):
            action = agent.act(state)
            next_state, reward, done, _, _ = env.step(action)
            agent.update(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward
            steps += 1

            if done:
                break

        agent.decay_epsilon()

        rewards.append(total_reward)
        steps_list.append(steps)
        success.append(1 if done else 0)

        if ep % 50 == 0 or ep == MAX_EPISODES - 1:
            print(
                f"[seed {seed} | ep {ep}] "
                f"reward={total_reward:.1f} steps={steps} success={done} eps={agent.epsilon:.3f}"
            )

    return {
        "rewards": np.array(rewards),
        "steps": np.array(steps_list),
        "success": np.array(success),
    }


if __name__ == "__main__":
    seeds = [0, 1, 2]
    window = 50

    plot_dir = os.path.join("plots", "q_learning")
    os.makedirs(plot_dir, exist_ok=True)

    all_rewards = []
    all_steps = []
    all_success = []

    print("\nTRAIN Q-LEARNING")
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

    plt.figure()
    for i, seed in enumerate(seeds):
        ma = moving_average(all_rewards[i], window)
        plt.plot(episodes[-len(ma):], ma, label=f"seed {seed}")
    plt.title("Q-learning: Episode return")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "reward.png"), dpi=150)
    plt.show()

    plt.figure()
    for i, seed in enumerate(seeds):
        ma = moving_average(all_success[i], window)
        plt.plot(episodes[-len(ma):], ma, label=f"seed {seed}")
    plt.title("Q-learning: Success rate")
    plt.xlabel("Episode")
    plt.ylabel("Success (0/1)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "success.png"), dpi=150)
    plt.show()

    plt.figure()
    for i, seed in enumerate(seeds):
        ma = moving_average(all_steps[i], window)
        plt.plot(episodes[-len(ma):], ma, label=f"seed {seed}")
    plt.title("Q-learning: Episode length")
    plt.xlabel("Episode")
    plt.ylabel("Steps")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "steps.png"), dpi=150)
    plt.show()
