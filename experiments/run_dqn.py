import os
import numpy as np
import matplotlib.pyplot as plt

from env.maze_env import MazeEnv
from agents.dqn import DQNAgent, DQNConfig

from utils.config import (
    GRID_SIZE,
    MAX_EPISODES,
    MAX_STEPS_PER_EPISODE,
)


def moving_average(x, window=50):
    if len(x) < window:
        return np.array(x)
    return np.convolve(x, np.ones(window) / window, mode="valid")


def train_one_seed(seed: int, cfg: DQNConfig):
    env = MazeEnv(grid_size=GRID_SIZE)
    state_dim = 2
    n_actions = env.action_space.n

    agent = DQNAgent(state_dim=state_dim, n_actions=n_actions, cfg=cfg, seed=seed)

    rewards = []
    steps_list = []
    success = []

    for ep in range(MAX_EPISODES):
        state, _ = env.reset(seed=seed)
        state = state.astype(np.float32)

        total_reward = 0.0
        done = False
        steps = 0

        for _ in range(MAX_STEPS_PER_EPISODE):
            action = agent.act(state)
            next_state, reward, done, _, _ = env.step(action)
            next_state = next_state.astype(np.float32)

            agent.push(state, action, float(reward), next_state, bool(done))
            agent.maybe_train()

            state = next_state
            total_reward += float(reward)
            steps += 1

            if done:
                break

        agent.decay_epsilon()

        rewards.append(total_reward)
        steps_list.append(steps)
        success.append(1 if done else 0)

        if ep % 50 == 0 or ep == MAX_EPISODES - 1:
            print(
                f"[DQN seed {seed} | ep {ep}] "
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

    plot_dir = os.path.join("plots", "dqn")
    os.makedirs(plot_dir, exist_ok=True)

    cfg = DQNConfig(
        gamma=0.99,
        lr=1e-3,
        batch_size=64,
        buffer_size=50_000,
        min_buffer=1_000,
        target_update_every=500,
        train_every=1,
        eps_start=1.0,
        eps_end=0.05,
        eps_decay=0.995,
    )

    all_rewards, all_steps, all_success = [], [], []

    print("\n TRAIN DQN")
    for seed in seeds:
        data = train_one_seed(seed, cfg)
        all_rewards.append(data["rewards"])
        all_steps.append(data["steps"])
        all_success.append(data["success"])

    all_rewards = np.array(all_rewards)
    all_steps = np.array(all_steps)
    all_success = np.array(all_success)

    last = 100
    print("\n FINAL TRAIN STATS")
    print(f"Avg reward: {all_rewards[:, -last:].mean():.2f} ± {all_rewards[:, -last:].std():.2f}")
    print(f"Success rate: {all_success[:, -last:].mean():.2f}")
    print(f"Avg steps: {all_steps[:, -last:].mean():.2f} ± {all_steps[:, -last:].std():.2f}")

    episodes = np.arange(MAX_EPISODES)

    plt.figure()
    for i, seed in enumerate(seeds):
        ma = moving_average(all_rewards[i], window)
        plt.plot(episodes[-len(ma):], ma, label=f"seed {seed}")
    plt.title("DQN: Episode return")
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
    plt.title("DQN: Success rate")
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
    plt.title("DQN: Episode length")
    plt.xlabel("Episode")
    plt.ylabel("Steps")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "steps.png"), dpi=150)
    plt.show()
