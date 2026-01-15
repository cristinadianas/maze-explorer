# experiments/run_value_iteration.py
import numpy as np

from env.maze_env import MazeEnv
from agents.value_iteration import ValueIterationAgent


def evaluate(agent, env, n_episodes=20, render_first=False):
    # ruleaza mai multe episoade si calculeaza reward-ul si numarul de pasi
    returns = []
    steps_list = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        total_r = 0.0
        steps = 0

        while not done:
            # afiseaza mediul doar pentru primul episod
            if render_first and ep == 0:
                env.render()

            # agentul alege actiunea conform politicii calculate
            a = agent.act(obs)

            # executa pasul in environment
            obs, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated

            total_r += float(r)
            steps += 1

        returns.append(total_r)
        steps_list.append(steps)

    return np.array(returns), np.array(steps_list)


def main():
    # dimensiunea gridului (se poate modifica pentru experimente)
    grid_size = 5
    env = MazeEnv(grid_size=grid_size)

    # initializam agentul value iteration
    agent = ValueIterationAgent(grid_size=grid_size, gamma=0.99, theta=1e-6)

    # calculam value function si politica optima
    agent.fit()

    # evaluam politica obtinuta
    returns, steps = evaluate(agent, env, n_episodes=30, render_first=True)

    print("\n=== Value Iteration ===")
    print(f"Grid size: {grid_size}x{grid_size}")
    print(f"Avg return: {returns.mean():.2f} ± {returns.std():.2f}")
    print(f"Avg steps : {steps.mean():.2f} ± {steps.std():.2f}")
    print(f"Min steps : {steps.min()} | Max steps: {steps.max()}")


if __name__ == "__main__":
    main()
