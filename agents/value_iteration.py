# agents/value_iteration.py
import numpy as np


class ValueIterationAgent:
    """
    agent value iteration pentru MazeEnv
    algoritm tabular, determinist
    """

    def __init__(
        self,
        grid_size: int,
        n_actions: int = 4,
        gamma: float = 0.99,
        theta: float = 1e-6,
        max_iters: int = 10_000,
        goal_reward: float = 100.0,
        step_reward: float = -1.0,
    ):
        # dimensiunea gridului
        self.grid_size = grid_size

        # numar actiuni (up, down, left, right)
        self.n_actions = n_actions

        # factor de discount
        self.gamma = gamma

        # prag de convergenta
        self.theta = theta

        # numar maxim de iteratii
        self.max_iters = max_iters

        # reward-uri din environment
        self.goal_reward = goal_reward
        self.step_reward = step_reward

        # value function V(x, y)
        self.V = np.zeros((grid_size, grid_size), dtype=np.float32)

        # politica optima pi(x, y) -> actiune
        self.pi = np.zeros((grid_size, grid_size), dtype=np.int64)

        # pozitia finala (goal)
        self.goal = (grid_size - 1, grid_size - 1)

    def _is_terminal(self, x: int, y: int) -> bool:
        # verifica daca starea este terminala
        return (x, y) == self.goal

    def _transition(self, x: int, y: int, a: int):
        """
        tranzitie determinista identica cu MazeEnv.step()
        0=up, 1=down, 2=left, 3=right
        """
        # daca suntem deja in goal, ramanem acolo
        if self._is_terminal(x, y):
            return x, y, 0.0, True

        nx, ny = x, y

        if a == 0:        # up
            nx = max(x - 1, 0)
        elif a == 1:      # down
            nx = min(x + 1, self.grid_size - 1)
        elif a == 2:      # left
            ny = max(y - 1, 0)
        elif a == 3:      # right
            ny = min(y + 1, self.grid_size - 1)

        done = self._is_terminal(nx, ny)
        reward = self.goal_reward if done else self.step_reward

        return nx, ny, reward, done

    def fit(self):
        """
        aplica value iteration pana la convergenta
        apoi construieste politica optima
        """
        for _ in range(self.max_iters):
            delta = 0.0

            # parcurgem toate starile
            for x in range(self.grid_size):
                for y in range(self.grid_size):
                    if self._is_terminal(x, y):
                        continue

                    old_v = self.V[x, y]

                    # calculam valoarea maxima peste actiuni
                    q_best = -1e9
                    for a in range(self.n_actions):
                        nx, ny, r, done = self._transition(x, y, a)
                        q = r + self.gamma * (0.0 if done else self.V[nx, ny])
                        if q > q_best:
                            q_best = q

                    self.V[x, y] = q_best
                    delta = max(delta, abs(old_v - q_best))

            # verificare convergenta
            if delta < self.theta:
                break

        # extragem politica greedy din V
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                if self._is_terminal(x, y):
                    self.pi[x, y] = 0
                    continue

                q_vals = np.empty(self.n_actions, dtype=np.float32)
                for a in range(self.n_actions):
                    nx, ny, r, done = self._transition(x, y, a)
                    q_vals[a] = r + self.gamma * (0.0 if done else self.V[nx, ny])

                self.pi[x, y] = int(np.argmax(q_vals))

        return self

    def act(self, state: np.ndarray) -> int:
        # alege actiunea optima din politica calculata
        x, y = int(state[0]), int(state[1])
        return int(self.pi[x, y])
