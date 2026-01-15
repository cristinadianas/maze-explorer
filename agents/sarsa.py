import numpy as np


class SARSAAgent:
    def __init__(
        self,
        grid_size,
        n_actions,
        learning_rate=0.1,
        gamma=0.99,
        epsilon=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.995,
        seed=None,
    ):
        self.grid_size = grid_size
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = gamma

        self.epsilon = epsilon
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.rng = np.random.default_rng(seed)

        # Same Q-table layout as QLearningAgent: Q[x, y, action]
        self.Q = np.zeros((grid_size, grid_size, n_actions), dtype=np.float32)

    def _state_to_xy(self, state):
        return int(state[0]), int(state[1])

    def act(self, state):
        # epsilon-greedy (same as QLearningAgent)
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))

        x, y = self._state_to_xy(state)
        return int(np.argmax(self.Q[x, y]))

    def update(self, state, action, reward, next_state, next_action, done):
        """
        SARSA update (on-policy):
        Q(s,a) <- Q(s,a) + lr * [r + gamma * Q(s',a') - Q(s,a)]

        If done=True, bootstrap term is 0.
        """
        x, y = self._state_to_xy(state)

        if done:
            td_target = reward
        else:
            nx, ny = self._state_to_xy(next_state)
            td_target = reward + self.gamma * self.Q[nx, ny, next_action]

        td_error = td_target - self.Q[x, y, action]
        self.Q[x, y, action] += self.lr * td_error

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
