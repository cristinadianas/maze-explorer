from dataclasses import dataclass
import random
from collections import deque
from typing import Deque, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, n_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
        )

    def forward(self, x):
        return self.net(x)


@dataclass
class DQNConfig:
    gamma: float = 0.99
    lr: float = 1e-3
    batch_size: int = 64
    buffer_size: int = 50_000
    min_buffer: int = 1_000
    target_update_every: int = 500  # steps
    train_every: int = 1           # steps
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay: float = 0.995


class ReplayBuffer:
    def __init__(self, capacity: int, seed: int = 0):
        self.buffer: Deque[Tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(maxlen=capacity)
        random.seed(seed)

    def push(self, s, a, r, s2, done):
        self.buffer.append((s, a, r, s2, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s2, d = zip(*batch)
        return (
            np.stack(s),
            np.array(a),
            np.array(r, dtype=np.float32),
            np.stack(s2),
            np.array(d, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(self, state_dim: int, n_actions: int, cfg: DQNConfig, seed: int = 0, device: str | None = None):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.cfg = cfg

        self.rng = np.random.default_rng(seed)
        random.seed(seed)
        torch.manual_seed(seed)

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.q = QNetwork(state_dim, n_actions).to(self.device)
        self.q_target = QNetwork(state_dim, n_actions).to(self.device)
        self.q_target.load_state_dict(self.q.state_dict())
        self.q_target.eval()

        self.opt = optim.Adam(self.q.parameters(), lr=cfg.lr)
        self.loss_fn = nn.MSELoss()

        self.buffer = ReplayBuffer(cfg.buffer_size, seed=seed)

        self.epsilon = cfg.eps_start
        self.total_steps = 0

    def act(self, state: np.ndarray) -> int:
        # epsilon-greedy
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))

        s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            qvals = self.q(s)
        return int(torch.argmax(qvals, dim=1).item())

    def push(self, s, a, r, s2, done):
        self.buffer.push(s, a, r, s2, done)

    def decay_epsilon(self):
        self.epsilon = max(self.cfg.eps_end, self.epsilon * self.cfg.eps_decay)

    def maybe_train(self):
        self.total_steps += 1

        if len(self.buffer) < self.cfg.min_buffer:
            return None

        if self.total_steps % self.cfg.train_every != 0:
            return None

        s, a, r, s2, d = self.buffer.sample(self.cfg.batch_size)

        s = torch.tensor(s, dtype=torch.float32, device=self.device)
        a = torch.tensor(a, dtype=torch.int64, device=self.device).unsqueeze(1)
        r = torch.tensor(r, dtype=torch.float32, device=self.device).unsqueeze(1)
        s2 = torch.tensor(s2, dtype=torch.float32, device=self.device)
        d = torch.tensor(d, dtype=torch.float32, device=self.device).unsqueeze(1)

        q_sa = self.q(s).gather(1, a)

        with torch.no_grad():
            max_next = self.q_target(s2).max(dim=1, keepdim=True).values
            target = r + self.cfg.gamma * max_next * (1.0 - d)

        loss = self.loss_fn(q_sa, target)

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        if self.total_steps % self.cfg.target_update_every == 0:
            self.q_target.load_state_dict(self.q.state_dict())

        return float(loss.item())
