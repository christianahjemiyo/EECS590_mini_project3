"""
agents.py
==========
Tabular RL agents for reactor control.

Implements:
  1. Q-Learning  — off-policy TD(0)
  2. SARSA       — on-policy TD(0)
  3. SARSA(λ)    — on-policy with eligibility traces (backward view)

Key design notes (from synthesis):
  - Q-learning is optimistic: max_a Q(s',a') target assumes greedy future
    behavior. This optimism is worst when ε is high (early training) and
    fades as ε → ε_min. This explains why Q-learning shows higher meltdown
    rates DURING training but converges to a similar final policy.
  - SARSA accounts for actual exploratory actions → implicitly assigns lower
    value to states near the meltdown boundary → safer during training.
  - SARSA(λ): eligibility traces are essential for the warm-up problem.
    During cold-start steps reward is exactly 0. TD(0) receives no gradient
    signal for warm-up states. λ > 0 lets productive-zone rewards propagate
    backward through traces to credit/blame early decisions.
"""

import numpy as np
from abc import ABC, abstractmethod


class BaseAgent(ABC):
    def __init__(self, n_states, n_actions, gamma=0.95, alpha=0.1,
                 epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.995):
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.Q = np.zeros((n_states, n_actions))

    def choose_action(self, state: int) -> int:
        """ε-greedy selection."""
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.Q[state]))

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def reset_episode(self):
        pass

    @abstractmethod
    def update(self, *args, **kwargs):
        pass


class QLearningAgent(BaseAgent):
    """
    Off-policy TD(0).

    Target: r + γ·max_{a'} Q(s', a')
    Optimistic because it assumes greedy behavior from s' regardless of
    what the behavior policy (ε-greedy) will actually do next.
    """
    def update(self, s, a, r, s_next, done):
        target = r if done else r + self.gamma * np.max(self.Q[s_next])
        self.Q[s, a] += self.alpha * (target - self.Q[s, a])


class SARSAAgent(BaseAgent):
    """
    On-policy TD(0).

    Target: r + γ·Q(s', a')  where a' is the ACTUAL next action.
    Conservative: accounts for exploratory noise in value estimates.
    Safer near absorbing meltdown boundary during training.
    """
    def update(self, s, a, r, s_next, a_next, done):
        target = r if done else r + self.gamma * self.Q[s_next, a_next]
        self.Q[s, a] += self.alpha * (target - self.Q[s, a])


class SARSALambdaAgent(BaseAgent):
    """
    SARSA with replacing eligibility traces (backward view of TD(λ)).

    δ_t  = r + γ·Q(s',a') − Q(s,a)
    e(s,a) ← γλ·e(s,a)  for all (s,a)   [decay]
    e(s_t,a_t) ← 1                        [replacing trace: spike]
    Q(s,a) ← Q(s,a) + α·δ_t·e(s,a)

    λ=0: equivalent to SARSA (no trace propagation)
    λ=1: approaches Monte Carlo (full-episode credit)
    λ=0.8: our choice — captures multi-step warm-up credit assignment
            without full Monte Carlo variance.
    """
    def __init__(self, *args, lam=0.8, **kwargs):
        super().__init__(*args, **kwargs)
        self.lam = lam
        self.e = np.zeros((self.n_states, self.n_actions))

    def reset_episode(self):
        self.e[:] = 0.0

    def update(self, s, a, r, s_next, a_next, done):
        target = r if done else r + self.gamma * self.Q[s_next, a_next]
        delta = target - self.Q[s, a]
        self.e *= self.gamma * self.lam
        self.e[s, a] = 1.0          # replacing trace
        self.Q += self.alpha * delta * self.e
        if done:
            self.e[:] = 0.0
