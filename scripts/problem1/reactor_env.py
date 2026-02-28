"""
reactor_env_v2.py
==================
Upgraded Cadmium Rod Reactor Environment — Synthesis Version

KEY UPGRADE over v1:
--------------------
Bin edges are now fixed to the TRUE PHYSICAL RANGE [mu_min, mu_max],
NOT to [mu_min - 3σ, mu_max + 3σ].

Rationale: When comparing experiments across different σ² values, using
σ-dependent bin edges means each noise level produces a different state
space. The agent at σ²=0.5 and the agent at σ²=2.0 would be solving
different MDPs, making comparison meaningless. By fixing bin edges to
the physical range, the state space is identical across all noise
experiments — σ² only affects how reliably the agent's observation
falls in the correct bin. This is the only experimental design that
isolates noise as a single variable.

Observations outside [mu_min, mu_max] (which happen due to sensor noise)
are clipped to the boundary bins, which is physically correct — the
agent cannot distinguish "very cold" from "at minimum" from a single
reading below the physical floor.
"""

import numpy as np


class ReactorEnv:
    """
    Partially Observable reactor control environment.

    The true state is the latent reactivity µt ∈ [µ_min, µ_max].
    The agent observes only zt ~ N(µt, σ²) — a noisy sensor reading.
    State space is the DISCRETIZED OBSERVATION, binned over the FIXED
    physical range [µ_min, µ_max] regardless of σ².

    Parameters
    ----------
    n_bins : int
        Number of uniform bins over [µ_min, µ_max].
    k : int
        Max rod increment. Actions ∈ {-k,...,0,...,+k}.
    sigma2 : float
        Sensor noise variance σ².
    sigma2_T : float
        Process noise variance σ²_T.
    sigma2_R : float
        Reward noise variance σ²_R.
    alpha : float
        Rod effectiveness per increment.
    delta : float
        Intrinsic drift rate above µ_hot.
    mu_min, mu_max : float
        Physical bounds (µ_max = meltdown threshold).
    mu_hot : float
        Drift onset threshold.
    mu_lo, mu_hi : float
        Productive power-generation range.
    c : float
        Rod movement cost per unit |a|.
    M : float
        Meltdown penalty.
    T : int
        Episode horizon.
    """

    def __init__(
        self,
        n_bins: int = 20,
        k: int = 2,
        sigma2: float = 0.5,
        sigma2_T: float = 0.1,
        sigma2_R: float = 0.3,
        alpha: float = 0.4,
        delta: float = 0.15,
        mu_min: float = 0.0,
        mu_max: float = 10.0,
        mu_hot: float = 4.0,
        mu_lo: float = 3.0,
        mu_hi: float = 7.0,
        c: float = 0.05,
        M: float = 50.0,
        T: int = 200,
    ):
        self.n_bins = n_bins
        self.k = k
        self.sigma2 = sigma2
        self.sigma2_T = sigma2_T
        self.sigma2_R = sigma2_R
        self.alpha = alpha
        self.delta = delta
        self.mu_min = mu_min
        self.mu_max = mu_max
        self.mu_hot = mu_hot
        self.mu_lo = mu_lo
        self.mu_hi = mu_hi
        self.c = c
        self.M = M
        self.T = T

        # Actions
        self.actions = list(range(-k, k + 1))
        self.n_actions = len(self.actions)

        # ── FIXED BIN EDGES (physical range only, σ-independent) ──────────
        # This is the critical design choice: bin edges are IDENTICAL
        # regardless of sigma2, so noise comparisons are valid.
        self.bin_edges = np.linspace(mu_min, mu_max, n_bins + 1)

        # Pre-compute bin centers for visualization
        self._bin_centers = 0.5 * (self.bin_edges[:-1] + self.bin_edges[1:])

        # Runtime state
        self.mu = None
        self.t = None
        self._mu_history = []
        self._z_history = []
        self._a_history = []

    # ------------------------------------------------------------------ #
    #  Core Interface                                                       #
    # ------------------------------------------------------------------ #

    def reset(self):
        """Cold start: µ near µ_min."""
        self.mu = self.mu_min + 0.3
        self.t = 0
        self._mu_history = []
        self._z_history = []
        self._a_history = []
        return self._get_obs_state()

    def step(self, action_idx: int):
        """
        Execute one step.

        Returns
        -------
        next_state : int   discretized bin of z_{t+1}
        reward     : float noisy reward sample
        done       : bool
        info       : dict  contains true µ for diagnostics
        """
        a = self.actions[action_idx]
        mu_before = self.mu

        # Log for POMDP trajectory visualization
        z_current = np.random.normal(mu_before, np.sqrt(self.sigma2))
        self._mu_history.append(mu_before)
        self._z_history.append(z_current)
        self._a_history.append(a)

        # Reward is based on µ BEFORE the action (correct per spec)
        expected_r = self._expected_reward(mu_before, a)
        reward = expected_r + np.random.randn() * np.sqrt(self.sigma2_R)

        # Transition: µ_{t+1} = clip(µ_t - α·a + d(µ_t) + ε, µ_min, µ_max)
        drift = self.delta if mu_before >= self.mu_hot else 0.0
        epsilon = np.random.randn() * np.sqrt(self.sigma2_T)
        self.mu = np.clip(
            mu_before - self.alpha * a + drift + epsilon,
            self.mu_min, self.mu_max
        )
        self.t += 1

        meltdown = self.mu >= self.mu_max
        done = meltdown or (self.t >= self.T)

        info = {
            "mu_before": mu_before,
            "mu_after": self.mu,
            "action": a,
            "meltdown": meltdown,
        }
        return self._get_obs_state(), reward, done, info

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    def _get_obs_state(self) -> int:
        z = np.random.normal(self.mu, np.sqrt(self.sigma2))
        return self._discretize(z)

    def _discretize(self, z: float) -> int:
        """
        Map z to a bin index, clipped to [0, n_bins-1].
        Observations outside the physical range are assigned to
        the nearest boundary bin — physically correct behavior.
        """
        idx = np.searchsorted(self.bin_edges, z, side="right") - 1
        return int(np.clip(idx, 0, self.n_bins - 1))

    def _expected_reward(self, mu: float, a: int) -> float:
        rod_cost = self.c * abs(a)
        if mu >= self.mu_max:
            return -self.M - rod_cost
        elif self.mu_lo <= mu <= self.mu_hi:
            return (mu - self.mu_lo) - rod_cost
        else:
            return -rod_cost

    def bin_centers(self) -> np.ndarray:
        return self._bin_centers.copy()

    def get_trajectory(self):
        """Return logged µ and z histories for POMDP visualization."""
        return (
            np.array(self._mu_history),
            np.array(self._z_history),
            np.array(self._a_history),
        )
