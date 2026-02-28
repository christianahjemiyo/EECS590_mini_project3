"""
experiments.py — Synthesis Version
=====================================
Six experiments covering all required and synthesis additions:

  Fig 1 — Physics Validation (passive a=0 test) [NEW from their solution]
  Fig 2 — Learning Curves: two noise levels, FIXED bin edges [UPGRADED]
  Fig 3 — Q-Function Heatmaps
  Fig 4 — Algorithm Comparison (SARSA / Q-Learning / SARSA(λ=0) / SARSA(λ=0.8))
  Fig 5 — Greedy Policy Visualization
  Fig 6 — POMDP Trajectory: µt vs zt in a single episode [NEW from their solution]
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
import os

from reactor_env import ReactorEnv
from agents import QLearningAgent, SARSAAgent, SARSALambdaAgent

OUT = os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUT, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Training loop helpers
# ─────────────────────────────────────────────────────────────────────────────

def run_qlearning(env, agent, n_ep):
    returns, melts = [], []
    for _ in range(n_ep):
        s = env.reset(); ep_r = 0.0; done = False; melted = False
        while not done:
            a = agent.choose_action(s)
            s2, r, done, info = env.step(a)
            agent.update(s, a, r, s2, done)
            s = s2; ep_r += r
            if info["meltdown"]: melted = True
        agent.decay_epsilon()
        returns.append(ep_r); melts.append(int(melted))
    return np.array(returns), np.array(melts)


def run_sarsa(env, agent, n_ep):
    returns, melts = [], []
    for _ in range(n_ep):
        s = env.reset(); a = agent.choose_action(s)
        ep_r = 0.0; done = False; melted = False
        while not done:
            s2, r, done, info = env.step(a)
            a2 = agent.choose_action(s2) if not done else 0
            agent.update(s, a, r, s2, a2, done)
            s, a = s2, a2; ep_r += r
            if info["meltdown"]: melted = True
        agent.decay_epsilon()
        returns.append(ep_r); melts.append(int(melted))
    return np.array(returns), np.array(melts)


def run_sarsa_lambda(env, agent, n_ep):
    returns, melts = [], []
    for _ in range(n_ep):
        agent.reset_episode()
        s = env.reset(); a = agent.choose_action(s)
        ep_r = 0.0; done = False; melted = False
        while not done:
            s2, r, done, info = env.step(a)
            a2 = agent.choose_action(s2) if not done else 0
            agent.update(s, a, r, s2, a2, done)
            s, a = s2, a2; ep_r += r
            if info["meltdown"]: melted = True
        agent.decay_epsilon()
        returns.append(ep_r); melts.append(int(melted))
    return np.array(returns), np.array(melts)


def smooth(x, w=40):
    return np.convolve(x, np.ones(w) / w, mode='valid')


def make_agent(kind, n_s, n_a, lam=0.8):
    kw = dict(n_states=n_s, n_actions=n_a, gamma=0.95, alpha=0.1,
              epsilon=1.0, epsilon_min=0.04, epsilon_decay=0.994)
    if kind == "ql":    return QLearningAgent(**kw)
    if kind == "sarsa": return SARSAAgent(**kw)
    return SARSALambdaAgent(**kw, lam=lam)


# ─────────────────────────────────────────────────────────────────────────────
# FIG 1 — Physics Validation: Passive a=0 Test
# ─────────────────────────────────────────────────────────────────────────────

def fig1_physics_validation():
    """
    Run the reactor with zero sensor noise and a=0 (no control).
    Validates that:
      1. Core stays stable while µ < µ_hot
      2. Once µ ≥ µ_hot, drift causes exponential runaway toward µ_max
    This confirms the non-stationary bandit property: the arm is safe
    initially but becomes increasingly dangerous as the episode progresses.
    """
    np.random.seed(0)
    # Zero sensor noise so observations perfectly reflect true µ
    env = ReactorEnv(sigma2=0.0, sigma2_T=0.05, T=200)
    env.reset()

    mu_hist = []
    done = False
    action_idx_zero = env.actions.index(0)   # index of a=0

    while not done:
        _, _, done, info = env.step(action_idx_zero)
        mu_hist.append(info["mu_before"])

    steps = np.arange(len(mu_hist))

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(steps, mu_hist, color='#c0392b', lw=2.5, label='True Reactivity µt (σ²_obs = 0)')
    ax.axhline(env.mu_hot, color='#e67e22', ls='--', lw=1.8,
               label=f'Drift Onset  µ_hot = {env.mu_hot}')
    ax.axhline(env.mu_hi,  color='#27ae60', ls=':', lw=1.5,
               label=f'Power Zone Ceiling  µ_hi = {env.mu_hi}')
    ax.axhline(env.mu_max, color='#2c3e50', ls='-', lw=2,
               label=f'Meltdown Threshold  µ_max = {env.mu_max}')
    ax.fill_between(steps, env.mu_lo, env.mu_hi, alpha=0.08, color='green',
                    label='Power Zone [µ_lo, µ_hi]')

    # Annotate the drift-onset moment
    hot_step = next((i for i, m in enumerate(mu_hist) if m >= env.mu_hot), None)
    if hot_step is not None:
        ax.annotate('Drift activates here\n(self-heating begins)',
                    xy=(hot_step, env.mu_hot),
                    xytext=(hot_step + 10, env.mu_hot + 1.5),
                    arrowprops=dict(arrowstyle='->', color='#e67e22'),
                    fontsize=9, color='#e67e22')

    ax.set_xlabel("Time Step", fontsize=11)
    ax.set_ylabel("Reactivity µt", fontsize=11)
    ax.set_title("Fig 1 — Physics Validation: Passive Test (a = 0, No Control)\n"
                 "Confirms non-stationary bandit: safe initially, runaway once µ ≥ µ_hot",
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.25)
    ax.set_xlim(0, len(mu_hist))
    ax.set_ylim(env.mu_min - 0.5, env.mu_max + 0.5)

    plt.tight_layout()
    path = f"{OUT}/fig1_physics_validation.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 2 — Learning Curves: Two Noise Levels (Fixed Bin Edges)
# ─────────────────────────────────────────────────────────────────────────────

def fig2_learning_curves(n_ep=1000):
    """
    Compare SARSA(λ=0.8) vs Q-learning under σ²=0.5 and σ²=2.0.
    CRITICAL: Both conditions use IDENTICAL fixed bin edges [µ_min, µ_max].
    σ² only affects how reliably the observation falls in the correct bin.
    This is the only design that isolates noise as a single variable.
    """
    np.random.seed(42)
    noise_configs = [
        (r"Low Noise  σ² = 0.5", 0.5),
        (r"High Noise  σ² = 2.0", 2.0),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        "Fig 2 — Learning Curves: Two Sensor Noise Levels\n"
        "Bin edges fixed to physical range [µ_min, µ_max] — noise is the ONLY variable",
        fontsize=12, fontweight='bold'
    )

    for col, (label, sig2) in enumerate(noise_configs):
        # Both share the same physical bin structure
        env_ql = ReactorEnv(sigma2=sig2)
        env_sl = ReactorEnv(sigma2=sig2)

        n_s, n_a = env_ql.n_bins, env_ql.n_actions

        ql = make_agent("ql", n_s, n_a)
        sl = make_agent("sl", n_s, n_a, lam=0.8)

        ql_ret, ql_melt = run_qlearning(env_ql, ql, n_ep)
        sl_ret, sl_melt = run_sarsa_lambda(env_sl, sl, n_ep)

        w = 40
        ax_r = axes[0, col]
        ax_m = axes[1, col]

        ax_r.plot(smooth(ql_ret, w), color='#e74c3c', lw=1.8, label='Q-Learning')
        ax_r.plot(smooth(sl_ret, w), color='#2980b9', lw=1.8, label='SARSA(λ=0.8)')
        ax_r.axhline(0, color='gray', lw=0.8, ls='--')
        ax_r.set_title(label, fontsize=11)
        ax_r.set_xlabel("Episode"); ax_r.set_ylabel("Smoothed Return")
        ax_r.legend(fontsize=9)

        ax_m.plot(smooth(ql_melt.astype(float), 50), color='#e74c3c', lw=1.5, label='Q-Learning')
        ax_m.plot(smooth(sl_melt.astype(float), 50), color='#2980b9', lw=1.5, label='SARSA(λ=0.8)')
        ax_m.set_ylim(-0.05, 1.05)
        ax_m.set_title(f"Meltdown Rate — {label}", fontsize=10)
        ax_m.set_xlabel("Episode"); ax_m.set_ylabel("Meltdown Rate (50-ep window)")
        ax_m.legend(fontsize=9)

        final_ql = ql_ret[-100:].mean()
        final_sl = sl_ret[-100:].mean()
        melt_ql  = ql_melt[-100:].mean()
        melt_sl  = sl_melt[-100:].mean()
        print(f"  σ²={sig2}: Q-Learning final={final_ql:.1f} melt={melt_ql:.1%} | "
              f"SARSA(λ) final={final_sl:.1f} melt={melt_sl:.1%}")

    plt.tight_layout()
    path = f"{OUT}/fig2_learning_curves.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 3 — Q-Function Heatmaps
# ─────────────────────────────────────────────────────────────────────────────

def fig3_qfunction_heatmaps(n_ep=1500):
    np.random.seed(0)
    env_ql = ReactorEnv(sigma2=0.5)
    env_sl = ReactorEnv(sigma2=0.5)
    n_s, n_a = env_ql.n_bins, env_ql.n_actions

    ql = make_agent("ql", n_s, n_a)
    sl = make_agent("sl", n_s, n_a, lam=0.8)

    # Train lower epsilon_min to get cleaner converged Q
    ql.epsilon_min = 0.02; sl.epsilon_min = 0.02

    run_qlearning(env_ql, ql, n_ep)
    run_sarsa_lambda(env_sl, sl, n_ep)

    bc = env_ql.bin_centers()
    action_labels = [str(a) for a in env_ql.actions]

    # Reference bin indices for annotation lines
    lo_bin  = int(np.searchsorted(env_ql.bin_edges, env_ql.mu_lo))
    hi_bin  = int(np.searchsorted(env_ql.bin_edges, env_ql.mu_hi))
    hot_bin = int(np.searchsorted(env_ql.bin_edges, env_ql.mu_hot))
    max_bin = int(np.searchsorted(env_ql.bin_edges, env_ql.mu_max - 0.01))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "Fig 3 — Learned Q(z, a) Heatmaps  (σ² = 0.5, 1500 episodes)\n"
        "Rows = discretized observation bin;  Columns = rod action",
        fontsize=12, fontweight='bold'
    )

    for ax, agent, title in zip(axes, [ql, sl], ["Q-Learning", "SARSA(λ=0.8)"]):
        Q = agent.Q
        vmax = max(np.percentile(np.abs(Q[Q != 0]), 95), 1.0)
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        im = ax.imshow(Q, aspect='auto', cmap='RdYlGn', norm=norm,
                       origin='lower', extent=[-0.5, n_a-0.5, -0.5, n_s-0.5])
        plt.colorbar(im, ax=ax, label="Q-value")

        ax.axhline(lo_bin,  color='cyan',   lw=1.5, ls='--', label=f'µ_lo  (bin {lo_bin})')
        ax.axhline(hi_bin,  color='cyan',   lw=1.5, ls=':',  label=f'µ_hi  (bin {hi_bin})')
        ax.axhline(hot_bin, color='orange', lw=1.2, ls='--', label=f'µ_hot (bin {hot_bin})')
        ax.axhline(max_bin, color='red',    lw=2.0, ls='-',  label=f'µ_max (bin {max_bin})')

        ax.set_xticks(range(n_a))
        ax.set_xticklabels(action_labels, fontsize=9)
        ax.set_xlabel("Action (− = withdraw, + = insert)", fontsize=10)
        ax.set_ylabel("Observation Bin (state)", fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=7, loc='upper right')

    plt.tight_layout()
    path = f"{OUT}/fig3_qfunction_heatmaps.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 4 — Algorithm Comparison
# ─────────────────────────────────────────────────────────────────────────────

def fig4_algorithm_comparison(n_ep=1200):
    np.random.seed(7)
    configs = [
        ("Q-Learning",      "ql",    "#e74c3c"),
        ("SARSA",           "sarsa", "#e67e22"),
        ("SARSA(λ=0)",      "sl0",   "#8e44ad"),
        ("SARSA(λ=0.8)",    "sl8",   "#2980b9"),
    ]

    all_ret, all_melt = {}, {}
    print("\n  Final 100-episode stats:")
    for name, key, _ in configs:
        env = ReactorEnv(sigma2=0.5)
        n_s, n_a = env.n_bins, env.n_actions
        if key == "ql":
            ag = make_agent("ql", n_s, n_a)
            r, m = run_qlearning(env, ag, n_ep)
        elif key == "sarsa":
            ag = make_agent("sarsa", n_s, n_a)
            r, m = run_sarsa(env, ag, n_ep)
        elif key == "sl0":
            ag = make_agent("sl", n_s, n_a, lam=0.0)
            r, m = run_sarsa_lambda(env, ag, n_ep)
        else:
            ag = make_agent("sl", n_s, n_a, lam=0.8)
            r, m = run_sarsa_lambda(env, ag, n_ep)
        all_ret[key] = r; all_melt[key] = m
        print(f"    {name:20s}: return={r[-100:].mean():7.1f}  meltdown={m[-100:].mean():.1%}")

    w = 40
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "Fig 4 — Algorithm Comparison: Return & Safety During Training\n"
        "σ² = 0.5 (fixed bins), γ = 0.95, α = 0.1",
        fontsize=12, fontweight='bold'
    )

    for name, key, color in configs:
        ax1.plot(smooth(all_ret[key], w), color=color, lw=1.8, label=name)
        ax2.plot(smooth(all_melt[key].astype(float), 50), color=color, lw=1.5, label=name)

    ax1.axhline(0, color='gray', lw=0.8, ls='--')
    ax1.set_xlabel("Episode"); ax1.set_ylabel("Smoothed Return")
    ax1.set_title("Episode Return"); ax1.legend(fontsize=9)

    ax2.set_ylim(-0.05, 1.05)
    ax2.set_xlabel("Episode"); ax2.set_ylabel("Meltdown Rate (50-ep window)")
    ax2.set_title("Meltdown Frequency During Training"); ax2.legend(fontsize=9)

    plt.tight_layout()
    path = f"{OUT}/fig4_algorithm_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 5 — Greedy Policy Visualization
# ─────────────────────────────────────────────────────────────────────────────

def fig5_greedy_policy(n_ep=1500):
    np.random.seed(3)
    env_ql = ReactorEnv(sigma2=0.5)
    env_sl = ReactorEnv(sigma2=0.5)
    n_s, n_a = env_ql.n_bins, env_ql.n_actions

    ql = make_agent("ql", n_s, n_a); ql.epsilon_min = 0.01
    sl = make_agent("sl", n_s, n_a, lam=0.8); sl.epsilon_min = 0.01

    run_qlearning(env_ql, ql, n_ep)
    run_sarsa_lambda(env_sl, sl, n_ep)

    bc = env_ql.bin_centers()
    ql_pol = [env_ql.actions[np.argmax(ql.Q[s])] for s in range(n_s)]
    sl_pol = [env_ql.actions[np.argmax(sl.Q[s])] for s in range(n_s)]

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(bc, ql_pol, 'o-', color='#e74c3c', lw=2, ms=6, label='Q-Learning')
    ax.plot(bc, sl_pol, 's--', color='#2980b9', lw=2, ms=6, label='SARSA(λ=0.8)')

    ax.axvspan(env_ql.mu_lo, env_ql.mu_hi, alpha=0.12, color='green', label='Power Zone')
    ax.axvline(env_ql.mu_max, color='red',    lw=2,   ls='-',  label='µ_max (meltdown)')
    ax.axvline(env_ql.mu_hot, color='orange', lw=1.5, ls='--', label='µ_hot (drift onset)')
    ax.axhline(0, color='gray', lw=0.8)

    ax.set_xlabel("Observation Bin Center (physical µ scale)", fontsize=11)
    ax.set_ylabel("Greedy Action  a* = argmax Q(z,a)", fontsize=11)
    ax.set_title(
        "Fig 5 — Learned Greedy Policies\n"
        "Note: SARSA(λ) targets lower µ within power zone — the 'safety buffer' from sensor uncertainty",
        fontsize=11, fontweight='bold'
    )
    ax.legend(fontsize=9)
    ax.set_yticks(sorted(set(env_ql.actions)))
    ax.set_yticklabels([f"a={a}" for a in sorted(set(env_ql.actions))])
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    path = f"{OUT}/fig5_greedy_policy.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 6 — POMDP Trajectory: µt vs zt (Perception vs Reality)
# ─────────────────────────────────────────────────────────────────────────────

def fig6_pomdp_trajectory(n_ep=800):
    """
    Train an agent then run one evaluation episode with full trajectory
    logging. Plot the true hidden µt against the noisy observation zt
    at each step. This visualizes the core POMDP challenge: the observation
    LAGS true physics, especially after the drift threshold is crossed.
    The gap between µt and zt is what causes the agent to sometimes
    fail to insert rods in time — it sees a 'safe' reading while µt
    is already drifting toward meltdown.
    """
    np.random.seed(99)
    env_train = ReactorEnv(sigma2=1.0)   # High noise to show the gap clearly
    n_s, n_a = env_train.n_bins, env_train.n_actions

    agent = make_agent("sl", n_s, n_a, lam=0.8)
    agent.epsilon_min = 0.05
    run_sarsa_lambda(env_train, agent, n_ep)

    # Evaluation episode — greedy
    agent.epsilon = 0.0
    env_eval = ReactorEnv(sigma2=1.0)
    env_eval.reset()

    s = env_eval.reset()
    done = False
    while not done:
        a_idx = agent.choose_action(s)
        s, _, done, _ = env_eval.step(a_idx)

    mu_hist, z_hist, a_hist = env_eval.get_trajectory()
    steps = np.arange(len(mu_hist))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True,
                                    gridspec_kw={'height_ratios': [3, 1]})
    fig.suptitle(
        "Fig 6 — POMDP Trajectory: True Reactivity µt vs Noisy Observation zt\n"
        "σ² = 1.0 (high noise) — visualizes the observability gap and its consequences",
        fontsize=12, fontweight='bold'
    )

    ax1.plot(steps, mu_hist, color='#c0392b', lw=2.5, label='True µt (hidden from agent)', zorder=3)
    ax1.plot(steps, z_hist,  color='#3498db', lw=1.2, alpha=0.75, label='Observed zt (noisy sensor)', zorder=2)
    ax1.fill_between(steps, mu_hist, z_hist, alpha=0.12, color='purple',
                     label='Observability Gap  |µt − zt|')

    ax1.axhspan(env_eval.mu_lo, env_eval.mu_hi, alpha=0.08, color='green', label='Power Zone')
    ax1.axhline(env_eval.mu_hot, color='orange', ls='--', lw=1.5, label='µ_hot (drift onset)')
    ax1.axhline(env_eval.mu_max, color='red',    ls='-',  lw=2,   label='µ_max (meltdown)')

    # Annotate any moment where µt > µ_hot but z_t < µ_hot (dangerous lag)
    lag_steps = [i for i in range(len(mu_hist))
                 if mu_hist[i] >= env_eval.mu_hot and z_hist[i] < env_eval.mu_hot]
    if lag_steps:
        i = lag_steps[0]
        ax1.annotate(
            'Agent sees "safe" reading\nbut µt already drifting',
            xy=(i, z_hist[i]), xytext=(i + 8, z_hist[i] - 1.5),
            arrowprops=dict(arrowstyle='->', color='purple'),
            fontsize=9, color='purple'
        )

    ax1.set_ylabel("Reactivity", fontsize=11)
    ax1.legend(fontsize=8, loc='upper left')
    ax1.grid(True, alpha=0.2)
    ax1.set_ylim(env_eval.mu_min - 0.5, env_eval.mu_max + 0.5)

    # Bottom panel: rod actions taken
    ax2.bar(steps, a_hist, color=['#e74c3c' if a > 0 else '#2980b9' if a < 0 else 'gray'
                                   for a in a_hist], width=0.8, alpha=0.8)
    ax2.axhline(0, color='black', lw=0.8)
    ax2.set_ylabel("Action a", fontsize=10)
    ax2.set_xlabel("Time Step", fontsize=11)
    ax2.set_yticks(sorted(set(env_eval.actions)))
    ax2.grid(True, alpha=0.2)

    plt.tight_layout()
    path = f"{OUT}/fig6_pomdp_trajectory.png"
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 65)
    print("Fig 1: Physics Validation...")
    fig1_physics_validation()

    print("\nFig 2: Learning Curves (two noise levels, fixed bins)...")
    fig2_learning_curves(n_ep=1000)

    print("\nFig 3: Q-function heatmaps...")
    fig3_qfunction_heatmaps(n_ep=1500)

    print("\nFig 4: Algorithm comparison...")
    fig4_algorithm_comparison(n_ep=1200)

    print("\nFig 5: Greedy policy...")
    fig5_greedy_policy(n_ep=1500)

    print("\nFig 6: POMDP trajectory...")
    fig6_pomdp_trajectory(n_ep=800)

    print("\nAll figures saved to", OUT)
