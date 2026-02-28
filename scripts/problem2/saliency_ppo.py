"""
saliency_ppo.py
================
Saliency analysis for a trained PPO (actor-critic) Breakout agent.

Implements Problem 2 requirements:
1) Pivotal frame selection from evaluation rollout
2) Perturbation-based saliency for greedy and non-greedy actions
3) Patch-size comparison
4) Gradient-based saliency (challenge)
5) Adversarial perturbation (challenge)

Note: PPO does not produce Q-values. We use the policy logits as the
action score S(f, a). All saliency and perturbation analysis is then
computed using |S(f, a) - S(f_pert, a)|.
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
# Ensure Atari environments are registered with Gymnasium
import ale_py  # noqa: F401
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_ID = "BreakoutNoFrameskip-v4"
DEFAULT_MODEL = "./models/ppo_breakout_final.zip"

ACTION_NAMES = ["NOOP", "FIRE", "RIGHT", "LEFT", "RIGHTFIRE", "LEFTFIRE"]


def build_env(env_id: str, seed: int, render_mode: str | None):
    env = make_atari_env(
        env_id,
        n_envs=1,
        seed=seed,
        env_kwargs={"render_mode": render_mode} if render_mode else None,
    )
    env = VecFrameStack(env, n_stack=4, channels_order="last")
    return env


def get_logits(model: PPO, obs: np.ndarray) -> np.ndarray:
    obs_tensor, _ = model.policy.obs_to_tensor(obs)
    dist = model.policy.get_distribution(obs_tensor)
    logits = dist.distribution.logits.detach().cpu().numpy()[0]
    return logits


def select_pivotal_frames(model: PPO, env, n_frames: int, sample_every: int):
    obs = env.reset()
    saved = []
    step = 0
    done = False

    while len(saved) < n_frames * 5 and step < 10_000:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, dones, _ = env.step(action)
        done = bool(dones[0])
        step += 1

        if step % sample_every == 0:
            logits = get_logits(model, obs)
            greedy = int(np.argmax(logits))
            max_logit = float(logits[greedy])
            frame = env.render()
            saved.append(
                {
                    "step": step,
                    "obs": obs.copy(),
                    "frame": frame.copy(),
                    "logits": logits.copy(),
                    "greedy_action": greedy,
                    "max_logit": max_logit,
                }
            )

        if done:
            obs = env.reset()
            done = False

    if len(saved) < 2:
        return []

    # Pivotal frames: largest |delta max_logit| between consecutive saved frames
    deltas = []
    for i in range(1, len(saved)):
        delta = abs(saved[i]["max_logit"] - saved[i - 1]["max_logit"])
        deltas.append((delta, i))

    deltas.sort(reverse=True, key=lambda x: x[0])
    top_idxs = sorted({i for _, i in deltas[:n_frames]})
    pivotal = [saved[i] for i in top_idxs]
    return pivotal


def perturbation_saliency(model: PPO, obs: np.ndarray, action_idx: int, patch: int):
    h, w = 84, 84
    baseline = get_logits(model, obs)[action_idx]

    neutral = int(obs.mean())
    n_h, n_w = h // patch, w // patch
    sal = np.zeros((n_h, n_w), dtype=np.float32)

    for i in range(n_h):
        for j in range(n_w):
            pert = obs.copy()
            y0, y1 = i * patch, (i + 1) * patch
            x0, x1 = j * patch, (j + 1) * patch
            pert[:, y0:y1, x0:x1, :] = neutral
            score = get_logits(model, pert)[action_idx]
            sal[i, j] = abs(baseline - score)

    sal_full = sal.repeat(patch, axis=0).repeat(patch, axis=1)
    s_min, s_max = sal_full.min(), sal_full.max()
    if s_max > s_min:
        sal_full = (sal_full - s_min) / (s_max - s_min)
    return sal_full


def gradient_saliency(model: PPO, obs: np.ndarray, action_idx: int):
    obs_tensor, _ = model.policy.obs_to_tensor(obs)
    obs_tensor = obs_tensor.float().detach().clone().requires_grad_(True)
    dist = model.policy.get_distribution(obs_tensor)
    logit = dist.distribution.logits[0, action_idx]

    model.policy.zero_grad()
    logit.backward()

    grad = obs_tensor.grad[0].abs()  # (n_stack, 84, 84)
    sal = grad.max(dim=0).values.cpu().numpy()
    s_min, s_max = sal.min(), sal.max()
    if s_max > s_min:
        sal = (sal - s_min) / (s_max - s_min)
    return sal


def overlay(frame, saliency, title, out_path, alpha=0.55):
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].imshow(frame)
    ax[0].set_title("Original Frame")
    ax[0].axis("off")

    ax[1].imshow(frame)
    ax[1].imshow(
        saliency,
        cmap="hot",
        alpha=alpha,
        interpolation="bilinear",
        extent=(0, frame.shape[1], frame.shape[0], 0),
    )
    ax[1].set_title(title)
    ax[1].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def pgd_adversarial(model: PPO, obs: np.ndarray, target_action: int,
                    eps_max: float, step_size: float, n_steps: int):
    obs_tensor, _ = model.policy.obs_to_tensor(obs)
    obs_tensor = obs_tensor.float().detach()

    delta = torch.zeros_like(obs_tensor, requires_grad=True)
    result = {
        "success": False,
        "eps_needed": eps_max,
        "new_action": target_action,
        "delta": np.zeros_like(obs_tensor.cpu().numpy()),
        "step": None,
    }

    for step in range(n_steps):
        pert = torch.clamp(obs_tensor + delta, 0.0, 1.0)
        dist = model.policy.get_distribution(pert)
        logits = dist.distribution.logits[0]
        target_logit = logits[target_action]
        other_logits = torch.cat([logits[:target_action], logits[target_action + 1:]])
        loss = target_logit - other_logits.max()

        if delta.grad is not None:
            delta.grad.zero_()
        loss.backward()

        with torch.no_grad():
            delta = delta + step_size * delta.grad.sign()
            delta = torch.clamp(delta, -eps_max, eps_max)
            delta = delta.detach().requires_grad_(True)

            pert2 = torch.clamp(obs_tensor + delta, 0.0, 1.0)
            logits2 = model.policy.get_distribution(pert2).distribution.logits[0]
            new_action = int(torch.argmax(logits2).item())
            eps_used = float(delta.abs().max().item())

        if new_action != target_action and not result["success"]:
            result = {
                "success": True,
                "eps_needed": eps_used,
                "new_action": new_action,
                "delta": delta.detach().cpu().numpy(),
                "step": step,
            }
            eps_max = eps_used

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--env_id", default=ENV_ID)
    parser.add_argument("--n_pivotal", type=int, default=5)
    parser.add_argument("--sample_every", type=int, default=25)
    parser.add_argument("--patch", type=int, default=8)
    parser.add_argument("--eps_max", type=float, default=0.05)
    parser.add_argument("--step_size", type=float, default=0.002)
    parser.add_argument("--pgd_steps", type=int, default=75)
    args = parser.parse_args()

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model not found: {args.model}")

    out_dirs = {
        "pivotal": "outputs/ppo_pivotal_frames",
        "greedy": "outputs/ppo_saliency_greedy",
        "nongreedy": "outputs/ppo_saliency_nongreedy",
        "patch": "outputs/ppo_patch_comparison",
        "grad": "outputs/ppo_saliency_gradient",
        "adv": "outputs/ppo_adversarial",
    }
    for d in out_dirs.values():
        os.makedirs(os.path.join(SCRIPT_DIR, d), exist_ok=True)

    model = PPO.load(args.model, device="cpu")
    env = build_env(args.env_id, seed=123, render_mode="rgb_array")

    pivotal = select_pivotal_frames(model, env, args.n_pivotal, args.sample_every)
    if not pivotal:
        print("No pivotal frames found.")
        return

    # Save pivotal frames
    for i, pf in enumerate(pivotal):
        path = os.path.join(SCRIPT_DIR, out_dirs["pivotal"], f"pivotal_{i+1}.png")
        plt.imsave(path, pf["frame"])

    # Q4: perturbation saliency for greedy action
    for i, pf in enumerate(pivotal):
        a_star = pf["greedy_action"]
        sal = perturbation_saliency(model, pf["obs"], a_star, args.patch)
        name = ACTION_NAMES[a_star] if a_star < len(ACTION_NAMES) else str(a_star)
        out = os.path.join(SCRIPT_DIR, out_dirs["greedy"], f"greedy_{i+1}.png")
        overlay(pf["frame"], sal, f"Greedy action: {name}", out)

    # Q5: saliency for non-greedy action
    for i, pf in enumerate(pivotal):
        logits = pf["logits"]
        a_star = int(np.argmax(logits))
        a_prime = int(np.argsort(logits)[-2])
        sal = perturbation_saliency(model, pf["obs"], a_prime, args.patch)
        name = ACTION_NAMES[a_prime] if a_prime < len(ACTION_NAMES) else str(a_prime)
        out = os.path.join(SCRIPT_DIR, out_dirs["nongreedy"], f"nongreedy_{i+1}.png")
        overlay(pf["frame"], sal, f"Non-greedy action: {name}", out)

    # Q6a: patch-size comparison on the first pivotal frame
    pf0 = pivotal[0]
    a0 = pf0["greedy_action"]
    for p in [4, 8, 16]:
        sal = perturbation_saliency(model, pf0["obs"], a0, p)
        out = os.path.join(SCRIPT_DIR, out_dirs["patch"], f"patch_{p}.png")
        overlay(pf0["frame"], sal, f"Patch size P={p}", out)

    # Q7a: gradient-based saliency
    for i, pf in enumerate(pivotal[:3]):
        a_star = pf["greedy_action"]
        sal = gradient_saliency(model, pf["obs"], a_star)
        out = os.path.join(SCRIPT_DIR, out_dirs["grad"], f"grad_{i+1}.png")
        overlay(pf["frame"], sal, "Gradient saliency", out)

    # Q7c: adversarial perturbation
    for i, pf in enumerate(pivotal[:2]):
        a_star = pf["greedy_action"]
        result = pgd_adversarial(
            model,
            pf["obs"],
            a_star,
            eps_max=args.eps_max,
            step_size=args.step_size,
            n_steps=args.pgd_steps,
        )
        delta = result["delta"][0]  # (n_stack, 84, 84)
        delta_vis = np.abs(delta).max(axis=0)
        if delta_vis.max() > 0:
            delta_vis /= delta_vis.max()
        out = os.path.join(SCRIPT_DIR, out_dirs["adv"], f"adv_{i+1}.png")
        plt.imsave(out, delta_vis, cmap="hot")

    env.close()
    print("Saliency outputs saved under outputs/ (ppo_*)")


if __name__ == "__main__":
    main()
