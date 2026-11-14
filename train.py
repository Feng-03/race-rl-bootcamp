import os
import csv
import pickle
import random
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
from car_env.env import RaceEnv
from car_env.agent import DQNAgent

env = RaceEnv()
agent = DQNAgent(obs_dim=env.observation_space.shape[0], n_actions=env.N_ACTIONS, buffer_size=200_000)

EPISODES_THIS_RUN = 1000
EPS_START, EPS_END, EPS_DECAY_EPISODES = 1.0, 0.05, 1800
PRINT_EVERY = 10
OVERLAY_EVERY = 500
CHECKPOINT_EVERY = 100
TRAIL_BUFFER_SIZE = 300

RUNS_DIR = "runs"
TRAILS_DIR = os.path.join(RUNS_DIR, "trails")
LOG_FILE = os.path.join(RUNS_DIR, "training_log.csv")
CHECKPOINT_FILE = os.path.join(RUNS_DIR, "checkpoint_train.pt")
BUFFER_FILE = os.path.join(RUNS_DIR, "replay_buffer.pkl")
QNET_FILE = os.path.join(RUNS_DIR, "q_net.pt")

os.makedirs(TRAILS_DIR, exist_ok=True)

reward_history = []
trail_history = []
best_progress_frac = 0.0
start_time = time.time()
exploration_actions, total_actions = 0, 0
episodes_so_far = 0

if os.path.exists(CHECKPOINT_FILE):
    ckpt = torch.load(CHECKPOINT_FILE, weights_only=False)
    agent.q_net.load_state_dict(ckpt["q_net"])
    agent.target_net.load_state_dict(ckpt["target_net"])
    agent.optimizer.load_state_dict(ckpt["optimizer"])
    agent.train_steps = ckpt["train_steps"]
    episodes_so_far = ckpt["episodes_so_far"]
    best_progress_frac = ckpt["best_progress_frac"]
    print(f"Resumed from checkpoint: {episodes_so_far} episodes already trained")

    if os.path.exists(BUFFER_FILE):
        try:
            with open(BUFFER_FILE, "rb") as f:
                agent.buffer.buffer = pickle.load(f)
            print(f"Loaded replay buffer: {len(agent.buffer)} experiences")
        except (EOFError, pickle.UnpicklingError):
            print("Warning: replay_buffer.pkl was corrupted/incomplete — starting with an empty buffer")
else:
    print("No checkpoint found — starting fresh")

write_header = not os.path.exists(LOG_FILE)
if write_header:
    with open(LOG_FILE, "w", newline="") as f:
        csv.writer(f).writerow(["episode", "epsilon", "reward", "avg10", "steps", "buffer_size", "track"])


def log_to_csv(ep, epsilon, total_reward, avg10, steps, buffer_size, track_file):
    with open(LOG_FILE, "a", newline="") as f:
        csv.writer(f).writerow([ep, round(epsilon, 3), round(total_reward, 2),
                                 round(avg10, 2), steps, buffer_size, track_file])


def save_checkpoint():
    torch.save({
        "q_net": agent.q_net.state_dict(),
        "target_net": agent.target_net.state_dict(),
        "optimizer": agent.optimizer.state_dict(),
        "train_steps": agent.train_steps,
        "episodes_so_far": episodes_so_far,
        "best_progress_frac": float(best_progress_frac),
    }, CHECKPOINT_FILE)

    buffer_tmp = BUFFER_FILE + ".tmp"
    with open(buffer_tmp, "wb") as f:
        pickle.dump(agent.buffer.buffer, f)
    os.replace(buffer_tmp, BUFFER_FILE)   # atomic — never leaves a half-written replay_buffer.pkl

    torch.save(agent.q_net.state_dict(), QNET_FILE)


def _draw_hud(ax, label, extra_note=""):
    elapsed_h = (time.time() - start_time) / 3600
    exploration_pct = 100 * exploration_actions / max(1, total_actions)
    hud = (f"Best: {best_progress_frac*100:.0f}% of track\n"
           f"Training time (this run): {elapsed_h:.1f}h\n"
           f"Exploration: {exploration_pct:.0f}%")
    ax.text(0.98, 0.02, hud, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=13, color="white", family="monospace",
            bbox=dict(facecolor="gray", alpha=0.6, boxstyle="round"))
    ax.text(0.02, 0.98, label, transform=ax.transAxes, ha="left", va="top",
            fontsize=16, color="red", weight="bold")
    if extra_note:
        ax.text(0.02, 0.02, extra_note, transform=ax.transAxes, ha="left", va="bottom",
                fontsize=10, color="#999999")


def save_overlays(ep_start, ep_end, prefix):
    fixed_entries = [e for e in trail_history if not e[0].startswith("random:")]
    random_entries = [e for e in trail_history if e[0].startswith("random:")]

    if fixed_entries:
        fig, ax = plt.subplots(figsize=(10, 8))
        fig.patch.set_facecolor("black"); ax.set_facecolor("black")
        seen = {}
        for track_file, track, trail, crashed in fixed_entries:
            if track_file not in seen:
                seen[track_file] = track
        for track in seen.values():
            ax.plot(track.left_edge[:, 0], track.left_edge[:, 1], color="white", lw=1, alpha=0.4)
            ax.plot(track.right_edge[:, 0], track.right_edge[:, 1], color="white", lw=1, alpha=0.4)
        for track_file, track, trail, crashed in fixed_entries:
            trail = np.array(trail)
            ax.plot(trail[:, 0], trail[:, 1], color="red", alpha=0.08, lw=1.2, zorder=1)
            if crashed:
                ax.plot(trail[-1, 0], trail[-1, 1], marker="x", color="gray", ms=4, alpha=0.6, zorder=2)
        _draw_hud(ax, f"#{ep_start}-{ep_end} (hand-drawn tracks)")
        ax.set_aspect("equal"); ax.axis("off"); fig.tight_layout()
        path = os.path.join(TRAILS_DIR, f"{prefix}_fixed.png")
        fig.savefig(path, dpi=120, facecolor="black"); plt.close(fig)
        print(f"Saved {path}")

    if random_entries:
        fig, ax = plt.subplots(figsize=(10, 8))
        fig.patch.set_facecolor("black"); ax.set_facecolor("black")
        for track_file, track, trail, crashed in random_entries:
            trail = np.array(trail)
            ax.plot(trail[:, 0], trail[:, 1], color="orange", alpha=0.06, lw=1.0, zorder=1)
            if crashed:
                ax.plot(trail[-1, 0], trail[-1, 1], marker="x", color="gray", ms=4, alpha=0.6, zorder=2)
        _draw_hud(ax, f"#{ep_start}-{ep_end} (random procedural tracks)",
                  extra_note=f"{len(set(e[0] for e in random_entries))} distinct generated tracks, outlines omitted")
        ax.set_aspect("equal"); ax.axis("off"); fig.tight_layout()
        path = os.path.join(TRAILS_DIR, f"{prefix}_random.png")
        fig.savefig(path, dpi=120, facecolor="black"); plt.close(fig)
        print(f"Saved {path}")


try:
    for i in range(EPISODES_THIS_RUN):
        episodes_so_far += 1
        epsilon = max(EPS_END, EPS_START - episodes_so_far / EPS_DECAY_EPISODES)
        obs, info = env.reset()
        total_reward = 0.0

        while True:
            action = agent.act(obs, epsilon)
            total_actions += 1
            if random.random() < epsilon:
                exploration_actions += 1
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            agent.store(obs, action, reward, next_obs, terminated)
            agent.train_step()

            obs = next_obs
            total_reward += reward
            if done:
                break

        reward_history.append(total_reward)
        best_progress_frac = max(best_progress_frac, env.progress / env.track.N_PTS)

        trail_history.append((env.current_track_file, env.track, env.trail, info["off_track"]))
        if len(trail_history) > TRAIL_BUFFER_SIZE:
            trail_history.pop(0)

        avg10 = np.mean(reward_history[-10:])
        log_to_csv(episodes_so_far, epsilon, total_reward, avg10, env.steps, len(agent.buffer), env.current_track_file)

        if episodes_so_far % PRINT_EVERY == 0:
            print(f"episode {episodes_so_far:5d}  epsilon={epsilon:.2f}  reward={total_reward:7.2f}  "
                  f"avg10={avg10:7.2f}  steps={env.steps:4d}  buffer={len(agent.buffer)}")

        if episodes_so_far % OVERLAY_EVERY == 0:
            save_overlays(episodes_so_far - OVERLAY_EVERY + 1, episodes_so_far,
                          f"trails_ep{episodes_so_far-OVERLAY_EVERY+1}-{episodes_so_far}")

        if episodes_so_far % CHECKPOINT_EVERY == 0:
            save_checkpoint()

except KeyboardInterrupt:
    print("Interrupted — saving checkpoint...")
finally:
    save_checkpoint()
    if trail_history:
        save_overlays(episodes_so_far - len(trail_history) + 1, episodes_so_far, "trails_final")
    print(f"Checkpoint saved. Total episodes trained so far: {episodes_so_far}")