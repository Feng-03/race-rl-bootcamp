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
agent = DQNAgent(obs_dim=env.observation_space.shape[0], n_actions=env.N_ACTIONS)

EPISODES_THIS_RUN = 1000
EPS_START, EPS_END, EPS_DECAY_EPISODES = 1.0, 0.05, 1800
PRINT_EVERY = 10
OVERLAY_EVERY = 500
CHECKPOINT_EVERY = 100
TRAIL_BUFFER_SIZE = 300

RUNS_DIR = "runs"
TRAILS_DIR = os.path.join(RUNS_DIR, "trails")
LOG_FILE = os.path.join(RUNS_DIR, "training_log.csv")
CHECKPOINT_FILE = os.path.join(RUNS_DIR, "checkpoint.pt")
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
        with open(BUFFER_FILE, "rb") as f:
            agent.buffer.buffer = pickle.load(f)
        print(f"Loaded replay buffer: {len(agent.buffer)} experiences")
else:
    print("No checkpoint found — starting fresh")

write_header = not os.path.exists(LOG_FILE)
if write_header:
    with open(LOG_FILE, "w", newline="") as f:
        csv.writer(f).writerow(["episode", "epsilon", "reward", "avg10", "steps", "buffer_size"])


def log_to_csv(ep, epsilon, total_reward, avg10, steps, buffer_size):
    with open(LOG_FILE, "a", newline="") as f:
        csv.writer(f).writerow([ep, round(epsilon, 3), round(total_reward, 2),
                                 round(avg10, 2), steps, buffer_size])


def save_checkpoint():
    torch.save({
        "q_net": agent.q_net.state_dict(),
        "target_net": agent.target_net.state_dict(),
        "optimizer": agent.optimizer.state_dict(),
        "train_steps": agent.train_steps,
        "episodes_so_far": episodes_so_far,
        "best_progress_frac": float(best_progress_frac),
    }, CHECKPOINT_FILE)
    with open(BUFFER_FILE, "wb") as f:
        pickle.dump(agent.buffer.buffer, f)
    torch.save(agent.q_net.state_dict(), QNET_FILE)


def save_overlay(filename, ep_range_label):
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    last_track = trail_history[-1][0]
    left, right = last_track.left_edge, last_track.right_edge
    ax.fill(np.r_[left[:, 0], right[::-1, 0]], np.r_[left[:, 1], right[::-1, 1]],
            color="white", alpha=0.08, zorder=0)
    ax.plot(left[:, 0], left[:, 1], color="white", lw=1, alpha=0.5)
    ax.plot(right[:, 0], right[:, 1], color="white", lw=1, alpha=0.5)

    for track, trail, crashed in trail_history:
        trail = np.array(trail)
        ax.plot(trail[:, 0], trail[:, 1], color="red", alpha=0.08, lw=1.2, zorder=1)
        if crashed:
            ax.plot(trail[-1, 0], trail[-1, 1], marker="x", color="gray", ms=4, alpha=0.6, zorder=2)

    elapsed_h = (time.time() - start_time) / 3600
    exploration_pct = 100 * exploration_actions / max(1, total_actions)
    hud = (f"Best: {best_progress_frac*100:.0f}% of track\n"
           f"Training time (this run): {elapsed_h:.1f}h\n"
           f"Exploration: {exploration_pct:.0f}%")
    ax.text(0.98, 0.02, hud, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=13, color="white", family="monospace",
            bbox=dict(facecolor="gray", alpha=0.6, boxstyle="round"))
    ax.text(0.02, 0.98, ep_range_label, transform=ax.transAxes, ha="left", va="top",
            fontsize=16, color="red", weight="bold")

    ax.set_aspect("equal"); ax.axis("off")
    fig.tight_layout()
    filepath = os.path.join(TRAILS_DIR, filename)
    fig.savefig(filepath, dpi=120, facecolor="black")
    plt.close(fig)
    print(f"Saved {filepath}")


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

        trail_history.append((env.track, env.trail, info["off_track"]))
        if len(trail_history) > TRAIL_BUFFER_SIZE:
            trail_history.pop(0)

        avg10 = np.mean(reward_history[-10:])
        log_to_csv(episodes_so_far, epsilon, total_reward, avg10, env.steps, len(agent.buffer))

        if episodes_so_far % PRINT_EVERY == 0:
            print(f"episode {episodes_so_far:5d}  epsilon={epsilon:.2f}  reward={total_reward:7.2f}  "
                  f"avg10={avg10:7.2f}  steps={env.steps:4d}  buffer={len(agent.buffer)}")

        if episodes_so_far % OVERLAY_EVERY == 0:
            save_overlay(f"trails_ep{episodes_so_far-OVERLAY_EVERY+1}-{episodes_so_far}.png",
                         f"#{episodes_so_far-OVERLAY_EVERY+1}-{episodes_so_far}")

        if episodes_so_far % CHECKPOINT_EVERY == 0:
            save_checkpoint()

except KeyboardInterrupt:
    print("Interrupted — saving checkpoint...")
finally:
    save_checkpoint()
    if trail_history:
        save_overlay("trails_final.png", f"up to ep{episodes_so_far}")
    print(f"Checkpoint saved. Total episodes trained so far: {episodes_so_far}")