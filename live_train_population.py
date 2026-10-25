import os
import glob
import random
import torch
from car_env.env import RaceEnv
from car_env.renderer import Renderer
from car_env.agent import DQNAgent

N_CARS = 100
SWITCH_TRACK_EVERY = 500

RUNS_DIR = "runs"
CHECKPOINT_FILE = os.path.join(RUNS_DIR, "checkpoint.pt")
QNET_FILE = os.path.join(RUNS_DIR, "q_net.pt")
os.makedirs(RUNS_DIR, exist_ok=True)

track_files = sorted(glob.glob("tracks/waypoints*.json"))
if not track_files:
    raise FileNotFoundError("No tracks/waypoints*.json files found — draw at least one track first.")

current_track_idx = 0
current_file = track_files[current_track_idx]

envs = [RaceEnv(track_files=[current_file]) for _ in range(N_CARS)]
agent = DQNAgent(obs_dim=envs[0].observation_space.shape[0], n_actions=envs[0].N_ACTIONS)

episodes_so_far = 0
total_resets = 0
best_progress_frac = 0.0
exploration_actions, total_actions = 0, 0

if os.path.exists(CHECKPOINT_FILE):
    ckpt = torch.load(CHECKPOINT_FILE, weights_only=False)
    agent.q_net.load_state_dict(ckpt["q_net"])
    agent.target_net.load_state_dict(ckpt["target_net"])
    agent.optimizer.load_state_dict(ckpt["optimizer"])
    agent.train_steps = ckpt["train_steps"]
    episodes_so_far = ckpt["episodes_so_far"]
    best_progress_frac = ckpt["best_progress_frac"]
    print(f"Resumed from checkpoint: {episodes_so_far} episodes already trained")

obs_list = [env.reset()[0] for env in envs]
renderer = Renderer(envs[0].track)

running = True
while running:
    epsilon = max(0.05, 1.0 - episodes_so_far / 1800)

    for i, env in enumerate(envs):
        action = agent.act(obs_list[i], epsilon)
        total_actions += 1
        if random.random() < epsilon:
            exploration_actions += 1

        next_obs, reward, terminated, truncated, info = env.step(action)
        agent.store(obs_list[i], action, reward, next_obs, terminated)

        if terminated or truncated:
            episodes_so_far += 1
            total_resets += 1
            best_progress_frac = max(best_progress_frac, env.progress / env.track.N_PTS)
            obs_list[i], _ = env.reset()
        else:
            obs_list[i] = next_obs

    agent.train_step()

    if total_resets >= SWITCH_TRACK_EVERY:
        total_resets = 0
        current_track_idx = (current_track_idx + 1) % len(track_files)
        current_file = track_files[current_track_idx]
        for idx, env in enumerate(envs):
            env.track_files = [current_file]
            obs_list[idx], _ = env.reset()
        renderer.update_track(envs[0].track)

    exploration_pct = 100 * exploration_actions / max(1, total_actions)
    hud = [f"Episodes: {episodes_so_far}", f"Epsilon: {epsilon:.2f}",
           f"Best: {best_progress_frac*100:.0f}%", f"Exploration: {exploration_pct:.0f}%",
           f"Track: {current_file}"]
    cars = [(env.pos, env.angle) for env in envs]
    running = renderer.draw_population(cars, hud, title=f"{N_CARS} cars")

    if episodes_so_far % 100 < N_CARS:
        float_best = float(best_progress_frac)  # ensure plain python float, not numpy
        torch.save({
            "q_net": agent.q_net.state_dict(), "target_net": agent.target_net.state_dict(),
            "optimizer": agent.optimizer.state_dict(), "train_steps": agent.train_steps,
            "episodes_so_far": episodes_so_far, "best_progress_frac": float_best,
        }, CHECKPOINT_FILE)
        torch.save(agent.q_net.state_dict(), QNET_FILE)

print(f"Stopped. Total episodes trained: {episodes_so_far}")