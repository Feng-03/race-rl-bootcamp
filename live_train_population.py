import os
import glob
import random
import torch
import numpy as np
from car_env.batch_env import BatchRaceEnv
from car_env.renderer import Renderer
from car_env.agent import DQNAgent

N_CARS = 100
SWITCH_TRACK_EVERY = 500
RANDOM_TRACK_PROB_ON_SWITCH = 0.6   # how often a switch goes to a brand-new random track vs a fixed one
STEPS_PER_FRAME = 3
DISPLAY_SMOOTHING = 0.4

RUNS_DIR = "runs"
CHECKPOINT_FILE = os.path.join(RUNS_DIR, "checkpoint_population.pt")
QNET_FILE = os.path.join(RUNS_DIR, "q_net.pt")
os.makedirs(RUNS_DIR, exist_ok=True)

track_files = sorted(glob.glob("tracks/waypoints*.json"))
if not track_files:
    raise FileNotFoundError("No tracks/waypoints*.json files found — draw at least one track first.")

env = BatchRaceEnv(n_cars=N_CARS, track_files=track_files)
agent = DQNAgent(obs_dim=env.OBS_DIM, n_actions=env.N_ACTIONS)

current_track_idx = track_files.index(env.current_track_file) if env.current_track_file in track_files else 0
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
    print(f"Resumed from population checkpoint: {episodes_so_far} episodes already trained")
elif os.path.exists(QNET_FILE):
    agent.q_net.load_state_dict(torch.load(QNET_FILE, weights_only=True))
    agent.target_net.load_state_dict(agent.q_net.state_dict())
    print("No population checkpoint found — bootstrapped from existing q_net.pt")
else:
    print("No checkpoint or q_net.pt found — starting completely fresh")

obs = env.get_obs_all()
renderer = Renderer(env.track)

display_pos = env.pos.copy()
display_angle = env.angle.copy()

running = True
while running:
    epsilon = max(0.05, 1.0 - episodes_so_far / 1800)

    for _ in range(STEPS_PER_FRAME):
        actions = agent.act_batch(obs, epsilon)
        exploration_actions += int(np.sum(np.random.rand(N_CARS) < epsilon))
        total_actions += N_CARS

        next_obs, reward, terminated, truncated, off_track = env.step_all(actions)
        done = terminated | truncated

        for i in range(N_CARS):
            agent.store(obs[i], actions[i], reward[i], next_obs[i], bool(terminated[i]))
        agent.train_step()

        done_idx = np.where(done)[0]
        for i in done_idx:
            episodes_so_far += 1
            total_resets += 1
            best_progress_frac = max(best_progress_frac, env.progress[i] / env.track.N_PTS)
            env.reset_one(i)
            display_pos[i] = env.pos[i]
            display_angle[i] = env.angle[i]

        obs = env.get_obs_all()

        if total_resets >= SWITCH_TRACK_EVERY:
            total_resets = 0
            if random.random() < RANDOM_TRACK_PROB_ON_SWITCH:
                env.set_track(random_seed=random.randint(0, 2**31 - 1))
            else:
                current_track_idx = (current_track_idx + 1) % len(track_files)
                env.set_track(track_files[current_track_idx])
            env.reset_all()
            obs = env.get_obs_all()
            display_pos = env.pos.copy()
            display_angle = env.angle.copy()
            renderer.update_track(env.track)

    display_pos += (env.pos - display_pos) * DISPLAY_SMOOTHING
    display_angle += (env.angle - display_angle) * DISPLAY_SMOOTHING

    exploration_pct = 100 * exploration_actions / max(1, total_actions)
    hud = [f"Episodes: {episodes_so_far}", f"Epsilon: {epsilon:.2f}",
           f"Best: {best_progress_frac*100:.0f}%", f"Exploration: {exploration_pct:.0f}%",
           f"Track: {env.current_track_file}"]
    cars = list(zip(display_pos.tolist(), display_angle.tolist()))
    running = renderer.draw_population(cars, hud, title=f"{N_CARS} cars")

    if episodes_so_far % 100 < N_CARS:
        torch.save({
            "q_net": agent.q_net.state_dict(), "target_net": agent.target_net.state_dict(),
            "optimizer": agent.optimizer.state_dict(), "train_steps": agent.train_steps,
            "episodes_so_far": episodes_so_far, "best_progress_frac": float(best_progress_frac),
        }, CHECKPOINT_FILE)
        torch.save(agent.q_net.state_dict(), QNET_FILE)

print(f"Stopped. Total episodes trained: {episodes_so_far}")