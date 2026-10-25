import os
import random
import torch
from car_env.env import RaceEnv
from car_env.renderer import Renderer
from car_env.agent import DQNAgent

EPISODES_THIS_RUN = 500
TRAIL_BUFFER_SIZE = 400

RUNS_DIR = "runs"
CHECKPOINT_FILE = os.path.join(RUNS_DIR, "checkpoint.pt")
QNET_FILE = os.path.join(RUNS_DIR, "q_net.pt")
os.makedirs(RUNS_DIR, exist_ok=True)

env = RaceEnv()
agent = DQNAgent(obs_dim=env.observation_space.shape[0], n_actions=env.N_ACTIONS)

episodes_so_far = 0
best_progress_frac = 0.0
exploration_actions, total_actions = 0, 0
trail_history = []

if os.path.exists(CHECKPOINT_FILE):
    ckpt = torch.load(CHECKPOINT_FILE, weights_only=False)
    agent.q_net.load_state_dict(ckpt["q_net"])
    agent.target_net.load_state_dict(ckpt["target_net"])
    agent.optimizer.load_state_dict(ckpt["optimizer"])
    agent.train_steps = ckpt["train_steps"]
    episodes_so_far = ckpt["episodes_so_far"]
    best_progress_frac = ckpt["best_progress_frac"]
    print(f"Resumed from checkpoint: {episodes_so_far} episodes already trained")

obs, info = env.reset()
renderer = Renderer(env.track)

running = True
for i in range(EPISODES_THIS_RUN):
    if not running:
        break
    episodes_so_far += 1
    epsilon = max(0.05, 1.0 - episodes_so_far / 1800)
    obs, info = env.reset()
    renderer.update_track(env.track)
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

    best_progress_frac = max(best_progress_frac, env.progress / env.track.N_PTS)
    trail_history.append((env.track, env.trail, info["off_track"]))
    if len(trail_history) > TRAIL_BUFFER_SIZE:
        trail_history.pop(0)

    exploration_pct = 100 * exploration_actions / max(1, total_actions)
    hud = [f"Episode: {episodes_so_far}", f"Epsilon: {epsilon:.2f}",
           f"Best: {best_progress_frac*100:.0f}%", f"Exploration: {exploration_pct:.0f}%"]
    running = renderer.draw_trail_buildup(trail_history, hud, title=f"#{episodes_so_far}")

torch.save({
    "q_net": agent.q_net.state_dict(), "target_net": agent.target_net.state_dict(),
    "optimizer": agent.optimizer.state_dict(), "train_steps": agent.train_steps,
    "episodes_so_far": episodes_so_far, "best_progress_frac": float(best_progress_frac),
}, CHECKPOINT_FILE)
torch.save(agent.q_net.state_dict(), QNET_FILE)
print(f"Saved checkpoint at episode {episodes_so_far}")