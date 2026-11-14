import sys
import os
import torch
from car_env.env import RaceEnv
from car_env.renderer import Renderer
from car_env.agent import DQNAgent

if len(sys.argv) < 2:
    print("Usage: python3 test_new_track.py tracks/my_new_track.json")
    sys.exit(1)

track_path = sys.argv[1]
QNET_FILE = os.path.join("runs", "q_net.pt")

env = RaceEnv(track_files=[track_path], random_track_prob=0.0)  # force exactly this track, every episode
agent = DQNAgent(obs_dim=env.observation_space.shape[0], n_actions=env.N_ACTIONS)
agent.q_net.load_state_dict(torch.load(QNET_FILE, weights_only=True))
agent.q_net.eval()

obs, info = env.reset()
renderer = Renderer(env.track)

running = True
while running:
    action = agent.act(obs, epsilon=0.0)
    obs, reward, terminated, truncated, info = env.step(action)
    running = renderer.draw(env.pos, env.angle)
    if terminated or truncated:
        result = "LAP" if info["lap_done"] else ("CRASH" if info["off_track"] else "timeout")
        print(f"Episode ended — {result}, {env.progress} progress units, {env.steps} steps")
        obs, info = env.reset()