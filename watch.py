import os
import torch
from car_env.env import RaceEnv
from car_env.renderer import Renderer
from car_env.agent import DQNAgent

QNET_FILE = os.path.join("runs", "q_net.pt")

env = RaceEnv()
agent = DQNAgent(obs_dim=env.observation_space.shape[0], n_actions=env.N_ACTIONS)
agent.q_net.load_state_dict(torch.load(QNET_FILE))
agent.q_net.eval()

obs, info = env.reset()
renderer = Renderer(env.track)

running = True
while running:
    action = agent.act(obs, epsilon=0.0)
    obs, reward, terminated, truncated, info = env.step(action)
    running = renderer.draw(env.pos, env.angle)
    if terminated or truncated:
        obs, info = env.reset()
        renderer.update_track(env.track)