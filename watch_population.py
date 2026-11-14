import os
import torch
import numpy as np
from car_env.batch_env import BatchRaceEnv
from car_env.renderer import Renderer
from car_env.agent import DQNAgent

N_CARS = 100
QNET_FILE = os.path.join("runs", "q_net.pt")

STEPS_PER_FRAME = 3      # higher = faster on-screen playback (more sim steps per render)
DISPLAY_SMOOTHING = 0.4  # 1.0 = no smoothing (snap to true pos); lower = smoother but laggier

env = BatchRaceEnv(n_cars=N_CARS)
agent = DQNAgent(obs_dim=env.OBS_DIM, n_actions=env.N_ACTIONS)
agent.q_net.load_state_dict(torch.load(QNET_FILE, weights_only=True))
agent.q_net.eval()

renderer = Renderer(env.track)
obs = env.get_obs_all()

display_pos = env.pos.copy()
display_angle = env.angle.copy()

running = True
while running:
    for _ in range(STEPS_PER_FRAME):
        actions = agent.act_batch(obs, epsilon=0.0)
        obs, reward, terminated, truncated, off_track = env.step_all(actions)
        done = terminated | truncated

        for i in np.where(done)[0]:
            env.reset_one(i)
            display_pos[i] = env.pos[i]       # snap smoothing state on respawn — no "sliding back"
            display_angle[i] = env.angle[i]
        obs = env.get_obs_all()

    display_pos += (env.pos - display_pos) * DISPLAY_SMOOTHING
    display_angle += (env.angle - display_angle) * DISPLAY_SMOOTHING

    cars = list(zip(display_pos.tolist(), display_angle.tolist()))
    running = renderer.draw_population(cars, [f"{N_CARS} cars — greedy policy"], title="Trained policy")