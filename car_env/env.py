# car_env/env.py
import glob
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from car_env.track import Track

class RaceEnv(gym.Env):
    RAY_ANGLES_DEG = [-60, -30, 0, 30, 60]
    RAY_MAX_LEN = 150.0
    DT = 0.1
    MAX_SPEED = 14.0
    MAX_STEPS = 1000
    N_ACTIONS = 4

    def __init__(self, track_files=None):
        super().__init__()
        self.track_files = track_files or sorted(glob.glob("tracks/waypoints*.json"))
        if not self.track_files:
            raise FileNotFoundError("No tracks/waypoints*.json files found — draw at least one track first.")

        self.track = Track(random.choice(self.track_files))

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(len(self.RAY_ANGLES_DEG) + 1,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(self.N_ACTIONS)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.track = Track(random.choice(self.track_files))

        self.pos = self.track.centerline[0].copy()
        start_tangent = self.track.centerline[1] - self.track.centerline[0]
        self.angle = np.degrees(np.arctan2(start_tangent[1], start_tangent[0]))
        self.speed = 4.0
        self.steps = 0
        self.prev_idx = 0
        self.progress = 0
        self.trail = [self.pos.copy()]
        return self._get_obs(), {}

    def step(self, action):
        if action == 0:
            self.angle += 3
        elif action == 1:
            self.angle -= 3
        elif action == 2:
            self.speed = max(1.0, self.speed - 1.0)
        elif action == 3:
            self.speed = min(self.MAX_SPEED, self.speed + 1.0)

        rad = np.radians(self.angle)
        self.pos = self.pos + self.speed * self.DT * np.array([np.cos(rad), np.sin(rad)])
        self.trail.append(self.pos.copy())
        self.steps += 1

        idx, dist = self.track.nearest_point(self.pos)
        delta = (idx - self.prev_idx + self.track.N_PTS // 2) % self.track.N_PTS - self.track.N_PTS // 2
        self.prev_idx = idx
        self.progress += delta

        off_track = dist > self.track.HALF_WIDTH
        lap_done = self.progress >= self.track.N_PTS

        reward = delta * 0.3 - 0.01
        if off_track:
            reward = -10.0
        elif lap_done:
            reward += 20.0

        terminated = off_track or lap_done
        truncated = self.steps >= self.MAX_STEPS

        return self._get_obs(), reward, terminated, truncated, {"off_track": off_track, "lap_done": lap_done}

    def _get_obs(self):
        rays = self._cast_rays()
        return np.append(rays / self.RAY_MAX_LEN, self.speed / self.MAX_SPEED).astype(np.float32)

    def _cast_rays(self):
        readings = []
        for deg in self.RAY_ANGLES_DEG:
            ray_angle = np.radians(self.angle + deg)
            direction = np.array([np.cos(ray_angle), np.sin(ray_angle)])
            dist = self.RAY_MAX_LEN
            for step_len in np.arange(2, self.RAY_MAX_LEN, 2):
                point = self.pos + direction * step_len
                _, edge_dist = self.track.nearest_point(point)
                if edge_dist > self.track.HALF_WIDTH:
                    dist = step_len
                    break
            readings.append(dist)
        return np.array(readings)