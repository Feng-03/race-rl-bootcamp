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
    N_ACTIONS = 6
    STEER_SIGN = {0: -1, 1: -1, 2: 1, 3: 1}
    SMOOTHNESS_PENALTY = 0.1
    CENTER_BONUS = 0.5     # up to +50% reward on progress made while centered; 0% bonus at the wall
    TIME_COST = 0.03        # per-step cost — makes lingering/slow laps genuinely costly

    def __init__(self, track_files=None, random_track_prob=0.85):
        super().__init__()
        self.track_files = track_files if track_files is not None else sorted(glob.glob("tracks/waypoints*.json"))
        self.random_track_prob = random_track_prob
        if not self.track_files and self.random_track_prob < 1.0:
            self.random_track_prob = 1.0

        self._pick_new_track()

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(len(self.RAY_ANGLES_DEG) + 1,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(self.N_ACTIONS)

    def _pick_new_track(self):
        if self.track_files and random.random() > self.random_track_prob:
            self.current_track_file = random.choice(self.track_files)
            self.track = Track(waypoints_file=self.current_track_file)
        else:
            seed = random.randint(0, 2**31 - 1)
            self.current_track_file = f"random:{seed}"
            self.track = Track(random_seed=seed)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._pick_new_track()

        self.pos = self.track.centerline[0].copy()
        start_tangent = self.track.centerline[1] - self.track.centerline[0]
        self.angle = np.degrees(np.arctan2(start_tangent[1], start_tangent[0]))
        self.speed = 4.0
        self.steps = 0
        self.prev_idx = 0
        self.progress = 0
        self.prev_action = None
        self.trail = [self.pos.copy()]
        return self._get_obs(), {}

    def step(self, action):
        if action == 0:
            self.angle += 6
        elif action == 1:
            self.angle += 3
        elif action == 2:
            self.angle -= 3
        elif action == 3:
            self.angle -= 6
        elif action == 4:
            self.speed = max(1.0, self.speed - 1.0)
        elif action == 5:
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

        # Centering only pays out as a MULTIPLIER on real progress — zero progress means
        # zero bonus regardless of position, which removes the "linger near center/go fast
        # in place" exploit. TIME_COST makes every step genuinely cost something, so slow,
        # cautious laps can no longer out-score fast, confident ones.
        center_multiplier = 1.0 + self.CENTER_BONUS * max(0.0, 1.0 - dist / self.track.HALF_WIDTH)
        reward = delta * 0.3 * center_multiplier - self.TIME_COST

        cur_sign = self.STEER_SIGN.get(action, 0)
        prev_sign = self.STEER_SIGN.get(self.prev_action, 0)
        if cur_sign != 0 and prev_sign != 0 and cur_sign != prev_sign:
            reward -= self.SMOOTHNESS_PENALTY
        self.prev_action = action

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