# car_env/batch_env.py
import glob
import random
import numpy as np
from car_env.track import Track

class BatchRaceEnv:
    RAY_ANGLES_DEG = np.array([-60, -30, 0, 30, 60])
    RAY_MAX_LEN = 120.0
    RAY_STEP = 8.0
    DT = 0.1
    MAX_SPEED = 14.0
    MAX_STEPS = 1000
    N_ACTIONS = 6
    OBS_DIM = len(RAY_ANGLES_DEG) + 1
    STEER_SIGN = np.array([-1, -1, 1, 1, 0, 0])
    SMOOTHNESS_PENALTY = 0.1
    CENTER_BONUS = 0.5
    TIME_COST = 0.03

    def __init__(self, n_cars, track_files=None):
        self.n_cars = n_cars
        self.track_files = track_files or sorted(glob.glob("tracks/waypoints*.json"))
        if not self.track_files:
            raise FileNotFoundError("No tracks/waypoints*.json files found — draw at least one track first.")
        self.set_track(random.choice(self.track_files))
        self.reset_all()

    def set_track(self, track_file=None, random_seed=None):
        if random_seed is not None:
            self.current_track_file = f"random:{random_seed}"
            self.track = Track(random_seed=random_seed)
        else:
            self.current_track_file = track_file
            self.track = Track(waypoints_file=track_file)
        self._centerline32 = self.track.centerline.astype(np.float32)

    def reset_all(self):
        start = self.track.centerline[0]
        tangent = self.track.centerline[1] - self.track.centerline[0]
        start_angle = np.degrees(np.arctan2(tangent[1], tangent[0]))

        self.pos = np.tile(start, (self.n_cars, 1)).astype(np.float64)
        self.angle = np.full(self.n_cars, start_angle, dtype=np.float64)
        self.speed = np.full(self.n_cars, 4.0, dtype=np.float64)
        self.steps = np.zeros(self.n_cars, dtype=int)
        self.prev_idx = np.zeros(self.n_cars, dtype=int)
        self.progress = np.zeros(self.n_cars, dtype=int)
        self.prev_action = np.full(self.n_cars, -1, dtype=int)

    def reset_one(self, i):
        start = self.track.centerline[0]
        tangent = self.track.centerline[1] - self.track.centerline[0]
        self.pos[i] = start
        self.angle[i] = np.degrees(np.arctan2(tangent[1], tangent[0]))
        self.speed[i] = 4.0
        self.steps[i] = 0
        self.prev_idx[i] = 0
        self.progress[i] = 0
        self.prev_action[i] = -1

    def _nearest_batch(self, points):
        pts32 = points.astype(np.float32)
        diffs = pts32[:, None, :] - self._centerline32[None, :, :]
        d = np.linalg.norm(diffs, axis=2)
        idx = d.argmin(axis=1)
        dist = d[np.arange(len(points)), idx]
        return idx, dist

    def _cast_rays_all(self):
        angles = np.radians(self.angle[:, None] + self.RAY_ANGLES_DEG[None, :])
        dirs = np.stack([np.cos(angles), np.sin(angles)], axis=-1)

        steps = np.arange(self.RAY_STEP, self.RAY_MAX_LEN, self.RAY_STEP)
        pts = self.pos[:, None, None, :] + dirs[:, :, None, :] * steps[None, None, :, None]
        flat = pts.reshape(-1, 2)
        _, dist_flat = self._nearest_batch(flat)
        off = (dist_flat > self.track.HALF_WIDTH).reshape(self.n_cars, len(self.RAY_ANGLES_DEG), len(steps))

        hit = off.any(axis=2)
        first_hit = np.where(hit, off.argmax(axis=2), len(steps) - 1)
        return np.where(hit, steps[first_hit], self.RAY_MAX_LEN)

    def get_obs_all(self):
        rays = self._cast_rays_all()
        return np.concatenate(
            [rays / self.RAY_MAX_LEN, (self.speed / self.MAX_SPEED)[:, None]], axis=1
        ).astype(np.float32)

    def step_all(self, actions):
        actions = np.asarray(actions)
        self.angle = (self.angle
                      + np.where(actions == 0, 6, 0) + np.where(actions == 1, 3, 0)
                      - np.where(actions == 2, 3, 0) - np.where(actions == 3, 6, 0))
        self.speed = np.where(actions == 4, np.maximum(1.0, self.speed - 1.0), self.speed)
        self.speed = np.where(actions == 5, np.minimum(self.MAX_SPEED, self.speed + 1.0), self.speed)

        rad = np.radians(self.angle)
        self.pos = self.pos + self.speed[:, None] * self.DT * np.stack([np.cos(rad), np.sin(rad)], axis=1)
        self.steps += 1

        idx, dist = self._nearest_batch(self.pos)
        delta = (idx - self.prev_idx + self.track.N_PTS // 2) % self.track.N_PTS - self.track.N_PTS // 2
        self.prev_idx = idx
        self.progress += delta

        off_track = dist > self.track.HALF_WIDTH
        lap_done = self.progress >= self.track.N_PTS

        center_multiplier = 1.0 + self.CENTER_BONUS * np.maximum(0.0, 1.0 - dist / self.track.HALF_WIDTH)
        reward = delta * 0.3 * center_multiplier - self.TIME_COST

        cur_sign = self.STEER_SIGN[actions]
        prev_valid = self.prev_action >= 0
        prev_sign = np.where(prev_valid, self.STEER_SIGN[np.clip(self.prev_action, 0, 5)], 0)
        flip = (cur_sign != 0) & (prev_sign != 0) & (cur_sign != prev_sign)
        reward = reward - np.where(flip, self.SMOOTHNESS_PENALTY, 0.0)
        self.prev_action = actions.copy()

        reward = np.where(off_track, -10.0, reward)
        reward = np.where(lap_done, reward + 20.0, reward)

        terminated = off_track | lap_done
        truncated = self.steps >= self.MAX_STEPS

        return self.get_obs_all(), reward, terminated, truncated, off_track