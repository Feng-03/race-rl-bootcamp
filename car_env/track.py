# car_env/track.py
import json
import numpy as np
from scipy.interpolate import splprep, splev

class Track:
    N_PTS = 600
    HALF_WIDTH = 20.0

    def __init__(self, waypoints_file=None, raw_points=None, random_seed=None):
        if random_seed is not None:
            self.centerline = self._generate_random_centerline(random_seed)
        else:
            if raw_points is not None:
                raw = np.array(raw_points, dtype=float)
            elif waypoints_file is not None:
                with open(waypoints_file) as f:
                    raw = np.array(json.load(f), dtype=float)
            else:
                raise ValueError("Track needs waypoints_file, raw_points, or random_seed")

            tck, _ = splprep([raw[:, 0], raw[:, 1]], s=0, per=1)
            u = np.linspace(0, 1, self.N_PTS, endpoint=False)
            x, y = splev(u, tck)
            self.centerline = np.column_stack([x, y])

        self._compute_edges()

    def _generate_random_centerline(self, seed):
        """Procedurally generates a brand-new closed-loop track shape each call.
        Same 'radius varies with angle' idea as the original hand-written formula
        track, with randomized terms — guaranteed non-self-intersecting (radius is
        single-valued per angle), no manual drawing needed."""
        rng = np.random.default_rng(seed)
        base_r = rng.uniform(90, 130)
        n_terms = int(rng.integers(2, 5))
        t = np.linspace(0, 2 * np.pi, self.N_PTS, endpoint=False)
        r = np.full_like(t, base_r)

        amp_budget = base_r * 0.35   # keeps combined wobble safely under the base radius
        for _ in range(n_terms):
            freq = int(rng.integers(2, 6))
            amp = rng.uniform(0.2, 0.5) * amp_budget / n_terms
            phase = rng.uniform(0, 2 * np.pi)
            r = r + amp * np.cos(freq * t + phase)

        x = r * np.cos(t)
        y = r * np.sin(t)
        return np.column_stack([x, y])

    def _compute_edges(self):
        next_pts = np.roll(self.centerline, -1, axis=0)
        tangent = next_pts - self.centerline
        tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
        normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])

        self.left_edge = self.centerline + normal * self.HALF_WIDTH
        self.right_edge = self.centerline - normal * self.HALF_WIDTH

    def nearest_point(self, pos):
        dists = np.linalg.norm(self.centerline - pos, axis=1)
        idx = dists.argmin()
        return idx, dists[idx]

    def is_off_track(self, pos):
        _, dist = self.nearest_point(pos)
        return dist > self.HALF_WIDTH