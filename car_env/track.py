# car_env/track.py
import json
import numpy as np
from scipy.interpolate import splprep, splev

class Track:
    N_PTS = 600
    HALF_WIDTH = 20.0

    def __init__(self, waypoints_file="waypoints.json"):
        with open(waypoints_file) as f:
            raw = np.array(json.load(f), dtype=float)

        tck, _ = splprep([raw[:, 0], raw[:, 1]], s=0, per=1)
        u = np.linspace(0, 1, self.N_PTS, endpoint=False)
        x, y = splev(u, tck)
        self.centerline = np.column_stack([x, y])

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