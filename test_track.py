import glob
import matplotlib.pyplot as plt
from car_env.track import Track

files = sorted(glob.glob("tracks/waypoints*.json"))
print(f"Found {len(files)} track file(s): {files}")

for file in files:
    track = Track(file)
    fig, ax = plt.subplots()
    ax.plot(*track.centerline.T, 'k--', label='centerline')
    ax.plot(*track.left_edge.T, 'b-')
    ax.plot(*track.right_edge.T, 'r-')
    ax.set_aspect('equal')
    ax.set_title(file)
    ax.legend()

plt.show()