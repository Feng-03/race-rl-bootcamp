# track_editor.py
import os
import pygame
import json
import sys

pygame.init()
screen = pygame.display.set_mode((900, 700))
pygame.display.set_caption("Click to place waypoints. Press S to save, C to clear.")
clock = pygame.time.Clock()

os.makedirs("tracks", exist_ok=True)
points = []

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()
        elif event.type == pygame.MOUSEBUTTONDOWN:
            points.append(list(event.pos))
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_c:
                points.clear()
            elif event.key == pygame.K_s:
                filename = input("Save as (e.g. waypoints_4.json): ")
                if not filename.endswith(".json"):
                    filename += ".json"
                filepath = os.path.join("tracks", filename)
                with open(filepath, "w") as f:
                    json.dump(points, f)
                print(f"Saved {len(points)} points to {filepath}")

    screen.fill((30, 30, 30))
    if len(points) > 1:
        pygame.draw.lines(screen, (255, 255, 255), True, points, 2)
    for p in points:
        pygame.draw.circle(screen, (255, 0, 0), p, 4)

    pygame.display.flip()
    clock.tick(60)