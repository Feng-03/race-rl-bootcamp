# car_env/renderer.py
import pygame
import numpy as np

class Renderer:
    def __init__(self, track, width=900, height=700, scale=1.0):
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Race Track")
        self.clock = pygame.time.Clock()
        self.scale = scale
        self.update_track(track)

    def update_track(self, track):
        self.track = track
        self.offset = np.array([self.screen.get_width() / 2, self.screen.get_height() / 2]) \
            - track.centerline.mean(axis=0)

    def _to_screen(self, pos):
        return (pos + self.offset) * self.scale

    def draw(self, car_pos, car_angle=0):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return False

        self.screen.fill((30, 30, 30))
        left = [self._to_screen(p) for p in self.track.left_edge]
        right = [self._to_screen(p) for p in self.track.right_edge]
        pygame.draw.lines(self.screen, (255, 255, 255), True, left, 2)
        pygame.draw.lines(self.screen, (255, 255, 255), True, right, 2)

        car_screen = self._to_screen(np.array(car_pos))
        car_surf = pygame.Surface((24, 14), pygame.SRCALPHA)
        pygame.draw.rect(car_surf, (255, 60, 60), (0, 0, 24, 14), border_radius=4)
        pygame.draw.rect(car_surf, (255, 255, 255), (18, 4, 6, 6))
        rotated = pygame.transform.rotate(car_surf, -car_angle)
        rect = rotated.get_rect(center=car_screen)
        self.screen.blit(rotated, rect)

        pygame.display.flip()
        self.clock.tick(60)
        return True

    def draw_trail_buildup(self, trail_history, hud_lines, title=""):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return False

        self.screen.fill((10, 10, 10))
        left = [self._to_screen(p) for p in self.track.left_edge]
        right = [self._to_screen(p) for p in self.track.right_edge]
        pygame.draw.lines(self.screen, (120, 120, 120), True, left, 1)
        pygame.draw.lines(self.screen, (120, 120, 120), True, right, 1)

        trail_surf = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
        for track, trail, crashed in trail_history:
            pts = [self._to_screen(p) for p in trail]
            if len(pts) > 1:
                pygame.draw.lines(trail_surf, (255, 40, 40, 18), False, pts, 2)
            if crashed:
                pygame.draw.circle(trail_surf, (150, 150, 150, 120), pts[-1], 3)
        self.screen.blit(trail_surf, (0, 0))

        font = pygame.font.SysFont("Arial", 18)
        for i, line in enumerate(hud_lines):
            text = font.render(line, True, (255, 255, 255))
            self.screen.blit(text, (self.screen.get_width() - 260, 20 + i * 24))

        title_font = pygame.font.SysFont("Arial", 28, bold=True)
        self.screen.blit(title_font.render(title, True, (255, 60, 60)), (20, 20))

        pygame.display.flip()
        self.clock.tick(30)
        return True

    def draw_population(self, cars, hud_lines, title=""):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return False

        self.screen.fill((20, 20, 20))
        left = [self._to_screen(p) for p in self.track.left_edge]
        right = [self._to_screen(p) for p in self.track.right_edge]
        pygame.draw.lines(self.screen, (255, 255, 255), True, left, 2)
        pygame.draw.lines(self.screen, (255, 255, 255), True, right, 2)

        for pos, angle in cars:
            p = self._to_screen(np.array(pos))
            pygame.draw.circle(self.screen, (255, 70, 70), p.astype(int), 4)

        font = pygame.font.SysFont("Arial", 18)
        for i, line in enumerate(hud_lines):
            text = font.render(line, True, (255, 255, 255))
            self.screen.blit(text, (self.screen.get_width() - 260, 20 + i * 24))

        title_font = pygame.font.SysFont("Arial", 28, bold=True)
        self.screen.blit(title_font.render(title, True, (255, 60, 60)), (20, 20))

        pygame.display.flip()
        self.clock.tick(60)   # was 30 — smoother render cadence
        return True