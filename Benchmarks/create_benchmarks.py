import numpy as np
import cv2
import pygame
import glob
import json
import os

def select_start_end(img, n = 5):
    pygame.init()
    gameDisplay = pygame.display.set_mode(img.shape[:2])

    clock = pygame.time.Clock()

    end = None
    starts = []

    img_surf = pygame.surfarray.make_surface(img)
    while len(starts) < n or end is None:
        for e in pygame.event.get():
            if e.type == pygame.MOUSEBUTTONUP:
                if end is None:
                    end = list(pygame.mouse.get_pos())
                else:
                    starts.append(list(pygame.mouse.get_pos()))

        gameDisplay.fill((0,0,0))
        gameDisplay.blit(img_surf, (0,0))
        if end is not None:
            pygame.draw.circle(gameDisplay, (0, 255, 0), end, 3)
        for s in starts:
            pygame.draw.circle(gameDisplay, (0, 0, 255), s, 3)
        pygame.display.update()
        clock.tick(60)

    pygame.quit()
    if end is None:
        return None, None
    for s in starts:
        s[1], s[0] = s[0], s[1]
    end[1], end[0] = end[0], end[1]
    return starts, end

if __name__ == '__main__':
    for file in glob.glob("../Scenes/*.png"):
        img = cv2.imread(file)

        name = os.path.split(file)[-1].split('.')[0]
        obj = {
            'Filepath' : name,
            'Goals' : []
        }
        starts, end = select_start_end(img.transpose(1,0,2), 5)
        for s in starts:
            obj['Goals'].append([s, end])
        
        with open(f'{name}.json', 'w') as fout:
            json.dump(obj, fout)