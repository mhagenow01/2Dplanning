import numpy as np
import cv2
import sys
from numpy.lib.function_base import gradient
import pygame
from matplotlib.animation import FuncAnimation, ArtistAnimation
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.ndimage import distance_transform_edt
from scipy.optimize import minimize
from scipy.interpolate import RectBivariateSpline
from scipy import ndimage
from mpl_toolkits.mplot3d import Axes3D


def genenrate_potential(start, end, img):
    img = np.sum(img, axis = 2) > 0

    goal_distance = np.ones_like(img)
    goal_distance[(*np.array(end, dtype = np.int),)] = 0
    goal_distance = distance_transform_edt(goal_distance)
    distance = distance_transform_edt(img)

    potential = np.zeros_like(distance)
    d0 = 100
    mask = distance < d0
    potential[mask] = (1/(distance[mask]+0.1) - 1/d0)**2
    potential += (goal_distance/10)**2

    ax = plt.gca(projection = '3d')
    x = np.arange(potential.shape[0])
    y = np.arange(potential.shape[1])
    X, Y = np.meshgrid(x, y)
    ax.plot_wireframe(X, Y, potential)
    plt.show()

    #return ndimage.gaussian_filter(potential, )
    return potential

def grad(cost, x, f):
    return np.array([
        cost(x + np.array((1, 0))) - f,
        cost(x + np.array((0,1))) - f
    ]).reshape((2,))
        
def gradient_descent(cost, x0):
    last_f = cost(x0)
    f = np.inf
    gamma = 1
    gd = grad(cost, x0, last_f)
    x = x0
    while True:
        x = x - gamma * gd
        print(gamma)
        f = cost(x)
        if abs(f - last_f) <= 1e-8:
            break

        if f > last_f:
            x += gamma * gd
            gamma *= 0.5
        else:
            yield x
            last_f = f
            gd = grad(cost, x, f)
            gamma *= 1.1
            gamma = min(10, gamma * np.linalg.norm(gd)) / np.linalg.norm(gd)
    return x

def find_path(start, end, img):
    potential = genenrate_potential(start, end, img)

    interpolator = RectBivariateSpline(np.arange(potential.shape[0]), np.arange(potential.shape[1]), potential)
    
    L = 0.01

    start, end = np.array(start, dtype = np.float), np.array(end, dtype = np.float)

    def cost(x):
        index = np.array(x, dtype = np.int)
        if np.any((index < 0) | (index > img.shape[:2])):
            return 1e5
        return interpolator(*x)
    

    for x in list(gradient_descent(cost, start)):
        yield x
    return None

    history = []
    def callback(x):
        nonlocal history
        history.append(x)
    res = minimize(cost, start, method = 'SLSQP', callback = callback)
    print(res)
    print(end)
    print(history)
    for x in history:
        yield x

    return None
    position = start
    while np.linalg.norm(position - end) > 1:
        x, y = np.array(position, dtype = np.int)
        gradient = np.array([
            (potential[x + 1, y] - potential[x - 1, y]) / (2),
            (potential[x, y + 1] - potential[x, y - 1]) / (2),
        ])

        position -= 1 / (2 * L) * gradient
        yield position
    return None


def show_path_finding(start, end, img):
    plt.axis('off')
    def draw_line(start, end, img):
        delta = end - start

        h = 1
        for p in np.linspace(start, end, int(np.linalg.norm(delta) / h)):
            x = int(p[0])
            y = int(p[1])
            img[x-3:x+3, y-3:y+3,:] = (0,0,255)


    path_generator = find_path(start, end, img)
    last = start.copy()
    running = True
    def update(t):
        nonlocal last, path_generator, running
        if running:
            try: 
                position = next(path_generator)
                draw_line(last, position, img)
                last = position.copy()
            except StopIteration:
                running = False

        plt.gca().clear()
        plt.imshow(img.transpose((1,0,2)), origin = 'upper')
        return

    #ani = FuncAnimation(plt.gcf(), update, np.arange(0, 1000, 1e-4), interval = 1)


    last = start.copy()
    for x in find_path(start, end, img):
        draw_line(last, x, img)
        last = x
    plt.imshow(img.transpose((1,0,2)), origin = 'upper')
    plt.show()

    return

def select_start_end(img):
    pygame.init()
    gameDisplay = pygame.display.set_mode(img.shape[:2])

    clock = pygame.time.Clock()

    start, end = None, None

    img_surf = pygame.surfarray.make_surface(img)
    while start is None or end is None:
        for e in pygame.event.get():
            if e.type == pygame.MOUSEBUTTONUP:
                if start is None:
                    start = pygame.mouse.get_pos()
                else:
                    end = pygame.mouse.get_pos()

        gameDisplay.fill((0,0,0))
        gameDisplay.blit(img_surf, (0,0))
        if start is not None:
            pygame.draw.circle(gameDisplay, (0, 0, 255), start, 3)
        pygame.display.update()
        clock.tick(60)

    pygame.quit()
    return np.array(start), np.array(end)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Enter a png to use as terrain')
        raise ValueError('Needs input')
    
    img = cv2.imread(sys.argv[1])
    
    show_path_finding(*select_start_end(img), img)
