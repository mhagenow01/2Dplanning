import glob
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import json


for benchmark_file in glob.glob('./*.json'):
    with open(benchmark_file) as fin:
        benchmark = json.load(fin)
    name = benchmark['Filepath']
    img_path = os.path.join('../Scenes/', name + '.png')
    img = cv2.imread(img_path)
    plt.figure()
    plt.imshow(img)
    for start, end in benchmark['Goals']:
        start[0], start[1] = start[1], start[0]
        end[0], end[1] = end[1], end[0]
        plt.scatter(*np.array((start, end)).T, color = 'blue')
    plt.savefig(f'{name}.png')
