import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import glob
import json
import cv2
import Utilities.sceneProcessor as SP

for benchmark_file in glob.glob('./Benchmarks/*.json'):
    with open(benchmark_file) as fin:
        benchmark = json.load(fin)
    name = benchmark['Filepath'].split('.')[0]
    img_path = os.path.join('./Scenes/', name + '.png')
    print(img_path)
    img = cv2.imread(img_path)

    SP.pngToCSV(img_path, 'output.csv')
    
    with open('goals.csv', 'w') as fout:
        end = benchmark['Goals'][0][1]
        fout.write(','.join(map(str,end)))
        fout.write('\n')
        for goal in benchmark['Goals']:
            fout.write(','.join(map(str,goal[0])))
            fout.write('\n')

    if not os.path.isdir('./Results'):
        os.mkdir('./Results')
    results_folder = os.path.join('./Results/', name)
    if not os.path.isdir(results_folder):
        os.mkdir(results_folder)
    for file in glob.glob(f'{results_folder}/**', recursive = True):
        if os.path.isfile(file):
            os.remove(file)
    if not os.path.isdir(f'./Results/{name}/PotentialFields/'):
        os.mkdir(f'./Results/{name}/PotentialFields/')
    if not os.path.isdir(f'./Results/{name}/HarmonicFunctions/'):
        os.mkdir(f'./Results/{name}/HarmonicFunctions/')
        
    os.system(f'./PotentialFields/a.out output.csv goals.csv ./Results/{name}/PotentialFields/')
    os.system(f'./HarmonicFunctions/a.out output.csv ./goals.csv ./Results/{name}/HarmonicFunctions/')

    #os.remove('./output.csv')
    #os.remove('./goals.csv')
    
    for file in glob.glob(f'./Results/{name}/*/*.csv'):
        plt.figure()
        plt.imshow(img)
        plt.axis('off')
        with open(file) as fin:
            solutions = list(fin)
        for solution in solutions:
            start, end, traj = solution.split('|')
            traj = np.array(list(map(lambda x: list(map(float, x.split(','))), traj.split(';')[:-1])))
            start = np.array(list(map(float, start.split(','))))
            end = np.array(list(map(float, end.split(','))))


            traj[:,[0,1]] = traj[:,[1,0]]
            start[[0,1]] = start[[1,0]]
            end[[0,1]] = end[[1,0]]
            plt.plot(*traj.T, color = 'blue')
            plt.scatter(*traj.T, color = 'blue', s = 2)
            plt.scatter(*end, color = 'green', zorder = 5)
            plt.scatter(*start, color = 'red', zorder = 5)
        method_name = os.path.split(file)[-1].split('.')[0]
        fig_name = f'./Results/{name}/{method_name}.png'
        print('Saving figure: ' + fig_name)
        plt.tight_layout()
        plt.savefig(fig_name)
        
        