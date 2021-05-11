#!/usr/bin/env python

""" Set of functions that can be used to process scenes, send data to C++,
    plot results etc.
 Created: 04/19/2021
"""

__author__ = "Mike Hagenow, Kevin Welsh"

import numpy as np
from numpy import genfromtxt
import cv2
import matplotlib.pyplot as plt

"""
Temp: takes full filename and stores in the working directory as a boolean array CSV
where 1 is a collision and 0 is free
"""
def pngToCSV(file, outfile = None):
    img = cv2.imread(file,0)
    # cv2.imshow('image',img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    if outfile is None:
        outfile = file.split("/")[-1].split(".")[0]+".csv"
    f = open(outfile, "w")

    print(img.shape)
    print(img.size)

    # First 2 numbers are the size of the problem
    f.write(str(np.shape(img)[0])+","+str(np.shape(img)[1])+"\n")
    for ii in range(0,np.shape(img)[0]):
        for jj in range(0,np.shape(img)[1]):
            if(img[ii,jj]<=127.0 and jj==np.shape(img)[1]-1):
                f.write("1")
            elif(img[ii,jj]>127.0 and jj==np.shape(img)[1]-1):
                f.write("0")
            elif(img[ii,jj]<=127.0):
                f.write("1,")
            else:
                f.write("0,")
        f.write("\n")
    f.close()


def plotOutput(file):
    processedFile = genfromtxt(file, delimiter=',')
    im = plt.imshow(processedFile, cmap=plt.cm.RdBu, extent=(0,18446744073709551615/4+1,18446744073709551615/4+1,0))
    plt.colorbar(im)  
    # plt.title('$z=(1-x^2+y^3) e^{-(x^2+y^2)/2}$')
    plt.show()

def plotTraj(coll_file_png,traj_file):
    img = cv2.imread(coll_file_png)
    plt.figure()
    plt.imshow(img.transpose(1,0,2))
    with open(traj_file) as fin:
        solutions = list(fin)#[:-1]
        print(list(fin))
    print(len(solutions))
    for solution in solutions:
        start, end, traj = solution.split('|')
        traj = np.array(list(map(lambda x: list(map(float, x.split(','))), traj.split(';')[:-1]))).T
        start = np.array(list(map(float, start.split(',')))).T
        end = np.array(list(map(float, end.split(',')))).T

        plt.scatter(*traj, color = 'blue')
        plt.scatter(*end, color = 'green',s=95)
        plt.scatter(*start, color = 'red',s=95)
    plt.show()

def getVal(x,y,grid):
    x_lim = np.shape(grid)[0]
    y_lim = np.shape(grid)[1]
    if(x>=0 and y>=0 and x<x_lim and y<y_lim):
        return grid[x,y]
    else:
        return 0.0

def getPath(grid_og,grid,row_start,col_start,row_goal,col_goal):
    finished = False
    i = row_start
    j = col_start
    xs = []
    ys = []
    num_its = 0
    while not finished:
        l = getVal(i,j-1,grid)
        r = getVal(i,j+1,grid)
        u = getVal(i-1,j,grid)
        d = getVal(i+1,j,grid)

        if(l<r and l<u and l<d):
            i = i
            j = j-1
        elif (r < u and r < d):
            i = i
            j = j+1
        elif(u<d):
            i = i-1
            j = j
        else:
            i = i+1
            j = j

        num_its+=1

        if (i==row_goal and j==col_goal) or num_its>100000:
            finished=True

        # print(i,j," - ",l,r,u,d)
        xs.append(i)
        ys.append(j)

    fig, ax = plt.subplots()
    ax.imshow(grid_og, cmap="Blues")
    plt.plot(ys,xs,color='red')
    plt.scatter(col_start,row_start,s=20,color='yellow')
    plt.scatter(col_goal,row_goal,s=20,c='green')
    plt.show()

def plotSolution(original_file,processed_file,row_start,col_start,row_goal,col_goal):
    grid = genfromtxt(processed_file, delimiter=',')
    grid_og = genfromtxt(original_file, delimiter=',',skip_header=1)
    getPath(grid_og,grid,row_start,col_start,row_goal,col_goal)


if __name__ == "__main__":
    # pngToCSV('/home/mike/Documents/ME759/FinalProject/Scenes/Berlin_2_1024.png')
    plotOutput('/home/mike/Documents/ME759/FinalProject/Utilities/output_processed.csv')
    # plotSolution('/home/mike/Documents/ME759/FinalProject/Utilities/output.csv','/home/mike/Documents/ME759/FinalProject/Utilities/output_processed.csv',970,286,200,300)write