#!/usr/bin/env python

""" Segment the demonstration based on force contact events using filtered
    force signals
 Created: 01/20/2021
"""

__author__ = "Mike Hagenow"

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from scipy import interpolate

def getValInterp(x,y,grid):
    x_lim = np.shape(grid)[0]
    y_lim = np.shape(grid)[1]
    if(x>=0 and y>=0 and x<x_lim and y<y_lim):
        a = range(0,x_lim)
        b = range(0,y_lim)
        f = interpolate.interp2d(a,b,grid)
        return f(x,y)
    else:
        return 0.0

def getVal(x,y,grid):
    x_lim = np.shape(grid)[0]
    y_lim = np.shape(grid)[1]
    if(x>=0 and y>=0 and x<x_lim and y<y_lim):
        return grid[x,y]
    else:
        return 0.0

def setBound(grid,val=0.0):
    grid[15:25,10]=val
    grid[25,10:20]=val
    grid[15,10:20]=val
    grid[15:22,20] = val
    grid[18:25,15] = val

    #start/goal
    grid[20,11] = 5.0

def main():
    grid = 0.5*np.ones((30,30))

    setBound(grid)

    fig, ax = plt.subplots()
    ax.imshow(grid, cmap="Blues")
    plt.show()


    for aa in range(0,2000):

        grid2 = np.copy(grid)
        for ii in range(0,30):
            for jj in range(0,30):
                    grid2[ii,jj] = 1/4*(getVal(ii+1,jj,grid)+getVal(ii-1,jj,grid)+getVal(ii,jj+1,grid)+getVal(ii,jj-1,grid))
        # reinforce boundaries
        setBound(grid2)
        grid = grid2

    print("YO")

    i = 3
    j = 3
    xs = []
    ys = []
    for ii in range(0,300):
        l = getVal(i,j-1,grid)
        r = getVal(i,j+1,grid)
        u = getVal(i-1,j,grid)
        d = getVal(i+1,j,grid)

        if(l>r and l>u and l>d):
            i = i
            j = j-1
        elif (r > u and r > d):
            i = i
            j = j+1
        elif(u>d):
            i = i-1
            j = j
        else:
            i = i+1
            j = j

        print(i,j)
        xs.append(i)
        ys.append(j)

    fig, ax = plt.subplots()
    # make the obstacles dark for vis
    setBound(grid,5.0)
    ax.imshow(grid, cmap="Blues")
    plt.plot(ys,xs,color='red')
    plt.show()


if __name__ == "__main__":
    main()




