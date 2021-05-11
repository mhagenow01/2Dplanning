#!/usr/bin/env python

""" Rewrite of python baseline harmonic field for comparison
 Created: 05/05/2021
"""

__author__ = "Mike Hagenow"

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from scipy import interpolate
import time

def getVal(x,y,grid):
    x_lim = np.shape(grid)[0]
    y_lim = np.shape(grid)[1]
    if(x>=0 and y>=0 and x<x_lim and y<y_lim):
        return grid[x,y]
    else:
        return 0.0

def setConstraints(grid,mask):
    grid = np.where(mask == 18446744073709551615/4, 18446744073709551615/4, grid)
    grid = np.where(mask == 0, 0, grid)

def main(file,attractor_row,attractor_col,max_it):
    mask = np.genfromtxt(file, delimiter=',',skip_header=1).astype(int)
    # Use same data ranges as C++ for fair comparison of performance
    
    mask = np.where(mask == 1, int(int(18446744073709551615)/4), mask)
    mask = np.where(mask == 0, int((int(18446744073709551615)/4)/2), mask)
    mask[attractor_row,attractor_col] = int(0)
    grid = np.copy(mask)

    if max_it>20:
        print_cond = max_it/20
    else:
        print_cond=1

    converged = False
    it=0

    start = time.time()
    np.seterr(all = "raise")

    while(not converged and it<max_it):
        grid2 = np.copy(grid)
        if it%print_cond==0:
                    print("Run "+str(it)+" of max "+str(max_it))
        for ii in range(0,np.shape(grid)[0]):
            for jj in range(0,np.shape(grid)[1]):
                grid2[ii,jj] = (int(getVal(ii+1,jj,grid))+int(getVal(ii-1,jj,grid))+int(getVal(ii,jj+1,grid))+int(getVal(ii,jj-1,grid)))/4
        # reinforce boundaries
        setConstraints(grid2,mask)
        grid = grid2
        it+=1

    end = time.time()

    print("Total Time (s): ",end-start)
    # Write Processed File
    outfile = file.split("/")[-1].split(".")[0]+"_processed.csv"
    f = open(outfile, "w")
    for ii in range(0,np.shape(grid)[0]):
        for jj in range(0,np.shape(grid)[1]):
            f.write(str(grid[ii,jj]))
            if(jj<(np.shape(grid)[1]-1)):
                f.write(",")
        f.write("\n")
    f.close()

if __name__ == "__main__":
    main(file)




