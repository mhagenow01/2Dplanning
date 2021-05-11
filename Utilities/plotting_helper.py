#!/usr/bin/env python

""" Set of plotting functions for particular long-running data
 Created: 05/04/2021
"""

__author__ = "Mike Hagenow, Kevin Welsh"

import numpy as np
from numpy import genfromtxt
import cv2
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'errorbar.capsize': 5})


def plot_SOR_results():
    x = np.array([1.0, 0.95, 0.90, 0.85, 0.80])
    Y = np.array([[137480, 143366, 150377, 158155, 166836],
                  [216795, 226671, 237642, 249805, 263371],
                  [272420, 284841, 298574, 313797, 330771],
                  [379469, 392054, 410711, 431373, 454393],
                  [230827, 241225, 252900, 265845, 280283],
                  [104168, 108781, 114124, 120053, 126673]])
    y_mean = np.mean(Y,axis=0)
    y_std = np.std(Y,axis=0)

    plt.errorbar(x, y_mean, y_std)
    plt.xlim(1.01,0.79)
    plt.xlabel("Update weight (w)")
    plt.ylabel("Number of Iterations")
    plt.title("Convergence using SOR")

    plt.show()
                

if __name__ == "__main__":
    plot_SOR_results()