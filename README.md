# ME759 Final Project - Path Planning

Final project for Mike Hagenow and Kevin Welsh

This repository contains parallel version of Artificial Potential Fields (APF) and Harmonic Functions (HF) for use on 2D planning problems.
The APF approaches use OpenMP whereas the HF approach uses CUDA. Information regarding required  

![GitHub Logo](/Utilities/teaser_path_planning.png)

## Required Libraries
### Python
- OpenCV2
- Numpy
- PyGame
- Matplotlib
- JSON
- Glob

### C++
- CUDA (tested on 10.2)
- OpenMP

### Contents
* Scripts to launch routines
    * **compile.sh**: compiles all C++, including CUDA and OpenMP
    * **main.py:** runs artificial potential fields and harmonic potential fields for all JSON benchmarks in the Benchmarks directory. Stores pngs of the results in the Results directory.
    * **harmonic.sh**: runs one example of the harmonic field and plots the results.
* Benchmarks
    * **create_benchmarks.py**: Uses a gui to select start and end points for all files in the Scenes directory. The benchmarks are then saved as JSON files.
* Utilities
    * **sceneProcessor**: converts pngs to CSV, plots harmonic potential fields, plots results and paths
* Harmonic Functions
    * **harmonic_main**: loads CSV, times kernel calls, and writes results to a processed CSV
    * **harmonic_kernel.cu**: contains the kernel launching, exit conditions, as well as a variety of kernels (e.g., global memory, neumann conditions, shared memory, looping/double buffered attempts)
    * **pyHarmonic**: serial python version mimicking the CUDA kernel (e.g., same datatypes, conditions)
    * **testHarmonic**: proof of concept that runs and plots a simple harmonic potential field solution (starting point for project)
* Potential Fields
    * **main.cpp**: loads CSV image, writes trajectories, implements several utility functions (joins trajectories based on proximities, computes gradients of a function, performs gradient descent on a function). Implements primary search algorithms: gradient descent, bacterial foraging, genetic descent, and random descent.
    * **base_line.py**: Serial python gradient descent visualization.
    * **DistanceField.cpp**: parallel computation of euclidean distance transform from a binary image.
