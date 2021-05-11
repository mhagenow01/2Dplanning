#!/usr/bin/env bash
#SBATCH --job-name=FinalProjectProfileHagenow
#SBATCH -p wacc
#SBATCH --ntasks=1 --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=0-00:30:00
#SBATCH --output="FinalProjectProfileHagenow.out"
#SBATCH --error="FinalProjectProfileHagenow.err"

module load cuda/10.2

nvcc harmonic_main.cu harmonickernel.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std c++14 -o harmonicmain

rm profileharmonic.nsight-cuprof-report

ncu -o profileharmonic --set full ./harmonicmain Denver_1_256.csv 75 75 1000

./task3