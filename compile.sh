
echo "--------------------------"
echo "| Compiling CUDA         |"
echo "--------------------------"


cd HarmonicFunctions
nvcc harmonic_main.cu harmonickernel.cu -Xcompiler -Wall -Xptxas -O3 -std c++14
cd ..


echo "--------------------------"
echo "| Compiling OpenMP        |"
echo "--------------------------"

cd PotentialFields
g++ main.cpp DistanceField.cpp -O3 -fopenmp
cd ..