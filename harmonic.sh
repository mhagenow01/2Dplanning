
echo "--------------------------"
echo "| Processing PNG TO CSV  |"
echo "--------------------------"

python3 -c'import Utilities.sceneProcessor as SP; SP.pngToCSV("/home/mike/Documents/ME759/FinalProject/Scenes/Denver_1_256.png")'

# Compile
cd HarmonicFunctions
nvcc harmonic_main.cu harmonickernel.cu -Xcompiler -Wall -Xptxas -O3 -std c++14
cd ..

# Call C++
echo "------------------------------------"
echo "| Running CUDA (/Python) Kernel    |"
echo "------------------------------------"
./HarmonicFunctions/a.out Denver_1_256.csv Scenes/Denver_1_256_benchmark.csv ./


# Call plotting results
echo "--------------------------"
echo "| Plot Results           |"
echo "--------------------------"

python3 -c'import Utilities.sceneProcessor as SP; SP.plotTraj("Scenes/Denver_1_256.png","Denver_1_256_harmonic_paths.csv")'



################################
# Run pyHarmonic ###############
################################
##  Note: This takes a very, very long time to run and with 500 iterations, will not be very good!
# python3 -c'import Utilities.sceneProcessor as SP; SP.pngToCSV("/home/mike/Documents/ME759/FinalProject/Scenes/Denver_1_256.png")'
# python3 -c'import HarmonicFunctions.pyHarmonic as PH; PH.main("Denver_1_256.csv",200,300,500)'
# python3 -c'import Utilities.sceneProcessor as SP; SP.plotSolution("Denver_1_256.csv","Denver_1_256_processed.csv",192,945,200,300)'