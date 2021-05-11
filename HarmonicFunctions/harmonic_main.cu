// Mike Hagenow
// ME759 - Final Project
// Loads a collision map from a CSV and calls the CUDA kernel
// to calculate the Laplacian


// Compile: nvcc harmonic_main.cu harmonickernel.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std c++14 -o harmonicmain
// Debug: nvcc -g -G harmonic_main.cu harmonickernel.cu -Xcompiler -O3 -Xcompiler -Wall -O3 -std c++14 -o harmonicmain
// Example call: ./harmonicmain /home/mike/Documents/ME759/FinalProject/Utilities/output.csv 200 300
#include <cuda.h>
#include <limits.h>
#include <stdio.h>
#include <chrono>
#include <fstream>
#include "harmonickernel.cuh"
#include <vector>

using namespace std;


// Used during the final gradient descent
// This looks into the matrix and gets a value of a specific row and column
// while also handling bounds on the array
unsigned long long int getVal( int row, int col,unsigned long long int *hB, unsigned int num_rows, unsigned int num_cols){
    if(row>=0 && col>=0 && row<(int)num_rows && col<(int)num_cols){
        return hB[num_cols*row + col];
    }
    else{
        return 0;
    }
}

// This writes a CSV with the trajectory for a particular start point
// to the goal point
void write_trajectory(string filename, vector<double> start, vector<double> end, vector<vector<double>> traj) {
    ofstream fout(filename, ios_base::app);
    fout << start[0] << "," << start[1] << "|" << end[0] << "," << end[1] << "|";
    for(int i = 0; i < (int)traj.size(); i++) {
        fout << traj[i][0] << "," << traj[i][1] << ";";
    }
    fout << "\n";
    fout.close();
}

int main(int argc, char **argv){
    if(argc<=3){
        // No command line argument
        printf("Insufficient Command Line Arg!\n Expected file goal_row goal_col\n");
        return 0;
    }

    // CUDA configuration
    unsigned int max_iters = 4000000;
    unsigned int threads_per_block = 1024;

    string filename = argv[1]; // file of collsion info
    ifstream collisionFile(filename);
    string temp;

    // Results directory for storing paths
    string results_dir = argv[3];

    // Row and columns at top of the CSV
    unsigned int num_rows = 0;
    unsigned int num_cols = 0;
    if(collisionFile.good()){
        getline(collisionFile,temp,',');
        num_rows = atoi(temp.c_str());
        getline(collisionFile,temp);
        num_cols = atoi(temp.c_str());
    }

    // Allocate host and device memory
    unsigned long long int *hA, *dA;
    unsigned long long int *hB, *dB;
    unsigned long long int *dMask;
    hA = new unsigned long long int[num_rows*num_cols];
    hB = new unsigned long long int[num_rows*num_cols];
    cudaMalloc((void **)&dA, sizeof(unsigned long long int) * (num_rows*num_cols));
    cudaMalloc((void **)&dB, sizeof(unsigned long long int) * (num_rows*num_cols));
    cudaMalloc((void **)&dMask, sizeof(unsigned long long int) * (num_rows*num_cols));

    unsigned long long int max_val = ULLONG_MAX/4;

    // The collision file has 0 and 1. Convert to 0 and the maximum int
    // value for the unsigned long long int
    for(unsigned long long int i=0;i<num_rows*num_cols;i++){
        if(i%num_cols==num_cols-1){
            getline(collisionFile,temp);
        }
        else{
            getline(collisionFile,temp,',');
        }
        if(temp=="1"){
            hA[i]=max_val;
        }
        else{
            hA[i] = max_val/2;
        }
    }

    collisionFile.close();

    // Load the CSV of the goal position and starting positions
    string filepts = argv[2]; // size of one dimension of matrix
    ifstream ptsFile(filepts);

    getline(ptsFile,temp,',');
    int goal_row = atoi(temp.c_str());
    getline(ptsFile,temp);
    int goal_col = atoi(temp.c_str());

    printf("Goal: %d %d\n",goal_row,goal_col);

    // Get the goal position and adjust mask!
    unsigned int goal_index = goal_row*num_cols + goal_col;

    if(goal_index>(num_rows*num_cols-1)){
        printf("Invalid goal (bounds)\n");
        return 0;
    }
    else if(hA[goal_index]==max_val){
        printf("Invalid goal (goal is a collision)\n");
        return 0;
    }
    else{
        hA[goal_index] = 0; // negative potential (pull towards)
    }

    printf("File: %s\n",filename.c_str());
    // printf("%d %d\n",num_rows,num_cols);

    cudaMemcpy(dA, hA, sizeof(unsigned long long int) * (num_rows*num_cols), cudaMemcpyHostToDevice);
    cudaMemcpy(dMask, hA, sizeof(unsigned long long int) * (num_rows*num_cols), cudaMemcpyHostToDevice);

    // Run and time the kernel call (calling matmul function in this case)
    float ms;
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    harmonic(&dA,&dB,dMask,num_rows,num_cols,threads_per_block,max_iters);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);

    // Bring back results
    cudaMemcpy(hB, dB, sizeof(unsigned long long int) * (num_rows*num_cols), cudaMemcpyDeviceToHost);


    ///////////////////////////////////////////////////////
    // Gradient Descent for all remaining start points   //
    ///////////////////////////////////////////////////////

    // Load the remaining points
    vector<double> goal = vector<double>();
    goal.push_back((double)goal_row);
    goal.push_back((double)goal_col);

    vector<vector<vector<double>>> result;

    while (ptsFile.peek() != EOF) {
        vector<double> start = vector<double>();
        getline(ptsFile, temp, ',');
        start.push_back(stof(temp.c_str()));
        getline(ptsFile, temp);
        start.push_back(stof(temp.c_str()));
        result.push_back(vector<vector<double>>({start, goal}));
    }
    ptsFile.close();

    vector<vector<vector<double>>> trajs;

    int goal_row_temp, goal_col_temp, i , j;
    unsigned long long int l,r,u,d;
    // run the gradient descent for each goal point
    for(int k=0;k<(int)result.size();k++){
        vector<vector<double>> traj_temp;
        vector<double> point_temp;

        i = (int) result[k][0][0];
        j = (int) result[k][0][1];
        goal_row_temp = (int) result[k][1][0];
        goal_col_temp = (int) result[k][1][1];

        bool finished = false;
        traj_temp.clear();
        while (!finished){
            l = getVal(i,j-1,hB,num_rows,num_cols);
            r = getVal(i,j+1,hB,num_rows,num_cols);
            u = getVal(i-1,j,hB,num_rows,num_cols);
            d = getVal(i+1,j,hB,num_rows,num_cols);

            // printf("path %llu %llu %llu %llu\n",l,r,u,d);

            unsigned long long int curr_val = getVal(i,j,hB,num_rows,num_cols);
            // printf("path %llu %llu %llu %llu %llu\n",l,r,u,d,curr_val);
            if(curr_val<=l && curr_val<= r && curr_val <=u && curr_val<=d){
                finished=true;
                printf("I have failed my mission!!!\n");
                continue;
            }
            else if(l<r && l<u && l<d){
                i = i;
                j = j-1;
            }
            else if (r < u && r < d){
                i = i;
                j = j+1;
            }
            else if(u<d){
                i = i-1;
                j = j;
            }
            else{
                i = i+1;
                j = j;
            }

            if (i==goal_row_temp and j==goal_col_temp){
                finished=true;
                printf("I have completed my mission!!!\n");
            }
            
            point_temp.clear();
            point_temp.push_back(i);
            point_temp.push_back(j);
            traj_temp.push_back(point_temp);
            
        }
        trajs.push_back(traj_temp);
    }

    string pathfile = results_dir+filename.substr(0,filename.find("."))+"_harmonic_paths.csv";
    
    // Write trajectories to file
    for(int k=0;k<(int)result.size();k++){
        write_trajectory(pathfile, result[k][0], result[k][1], trajs[k]);
    }

    // // Write to output file
    // string fileout = filename.substr(0,filename.find("."))+"_processed.csv";
    // ofstream outfile;
    // outfile.open(fileout);

    // for(unsigned long long int ii=0;ii<num_rows;ii++){
    //     for(unsigned long long int jj=0;jj<num_rows;jj++){
    //         if(jj==num_cols-1){
    //             outfile << to_string(hB[num_rows*ii+jj]) + "\n";
    //         }
    //         else{
    //             // if(hB[num_rows*ii+jj]>0.0){
    //             //     printf("  %u\n",hB[num_rows*ii+jj]);
    //             // }
    //             outfile << to_string(hB[num_rows*ii+jj]) + ",";
    //         }
    //     }
    // }
    // outfile.close();

    printf("CUDA RUN (ms): %f\n",ms/1000);
    
    // clean up memory
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(dA);
    delete[] hA;
    cudaFree(dB);
    delete[] hB;
    cudaFree(dMask);

    return 0;
}