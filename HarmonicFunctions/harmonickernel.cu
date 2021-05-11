// Mike Hagenow
// ME759 Final Project - Kernel to iteratively solve Laplacian

#include "harmonickernel.cuh"
#include <stdio.h>

// Dirichlet CUDA kernel for the harmonic potential fields using Shared Memory and Increased Arithmetic Density
// The approach is similar to the shared memory, but with multiple loops through the shared memory
// to promote faster diffusion! It also uses a harmonic mean. Unfortunately, none of these things actually worked.
__global__ void harmonic_kernel_tiled_2(unsigned long long int* A, unsigned long long int* B, const unsigned long long int* Mask, size_t num_rows, size_t num_cols, unsigned long long int delta_threshold, int* delta_count){
    extern __shared__ unsigned long long int A_tile_sh[];
    int BLOCK_SIZE = 32;
    int buffer_size = (BLOCK_SIZE+2)*(BLOCK_SIZE+2);

    /////////////////////////////
    //   Loading the tile      //
    /////////////////////////////
    // Each thread will load an element of the tile and TODO (max 1) an element from the edges
    
    int ty = threadIdx.y; //the row index in the sub-block
    int tx = threadIdx.x; //the column index in the sub-block
    
    int whichRow = blockIdx.y*BLOCK_SIZE+ty;
    int whichCol = blockIdx.x*BLOCK_SIZE+tx;

    // Log the 32x32 grid of interest
    if(whichCol<num_cols && whichRow<num_rows)
    {
        // Logging is done with an offset from (1,1) since the first row and column are the extra padding for calculations
        A_tile_sh[(ty+1)*(BLOCK_SIZE+2)+(tx+1)] = A[whichRow*num_cols+whichCol];
    }
    else{
        A_tile_sh[(ty+1)*(BLOCK_SIZE+2)+(tx+1)] = (ULLONG_MAX/4); //treat as collision if outside bounds of array
    }

    // Load the edges of the shared memory array!!
    int threadId = ty*BLOCK_SIZE+tx;
    if(threadId<(BLOCK_SIZE+2)){

        // for each check that row and column are ok (including both ends for the threadID sweep direction)

        // Row on top of tile
        if((blockIdx.y*BLOCK_SIZE)>=1 && (blockIdx.x*BLOCK_SIZE-1+threadId)<num_cols && (blockIdx.x*BLOCK_SIZE+threadId)>=1){
            // [0][threadId]
            A_tile_sh[0*(BLOCK_SIZE+2)+threadId] = A[(blockIdx.y*BLOCK_SIZE-1)*num_cols+blockIdx.x*BLOCK_SIZE-1+threadId];
        }
        else{
            // A_tile_sh[0*(BLOCK_SIZE+2)+threadId] = (ULLONG_MAX/4); //treat as collision if outside bounds
            A_tile_sh[0*(BLOCK_SIZE+2)+threadId] = (ULLONG_MAX/4); //treat as collision if outside bounds
        }

        // Row below tile
        if(((blockIdx.y+1)*BLOCK_SIZE)<num_rows && (blockIdx.x*BLOCK_SIZE-1+threadId)<num_cols && (blockIdx.x*BLOCK_SIZE+threadId)>=1){
            // A_tile_sh[BLOCK_SIZE+2][threadId] = A[(blockIdx.y+1)*BLOCK_SIZE+2)*num_cols+threadId];
            A_tile_sh[(BLOCK_SIZE+1)*(BLOCK_SIZE+2)+threadId] = A[(blockIdx.y+1)*BLOCK_SIZE*num_cols+blockIdx.x*BLOCK_SIZE-1+threadId];
        }
        else{
            // A_tile_sh[(BLOCK_SIZE+1)*(BLOCK_SIZE+2)+threadId] = (ULLONG_MAX/4); //treat as collision if outside bounds
            A_tile_sh[0*(BLOCK_SIZE+2)+threadId] = (ULLONG_MAX/4); //treat as collision if outside bounds
        }

        // Col to left of tile
        // check col and row
        if((blockIdx.x*BLOCK_SIZE)>=1 && (blockIdx.y*BLOCK_SIZE-1+threadId)<num_rows && (blockIdx.y*BLOCK_SIZE+threadId)>=1){
            // A_tile_sh[threadId][0] = A[(blockIdx.y+1)*BLOCK_SIZE+2)*num_cols+threadId];
            A_tile_sh[threadId*(BLOCK_SIZE+2)+0] = A[(blockIdx.y*BLOCK_SIZE-1+threadId)*num_cols+blockIdx.x*BLOCK_SIZE-1];
        }
        else{
            // A_tile_sh[threadId*(BLOCK_SIZE+2)+0] = (ULLONG_MAX/4); //treat as collision if outside bounds
            A_tile_sh[0*(BLOCK_SIZE+2)+threadId] = (ULLONG_MAX/4); //treat as collision if outside bounds

        }

        // Col to right of tile
        if((blockIdx.x+1)*BLOCK_SIZE<num_cols && (blockIdx.y*BLOCK_SIZE-1+threadId)<num_rows && (blockIdx.y*BLOCK_SIZE+threadId)>=1){
            // A_tile_sh[threadId][BLOCK_SIZE+2] = A[(blockIdx.y+1)*BLOCK_SIZE+2)*num_cols+threadId];
            A_tile_sh[threadId*(BLOCK_SIZE+2)+(BLOCK_SIZE+1)] = A[(blockIdx.y*BLOCK_SIZE-1+threadId)*num_cols+(blockIdx.x+1)*BLOCK_SIZE];
        }
        else{
            // A_tile_sh[threadId*(BLOCK_SIZE+2)+(BLOCK_SIZE+1)] = (ULLONG_MAX/4); //treat as collision if outside bounds
            A_tile_sh[0*(BLOCK_SIZE+2)+threadId] = (ULLONG_MAX/4); //treat as collision if outside bounds
        }
    }

    __syncthreads();

    //////////////////////////////////////////////
    // Calculate the new tile and store in B    //
    //////////////////////////////////////////////
    if(whichRow<num_rows && whichCol<num_cols){
        int num_subloops = 1;
        int pout = 0, pin = 1;

        for(int i=0;i<num_subloops;i++){
            pout = 1 - pout; // swap double buffer indices
            pin  = 1 - pout;

            // One store into the global memory (or enforce boundary condition)
            if(Mask[whichRow*num_cols+whichCol]==(ULLONG_MAX/4)/2){
                // shared memory already accounts for bounds so no checks are necessary
                // left, right, up, down
                float new_val = 0;
                new_val+=1.0/float(A_tile_sh[pin*buffer_size+(ty+1)*(BLOCK_SIZE+2)+(tx+1)-1]);
                new_val+=1.0/float(A_tile_sh[pin*buffer_size+(ty+1)*(BLOCK_SIZE+2)+(tx+1)+1]);
                new_val+=1.0/float(A_tile_sh[pin*buffer_size+(ty+1-1)*(BLOCK_SIZE+2)+(tx+1)]);
                new_val+=1.0/float(A_tile_sh[pin*buffer_size+(ty+1+1)*(BLOCK_SIZE+2)+(tx+1)]);
                unsigned long long int temp = (long long int)(4.0/new_val);
                A_tile_sh[pout*buffer_size+(ty+1)*(BLOCK_SIZE+2)+(tx+1)]=temp;
                // B[whichRow*num_cols+whichCol] = temp;
                if(i==0 && (unsigned long long int)abs((long long int)(temp)-(long long int)A_tile_sh[pin*buffer_size+(ty+1)*(BLOCK_SIZE+2)+(tx+1)])>delta_threshold){
                    atomicAdd(delta_count,1);
                }
            }
            else{
                A_tile_sh[pout*buffer_size+(ty+1)*(BLOCK_SIZE+2)+(tx+1)]=Mask[whichRow*num_cols+whichCol];
                // B[whichRow*num_cols+whichCol] = Mask[whichRow*num_cols+whichCol];
            }

            __syncthreads();
        }

        // Store the final result back in global memory
        B[whichRow*num_cols+whichCol]=A_tile_sh[pout*buffer_size+(ty+1)*(BLOCK_SIZE+2)+(tx+1)];
    }
}

// Dirichlet CUDA kernel for the harmonic potential fields using Shared Memory
// The approach is similar to matrix multiplication except that there are padding requirements and each block
// in the final matrix only requires the equivalent tile in the original matrix.
__global__ void harmonic_kernel_tiled(unsigned long long int* A, unsigned long long int* B, const unsigned long long int* Mask, size_t num_rows, size_t num_cols, unsigned long long int delta_threshold, int* delta_count){
    extern __shared__ unsigned long long int A_tile_sh[];
    int BLOCK_SIZE = 32;

    /////////////////////////////
    //   Loading the tile      //
    /////////////////////////////
    // Each thread will load an element of the tile and TODO (max 1) an element from the edges
    
    int ty = threadIdx.y; //the row index in the sub-block
    int tx = threadIdx.x; //the column index in the sub-block
    
    int whichRow = blockIdx.y*BLOCK_SIZE+ty;
    int whichCol = blockIdx.x*BLOCK_SIZE+tx;

    // Log the 32x32 grid of interest
    if(whichCol<num_cols && whichRow<num_rows)
    {
        // Logging is done with an offset from (1,1) since the first row and column are the extra padding for calculations
        A_tile_sh[(ty+1)*(BLOCK_SIZE+2)+(tx+1)] = A[whichRow*num_cols+whichCol];
    }
    else{
        A_tile_sh[(ty+1)*(BLOCK_SIZE+2)+(tx+1)] = (ULLONG_MAX/4); //treat as collision if outside bounds of array
    }

    // Load the edges of the shared memory array!!
    int threadId = ty*BLOCK_SIZE+tx;
    if(threadId<(BLOCK_SIZE+2)){

        // for each check that row and column are ok (including both ends for the threadID sweep direction)

        // Row on top of tile
        if((blockIdx.y*BLOCK_SIZE)>=1 && (blockIdx.x*BLOCK_SIZE-1+threadId)<num_cols && (blockIdx.x*BLOCK_SIZE+threadId)>=1){
            // [0][threadId]
            A_tile_sh[0*(BLOCK_SIZE+2)+threadId] = A[(blockIdx.y*BLOCK_SIZE-1)*num_cols+blockIdx.x*BLOCK_SIZE-1+threadId];
        }
        else{
            // A_tile_sh[0*(BLOCK_SIZE+2)+threadId] = (ULLONG_MAX/4); //treat as collision if outside bounds
            A_tile_sh[0*(BLOCK_SIZE+2)+threadId] = (ULLONG_MAX/4); //treat as collision if outside bounds
        }

        // Row below tile
        if(((blockIdx.y+1)*BLOCK_SIZE)<num_rows && (blockIdx.x*BLOCK_SIZE-1+threadId)<num_cols && (blockIdx.x*BLOCK_SIZE+threadId)>=1){
            // A_tile_sh[BLOCK_SIZE+2][threadId] = A[(blockIdx.y+1)*BLOCK_SIZE+2)*num_cols+threadId];
            A_tile_sh[(BLOCK_SIZE+1)*(BLOCK_SIZE+2)+threadId] = A[(blockIdx.y+1)*BLOCK_SIZE*num_cols+blockIdx.x*BLOCK_SIZE-1+threadId];
        }
        else{
            // A_tile_sh[(BLOCK_SIZE+1)*(BLOCK_SIZE+2)+threadId] = (ULLONG_MAX/4); //treat as collision if outside bounds
            A_tile_sh[0*(BLOCK_SIZE+2)+threadId] = (ULLONG_MAX/4); //treat as collision if outside bounds
        }

        // Col to left of tile
        // check col and row
        if((blockIdx.x*BLOCK_SIZE)>=1 && (blockIdx.y*BLOCK_SIZE-1+threadId)<num_rows && (blockIdx.y*BLOCK_SIZE+threadId)>=1){
            // A_tile_sh[threadId][0] = A[(blockIdx.y+1)*BLOCK_SIZE+2)*num_cols+threadId];
            A_tile_sh[threadId*(BLOCK_SIZE+2)+0] = A[(blockIdx.y*BLOCK_SIZE-1+threadId)*num_cols+blockIdx.x*BLOCK_SIZE-1];
        }
        else{
            // A_tile_sh[threadId*(BLOCK_SIZE+2)+0] = (ULLONG_MAX/4); //treat as collision if outside bounds
            A_tile_sh[0*(BLOCK_SIZE+2)+threadId] = (ULLONG_MAX/4); //treat as collision if outside bounds

        }

        // Col to right of tile
        if((blockIdx.x+1)*BLOCK_SIZE<num_cols && (blockIdx.y*BLOCK_SIZE-1+threadId)<num_rows && (blockIdx.y*BLOCK_SIZE+threadId)>=1){
            // A_tile_sh[threadId][BLOCK_SIZE+2] = A[(blockIdx.y+1)*BLOCK_SIZE+2)*num_cols+threadId];
            A_tile_sh[threadId*(BLOCK_SIZE+2)+(BLOCK_SIZE+1)] = A[(blockIdx.y*BLOCK_SIZE-1+threadId)*num_cols+(blockIdx.x+1)*BLOCK_SIZE];
        }
        else{
            // A_tile_sh[threadId*(BLOCK_SIZE+2)+(BLOCK_SIZE+1)] = (ULLONG_MAX/4); //treat as collision if outside bounds
            A_tile_sh[0*(BLOCK_SIZE+2)+threadId] = (ULLONG_MAX/4); //treat as collision if outside bounds
        }
    }

    __syncthreads();

    //////////////////////////////////////////////
    // Calculate the new tile and store in B    //
    //////////////////////////////////////////////
    if(whichRow<num_rows && whichCol<num_cols){
        unsigned long long int new_val = 0;

        // shared memory already accounts for bounds so no checks are necessary
        // left, right, up, down
        new_val+=A_tile_sh[(ty+1)*(BLOCK_SIZE+2)+(tx+1)-1];
        new_val+=A_tile_sh[(ty+1)*(BLOCK_SIZE+2)+(tx+1)+1];
        new_val+=A_tile_sh[(ty+1-1)*(BLOCK_SIZE+2)+(tx+1)];
        new_val+=A_tile_sh[(ty+1+1)*(BLOCK_SIZE+2)+(tx+1)];
        
        // One store into the global memory (or enforce boundary condition)
        if(Mask[whichRow*num_cols+whichCol]==(ULLONG_MAX/4)/2){
            unsigned long long int temp = (new_val/4);
            // unsigned long long int temp = (new_val/4)/100*85+(A_tile_sh[(ty+1)*(BLOCK_SIZE+2)+(tx+1)])/100*15;
            B[whichRow*num_cols+whichCol] = temp;
            if((unsigned long long int)abs((long long int)(temp)-(long long int)A_tile_sh[(ty+1)*(BLOCK_SIZE+2)+(tx+1)])>delta_threshold){
                atomicAdd(delta_count,1);
                // (*delta_count)+=1;
            }
        }
        else{
            B[whichRow*num_cols+whichCol] = Mask[whichRow*num_cols+whichCol];
        }

    }
}

// Neumann CUDA kernel for the harmonic potential fields
// Also uses global memory access and enforces Neumann boundary via three-point derivative (i.e., sets current point to make derivative 0)
// To enforce 2d laplacian, the values of the current point to create a zero-partial in each direction are averaged.
__global__ void global_neumann_harmonic_kernel(unsigned long long int* A, unsigned long long int* B, const unsigned long long int* Mask, size_t num_rows, size_t num_cols, unsigned long long int delta_threshold, int* delta_count){
    int index = threadIdx.x+ blockIdx.x* blockDim.x;
    
    int whichRow = index/num_cols;
    int whichCol = index%num_cols;

    if(index<num_rows*num_cols){
        unsigned long long int new_val = 0;

        // 4 Checks - Left, right, up, and down (each has different issues)
        // If not passed, don't add anything
        
        // Left (need a column to left)
        if(whichCol>=1){
            new_val+=A[whichRow*num_cols+whichCol-1];
        }
        // Right (need a column to right)
        if((whichCol+1)<num_cols){
            new_val+=A[whichRow*num_cols+whichCol+1];
        }
        // Up (need a previous row)
        if(whichRow>=1){
            new_val+=A[(whichRow-1)*num_cols+whichCol];
        }
        // Down (need a row after)
        if((whichRow+1)<num_rows){
            new_val+=A[(whichRow+1)*num_cols+whichCol];
        }

        // printf("  ** %u %u\n",new_val,new_val/4);
        // One store into the global memory (or enforce boundary condition)
        if(Mask[index]==(ULLONG_MAX/4)/2){
            B[index] = new_val/4;
            // printf("  %llu \n",(unsigned long long int)abs((long long int)(new_val/4)-(long long int)A[index]));
            if((unsigned long long int)abs((long long int)(new_val/4)-(long long int)A[index])>delta_threshold){
                atomicAdd(delta_count,1);
                // (*delta_count)+=1;
            }
        }
        else{ // For the neumann, we will update left, right, above, below (i.e., central difference for velocity)
            if(Mask[index]==0)
            {
                B[index]=0;
            }

            else{
                unsigned long long int left = A[index];
                unsigned long long int right = left;
                unsigned long long int up = left;
                unsigned long long int down = left;

                
                if(whichCol>=1){
                    left=A[whichRow*num_cols+whichCol-1];
                }
                // Right (need a column to right)
                if((whichCol+1)<num_cols){
                    right=A[whichRow*num_cols+whichCol+1];
                }
                // Up (need a previous row)
                if(whichRow>=1){
                    up=A[(whichRow-1)*num_cols+whichCol];
                }
                // Down (need a row after)
                if((whichRow+1)<num_rows){
                    down=A[(whichRow+1)*num_cols+whichCol];
                }
            
                B[index] = (3*left+right)/8+(3*up+down)/8;
            }
        }
    }
}

// Baseline CUDA kernel for the harmonic potential fields
// Uses global memory access and Dirichlet boundary conditions (i.e., all obstacles are repulsive fields)
__global__ void global_harmonic_kernel(unsigned long long int* A, unsigned long long int* B, const unsigned long long int* Mask, size_t num_rows, size_t num_cols, unsigned long long int delta_threshold, int* delta_count){
    int index = threadIdx.x+ blockIdx.x* blockDim.x;
    
    int whichRow = index/num_cols;
    int whichCol = index%num_cols;

    if(index<num_rows*num_cols){
        unsigned long long int new_val = 0;

        // 4 Checks - Left, right, up, and down (each has different issues)
        // If not passed, don't add anything
        
        // Left (need a column to left)
        if(whichCol>=1){
            new_val+=A[whichRow*num_cols+whichCol-1];
        }
        // Right (need a column to right)
        if((whichCol+1)<num_cols){
            new_val+=A[whichRow*num_cols+whichCol+1];
        }
        // Up (need a previous row)
        if(whichRow>=1){
            new_val+=A[(whichRow-1)*num_cols+whichCol];
        }
        // Down (need a row after)
        if((whichRow+1)<num_rows){
            new_val+=A[(whichRow+1)*num_cols+whichCol];
        }

        // printf("  ** %u %u\n",new_val,new_val/4);
        // One store into the global memory (or enforce boundary condition)
        if(Mask[index]==(ULLONG_MAX/4)/2){
            B[index] = new_val/4;
            // printf("  %llu \n",(unsigned long long int)abs((long long int)(new_val/4)-(long long int)A[index]));
            if((unsigned long long int)abs((long long int)(new_val/4)-(long long int)A[index])>delta_threshold){
                atomicAdd(delta_count,1);
                // (*delta_count)+=1;
            }
        }
        else{
            B[index] = Mask[index];
        }
    }
}

// Harmonic function is responsible for launching the successive kernel
// calls and setting up appropriate dimensions for each kernel (e.g., 2d vs 1d)
void harmonic(unsigned long long int** A, unsigned long long int** B, const unsigned long long int* Mask, size_t num_rows, size_t num_cols, unsigned int threads_per_block, unsigned int max_iters){
    
    // Determine appropriate number of blocks (n^2 operations for row-col combinations)
    // code asks for 1D kernel configuration
    const unsigned int blocks_per_grid = ((num_rows*num_cols)+threads_per_block-1)/threads_per_block;

    // For the tiling, need to compute a tile and the appopriate grid of tiles
    unsigned int block_dim = 32;
    unsigned int grid_dim = ((num_rows) + block_dim - 1) / block_dim; // Assume rows and columns are the same

    // Printing iterations
    unsigned iter_print = 1;
    if(max_iters>20){
        iter_print = max_iters/20;
    }

    unsigned long long int* temp_switch;
    
    int *d_delta_count;
    int delta_count;
    cudaMalloc((void **)&d_delta_count, sizeof(int));

    // unsigned long long int delta_threshold=ULLONG_MAX/4*0.00000000000001;
    unsigned long long int delta_threshold=0;
    printf("Delta Threshold: %llu\n",delta_threshold);

    // TODO: multiple runs within a kernel call to try and improve arithmetic intensity?
    unsigned int i=0;
    bool converged = false;

    while(i<max_iters && !converged) {
        cudaMemset(d_delta_count, 0, sizeof(int));

        /////////////////////
        // Kernel Options //
        ////////////////////

        // global_harmonic_kernel<<<blocks_per_grid, threads_per_block>>>(*A,*B,Mask,num_rows,num_cols,delta_threshold,d_delta_count);
        harmonic_kernel_tiled<<<dim3(grid_dim,grid_dim),dim3(block_dim,block_dim),(block_dim+2)*(block_dim+2)*sizeof(unsigned long long int)>>>(*A,*B,Mask,num_rows,num_cols,delta_threshold,d_delta_count);
        // harmonic_kernel_tiled_2<<<dim3(grid_dim,grid_dim),dim3(block_dim,block_dim),2*(block_dim+2)*(block_dim+2)*sizeof(unsigned long long int)>>>(*A,*B,Mask,num_rows,num_cols,delta_threshold,d_delta_count);
        // global_neumann_harmonic_kernel<<<blocks_per_grid, threads_per_block>>>(*A,*B,Mask,num_rows,num_cols,delta_threshold,d_delta_count);
        cudaDeviceSynchronize();
        cudaMemcpy(&delta_count, d_delta_count, sizeof(int), cudaMemcpyDeviceToHost);
        if(i%(iter_print)==0){
            printf("Run %d of Max %d\n",i,max_iters);
            printf("  DC: %d\n",delta_count);
        }

        i++;
        
        // if(delta_count<(num_rows*num_cols*0.00001)){
        if(delta_count==0){
            converged=true;
            printf("Iterations to finish: %u\n",i);
        }
        //switch A and B for next iteration: todo if another run -- otherwise, B has result
        else if(i<(max_iters-1)){
            temp_switch = *A;
            *A = *B;
            *B = temp_switch;
        }
    }

    cudaDeviceSynchronize();
    cudaFree(d_delta_count);
    // printf("Cuda Error Check: %s\n",cudaGetErrorString(cudaGetLastError()));
    
}
