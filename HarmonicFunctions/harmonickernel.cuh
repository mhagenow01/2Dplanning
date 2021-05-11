// Mike Hagenow

#ifndef HARM_CUH
#define HARM_CUH

// TODO: fill out
__global__ void harmonic_kernel(unsigned long long int* A, unsigned long long int* B, const unsigned long long int* Mask, size_t num_rows, size_t num_cols, unsigned long long int delta_threshold, int* delta_count);

// TODO: fill out
void harmonic(unsigned long long int** A, unsigned long long int** B, const unsigned long long int* Mask, size_t num_rows, size_t num_cols, unsigned int threads_per_block, unsigned int max_iters);

#endif
