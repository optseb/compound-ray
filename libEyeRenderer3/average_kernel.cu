// Averaging kernel implementation
#include "average_kernel.h"

__global__ void do_average (const float3* d_omm, const float3* d_avg)
{
    // Find the average from d_omm
}

__host__ void
average_kernel (const float3* d_omm, const float3* d_avg, int n_pixels, int n_samples)
{
    // launch a kernel here as in the example below
    dim3 threadsPerBlock(n_pixels, n_samples);
    int numBlocks = 1;
    // do_average <<< numBlocks, threadsPerBlock >>> (d_omm, d_avg);
}
