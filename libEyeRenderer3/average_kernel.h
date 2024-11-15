#pragma once

#include <cuda.h>
#include <vector_types.h> // for float3 type (a struct of 3 floats)

__inline__ __device__ float3 warpReduceSum (float valR, float valG, float valB);
__inline__ __device__ float3 blockReduceSum (float3 val);
__global__ void reduceit_arrays (const float3* in, float3* out, int n_arrays, int n_elements);
__host__ void shufflesum_arrays (const float3* in, int n_arrays, int n_elements, float3* d_final);

/*
 * A CUDA averaging kernel
 */
__host__ void average_kernel (const float3* d_omm, float3* d_avg, int n_pixels, int n_samples_per_pixel);
