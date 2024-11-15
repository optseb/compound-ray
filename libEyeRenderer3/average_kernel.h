#pragma once

#include <cuda.h>
#include <vector_types.h> // for float3 type (a struct of 3 floats)

/*
 * A CUDA averaging kernel
 */
__host__ void average_kernel (float3* d_omm, float3* d_avg, int n_pixels, int n_samples_per_pixel);
