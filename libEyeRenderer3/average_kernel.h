#pragma once

#include <cuda.h>

/*
 * A CUDA averaging kernel
 */
__host__ void average_kernel (const float3* d_omm, const float3* d_avg, int n_pixels, int n_samples);
