// Averaging kernel implementation
#include "average_kernel.h"

#include <cmath>
#include <algorithm>
#include <iostream>
#include <cuda.h>
#include <vector_types.h> // for float3 type (a struct of 3 floats)

// Ideal number of threads per block
static constexpr int threadsperblock = 512;
// Mask for __shfl_down_sync
static constexpr unsigned int all_in_warp = 0xffffffff;

// In each warp reduce three values per thread
__inline__ __device__ float3 warpReduceSum (float valR, float valG, float valB)
{
    for (int offset = warpSize/2; offset > 0; offset >>= 1) {
        valR += __shfl_down_sync(all_in_warp, valR, offset);
        valG += __shfl_down_sync(all_in_warp, valG, offset);
        valB += __shfl_down_sync(all_in_warp, valB, offset);
    }
    return make_float3 (valR, valG, valB);
}

// Run by the 32 threads of a warp
__inline__ __device__ float3 blockReduceSum (float3 val)
{
    static __shared__ float3 shared[32];    // Shared mem for 32 partial sums
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;
    val = warpReduceSum (val.x, val.y, val.z); // Each warp performs partial reduction
    if (lane == 0) { shared[wid] = val; }  // Write reduced value to shared memory
    __syncthreads();                       // Wait for all partial reductions
    // read from shared memory only if that warp existed

    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : make_float3(0.0f, 0.0f, 0.0f);

    if (wid == 0) { val = warpReduceSum (val.x, val.y, val.z); } // Final reduce within first warp
    return val;
}

#if 0
// A kernel function that doesn't do anything
__global__ void reduceit_arrays_dummy (const float3* in, float3* out, int n_arrays, int n_elements)
{
    float3 sum = make_float3(0.0f, 1.0f, 0.0f);
    __syncthreads();
}
#endif

// Input is float3 format.
__global__ void reduceit_arrays (float3* in, float3* out, int n_arrays, int n_elements)
{
    float3 sum = make_float3(0.0f, 0.0f, 0.0f);
    // The y axis of our threads/threadblocks indexes which of the n_arrays this sum relates to
    int omm_id = blockIdx.y * blockDim.y /* + threadIdx.y == 0 */;
    // This gives a memory offset to get to the right part of the input memory
    int mem_offset = omm_id * n_elements;
    // Number of sums is the number of 1D threadblocks that span n_elements. This is gridDim.x.
    int n_sums = gridDim.x;
    // For array index checking
    int data_sz = n_arrays * n_elements;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n_elements && omm_id < n_arrays;
         i += blockDim.x * gridDim.x) {
        int midx = mem_offset + i;
        if (midx < data_sz) {
#if 1 // Causes illegal memory access
            sum.x += in[midx].x;
            sum.y += in[midx].y;
            sum.z += in[midx].z;
#else
            sum.x += 0.1f;
            sum.y += 0.1f;
            sum.z += 0.1f;
#endif
        } else {
            sum.x = -1.0f;
            sum.y = -1.0f;
            sum.z = -1.0f;
        }
    }

    sum = blockReduceSum (sum);
    __syncthreads();

    // This gets the correct output location in out.
    if (threadIdx.x == 0 && omm_id < n_arrays) {
        size_t out_idx = omm_id * n_sums + blockIdx.x;
        if (out_idx < n_arrays * n_sums) {
#if 1 // Causes illegal mem access
            out[omm_id * n_sums + blockIdx.x] = sum;
#endif
        }
    }
}

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert (cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) {
       std::cerr << "CUDA_AVG: " << "GPUassert: " << cudaGetErrorString(code) << " " << file << ":" << line << std::endl;
       if (abort) { exit (code); }
   } // else { std::cout << "CUDA_AVG: " << "GPU Success: code was " << code << " (cudaSuccess)\n"; }
}

// in points to device memory containing n_arrays of n_elements float3s (RGB
// triplets). These are arranged one-after-the other. This host function launches
// kernels to compute the mean of each array and places these means in d_final, which
// should be n_arrays in length.
__host__ void shufflesum_arrays (float3* in, int n_arrays, int n_elements, float3* d_final)
{
    // the constexpr threadsperblock gives an ideal size (512). warp size is 32. I can't
    // mix memory values from different arrays in a threadblock, so the threadblock has
    // to be 1D and thus configurable in size - dynamically sized to match the number of
    // elements in the array (n_elements - i.e. n_samples - may be <512)
    int warps_base = n_elements / 32;
    int warps_extra = n_elements % 32;
    int tbx = (warps_base * 32) + (warps_extra ? 32 : 0);
    tbx = std::min (tbx, threadsperblock);
    dim3 stg1_blockdim(tbx, 1);

    // Then figure out how many threadblocks to run.
    dim3 stg1_griddim(1, 1);
    stg1_griddim.x = n_elements / stg1_blockdim.x + (n_elements % stg1_blockdim.x ? 1 : 0);
    stg1_griddim.y = n_arrays / stg1_blockdim.y + (n_arrays % stg1_blockdim.y ? 1 : 0);


    // d_output is an intermediate piece ofmemory. May (probably) want to allocate externally with the averages memory.
    float3* d_output = nullptr;
    // Malloc n_arrays * n_sums (which is stg1_griddim.x) elements
    gpuErrchk(cudaMalloc (&d_output, n_arrays * stg1_griddim.x * 3 * sizeof(float)));

    std::cout << "CUDA_AVG: About to run with stg1_griddim = (" << stg1_griddim.x << " x " << stg1_griddim.y << " x " << stg1_griddim.z
              << ") and stg1_blockdim = (" << stg1_blockdim.x << " x " << stg1_blockdim.y << " x " << stg1_blockdim.z << ") thread blocks\n";
    reduceit_arrays<<<stg1_griddim, stg1_blockdim>>>(in, d_output, n_arrays, n_elements);

    gpuErrchk(cudaDeviceSynchronize());

    // stg1_griddim.x is 'n_sums'
    warps_base = stg1_griddim.x / 32;
    warps_extra = stg1_griddim.x % 32;
    tbx = (warps_base * 32) + (warps_extra ? 32 : 0);
    tbx = std::min (tbx, threadsperblock);
    dim3 stg2_blockdim(tbx, 1);
    dim3 stg2_griddim(1, 1);
    stg2_griddim.x = stg1_griddim.x / stg1_blockdim.x + (stg1_griddim.x % stg2_blockdim.x ? 1 : 0);
    stg2_griddim.y = n_arrays / stg2_blockdim.y + (n_arrays % stg2_blockdim.y ? 1 : 0);

    std::cout << "CUDA_AVG: About to run with stg2_griddim = (" << stg2_griddim.x << " x " << stg2_griddim.y
              << ") and stg2_blockdim = (" << stg2_blockdim.x << " x " << stg2_blockdim.y << ") thread blocks\n";

    reduceit_arrays<<<stg2_griddim, stg2_blockdim>>>(d_output, d_final, n_arrays, stg1_griddim.x);
    // out_final can be only n_arrays in size

    cudaFree (d_output);
}

__host__ void
average_kernel (float3* d_omm, float3* d_avg, int n_pixels, int n_samples)
{
    std::cout << "CUDA_AVG: " << __func__ << " called for n_pixels = " << n_pixels << " and n_samples = " << n_samples << std::endl;
    if (d_omm == nullptr || d_avg == nullptr) { return; }
    shufflesum_arrays (d_omm, n_pixels, n_samples, d_avg);
}
