/*
 * An implementation of a summing kernel to add up the samples (which are already
 * divided by n_ommatidial_samples) to get the average.
 *
 * Adapted by Seb James from an example of the summing algorithm at
 * https://developer.nvidia.com/blog/faster-parallel-reductions-kepler/
 *
 * \author Seb James
 * \date November, 2024
 */
#include "summing_kernel.h"

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
    static __shared__ float3 shared[32];       // Shared mem for 32 partial sums
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;
    val = warpReduceSum (val.x, val.y, val.z); // Each warp performs partial reduction
    if (lane == 0) { shared[wid] = val; }      // Write reduced value to shared memory
    __syncthreads();                           // Wait for all partial reductions to complete

    // read from shared memory only if that warp existed
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : make_float3(0.0f, 0.0f, 0.0f);

    // Final reduction within first warp
    if (wid == 0) { val = warpReduceSum (val.x, val.y, val.z); }

    return val;
}

/*
 * Perform a sum of n_arrays, each containing n_elements float3-sized data points,
 * arranged in the device memory pointed to by in.
 *
 * The elements of one array are dispersed across the memory. Element 0 of array 0 is at
 * memory location 0; element 1 of array 0 is at memory location (0 + n_elements) and so
 * on. This results in a need to translate the thread index (called tidx in the code)
 * which has an x component that indexes elements and a y component that indexes arrays
 * into a memory index (midx).
 *
 * The result is written to out, which should be a region of device memory of size
 * gridDim.x * n_arrays * sizeof(float3) (gridDim.x is usually 1).
 */
__global__ void reduceit_arrays (float3* in, float3* out, int n_arrays, int n_elements)
{
    float3 sum = make_float3(0.0f, 0.0f, 0.0f);
    // The y axis of our threads/threadblocks indexes which of the n_arrays this sum relates to
    int omm_id = blockIdx.y * blockDim.y  + threadIdx.y;
    // This gives a memory offset to get to the right part of the input memory
    int thread_offset = omm_id * n_elements;
    // For array index checking
    int data_sz = n_arrays * n_elements;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n_elements && omm_id < n_arrays;
         i += blockDim.x * gridDim.x) { // jumps the whole threadblock as threadblock may not span all data

        int tidx = thread_offset + i;
        // Convert to x/y
        int tx = tidx % n_elements;
        int ty = tidx / n_elements;
        // Convert to real memory index
        int midx = tx * n_arrays + ty;
        if (midx < data_sz) {
            sum.x += in[midx].x;
            sum.y += in[midx].y;
            sum.z += in[midx].z;
        } else {
            sum.x = 0.0f;
            sum.y = 0.0f;
            sum.z = 0.0f;
        }
    }

    sum = blockReduceSum (sum);
    __syncthreads();

    // This gets the correct output location in out. gridDim.x is "n_sums"
    if (threadIdx.x == 0 && omm_id < n_arrays) {
        // Number of sums is the number of 1D threadblocks that span n_elements. This is gridDim.x.
        size_t out_idx = omm_id * gridDim.x + blockIdx.x;
        if (out_idx < n_arrays * gridDim.x) {
            out[omm_id * gridDim.x + blockIdx.x] = sum;
        }
    }
}

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert (cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) {
       std::cerr << "CUDA_SUM: " << "GPUassert: " << cudaGetErrorString(code) << " " << file << ":" << line << std::endl;
       if (abort) { exit (code); }
   } // else { std::cout << "CUDA_SUM: " << "GPU Success: code was " << code << " (cudaSuccess)\n"; }
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
    // We will combine n_sums before warping
    // int n_sums = n_elements / stg1_blockdim.x + (n_elements % stg1_blockdim.x ? 1 : 0);
    //stg1_griddim.x = n_sums; // but leave this at 1
    stg1_griddim.y = n_arrays / stg1_blockdim.y + (n_arrays % stg1_blockdim.y ? 1 : 0);


    // d_output is an intermediate piece of memory. May want to allocate this externally
    // with the reset of the device memory
    float3* d_output = nullptr;
    // Malloc n_arrays * n_sums (which is stg1_griddim.x) elements
    gpuErrchk(cudaMalloc (&d_output, n_arrays * stg1_griddim.x * 3 * sizeof(float)));

#ifdef DEBUG_CUDA_SUM1
    std::cout << "CUDA_SUM: About to run with stg1_griddim = (" << stg1_griddim.x << " x " << stg1_griddim.y
              << ") and stg1_blockdim = (" << stg1_blockdim.x << " x " << stg1_blockdim.y << ") thread blocks\n";
#endif
    reduceit_arrays<<<stg1_griddim, stg1_blockdim>>>(in, d_output, n_arrays, n_elements);

    gpuErrchk(cudaDeviceSynchronize());

    // stg1_griddim.x is 'n_sums'. Second stage is NOT needed in this code!
    warps_base = stg1_griddim.x / 32;
    warps_extra = stg1_griddim.x % 32;
    tbx = (warps_base * 32) + (warps_extra ? 32 : 0);
    tbx = std::min (tbx, threadsperblock);
    dim3 stg2_blockdim(tbx, 1);
    dim3 stg2_griddim(1, 1);
    stg2_griddim.x = stg1_griddim.x / stg1_blockdim.x + (stg1_griddim.x % stg2_blockdim.x ? 1 : 0);
    stg2_griddim.y = n_arrays / stg2_blockdim.y + (n_arrays % stg2_blockdim.y ? 1 : 0);

#ifdef DEBUG_CUDA_SUM2
    std::cout << "CUDA_SUM: About to run with stg2_griddim = (" << stg2_griddim.x << " x " << stg2_griddim.y
              << ") and stg2_blockdim = (" << stg2_blockdim.x << " x " << stg2_blockdim.y << ") thread blocks\n";
#endif

    reduceit_arrays<<<stg2_griddim, stg2_blockdim>>>(d_output, d_final, n_arrays, stg1_griddim.x);
    // out_final can be only n_arrays in size

    cudaFree (d_output);
}

__host__ void
summing_kernel (float3* d_omm, float3* d_avg, int n_pixels, int n_samples)
{
#ifdef DEBUG_CUDA_SUM
    std::cout << "CUDA_SUM: " << __func__ << " called for n_pixels = " << n_pixels
              << " and n_samples = " << n_samples << std::endl;
#endif
    if (d_omm == nullptr || d_avg == nullptr) { return; }
    shufflesum_arrays (d_omm, n_pixels, n_samples, d_avg);
}
