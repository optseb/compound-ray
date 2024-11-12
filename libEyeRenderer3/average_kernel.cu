// Averaging kernel implementation
#include "average_kernel.h"

#if 0
// From https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda
#define NUM_BANKS 16
#define LOG_NUM_BANKS 4
#define CONFLICT_FREE_OFFSET(n)((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS))

// From https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda
__global__ void prefix_scan (float *g_odata, float *g_idata, int n)
{
    extern __shared__ float temp[];
    // allocated on invocation int thid = threadIdx.x; int offset = 1;

    int ai = thid;
    int bi = thid + (n / 2);
    int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
    int bankOffsetB = CONFLICT_FREE_OFFSET(bi);
    temp[ai + bankOffsetA] = g_idata[ai];
    temp[bi + bankOffsetB] = g_idata[bi];

    for (int d = n >> 1; d > 0; d >> = 1) {
        // build sum in place up the tree
        __syncthreads();
        if (thid < d) {

            int ai = offset * (2 * thid + 1) - 1;
            int bi = offset * (2 * thid + 2) - 1;
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);

            temp[bi] += temp[ai];
        }
        offset *= 2;
    }

    if (thid == 0) { temp[n - 1 + CONFLICT_FREE_OFFSET(n - 1)] = 0; } // clear the last element

    for (int d = 1; d < n; d *= 2) { // traverse down tree & build scan
        offset >> = 1;
        __syncthreads();
        if (thid < d) {

            int ai = offset * (2 * thid + 1) - 1;
            int bi = offset * (2 * thid + 2) - 1;
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);

            float t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        }
    }
    __syncthreads();

    g_odata[ai] = temp[ai + bankOffsetA];
    g_odata[bi] = temp[bi + bankOffsetB];
}
#endif

#if 0
// C version of my python shifted index fn.
__device__ int shifted_idx (int idx)
{
    int num_banks = 16; // 16 and 768 works for GTX1070/Compute capability 6.1
    int bank_width_int32 = 768;
    // max_idx = (bank_width_int32 * num_banks)
    int idx_idiv_num_banks = idx; // num_banks
    int idx_mod_num_banks = idx % num_banks;
    int offs_idx = ((bank_width_int32 * idx_mod_num_banks) + (idx_idiv_num_banks));

    return offs_idx;
}

// This should work - it comes from my PyCuda implementation in VisualAttention and was fairly easy to convert to C
__global__ void reduceit_from_py (float* scan_ar_, float* nonzero_ar_, float* carry_, int n, int arraysz)
{
    // Access thread/block/grid info like this:
    // int i = threadIdx.x + (blockIdx.x * blockDim.x) + (blockIdx.y * blockDim.x * gridDim.x);

    int thid = threadIdx.x;
    int tb_offset = blockIdx.x * blockDim.x; // threadblock offset
    int d = n / 2; // Integer division

    // This runs for every element in nonzero_ar_
    if ((thid + tb_offset) < (arraysz - d)) {

        extern __shared__ float temp[]; // You have to use an argument in the <<< >>>
                                        // invocation to set this at runtime. See
                                        // https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/. Also,
                                        // it could be statically set at compile time
                                        // (but that won't work here)
        //float* temp = cuda.shared.array(12288, dtype=float32); // Note - allocating ALL shared memory here.

        int ai = thid; // within one block
        int bi = ai + d; // ai and bi are well separated across the shared memory
        int ai_s = shifted_idx (ai);
        int bi_s = shifted_idx (bi);

        // Summing scheme
        temp[ai_s] = nonzero_ar_[ai + tb_offset];
        temp[bi_s] = nonzero_ar_[bi + tb_offset];

        int offset = 1;
        // Upsweep: Ebuild sum in place up the tree
        while (d > 0) {
            __syncthreads();
            if (thid < d) {
                // Block B
                ai = offset * (2 * thid + 1) - 1;
                bi = offset * (2 * thid + 2) - 1;
                ai_s = shifted_idx(ai);
                bi_s = shifted_idx(bi);
                temp[bi_s] += temp[ai_s];

                offset *= 2;
                d >>= 1;
            }
        }
        __syncthreads();

        // Block C: clear the last element - the first step of the downsweep
        if (thid == 0) {
            int nm1s = shifted_idx (n - 1);
            // Carry last number in the block
            carry_[blockIdx.x] = temp[nm1s];
            temp[nm1s] = 0;
        }

        // Downsweep: traverse down tree & build scan
        d = 1;
        while (d < n) {
            offset >>= 1;
            __syncthreads();
            if (thid < d) {
                // Block D
                ai = offset * (2 * thid + 1) - 1;
                bi = offset * (2 * thid + 2) - 1;
                ai_s = shifted_idx(ai);
                bi_s = shifted_idx(bi);
                float t = temp[ai_s];
                temp[ai_s] = temp[bi_s];
                temp[bi_s] += t;
            }
            d *= 2;
        }
        __syncthreads();

        // Block E: write results to device memory
        scan_ar_[ai + tb_offset] = temp[ai_s];
        if (bi < n) { scan_ar_[bi + tb_offset] = temp[bi_s]; }
    }
    __syncthreads();
    // End of reduceit()
}

// Last job is to add on the carry to each part of scan_ar WHILE AT THE SAME TIME SUMMING WITHIN A BLOCK
__global__ void sum_scans (float* new_carry_ar_, float* scan_ar_, int scan_ar_sz, float* carry_ar_)
{
    int thid = threadIdx.x;
    int tb_offset = blockIdx.x * blockDim.x; // threadblock offset
    int arr_addr = thid + tb_offset;
    if (blockIdx.x > 0 && arr_addr < scan_ar_sz) {
        new_carry_ar_[arr_addr] = scan_ar_[arr_addr] + carry_ar_[blockIdx.x];
    } else if (blockIdx.x == 0 && arr_addr < scan_ar_sz) {
        new_carry_ar_[arr_addr] = scan_ar_[arr_addr];
    }
    __syncthreads();
}
#endif

__global__ void do_average (const float3* d_omm, const float3* d_avg)
{
    // whatever d_avg[some_index] += whatever
    // call reduceit_from_py() somehow
}


__host__ void
average_kernel (const float3* d_omm, const float3* d_avg, int n_pixels, int n_samples)
{
    // launch a kernel here as in the example below
    dim3 threadsPerBlock( n_pixels, n_samples );
    int numBlocks = 1;
    do_average <<< numBlocks, threadsPerBlock >>> (d_omm, d_avg);
}
