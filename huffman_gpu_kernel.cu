// huffman_gpu_kernel.cu
// CUDA Kernel: compute per-chunk (sub-block) histograms for Huffman (Layer 7)
// This file provides a performant histogram kernel which is used as the
// first step in a parallel canonical-Huffman pipeline (sub-chunking).

#include <cuda.h>
#include <stdint.h>

extern "C" __global__ void compute_histograms_kernel(const unsigned char* data,
                                                      unsigned long long n,
                                                      unsigned int* out_hist,
                                                      unsigned long long chunk_size) {
    // shared histogram per block (one block == one chunk)
    extern __shared__ unsigned int s_hist[]; // size must be 256 * sizeof(unsigned int)

    int tid = threadIdx.x;
    // initialize shared histogram
    for (int i = tid; i < 256; i += blockDim.x) {
        s_hist[i] = 0u;
    }
    __syncthreads();

    unsigned long long block_start = (unsigned long long)blockIdx.x * chunk_size;
    unsigned long long block_end = block_start + chunk_size;
    if (block_end > n) block_end = n;

    // Each thread processes a strided subset of bytes within the chunk.
    for (unsigned long long pos = block_start + tid; pos < block_end; pos += blockDim.x) {
        unsigned char v = data[pos];
        atomicAdd(&s_hist[(int)v], 1u);
    }
    __syncthreads();

    // Write shared histogram to global memory
    unsigned long long out_offset = (unsigned long long)blockIdx.x * 256ull;
    for (int i = tid; i < 256; i += blockDim.x) {
        out_hist[out_offset + (unsigned long long)i] = s_hist[i];
    }
}

