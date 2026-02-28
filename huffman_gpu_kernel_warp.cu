#include <cuda.h>
#include <stdint.h>

// Warp-aggregation histogram kernel. Each thread accumulates a small local
// associative histogram (SMALL_BUCKETS entries) in registers. Warps then
// perform pairwise merges using __shfl_xor_sync so that only the warp leader
// writes the aggregated entries to global memory, reducing global/shared
// atomic contention.

#define SMALL_BUCKETS 16

struct SmallEntry {
    unsigned char sym;
    unsigned int cnt;
};

__device__ inline void small_add_count(SmallEntry *arr, int maxb, unsigned char sym, unsigned int delta) {
    for (int i = 0; i < maxb; ++i) {
        if (arr[i].cnt == 0) {
            arr[i].sym = sym;
            arr[i].cnt = delta;
            return;
        }
        if (arr[i].sym == sym) {
            arr[i].cnt += delta;
            return;
        }
    }
    // Fallback: simple modulo slot to avoid overflow
    int idx = sym & (maxb - 1);
    arr[idx].cnt += delta;
}

extern "C" __global__ void compute_histograms_warp_kernel(const unsigned char* data,
                                                           unsigned long long n,
                                                           unsigned int* out_hist,
                                                           unsigned long long chunk_size) {
    int tid = threadIdx.x;
    int lane = tid & 31;
    // each block processes one chunk
    unsigned long long block_start = (unsigned long long)blockIdx.x * chunk_size;
    unsigned long long block_end = block_start + chunk_size;
    if (block_end > n) block_end = n;

    SmallEntry local[SMALL_BUCKETS];
    for (int i = 0; i < SMALL_BUCKETS; ++i) {
        local[i].sym = 0;
        local[i].cnt = 0u;
    }

    // Each thread processes a strided subset of bytes within the chunk.
    for (unsigned long long pos = block_start + tid; pos < block_end; pos += blockDim.x) {
        unsigned char v = data[pos];
        small_add_count(local, SMALL_BUCKETS, v, 1u);
    }

    // Warp-level pairwise merging using shuffles
    unsigned int mask = 0xffffffffu;
    for (int offset = 1; offset < 32; offset <<= 1) {
        // For each small slot, fetch counterpart's sym/cnt and merge
        for (int s = 0; s < SMALL_BUCKETS; ++s) {
            unsigned int other_cnt = __shfl_xor_sync(mask, local[s].cnt, offset);
            unsigned int other_sym = __shfl_xor_sync(mask, (unsigned int)local[s].sym, offset);
            if (other_cnt == 0) continue;
            unsigned char os = (unsigned char)other_sym;
            // merge other's entry into our local map
            small_add_count(local, SMALL_BUCKETS, os, other_cnt);
        }
    }

    // After reduction, lane 0 of each warp holds aggregated small map for the warp
    int lane_id = tid & 31;

    if (lane_id == 0) {
        unsigned long long out_offset = (unsigned long long)blockIdx.x * 256ull;
        for (int i = 0; i < SMALL_BUCKETS; ++i) {
            unsigned char sym = local[i].sym;
            unsigned int cnt = local[i].cnt;
            if (cnt == 0) continue;
            atomicAdd(&out_hist[out_offset + (unsigned long long)sym], cnt);
        }
    }
}
