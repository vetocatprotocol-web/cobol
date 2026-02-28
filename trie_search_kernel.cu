// trie_search_kernel.cu
// CUDA Kernel untuk Trie Pattern Search (Layer 6)
// Untuk integrasi dengan Python via PyCUDA/CuPy

#include <cuda_runtime.h>
#include <stdint.h>

#define MAX_PATTERN_LEN 64
#define MAX_TRIE_NODES 4096

// Struktur node Trie di GPU
struct TrieNode {
    int32_t children[256]; // Indeks anak, -1 jika tidak ada
    int32_t pattern_id;    // -1 jika bukan endpoint
};

extern "C" __global__ void trie_search_kernel(
    const uint8_t* data, int data_len,
    const TrieNode* trie, int trie_size,
    int* match_offsets, int* match_ids, int* match_count
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= data_len) return;

    int offset = idx;
    int node = 0;
    for (int i = 0; i < MAX_PATTERN_LEN && offset + i < data_len; ++i) {
        int c = data[offset + i];
        int next = trie[node].children[c];
        if (next == -1) break;
        node = next;
        if (trie[node].pattern_id != -1) {
            int pos = atomicAdd(match_count, 1);
            match_offsets[pos] = offset;
            match_ids[pos] = trie[node].pattern_id;
        }
    }
}
