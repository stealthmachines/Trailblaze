/*
 * layer4/tb_gguf_dequant.cu — CUDA fused dequant+matvec kernels
 *
 * Target:  RTX 2060 (sm_75), CUDA 13.2
 * Build:   nvcc -O3 -arch=sm_75 -c layer4/tb_gguf_dequant.cu -o obj/tb_gguf_dequant.obj
 *
 * Architecture:
 *   - Weights are uploaded to device VRAM once at model-load time via
 *     tb_cuda_upload_tensor().  Pointer is stored in TB_GGUFTensorInfo.d_data.
 *   - tb_cuda_matvec_device() dispatches the correct kernel using the
 *     device-resident W — NO PCIe transfer for weights during inference.
 *   - The hidden-state vector x is tiny (~16 KB for hidden=4096) and is
 *     uploaded to a persistent device buffer once per matvec call.
 *   - With RTX 2060 12 GB and a 8.2 GB Q6_K_XL model, the entire model fits
 *     in VRAM: zero PCIe pressure per token.
 *
 * FMA trick (from flash-moe-HDGL shaders.metal):
 *   Instead of: out += (scale * q - min) * x[k]
 *   Use:        out += __fmaf_rn(scale * q, x[k], -(min * x[k]))
 *   GPU FMA unit fuses dequant+multiply in one instruction: +12% vs naive.
 *
 * Kernel design:
 *   grid = (M,)   — one block per output element (row)
 *   block = 128   — 4 warps; good register pressure for sm_75
 *   Threads within a block stripe over the K-dimension superblocks.
 *   Partial sums reduce via warp shuffle + shared-memory inter-warp reduce.
 */

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Constants matching GGUF block layouts ──────────────────────────────── */
#define QK_K        256     /* weights per K-quant superblock               */
#define QK4_0        32     /* weights per Q4_0 block                       */
#define QK8_0        32     /* weights per Q8_0 block                       */
#define Q4K_BLOCK   144     /* bytes per Q4_K superblock                    */
#define Q6K_BLOCK   210     /* bytes per Q6_K superblock                    */
#define Q4_0_BLOCK   18     /* bytes per Q4_0 block                         */
#define Q8_0_BLOCK   34     /* bytes per Q8_0 block                         */
#define THREADS     128     /* threads per block (4 warps on sm_75)         */
#define MAX_WARPS     4     /* THREADS / 32                                 */

/* ── Device helpers ─────────────────────────────────────────────────────── */
__device__ __forceinline__ float d_f16_to_f32(uint16_t h)
{
    uint32_t sign     = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exponent = (h & 0x7C00u) >> 10;
    uint32_t mantissa =  h & 0x03FFu;
    uint32_t bits;
    if      (exponent == 0)   bits = sign | (mantissa << 13);
    else if (exponent == 31)  bits = sign | 0x7F800000u | (mantissa << 13);
    else                      bits = sign | ((exponent + 112u) << 23) | (mantissa << 13);
    return __uint_as_float(bits);
}

/* ── Block-level warp-shuffle reduce ────────────────────────────────────── */
__device__ __forceinline__ float block_reduce_sum(float val, float *warp_sums)
{
    /* Warp reduce */
    for (int off = warpSize / 2; off > 0; off >>= 1)
        val += __shfl_down_sync(0xffffffff, val, off);

    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    if (lane == 0) warp_sums[warp] = val;
    __syncthreads();

    /* Inter-warp reduce in thread 0 */
    float total = 0.0f;
    if (threadIdx.x == 0) {
        int nw = (blockDim.x + 31) >> 5;
        for (int w = 0; w < nw; w++) total += warp_sums[w];
    }
    return total;
}

/* ════════════════════════════════════════════════════════════════════════════
 * Q4_K fused dequant + matvec
 * Superblock: 144 bytes, 256 weights
 *   [d:f16 2][dmin:f16 2][scales:12][qs:128]
 *   weight[i] = scale[i/32] * nibble[i] - min[i/32]
 * ════════════════════════════════════════════════════════════════════════════ */
__global__ void k_q4k_matvec(
    const uint8_t * __restrict__ W,    /* device pointer to Q4_K weights   */
    int M, int K,
    const float   * __restrict__ x,    /* hidden vector (K floats)          */
    float         * __restrict__ out)  /* output vector (M floats)          */
{
    int row = blockIdx.x;
    if (row >= M) return;

    int num_sb       = K / QK_K;
    size_t row_bytes = (size_t)num_sb * Q4K_BLOCK;

    __shared__ float warp_sums[MAX_WARPS];
    float acc = 0.0f;

    for (int sbi = threadIdx.x; sbi < num_sb; sbi += blockDim.x) {
        const uint8_t *sb = W + (size_t)row * row_bytes + (size_t)sbi * Q4K_BLOCK;
        float d    = d_f16_to_f32(*(const uint16_t *)(sb + 0));
        float dmin = d_f16_to_f32(*(const uint16_t *)(sb + 2));
        const uint8_t *sc = sb + 4;    /* scales[12] — 6-bit packed */
        const uint8_t *qs = sb + 16;   /* qs[128]    — nibble packed */

        /* Unpack 8 scale+min pairs from 6-bit encoding */
        float scales[8], mins[8];
        for (int j = 0; j < 4; j++) {
            scales[j]   = d    * (float)(sc[j]   & 0x3F);
            mins[j]     = dmin * (float)(sc[j+4] & 0x3F);
            scales[j+4] = d    * (float)((sc[j] >> 6) | ((sc[j+8]  & 0x0F) << 2));
            mins[j+4]   = dmin * (float)((sc[j+4] >> 6) | ((sc[j+12] & 0x0F) << 2));
        }

        /* 128 packed bytes → 256 nibbles; fused dequant+dot via FMA */
        int base_k = sbi * QK_K;
        for (int i = 0; i < 128; i++) {
            uint8_t byte = qs[i];
            int g0 = (2 * i)     >> 5;   /* sub-block index */
            int g1 = (2 * i + 1) >> 5;
            float lo = (float)(byte & 0xFu);
            float hi = (float)(byte >>  4);
            float x0 = x[base_k + 2*i    ];
            float x1 = x[base_k + 2*i + 1];
            /* FMA: scale*q*x - min*x = fmaf(scale*q, x, -min*x) */
            acc += __fmaf_rn(scales[g0] * lo, x0, -mins[g0] * x0);
            acc += __fmaf_rn(scales[g1] * hi, x1, -mins[g1] * x1);
        }
    }

    float total = block_reduce_sum(acc, warp_sums);
    if (threadIdx.x == 0) out[row] = total;
}

/* ════════════════════════════════════════════════════════════════════════════
 * Q6_K fused dequant + matvec
 * Superblock: 210 bytes, 256 weights
 *   [ql:128][qh:64][scales:16 int8][d:f16 2]
 *   q6 = (ql_nibble | (qh_2bit << 4)) - 32
 *   weight[i] = d * scales[i/16] * q6[i]
 * ════════════════════════════════════════════════════════════════════════════ */
__global__ void k_q6k_matvec(
    const uint8_t * __restrict__ W,
    int M, int K,
    const float   * __restrict__ x,
    float         * __restrict__ out)
{
    int row = blockIdx.x;
    if (row >= M) return;

    int num_sb       = K / QK_K;
    size_t row_bytes = (size_t)num_sb * Q6K_BLOCK;

    __shared__ float warp_sums[MAX_WARPS];
    float acc = 0.0f;

    for (int sbi = threadIdx.x; sbi < num_sb; sbi += blockDim.x) {
        const uint8_t *sb   = W + (size_t)row * row_bytes + (size_t)sbi * Q6K_BLOCK;
        const uint8_t *ql   = sb;
        const uint8_t *qh   = sb + 128;
        const int8_t  *sc   = (const int8_t *)(sb + 192);
        float d = d_f16_to_f32(*(const uint16_t *)(sb + 208));

        int base_k = sbi * QK_K;
        for (int i = 0; i < QK_K; i++) {
            int lo = (ql[i >> 1] >> (4 * (i & 1))) & 0xF;
            int hi = (qh[i >> 2] >> (2 * (i & 3))) & 0x3;
            int q  = (lo | (hi << 4)) - 32;
            float wf = d * (float)sc[i >> 4] * (float)q;
            acc += wf * x[base_k + i];
        }
    }

    float total = block_reduce_sum(acc, warp_sums);
    if (threadIdx.x == 0) out[row] = total;
}

/* ════════════════════════════════════════════════════════════════════════════
 * Q4_0 fused dequant + matvec
 * Block: 18 bytes, 32 weights: [d:f16 2][qs:16]
 *   weight[i] = d * (nibble[i] - 8)
 * ════════════════════════════════════════════════════════════════════════════ */
__global__ void k_q4_0_matvec(
    const uint8_t * __restrict__ W,
    int M, int K,
    const float   * __restrict__ x,
    float         * __restrict__ out)
{
    int row = blockIdx.x;
    if (row >= M) return;

    int num_blocks   = K / QK4_0;
    size_t row_bytes = (size_t)num_blocks * Q4_0_BLOCK;

    __shared__ float warp_sums[MAX_WARPS];
    float acc = 0.0f;

    for (int bi = threadIdx.x; bi < num_blocks; bi += blockDim.x) {
        const uint8_t *blk = W + (size_t)row * row_bytes + (size_t)bi * Q4_0_BLOCK;
        float d = d_f16_to_f32(*(const uint16_t *)blk);
        const uint8_t *qs = blk + 2;
        int base_k = bi * QK4_0;
        for (int i = 0; i < 16; i++) {
            uint8_t b = qs[i];
            float lo = (float)((int)(b & 0xF) - 8);
            float hi = (float)((int)(b >>  4) - 8);
            acc += __fmaf_rn(d, lo * x[base_k + 2*i    ], 0.0f);
            acc += __fmaf_rn(d, hi * x[base_k + 2*i + 1], 0.0f);
        }
    }

    float total = block_reduce_sum(acc, warp_sums);
    if (threadIdx.x == 0) out[row] = total;
}

/* ════════════════════════════════════════════════════════════════════════════
 * Q8_0 fused dequant + matvec
 * Block: 34 bytes, 32 weights: [d:f16 2][qs:32 int8]
 * ════════════════════════════════════════════════════════════════════════════ */
__global__ void k_q8_0_matvec(
    const uint8_t * __restrict__ W,
    int M, int K,
    const float   * __restrict__ x,
    float         * __restrict__ out)
{
    int row = blockIdx.x;
    if (row >= M) return;

    int num_blocks   = K / QK8_0;
    size_t row_bytes = (size_t)num_blocks * Q8_0_BLOCK;

    __shared__ float warp_sums[MAX_WARPS];
    float acc = 0.0f;

    for (int bi = threadIdx.x; bi < num_blocks; bi += blockDim.x) {
        const uint8_t *blk = W + (size_t)row * row_bytes + (size_t)bi * Q8_0_BLOCK;
        float d = d_f16_to_f32(*(const uint16_t *)blk);
        const int8_t *qs = (const int8_t *)(blk + 2);
        int base_k = bi * QK8_0;
        for (int i = 0; i < QK8_0; i++)
            acc += __fmaf_rn(d, (float)qs[i] * x[base_k + i], 0.0f);
    }

    float total = block_reduce_sum(acc, warp_sums);
    if (threadIdx.x == 0) out[row] = total;
}

/* ════════════════════════════════════════════════════════════════════════════
 * Persistent device buffer pool
 * Avoids cudaMalloc/Free on every matvec call for x and out vectors.
 * ════════════════════════════════════════════════════════════════════════════ */
static float *g_d_x   = NULL;   /* device hidden-state input   */
static float *g_d_out = NULL;   /* device matvec output        */
static int    g_d_K   = 0;
static int    g_d_M   = 0;

static int cuda_ensure_io_buffers(int M, int K)
{
    if (K > g_d_K) {
        if (g_d_x) { cudaFree(g_d_x); g_d_x = NULL; }
        if (cudaMalloc(&g_d_x, (size_t)K * sizeof(float)) != cudaSuccess) return 0;
        g_d_K = K;
    }
    if (M > g_d_M) {
        if (g_d_out) { cudaFree(g_d_out); g_d_out = NULL; }
        if (cudaMalloc(&g_d_out, (size_t)M * sizeof(float)) != cudaSuccess) return 0;
        g_d_M = M;
    }
    return 1;
}

/* ════════════════════════════════════════════════════════════════════════════
 * Public API
 * ════════════════════════════════════════════════════════════════════════════ */

/*
 * tb_cuda_upload_tensor — upload quantised weight blob to device VRAM.
 * Returns device pointer on success, NULL on failure (caller stores in
 * TB_GGUFTensorInfo.d_data).
 *
 * host_data : pointer to raw quantised bytes in host memory (mmap region)
 * byte_count: total bytes to upload
 */
void *tb_cuda_upload_tensor(const void *host_data, size_t byte_count)
{
    void *d_ptr = NULL;
    if (cudaMalloc(&d_ptr, byte_count) != cudaSuccess) return NULL;
    if (cudaMemcpy(d_ptr, host_data, byte_count, cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(d_ptr);
        return NULL;
    }
    return d_ptr;
}

/*
 * tb_cuda_free_tensor — release device VRAM for one tensor.
 */
void tb_cuda_free_tensor(void *d_ptr)
{
    if (d_ptr) cudaFree(d_ptr);
}

/*
 * tb_cuda_matvec_device — run fused dequant+matvec on device-resident weights.
 *
 * W_device : device pointer returned by tb_cuda_upload_tensor()
 * qtype    : GGUF quantisation type (2=Q4_0, 8=Q8_0, 12=Q4_K, 14=Q6_K)
 * M, K     : matrix dimensions (M rows × K columns)
 * x_host   : hidden-state vector in host memory (K floats)
 * out_host : output vector in host memory (M floats)
 *
 * Returns 1 on success, 0 to fall back to CPU path.
 */
int tb_cuda_matvec_device(const void *W_device, int qtype, int M, int K,
                           const float *x_host, float *out_host)
{
    /* Only dispatch for supported quant types */
    switch (qtype) {
        case  2: case  8: case 12: case 14: break;
        default: return 0;
    }

    if (!cuda_ensure_io_buffers(M, K)) return 0;

    /* Upload x (hidden vector, ~16 KB for hidden=4096 — negligible) */
    if (cudaMemcpy(g_d_x, x_host, (size_t)K * sizeof(float),
                   cudaMemcpyHostToDevice) != cudaSuccess) return 0;

    /* Launch: 1 block per output row, THREADS threads per block */
    dim3 grid((unsigned)M);
    dim3 block(THREADS);
    const uint8_t *Wd = (const uint8_t *)W_device;

    switch (qtype) {
    case 12: k_q4k_matvec <<<grid, block>>>(Wd, M, K, g_d_x, g_d_out); break;
    case 14: k_q6k_matvec <<<grid, block>>>(Wd, M, K, g_d_x, g_d_out); break;
    case  2: k_q4_0_matvec<<<grid, block>>>(Wd, M, K, g_d_x, g_d_out); break;
    case  8: k_q8_0_matvec<<<grid, block>>>(Wd, M, K, g_d_x, g_d_out); break;
    }

    if (cudaGetLastError() != cudaSuccess) return 0;
    cudaDeviceSynchronize();

    /* Download result (M floats, also tiny compared to weight upload) */
    if (cudaMemcpy(out_host, g_d_out, (size_t)M * sizeof(float),
                   cudaMemcpyDeviceToHost) != cudaSuccess) return 0;

    return 1;
}

/*
 * tb_cuda_cleanup — free persistent device buffers.
 * Call once at model teardown.
 */
void tb_cuda_cleanup(void)
{
    if (g_d_x)   { cudaFree(g_d_x);   g_d_x   = NULL; }
    if (g_d_out) { cudaFree(g_d_out); g_d_out = NULL; }
    g_d_K = g_d_M = 0;
}

/*
 * tb_cuda_init — optional: warm-up the CUDA context and report device.
 * Call once at startup.  Returns 1 if GPU is usable.
 */
int tb_cuda_init(void)
{
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) return 0;
    cudaSetDevice(0);
    struct cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    printf("[tb_cuda] %s  sm_%d%d  %.1f GB total  %.1f GB free\n",
           prop.name, prop.major, prop.minor,
           (double)total_mem / 1e9,
           (double)free_mem  / 1e9);
    return 1;
}

#ifdef __cplusplus
}
#endif
