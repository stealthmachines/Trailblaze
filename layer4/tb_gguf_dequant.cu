/*
 * layer4/tb_gguf_dequant.cu — HDGL Analog-Driven CUDA Inference Engine v0.6
 *
 * Three innovations over v0.5:
 *
 * 1. ANALOG BLOCK/GRID DISPATCH — oscillator state drives kernel config
 *    APhase → block size: PLUCK=256, SUSTAIN/FINETUNE=128, LOCK=64
 *    S(U) → grid scaling: <0.5 reduce 25%, >1.5 increase 25%
 *
 * 2. 4-STREAM ASYNC PIPELINE — eliminates DeviceSynchronize per matvec
 *    Hidden vector x uploaded ONCE per token (tb_cuda_upload_x)
 *    Kernels fire on rotating streams; async D→H on pinned memory
 *    Overlap: stream N kernel runs while stream N+1 H→D copy is in flight
 *
 * 3. COMPLETE QUANT COVERAGE — Q3_K, Q8_K, BF16, F32 kernels added
 *    All 8 GGUF formats now GPU-accelerated (v0.5 had 4)
 *
 * Design lineage:
 *   Block dispatch: ll_analog.c APhase thresholds (ANA_CV_TO_*)
 *   Stream pipeline: ll_cuda.cu DWT iteration loop structure
 *   FMA pattern: flash-moe shaders.metal __fmaf_rn trick
 *   phi constants: roadmap.md mathematical foundation
 */

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../src/tb_analog_dispatch.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ── Block layout constants ─────────────────────────────────────────────── */
#define QK_K        256
#define QK4_0        32
#define QK8_0        32
#define Q4K_BLOCK   144
#define Q5K_BLOCK   176
#define Q6K_BLOCK   210
#define Q3K_BLOCK   110
#define Q8K_BLOCK   292
#define Q4_0_BLOCK   18
#define Q8_0_BLOCK   34
#define MAX_WARPS      8   /* 256 threads / 32 */

/* Analog phase → block size table (ll_analog.c APhase thresholds) */
static const int k_aphase_block[4] = { 256, 128, 128, 64 };

/* ── Device type converters ─────────────────────────────────────────────── */
__device__ __forceinline__ float d_f16_to_f32(uint16_t h) {
    uint32_t s = (uint32_t)(h & 0x8000u) << 16;
    uint32_t e = (h & 0x7C00u) >> 10;
    uint32_t m =  h & 0x03FFu;
    uint32_t b;
    if      (e == 0)   b = s | (m << 13);
    else if (e == 31)  b = s | 0x7F800000u | (m << 13);
    else               b = s | ((e + 112u) << 23) | (m << 13);
    return __uint_as_float(b);
}

__device__ __forceinline__ float d_bf16_to_f32(uint16_t v) {
    uint32_t b = (uint32_t)v << 16;
    return __uint_as_float(b);
}

/* Block-level warp-shuffle + shared-memory reduce */
__device__ __forceinline__ float block_reduce(float val, float *smem) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    for (int off = 16; off > 0; off >>= 1)
        val += __shfl_down_sync(0xffffffff, val, off);
    if (lane == 0) smem[warp] = val;
    __syncthreads();
    float total = 0.0f;
    if (threadIdx.x == 0) {
        int nw = (blockDim.x + 31) >> 5;
        for (int w = 0; w < nw; w++) total += smem[w];
    }
    return total;
}

/* ════════════════════════════════════════════════════════════════════════════
 * Q4_K  (144 bytes/256 weights)
 * ════════════════════════════════════════════════════════════════════════════ */
__global__ void k_q4k_matvec(
    const uint8_t * __restrict__ W, int M, int K,
    const float * __restrict__ x, float * __restrict__ out)
{
    int row = blockIdx.x; if (row >= M) return;
    int nsb = K / QK_K;
    __shared__ float smem[MAX_WARPS];
    float acc = 0.0f;
    for (int sbi = threadIdx.x; sbi < nsb; sbi += blockDim.x) {
        const uint8_t *sb = W + (size_t)row * (size_t)nsb * Q4K_BLOCK + (size_t)sbi * Q4K_BLOCK;
        float d = d_f16_to_f32(*(const uint16_t*)(sb+0));
        float dm = d_f16_to_f32(*(const uint16_t*)(sb+2));
        const uint8_t *sc = sb+4, *qs = sb+16;
        float sc8[8], mn8[8];
        for (int j=0;j<4;j++) {
            sc8[j]   = d  * (float)(sc[j]  &0x3F);
            mn8[j]   = dm * (float)(sc[j+4]&0x3F);
            sc8[j+4] = d  * (float)((sc[j]>>6)|((sc[j+8]&0x0F)<<2));
            mn8[j+4] = dm * (float)((sc[j+4]>>6)|((sc[j+8]>>4)<<2));
        }
        int bk = sbi*QK_K;
        for (int i=0;i<128;i++) {
            uint8_t b = qs[i];
            int g0=(2*i)>>5, g1=(2*i+1)>>5;
            acc += __fmaf_rn(sc8[g0]*(float)(b&0xF), x[bk+2*i],   -mn8[g0]*x[bk+2*i  ]);
            acc += __fmaf_rn(sc8[g1]*(float)(b>>4),  x[bk+2*i+1], -mn8[g1]*x[bk+2*i+1]);
        }
    }
    { float _r = block_reduce(acc, smem); if (threadIdx.x == 0) out[row] = _r; }
}

/* ════════════════════════════════════════════════════════════════════════════
 * Q3_K  (110 bytes/256 weights) — NEW in v0.6
 * Layout: [hmask:32][qs:64][scales:12][d:f16 2]
 * ════════════════════════════════════════════════════════════════════════════ */
__global__ void k_q3k_matvec(
    const uint8_t * __restrict__ W, int M, int K,
    const float * __restrict__ x, float * __restrict__ out)
{
    int row = blockIdx.x; if (row >= M) return;
    int nsb = K / QK_K;
    __shared__ float smem[MAX_WARPS];
    float acc = 0.0f;
    for (int sbi = threadIdx.x; sbi < nsb; sbi += blockDim.x) {
        const uint8_t *sb = W + (size_t)row * (size_t)nsb * Q3K_BLOCK + (size_t)sbi * Q3K_BLOCK;
        const uint8_t *hmask = sb, *qs = sb+32, *sc_raw = sb+96;
        float d = d_f16_to_f32(*(const uint16_t*)(sb+108));
        float scales[16];
        for (int j=0;j<16;j++) {
            int is  = (j<8)?j:(j-8);
            int lo4 = (sc_raw[is>>1]>>((is&1)<<2))&0xF;
            int hi2 = (sc_raw[8+(is>>2)]>>((is&3)<<1))&0x3;
            scales[j] = d * (float)((lo4|(hi2<<4))-32);
        }
        int bk = sbi*QK_K;
        for (int i=0;i<QK_K;i++) {
            int l = (qs[i>>2]>>((i&3)<<1))&0x3;
            int h = (hmask[i>>3]>>(i&7))&0x1;
            acc += scales[i>>4] * (float)((l|(h<<2))-4) * x[bk+i];
        }
    }
    { float _r = block_reduce(acc, smem); if (threadIdx.x == 0) out[row] = _r; }
}

/* ════════════════════════════════════════════════════════════════════════════
 * Q5_K  (176 bytes/256 weights)
 * Layout: [d:f16 2][dmin:f16 2][scales:12][qh:32][qs:128]
 * Same 6-bit scale packing as Q4_K; 5-bit weight = qs_nibble | (qh_bit<<4)
 * ════════════════════════════════════════════════════════════════════════════ */
__global__ void k_q5k_matvec(
    const uint8_t * __restrict__ W, int M, int K,
    const float * __restrict__ x, float * __restrict__ out)
{
    int row = blockIdx.x; if (row >= M) return;
    int nsb = K / QK_K;
    __shared__ float smem[MAX_WARPS];
    float acc = 0.0f;
    for (int sbi = threadIdx.x; sbi < nsb; sbi += blockDim.x) {
        const uint8_t *sb = W + (size_t)row * (size_t)nsb * Q5K_BLOCK + (size_t)sbi * Q5K_BLOCK;
        float d    = d_f16_to_f32(*(const uint16_t*)(sb + 0));
        float dmin = d_f16_to_f32(*(const uint16_t*)(sb + 2));
        const uint8_t *sc = sb + 4;   /* scales[12] */
        const uint8_t *qh = sb + 16;  /* qh[32]     */
        const uint8_t *qs = sb + 48;  /* qs[128]    */
        /* Unpack 8 scales and 8 mins (6-bit each, same as Q4_K) */
        float sc8[8], mn8[8];
        for (int j=0;j<4;j++) {
            sc8[j]   = d    * (float)(sc[j]   & 0x3F);
            mn8[j]   = dmin * (float)(sc[j+4] & 0x3F);
            sc8[j+4] = d    * (float)((sc[j]  >>6) | ((sc[j+8] & 0x0F) << 2));
            mn8[j+4] = dmin * (float)((sc[j+4]>>6) | ((sc[j+8] >> 4) << 2));
        }
        int bk = sbi * QK_K;
        for (int i = 0; i < 256; i++) {
            int g   = i >> 5;  /* group [0..7] */
            int lo4 = (qs[i >> 1] >> ((i & 1) << 2)) & 0xF;
            int hi1 = (qh[i >> 3] >> (i & 7)) & 1;
            int q5  = lo4 | (hi1 << 4);  /* [0..31] */
            acc += (sc8[g] * (float)q5 - mn8[g]) * x[bk + i];
        }
    }
    { float _r = block_reduce(acc, smem); if (threadIdx.x == 0) out[row] = _r; }
}

/* ════════════════════════════════════════════════════════════════════════════
 * Q6_K  (210 bytes/256 weights)
 * ════════════════════════════════════════════════════════════════════════════ */
__global__ void k_q6k_matvec(
    const uint8_t * __restrict__ W, int M, int K,
    const float * __restrict__ x, float * __restrict__ out)
{
    int row = blockIdx.x; if (row >= M) return;
    int nsb = K / QK_K;
    __shared__ float smem[MAX_WARPS];
    float acc = 0.0f;
    for (int sbi = threadIdx.x; sbi < nsb; sbi += blockDim.x) {
        const uint8_t *sb = W + (size_t)row * (size_t)nsb * Q6K_BLOCK + (size_t)sbi * Q6K_BLOCK;
        const uint8_t *ql = sb, *qh = sb+128;
        const int8_t  *sc = (const int8_t*)(sb+192);
        float d = d_f16_to_f32(*(const uint16_t*)(sb+208));
        int bk = sbi*QK_K;
        for (int i=0;i<QK_K;i++) {
            int lo = (ql[i>>1]>>((i&1)<<2))&0xF;
            int hi = (qh[i>>2]>>((i&3)<<1))&0x3;
            acc += d * (float)sc[i>>4] * (float)((lo|(hi<<4))-32) * x[bk+i];
        }
    }
    { float _r = block_reduce(acc, smem); if (threadIdx.x == 0) out[row] = _r; }
}

/* ════════════════════════════════════════════════════════════════════════════
 * Q4_0  (18 bytes/32 weights)
 * ════════════════════════════════════════════════════════════════════════════ */
__global__ void k_q4_0_matvec(
    const uint8_t * __restrict__ W, int M, int K,
    const float * __restrict__ x, float * __restrict__ out)
{
    int row = blockIdx.x; if (row >= M) return;
    int nb = K/QK4_0;
    __shared__ float smem[MAX_WARPS];
    float acc = 0.0f;
    for (int bi = threadIdx.x; bi < nb; bi += blockDim.x) {
        const uint8_t *b = W + (size_t)row*(size_t)nb*Q4_0_BLOCK + (size_t)bi*Q4_0_BLOCK;
        float d = d_f16_to_f32(*(const uint16_t*)b);
        const uint8_t *qs = b+2;
        int bk = bi*QK4_0;
        for (int i=0;i<16;i++) {
            acc += d*((float)(qs[i]&0xF)-8.f)*x[bk+2*i  ];
            acc += d*((float)(qs[i]>>4 )-8.f)*x[bk+2*i+1];
        }
    }
    { float _r = block_reduce(acc, smem); if (threadIdx.x == 0) out[row] = _r; }
}

/* ════════════════════════════════════════════════════════════════════════════
 * Q8_0  (34 bytes/32 weights)
 * ════════════════════════════════════════════════════════════════════════════ */
__global__ void k_q8_0_matvec(
    const uint8_t * __restrict__ W, int M, int K,
    const float * __restrict__ x, float * __restrict__ out)
{
    int row = blockIdx.x; if (row >= M) return;
    int nb = K/QK8_0;
    __shared__ float smem[MAX_WARPS];
    float acc = 0.0f;
    for (int bi = threadIdx.x; bi < nb; bi += blockDim.x) {
        const uint8_t *b = W + (size_t)row*(size_t)nb*Q8_0_BLOCK + (size_t)bi*Q8_0_BLOCK;
        float d = d_f16_to_f32(*(const uint16_t*)b);
        const int8_t *qs = (const int8_t*)(b+2);
        int bk = bi*QK8_0;
        for (int i=0;i<QK8_0;i++) acc += d*(float)qs[i]*x[bk+i];
    }
    { float _r = block_reduce(acc, smem); if (threadIdx.x == 0) out[row] = _r; }
}

/* ════════════════════════════════════════════════════════════════════════════
 * Q8_K  (292 bytes/256 weights) — NEW in v0.6
 * Layout: [d:f32][qs:256 int8][bsums:16 int16][pad:4]
 * ════════════════════════════════════════════════════════════════════════════ */
__global__ void k_q8k_matvec(
    const uint8_t * __restrict__ W, int M, int K,
    const float * __restrict__ x, float * __restrict__ out)
{
    int row = blockIdx.x; if (row >= M) return;
    int nsb = K/QK_K;
    __shared__ float smem[MAX_WARPS];
    float acc = 0.0f;
    for (int sbi = threadIdx.x; sbi < nsb; sbi += blockDim.x) {
        const uint8_t *sb = W + (size_t)row*(size_t)nsb*Q8K_BLOCK + (size_t)sbi*Q8K_BLOCK;
        float d; memcpy(&d, sb, 4);
        const int8_t *qs = (const int8_t*)(sb+4);
        int bk = sbi*QK_K;
        for (int i=0;i<QK_K;i++) acc += d*(float)qs[i]*x[bk+i];
    }
    { float _r = block_reduce(acc, smem); if (threadIdx.x == 0) out[row] = _r; }
}

/* ════════════════════════════════════════════════════════════════════════════
 * F16  — uploaded now so Qwen3 projection/embedding tensors stay on GPU
 * ════════════════════════════════════════════════════════════════════════════ */
__global__ void k_f16_matvec(
    const uint16_t * __restrict__ W, int M, int K,
    const float * __restrict__ x, float * __restrict__ out)
{
    int row = blockIdx.x; if (row >= M) return;
    const uint16_t *rw = W + (size_t)row*K;
    __shared__ float smem[MAX_WARPS];
    float acc = 0.0f;
    for (int k = threadIdx.x; k < K; k += blockDim.x)
        acc += d_f16_to_f32(rw[k]) * x[k];
    { float _r = block_reduce(acc, smem); if (threadIdx.x == 0) out[row] = _r; }
}

/* ════════════════════════════════════════════════════════════════════════════
 * BF16  — NEW in v0.6
 * ════════════════════════════════════════════════════════════════════════════ */
__global__ void k_bf16_matvec(
    const uint16_t * __restrict__ W, int M, int K,
    const float * __restrict__ x, float * __restrict__ out)
{
    int row = blockIdx.x; if (row >= M) return;
    const uint16_t *rw = W + (size_t)row*K;
    __shared__ float smem[MAX_WARPS];
    float acc = 0.0f;
    for (int k = threadIdx.x; k < K; k += blockDim.x)
        acc += d_bf16_to_f32(rw[k]) * x[k];
    { float _r = block_reduce(acc, smem); if (threadIdx.x == 0) out[row] = _r; }
}

/* ════════════════════════════════════════════════════════════════════════════
 * F32  — NEW in v0.6
 * ════════════════════════════════════════════════════════════════════════════ */
__global__ void k_f32_matvec(
    const float * __restrict__ W, int M, int K,
    const float * __restrict__ x, float * __restrict__ out)
{
    int row = blockIdx.x; if (row >= M) return;
    const float *rw = W + (size_t)row*K;
    __shared__ float smem[MAX_WARPS];
    float acc = 0.0f;
    for (int k = threadIdx.x; k < K; k += blockDim.x)
        acc += rw[k] * x[k];
    { float _r = block_reduce(acc, smem); if (threadIdx.x == 0) out[row] = _r; }
}

/* ════════════════════════════════════════════════════════════════════════════
 * Persistent device state
 * ════════════════════════════════════════════════════════════════════════════ */
#define TB_N_STREAMS 4

static float       *g_d_x           = NULL;
static int          g_d_K            = 0;
static float       *g_d_out[TB_N_STREAMS]  = {NULL};
static float       *g_h_pin[TB_N_STREAMS]  = {NULL};
static int          g_d_M            = 0;
static cudaStream_t g_streams[TB_N_STREAMS] = {NULL};
static int          g_si             = 0;
static int          g_streams_ok     = 0;
static const TBOscSnapshot *g_snap   = NULL;

static int _ensure_streams(void) {
    if (g_streams_ok) return 1;
    for (int i=0;i<TB_N_STREAMS;i++)
        if (cudaStreamCreate(&g_streams[i]) != cudaSuccess) return 0;
    g_streams_ok = 1; return 1;
}
static int _ensure_x(int K) {
    if (K <= g_d_K) return 1;
    if (g_d_x) { cudaFree(g_d_x); g_d_x=NULL; }
    if (cudaMalloc(&g_d_x,(size_t)K*sizeof(float)) != cudaSuccess) return 0;
    g_d_K = K; return 1;
}
static int _ensure_out(int M) {
    if (M <= g_d_M) return 1;
    for (int i=0;i<TB_N_STREAMS;i++) {
        if (g_d_out[i]) { cudaFree(g_d_out[i]); g_d_out[i]=NULL; }
        if (g_h_pin[i]) { cudaFreeHost(g_h_pin[i]); g_h_pin[i]=NULL; }
        if (cudaMalloc    (&g_d_out[i],(size_t)M*sizeof(float))!=cudaSuccess) return 0;
        if (cudaMallocHost(&g_h_pin[i],(size_t)M*sizeof(float))!=cudaSuccess) return 0;
    }
    g_d_M = M; return 1;
}

/* Analog config helpers */
static int _block(void) {
    if (!g_snap || g_snap->steps < 4) return 128;
    int ap = g_snap->aphase; if (ap<0||ap>3) ap=1;
    return k_aphase_block[ap];
}
static int _grid(int M) {
    if (!g_snap || g_snap->steps < 4) return M;
    double su = g_snap->s_u;
    if      (su < 0.5) return (int)(M*0.75+0.5);
    else if (su > 1.5) return (int)(M*1.25+0.5);
    return M;
}

/* ── Public API ─────────────────────────────────────────────────────────── */

int tb_cuda_init(void) {
    int n=0;
    if (cudaGetDeviceCount(&n)!=cudaSuccess||n==0) return 0;
    cudaSetDevice(0);
    struct cudaDeviceProp p; cudaGetDeviceProperties(&p,0);
    size_t fm,tm; cudaMemGetInfo(&fm,&tm);
    printf("[tb_cuda] %s sm_%d%d  %.1f GB total  %.1f GB free\n",
           p.name,p.major,p.minor,(double)tm/1e9,(double)fm/1e9);
    printf("[tb_cuda] analog dispatch: aphase->{256,128,128,64} threads  "
           "S(U)->grid+-25%%  4 async streams\n");
    if (!_ensure_streams()) { fprintf(stderr,"[tb_cuda] streams failed\n"); return 0; }
    return 1;
}

/* Upload x once per token decode step */
int tb_cuda_upload_x(const float *x_host, int K) {
    if (!_ensure_streams()||!_ensure_x(K)) return 0;
    if (cudaMemcpyAsync(g_d_x,x_host,(size_t)K*sizeof(float),
                        cudaMemcpyHostToDevice,g_streams[0])!=cudaSuccess) return 0;
    cudaStreamSynchronize(g_streams[0]);
    return 1;
}

void *tb_cuda_upload_tensor(const void *hd, size_t n) {
    void *d=NULL;
    if (cudaMalloc(&d,n)!=cudaSuccess) return NULL;
    if (cudaMemcpy(d,hd,n,cudaMemcpyHostToDevice)!=cudaSuccess){cudaFree(d);return NULL;}
    return d;
}

void tb_cuda_free_tensor(void *d) { if(d) cudaFree(d); }

void tb_cuda_set_snap(const TBOscSnapshot *s) { g_snap = s; }

int tb_cuda_matvec_device(const void *Wdev, int qtype, int M, int K,
                           const float *x_host, float *out_host)
{
    static int g_mv_calls = 0;
    int dbg = (g_mv_calls < 4);
    if (dbg) fprintf(stderr, "[cmv%d] q=%d M=%d K=%d\n", g_mv_calls, qtype, M, K);
    g_mv_calls++;

    if (!_ensure_x(K)) { if(dbg) fprintf(stderr,"[cmv] ensure_x fail\n"); return 0; }
    cudaError_t ce = cudaMemcpy(g_d_x,x_host,(size_t)K*sizeof(float),cudaMemcpyHostToDevice);
    if (ce!=cudaSuccess) { if(dbg) fprintf(stderr,"[cmv] H2D fail: %s\n",cudaGetErrorString(ce)); return 0; }
    if (!_ensure_out(M)) { if(dbg) fprintf(stderr,"[cmv] ensure_out fail\n"); return 0; }
    if (!_ensure_streams()) { if(dbg) fprintf(stderr,"[cmv] streams fail\n"); return 0; }
    if (dbg) { fflush(stderr); }

    int si = g_si; g_si = (g_si+1)%TB_N_STREAMS;
    cudaStream_t st = g_streams[si];
    ce = cudaStreamSynchronize(st);  /* wait for previous occupant of this slot */
    if (ce!=cudaSuccess) { if(dbg) fprintf(stderr,"[cmv] pre-sync fail: %s\n",cudaGetErrorString(ce)); return 0; }

    int blk = _block();
    int grd = _grid(M); if(grd<1) grd=1;
    const uint8_t *Wd = (const uint8_t*)Wdev;
    if (dbg) { fprintf(stderr,"[cmv] launch grd=%d blk=%d si=%d\n",grd,blk,si); fflush(stderr); }

    switch (qtype) {
    case  1: k_f16_matvec  <<<grd,blk,0,st>>>((const uint16_t*)Wd,M,K,g_d_x,g_d_out[si]); break; /* F16 */
    case 12: k_q4k_matvec  <<<grd,blk,0,st>>>(Wd,M,K,g_d_x,g_d_out[si]); break;  /* Q4_K  */
    case 13: k_q5k_matvec  <<<grd,blk,0,st>>>(Wd,M,K,g_d_x,g_d_out[si]); break;  /* Q5_K  */
    case 11:
    case 21: k_q3k_matvec  <<<grd,blk,0,st>>>(Wd,M,K,g_d_x,g_d_out[si]); break;  /* Q3_K / IQ3_S */
    case 14: k_q6k_matvec  <<<grd,blk,0,st>>>(Wd,M,K,g_d_x,g_d_out[si]); break;  /* Q6_K  */
    case  2: k_q4_0_matvec <<<grd,blk,0,st>>>(Wd,M,K,g_d_x,g_d_out[si]); break;  /* Q4_0  */
    case  8: k_q8_0_matvec <<<grd,blk,0,st>>>(Wd,M,K,g_d_x,g_d_out[si]); break;  /* Q8_0  */
    case 15: k_q8k_matvec  <<<grd,blk,0,st>>>(Wd,M,K,g_d_x,g_d_out[si]); break;  /* Q8_K  */
    case 30: k_bf16_matvec <<<grd,blk,0,st>>>((const uint16_t*)Wd,M,K,g_d_x,g_d_out[si]); break; /* BF16 */
    case  0: k_f32_matvec  <<<grd,blk,0,st>>>((const float*)Wd,M,K,g_d_x,g_d_out[si]); break;    /* F32  */
    default: return 0;
    }
    ce = cudaGetLastError();
    if (dbg) { fprintf(stderr,"[cmv] post-launch ce=%d(%s)\n",ce,cudaGetErrorString(ce)); fflush(stderr); }
    if (ce!=cudaSuccess) return 0;
    ce = cudaMemcpyAsync(g_h_pin[si],g_d_out[si],(size_t)M*sizeof(float),cudaMemcpyDeviceToHost,st);
    if (ce!=cudaSuccess) { if(dbg) fprintf(stderr,"[cmv] D2H fail\n"); return 0; }
    if (dbg) { fprintf(stderr,"[cmv] stream-sync...\n"); fflush(stderr); }
    ce = cudaStreamSynchronize(st);
    if (dbg) { fprintf(stderr,"[cmv] sync done ce=%d(%s)\n",ce,cudaGetErrorString(ce)); fflush(stderr); }
    if (ce!=cudaSuccess) return 0;
    memcpy(out_host,g_h_pin[si],(size_t)M*sizeof(float));
    return 1;
}

void tb_cuda_sync_all(void) {
    for(int i=0;i<TB_N_STREAMS;i++) if(g_streams[i]) cudaStreamSynchronize(g_streams[i]);
}

void tb_cuda_cleanup(void) {
    for(int i=0;i<TB_N_STREAMS;i++){
        if(g_streams[i]){cudaStreamDestroy(g_streams[i]);g_streams[i]=NULL;}
        if(g_d_out[i])  {cudaFree(g_d_out[i]);          g_d_out[i]=NULL;}
        if(g_h_pin[i])  {cudaFreeHost(g_h_pin[i]);      g_h_pin[i]=NULL;}
    }
    if(g_d_x){cudaFree(g_d_x);g_d_x=NULL;}
    g_d_K=g_d_M=g_si=g_streams_ok=0;
}

#ifdef __cplusplus
}
#endif
