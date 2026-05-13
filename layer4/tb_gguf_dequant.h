/*
 * layer4/tb_gguf_dequant.h — C interface to CUDA dequant+matvec kernels
 *
 * v0.6: analog-driven block/grid dispatch, 4-stream async pipeline,
 *       Q3_K + Q8_K + BF16 + F32 kernels, persistent x buffer.
 */
#pragma once
#ifndef TB_GGUF_DEQUANT_H
#define TB_GGUF_DEQUANT_H

#include <stddef.h>
#include "../src/tb_analog_dispatch.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Warm up CUDA, create 4 async streams, print device info. */
int  tb_cuda_init(void);

/* Upload quantised weight blob to VRAM once at model-load time. */
void *tb_cuda_upload_tensor(const void *host_data, size_t byte_count);
void  tb_cuda_free_tensor(void *d_ptr);

/* Upload hidden-state vector x ONCE per token decode step.
 * All matvec calls in that step reuse g_d_x — zero redundant H→D copies. */
int  tb_cuda_upload_x(const float *x_host, int K);

/* Wire the live oscillator snapshot into GPU dispatch.
 * Call from tb_dispatch_context_set() so CUDA sees the same snap as CPU. */
void tb_cuda_set_snap(const TBOscSnapshot *snap);

/* Analog-dispatched fused dequant+matvec.
 * Block size from aphase; grid size from S(U); async stream round-robin.
 * Supported qtypes: 0=F32 2=Q4_0 8=Q8_0 11=Q3_K 12=Q4_K 14=Q6_K 15=Q8_K 30=BF16 */
int  tb_cuda_matvec_device(const void *W_device, int qtype, int M, int K,
                            const float *x_host, float *out_host);

/* Synchronize all 4 streams — call at end of each decode step. */
void tb_cuda_sync_all(void);

/* Free all device/host buffers and destroy streams. */
void tb_cuda_cleanup(void);

#ifdef __cplusplus
}
#endif

#endif /* TB_GGUF_DEQUANT_H */
