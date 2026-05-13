/*
 * layer4/tb_gguf_dequant.h — C interface to CUDA dequant+matvec kernels
 *
 * Compile tb_gguf_dequant.cu with nvcc and link the resulting object.
 * Guard every call site with #ifdef TB_CUDA.
 */
#pragma once
#ifndef TB_GGUF_DEQUANT_H
#define TB_GGUF_DEQUANT_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * tb_cuda_init — warm up CUDA context, print device info.
 * Returns 1 if a usable GPU was found, 0 otherwise.
 * Call once at startup before tb_gguf_load().
 */
int tb_cuda_init(void);

/*
 * tb_cuda_upload_tensor — copy quantised weight blob to device VRAM.
 *
 *   host_data  : raw quantised bytes in host memory (mmap pointer + offset)
 *   byte_count : exact byte size of the tensor
 *
 * Returns an opaque device pointer on success, NULL on failure.
 * Store the return value in TB_GGUFTensorInfo.d_data.
 * On failure the CPU fallback path in tb_gguf_dequant_matvec() is used.
 */
void *tb_cuda_upload_tensor(const void *host_data, size_t byte_count);

/*
 * tb_cuda_free_tensor — release the device buffer for one tensor.
 * Called from tb_gguf_free() for every tensor that has d_data != NULL.
 */
void tb_cuda_free_tensor(void *d_ptr);

/*
 * tb_cuda_matvec_device — fused dequant+matvec entirely on GPU.
 *
 *   W_device  : device pointer (from tb_cuda_upload_tensor)
 *   qtype     : GGUF type (2=Q4_0, 8=Q8_0, 12=Q4_K, 14=Q6_K)
 *   M, K      : weight matrix dimensions
 *   x_host    : hidden vector (host, K floats)
 *   out_host  : output vector (host, M floats)
 *
 * Returns 1 on success; 0 signals the caller to fall back to CPU.
 */
int tb_cuda_matvec_device(const void *W_device, int qtype, int M, int K,
                           const float *x_host, float *out_host);

/*
 * tb_cuda_cleanup — free persistent device I/O buffers.
 * Call once at model teardown after all tb_cuda_free_tensor() calls.
 */
void tb_cuda_cleanup(void);

#ifdef __cplusplus
}
#endif

#endif /* TB_GGUF_DEQUANT_H */
