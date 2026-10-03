#pragma once

#include <cuda_runtime_api.h>

#include "vendor/flash_attn_hopper/flash.h"

void fa2mask_fa3_launch_forward(Flash_fwd_params& params, cudaStream_t stream);
void fa2mask_fa3_launch_backward(Flash_bwd_params& params, cudaStream_t stream);
void fa2mask_fa3_pack_mask(
    bool const* mask,
    uint32_t* packed,
    int batch,
    int packed_heads,
    int lq,
    int lkv,
    int64_t mask_batch_stride,
    int64_t mask_head_stride,
    int64_t mask_q_stride,
    int64_t mask_k_stride,
    cudaStream_t stream);
void fa2mask_fa3_build_metadata(
    uint32_t const* packed,
    uint8_t* fwd_tile_state,
    uint8_t* bwd_tile_state,
    uint32_t* fwd_row_state,
    int batch,
    int packed_heads,
    int lq,
    int lkv,
    int words,
    cudaStream_t stream);
