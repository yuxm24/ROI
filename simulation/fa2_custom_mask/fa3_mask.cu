#include "fa3_launch.h"
#include "fa3_config.h"

#include <cstdint>

namespace {

constexpr int kWarpsPerBlock = 8;

__global__ void pack_mask_kernel(
    bool const* __restrict__ mask,
    uint32_t* __restrict__ packed,
    int batch,
    int packed_heads,
    int lq,
    int lkv,
    int words,
    int64_t mask_batch_stride,
    int64_t mask_head_stride,
    int64_t mask_q_stride,
    int64_t mask_k_stride) {
  int const warp = threadIdx.x >> 5;
  int const lane = threadIdx.x & 31;
  int64_t const row = int64_t(blockIdx.x) * kWarpsPerBlock + warp;
  int64_t const rows = int64_t(batch) * packed_heads * lq;
  if (row >= rows) {
    return;
  }
  int const q = row % lq;
  int64_t const bh = row / lq;
  int const h = bh % packed_heads;
  int const b = bh / packed_heads;
  int64_t const row_offset = int64_t(b) * mask_batch_stride
                           + int64_t(h) * mask_head_stride
                           + int64_t(q) * mask_q_stride;
  uint32_t* const packed_row = packed + row * words;
  for (int word = 0; word < words; ++word) {
    int const k = word * 32 + lane;
    bool const keep = k < lkv && mask[row_offset + int64_t(k) * mask_k_stride];
    uint32_t const bits = __ballot_sync(0xffffffffu, keep);
    if (lane == 0) {
      packed_row[word] = bits;
    }
  }
}

__device__ __forceinline__ uint32_t bit_range_mask(int lo, int hi) {
  uint32_t const below_lo = lo == 0 ? 0u : ((uint32_t{1} << lo) - 1u);
  uint32_t const below_hi = hi == 32 ? 0xffffffffu : ((uint32_t{1} << hi) - 1u);
  return below_hi & ~below_lo;
}

__device__ __forceinline__ void build_tile_map_task(
    uint32_t const* __restrict__ packed,
    uint8_t* __restrict__ tile_state,
    int64_t tile_linear,
    int packed_heads,
    int lq,
    int lkv,
    int words,
    int tiles_m,
    int tiles_n,
    int block_m,
    int block_n) {
  int const lane = threadIdx.x & 31;
  int64_t tile_index = tile_linear;
  int const tile_n = tile_index % tiles_n;
  tile_index /= tiles_n;
  int const tile_m = tile_index % tiles_m;
  int const bh = tile_index / tiles_m;
  int const q_begin = tile_m * block_m;
  int const q_end_raw = q_begin + block_m;
  int const q_end = q_end_raw < lq ? q_end_raw : lq;
  int const k_begin = tile_n * block_n;
  int const k_end_raw = k_begin + block_n;
  int const k_end = k_end_raw < lkv ? k_end_raw : lkv;
  int const word_begin = k_begin >> 5;
  int const word_end = (k_end + 31) >> 5;
  int const tile_words = word_end - word_begin;

  bool local_any = false;
  bool local_all = true;
  int const pair_count = (q_end - q_begin) * tile_words;
  for (int pair = lane; pair < pair_count; pair += 32) {
    int const q = q_begin + pair / tile_words;
    int const word = word_begin + pair % tile_words;
    int const word_k_begin = word * 32;
    int const lo = k_begin > word_k_begin ? k_begin - word_k_begin : 0;
    int const word_k_end = word_k_begin + 32;
    int const hi = k_end < word_k_end ? k_end - word_k_begin : 32;
    uint32_t const valid_bits = bit_range_mask(lo, hi);
    uint32_t const bits = packed[(int64_t(bh) * lq + q) * words + word] & valid_bits;
    local_any |= bits != 0u;
    local_all &= bits == valid_bits;
  }
  bool const tile_any = __any_sync(0xffffffffu, local_any);
  bool const tile_all = __all_sync(0xffffffffu, local_all);
  if (lane == 0) {
    tile_state[tile_linear] =
        !tile_any ? uint8_t{0} : (tile_all ? uint8_t{2} : uint8_t{1});
  }
}

__device__ __forceinline__ void build_row_map_task(
    uint32_t const* __restrict__ packed,
    uint32_t* __restrict__ row_state,
    int64_t row_tile_linear,
    int lq,
    int lkv,
    int words,
    int tiles_n,
    int block_n) {
  int const lane = threadIdx.x & 31;
  int const tile_n = row_tile_linear % tiles_n;
  int64_t const row_linear = row_tile_linear / tiles_n;
  int const k_begin = tile_n * block_n;
  int const k_end_raw = k_begin + block_n;
  int const k_end = k_end_raw < lkv ? k_end_raw : lkv;
  int const word_begin = k_begin >> 5;
  int const word_end = (k_end + 31) >> 5;

  bool local_any = false;
  bool local_all = true;
  int local_count = 0;
  int local_first = 0x7fffffff;
  int local_last = -1;
  for (int word = word_begin + lane; word < word_end; word += 32) {
    int const word_k_begin = word * 32;
    int const lo = k_begin > word_k_begin ? k_begin - word_k_begin : 0;
    int const word_k_end = word_k_begin + 32;
    int const hi = k_end < word_k_end ? k_end - word_k_begin : 32;
    uint32_t const valid_bits = bit_range_mask(lo, hi);
    uint32_t const bits = packed[row_linear * words + word] & valid_bits;
    local_any |= bits != 0u;
    local_all &= bits == valid_bits;
    local_count += __popc(bits);
    if (bits != 0u) {
      int const first = word_k_begin + __ffs(bits) - 1;
      int const last = word_k_begin + 31 - __clz(bits);
      local_first = first < local_first ? first : local_first;
      local_last = last > local_last ? last : local_last;
    }
  }
  bool const row_any = __any_sync(0xffffffffu, local_any);
  bool const row_all = __all_sync(0xffffffffu, local_all);
  for (int offset = 16; offset > 0; offset >>= 1) {
    local_count += __shfl_down_sync(0xffffffffu, local_count, offset);
    int const other_first = __shfl_down_sync(0xffffffffu, local_first, offset);
    int const other_last = __shfl_down_sync(0xffffffffu, local_last, offset);
    local_first = other_first < local_first ? other_first : local_first;
    local_last = other_last > local_last ? other_last : local_last;
  }
  if (lane == 0) {
    constexpr uint32_t kEmpty = 0xffffffffu;
    constexpr uint32_t kFull = 0xfffffffeu;
    constexpr uint32_t kFallback = 0xfffffffdu;
    uint32_t descriptor = kFallback;
    if (!row_any) {
      descriptor = kEmpty;
    } else if (row_all) {
      descriptor = kFull;
    } else if (local_first <= 0xffff && local_last < 0xffff
               && local_count == local_last - local_first + 1) {
      descriptor = uint32_t(local_first) | (uint32_t(local_last + 1) << 16);
    }
    row_state[row_tile_linear] = descriptor;
  }
}

__global__ void build_metadata_kernel(
    uint32_t const* __restrict__ packed,
    uint8_t* __restrict__ fwd_tile_state,
    uint8_t* __restrict__ bwd_tile_state,
    uint32_t* __restrict__ fwd_row_state,
    int batch,
    int packed_heads,
    int lq,
    int lkv,
    int words,
    int tiles_n,
    int fwd_tiles_m,
    int bwd_tiles_m,
    int64_t fwd_tile_count,
    int64_t bwd_tile_count,
    int64_t row_tile_count) {
  int const warp = threadIdx.x >> 5;
  int64_t const task = int64_t(blockIdx.x) * kWarpsPerBlock + warp;
  if (task < fwd_tile_count) {
    build_tile_map_task(
        packed, fwd_tile_state, task, packed_heads, lq, lkv, words,
        fwd_tiles_m, tiles_n, fa2mask_fwd_block_m(lq), 128);
  } else if (task < fwd_tile_count + bwd_tile_count) {
    build_tile_map_task(
        packed, bwd_tile_state, task - fwd_tile_count, packed_heads, lq, lkv,
        words, bwd_tiles_m, tiles_n, 64, 128);
  } else if (task < fwd_tile_count + bwd_tile_count + row_tile_count) {
    build_row_map_task(
        packed, fwd_row_state, task - fwd_tile_count - bwd_tile_count,
        lq, lkv, words, tiles_n, 128);
  }
}

}  // namespace

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
    cudaStream_t stream) {
  int64_t const rows = int64_t(batch) * packed_heads * lq;
  int const blocks = static_cast<int>((rows + kWarpsPerBlock - 1) / kWarpsPerBlock);
  pack_mask_kernel<<<blocks, kWarpsPerBlock * 32, 0, stream>>>(
      mask, packed, batch, packed_heads, lq, lkv, (lkv + 31) / 32,
      mask_batch_stride, mask_head_stride, mask_q_stride, mask_k_stride);
}

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
    cudaStream_t stream) {
  int tiles_n = (lkv + 127) / 128;
  int const fwd_block_m = fa2mask_fwd_block_m(lq);
  int fwd_tiles_m = (lq + fwd_block_m - 1) / fwd_block_m;
  int bwd_tiles_m = (lq + 63) / 64;
  int64_t const bh = int64_t(batch) * packed_heads;
  int64_t fwd_tile_count = bh * fwd_tiles_m * tiles_n;
  int64_t bwd_tile_count = bh * bwd_tiles_m * tiles_n;
  int64_t row_tile_count = bh * lq * tiles_n;
  int64_t const task_count = fwd_tile_count + bwd_tile_count + row_tile_count;
  int const blocks = static_cast<int>(
      (task_count + kWarpsPerBlock - 1) / kWarpsPerBlock);
  build_metadata_kernel<<<blocks, kWarpsPerBlock * 32, 0, stream>>>(
      packed, fwd_tile_state, bwd_tile_state, fwd_row_state, batch,
      packed_heads, lq, lkv, words, tiles_n, fwd_tiles_m, bwd_tiles_m,
      fwd_tile_count, bwd_tile_count, row_tile_count);
}
