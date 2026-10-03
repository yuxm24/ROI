/******************************************************************************
 * Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
 ******************************************************************************/

#pragma once

#include <cstdint>

#include <cute/tensor.hpp>

#include "cutlass/fast_math.h"  // For cutlass::FastDivmod

#include "utils.h"
#include "fa3_config.h"

namespace flash {

using namespace cute;

template <int kBlockM, int kBlockN, bool PackGQA, typename TiledMma, bool SwapAB=false>
struct Mask {

    static_assert(!(PackGQA && SwapAB), "Cannot be both PackGQA and SwapAB");

    int const thread_idx;
    int const seqlen_q, seqlen_k;
    int const window_size_left, window_size_right, sink_token_length;
    cutlass::FastDivmod const attention_chunk_divmod;
    cutlass::FastDivmod const qhead_per_khead_divmod;
    uint32_t const* const custom_mask_ptr;
    int64_t const custom_mask_batch_stride, custom_mask_head_stride;
    int64_t const custom_mask_q_stride, custom_mask_k_stride;
    uint8_t const* const custom_mask_tile_ptr;
    int64_t const custom_mask_tile_batch_stride, custom_mask_tile_head_stride;
    int64_t const custom_mask_tile_q_stride, custom_mask_tile_k_stride;
    uint32_t const* const custom_mask_row_ptr;
    int64_t const custom_mask_row_batch_stride, custom_mask_row_head_stride;
    int64_t const custom_mask_row_q_stride, custom_mask_row_k_stride;
    int const bidb, bidh;

    CUTLASS_DEVICE
    Mask(const int thread_idx, const int seqlen_q, const int seqlen_k,
         const int window_size_left, const int window_size_right, const int sink_token_length,
         cutlass::FastDivmod const &attention_chunk_divmod,
         cutlass::FastDivmod const &qhead_per_khead_divmod,
         uint32_t const* const custom_mask_ptr = nullptr,
         int64_t const custom_mask_batch_stride = 0,
         int64_t const custom_mask_head_stride = 0,
         int64_t const custom_mask_q_stride = 0,
         int64_t const custom_mask_k_stride = 0,
         uint8_t const* const custom_mask_tile_ptr = nullptr,
         int64_t const custom_mask_tile_batch_stride = 0,
         int64_t const custom_mask_tile_head_stride = 0,
         int64_t const custom_mask_tile_q_stride = 0,
         int64_t const custom_mask_tile_k_stride = 0,
         uint32_t const* const custom_mask_row_ptr = nullptr,
         int64_t const custom_mask_row_batch_stride = 0,
         int64_t const custom_mask_row_head_stride = 0,
         int64_t const custom_mask_row_q_stride = 0,
         int64_t const custom_mask_row_k_stride = 0,
         int const bidb = 0, int const bidh = 0)
        : thread_idx(thread_idx)
        , seqlen_q(seqlen_q)
        , seqlen_k(seqlen_k)
        , window_size_left(window_size_left)
        , window_size_right(window_size_right)
        , sink_token_length(sink_token_length)
        , attention_chunk_divmod(attention_chunk_divmod)
        , qhead_per_khead_divmod(qhead_per_khead_divmod)
        , custom_mask_ptr(custom_mask_ptr)
        , custom_mask_batch_stride(custom_mask_batch_stride)
        , custom_mask_head_stride(custom_mask_head_stride)
        , custom_mask_q_stride(custom_mask_q_stride)
        , custom_mask_k_stride(custom_mask_k_stride)
        , custom_mask_tile_ptr(custom_mask_tile_ptr)
        , custom_mask_tile_batch_stride(custom_mask_tile_batch_stride)
        , custom_mask_tile_head_stride(custom_mask_tile_head_stride)
        , custom_mask_tile_q_stride(custom_mask_tile_q_stride)
        , custom_mask_tile_k_stride(custom_mask_tile_k_stride)
        , custom_mask_row_ptr(custom_mask_row_ptr)
        , custom_mask_row_batch_stride(custom_mask_row_batch_stride)
        , custom_mask_row_head_stride(custom_mask_row_head_stride)
        , custom_mask_row_q_stride(custom_mask_row_q_stride)
        , custom_mask_row_k_stride(custom_mask_row_k_stride)
        , bidb(bidb)
        , bidh(bidh)
    {
    };

    CUTLASS_DEVICE
    uint8_t tile_state(const int m_block, const int n_block) const {
        if (custom_mask_tile_ptr == nullptr) { return uint8_t{1}; }
        return custom_mask_tile_ptr[
            int64_t(bidb) * custom_mask_tile_batch_stride
          + int64_t(bidh) * custom_mask_tile_head_stride
          + int64_t(m_block) * custom_mask_tile_q_stride
          + int64_t(n_block) * custom_mask_tile_k_stride];
    }

    CUTLASS_DEVICE
    uint32_t row_descriptor(const int q_idx, const int n_block) const {
        if (custom_mask_row_ptr == nullptr) { return 0xfffffffdu; }
        return custom_mask_row_ptr[
            int64_t(bidb) * custom_mask_row_batch_stride
          + int64_t(bidh) * custom_mask_row_head_stride
          + int64_t(q_idx) * custom_mask_row_q_stride
          + int64_t(n_block) * custom_mask_row_k_stride];
    }

    template <bool Seqlenk_mask=false, bool Causal_mask=false, bool Local_mask=false,
        typename Engine, typename Layout>
    CUTLASS_DEVICE
    void apply(Tensor<Engine, Layout> &tSrS, const int m_block, const int n_block) const {
        static_assert(!(Causal_mask && Local_mask), "Cannot be both causal and local");
        static_assert(Layout::rank == 3, "Only support 3D Tensor");
        uint8_t const custom_tile_state = tile_state(m_block, n_block);
        if constexpr (!Causal_mask && !Local_mask) {
            bool const needs_tail_mask = Seqlenk_mask && (n_block + 1) * kBlockN > seqlen_k;
            if (!needs_tail_mask && (custom_mask_ptr == nullptr || custom_tile_state == uint8_t{2})) {
                return;
            }
        }

        auto thread_mma = TiledMma{}.get_thread_slice(thread_idx);
        auto thread0_mma = TiledMma{}.get_thread_slice(_0{});

        static constexpr int Row = !SwapAB ? 0 : 1, Col = !SwapAB ? 1 : 0;

        Tensor cS = cute::make_identity_tensor(Shape<Int<!SwapAB ? kBlockM : kBlockN>, Int<!SwapAB ? kBlockN : kBlockM>>{});
        Tensor tScS = thread_mma.partition_C(cS);
        Tensor tSrS_rowcol = make_tensor(tSrS.data(), flash::convert_layout_acc_rowcol</*Transposed=*/SwapAB>(tSrS.layout()));
        Tensor tScS_rowcol = make_tensor(tScS.data(), flash::convert_layout_acc_rowcol</*Transposed=*/SwapAB>(tScS.layout()));
        Tensor t0ScS = thread0_mma.partition_C(cS);
        Tensor t0ScS_rowcol = make_tensor(t0ScS.data(), flash::convert_layout_acc_rowcol</*Transposed=*/SwapAB>(t0ScS.layout()));
        // We want to use the col indices of thread0 to compare, since that is known at compile time.
        // So we subtract the limit by the first col index of this thread (get<Col>(tScS_rowcol(_0{}, _0{})))
        int const thread_col_offset = get<Col>(tScS_rowcol(_0{}, _0{}));
        int const seqlenk_col_limit = seqlen_k - n_block * kBlockN - thread_col_offset;
        if constexpr (!Causal_mask && !Local_mask) {
            if constexpr (Seqlenk_mask) {  // Just masking based on col
                #pragma unroll
                for (int n = 0; n < size<1>(tSrS_rowcol); ++n) {
                    if (int(get<Col>(t0ScS_rowcol(_0{}, n))) >= seqlenk_col_limit) {
                        #pragma unroll
                        for (int m = 0; m < size<0>(tSrS_rowcol); ++m) { tSrS_rowcol(m, n) = -INFINITY; }
                    }
                }
            }
        } else {  // mask based on both row and col
            if constexpr (!SwapAB) {
                // If PackGQA, we split the work of compute divmod among threads in the same row
                static constexpr int kMmaThreadsPerRow = size<0, 0>(typename TiledMma::AtomLayoutC_TV{});
                static_assert(cutlass::NumThreadsPerWarp % kMmaThreadsPerRow == 0);
                static_assert(!PackGQA || CUTE_STATIC_V(size<0>(tSrS_rowcol)) <= kMmaThreadsPerRow);
                int mma_m_idx;
                // Might get OOB but it's ok since we'll check it later
                if constexpr (PackGQA) {
                    mma_m_idx = qhead_per_khead_divmod.divide(m_block * kBlockM + get<Row>(tScS_rowcol(thread_idx % kMmaThreadsPerRow, _0{})));
                }
                int const causal_row_offset = 1 + seqlen_k - n_block * kBlockN - seqlen_q - thread_col_offset;
                if constexpr (Causal_mask) {
                    #pragma unroll
                    for (int m = 0; m < size<0>(tSrS_rowcol); ++m) {
                        int const row_idx = !PackGQA
                            ? get<Row>(tScS_rowcol(m, _0{})) + m_block * kBlockM
                            :  __shfl_sync(0xffffffff, mma_m_idx, m % kMmaThreadsPerRow, kMmaThreadsPerRow);
                        int const col_limit_right = !Seqlenk_mask
                            ? row_idx + causal_row_offset
                            : __viaddmin_s32(row_idx, causal_row_offset, seqlenk_col_limit);
                        #pragma unroll
                        for (int n = 0; n < size<1>(tSrS_rowcol); ++n) {
                            if (int(get<Col>(t0ScS_rowcol(_0{}, n))) >= col_limit_right) { tSrS_rowcol(m, n) = -INFINITY; }
                        }
                    }
                } else {
                    int const local_row_offset_right = causal_row_offset + window_size_right;
                    int const local_row_offset_left = causal_row_offset - 1 - window_size_left;
                    int const col_limit_sink = sink_token_length - n_block * kBlockN;  // TODO: subtract thread_col_offset?
                    #pragma unroll
                    for (int m = 0; m < size<0>(tSrS_rowcol); ++m) {
                        int const row_idx = !PackGQA
                            ? get<Row>(tScS_rowcol(m, _0{})) + m_block * kBlockM
                            :  __shfl_sync(0xffffffff, mma_m_idx, m % kMmaThreadsPerRow, kMmaThreadsPerRow);
                        int col_limit_right = !Seqlenk_mask
                            ? row_idx + local_row_offset_right
                            : __viaddmin_s32(row_idx, local_row_offset_right, seqlenk_col_limit);
                        int col_limit_left = row_idx + local_row_offset_left;
                        if (attention_chunk_divmod.divisor > 0) {
                            int col_limit_left_chunk = flash::round_down(attention_chunk_divmod, row_idx + seqlen_k - seqlen_q) - n_block * kBlockN - thread_col_offset;
                            col_limit_left = std::max(col_limit_left, col_limit_left_chunk);
                            col_limit_right = std::min(col_limit_right, col_limit_left_chunk + attention_chunk_divmod.divisor);
                        }
                        #pragma unroll
                        for (int n = 0; n < size<1>(tSrS_rowcol); ++n) {
                            int const col_idx = int(get<Col>(t0ScS_rowcol(m, n)));
                            if (col_idx >= col_limit_right || (col_idx < col_limit_left && col_idx >= col_limit_sink)) { tSrS_rowcol(m, n) = -INFINITY; }
                        }
                    }
                }
            } else {
                // TODO: backward does not support attention_chunk yet
                int const thread_row_offset = get<Row>(tScS_rowcol(_0{}, _0{}));
                int const causal_row_offset = seqlenk_col_limit - seqlen_q + m_block * kBlockM + thread_row_offset;
                if constexpr (Causal_mask) {
                    #pragma unroll
                    for (int n = 0; n < size<1>(tSrS_rowcol); ++n) {
                        int const col0 = int(get<Col>(t0ScS_rowcol(_0{}, n)));
                        // If col0 is beyond the column limit, we want to mask out the entire column, by setting
                        // row limit to be kBlockM.
                        int const row_limit_top = col0 >= seqlenk_col_limit ? kBlockM : col0 - causal_row_offset;
                        #pragma unroll
                        for (int m = 0; m < size<0>(tSrS_rowcol); ++m) {
                            if (int(get<Row>(t0ScS_rowcol(m, _0{}))) < row_limit_top) { tSrS_rowcol(m, n) = -INFINITY; }
                        }
                    }
                } else {
                    int const col_limit_sink = sink_token_length - n_block * kBlockN - thread_col_offset;
                    #pragma unroll
                    for (int n = 0; n < size<1>(tSrS_rowcol); ++n) {
                        int const col0 = int(get<Col>(t0ScS_rowcol(_0{}, n)));
                        // If col0 is beyond the column limit, we want to mask out the entire column, by setting
                        // row limit to be kBlockM.
                        int const row_limit_top = col0 >= seqlenk_col_limit ? kBlockM : col0 - causal_row_offset - window_size_right;
                        int const row_limit_bot = col0 < col_limit_sink ? kBlockM : col0 - causal_row_offset + window_size_left;
                        #pragma unroll
                        for (int m = 0; m < size<0>(tSrS_rowcol); ++m) {
                            int const row_idx = int(get<Row>(t0ScS_rowcol(m, _0{})));
                            if (row_idx < row_limit_top || row_idx > row_limit_bot) { tSrS_rowcol(m, n) = -INFINITY; }
                        }
                    }
                }
            }
        }

        // Apply the dense user mask after structural/tail masking. The direct
        // coordinate lookup works for both the forward accumulator layout and
        // the transposed score layout used by the backward kernel.
        if (custom_mask_ptr != nullptr && custom_tile_state != uint8_t{2}) {
            int64_t const bh_offset = int64_t(bidb) * custom_mask_batch_stride
                                    + int64_t(bidh) * custom_mask_head_stride;
            #pragma unroll
            for (int m = 0; m < size<0>(tSrS_rowcol); ++m) {
                int const q_idx = m_block * kBlockM + int(get<Row>(tScS_rowcol(m, _0{})));
                if constexpr (!SwapAB) {
                    bool const row_in_bounds = q_idx >= 0 && q_idx < seqlen_q;
                    uint32_t const row_desc =
                        row_in_bounds ? row_descriptor(q_idx, n_block) : 0xffffffffu;
                    if (row_desc == 0xffffffffu) {  // empty
                        #pragma unroll
                        for (int n = 0; n < size<1>(tSrS_rowcol); ++n) {
                            tSrS_rowcol(m, n) = -INFINITY;
                        }
                    } else if (row_desc != 0xfffffffeu && row_desc != 0xfffffffdu) {
                        // Proven single contiguous interval, packed as global
                        // uint16 [lo, hi). Structural tail masking has already
                        // run, so two comparisons are sufficient and exact.
                        int const lo = int(row_desc & 0xffffu);
                        int const hi = int(row_desc >> 16);
                        int const tile_begin = n_block * kBlockN;
                        #pragma unroll
                        for (int n = 0; n < size<1>(tSrS_rowcol); ++n) {
                            int const k_idx = tile_begin
                                            + int(get<Col>(tScS_rowcol(_0{}, n)));
                            if (k_idx < lo || k_idx >= hi) {
                                tSrS_rowcol(m, n) = -INFINITY;
                            }
                        }
                    } else if (row_desc == 0xfffffffdu) {  // arbitrary fallback
                        // Forward's logical columns are ordered. Cache the
                        // packed word on first use so all scores from the same
                        // 32-key group reuse one global load. Packed tail bits
                        // are zero, so the word value also predicates K tails.
                        int cached_word_idx = -1;
                        uint32_t cached_word = 0u;
                        int const mask_words = (seqlen_k + 31) >> 5;
                        #pragma unroll
                        for (int n = 0; n < size<1>(tSrS_rowcol); ++n) {
                            int const k_idx = n_block * kBlockN + int(get<Col>(tScS_rowcol(_0{}, n)));
                            int const word_idx = k_idx >> 5;
                            if (word_idx != cached_word_idx) {
                                cached_word = word_idx >= mask_words
                                    ? 0u
                                    : custom_mask_ptr[
                                        bh_offset + int64_t(q_idx) * custom_mask_q_stride
                                                  + int64_t(word_idx) * custom_mask_k_stride];
                                cached_word_idx = word_idx;
                            }
                            if (cached_word == 0u) {
                                tSrS_rowcol(m, n) = -INFINITY;
                            } else if (cached_word != 0xffffffffu
                                       && ((cached_word >> (k_idx & 31)) & 1u) == 0u) {
                                tSrS_rowcol(m, n) = -INFINITY;
                            }
                        }
                    }
                } else {
                    #pragma unroll
                    for (int n = 0; n < size<1>(tSrS_rowcol); ++n) {
                        int const k_idx = n_block * kBlockN
                                        + int(get<Col>(tScS_rowcol(_0{}, n)));
                        bool const in_bounds = q_idx >= 0 && q_idx < seqlen_q
                                            && k_idx >= 0 && k_idx < seqlen_k;
                        uint32_t const packed_word = !in_bounds ? 0u : custom_mask_ptr[
                            bh_offset + int64_t(q_idx) * custom_mask_q_stride
                                      + int64_t(k_idx >> 5) * custom_mask_k_stride];
                        bool const keep = in_bounds
                            && ((packed_word >> (k_idx & 31)) & 1u);
                        if (!keep) { tSrS_rowcol(m, n) = -INFINITY; }
                    }
                }
            }
        }
    };

};

} // namespace flash
