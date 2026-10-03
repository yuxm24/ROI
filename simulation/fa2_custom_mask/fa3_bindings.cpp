#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

#include <algorithm>
#include <cstdint>
#include <vector>

#include "fa3_launch.h"
#include "fa3_config.h"

namespace {

void check_common(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v) {
  TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "q/k/v must be CUDA tensors");
  TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4,
              "q/k/v must have shape [B,H,L,D]");
  TORCH_CHECK(q.is_contiguous() && k.is_contiguous(),
              "q/k must be contiguous");
  // Wan's V projection is a dense (B,L,H,D)->(B,H,L,D) transpose view.
  // FA3 already consumes the explicit batch/head/row strides below, and its
  // selected row-major V specialization only requires contiguous D.
  TORCH_CHECK(v.stride(3) == 1,
              "v must be contiguous in its head dimension");
  auto const dtype = q.scalar_type();
  TORCH_CHECK(dtype == at::kHalf || dtype == at::kBFloat16,
              "q/k/v must be FP16 or BF16");
  TORCH_CHECK(k.scalar_type() == dtype && v.scalar_type() == dtype,
              "q/k/v dtype mismatch");
  TORCH_CHECK(q.size(0) == k.size(0) && q.size(0) == v.size(0), "batch mismatch");
  TORCH_CHECK(q.size(1) == k.size(1) && q.size(1) == v.size(1), "head mismatch");
  TORCH_CHECK(k.size(2) == v.size(2), "K/V length mismatch");
  TORCH_CHECK(q.size(3) == 128 && k.size(3) == 128 && v.size(3) == 128,
              "head_dim must be 128");
  TORCH_CHECK(q.get_device() == k.get_device() && q.get_device() == v.get_device(),
              "q/k/v device mismatch");
  auto* props = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(props->major == 9, "FA3 candidate requires an SM90 GPU");
}

void set_common_params(
    Flash_fwd_params& params,
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& out,
    const torch::Tensor& lse,
    const torch::Tensor& packed_mask,
    const torch::Tensor& tile_state,
    const torch::Tensor& row_state,
    float sm_scale,
    bool has_mask) {
  params = {};
  const int b = static_cast<int>(q.size(0));
  const int h = static_cast<int>(q.size(1));
  const int lq = static_cast<int>(q.size(2));
  const int lkv = static_cast<int>(k.size(2));

  params.q_ptr = q.data_ptr();
  params.k_ptr = k.data_ptr();
  params.v_ptr = v.data_ptr();
  params.o_ptr = out.data_ptr();
  params.q_batch_stride = q.stride(0);
  params.k_batch_stride = k.stride(0);
  params.v_batch_stride = v.stride(0);
  params.o_batch_stride = out.stride(0);
  params.q_head_stride = q.stride(1);
  params.k_head_stride = k.stride(1);
  params.v_head_stride = v.stride(1);
  params.o_head_stride = out.stride(1);
  params.q_row_stride = q.stride(2);
  params.k_row_stride = k.stride(2);
  params.v_row_stride = v.stride(2);
  params.o_row_stride = out.stride(2);
  params.v_dim_stride = v.stride(3);
  params.softmax_lse_ptr = lse.data_ptr();

  params.b = b;
  params.b_k = b;
  params.h = h;
  params.h_k = h;
  params.seqlen_q = lq;
  params.seqlen_k = lkv;
  int const fwd_block_m = fa2mask_fwd_block_m(lq);
  params.seqlen_q_rounded = (lq + fwd_block_m - 1) / fwd_block_m * fwd_block_m;
  params.seqlen_k_rounded = (lkv + 127) / 128 * 128;
  params.total_q = b * lq;
  params.total_k = b * lkv;
  params.d = 128;
  params.d_rounded = 128;
  params.dv = 128;
  params.dv_rounded = 128;
  params.scale_softmax = sm_scale;
  params.p_dropout = 1.0f;
  params.rp_dropout = 1.0f;
  params.is_bf16 = q.scalar_type() == at::kBFloat16;
  params.window_size_left = lkv - 1;
  params.window_size_right = lq - 1;
  params.num_splits = 1;
  params.page_size = 1;
  params.arch = 90;
  params.num_sm = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;

  if (has_mask) {
    params.custom_mask_ptr = reinterpret_cast<uint32_t const*>(packed_mask.data_ptr<int>());
    params.custom_mask_batch_stride = packed_mask.stride(0);
    params.custom_mask_head_stride = packed_mask.size(1) == 1 ? 0 : packed_mask.stride(1);
    params.custom_mask_q_stride = packed_mask.stride(2);
    params.custom_mask_k_stride = packed_mask.stride(3);
    params.custom_mask_tile_ptr = tile_state.data_ptr<uint8_t>();
    params.custom_mask_tile_batch_stride = tile_state.stride(0);
    params.custom_mask_tile_head_stride =
        tile_state.size(1) == 1 ? 0 : tile_state.stride(1);
    params.custom_mask_tile_q_stride = tile_state.stride(2);
    params.custom_mask_tile_k_stride = tile_state.stride(3);
    if (row_state.defined() && row_state.numel() != 0) {
      params.custom_mask_row_ptr =
          reinterpret_cast<uint32_t const*>(row_state.data_ptr<int>());
      params.custom_mask_row_batch_stride = row_state.stride(0);
      params.custom_mask_row_head_stride =
          row_state.size(1) == 1 ? 0 : row_state.stride(1);
      params.custom_mask_row_q_stride = row_state.stride(2);
      params.custom_mask_row_k_stride = row_state.stride(3);
    }
  }
}

struct PreparedMask {
  torch::Tensor packed;
  torch::Tensor fwd_tiles;
  torch::Tensor bwd_tiles;
  torch::Tensor fwd_rows;
};

PreparedMask prepare_mask(const torch::Tensor& mask, int logical_heads) {
  int const packed_heads = mask.stride(1) == 0 ? 1 : logical_heads;
  int const words = static_cast<int>((mask.size(3) + 31) / 32);
  auto packed = torch::empty(
      {mask.size(0), packed_heads, mask.size(2), words},
      mask.options().dtype(torch::kInt32));
  fa2mask_fa3_pack_mask(
      mask.data_ptr<bool>(), reinterpret_cast<uint32_t*>(packed.data_ptr<int>()),
      static_cast<int>(mask.size(0)), packed_heads, static_cast<int>(mask.size(2)),
      static_cast<int>(mask.size(3)), mask.stride(0), mask.stride(1),
      mask.stride(2), mask.stride(3), at::cuda::getCurrentCUDAStream().stream());
  int const lq = static_cast<int>(mask.size(2));
  int const lkv = static_cast<int>(mask.size(3));
  int const tiles_n = (lkv + 127) / 128;
  auto fwd_tiles = torch::empty(
      {packed.size(0), packed.size(1),
       (lq + fa2mask_fwd_block_m(lq) - 1) / fa2mask_fwd_block_m(lq), tiles_n},
      packed.options().dtype(torch::kUInt8));
  auto bwd_tiles = torch::empty(
      {packed.size(0), packed.size(1), (lq + 63) / 64, tiles_n},
      packed.options().dtype(torch::kUInt8));
  auto fwd_rows = torch::empty(
      {packed.size(0), packed.size(1), lq, tiles_n},
      packed.options().dtype(torch::kInt32));
  fa2mask_fa3_build_metadata(
      reinterpret_cast<uint32_t const*>(packed.data_ptr<int>()),
      fwd_tiles.data_ptr<uint8_t>(), bwd_tiles.data_ptr<uint8_t>(),
      reinterpret_cast<uint32_t*>(fwd_rows.data_ptr<int>()),
      static_cast<int>(packed.size(0)), packed_heads, lq, lkv, words,
      at::cuda::getCurrentCUDAStream().stream());
  return {packed, fwd_tiles, bwd_tiles, fwd_rows};
}

}  // namespace

std::vector<torch::Tensor> fa2mask_fa3_forward(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor mask,
    double sm_scale,
  bool has_mask) {
  c10::cuda::CUDAGuard guard(q.device());
  check_common(q, k, v);
  if (has_mask) {
    TORCH_CHECK(mask.is_cuda() && mask.scalar_type() == at::kBool,
                "mask must be a CUDA bool tensor");
    TORCH_CHECK(mask.get_device() == q.get_device(), "mask device mismatch");
    TORCH_CHECK(mask.dim() == 4 && mask.size(0) == q.size(0) &&
                    mask.size(1) == q.size(1) && mask.size(2) == q.size(2) &&
                    mask.size(3) == k.size(2),
                "mask must have shape [B,H,LQ,LKV]");
  }
  auto out = torch::empty_like(q);
  auto lse = torch::empty(
      {q.size(0), q.size(1), q.size(2)}, q.options().dtype(torch::kFloat32));
  auto empty_packed = torch::empty({0}, q.options().dtype(torch::kInt32));
  auto empty_tiles = torch::empty({0}, q.options().dtype(torch::kUInt8));
  PreparedMask prepared{empty_packed, empty_tiles, empty_tiles, empty_tiles};
  if (has_mask) {
    prepared = prepare_mask(mask, static_cast<int>(q.size(1)));
  }
  Flash_fwd_params params;
  set_common_params(params, q, k, v, out, lse, prepared.packed, prepared.fwd_tiles,
                    prepared.fwd_rows, static_cast<float>(sm_scale), has_mask);
  fa2mask_fa3_launch_forward(params, at::cuda::getCurrentCUDAStream().stream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {out, lse, prepared.packed, prepared.bwd_tiles,
          prepared.fwd_tiles, prepared.fwd_rows};
}

std::vector<torch::Tensor> fa2mask_fa3_forward_prepared(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor packed_mask,
    torch::Tensor fwd_tile_state,
    torch::Tensor fwd_row_state,
    double sm_scale) {
  c10::cuda::CUDAGuard guard(q.device());
  check_common(q, k, v);
  int const lq = static_cast<int>(q.size(2));
  int const lkv = static_cast<int>(k.size(2));
  int const fwd_block_m = fa2mask_fwd_block_m(lq);
  int const tiles_n = (lkv + 127) / 128;
  TORCH_CHECK(packed_mask.is_cuda() && packed_mask.scalar_type() == at::kInt &&
                  packed_mask.is_contiguous() && packed_mask.get_device() == q.get_device(),
              "packed mask must be contiguous CUDA int32 on the Q device");
  TORCH_CHECK(packed_mask.dim() == 4 && packed_mask.size(0) == q.size(0) &&
                  (packed_mask.size(1) == 1 || packed_mask.size(1) == q.size(1)) &&
                  packed_mask.size(2) == lq && packed_mask.size(3) == (lkv + 31) / 32,
              "packed mask shape mismatch");
  TORCH_CHECK(fwd_tile_state.is_cuda() &&
                  fwd_tile_state.scalar_type() == at::kByte &&
                  fwd_tile_state.is_contiguous() &&
                  fwd_tile_state.get_device() == q.get_device(),
              "forward tile state must be contiguous CUDA uint8 on the Q device");
  TORCH_CHECK(fwd_tile_state.dim() == 4 &&
                  fwd_tile_state.size(0) == q.size(0) &&
                  fwd_tile_state.size(1) == packed_mask.size(1) &&
                  fwd_tile_state.size(2) == (lq + fwd_block_m - 1) / fwd_block_m &&
                  fwd_tile_state.size(3) == tiles_n,
              "forward tile-state shape mismatch");
  TORCH_CHECK(fwd_row_state.is_cuda() &&
                  fwd_row_state.scalar_type() == at::kInt &&
                  fwd_row_state.is_contiguous() &&
                  fwd_row_state.get_device() == q.get_device(),
              "forward row state must be contiguous CUDA int32 on the Q device");
  TORCH_CHECK(fwd_row_state.dim() == 4 &&
                  fwd_row_state.size(0) == q.size(0) &&
                  fwd_row_state.size(1) == packed_mask.size(1) &&
                  fwd_row_state.size(2) == lq &&
                  fwd_row_state.size(3) == tiles_n,
              "forward row-state shape mismatch");

  auto out = torch::empty_like(q);
  auto lse = torch::empty(
      {q.size(0), q.size(1), q.size(2)}, q.options().dtype(torch::kFloat32));
  Flash_fwd_params params;
  set_common_params(params, q, k, v, out, lse, packed_mask, fwd_tile_state,
                    fwd_row_state, static_cast<float>(sm_scale), true);
  fa2mask_fa3_launch_forward(params, at::cuda::getCurrentCUDAStream().stream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {out, lse};
}

std::vector<torch::Tensor> fa2mask_fa3_backward(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor out,
    torch::Tensor dout,
    torch::Tensor packed_mask,
    torch::Tensor bwd_tile_state,
    torch::Tensor lse,
    double sm_scale,
  bool has_mask) {
  c10::cuda::CUDAGuard guard(q.device());
  check_common(q, k, v);
  if (has_mask) {
    TORCH_CHECK(packed_mask.is_cuda() && packed_mask.scalar_type() == at::kInt,
                "packed mask must be CUDA int32");
    TORCH_CHECK(packed_mask.dim() == 4 && packed_mask.size(0) == q.size(0) &&
                    (packed_mask.size(1) == 1 || packed_mask.size(1) == q.size(1)) &&
                    packed_mask.size(2) == q.size(2) &&
                    packed_mask.size(3) == (k.size(2) + 31) / 32,
                "packed mask shape mismatch");
    TORCH_CHECK(bwd_tile_state.is_cuda() &&
                    bwd_tile_state.scalar_type() == at::kByte &&
                    bwd_tile_state.is_contiguous(),
                "backward tile state must be contiguous CUDA uint8");
    TORCH_CHECK(
        bwd_tile_state.dim() == 4 && bwd_tile_state.size(0) == q.size(0) &&
            bwd_tile_state.size(1) == packed_mask.size(1) &&
            bwd_tile_state.size(2) == (q.size(2) + 63) / 64 &&
            bwd_tile_state.size(3) == (k.size(2) + 127) / 128,
        "backward tile-state shape mismatch");
  }
  TORCH_CHECK(out.is_contiguous() && dout.is_contiguous(), "out/dout must be contiguous");
  TORCH_CHECK(out.sizes() == q.sizes() && dout.sizes() == q.sizes(),
              "out/dout shape mismatch");
  TORCH_CHECK(out.scalar_type() == q.scalar_type() &&
                  dout.scalar_type() == q.scalar_type(),
              "out/dout dtype must match q");
  TORCH_CHECK(lse.is_cuda() && lse.scalar_type() == at::kFloat && lse.is_contiguous(),
              "lse must be contiguous CUDA FP32");
  TORCH_CHECK(lse.size(0) == q.size(0) && lse.size(1) == q.size(1) &&
                  lse.size(2) == q.size(2),
              "lse shape mismatch");

  const int64_t b = q.size(0);
  const int64_t h = q.size(1);
  const int64_t lq = q.size(2);
  // The selected FA3 backward uses kBlockM=64 (not the forward's 128).
  // This stride must cover every 64-row tile.
  const int64_t lq_rounded = (lq + 63) / 64 * 64;
  auto dq = torch::empty_like(q);
  auto dk = torch::empty_like(k);
  auto dv = torch::empty_like(v);
  auto softmax_d = torch::empty(
      {b, h, lq_rounded}, q.options().dtype(torch::kFloat32));
  auto lse_log2 = torch::empty(
      {b, h, lq_rounded}, q.options().dtype(torch::kFloat32));
  auto dq_accum = torch::empty(
      {b, h, lq_rounded * 128}, q.options().dtype(torch::kFloat32));
  auto dq_semaphore = torch::empty(
      {(lq + 63) / 64, b, h}, q.options().dtype(torch::kInt32));
  torch::Tensor empty_row_state;

  Flash_bwd_params params;
  set_common_params(params, q, k, v, out, lse, packed_mask, bwd_tile_state,
                    empty_row_state, static_cast<float>(sm_scale), has_mask);
  params.seqlen_q_rounded = static_cast<int>(lq_rounded);
  params.seqlen_k_rounded = static_cast<int>((k.size(2) + 127) / 128 * 128);
  params.do_ptr = dout.data_ptr();
  params.dq_ptr = dq.data_ptr();
  params.dk_ptr = dk.data_ptr();
  params.dv_ptr = dv.data_ptr();
  params.do_batch_stride = dout.stride(0);
  params.dq_batch_stride = dq.stride(0);
  params.dk_batch_stride = dk.stride(0);
  params.dv_batch_stride = dv.stride(0);
  params.do_head_stride = dout.stride(1);
  params.dq_head_stride = dq.stride(1);
  params.dk_head_stride = dk.stride(1);
  params.dv_head_stride = dv.stride(1);
  params.do_row_stride = dout.stride(2);
  params.dq_row_stride = dq.stride(2);
  params.dk_row_stride = dk.stride(2);
  params.dv_row_stride = dv.stride(2);
  params.dq_accum_ptr = dq_accum.data_ptr();
  params.dsoftmax_sum = softmax_d.data_ptr();
  params.softmax_lse_log2_ptr = lse_log2.data_ptr();
  params.dq_semaphore = dq_semaphore.data_ptr<int>();
  params.deterministic = false;

  fa2mask_fa3_launch_backward(params, at::cuda::getCurrentCUDAStream().stream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {dq, dk, dv};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &fa2mask_fa3_forward, "fa2mask FA3 forward");
  m.def("forward_prepared", &fa2mask_fa3_forward_prepared,
        "fa2mask FA3 forward with cached mask metadata");
  m.def("backward", &fa2mask_fa3_backward, "fa2mask FA3 backward");
}
