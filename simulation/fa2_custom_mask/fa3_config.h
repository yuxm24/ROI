#pragma once

#ifdef __CUDACC__
__host__ __device__
#endif
inline int fa2mask_fwd_block_m(int seqlen_q) {
  return seqlen_q <= 160 ? 64 : 128;
}
