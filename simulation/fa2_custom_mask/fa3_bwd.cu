#include "fa3_launch.h"

#include <cutlass/numeric_types.h>

#include "vendor/flash_attn_hopper/flash_bwd_launch_template.h"

namespace {

template <typename Element>
void launch_backward_typed(Flash_bwd_params& params, cudaStream_t stream) {
  run_flash_bwd<
      90, 128, 64, 128, Element,
      false, false, false, false, false, false,
      2, 2, true, false, true, 2, 1, 2, 1, false>(params, stream);
}

}  // namespace

void fa2mask_fa3_launch_backward(Flash_bwd_params& params, cudaStream_t stream) {
  if (params.is_bf16) {
    launch_backward_typed<cutlass::bfloat16_t>(params, stream);
  } else {
    launch_backward_typed<cutlass::half_t>(params, stream);
  }
}
