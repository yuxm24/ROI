#include "fa3_launch.h"
#include "fa3_config.h"

#include <cutlass/numeric_types.h>

#include "vendor/flash_attn_hopper/flash_fwd_launch_template.h"

namespace {

template <typename Element>
void launch_forward_typed(Flash_fwd_params& params, cudaStream_t stream) {
  if (params.seqlen_q <= 160) {
    run_flash_fwd<
        90, 128, 128, 1,
        Element, Element,
        false, false, false, false, false,
        false, false, false, false, false,
        64, 128, 1, false>(params, stream);
    return;
  }
  run_flash_fwd<
      90, 128, 128, 1,
      Element, Element,
      false, false, false, false, false,
      false, false, false, false, false>(params, stream);
}

}  // namespace

void fa2mask_fa3_launch_forward(Flash_fwd_params& params, cudaStream_t stream) {
  // The dtype dispatch is host-side; each Hopper kernel remains a dedicated
  // FP16 or BF16 specialization with FP32 score/softmax accumulation.
  if (params.is_bf16) {
    launch_forward_typed<cutlass::bfloat16_t>(params, stream);
  } else {
    launch_forward_typed<cutlass::half_t>(params, stream);
  }
}
