"""Optional Triton kernels for bandwidth-bound Zerith VA primitives."""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - CPU/minimal deployments use fallback
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _rope_fwd_kernel(
        x,
        cosine,
        sine,
        output,
        n_elements,
        heads: tl.constexpr,
        length: tl.constexpr,
        head_dim: tl.constexpr,
        pairs: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        pair_offsets = offsets // 2
        pair_index = pair_offsets % pairs
        sequence_pair = pair_offsets // pairs
        batch_index = sequence_pair // (heads * length)
        token_index = sequence_pair % length
        frequency_offset = (batch_index * length + token_index) * pairs + pair_index
        x0 = tl.load(x + pair_offsets * 2, mask=mask, other=0.0).to(tl.float32)
        x1 = tl.load(x + pair_offsets * 2 + 1, mask=mask, other=0.0).to(tl.float32)
        c = tl.load(cosine + frequency_offset, mask=mask).to(tl.float32)
        s = tl.load(sine + frequency_offset, mask=mask).to(tl.float32)
        even = offsets % 2 == 0
        y = tl.where(even, x0 * c - x1 * s, x1 * c + x0 * s)
        tl.store(output + offsets, y, mask=mask)


    @triton.jit
    def _rope_bwd_kernel(
        grad_output,
        cosine,
        sine,
        grad_input,
        n_elements,
        heads: tl.constexpr,
        length: tl.constexpr,
        head_dim: tl.constexpr,
        pairs: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        pair_offsets = offsets // 2
        pair_index = pair_offsets % pairs
        sequence_pair = pair_offsets // pairs
        batch_index = sequence_pair // (heads * length)
        token_index = sequence_pair % length
        frequency_offset = (batch_index * length + token_index) * pairs + pair_index
        g0 = tl.load(
            grad_output + pair_offsets * 2, mask=mask, other=0.0
        ).to(tl.float32)
        g1 = tl.load(
            grad_output + pair_offsets * 2 + 1, mask=mask, other=0.0
        ).to(tl.float32)
        c = tl.load(cosine + frequency_offset, mask=mask).to(tl.float32)
        s = tl.load(sine + frequency_offset, mask=mask).to(tl.float32)
        even = offsets % 2 == 0
        dx = tl.where(even, g0 * c + g1 * s, g1 * c - g0 * s)
        tl.store(grad_input + offsets, dx, mask=mask)


    @triton.jit
    def _masked_smooth_l1_forward_kernel(
        predicted,
        target,
        valid,
        partial,
        rows,
        row_width,
        spatial,
        head_dim: tl.constexpr,
        beta,
        scale,
        BLOCK_SIZE: tl.constexpr,
    ):
        row = tl.program_id(0)
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = (row < rows) & (offsets < row_width)
        is_valid = tl.load(valid + row, mask=row < rows, other=0)
        load_mask = mask & is_valid
        sample_time = row // spatial
        spatial_index = row % spatial
        head = offsets // head_dim
        feature = offsets % head_dim
        tensor_offsets = (
            sample_time * row_width * spatial
            + head * spatial * head_dim
            + spatial_index * head_dim
            + feature
        )
        inv_scale = 1.0 / tl.load(scale).to(tl.float32)
        p = tl.load(
            predicted + tensor_offsets, mask=load_mask, other=0.0
        ).to(tl.float32) * inv_scale
        t = tl.load(
            target + tensor_offsets, mask=load_mask, other=0.0
        ).to(tl.float32) * inv_scale
        difference = tl.abs(p - t)
        loss = tl.where(
            difference < beta,
            0.5 * difference * difference / beta,
            difference - 0.5 * beta,
        )
        loss = tl.where(mask & is_valid, loss, 0.0)
        tl.store(partial + row, tl.sum(loss, axis=0), mask=row < rows)


    @triton.jit
    def _masked_smooth_l1_backward_kernel(
        predicted,
        target,
        valid,
        grad_input,
        rows,
        row_width,
        spatial,
        head_dim: tl.constexpr,
        beta,
        scale,
        grad_output,
        denominator,
        BLOCK_SIZE: tl.constexpr,
    ):
        row = tl.program_id(0)
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = (row < rows) & (offsets < row_width)
        is_valid = tl.load(valid + row, mask=row < rows, other=0)
        load_mask = mask & is_valid
        sample_time = row // spatial
        spatial_index = row % spatial
        head = offsets // head_dim
        feature = offsets % head_dim
        tensor_offsets = (
            sample_time * row_width * spatial
            + head * spatial * head_dim
            + spatial_index * head_dim
            + feature
        )
        inv_scale = 1.0 / tl.load(scale).to(tl.float32)
        p = tl.load(
            predicted + tensor_offsets, mask=load_mask, other=0.0
        ).to(tl.float32) * inv_scale
        t = tl.load(
            target + tensor_offsets, mask=load_mask, other=0.0
        ).to(tl.float32) * inv_scale
        difference = p - t
        derivative = tl.where(
            tl.abs(difference) < beta,
            difference / beta,
            tl.where(difference > 0.0, 1.0, -1.0),
        )
        derivative *= (
            inv_scale
            * tl.load(grad_output).to(tl.float32)
            / tl.load(denominator).to(tl.float32)
        )
        derivative = tl.where(mask & is_valid, derivative, 0.0)
        tl.store(grad_input + tensor_offsets, derivative, mask=mask)


    @triton.jit
    def _masked_square_sum_kernel(
        value,
        valid,
        partial,
        rows,
        times: tl.constexpr,
        heads: tl.constexpr,
        spatial: tl.constexpr,
        head_dim: tl.constexpr,
        value_stride_b,
        value_stride_t,
        value_stride_h,
        value_stride_s,
        value_stride_d,
        valid_stride_b,
        valid_stride_t,
        valid_stride_s,
        BLOCK_SIZE: tl.constexpr,
    ):
        row = tl.program_id(0)
        offsets = tl.arange(0, BLOCK_SIZE)
        row_width = heads * head_dim
        mask = (row < rows) & (offsets < row_width)
        spatial_index = row % spatial
        sample_time = row // spatial
        time_index = sample_time % times
        batch_index = sample_time // times
        head = offsets // head_dim
        feature = offsets % head_dim
        value_offsets = (
            batch_index * value_stride_b
            + time_index * value_stride_t
            + head * value_stride_h
            + spatial_index * value_stride_s
            + feature * value_stride_d
        )
        valid_offset = (
            batch_index * valid_stride_b
            + time_index * valid_stride_t
            + spatial_index * valid_stride_s
        )
        is_valid = tl.load(valid + valid_offset, mask=row < rows, other=0)
        load_mask = mask & is_valid
        item = tl.load(value + value_offsets, mask=load_mask, other=0.0).to(tl.float32)
        squared = tl.where(mask & is_valid, item * item, 0.0)
        tl.store(partial + row, tl.sum(squared, axis=0), mask=row < rows)


    @triton.jit
    def _kv_layer_loss_forward_kernel(
        predicted_key,
        target_key,
        predicted_value,
        target_value,
        valid,
        partial,
        rows,
        observation_denominator,
        future_denominator,
        scale,
        beta,
        times: tl.constexpr,
        heads: tl.constexpr,
        spatial: tl.constexpr,
        head_dim: tl.constexpr,
        pk_stride_b,
        pk_stride_t,
        pk_stride_h,
        pk_stride_s,
        pk_stride_d,
        tk_stride_b,
        tk_stride_t,
        tk_stride_h,
        tk_stride_s,
        tk_stride_d,
        pv_stride_b,
        pv_stride_t,
        pv_stride_h,
        pv_stride_s,
        pv_stride_d,
        tv_stride_b,
        tv_stride_t,
        tv_stride_h,
        tv_stride_s,
        tv_stride_d,
        valid_stride_b,
        valid_stride_t,
        valid_stride_s,
        BLOCK_SIZE: tl.constexpr,
    ):
        row = tl.program_id(0)
        offsets = tl.arange(0, BLOCK_SIZE)
        row_width = heads * head_dim
        mask = (row < rows) & (offsets < row_width)
        spatial_index = row % spatial
        sample_time = row // spatial
        time_index = sample_time % times
        batch_index = sample_time // times
        head = offsets // head_dim
        feature = offsets % head_dim

        pk_offset = (
            batch_index * pk_stride_b + time_index * pk_stride_t
            + head * pk_stride_h + spatial_index * pk_stride_s
            + feature * pk_stride_d
        )
        tk_offset = (
            batch_index * tk_stride_b + time_index * tk_stride_t
            + head * tk_stride_h + spatial_index * tk_stride_s
            + feature * tk_stride_d
        )
        pv_offset = (
            batch_index * pv_stride_b + time_index * pv_stride_t
            + head * pv_stride_h + spatial_index * pv_stride_s
            + feature * pv_stride_d
        )
        tv_offset = (
            batch_index * tv_stride_b + time_index * tv_stride_t
            + head * tv_stride_h + spatial_index * tv_stride_s
            + feature * tv_stride_d
        )
        valid_offset = (
            batch_index * valid_stride_b + time_index * valid_stride_t
            + spatial_index * valid_stride_s
        )
        is_valid = tl.load(valid + valid_offset, mask=row < rows, other=0)
        load_mask = mask & is_valid
        pk = tl.load(predicted_key + pk_offset, mask=load_mask, other=0.0).to(tl.float32)
        tk = tl.load(target_key + tk_offset, mask=load_mask, other=0.0).to(tl.float32)
        pv = tl.load(predicted_value + pv_offset, mask=load_mask, other=0.0).to(tl.float32)
        tv = tl.load(target_value + tv_offset, mask=load_mask, other=0.0).to(tl.float32)

        key_difference = tl.abs(pk - tk)
        inv_scale = 1.0 / tl.load(scale).to(tl.float32)
        value_difference = tl.abs(pv * inv_scale - tv * inv_scale)
        key_loss = tl.where(
            key_difference < beta,
            0.5 * key_difference * key_difference / beta,
            key_difference - 0.5 * beta,
        )
        value_loss = tl.where(
            value_difference < beta,
            0.5 * value_difference * value_difference / beta,
            value_difference - 0.5 * beta,
        )
        denominator = tl.where(
            time_index == 0,
            tl.load(observation_denominator),
            tl.load(future_denominator),
        ).to(tl.float32)
        loss = tl.where(
            mask & is_valid,
            0.25 * (key_loss + value_loss) / denominator,
            0.0,
        )
        tl.store(partial + row, tl.sum(loss, axis=0), mask=row < rows)


    @triton.jit
    def _kv_layer_loss_backward_kernel(
        predicted_key,
        target_key,
        predicted_value,
        target_value,
        valid,
        grad_key,
        grad_value,
        rows,
        observation_denominator,
        future_denominator,
        scale,
        beta,
        grad_output,
        times: tl.constexpr,
        heads: tl.constexpr,
        spatial: tl.constexpr,
        head_dim: tl.constexpr,
        pk_stride_b,
        pk_stride_t,
        pk_stride_h,
        pk_stride_s,
        pk_stride_d,
        tk_stride_b,
        tk_stride_t,
        tk_stride_h,
        tk_stride_s,
        tk_stride_d,
        pv_stride_b,
        pv_stride_t,
        pv_stride_h,
        pv_stride_s,
        pv_stride_d,
        tv_stride_b,
        tv_stride_t,
        tv_stride_h,
        tv_stride_s,
        tv_stride_d,
        valid_stride_b,
        valid_stride_t,
        valid_stride_s,
        BLOCK_SIZE: tl.constexpr,
    ):
        row = tl.program_id(0)
        offsets = tl.arange(0, BLOCK_SIZE)
        row_width = heads * head_dim
        mask = (row < rows) & (offsets < row_width)
        spatial_index = row % spatial
        sample_time = row // spatial
        time_index = sample_time % times
        batch_index = sample_time // times
        head = offsets // head_dim
        feature = offsets % head_dim

        pk_offset = (
            batch_index * pk_stride_b + time_index * pk_stride_t
            + head * pk_stride_h + spatial_index * pk_stride_s
            + feature * pk_stride_d
        )
        tk_offset = (
            batch_index * tk_stride_b + time_index * tk_stride_t
            + head * tk_stride_h + spatial_index * tk_stride_s
            + feature * tk_stride_d
        )
        pv_offset = (
            batch_index * pv_stride_b + time_index * pv_stride_t
            + head * pv_stride_h + spatial_index * pv_stride_s
            + feature * pv_stride_d
        )
        tv_offset = (
            batch_index * tv_stride_b + time_index * tv_stride_t
            + head * tv_stride_h + spatial_index * tv_stride_s
            + feature * tv_stride_d
        )
        valid_offset = (
            batch_index * valid_stride_b + time_index * valid_stride_t
            + spatial_index * valid_stride_s
        )
        is_valid = tl.load(valid + valid_offset, mask=row < rows, other=0)
        load_mask = mask & is_valid
        pk = tl.load(predicted_key + pk_offset, mask=load_mask, other=0.0).to(tl.float32)
        tk = tl.load(target_key + tk_offset, mask=load_mask, other=0.0).to(tl.float32)
        pv = tl.load(predicted_value + pv_offset, mask=load_mask, other=0.0).to(tl.float32)
        tv = tl.load(target_value + tv_offset, mask=load_mask, other=0.0).to(tl.float32)

        key_difference = pk - tk
        inv_scale = 1.0 / tl.load(scale).to(tl.float32)
        value_difference = pv * inv_scale - tv * inv_scale
        key_derivative = tl.where(
            tl.abs(key_difference) < beta,
            key_difference / beta,
            tl.where(key_difference > 0.0, 1.0, -1.0),
        )
        value_derivative = tl.where(
            tl.abs(value_difference) < beta,
            value_difference / beta,
            tl.where(value_difference > 0.0, 1.0, -1.0),
        ) * inv_scale
        denominator = tl.where(
            time_index == 0,
            tl.load(observation_denominator),
            tl.load(future_denominator),
        ).to(tl.float32)
        common = 0.25 * tl.load(grad_output).to(tl.float32) / denominator
        key_derivative = tl.where(mask & is_valid, key_derivative * common, 0.0)
        value_derivative = tl.where(mask & is_valid, value_derivative * common, 0.0)
        # Gradients are returned in canonical contiguous B,T,H,S,D order,
        # independently of the source view strides.
        grad_offset = (
            sample_time * heads * spatial * head_dim
            + head * spatial * head_dim
            + spatial_index * head_dim
            + feature
        )
        tl.store(grad_key + grad_offset, key_derivative, mask=mask)
        tl.store(grad_value + grad_offset, value_derivative, mask=mask)


class _TritonRotary(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value: torch.Tensor, cosine: torch.Tensor, sine: torch.Tensor):
        if not value.is_contiguous():
            value = value.contiguous()
        batch, heads, length, head_dim = value.shape
        if tuple(cosine.shape) != (batch, 1, length, head_dim // 2):
            raise ValueError("RoPE cosine must have shape (B,1,L,D/2)")
        if tuple(sine.shape) != tuple(cosine.shape):
            raise ValueError("RoPE sine/cosine shapes differ")
        cosine = cosine.contiguous()
        sine = sine.contiguous()
        output = torch.empty_like(value)
        n = value.numel()
        block = 256
        _rope_fwd_kernel[(triton.cdiv(n, block),)](
            value,
            cosine,
            sine,
            output,
            n,
            heads=heads,
            length=length,
            head_dim=head_dim,
            pairs=head_dim // 2,
            BLOCK_SIZE=block,
            num_warps=4,
        )
        ctx.save_for_backward(cosine, sine)
        ctx.heads = heads
        ctx.length = length
        ctx.head_dim = head_dim
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        cosine, sine = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grad_input = torch.empty_like(grad_output)
        n = grad_output.numel()
        block = 256
        _rope_bwd_kernel[(triton.cdiv(n, block),)](
            grad_output,
            cosine,
            sine,
            grad_input,
            n,
            heads=ctx.heads,
            length=ctx.length,
            head_dim=ctx.head_dim,
            pairs=ctx.head_dim // 2,
            BLOCK_SIZE=block,
            num_warps=4,
        )
        return grad_input, None, None


class _TritonMaskedSmoothL1(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        predicted: torch.Tensor,
        target: torch.Tensor,
        valid: torch.Tensor,
        beta: float,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        predicted = predicted.contiguous()
        target = target.contiguous()
        valid = valid.contiguous()
        rows = valid.numel()
        spatial = predicted.shape[3]
        head_dim = predicted.shape[4]
        row_width = predicted.shape[2] * head_dim
        if row_width > 65536:
            raise RuntimeError("masked Smooth-L1 row exceeds Triton capacity")
        block = triton.next_power_of_2(row_width)
        partial = torch.empty(rows, device=predicted.device, dtype=torch.float32)
        _masked_smooth_l1_forward_kernel[(rows,)](
            predicted,
            target,
            valid,
            partial,
            rows,
            row_width,
            spatial,
            head_dim,
            beta,
            scale,
            BLOCK_SIZE=block,
            num_warps=min(max(block // 256, 1), 8),
        )
        denominator = valid.sum(dtype=torch.float32) * row_width
        denominator = denominator.clamp_min(1.0)
        ctx.save_for_backward(predicted, target, valid, scale, denominator)
        ctx.beta = float(beta)
        ctx.row_width = row_width
        ctx.spatial = spatial
        ctx.head_dim = head_dim
        ctx.block = block
        return partial.sum() / denominator

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        predicted, target, valid, scale, denominator = ctx.saved_tensors
        rows = valid.numel()
        grad_input = torch.empty_like(predicted)
        _masked_smooth_l1_backward_kernel[(rows,)](
            predicted,
            target,
            valid,
            grad_input,
            rows,
            ctx.row_width,
            ctx.spatial,
            ctx.head_dim,
            ctx.beta,
            scale,
            grad_output,
            denominator,
            BLOCK_SIZE=ctx.block,
            num_warps=min(max(ctx.block // 256, 1), 8),
        )
        return grad_input, None, None, None, None


class _TritonKVLayerLoss(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        predicted_key: torch.Tensor,
        target_key: torch.Tensor,
        predicted_value: torch.Tensor,
        target_value: torch.Tensor,
        valid: torch.Tensor,
        beta: float,
        scale: torch.Tensor,
        observation_denominator: torch.Tensor,
        future_denominator: torch.Tensor,
    ) -> torch.Tensor:
        batch, times, heads, spatial, head_dim = predicted_key.shape
        rows = batch * times * spatial
        row_width = heads * head_dim
        if row_width > 65536:
            raise RuntimeError("K/V loss row exceeds Triton capacity")
        block = triton.next_power_of_2(row_width)
        partial = torch.empty(rows, device=predicted_key.device, dtype=torch.float32)
        _kv_layer_loss_forward_kernel[(rows,)](
            predicted_key,
            target_key,
            predicted_value,
            target_value,
            valid,
            partial,
            rows,
            observation_denominator,
            future_denominator,
            scale,
            beta,
            times=times,
            heads=heads,
            spatial=spatial,
            head_dim=head_dim,
            pk_stride_b=predicted_key.stride(0),
            pk_stride_t=predicted_key.stride(1),
            pk_stride_h=predicted_key.stride(2),
            pk_stride_s=predicted_key.stride(3),
            pk_stride_d=predicted_key.stride(4),
            tk_stride_b=target_key.stride(0),
            tk_stride_t=target_key.stride(1),
            tk_stride_h=target_key.stride(2),
            tk_stride_s=target_key.stride(3),
            tk_stride_d=target_key.stride(4),
            pv_stride_b=predicted_value.stride(0),
            pv_stride_t=predicted_value.stride(1),
            pv_stride_h=predicted_value.stride(2),
            pv_stride_s=predicted_value.stride(3),
            pv_stride_d=predicted_value.stride(4),
            tv_stride_b=target_value.stride(0),
            tv_stride_t=target_value.stride(1),
            tv_stride_h=target_value.stride(2),
            tv_stride_s=target_value.stride(3),
            tv_stride_d=target_value.stride(4),
            valid_stride_b=valid.stride(0),
            valid_stride_t=valid.stride(1),
            valid_stride_s=valid.stride(2),
            BLOCK_SIZE=block,
            num_warps=min(max(block // 256, 1), 8),
        )
        ctx.save_for_backward(
            predicted_key,
            target_key,
            predicted_value,
            target_value,
            valid,
            scale,
            observation_denominator,
            future_denominator,
        )
        ctx.beta = float(beta)
        ctx.block = block
        return partial.sum()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (
            predicted_key,
            target_key,
            predicted_value,
            target_value,
            valid,
            scale,
            observation_denominator,
            future_denominator,
        ) = ctx.saved_tensors
        batch, times, heads, spatial, head_dim = predicted_key.shape
        rows = batch * times * spatial
        grad_key = torch.empty_like(
            predicted_key, memory_format=torch.contiguous_format
        )
        grad_value = torch.empty_like(
            predicted_value, memory_format=torch.contiguous_format
        )
        _kv_layer_loss_backward_kernel[(rows,)](
            predicted_key,
            target_key,
            predicted_value,
            target_value,
            valid,
            grad_key,
            grad_value,
            rows,
            observation_denominator,
            future_denominator,
            scale,
            ctx.beta,
            grad_output,
            times=times,
            heads=heads,
            spatial=spatial,
            head_dim=head_dim,
            pk_stride_b=predicted_key.stride(0),
            pk_stride_t=predicted_key.stride(1),
            pk_stride_h=predicted_key.stride(2),
            pk_stride_s=predicted_key.stride(3),
            pk_stride_d=predicted_key.stride(4),
            tk_stride_b=target_key.stride(0),
            tk_stride_t=target_key.stride(1),
            tk_stride_h=target_key.stride(2),
            tk_stride_s=target_key.stride(3),
            tk_stride_d=target_key.stride(4),
            pv_stride_b=predicted_value.stride(0),
            pv_stride_t=predicted_value.stride(1),
            pv_stride_h=predicted_value.stride(2),
            pv_stride_s=predicted_value.stride(3),
            pv_stride_d=predicted_value.stride(4),
            tv_stride_b=target_value.stride(0),
            tv_stride_t=target_value.stride(1),
            tv_stride_h=target_value.stride(2),
            tv_stride_s=target_value.stride(3),
            tv_stride_d=target_value.stride(4),
            valid_stride_b=valid.stride(0),
            valid_stride_t=valid.stride(1),
            valid_stride_s=valid.stride(2),
            BLOCK_SIZE=ctx.block,
            num_warps=min(max(ctx.block // 256, 1), 8),
        )
        return grad_key, None, grad_value, None, None, None, None, None, None


def triton_rotary(
    value: torch.Tensor, cosine: torch.Tensor, sine: torch.Tensor
) -> torch.Tensor:
    """Apply RoPE with FP32 arithmetic and a fused backward on CUDA."""
    if (
        triton is None
        or not value.is_cuda
        or value.dtype not in (torch.float16, torch.bfloat16)
    ):
        raise RuntimeError("Triton rotary is unavailable for this tensor")
    return _TritonRotary.apply(value, cosine, sine)


def has_triton_rotary(value: torch.Tensor) -> bool:
    return (
        triton is not None
        and value.is_cuda
        and value.dtype in (torch.float16, torch.bfloat16)
    )


def triton_masked_smooth_l1(
    predicted: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    *,
    beta: float,
    scale: torch.Tensor,
) -> torch.Tensor:
    """Masked Smooth-L1 with fused scale, mask and per-token reduction."""
    if (
        triton is None
        or not predicted.is_cuda
        or predicted.dtype not in (torch.float16, torch.bfloat16)
    ):
        raise RuntimeError("Triton masked Smooth-L1 is unavailable")
    return _TritonMaskedSmoothL1.apply(
        predicted, target.detach(), valid, beta, scale
    )


def triton_masked_square_sum(
    value: torch.Tensor, valid: torch.Tensor
) -> torch.Tensor:
    """Sum valid squared K/V elements without FP32 tensor materialization."""
    if (
        triton is None
        or not value.is_cuda
        or value.dtype not in (torch.float16, torch.bfloat16)
        or value.ndim != 5
        or valid.ndim != 3
    ):
        raise RuntimeError("Triton masked square reduction is unavailable")
    batch, times, heads, spatial, head_dim = value.shape
    if tuple(valid.shape) != (batch, times, spatial):
        raise ValueError("valid mask does not match K/V tensor")
    row_width = heads * head_dim
    if row_width > 65536:
        raise RuntimeError("masked square row exceeds Triton capacity")
    rows = valid.numel()
    block = triton.next_power_of_2(row_width)
    partial = torch.empty(rows, device=value.device, dtype=torch.float32)
    _masked_square_sum_kernel[(rows,)](
        value.detach(),
        valid,
        partial,
        rows,
        times=times,
        heads=heads,
        spatial=spatial,
        head_dim=head_dim,
        value_stride_b=value.stride(0),
        value_stride_t=value.stride(1),
        value_stride_h=value.stride(2),
        value_stride_s=value.stride(3),
        value_stride_d=value.stride(4),
        valid_stride_b=valid.stride(0),
        valid_stride_t=valid.stride(1),
        valid_stride_s=valid.stride(2),
        BLOCK_SIZE=block,
        num_warps=min(max(block // 256, 1), 8),
    )
    return partial.sum()


def triton_kv_layer_loss(
    predicted_key: torch.Tensor,
    target_key: torch.Tensor,
    predicted_value: torch.Tensor,
    target_value: torch.Tensor,
    valid: torch.Tensor,
    *,
    beta: float,
    scale: torch.Tensor,
    observation_denominator: torch.Tensor,
    future_denominator: torch.Tensor,
) -> torch.Tensor:
    """Fuse the four observation/future K/V losses for one model layer."""
    tensors = (predicted_key, target_key, predicted_value, target_value)
    if (
        triton is None
        or not predicted_key.is_cuda
        or predicted_key.dtype not in (torch.float16, torch.bfloat16)
        or any(tuple(item.shape) != tuple(predicted_key.shape) for item in tensors[1:])
        or any(item.dtype != predicted_key.dtype for item in tensors[1:])
    ):
        raise RuntimeError("Triton fused K/V layer loss is unavailable")
    if predicted_key.ndim != 5 or predicted_key.shape[1] != 4:
        raise ValueError("fused K/V loss expects (B,4,H,S,D)")
    if tuple(valid.shape) != (
        predicted_key.shape[0], predicted_key.shape[1], predicted_key.shape[3]
    ):
        raise ValueError("valid mask does not match K/V tensor")
    return _TritonKVLayerLoss.apply(
        predicted_key,
        target_key.detach(),
        predicted_value,
        target_value.detach(),
        valid,
        beta,
        scale,
        observation_denominator,
        future_denominator,
    )
