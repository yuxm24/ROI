"""Small Wan-compatible transformer primitives with a Hopper FA3 path."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def causal_temporal_frame_mask(
    frames: int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Allow each frame query to read only itself and earlier frames."""
    if frames <= 0:
        raise ValueError("frames must be positive")
    return torch.ones(frames, frames, device=device, dtype=torch.bool).tril()


def async_tensor_assert(condition: torch.Tensor, message: str) -> None:
    """Validate a scalar tensor without synchronizing a CUDA stream."""
    if condition.numel() != 1:
        raise ValueError(
            "async tensor assertions require one scalar condition")
    if condition.device.type == "cuda" and hasattr(torch, "_assert_async"):
        torch._assert_async(condition, message)
    elif not bool(condition):
        raise ValueError(message)


def prepare_zero_padded_text(
    text_emb: torch.Tensor,
    *,
    prefix_length: int,
    logical_length: int,
) -> torch.Tensor:
    """Represent a zero-padded logical context as prefix + one zero row.

    Inputs may already be compact (``prefix_length`` rows) or retain the
    historical fully padded ``logical_length`` rows.  The returned final row
    represents the identical logical suffix; attention restores its exact
    multiplicity with a logit correction.
    """
    if text_emb.ndim != 3:
        raise ValueError("text_emb must have shape (B,L,D)")
    actual = text_emb.shape[1]
    if actual not in (prefix_length, logical_length):
        raise ValueError(
            "text_emb length must equal compact prefix or logical Wan length: "
            f"got {actual}, expected {prefix_length} or {logical_length}")
    prefix = text_emb[:, :prefix_length]
    if actual == logical_length and logical_length > prefix_length:
        suffix = text_emb[:, prefix_length:]
        async_tensor_assert(
            suffix.eq(0).all(),
            "text_emb rows beyond text_prefix_length must be exact zeros",
        )
    if logical_length == prefix_length:
        return prefix
    return torch.cat((prefix, torch.zeros_like(prefix[:, :1])), dim=1)


def padding_multiplicity_bias(
    reference: torch.Tensor,
    *,
    prefix_length: int,
    logical_length: int,
) -> Optional[torch.Tensor]:
    """Logit correction replacing repeated identical padding K/V by one row."""
    padding = logical_length - prefix_length
    if padding <= 0:
        return None
    bias = reference.new_zeros(1, 1, 1, prefix_length + 1)
    bias[..., -1] = math.log(float(padding))
    return bias


def _as_sdpa_mask(mask: Optional[torch.Tensor],
                  query: torch.Tensor) -> Optional[torch.Tensor]:
    if mask is None:
        return None
    if mask.dtype != torch.bool:
        raise TypeError("attention masks must be boolean")
    if mask.ndim == 2:
        # Input convention is token validity (B, K), not a pairwise mask.
        mask = mask[:, None, None, :]
    elif mask.ndim == 3:
        mask = mask[:, None, :, :]
    elif mask.ndim != 4:
        raise ValueError(
            f"attention mask must have rank 2, 3 or 4, got {mask.ndim}")
    return mask.to(device=query.device)


def _scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    attn_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    """Prefer the repository's mask-aware FlashAttention-3 on Hopper.

    The custom CuTe kernel implements arbitrary boolean masks for SM90,
    FP16/BF16, and 128-wide heads.  Additive biases and unsupported layouts
    retain the PyTorch dispatcher; cuDNN remains the correctness/performance
    fallback for eligible Hopper masks. Set
    ``ZVLA_REQUIRE_FLASH_ATTENTION3=1`` to reject unsupported boolean layouts
    or a missing or broken FA3 build instead of changing the kernel.
    """
    require_fa3 = os.environ.get(
        "ZVLA_REQUIRE_FLASH_ATTENTION3", "0").strip().lower() in {
            "1", "true", "yes", "on"
        }
    boolean_mask = attn_mask is not None and attn_mask.dtype == torch.bool
    use_fa3 = (
        boolean_mask
        and query.is_cuda
        and key.is_cuda
        and value.is_cuda
        and query.dtype in (torch.bfloat16, torch.float16)
        and key.dtype == query.dtype
        and value.dtype == query.dtype
        and query.ndim == key.ndim == value.ndim == 4
        and query.shape[-1] == key.shape[-1] == value.shape[-1] == 128
        and query.shape[:2] == key.shape[:2] == value.shape[:2]
        and key.shape[2] == value.shape[2]
        and torch.cuda.get_device_capability(query.device) == (9, 0)
    )
    if require_fa3 and boolean_mask and not use_fa3:
        raise RuntimeError(
            "required Hopper FlashAttention-3 received an unsupported boolean "
            "attention layout; production must not silently dispatch this "
            "call to SDPA")
    if use_fa3:
        target_mask_shape = (
            query.shape[0],
            query.shape[1],
            query.shape[2],
            key.shape[2],
        )
        broadcastable = (
            attn_mask.ndim == 4
            and all(actual in (1, expected) for actual, expected in zip(
                attn_mask.shape, target_mask_shape))
        )
        if not broadcastable and require_fa3:
            raise RuntimeError(
                "required Hopper FlashAttention-3 mask is not broadcastable "
                f"from {tuple(attn_mask.shape)} to {target_mask_shape}")
        if broadcastable:
            try:
                from fa2_custom_mask import flash_attention_custom_mask_fa3
                mask = attn_mask.to(device=query.device).expand(
                    target_mask_shape)
                return flash_attention_custom_mask_fa3(
                    query.contiguous(),
                    key.contiguous(),
                    (value if value.stride(-1) == 1 else value.contiguous()),
                    mask,
                    query.shape[-1]**-0.5,
                )
            except (ImportError, OSError, RuntimeError) as error:
                if require_fa3:
                    raise RuntimeError(
                        "required Hopper FlashAttention-3 execution failed; "
                        "prebuild with fa2_custom_mask.prebuild_flash_attention3() "
                        "on this node; see the optional Hopper extension in README") from error
    use_hopper_cudnn = (
        attn_mask is not None
        and query.is_cuda
        and query.dtype in (torch.bfloat16, torch.float16)
        and torch.cuda.get_device_capability(query.device) >= (9, 0)
        and hasattr(torch.backends.cuda, "cudnn_sdp_enabled")
        and torch.backends.cuda.cudnn_sdp_enabled()
    )
    if use_hopper_cudnn:
        try:
            from torch.nn.attention import SDPBackend, sdpa_kernel
            with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
                return F.scaled_dot_product_attention(
                    query, key, value, attn_mask=attn_mask)
        except (ImportError, RuntimeError):
            # Preserve correctness on a cuDNN/PyTorch combination that does
            # not implement a particular mask, head width, or stride layout.
            pass
    return F.scaled_dot_product_attention(
        query, key, value, attn_mask=attn_mask)


def masked_mean(value: torch.Tensor, mask: torch.Tensor,
                dim: int) -> torch.Tensor:
    mask = mask.to(device=value.device, dtype=value.dtype)
    while mask.ndim < value.ndim:
        mask = mask.unsqueeze(-1)
    denominator = mask.sum(dim=dim).clamp_min(1.0)
    return (value * mask).sum(dim=dim) / denominator


def layer_norm_fp32(norm: nn.LayerNorm, value: torch.Tensor) -> torch.Tensor:
    """Apply LayerNorm in FP32 even when deployment weights are BF16."""
    # CUDA's native low-precision LayerNorm accumulates its statistics and
    # affine math in FP32 before casting the result.  When the forward
    # parameters already match the activation dtype (the FSDP BF16 path), it
    # is bitwise equivalent to the explicit casts below while avoiding two
    # full-size FP32 activation tensors.
    native_dtype = value.dtype in (torch.float16, torch.bfloat16)
    matching_weight = norm.weight is None or norm.weight.dtype == value.dtype
    matching_bias = norm.bias is None or norm.bias.dtype == value.dtype
    if value.is_cuda and native_dtype and matching_weight and matching_bias:
        return F.layer_norm(
            value,
            norm.normalized_shape,
            norm.weight,
            norm.bias,
            norm.eps,
        )
    weight = None if norm.weight is None else norm.weight.float()
    bias = None if norm.bias is None else norm.bias.float()
    normalized = F.layer_norm(value.float(), norm.normalized_shape, weight,
                              bias, norm.eps)
    return normalized.to(value.dtype)


def infer_text_valid(text_emb: torch.Tensor) -> torch.Tensor:
    """Infer the strict mask used by preprocessors that zero-pad UMT5 rows."""
    if text_emb.ndim != 3:
        raise ValueError("text_emb must have shape (B,L,D)")
    valid = text_emb.ne(0).any(dim=-1)
    async_tensor_assert(
        valid.any(dim=1).all(),
        "every sample must contain at least one non-zero text token",
    )
    return valid


@dataclass(frozen=True)
class RotaryFrequencies:
    cosine: torch.Tensor
    sine: torch.Tensor


def select_packed_token_indices(
        valid: torch.Tensor,
        capacity: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Select canonical token positions for a fixed-capacity packed sequence."""
    if valid.ndim != 2 or valid.dtype != torch.bool:
        raise ValueError("packed token validity must be boolean (B,L)")
    if capacity <= 0 or capacity > valid.shape[1]:
        raise ValueError(
            "packed token capacity lies outside the dense sequence")
    async_tensor_assert(
        (valid.sum(dim=1) <= capacity).all(),
        "visual sample exceeds configured active-camera token capacity",
    )
    # topk avoids a variable-sized nonzero allocation.  Sorting the selected
    # canonical indices keeps RoPE/KV ordering deterministic.
    indices = torch.topk(
        valid.to(torch.uint8), capacity, dim=1, largest=True,
        sorted=False).indices
    indices = indices.sort(dim=1).values
    packed_valid = valid.gather(1, indices)
    return indices, packed_valid


def gather_packed_tokens(value: torch.Tensor,
                         indices: torch.Tensor) -> torch.Tensor:
    """Gather token axis 1 from ``(B,L,...)`` into fixed packed capacity."""
    if value.ndim < 2 or value.shape[0] != indices.shape[0]:
        raise ValueError("packed gather batch mismatch")
    index = indices.reshape(*indices.shape, *([1] * (value.ndim - 2))).expand(
        indices.shape[0], indices.shape[1], *value.shape[2:])
    return value.gather(1, index)


def scatter_packed_tokens(
    value: torch.Tensor,
    indices: torch.Tensor,
    dense_length: int,
    valid: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Scatter ``(B,P,...)`` back to canonical token axis 1."""
    if value.ndim < 2 or tuple(value.shape[:2]) != tuple(indices.shape):
        raise ValueError("packed scatter shape mismatch")
    if valid is not None:
        mask = valid.reshape(*valid.shape,
                             *([1] * (value.ndim - 2))).to(value.dtype)
        value = value * mask
    index = indices.reshape(*indices.shape,
                            *([1] * (value.ndim - 2))).expand_as(value)
    dense = value.new_zeros(value.shape[0], dense_length, *value.shape[2:])
    return dense.scatter(1, index, value)


def build_rotary_frequencies(
    positions: torch.Tensor,
    head_dim: int,
    *,
    theta: float = 10_000.0,
) -> RotaryFrequencies:
    """Precompute Wan-style factorized frequencies for one token grid.

    ``positions`` is ``(B,L,A)`` for one or more causal/spatial axes.  The
    complex head dimensions are divided evenly across axes, with the first
    axis receiving the remainder.  For three axes and Wan's 128-D heads this
    is exactly the official 22/21/21 complex-dimension T/H/W split.
    """
    if head_dim % 2:
        raise ValueError("rotary head dimension must be even")
    if positions.ndim != 3:
        raise ValueError("rotary positions must have shape (B,L,A)")
    _, _, axes = positions.shape
    if axes <= 0:
        raise ValueError("rotary positions must contain at least one axis")
    pairs = head_dim // 2
    base, remainder = divmod(pairs, axes)
    pair_splits = [
        base + (1 if axis < remainder else 0) for axis in range(axes)
    ]
    if any(split == 0 for split in pair_splits):
        raise ValueError(
            f"head_dim={head_dim} is too small for {axes} rotary axes")

    cosines = []
    sines = []
    for axis, pair_count in enumerate(pair_splits):
        real_dim = pair_count * 2
        exponent = torch.arange(
            0, real_dim, 2, device=positions.device,
            dtype=torch.float32) / float(real_dim)
        inverse_frequency = torch.pow(
            torch.tensor(theta, device=positions.device, dtype=torch.float32),
            -exponent,
        )
        angle = (positions[..., axis].float()[..., None] *
                 inverse_frequency)[:, None]
        cosines.append(angle.cos())
        sines.append(angle.sin())
    return RotaryFrequencies(
        cosine=torch.cat(cosines, dim=-1),
        sine=torch.cat(sines, dim=-1),
    )


def apply_rotary_frequencies(
    value: torch.Tensor,
    frequencies: RotaryFrequencies,
) -> torch.Tensor:
    """Apply precomputed complex rotations to ``(B,H,L,D)`` Q/K."""
    if value.ndim != 4:
        raise ValueError("rotary input must have shape (B,H,L,D)")
    batch, _, length, head_dim = value.shape
    expected = (batch, 1, length, head_dim // 2)
    if (head_dim % 2 or tuple(frequencies.cosine.shape) != expected
            or tuple(frequencies.sine.shape) != expected):
        raise ValueError(f"rotary frequencies must each have shape {expected}")
    try:
        from .triton_ops import has_triton_rotary, triton_rotary
        if has_triton_rotary(value):
            return triton_rotary(
                value,
                frequencies.cosine,
                frequencies.sine,
            )
    except (ImportError, RuntimeError):
        pass
    paired = value.float().reshape(batch, value.shape[1], length,
                                   head_dim // 2, 2)
    real, imaginary = paired.unbind(dim=-1)
    cosine = frequencies.cosine
    sine = frequencies.sine
    rotated = torch.stack(
        (
            real * cosine - imaginary * sine,
            imaginary * cosine + real * sine,
        ),
        dim=-1,
    )
    return rotated.flatten(-2).to(value.dtype)


def apply_rotary_embedding(
    value: torch.Tensor,
    positions: torch.Tensor,
    *,
    theta: float = 10_000.0,
) -> torch.Tensor:
    """Convenience wrapper used by callers that do not reuse a token grid."""
    frequencies = build_rotary_frequencies(
        positions, value.shape[-1], theta=theta)
    return apply_rotary_frequencies(value, frequencies)


RotaryInput = Union[torch.Tensor, RotaryFrequencies]


def _apply_rotary_input(
    value: torch.Tensor,
    positions: RotaryInput,
) -> torch.Tensor:
    if isinstance(positions, RotaryFrequencies):
        return apply_rotary_frequencies(value, positions)
    return apply_rotary_embedding(value, positions)


def factorized_grid_positions(
    batch: int,
    frames: int,
    rows: int,
    cols: int,
    device: torch.device,
    row_bases: Optional[Sequence[int]] = None,
    time_positions: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Return flattened ``(time,row,column)`` positions in Wan token order."""
    if min(batch, frames, rows, cols) <= 0:
        raise ValueError("grid dimensions must be positive")
    if time_positions is None:
        time = torch.arange(frames, device=device, dtype=torch.float32)
        time = time[None].expand(batch, -1)
    else:
        if tuple(time_positions.shape) != (batch, frames):
            raise ValueError(
                f"time_positions must have shape {(batch, frames)}")
        time = time_positions.to(device=device, dtype=torch.float32)
    if row_bases is None:
        row = torch.arange(rows, device=device, dtype=torch.float32)
    else:
        bases = tuple(int(base) for base in row_bases)
        if not bases or rows % len(bases):
            raise ValueError(
                f"rows={rows} must divide evenly across {len(bases)} camera bases"
            )
        rows_per_camera = rows // len(bases)
        if any(rows_per_camera > right - left
               for left, right in zip(bases, bases[1:])):
            raise ValueError(
                f"{rows_per_camera} patch rows per camera overlap bases {bases}"
            )
        local_row = torch.arange(
            rows_per_camera, device=device, dtype=torch.float32)
        row = torch.cat([local_row + float(base) for base in bases])
    column = torch.arange(cols, device=device, dtype=torch.float32)
    spatial_row, spatial_col = torch.meshgrid(row, column, indexing="ij")
    spatial_row = spatial_row.flatten()[None, None].expand(batch, frames, -1)
    spatial_col = spatial_col.flatten()[None, None].expand(batch, frames, -1)
    temporal = time[:, :, None].expand(batch, frames, rows * cols)
    return torch.stack((temporal, spatial_row, spatial_col),
                       dim=-1).reshape(batch, frames * rows * cols, 3)


class WanAttention(nn.Module):
    """Attention with parameter names compatible with LingBot/Wan blocks."""

    def __init__(self,
                 dim: int,
                 heads: int,
                 context_dim: Optional[int] = None) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError("attention dim must be divisible by heads")
        context_dim = dim if context_dim is None else context_dim
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.to_q = nn.Linear(dim, dim, bias=True)
        self.to_k = nn.Linear(context_dim, dim, bias=True)
        self.to_v = nn.Linear(context_dim, dim, bias=True)
        self.to_out = nn.ModuleList(
            [nn.Linear(dim, dim, bias=True),
             nn.Dropout(0.0)])
        self.norm_q = nn.RMSNorm(dim, eps=1e-6, elementwise_affine=True)
        self.norm_k = nn.RMSNorm(dim, eps=1e-6, elementwise_affine=True)

    def project_q(
        self,
        query: torch.Tensor,
        positions: Optional[RotaryInput] = None,
    ) -> torch.Tensor:
        projected = self.to_q(query)
        query = self.norm_q(projected)
        query = query.unflatten(-1,
                                (self.heads, self.head_dim)).transpose(1, 2)
        if positions is not None:
            query = _apply_rotary_input(query, positions)
        return query

    def project_kv(
        self,
        context: torch.Tensor,
        positions: Optional[RotaryInput] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        key = self.to_k(context)
        value = self.to_v(context)
        key = self.norm_k(key)
        key = key.unflatten(-1, (self.heads, self.head_dim)).transpose(1, 2)
        value = value.unflatten(-1,
                                (self.heads, self.head_dim)).transpose(1, 2)
        if positions is not None:
            key = _apply_rotary_input(key, positions)
        return key, value

    @staticmethod
    def _repeat_projected_context(value: torch.Tensor,
                                  repeats: int) -> torch.Tensor:
        if repeats == 1:
            return value
        if repeats <= 0:
            raise ValueError("context repeats must be positive")
        return value[:, None].expand(value.shape[0], repeats,
                                     *value.shape[1:]).reshape(
                                         value.shape[0] * repeats,
                                         *value.shape[1:])

    def attend_repeated_context(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        *,
        repeats: int = 1,
        logical_context_length: Optional[int] = None,
        context_prefix_length: Optional[int] = None,
        key_valid: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Cross-attend after projecting one context per original sample.

        ``query`` may contain repeated frames/horizons.  Projecting context at
        the original batch size removes the dominant redundant cross-K/V GEMM;
        projected tensors are expanded only for the attention kernel.
        """
        key, value = self.project_kv(context)
        attention_bias = None
        if logical_context_length is not None:
            if context_prefix_length is None:
                raise ValueError("context_prefix_length is required")
            attention_bias = padding_multiplicity_bias(
                key,
                prefix_length=context_prefix_length,
                logical_length=logical_context_length,
            )
            if attention_bias is not None and key_valid is not None:
                raise ValueError(
                    "multiplicity-compressed context cannot use key_valid")
        key = self._repeat_projected_context(key, repeats)
        value = self._repeat_projected_context(value, repeats)
        if query.shape[0] != key.shape[0]:
            raise ValueError(
                "query batch must equal context batch times repeats")
        if key_valid is not None:
            key_valid = self._repeat_projected_context(key_valid, repeats)
        projected_query = self.project_q(query)
        mask = (
            attention_bias if attention_bias is not None else _as_sdpa_mask(
                key_valid, projected_query))
        attended = _scaled_dot_product_attention(
            projected_query, key, value, attn_mask=mask)
        attended = attended.transpose(1, 2).flatten(-2)
        return self.to_out[1](self.to_out[0](attended))

    def attend_projected(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_valid: Optional[torch.Tensor] = None,
        pair_mask: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
        query_positions: Optional[RotaryInput] = None,
    ) -> torch.Tensor:
        q = self.project_q(query, positions=query_positions)
        if sum(
                value is not None
                for value in (key_valid, pair_mask, attention_bias)) > 1:
            raise ValueError("provide only one attention mask or bias")
        mask = key_valid if pair_mask is None else pair_mask
        mask = attention_bias if attention_bias is not None else _as_sdpa_mask(
            mask, q)
        attended = _scaled_dot_product_attention(
            q, key, value, attn_mask=mask)
        attended = attended.transpose(1, 2).flatten(-2)
        output = self.to_out[0](attended)
        return self.to_out[1](output)

    def attend_preprojected(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        key_valid: Optional[torch.Tensor] = None,
        pair_mask: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Attend when Q/K/V are already normalized, split, and RoPE-rotated."""
        if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
            raise ValueError("projected Q/K/V must have shape (B,H,L,D)")
        if key_valid is not None and pair_mask is not None:
            raise ValueError("provide key_valid or pair_mask, not both")
        mask = key_valid if pair_mask is None else pair_mask
        mask = _as_sdpa_mask(mask, query)
        if attention_bias is not None:
            if attention_bias.shape[-1] != key.shape[2]:
                raise ValueError("attention bias length differs from K/V")
            bias = attention_bias.to(device=query.device, dtype=query.dtype)
            if mask is not None:
                bias = bias.masked_fill(~mask, -torch.inf)
            mask = bias
        attended = _scaled_dot_product_attention(
            query, key, value, attn_mask=mask).transpose(1, 2).flatten(-2)
        output = self.to_out[0](attended)
        return self.to_out[1](output)

    def world_component_under_shared_softmax(
        self,
        query: torch.Tensor,
        world_key: torch.Tensor,
        world_value: torch.Tensor,
        action_key: torch.Tensor,
        action_value: torch.Tensor,
        *,
        world_valid: torch.Tensor,
        action_valid: torch.Tensor,
        return_attention_mass: bool = True,
        world_attention_bias: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the world's exact share of asymmetric-MoT attention.

        The normalization denominator contains both world and action tokens,
        exactly as in :meth:`attend_preprojected`.  The numerator is then
        restricted to world values.  The output projection bias is excluded:
        it is shared by both sources and cannot be assigned to either edge.
        """
        projected = (query, world_key, world_value, action_key, action_value)
        if any(value.ndim != 4 for value in projected):
            raise ValueError("projected Q/K/V must have shape (B,H,L,D)")
        batch, heads, queries, head_dim = query.shape
        if (heads != self.heads or head_dim != self.head_dim
                or world_key.shape[:2] != (batch, heads)
                or world_value.shape != world_key.shape
                or action_key.shape != (batch, heads, queries, head_dim)
                or action_value.shape != action_key.shape):
            raise ValueError(
                "world/action projections do not share one head layout")
        if (tuple(world_valid.shape) != (batch, world_key.shape[2])
                or tuple(action_valid.shape) != (batch, queries)):
            raise ValueError(
                "world/action validity does not match projected K/V")

        return self.source_component_under_shared_softmax(
            query,
            world_key,
            world_value,
            action_key,
            action_value,
            world_valid=world_valid,
            action_valid=action_valid,
            source_start=0,
            source_stop=world_key.shape[2],
            return_attention_mass=return_attention_mass,
            world_attention_bias=world_attention_bias,
        )

    def source_component_under_shared_softmax(
        self,
        query: torch.Tensor,
        world_key: torch.Tensor,
        world_value: torch.Tensor,
        action_key: torch.Tensor,
        action_value: torch.Tensor,
        *,
        world_valid: torch.Tensor,
        action_valid: torch.Tensor,
        source_start: int,
        source_stop: int,
        return_attention_mass: bool = True,
        world_attention_bias: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return one world's contribution under the full MoT denominator."""
        projected = (query, world_key, world_value, action_key, action_value)
        if any(value.ndim != 4 for value in projected):
            raise ValueError("projected Q/K/V must have shape (B,H,L,D)")
        batch, heads, queries, head_dim = query.shape
        if (heads != self.heads or head_dim != self.head_dim
                or world_key.shape[:2] != (batch, heads)
                or world_value.shape != world_key.shape
                or action_key.shape != (batch, heads, queries, head_dim)
                or action_value.shape != action_key.shape):
            raise ValueError(
                "world/action projections do not share one head layout")
        if (tuple(world_valid.shape) != (batch, world_key.shape[2])
                or tuple(action_valid.shape) != (batch, queries)):
            raise ValueError(
                "world/action validity does not match projected K/V")
        if not 0 <= source_start < source_stop <= world_key.shape[2]:
            raise ValueError("source range lies outside world K/V")

        key = torch.cat((world_key, action_key), dim=2)
        partitioned_world = torch.zeros_like(world_value)
        partitioned_world[:, :, source_start:
                          source_stop] = world_value[:, :,
                                                     source_start:source_stop]
        partitioned_value = torch.cat(
            (partitioned_world, torch.zeros_like(action_value)), dim=2)
        valid = torch.cat((world_valid, action_valid), dim=1)
        attention_bias = None
        if world_attention_bias is not None:
            if world_attention_bias.shape[-1] != world_key.shape[2]:
                raise ValueError(
                    "world attention bias length differs from K/V")
            action_bias = world_attention_bias.new_zeros(
                *world_attention_bias.shape[:-1], action_key.shape[2])
            attention_bias = torch.cat((world_attention_bias, action_bias),
                                       dim=-1).to(query.dtype)
            attention_bias = attention_bias.masked_fill(
                ~valid[:, None, None], -torch.inf)
        attended = _scaled_dot_product_attention(
            query,
            key,
            partitioned_value,
            attn_mask=(attention_bias if attention_bias is not None else
                       _as_sdpa_mask(valid, query)),
        ).transpose(1, 2).flatten(-2)
        contribution = F.linear(
            attended,
            self.to_out[0].weight,
            bias=None,
        )
        if not return_attention_mass:
            return contribution, query.new_zeros(batch, queries)
        with torch.no_grad():
            logits = torch.matmul(
                query.float(),
                key.float().transpose(-1, -2),
            ) * self.head_dim**-0.5
            logits = logits.masked_fill(~valid[:, None, None], -torch.inf)
            if attention_bias is not None:
                logits = logits + attention_bias.float()
            probabilities = logits.softmax(dim=-1)
            attention_mass = probabilities[..., source_start:source_stop].sum(
                dim=-1).mean(dim=1)
        return contribution, attention_mass

    def forward(
        self,
        query: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        key_valid: Optional[torch.Tensor] = None,
        pair_mask: Optional[torch.Tensor] = None,
        query_positions: Optional[RotaryInput] = None,
        context_positions: Optional[RotaryInput] = None,
    ) -> torch.Tensor:
        self_attention = context is None
        context = query if self_attention else context
        if self_attention and context_positions is None:
            context_positions = query_positions
        key, value = self.project_kv(context, positions=context_positions)
        projected_query = self.project_q(query, positions=query_positions)
        if key_valid is not None and pair_mask is not None:
            raise ValueError("provide key_valid or pair_mask, not both")
        mask = key_valid if pair_mask is None else pair_mask
        mask = _as_sdpa_mask(mask, projected_query)
        attended = _scaled_dot_product_attention(
            projected_query, key, value, attn_mask=mask)
        attended = attended.transpose(1, 2).flatten(-2)
        return self.to_out[1](self.to_out[0](attended))


class _GELUProjection(nn.Module):

    def __init__(self, dim: int, inner_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, inner_dim)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.proj(value), approximate="tanh")


class WanFeedForward(nn.Module):
    """Feed-forward with ``ffn.net.0.proj``/``ffn.net.2`` checkpoint keys."""

    def __init__(self, dim: int, inner_dim: int) -> None:
        super().__init__()
        self.net = nn.ModuleList([
            _GELUProjection(dim, inner_dim),
            nn.Dropout(0.0),
            nn.Linear(inner_dim, dim)
        ])

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = self.net[0].proj(value)
        value = F.gelu(value, approximate="tanh")
        value = self.net[1](value)
        return self.net[2](value)


class FP32Linear(nn.Linear):
    """Linear layer that performs its matmul in FP32 under mixed precision.

    Subclassing :class:`nn.Linear` deliberately preserves the official Wan
    state-dict names (``weight`` and ``bias``).  Keeping the conversion inside
    ``forward`` also makes every managed parameter read visible to sharded
    module hooks; callers must not read these weights to issue a raw
    ``F.linear`` outside the module call boundary.
    """

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=value.device.type, enabled=False):
            return F.linear(
                value.float(),
                self.weight.float(),
                None if self.bias is None else self.bias.float(),
            )


class SinusoidalConditioner(nn.Module):
    """Wan-shaped time and text conditioner without a diffusers dependency."""

    def __init__(self,
                 dim: int,
                 text_dim: int,
                 frequency_dim: int = 256) -> None:
        super().__init__()
        self.frequency_dim = frequency_dim
        self.time_embedder = nn.Module()
        self.time_embedder.linear_1 = FP32Linear(frequency_dim, dim)
        self.time_embedder.linear_2 = FP32Linear(dim, dim)
        self.time_proj = FP32Linear(dim, dim * 6)
        self.text_embedder = nn.Module()
        self.text_embedder.linear_1 = nn.Linear(text_dim, dim)
        self.text_embedder.linear_2 = nn.Linear(dim, dim)

    def _frequencies(self, timestep: torch.Tensor) -> torch.Tensor:
        half = self.frequency_dim // 2
        exponent = -math.log(10_000.0) * torch.arange(
            half, device=timestep.device, dtype=torch.float64) / max(half, 1)
        phase = timestep.double().reshape(-1, 1) * exponent.exp().reshape(
            1, -1)
        value = torch.cat((phase.cos(), phase.sin()), dim=-1)
        if value.shape[-1] < self.frequency_dim:
            value = F.pad(value, (0, self.frequency_dim - value.shape[-1]))
        return value.float()

    def time(self, timestep: torch.Tensor,
             dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        # Official Wan keeps both ``e`` and its six-way AdaLN projection in
        # FP32.  This is not cosmetic: the gates multiply every residual and
        # define the K/V coordinate system being distilled.  Explicit linear
        # module calls also preserve FP32 when FSDP exposes BF16 forward
        # parameters, while remaining visible to sharding hooks.
        value = self._frequencies(timestep)
        value = self.time_embedder.linear_1(value)
        value = self.time_embedder.linear_2(F.silu(value))
        projection = self.time_proj(F.silu(value))
        return value, projection

    def text(self, text_emb: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        param_dtype = self.text_embedder.linear_1.weight.dtype
        value = text_emb.to(dtype=param_dtype)
        value = self.text_embedder.linear_1(value)
        value = F.gelu(value, approximate="tanh")
        value = self.text_embedder.linear_2(value)
        return value.to(dtype=dtype)


class WanCompatibleBlock(nn.Module):
    """A Wan block used by clean observation and frozen teacher encoders."""

    def __init__(self,
                 dim: int,
                 heads: int,
                 ffn_dim: int,
                 eps: float = 1e-6) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.attn1 = WanAttention(dim, heads)
        self.attn2 = WanAttention(dim, heads)
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=True)
        self.ffn = WanFeedForward(dim, ffn_dim)
        self.norm3 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.scale_shift_table = nn.Parameter(
            torch.randn(1, 6, dim) / math.sqrt(dim))

    def forward(
        self,
        hidden: torch.Tensor,
        time_projection: torch.Tensor,
        text: torch.Tensor,
        token_valid: Optional[torch.Tensor] = None,
        self_pair_mask: Optional[torch.Tensor] = None,
        text_valid: Optional[torch.Tensor] = None,
        text_repeats: int = 1,
        text_logical_length: Optional[int] = None,
        text_prefix_length: Optional[int] = None,
        self_positions: Optional[RotaryInput] = None,
        return_self_kv: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        if time_projection.ndim == 2:
            time_projection = time_projection[:, None, :]
        modulation = self.scale_shift_table.unsqueeze(
            0) + time_projection.unflatten(-1, (6, hidden.shape[-1]))
        shift, scale, gate, ff_shift, ff_scale, ff_gate = modulation.unbind(
            dim=-2)
        value = layer_norm_fp32(self.norm1, hidden.float())
        value = (value * (1.0 + scale) + shift).to(hidden.dtype)
        key, projected_value = self.attn1.project_kv(
            value, positions=self_positions)
        pair_mask = self_pair_mask
        if pair_mask is not None:
            expected = (hidden.shape[0], hidden.shape[1], hidden.shape[1])
            if pair_mask.dtype != torch.bool or tuple(
                    pair_mask.shape) != expected:
                raise ValueError(
                    f"self_pair_mask must be boolean {expected}, got "
                    f"{tuple(pair_mask.shape)}")
            if token_valid is not None:
                pair_mask = pair_mask & token_valid[:, None, :]
        attended = self.attn1.attend_projected(
            value,
            key,
            projected_value,
            key_valid=token_valid if pair_mask is None else None,
            pair_mask=pair_mask,
            query_positions=self_positions,
        )
        hidden = (hidden.float() + gate * attended.float()).to(hidden.dtype)
        value = layer_norm_fp32(self.norm2, hidden)
        hidden = hidden + self.attn2.attend_repeated_context(
            value,
            text,
            repeats=text_repeats,
            logical_context_length=text_logical_length,
            context_prefix_length=text_prefix_length,
            key_valid=text_valid,
        )
        value = layer_norm_fp32(self.norm3, hidden.float())
        value = (value * (1.0 + ff_scale) + ff_shift).to(hidden.dtype)
        feed_forward = self.ffn(value)
        hidden = (hidden.float() + ff_gate * feed_forward.float()).to(
            hidden.dtype)
        if token_valid is not None:
            hidden = hidden * token_valid.unsqueeze(-1).to(hidden.dtype)
        if return_self_kv:
            return hidden, key, projected_value
        return hidden


class QueryTransformerBlock(nn.Module):
    """Self-attending queries followed by one explicit cross-attention edge."""

    def __init__(self,
                 dim: int,
                 heads: int,
                 ffn_dim: int,
                 context_dim: Optional[int] = None) -> None:
        super().__init__()
        self.norm_self = nn.LayerNorm(dim)
        self.self_attn = WanAttention(dim, heads)
        self.norm_cross = nn.LayerNorm(dim)
        self.cross_attn = WanAttention(dim, heads, context_dim=context_dim)
        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = WanFeedForward(dim, ffn_dim)

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        context_valid: Optional[torch.Tensor] = None,
        query_valid: Optional[torch.Tensor] = None,
        causal: bool = False,
    ) -> torch.Tensor:
        pair_mask = None
        if causal:
            length = query.shape[1]
            pair_mask = torch.ones(
                length, length, device=query.device, dtype=torch.bool).tril()
            pair_mask = pair_mask[None].expand(query.shape[0], -1, -1)
            if query_valid is not None:
                pair_mask = pair_mask & query_valid[:, None, :]
        query = query + self.self_attn(
            self.norm_self(query),
            key_valid=None if causal else query_valid,
            pair_mask=pair_mask)
        query = query + self.cross_attn(
            self.norm_cross(query), context=context, key_valid=context_valid)
        query = query + self.ffn(self.norm_ffn(query))
        if query_valid is not None:
            query = query * query_valid.unsqueeze(-1).to(query.dtype)
        return query


@dataclass(frozen=True)
class ProjectedKV:
    key: torch.Tensor
    value: torch.Tensor
    valid: Optional[torch.Tensor]
    attention_bias: Optional[torch.Tensor] = None


def project_valid_kv(
    attention: WanAttention,
    context: torch.Tensor,
    valid: Optional[torch.Tensor],
) -> ProjectedKV:
    key, value = attention.project_kv(context)
    return ProjectedKV(key=key, value=value, valid=valid)
