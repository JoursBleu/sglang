"""Triton stand-in for `aiter_fp4_paged_mqa_logits` on gfx942.

The low-ratio indexer ships a single FlyDSL implementation built on CDNA4's
`v_mfma_scale_f32_16x16x128_f8f6f4`, which gfx942 cannot even compile, and
`low_ratio_backend_hip.py` exposes no env switch to route around it.

Semantics, matching `dsv41_sparse.Indexer.scores`:
    logits[t, n] = sum_h relu(q[t, h, :] . k[n, :]) * weights[t, h]

The index-K pool is swizzled for MFMA access (measured at runtime, not assumed):
    payload [pages, 1, 4, 64, 16] float4_e2m1fn_x2  stride (4096, 4096, 1024, 16, 1)
    scale   [pages, 1, 4, 64]     uint8             one ue8m0 per 32 elements
A slot's 64 payload bytes are split into 4 segments of 16 bytes with the slot
dimension in between. Element d lives in byte d // 2, inside segment d // 32
(which doubles as the scale block index), at offset (d // 2) % 16.

Q arrives as bf16: the fp4 packing exists for CDNA4's native FP4 MFMA, and
gfx942 has no such instruction.

Two things the first version got wrong, both worth keeping in mind:
  1. The grid is laid out over max_seq_len, which is the page-table width rather
     than the row's real length. Masking `tl.load` alone saves no arithmetic --
     the block-wide reduction still runs -- so whole blocks past the row's reach
     must return early.
  2. Reducing the 32 heads one at a time with `tl.sum` is 32 cross-lane
     reductions and uses no MFMA at all. A single [BLOCK_N, D] x [D, H] `tl.dot`
     puts v_mfma_f32_16x16x16_bf16 to work instead. The eight fp4 magnitudes need
     3 mantissa bits, so the bf16 conversion is lossless and accuracy is
     unchanged.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

HEAD_DIM = 128


@triton.jit
def _e2m1_decode(nib):
    """4-bit e2m1 -> f32, by arithmetic rather than a lookup table: a table needs a
    device tensor whose first allocation lands inside CUDA-graph capture, and that
    segfaults capture. Codes 0..7 are 0, .5, 1, 1.5, 2, 3, 4, 6; bit 3 is the sign."""
    code = nib & 7
    mag = tl.where(
        code < 2,
        0.5 * code.to(tl.float32),
        (1.0 + 0.5 * (code & 1).to(tl.float32)) * tl.exp2(((code >> 1) - 1).to(tl.float32)),
    )
    return tl.where((nib >> 3) & 1 == 1, -mag, mag)


@triton.jit
def _mqa_logits_kernel(
    q_ptr, w_ptr, kpay_ptr, ksc_ptr, pt_ptr, slen_ptr, out_ptr,
    s_qt, s_qh, s_pt, s_out,
    kp0, kp2, kp3, ks0, ks2,
    H: tl.constexpr, D: tl.constexpr, PAGE: tl.constexpr, BLOCK_N: tl.constexpr,
):
    t = tl.program_id(0)
    start = tl.program_id(1) * BLOCK_N
    slen = tl.load(slen_ptr + t)
    # The whole block lies past this row's reach. The grid covers the page-table
    # width, which is usually far larger than the real length, and masking the
    # loads alone would still run the full reduction.
    if start >= slen:
        return

    offs_n = start + tl.arange(0, BLOCK_N)
    mask_n = offs_n < slen

    page = tl.load(
        pt_ptr + t.to(tl.int64) * s_pt + (offs_n // PAGE).to(tl.int64),
        mask=mask_n, other=0,
    ).to(tl.int64)
    slot = (offs_n % PAGE).to(tl.int64)

    offs_d = tl.arange(0, D)
    seg = offs_d // 32              # payload segment, also the scale block index
    within = (offs_d // 2) % 16     # byte offset inside the segment

    # [BLOCK_N, D]: d is the inner dimension and is byte-contiguous within a
    # segment, so the loads coalesce.
    base = page[:, None] * kp0 + slot[:, None] * kp3
    byte = tl.load(
        kpay_ptr + base + seg[None, :] * kp2 + within[None, :],
        mask=mask_n[:, None], other=0,
    ).to(tl.int32)
    nib = tl.where((offs_d % 2)[None, :] == 1, (byte >> 4) & 0xF, byte & 0xF)
    kval = _e2m1_decode(nib)

    exp = tl.load(
        ksc_ptr + page[:, None] * ks0 + seg[None, :] * ks2 + slot[:, None],
        mask=mask_n[:, None], other=127,
    ).to(tl.int32)
    kval = kval * tl.exp2((exp - 127).to(tl.float32))

    # Load q straight as [D, H] to skip a tl.trans; q is small and stays in cache.
    offs_h = tl.arange(0, H)
    qt = tl.load(q_ptr + t * s_qt + offs_d[:, None] + offs_h[None, :] * s_qh)

    # The fp4 magnitudes need 3 mantissa bits, so bf16 holds them exactly.
    scores = tl.dot(kval.to(tl.bfloat16), qt.to(tl.bfloat16))  # [BLOCK_N, H] fp32
    w = tl.load(w_ptr + t * H + offs_h).to(tl.float32)
    acc = tl.sum(tl.maximum(scores, 0.0) * w[None, :], axis=1)

    # A prefill rectangle can reach ~1000 rows by ~500K columns, so the row offset
    # has to be int64: Triton specializes Python int arguments that fit in int32.
    tl.store(
        out_ptr + t.to(tl.int64) * s_out.to(tl.int64) + offs_n.to(tl.int64),
        acc, mask=mask_n,
    )


def triton_paged_mqa_logits(
    *,
    q: torch.Tensor,            # [T, H, D] bf16
    weights: torch.Tensor,      # [T, H] bf16
    k_payload: torch.Tensor,    # [pages, 1, 4, 64, 16]
    k_scale: torch.Tensor,      # [pages, 1, 4, 64] uint8
    page_table: torch.Tensor,   # [T, P] int32
    c4_seq_lens: torch.Tensor,  # [T] int32
    max_seq_len: int,
    page_size: int = 64,
    out: torch.Tensor = None,
    block_n: int = 64,
    num_warps: int = 4,
    num_stages: int = 2,
) -> torch.Tensor:
    T, H, D = q.shape
    assert D == HEAD_DIM, D
    kp = k_payload.view(torch.uint8)
    ks = k_scale.view(torch.uint8)
    if c4_seq_lens.dtype != torch.int32:
        c4_seq_lens = c4_seq_lens.to(torch.int32)
    if out is None:
        out = torch.empty((T, max_seq_len), dtype=torch.float32, device=q.device)

    _mqa_logits_kernel[(T, triton.cdiv(max_seq_len, block_n))](
        q, weights, kp, ks, page_table, c4_seq_lens, out,
        q.stride(0), q.stride(1), page_table.stride(0), out.stride(0),
        kp.stride(0), kp.stride(2), kp.stride(3), ks.stride(0), ks.stride(2),
        H=H, D=D, PAGE=page_size, BLOCK_N=block_n,
        num_warps=num_warps, num_stages=num_stages,
    )
    return out
