from __future__ import annotations

from typing import Optional

import torch

from sglang.kernels.jit.utils import (
    cache_once,
    is_arch_support_pdl,
    is_hip_runtime,
    load_jit,
    make_cpp_args,
)
from sglang.srt.utils import is_xpu

from .utils import make_name


@cache_once
def _jit_topk_v1_module():
    args = make_cpp_args(is_arch_support_pdl())
    return load_jit(
        make_name("topk_v1"),
        *args,
        cuda_files=["deepseek_v4/topk_v1.cuh"],
        cuda_wrappers=[("topk_transform", f"TopKKernel<{args}>::transform")],
    )


@cache_once
def _jit_topk_v2_module():
    from sglang.kernels.ops.misc import get_max_active_clusters

    args = make_cpp_args(is_arch_support_pdl())
    # Leave these undefined if the probe fails: topk_v2.cuh carries per-arch
    # defaults, and a 0 would size the persistent pool to an empty grid.
    extra_cuda_cflags = []
    if is_arch_support_pdl():  # set the persistent cluster size after hopper
        occ_8_2, occ_16_1 = 0, 0
        try:
            occ_8_2 = get_max_active_clusters(8, occupancy=2)
            # NOTE: cluster 16 might fail, but at least cluster 8 is ok
            occ_16_1 = get_max_active_clusters(16, occupancy=1)
        except Exception:
            pass
        extra_cuda_cflags = [
            f"-DSGL_TOPK_V2_MAX_C8_OCC2={occ_8_2}",
            f"-DSGL_TOPK_V2_MAX_C16_OCC1={occ_16_1}",
        ]
    kernel = f"TopKKernel<{args}>"
    return load_jit(
        make_name("topk_v2"),
        *args,
        extra_cuda_cflags=extra_cuda_cflags,
        cuda_files=["deepseek_v4/topk_v2.cuh"],
        cuda_wrappers=[
            ("topk_transform_paged", f"{kernel}::transform_paged"),
            ("topk_transform_ragged", f"{kernel}::transform_ragged"),
            ("topk_plan", f"{kernel}::plan"),
        ],
    )


@cache_once
def _jit_topk_bf16_small_module():
    args = make_cpp_args(is_arch_support_pdl())
    return load_jit(
        make_name("topk_bf16_small"),
        *args,
        cuda_files=["deepseek_v4/topk_bf16_small.cuh"],
        cuda_wrappers=[("topk_transform", f"TopKBF16Kernel<{args}>::transform")],
    )


def topk_transform_bf16_small(
    scores: torch.Tensor,
    seq_lens: torch.Tensor,
    page_table: torch.Tensor,
    out_page_indices: torch.Tensor,
    page_size: int,
) -> None:
    """bf16 top-k for rows of at most 16384 scores (the DeepSeek-V4.1 sparse
    indexer's consumer rows), fused with a page-table transform.

    Row ``b`` selects the ``k = out_page_indices.shape[1]`` best of its first
    ``seq_lens[b]`` scores; a selected index ``i`` is written as
    ``page_table[b, i // page_size] * page_size + i % page_size``, in no
    particular order, and ``-1`` fills the slots past ``min(k, seq_lens[b])``.
    Selection is by a 13-bit fp16-derived key, exact for bf16 in fp16's normal
    range; ties within a key are broken arbitrarily.
    """
    _jit_topk_bf16_small_module().topk_transform(
        scores, seq_lens, page_table, out_page_indices, page_size
    )


@cache_once
def _jit_amax_copy_module():
    args = make_cpp_args(is_arch_support_pdl())
    return load_jit(
        make_name("amax_copy"),
        *args,
        cuda_files=["deepseek_v4/amax_copy.cuh"],
        cuda_wrappers=[("amax8_varlen", f"AmaxCopyKernel<{args}>::amax8_varlen")],
    )


def amax8_varlen(
    scores: torch.Tensor,
    seq_lens: torch.Tensor,
    topk: int = 0,
    *,
    max_seqlen: int = 0,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Level-one keys of the two-level indexer: ``out[b, i]`` is the max of
    ``scores[b, 8 i : 8 i + 8]`` for ``i < ceil(seq_lens[b] / 8)``, the last of
    them ``+inf`` (the newest block is always selected), nothing written past
    that count. Rows with at most ``topk`` blocks are skipped (every block is
    selected anyway); ``topk=0`` never skips. ``out`` is allocated as
    ``[rows, ceil(max_seqlen / 8)]`` when not given, ``max_seqlen`` defaulting to
    the width of ``scores``; every ``seq_lens[b]`` must fit in ``8 * out.shape[1]``.
    fp32 only for now; ``scores`` rows must be 32-byte aligned (stride a multiple
    of 8). Returns ``out``.
    """
    if out is None:
        num_tokens, max_len = scores.shape
        if max_seqlen == 0:
            max_seqlen = max_len
        out = scores.new_empty(num_tokens, (max_seqlen + 7) // 8)
    _jit_amax_copy_module().amax8_varlen(scores, seq_lens, out, topk)
    return out


@cache_once
def _jit_sort_idx_module():
    args = make_cpp_args(is_arch_support_pdl())
    return load_jit(
        make_name("sort_idx"),
        *args,
        cuda_files=["deepseek_v4/sort_idx.cuh"],
        cuda_wrappers=[
            ("transform", f"SortIdxKernel<{args}>::transform"),
            ("transform_pages", f"SortIdxKernel<{args}>::transform_pages"),
        ],
    )


def sort_candidate_blocks(
    blocks: torch.Tensor,
    seq_lens: torch.Tensor,
    page_table: torch.Tensor,
    page_size: int,
    *,
    out_pages: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """The block table of the two-level indexer from a row's selected blocks,
    in place: ``blocks`` ``[rows, k]`` int32 block ids in any order, ``-1``
    padded, become the same ids ascending with ``INT32_MAX`` past ``min(k,
    ceil(seq_lens[b] / 8))``; the matching pool slots / 8 (``page_table[b, id //
    bpp] * bpp + id % bpp``, ``bpp = page_size // 8``, same padding) go to
    ``out_pages``. A row with at most ``k`` blocks gets the identity table
    regardless of its input. Returns ``out_pages``.
    """
    if out_pages is None:
        out_pages = torch.empty_like(blocks)
    _jit_sort_idx_module().transform(blocks, seq_lens, page_table, out_pages, page_size)
    return out_pages


def transform_candidate_blocks(
    blocks: torch.Tensor,
    seq_lens: torch.Tensor,
    page_table: torch.Tensor,
    page_size: int,
    *,
    out_pages: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """The page transform of ``sort_candidate_blocks`` alone, for a block top-k
    that already emits ascending ids: ``out_pages[b, t]`` is the pool slot / 8 of
    ``blocks[b, t]`` for ``t < min(k, ceil(seq_lens[b] / 8))`` (which must be
    valid block ids), ``INT32_MAX`` past that; ``blocks`` is not modified.
    Returns ``out_pages``.
    """
    if out_pages is None:
        out_pages = torch.empty_like(blocks)
    _jit_sort_idx_module().transform_pages(
        blocks, seq_lens, page_table, out_pages, page_size
    )
    return out_pages


def topk_transform_paged(
    scores: torch.Tensor,
    seq_lens: torch.Tensor,
    page_tables: torch.Tensor,
    out_page_indices: torch.Tensor,
    page_size: int,
    out_raw_indices: Optional[torch.Tensor] = None,
) -> None:
    if is_hip_runtime():
        # PATCH(AMD): neither v1 path exists on ROCm. The AOT kernel this used to
        # call (sgl_kernel.deepseek_v4_topk_transform_512) is not in the build,
        # and the JIT v1 kernel asks for 65540 B of dynamic shared memory -- more
        # than gfx9's 64 KiB LDS. topk_v2 is ROCm-ported and matches v1's
        # semantics (rows with seq_len <= topk take the sequential branch and
        # never read the scores), but it writes one output per launch.
        module = _jit_topk_v2_module()
        # v2 does 16-byte vectorized score loads; the identity branch hands us a
        # (batch, 1) placeholder whose row stride breaks that.
        if scores.dim() == 2 and scores.stride(0) % 4:
            padded = scores.new_empty(
                (scores.shape[0], (scores.shape[1] + 3) // 4 * 4)
            )
            padded[:, : scores.shape[1]] = scores
            scores = padded
        # plan_topk_v2 allocates with new_empty and topk_plan only writes row 0
        # when rows route to the cluster pool, so the uninitialised tail leaks in.
        metadata = torch.zeros(
            (seq_lens.shape[0] + 1, _PLAN_METADATA_INTS_PER_BATCH),
            dtype=torch.int32,
            device=seq_lens.device,
        )
        module.topk_plan(seq_lens, metadata, 0)
        if out_raw_indices is None:
            module.topk_transform_paged(
                scores, seq_lens, page_tables, out_page_indices, page_size, metadata
            )
            return
        # v1 writes both outputs from one selection. v2 writes one per launch and
        # is not reproducible across launches (the set matches torch.topk but the
        # order does not), so calling it twice would hand the caller two
        # *different* selections. Select once, then apply the page table here so
        # both outputs describe the same rows.
        module.topk_transform_paged(
            scores, seq_lens, None, out_raw_indices, page_size, metadata
        )
        if page_tables is None:
            out_page_indices.copy_(out_raw_indices)
            return
        cols = out_raw_indices.clamp_min(0).to(torch.int64)
        pages = page_tables.to(torch.int64).gather(1, cols // page_size)
        slots = pages * page_size + cols % page_size
        out_page_indices.copy_(
            torch.where(out_raw_indices >= 0, slots, slots.new_full((), -1)).to(
                out_page_indices.dtype
            )
        )
        return
    elif is_xpu():
        torch.ops.sgl_kernel.topk_transform(
            scores, seq_lens, page_tables, out_page_indices, page_size, out_raw_indices
        )
    else:
        module = _jit_topk_v1_module()
        module.topk_transform(
            scores, seq_lens, page_tables, out_page_indices, page_size, out_raw_indices
        )


# metadata is (batch+1, 2) int32: row 0 = {cluster_threshold, num_cluster_items};
# rows 1..N = {batch_id, seq_len} of items routed to the persistent cluster pool.
_PLAN_METADATA_INTS_PER_BATCH = 2


def plan_topk_v2(seq_lens: torch.Tensor, static_threshold: int = -1) -> torch.Tensor:
    """
    Preprocess the per-batch routing plan for :func:`topk_transform_paged_v2`.
    NOTE: every entry of ``seq_lens`` must be NON-NEGATIVE.

    :param static_threshold: If a batch item has `seq_len` > `static_threshold`,
                             prefer the cluster implementation.
                             Negative number means internal heuristic.
    """
    module = _jit_topk_v2_module()
    bs = seq_lens.shape[0]
    metadata = seq_lens.new_empty(bs + 1, _PLAN_METADATA_INTS_PER_BATCH)
    module.topk_plan(seq_lens, metadata, static_threshold)
    return metadata


def topk_transform_ragged_v2(
    scores: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    out_offsets: torch.Tensor,
    out_indices: torch.Tensor,
    row_starts: Optional[torch.Tensor] = None,
) -> None:
    """Ragged (prefill) fused top-k for a contiguous-KV score matrix.

    Row ``i`` selects the top-k of ``scores[i, ks : ks + seq_lens[i]]`` (``ks =
    row_starts[i]``, 0 when ``row_starts`` is omitted) and writes
    ``selected_position + out_offsets[i]`` into ``out_indices``, ``-1`` padded.
    With the production convention ``out_offsets == row_starts`` that is the
    column index itself, i.e. the token's slot in the batch's flattened KV.

    Unlike :func:`topk_transform_paged_v2` this needs no page table and no plan
    (the cluster path only pays off for very few rows, and prefill has many).

    NOTE: ``scores`` is written in place -- the <= 3 columns ahead of each
    row's window that the 16-byte-aligned read base pulls in are masked out.
    They are invalid for that row and the buffer must have no other consumer.
    ``seq_lens`` entries must be NON-NEGATIVE, as for the paged entry point.
    """
    if is_xpu():
        torch.ops.sgl_kernel.topk_transform_ragged(
            scores,
            seq_lens,
            out_indices,
            out_offsets,
            row_starts,
        )
        return
    module = _jit_topk_v2_module()
    module.topk_transform_ragged(scores, seq_lens, row_starts, out_offsets, out_indices)


def topk_transform_paged_v2(
    scores: torch.Tensor,
    seq_lens: torch.Tensor,
    page_tables: Optional[torch.Tensor],
    out_page_indices: torch.Tensor,
    page_size: int,
    metadata: torch.Tensor,
    out_raw_indices: Optional[torch.Tensor] = None,
) -> None:
    """Fused top-k + optional page-table transform (DeepSeek-V4 top-k v2 kernel).

    Output mode is chosen from ``page_tables`` and ``out_raw_indices`` and
    resolved to a device-side template parameter, so an unused page-table gather
    is compiled out rather than skipped at runtime:

    * ``page_tables=None`` -- ``out_page_indices`` receives the raw selected
      indices and no page table is read.
    * ``page_tables`` given -- ``out_page_indices`` receives the page-table
      transform of them.
    * Both outputs given -- ``out_page_indices`` receives the page-table
      transform and ``out_raw_indices`` receives the selected raw indices.

    NOTE: every entry of `seq_lens` must be NON-NEGATIVE, and `metadata` must
    come from :func:`plan_topk_v2` over the same `seq_lens` values.
    A length of 0 is the valid way to express "no tokens": the row takes the
    trivial path and the output is guaranteed to be all -1.
    """
    if is_xpu():
        if out_raw_indices is not None:
            topk_transform_paged(
                scores,
                seq_lens,
                page_tables,
                out_page_indices,
                page_size,
                out_raw_indices,
            )
            return
        torch.ops.sgl_kernel.topk_transform_paged(
            scores,
            seq_lens,
            page_tables,
            out_page_indices,
            page_size,
            metadata,
        )
        return
    module = _jit_topk_v2_module()
    module.topk_transform_paged(
        scores,
        seq_lens,
        page_tables,
        out_page_indices,
        page_size,
        metadata,
        out_raw_indices,
    )
