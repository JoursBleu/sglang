import math
import sys
from types import SimpleNamespace

import pytest
import torch

import sglang.srt.models.mimo_v2 as mimo_v2
from sglang.srt.models.mimo_v2 import (
    _fused_qkv_kv_chunk_rows,
    _shard_fp8_qkv_proj,
    _split_qkv,
    get_mimo_v2_fused_qkv_chunks,
    try_load_mimo_v2_fp8_qkv_proj,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

BLOCK = 128
COLS = 128
GA = (64, 4, 192, 128, 4)
SWA = (64, 8, 192, 128, 4)


def _chunk_sizes(geometry):
    num_heads, num_kv_heads, head_dim, v_head_dim, ckpt_tp = geometry
    return (
        num_heads // ckpt_tp * head_dim,
        num_kv_heads // ckpt_tp * head_dim,
        num_kv_heads // ckpt_tp * v_head_dim,
    )


def _checkpoint(geometry, dtype=torch.float8_e4m3fn):
    q_rows, k_rows, v_rows = _chunk_sizes(geometry)
    rows_per_chunk = q_rows + k_rows + v_rows
    ckpt_tp = geometry[-1]
    weight_bytes = (
        torch.arange(ckpt_tp * rows_per_chunk * COLS, dtype=torch.int64)
        .remainder(251)
        .to(torch.uint8)
        .view(ckpt_tp * rows_per_chunk, COLS)
    )
    weight = weight_bytes.view(dtype)
    scale_rows = ckpt_tp * math.ceil(rows_per_chunk / BLOCK)
    scale = torch.arange(scale_rows, dtype=torch.float32).view(scale_rows, 1)
    return weight, scale


def _bits(tensor):
    return tensor.contiguous().view(torch.uint8)


@pytest.mark.parametrize("tp_rank", [0, 1])
def test_tp2_global_attention_preserves_checkpoint_bytes(tp_rank):
    num_heads, num_kv_heads, head_dim, v_head_dim, ckpt_tp = GA
    q_rows, k_rows, v_rows = _chunk_sizes(GA)
    rows_per_chunk = q_rows + k_rows + v_rows
    weight, scale = _checkpoint(GA)
    kv_chunk_rows = _fused_qkv_kv_chunk_rows(*GA[:-1], 2, ckpt_tp)

    rank_weight, rank_scale = _shard_fp8_qkv_proj(
        weight,
        scale,
        num_heads,
        num_kv_heads,
        head_dim,
        v_head_dim,
        tp_rank,
        2,
        ckpt_tp,
        kv_chunk_rows=kv_chunk_rows,
    )

    chunks = range(tp_rank * 2, tp_rank * 2 + 2)
    padding = kv_chunk_rows - k_rows - v_rows
    expected_weight = [
        _bits(weight[chunk * rows_per_chunk : chunk * rows_per_chunk + q_rows])
        for chunk in chunks
    ]
    for chunk in chunks:
        base = chunk * rows_per_chunk
        expected_weight.extend(
            [
                _bits(weight[base + q_rows : base + rows_per_chunk]),
                torch.zeros((padding, COLS), dtype=torch.uint8),
            ]
        )
    assert torch.equal(_bits(rank_weight), torch.cat(expected_weight))

    scale_rows = math.ceil(rows_per_chunk / BLOCK)
    q_scale_rows = q_rows // BLOCK
    expected_scale = [
        scale[chunk * scale_rows : chunk * scale_rows + q_scale_rows]
        for chunk in chunks
    ] + [
        scale[chunk * scale_rows + q_scale_rows : (chunk + 1) * scale_rows]
        for chunk in chunks
    ]
    assert torch.equal(rank_scale, torch.cat(expected_scale))

    q_size = num_heads // 2 * head_dim
    k_size = num_kv_heads // 2 * head_dim
    v_size = num_kv_heads // 2 * v_head_dim
    q, k, v = _split_qkv(rank_weight.T, q_size, k_size, v_size, kv_chunk_rows)
    expected_q = torch.cat(
        [
            weight[chunk * rows_per_chunk : chunk * rows_per_chunk + q_rows]
            for chunk in chunks
        ]
    ).T
    expected_k = torch.cat(
        [
            weight[
                chunk * rows_per_chunk + q_rows : chunk * rows_per_chunk
                + q_rows
                + k_rows
            ]
            for chunk in chunks
        ]
    ).T
    expected_v = torch.cat(
        [
            weight[
                chunk * rows_per_chunk + q_rows + k_rows : (chunk + 1) * rows_per_chunk
            ]
            for chunk in chunks
        ]
    ).T
    for actual, expected in zip((q, k, v), (expected_q, expected_k, expected_v)):
        assert torch.equal(_bits(actual), _bits(expected))


@pytest.mark.parametrize("tp_rank", [0, 1])
def test_tp2_swa_is_exact_block_permutation(tp_rank):
    num_heads, num_kv_heads, head_dim, v_head_dim, ckpt_tp = SWA
    q_rows, k_rows, v_rows = _chunk_sizes(SWA)
    rows_per_chunk = q_rows + k_rows + v_rows
    weight, scale = _checkpoint(SWA)

    assert _fused_qkv_kv_chunk_rows(*SWA[:-1], 2, ckpt_tp) == 0
    rank_weight, rank_scale = _shard_fp8_qkv_proj(
        weight,
        scale,
        num_heads,
        num_kv_heads,
        head_dim,
        v_head_dim,
        tp_rank,
        2,
        ckpt_tp,
    )

    chunks = range(tp_rank * 2, tp_rank * 2 + 2)
    segments = (
        (0, q_rows),
        (q_rows, q_rows + k_rows),
        (q_rows + k_rows, rows_per_chunk),
    )
    expected_weight = torch.cat(
        [
            _bits(weight[chunk * rows_per_chunk + start : chunk * rows_per_chunk + end])
            for start, end in segments
            for chunk in chunks
        ]
    )
    assert torch.equal(_bits(rank_weight), expected_weight)

    scale_rows = rows_per_chunk // BLOCK
    expected_scale = torch.cat(
        [
            scale[
                chunk * scale_rows + start // BLOCK : chunk * scale_rows + end // BLOCK
            ]
            for start, end in segments
            for chunk in chunks
        ]
    )
    assert torch.equal(rank_scale, expected_scale)


@pytest.mark.parametrize("geometry", [GA, SWA], ids=["ga", "swa"])
@pytest.mark.parametrize("tp_rank", [0, 1, 2, 3])
def test_tp4_uses_checkpoint_chunk_directly(geometry, tp_rank):
    num_heads, num_kv_heads, head_dim, v_head_dim, ckpt_tp = geometry
    q_rows, k_rows, v_rows = _chunk_sizes(geometry)
    rows_per_chunk = q_rows + k_rows + v_rows
    weight, scale = _checkpoint(geometry)

    rank_weight, rank_scale = _shard_fp8_qkv_proj(
        weight,
        scale,
        num_heads,
        num_kv_heads,
        head_dim,
        v_head_dim,
        tp_rank,
        4,
        ckpt_tp,
    )
    scale_rows = math.ceil(rows_per_chunk / BLOCK)
    assert torch.equal(
        _bits(rank_weight),
        _bits(weight[tp_rank * rows_per_chunk : (tp_rank + 1) * rows_per_chunk]),
    )
    assert torch.equal(
        rank_scale,
        scale[tp_rank * scale_rows : (tp_rank + 1) * scale_rows],
    )


@pytest.mark.parametrize("weight_first", [False, True])
@pytest.mark.parametrize("weight_dtype", [torch.float8_e4m3fn, torch.float8_e4m3fnuz])
def test_paired_loader_handles_either_order_and_fp8_dtype(
    monkeypatch, weight_first, weight_dtype
):
    num_heads, num_kv_heads, head_dim, v_head_dim, ckpt_tp = GA
    weight, scale = _checkpoint(GA, weight_dtype)
    kv_chunk_rows = _fused_qkv_kv_chunk_rows(*GA[:-1], 2, ckpt_tp)
    expected_weight, expected_scale = _shard_fp8_qkv_proj(
        weight,
        scale,
        num_heads,
        num_kv_heads,
        head_dim,
        v_head_dim,
        0,
        2,
        ckpt_tp,
        kv_chunk_rows=kv_chunk_rows,
    )

    attention = torch.nn.Module()
    attention.total_num_heads = num_heads
    attention.total_num_kv_heads = num_kv_heads
    attention.head_dim = head_dim
    attention.v_head_dim = v_head_dim
    attention.kv_chunk_rows = kv_chunk_rows
    model = torch.nn.Module()
    model.add_module("attention", attention)

    prefix = "attention.qkv_proj"
    weight_param = torch.nn.Parameter(
        torch.empty_like(expected_weight), requires_grad=False
    )
    scale_param = torch.nn.Parameter(
        torch.empty_like(expected_scale), requires_grad=False
    )
    params = {
        f"{prefix}.weight": weight_param,
        f"{prefix}.weight_scale_inv": scale_param,
    }
    pending = {}
    config = SimpleNamespace(
        attention_projection_layout="fused_qkv",
        num_key_value_heads=ckpt_tp,
    )
    monkeypatch.setattr(
        mimo_v2,
        "get_parallel",
        lambda: SimpleNamespace(attn_tp_rank=0, attn_tp_size=2),
    )

    tensors = [
        (f"{prefix}.weight", weight),
        (f"{prefix}.weight_scale_inv", scale),
    ]
    if not weight_first:
        tensors.reverse()
    for name, tensor in tensors:
        assert try_load_mimo_v2_fp8_qkv_proj(
            model, config, name, tensor, params, pending
        )
        assert prefix in pending or name == tensors[-1][0]
    assert not pending
    assert torch.equal(_bits(weight_param), _bits(expected_weight))
    assert torch.equal(scale_param, expected_scale)


def test_paired_loader_ignores_split_layout():
    weight, _ = _checkpoint(GA)
    prefix = "attention.qkv_proj"
    params = {
        f"{prefix}.weight": torch.nn.Parameter(
            torch.empty((1, COLS), dtype=torch.float8_e4m3fn),
            requires_grad=False,
        ),
        f"{prefix}.weight_scale_inv": torch.nn.Parameter(
            torch.empty((1, 1)), requires_grad=False
        ),
    }
    pending = {}
    config = SimpleNamespace(attention_projection_layout="split")

    assert not try_load_mimo_v2_fp8_qkv_proj(
        torch.nn.Module(),
        config,
        f"{prefix}.weight",
        weight,
        params,
        pending,
    )
    assert not pending


def test_fused_qkv_chunk_resolver_returns_integer_for_all_layouts():
    block_fp8 = SimpleNamespace(weight_block_size=[128, 128])
    fused = SimpleNamespace(
        attention_projection_layout="fused_qkv", num_key_value_heads=4
    )
    split = SimpleNamespace(attention_projection_layout="split")
    unset = SimpleNamespace()

    assert get_mimo_v2_fused_qkv_chunks(fused, block_fp8) == 4
    assert get_mimo_v2_fused_qkv_chunks(split, block_fp8) == 0
    assert get_mimo_v2_fused_qkv_chunks(unset, block_fp8) == 0
    assert get_mimo_v2_fused_qkv_chunks(fused, None) == 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
