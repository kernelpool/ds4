"""Pure-PyTorch stand-ins for DeepSeek V4.1's six tilelang kernels.

Lets the checkpoint's own inference/model.py run unmodified on CPU, so a tiny random-weight
model can act as a parity oracle for the DS4 port.  The maths follows kernel.py exactly:
see hc_split_sinkhorn_kernel and sparse_attn_kernel there.
"""
import torch
import torch.nn.functional as F

E4M3_MAX = 448.0
# E2M1: the eight magnitudes representable in fp4
FP4_GRID = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
FP4_MAX = 6.0

# Set False to bypass the simulated quantisation and get a clean-maths oracle.
QUANTIZE = True


def _blocks(x, block):
    """Fold the last dim into [..., nblocks, block], padding when it does not divide."""
    n = x.shape[-1]
    pad = (-n) % block
    if pad:
        x = F.pad(x, (0, pad))
    return x.unflatten(-1, (-1, block)), n


def _round_pow2(s):
    """ue8m0 scales are pure powers of two."""
    return torch.exp2(torch.ceil(torch.log2(s.clamp_min(1e-30))))


def _quant_dequant(x, block, grid_max, to_fp8, pow2_scale):
    xb, n = _blocks(x.float(), block)
    amax = xb.abs().amax(dim=-1, keepdim=True)
    scale = (amax / grid_max).clamp_min(1e-30)
    if pow2_scale:
        scale = _round_pow2(scale)
    q = xb / scale
    if to_fp8:
        q = q.to(torch.float8_e4m3fn).float()
    else:
        g = FP4_GRID.to(q.device)
        sign = torch.sign(q)
        mag = q.abs().clamp(max=FP4_MAX)
        idx = (mag.unsqueeze(-1) - g).abs().argmin(dim=-1)
        q = sign * g[idx]
    return (q * scale).flatten(-2)[..., :n], scale.squeeze(-1)


def act_quant(x, block_size, scale_fmt=None, scale_dtype=None, inplace=False):
    """fp8 e4m3 with one scale per `block_size` values along the last dim."""
    if not QUANTIZE:
        return (x, None) if not inplace else None
    y, s = _quant_dequant(x, block_size, E4M3_MAX, True, scale_fmt == "ue8m0")
    if inplace:
        x.copy_(y.to(x.dtype))
        return None
    return y.to(x.dtype), s


def fp4_act_quant(x, block_size, inplace=False, scale_dtype=None):
    """fp4 e2m1.  The compressed KV passes block 16 with e4m3 scales, the indexer block 32 ue8m0."""
    if not QUANTIZE:
        return (x, None) if not inplace else None
    # only the ue8m0 path rounds the scale to a power of two
    pow2 = scale_dtype is None or getattr(scale_dtype, "name", "") == "float8_e8m0fnu"
    y, s = _quant_dequant(x, block_size, FP4_MAX, False, pow2)
    if inplace:
        x.copy_(y.to(x.dtype))
        return None
    return y.to(x.dtype), s


def fp8_gemm(x, xs, w, ws, scale_dtype, block_size=None, act_block_size=None):
    raise NotImplementedError("bf16 oracle: quantised GEMMs are not exercised")


fp4_gemm = fp8_gemm


def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, eps=1e-6):
    """Split one projection into pre / post / comb, comb made doubly stochastic by Sinkhorn."""
    hc = hc_mult
    m = mixes.float()
    pre = torch.sigmoid(m[..., :hc] * hc_scale[0] + hc_base[:hc]) + eps
    post = 2.0 * torch.sigmoid(m[..., hc:2 * hc] * hc_scale[1] + hc_base[hc:2 * hc])
    comb = m[..., 2 * hc:] * hc_scale[2] + hc_base[2 * hc:]
    comb = comb.unflatten(-1, (hc, hc))

    comb = comb.softmax(dim=-1) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    return pre, post, comb


def sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale):
    """Gathered attention over topk_idxs; -1 selects nothing.  attn_sink is a per-head logit that
    enters only the denominator, so a row with no valid index comes out all zero."""
    b, s, h, d = q.shape
    idxs = topk_idxs.long()
    valid = idxs >= 0
    gather = idxs.clamp_min(0)
    bi = torch.arange(b, device=q.device).view(b, 1, 1)
    kvg = kv[bi, gather].float()                                  # [b, s, topk, d]

    scores = torch.einsum("bshd,bstd->bsht", q.float(), kvg) * softmax_scale
    scores = scores.masked_fill(~valid.unsqueeze(2), float("-inf"))
    # kernel.py seeds the running max with a finite -1e30 so an all-invalid row is zero, not NaN
    mx = scores.amax(dim=-1, keepdim=True).clamp_min(-1e30)
    e = torch.exp(scores - mx)
    denom = e.sum(dim=-1) + torch.exp(attn_sink.float().view(1, 1, h) - mx.squeeze(-1))
    o = torch.einsum("bsht,bstd->bshd", e, kvg) / denom.unsqueeze(-1)
    return o.to(q.dtype)
