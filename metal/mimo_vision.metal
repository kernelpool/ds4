/* MiMo-V2.6 vision tower glue around kernel_qwen4_dense_mm and the Qwen3.8
 * vision helpers: fused GQA qkv rows with the 2-D rope, attention with a
 * banded window and a per-head sink on key 0, SwiGLU with biases, and the
 * merge-unit reorder of the column-window blocks.  Rows are patches in 2x2
 * merge-window order; positions travel with the rows. */

struct ds4_metal_args_mimo_vis {
    uint32_t rows;
    uint32_t width;
    uint32_t n_head;
    uint32_t n_head_kv;
    uint32_t head_dim;
    uint32_t window;     /* attention: 0 full, else |q - k| <= window */
    uint32_t has_sink;
    uint32_t n_units;    /* reorder: merge units of 4 rows */
    float    scale;
    uint32_t pad0;
    uint32_t pad1;
    uint32_t pad2;
};

/* qkv rows (+bias) -> q [rows][H*D], k [rows][Hkv*D], v [rows][Hkv*D]; q and
 * k get the 2-D rope of their row's (h, w): pair (i, i + D/2) rotates by
 * hpos*inv(i) for i < D/4 and by wpos*inv(i - D/4) above, inv(j) =
 * 10000^(-2j/(D/2)). */
kernel void kernel_mimo_vis_qkv_rope(
        constant ds4_metal_args_mimo_vis & args,
        device const float *qkv,
        device const float *bias,
        device const float *pos,       /* [rows][2] (h, w) */
        device float       *q,
        device float       *k,
        device float       *v,
        uint row [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]]) {
    if (row >= args.rows) return;
    const uint D = args.head_dim, H = args.n_head, Hkv = args.n_head_kv;
    const uint hd = D / 2, qd = D / 4, W = (H + 2 * Hkv) * D;
    const float hpos = pos[row * 2], wpos = pos[row * 2 + 1];
    device const float *src = qkv + (uint64_t)row * W;
    for (uint idx = tid; idx < (H + Hkv) * hd; idx += 256) {
        const bool is_q = idx < H * hd;
        const uint rem = is_q ? idx : idx - H * hd;
        const uint h = rem / hd, i = rem % hd;
        const uint base = (is_q ? h : H + h) * D;
        const float x0 = src[base + i] + bias[base + i];
        const float x1 = src[base + i + hd] + bias[base + i + hd];
        const uint j = i < qd ? i : i - qd;
        const float theta = (i < qd ? hpos : wpos) * pow(10000.0f, -2.0f * (float)j / (float)hd);
        const float c = cos(theta), s = sin(theta);
        device float *dst = is_q ? q + (uint64_t)row * H * D + h * D : k + (uint64_t)row * Hkv * D + h * D;
        dst[i] = x0 * c - x1 * s;
        dst[i + hd] = x0 * s + x1 * c;
    }
    for (uint i = tid; i < Hkv * D; i += 256) {
        v[(uint64_t)row * Hkv * D + i] = src[(H + Hkv) * D + i] + bias[(H + Hkv) * D + i];
    }
}

/* One simdgroup per (row, head), head_dim 32 or 64: lanes hold dims lane
 * and lane + 32.  Windowed blocks see keys within the window and add the
 * head's sink to key 0's logit when it is in range, as the reference's
 * additive mask does. */
kernel void kernel_mimo_vis_attention(
        constant ds4_metal_args_mimo_vis & args,
        device const float *q,
        device const float *k,
        device const float *v,
        device const float *sinks,
        device float       *out,
        uint2 tg [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]]) {
    const uint row = tg.x, head = tg.y;
    if (row >= args.rows) return;
    const uint D = args.head_dim, H = args.n_head, Hkv = args.n_head_kv;
    const uint kvh = head / (H / Hkv);
    const bool has1 = D > 32;
    const uint64_t qb = (uint64_t)row * H * D + head * D;
    const float q0 = q[qb + lane], q1 = has1 ? q[qb + lane + 32] : 0.0f;
    uint lo = 0, hi = args.rows - 1;
    if (args.window) {
        lo = row > args.window ? row - args.window : 0;
        hi = min(args.rows - 1, row + args.window);
    }
    float acc0 = 0.0f, acc1 = 0.0f, m = -INFINITY, denom = 0.0f;
    for (uint key = lo; key <= hi; key++) {
        const uint64_t kb = (uint64_t)key * Hkv * D + kvh * D;
        float score = simd_sum(q0 * k[kb + lane] + (has1 ? q1 * k[kb + lane + 32] : 0.0f)) * args.scale;
        if (args.has_sink && key == 0) score += sinks[head];
        const float nm = max(m, score);
        const float old = m == -INFINITY ? 0.0f : exp(m - nm);
        const float p = exp(score - nm);
        denom = denom * old + p;
        acc0 = acc0 * old + p * v[kb + lane];
        acc1 = acc1 * old + p * (has1 ? v[kb + lane + 32] : 0.0f);
        m = nm;
    }
    out[qb + lane] = acc0 / denom;
    if (has1) out[qb + lane + 32] = acc1 / denom;
}

/* out = silu(g + bg) * (u + bu) */
kernel void kernel_mimo_vis_swiglu(
        constant ds4_metal_args_mimo_vis & args,
        device const float *g,
        device const float *u,
        device const float *bg,
        device const float *bu,
        device float       *out,
        uint gid [[thread_position_in_grid]]) {
    if (gid >= args.rows * args.width) return;
    const float a = g[gid] + bg[gid % args.width];
    out[gid] = a / (1.0f + exp(-a)) * (u[gid] + bu[gid % args.width]);
}

/* dst merge unit u <- src merge unit idx[u] (4 rows each) */
kernel void kernel_mimo_vis_reorder(
        constant ds4_metal_args_mimo_vis & args,
        device const float    *src,
        device const uint32_t *idx,
        device float          *dst,
        uint row [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]]) {
    if (row >= args.n_units * 4) return;
    const uint from = idx[row / 4] * 4 + row % 4;
    device const float *s = src + (uint64_t)from * args.width;
    device float *d = dst + (uint64_t)row * args.width;
    for (uint i = tid; i < args.width; i += 256) d[i] = s[i];
}
