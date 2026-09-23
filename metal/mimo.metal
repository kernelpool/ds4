/* MiMo-V2.6 kernels: fused-qkv split with partial NeoX rope and the K/V ring
 * writes, and GQA attention with per-head sinks and a sliding window over
 * key splits.  Transients are f32, caches half; the qk width (head_dim) and
 * the value width differ, so the decode tile carries both per-lane strides.
 * Math mirrors the mimo_ref_* scalar reference in ds4.c. */

struct ds4_metal_args_mimo_attn_prep {
    uint32_t n_tokens;
    uint32_t n_head;
    uint32_t n_head_kv;
    uint32_t head_dim;
    uint32_t value_dim;
    uint32_t n_rot;
    uint32_t pos0;
    uint32_t ring;          /* cache rows; position p lives in row p % ring */
    float    v_scale;
    uint32_t pad0;
    uint32_t pad1;
    uint32_t pad2;
    float    rope_freq[32]; /* per-pair inverse frequencies */
};

/* NeoX rotation of the first n_rot dims of one head: pair (i, i + n_rot/2). */
static inline void mimo_rope_neox(thread float *x, uint n_rot, uint pos, constant float *freq) {
    const uint nh = n_rot / 2;
    for (uint i = 0; i < nh; i++) {
        const float theta = (float)pos * freq[i];
        const float c = cos(theta), s = sin(theta);
        const float x0 = x[i], x1 = x[i + nh];
        x[i] = x0 * c - x1 * s;
        x[i + nh] = x0 * s + x1 * c;
    }
}

/* One (head slot, token) per 32-lane threadgroup: q heads are roped into
 * q_out, kv heads are roped (k) / scaled (v) into the caches. */
kernel void kernel_mimo_attn_prep(
        constant ds4_metal_args_mimo_attn_prep & args,
        device const float *qkv,       /* [T][H*Dk + Hkv*Dk + Hkv*Dv] */
        device float       *q_out,     /* [T][H*Dk] */
        device half        *k_cache,   /* [ring][Hkv*Dk] */
        device half        *v_cache,   /* [ring][Hkv*Dv] */
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const uint slot = tgpig.x;
    const uint tok = tgpig.y;
    if (tok >= args.n_tokens) return;
    const uint H = args.n_head, Hkv = args.n_head_kv, Dk = args.head_dim, Dv = args.value_dim;
    const uint pos = args.pos0 + tok;
    const uint row = pos % args.ring;
    const uint width = H * Dk + Hkv * Dk + Hkv * Dv;
    device const float *base = qkv + (uint64_t)tok * width;
    threadgroup float stage[576];
    const uint npt = Dk / 32;
    if (slot < H) {
        device const float *src = base + slot * Dk;
        for (uint i = 0; i < npt; i++) stage[tiisg * npt + i] = src[tiisg * npt + i];
        simdgroup_barrier(mem_flags::mem_threadgroup);
        if (tiisg == 0) {
            float tmp[64];
            for (uint i = 0; i < args.n_rot; i++) tmp[i] = stage[i];
            mimo_rope_neox(tmp, args.n_rot, pos, args.rope_freq);
            for (uint i = 0; i < args.n_rot; i++) stage[i] = tmp[i];
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        device float *dst = q_out + ((uint64_t)tok * H + slot) * Dk;
        for (uint i = 0; i < npt; i++) dst[tiisg * npt + i] = stage[tiisg * npt + i];
        return;
    }
    const uint h = slot - H;
    if (h >= Hkv) return;
    device const float *ks = base + H * Dk + h * Dk;
    device const float *vs = base + H * Dk + Hkv * Dk + h * Dv;
    for (uint i = 0; i < npt; i++) stage[tiisg * npt + i] = ks[tiisg * npt + i];
    simdgroup_barrier(mem_flags::mem_threadgroup);
    if (tiisg == 0) {
        float tmp[64];
        for (uint i = 0; i < args.n_rot; i++) tmp[i] = stage[i];
        mimo_rope_neox(tmp, args.n_rot, pos, args.rope_freq);
        for (uint i = 0; i < args.n_rot; i++) stage[i] = tmp[i];
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    device half *dk = k_cache + ((uint64_t)row * Hkv + h) * Dk;
    device half *dv = v_cache + ((uint64_t)row * Hkv + h) * Dv;
    for (uint i = 0; i < npt; i++) dk[tiisg * npt + i] = (half)stage[tiisg * npt + i];
    const uint npv = Dv / 32;
    for (uint i = 0; i < npv; i++) dv[tiisg * npv + i] = (half)(vs[tiisg * npv + i] * args.v_scale);
}

/* --- attention --------------------------------------------------------- */

struct ds4_metal_args_mimo_attn {
    uint32_t n_tokens;
    uint32_t n_head;
    uint32_t n_head_kv;
    uint32_t head_dim;
    uint32_t value_dim;
    uint32_t pos0;
    uint32_t ring;
    uint32_t n_swa;          /* 0: every cached key; else the last n_swa positions */
    uint32_t first;          /* first cached position (an MTP cache starts late) */
    uint32_t has_sink;
    uint32_t n_splits;       /* key ranges per (token, kv head); > 1 writes partials */
    uint32_t keys_per_split;
    float    scale;
    uint32_t hi_end;         /* 0: causal (keys <= the query); else keys below hi_end (DFlash block) */
    uint32_t fuse_prep;      /* one token: the kernel ropes q and writes the K/V row itself */
    uint32_t n_rot;
    float    v_scale;
    uint32_t pad3;
    float    rope_freq[32];
};

#define MIMO_ATTN_NSG 4          /* simdgroups per threadgroup */
#define MIMO_ATTN_HPS 4          /* q heads per simdgroup: group <= NSG * HPS */

static inline uint mimo_attn_lo(constant ds4_metal_args_mimo_attn &args, uint pos) {
    uint lo = args.n_swa && pos + 1u > args.n_swa ? pos + 1u - args.n_swa : 0u;
    return max(lo, args.first);
}

/* One (key split, kv head, token): the simdgroups share the K/V rows and
 * own disjoint query heads; lane j owns qk dims j*NPTK.. and value dims
 * j*NPTV.. (template constants keep the per-lane arrays in registers).
 * The sink enters split 0 as an extra logit whose value is dropped. */
template <uint NPTK, uint NPTV>
static inline void mimo_attn_tile(
        constant ds4_metal_args_mimo_attn &args, uint split, uint kvh, uint tok,
        device const float *q, device const half *k_cache, device const half *v_cache,
        device const float *sinks, device float *out, device float *part,
        ushort sgitg, ushort tiisg) {
    const uint H = args.n_head, Hkv = args.n_head_kv;
    constexpr uint Dk = NPTK * 32, Dv = NPTV * 32;
    const uint group = H / Hkv;
    const uint hps = (group + MIMO_ATTN_NSG - 1) / MIMO_ATTN_NSG;
    const uint g0 = (uint)sgitg * hps;
    if (g0 >= group) return;
    const uint ng = min(hps, group - g0);
    const uint pos = args.pos0 + tok;
    const uint lo = mimo_attn_lo(args, pos);
    const uint k0 = lo + split * args.keys_per_split;
    const uint k1 = min(args.hi_end ? args.hi_end : pos + 1u, k0 + args.keys_per_split);

    float qv[MIMO_ATTN_HPS][NPTK];
    float m[MIMO_ATTN_HPS], l[MIMO_ATTN_HPS], acc[MIMO_ATTN_HPS][NPTV];
#pragma unroll
    for (uint g = 0; g < MIMO_ATTN_HPS; g++) {
        const uint h = kvh * group + g0 + min(g, ng - 1u);
        device const float *qh = q + ((uint64_t)tok * H + h) * Dk + tiisg * NPTK;
#pragma unroll
        for (uint i = 0; i < NPTK; i++) qv[g][i] = qh[i] * args.scale;
        const bool sink = args.has_sink && split == 0;
        m[g] = sink ? sinks[h] : -3.0e38f;
        l[g] = sink ? 1.0f : 0.0f;
#pragma unroll
        for (uint i = 0; i < NPTV; i++) acc[g][i] = 0.0f;
    }
    for (uint p = k0; p < k1; p++) {
        const uint row = p % args.ring;
        device const half *kr = k_cache + ((uint64_t)row * Hkv + kvh) * Dk + tiisg * NPTK;
        device const half *vr = v_cache + ((uint64_t)row * Hkv + kvh) * Dv + tiisg * NPTV;
        float kv[NPTK], vv[NPTV];
#pragma unroll
        for (uint i = 0; i < NPTK; i++) kv[i] = (float)kr[i];
#pragma unroll
        for (uint i = 0; i < NPTV; i++) vv[i] = (float)vr[i];
#pragma unroll
        for (uint g = 0; g < MIMO_ATTN_HPS; g++) {
            if (g < ng) {
                float s = 0.0f;
#pragma unroll
                for (uint i = 0; i < NPTK; i++) s += qv[g][i] * kv[i];
                s = simd_sum(s);
                const float m_new = max(m[g], s);
                const float corr = exp(m[g] - m_new);
                const float w = exp(s - m_new);
                l[g] = l[g] * corr + w;
#pragma unroll
                for (uint i = 0; i < NPTV; i++) acc[g][i] = acc[g][i] * corr + w * vv[i];
                m[g] = m_new;
            }
        }
    }
#pragma unroll
    for (uint g = 0; g < MIMO_ATTN_HPS; g++) {
        if (g >= ng) break;
        const uint h = kvh * group + g0 + g;
        if (args.n_splits == 1) {
            device float *dst = out + ((uint64_t)tok * H + h) * Dv + tiisg * NPTV;
            const float inv = l[g] > 0.0f ? 1.0f / l[g] : 0.0f;
#pragma unroll
            for (uint i = 0; i < NPTV; i++) dst[i] = acc[g][i] * inv;
        } else {
            device float *dst = part + ((((uint64_t)tok * Hkv + kvh) * args.n_splits + split) * group + g0 + g) * (2u + Dv);
            if (tiisg == 0) { dst[0] = m[g]; dst[1] = l[g]; }
#pragma unroll
            for (uint i = 0; i < NPTV; i++) dst[2u + tiisg * NPTV + i] = acc[g][i];
        }
    }
}

/* One head's prep as kernel_mimo_attn_prep does it, one rope pair per lane
 * (the pairs are independent, so the arithmetic is the same); every split
 * threadgroup of a kv head writes the same values, so the duplicate writes
 * are harmless. */
template <typename T>
static inline void mimo_prep_head(device const float *src, device T *dst, uint dim, uint n_rot, uint pos,
                                  constant float *freq, float scale, ushort tiisg) {
    const uint nh = n_rot / 2;
    if (tiisg < nh) {
        const float theta = (float)pos * freq[tiisg];
        const float c = cos(theta), s = sin(theta);
        const float x0 = src[tiisg], x1 = src[tiisg + nh];
        dst[tiisg] = (T)(x0 * c - x1 * s);
        dst[tiisg + nh] = (T)(x0 * s + x1 * c);
    }
    for (uint i = n_rot + tiisg; i < dim; i += 32) dst[i] = (T)(src[i] * scale);
}

template <uint NPTK, uint NPTV>
kernel void kernel_mimo_attn(
        constant ds4_metal_args_mimo_attn & args,
        device float       *q,          /* [T][H*Dk] */
        device half        *k_cache,    /* [ring][Hkv*Dk] */
        device half        *v_cache,    /* [ring][Hkv*Dv] */
        device const float *sinks,      /* [H] */
        device float       *out,        /* [T][H*Dv] */
        device float       *part,       /* [T][Hkv][n_splits][group][2+Dv] */
        device const float *qkv,        /* [1][H*Dk + Hkv*Dk + Hkv*Dv] when fuse_prep */
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const uint split = tgpig.x;
    const uint kvh = tgpig.y;
    const uint tok = tgpig.z;
    if (split >= args.n_splits || kvh >= args.n_head_kv || tok >= args.n_tokens) return;
    if (args.fuse_prep) {
        const uint H = args.n_head, Hkv = args.n_head_kv, Dk = args.head_dim, Dv = args.value_dim;
        const uint group = H / Hkv, pos = args.pos0, row = pos % args.ring;
        for (uint g = sgitg; g < group; g += MIMO_ATTN_NSG) {
            const uint h = kvh * group + g;
            mimo_prep_head<float>(qkv + h * Dk, q + h * Dk, Dk, args.n_rot, pos, args.rope_freq, 1.0f, tiisg);
        }
        if (sgitg == 0) {
            mimo_prep_head<half>(qkv + H * Dk + kvh * Dk, k_cache + ((uint64_t)row * Hkv + kvh) * Dk, Dk, args.n_rot,
                                 pos, args.rope_freq, 1.0f, tiisg);
        } else if (sgitg == 1) {
            device const float *vs = qkv + H * Dk + Hkv * Dk + kvh * Dv;
            device half *dv = v_cache + ((uint64_t)row * Hkv + kvh) * Dv;
            for (uint i = tiisg; i < Dv; i += 32) dv[i] = (half)(vs[i] * args.v_scale);
        }
        threadgroup_barrier(mem_flags::mem_device);
    }
    mimo_attn_tile<NPTK, NPTV>(args, split, kvh, tok, q, k_cache, v_cache, sinks, out, part, sgitg, tiisg);
}

/* Merge the split partials of one (token, head). */
template <uint NPTV>
kernel void kernel_mimo_attn_merge(
        constant ds4_metal_args_mimo_attn & args,
        device const float *part,
        device float       *out,
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const uint h = tgpig.x;
    const uint tok = tgpig.y;
    if (h >= args.n_head || tok >= args.n_tokens) return;
    const uint H = args.n_head, Hkv = args.n_head_kv;
    constexpr uint Dv = NPTV * 32;
    const uint group = H / Hkv;
    const uint kvh = h / group, g = h % group;
    const uint64_t stride = (uint64_t)group * (2u + Dv);
    device const float *base = part + (((uint64_t)tok * Hkv + kvh) * args.n_splits * group + g) * (2u + Dv);
    float mm = -3.0e38f;
    for (uint s = 0; s < args.n_splits; s++) mm = max(mm, base[s * stride]);
    float ll = 0.0f;
    float o[NPTV];
#pragma unroll
    for (uint i = 0; i < NPTV; i++) o[i] = 0.0f;
    for (uint s = 0; s < args.n_splits; s++) {
        device const float *p = base + s * stride;
        const float c = p[1] > 0.0f ? exp(p[0] - mm) : 0.0f;
        ll += p[1] * c;
#pragma unroll
        for (uint i = 0; i < NPTV; i++) o[i] += p[2u + tiisg * NPTV + i] * c;
    }
    const float inv = ll > 0.0f ? 1.0f / ll : 0.0f;
    device float *dst = out + ((uint64_t)tok * H + h) * Dv + tiisg * NPTV;
#pragma unroll
    for (uint i = 0; i < NPTV; i++) dst[i] = o[i] * inv;
}

#define MIMO_ATTN_INSTANCE(NPTK_, NPTV_) \
template [[host_name("kernel_mimo_attn_k" #NPTK_ "v" #NPTV_)]] \
kernel void kernel_mimo_attn<NPTK_, NPTV_>(constant ds4_metal_args_mimo_attn &, device float *, \
        device half *, device half *, device const float *, device float *, device float *, \
        device const float *, uint3, ushort, ushort);
MIMO_ATTN_INSTANCE(6, 4)   /* MiMo-V2.6: qk 192, value 128 */
MIMO_ATTN_INSTANCE(3, 2)   /* mini: qk 96, value 64 */
MIMO_ATTN_INSTANCE(4, 4)   /* DFlash drafter: 128/128 */
MIMO_ATTN_INSTANCE(1, 1)   /* mini drafter: 32/32 */

#define MIMO_MERGE_INSTANCE(NPTV_) \
template [[host_name("kernel_mimo_attn_merge_v" #NPTV_)]] \
kernel void kernel_mimo_attn_merge<NPTV_>(constant ds4_metal_args_mimo_attn &, device const float *, \
        device float *, uint3, ushort);
MIMO_MERGE_INSTANCE(4)
MIMO_MERGE_INSTANCE(2)
MIMO_MERGE_INSTANCE(1)

/* --- router ------------------------------------------------------------- */

struct ds4_metal_args_mimo_router {
    uint32_t n_tokens;
    uint32_t n_expert;
    uint32_t n_used;
    uint32_t own_lo;    /* TP: this rank's experts [own_lo, own_lo + own_n); own_n 0 owns all */
    uint32_t own_n;
    uint32_t pad0, pad1, pad2;
};

/* Sigmoid router for one token per threadgroup: experts are picked by
 * sigmoid(logit) + bias (ties to the lower index, as the reference), the
 * weights are the unbiased probabilities renormalised over the picks. */
kernel void kernel_mimo_router(
        constant ds4_metal_args_mimo_router & args,
        device const float *logits,     /* [T][n_expert] */
        device const float *bias,       /* [n_expert] */
        device int32_t     *selected,   /* [T][n_used] */
        device float       *weights,    /* [T][n_used] */
        uint tok [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    if (tok >= args.n_tokens) return;
    threadgroup float red_s[8];
    threadgroup uint red_i[8];
    threadgroup float prob[512];
    threadgroup uint taken[512];
    device const float *lg = logits + (uint64_t)tok * args.n_expert;
    for (uint e = tid; e < args.n_expert; e += 256u) {
        prob[e] = qwen4_sigmoid(lg[e]);
        taken[e] = 0u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float wsum = 0.0f;
    for (uint k = 0; k < args.n_used; k++) {
        float best = -INFINITY;
        uint besti = 0xffffffffu;
        for (uint e = tid; e < args.n_expert; e += 256u) {
            const float s = prob[e] + bias[e];
            if (!taken[e] && (s > best || (s == best && e < besti))) { best = s; besti = e; }
        }
        /* reduce (best, besti) over the threadgroup, lower index on ties */
        for (uint off = 16; off > 0; off >>= 1) {
            const float ob = simd_shuffle_xor(best, off);
            const uint oi = simd_shuffle_xor(besti, off);
            if (ob > best || (ob == best && oi < besti)) { best = ob; besti = oi; }
        }
        if (tiisg == 0) { red_s[sgitg] = best; red_i[sgitg] = besti; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            float b = red_s[0];
            uint bi = red_i[0];
            for (uint q = 1; q < 8; q++) {
                if (red_s[q] > b || (red_s[q] == b && red_i[q] < bi)) { b = red_s[q]; bi = red_i[q]; }
            }
            taken[bi] = 1u;
            selected[(uint64_t)tok * args.n_used + k] = (int32_t)bi;
            red_i[0] = bi;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        wsum += prob[red_i[0]];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid < args.n_used) {
        const uint64_t at = (uint64_t)tok * args.n_used + tid;
        const uint e = (uint)selected[at];
        float w = prob[e] / (wsum + 1e-20f);
        if (args.own_n) {   /* owned picks rebased, a peer's marked -1 with no weight */
            const uint r = e - args.own_lo;
            selected[at] = r < args.own_n ? (int32_t)r : -1;
            if (r >= args.own_n) w = 0.0f;
        }
        weights[at] = w;
    }
}

/* --- MTP input --------------------------------------------------------- */

struct ds4_metal_args_mimo_mtp_cat {
    uint32_t n_tokens;
    uint32_t n_embd;
    float    eps;
    uint32_t pad0;
};

/* cat[t] = [ rms(e[t]) * g_e | rms(h[t]) * g_h ]: the eh_proj input of one MTP row */
kernel void kernel_mimo_mtp_cat(
        constant ds4_metal_args_mimo_mtp_cat & args,
        device const float *e,       /* [T][E] token embeddings */
        device const float *h,       /* [T][E] previous-depth hidden */
        device const float *g_e,     /* [E] */
        device const float *g_h,     /* [E] */
        device float       *cat,     /* [T][2E] */
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort3 ntg [[threads_per_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const uint tok = tgpig.x;
    const uint half_id = tgpig.y;   /* 0: embedding half, 1: hidden half */
    if (tok >= args.n_tokens) return;
    const uint E = args.n_embd, nth = ntg.x, nsg = nth / 32;
    threadgroup float red[32];
    device const float *src = (half_id ? h : e) + (uint64_t)tok * E;
    device const float *g = half_id ? g_h : g_e;
    device float *dst = cat + (uint64_t)tok * 2u * E + half_id * E;
    float ss = 0.0f;
    for (uint i = tid; i < E; i += nth) ss += src[i] * src[i];
    ss = simd_sum(ss);
    if (tiisg == 0) red[sgitg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0.0f;
    for (uint q = 0; q < nsg; q++) tot += red[q];
    const float inv = rsqrt(tot / (float)E + args.eps);
    for (uint i = tid; i < E; i += nth) dst[i] = src[i] * inv * g[i];
}

/* --- DFlash drafter ------------------------------------------------------ */

struct ds4_metal_args_mimo_dflash_prep {
    uint32_t n_tokens;
    uint32_t n_head;        /* 0: context rows (K/V only) */
    uint32_t n_head_kv;
    uint32_t head_dim;
    uint32_t value_dim;
    uint32_t n_rot;
    uint32_t pos0;
    uint32_t ring;
    float    v_scale;
    float    eps;
    uint32_t pad0;
    uint32_t pad1;
    float    rope_freq[32];
};

/* Per-head RMSNorm (q_norm / k_norm) then the partial rope on one head
 * held in a 32-lane stage; the lanes own consecutive dims. */
static inline void mimo_head_norm(threadgroup float *stage, uint dim, device const float *g, float eps,
                                  ushort tiisg) {
    const uint npt = dim / 32;
    float ss = 0.0f;
    for (uint i = 0; i < npt; i++) ss += stage[tiisg * npt + i] * stage[tiisg * npt + i];
    ss = simd_sum(ss);
    const float inv = rsqrt(ss / (float)dim + eps);
    for (uint i = 0; i < npt; i++) stage[tiisg * npt + i] *= inv * g[tiisg * npt + i];
}

/* Qwen3-style attention inputs from separate q/k/v projections: q heads
 * normed and roped into q_out, k heads normed and roped into the ring, v
 * scaled into the ring.  One (head slot, token) per 32-lane threadgroup. */
kernel void kernel_mimo_dflash_prep(
        constant ds4_metal_args_mimo_dflash_prep & args,
        device const float *q,         /* [T][H*Dk] */
        device const float *k,         /* [T][Hkv*Dk] */
        device const float *v,         /* [T][Hkv*Dv] */
        device const float *q_norm,    /* [Dk] */
        device const float *k_norm,    /* [Dk] */
        device float       *q_out,     /* [T][H*Dk] */
        device half        *k_cache,   /* [ring][Hkv*Dk] */
        device half        *v_cache,   /* [ring][Hkv*Dv] */
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const uint slot = tgpig.x;
    const uint tok = tgpig.y;
    if (tok >= args.n_tokens) return;
    const uint H = args.n_head, Hkv = args.n_head_kv, Dk = args.head_dim, Dv = args.value_dim;
    const uint pos = args.pos0 + tok;
    const uint row = pos % args.ring;
    threadgroup float stage[256];
    const uint npt = Dk / 32;
    const bool is_q = slot < H;
    const uint h = is_q ? slot : slot - H;
    if (!is_q && h >= Hkv) return;
    device const float *src = is_q ? q + ((uint64_t)tok * H + h) * Dk : k + ((uint64_t)tok * Hkv + h) * Dk;
    for (uint i = 0; i < npt; i++) stage[tiisg * npt + i] = src[tiisg * npt + i];
    simdgroup_barrier(mem_flags::mem_threadgroup);
    mimo_head_norm(stage, Dk, is_q ? q_norm : k_norm, args.eps, tiisg);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    if (tiisg == 0) {
        float tmp[64];
        for (uint i = 0; i < args.n_rot; i++) tmp[i] = stage[i];
        mimo_rope_neox(tmp, args.n_rot, pos, args.rope_freq);
        for (uint i = 0; i < args.n_rot; i++) stage[i] = tmp[i];
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    if (is_q) {
        device float *dst = q_out + ((uint64_t)tok * H + h) * Dk;
        for (uint i = 0; i < npt; i++) dst[tiisg * npt + i] = stage[tiisg * npt + i];
        return;
    }
    device half *dk = k_cache + ((uint64_t)row * Hkv + h) * Dk;
    device half *dv = v_cache + ((uint64_t)row * Hkv + h) * Dv;
    device const float *vs = v + ((uint64_t)tok * Hkv + h) * Dv;
    for (uint i = 0; i < npt; i++) dk[tiisg * npt + i] = (half)stage[tiisg * npt + i];
    const uint npv = Dv / 32;
    for (uint i = 0; i < npv; i++) dv[tiisg * npv + i] = (half)(vs[tiisg * npv + i] * args.v_scale);
}

struct ds4_metal_args_mimo_scatter_cols {
    uint32_t n_tokens;
    uint32_t width;
    uint32_t dst_stride;
    uint32_t dst_col;
};

/* dst[t][dst_col + i] = src[t][i]: one residual stream into its slot of the
 * concatenated DFlash feature rows */
kernel void kernel_mimo_scatter_cols(
        constant ds4_metal_args_mimo_scatter_cols & args,
        device const float *src,
        device float       *dst,
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort3 ntg [[threads_per_threadgroup]]) {
    const uint tok = tgpig.x;
    if (tok >= args.n_tokens) return;
    device const float *s = src + (uint64_t)tok * args.width;
    device float *d = dst + (uint64_t)tok * args.dst_stride + args.dst_col;
    for (uint i = tid; i < args.width; i += ntg.x) d[i] = s[i];
}
