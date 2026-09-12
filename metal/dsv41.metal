// DeepSeek V4.1 Flash kernels.
//
// The engram gate closes the block whose wkv projection runs through the shared
// BF16 mat-vec: a normalised dot of the residual stream against the looked-up key,
// signed-sqrt then sigmoid, scaling one shared value into every hc copy. The n-gram
// rows themselves are gathered on the host, because the tables are 203 GB and are
// mapped without ever being made resident.

struct dsv41_engram_gate_args {
    uint dim;           // n_embd
    uint hc;            // residual-stream copies
    float eps;          // norm_eps, 1e-20 in the release
    uint rows;          // tokens, tgpig.y
    uint use_dead;      // read dead_rows; an image-span token shuts the gate
};

// out[t][c][j] = x[t][c][j] + gate[t][c] * value[t][j], one threadgroup per (token, copy).
kernel void kernel_dsv41_engram_gate(
        constant dsv41_engram_gate_args &args,
        device const float              *x,        // [rows, hc, dim]
        device const float              *kv,       // [rows][hc * dim keys, then dim value]
        device const float              *q_weight, // [hc, dim]
        device const float              *k_weight, // [hc, dim]
        device float                    *gate_out, // [rows, hc]
        device float                    *out,      // [rows, hc, dim]
        device const int                *dead_rows,// [rows]
        uint2  tgpig [[threadgroup_position_in_grid]],
        uint2  tpitg [[thread_position_in_threadgroup]],
        uint2  ntg   [[threads_per_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg   [[simdgroup_index_in_threadgroup]],
        ushort nsg  [[simdgroups_per_threadgroup]]) {
    const uint c = tgpig.x;
    const uint t = tgpig.y;
    if (c >= args.hc || t >= args.rows) return;

    const uint dim = args.dim;
    const ulong stream = (ulong)args.hc * dim;
    device const float *xc = x + t * stream + (ulong)c * dim;
    device const float *kc = kv + t * (stream + dim) + (ulong)c * dim;
    device const float *qw = q_weight + (ulong)c * dim;
    device const float *kw = k_weight + (ulong)c * dim;
    device const float *val = kv + t * (stream + dim) + stream;
    device float *o = out + t * stream + (ulong)c * dim;
    const bool dead = args.use_dead != 0u && dead_rows[t] != 0;

    float xs = 0.0f, ks = 0.0f, dot = 0.0f;
    for (uint j = tpitg.x; j < dim; j += ntg.x) {
        const float xv = xc[j];
        const float kvv = kc[j];
        xs = fma(xv, xv, xs);
        ks = fma(kvv, kvv, ks);
        // only ever used as the product, so the two weights fold together first
        dot = fma(xv * kvv, qw[j] * kw[j], dot);
    }

    threadgroup float shared[3 * 32];
    xs = simd_sum(xs);
    ks = simd_sum(ks);
    dot = simd_sum(dot);
    if (lane == 0u) {
        shared[sg] = xs;
        shared[32u + sg] = ks;
        shared[64u + sg] = dot;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float gate_share;
    if (tpitg.x == 0u) {
        float xt = 0.0f, kt = 0.0f, dt = 0.0f;
        for (ushort i = 0; i < nsg; i++) {
            xt += shared[i];
            kt += shared[32u + i];
            dt += shared[64u + i];
        }
        const float rstd = (1.0f / sqrt(xt / (float)dim + args.eps)) *
                           (1.0f / sqrt(kt / (float)dim + args.eps));
        float v = dt * rstd / sqrt((float)dim);
        float a = fabs(v);
        a = a < 1e-6f ? 1e-6f : a;
        // signed sqrt before the sigmoid, matching the training kernel
        const float s = copysign(sqrt(a), v);
        float g = 1.0f / (1.0f + exp(-s));
        if (dead) g = 0.0f;
        gate_share = g;
        gate_out[(ulong)t * args.hc + c] = g;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float g = gate_share;
    for (uint j = tpitg.x; j < dim; j += ntg.x) {
        o[j] = fma(g, val[j], xc[j]);
    }
}

// Sparse attention over a gathered index list.  MQA: one latent per position is both
// K and V for every query head, so the loop reads each candidate row once.  Entries of
// -1 gather nothing, and the per-head sink is a logit that enters the softmax
// denominator only -- never the weighted sum.
//
// One simdgroup owns one (query, head); lanes stride the head dimension so the candidate
// loads stay coalesced, and the running softmax keeps the whole output in registers.

#define DSV41_ATTN_MAX_PER_LANE 16u   // head_dim <= 512

struct dsv41_sparse_attn_args {
    uint n_window;      // ids below this index the window ring, the rest the compressed cache
    uint n_head;
    uint head_dim;
    uint topk;          // candidates per query, over both index arrays
    uint n_idx_win;     // the first n_idx_win come from idxs, the rest from idx_cmp
    float scale;
};

// Candidates are split across the simdgroups of one threadgroup, each keeping its own
// running (max, denominator, accumulator), and the partials are combined at the end.  A
// decode step has a single query, so without this the grid is only n_head simdgroups and
// the machine sits idle; splitting recovers the occupancy.
#define DSV41_ATTN_NSG 8u

kernel void kernel_dsv41_sparse_attn(
        constant dsv41_sparse_attn_args &args,
        device const float              *q,     // [s_len, n_head, head_dim]
        device const float              *kv,    // [n_window, head_dim], the window ring
        device const float              *kv_cmp,// [*, head_dim], the compressed cache
        device const float              *sink,  // [n_head]
        device const int                *idxs,  // [s_len, n_idx_win], -1 gathers nothing
        device float                    *out,   // [s_len, n_head, head_dim]
        threadgroup float               *shared [[threadgroup(0)]],
        device const int                *idx_cmp, // [s_len, topk - n_idx_win]
        uint2  tgpig [[threadgroup_position_in_grid]],
        ushort lane  [[thread_index_in_simdgroup]],
        ushort sg    [[simdgroup_index_in_threadgroup]],
        ushort nsg   [[simdgroups_per_threadgroup]]) {
    const uint h = tgpig.x;
    const uint s = tgpig.y;
    if (h >= args.n_head) return;

    const uint d = args.head_dim;
    const uint dpl = d / 32u;                       // elements per lane
    if (dpl == 0u || dpl > DSV41_ATTN_MAX_PER_LANE) return;

    device const float *qv = q + ((ulong)s * args.n_head + h) * d;
    const uint n_win = args.n_idx_win;
    device const int *idx = idxs + (ulong)s * n_win;
    device const int *idc = idx_cmp + (ulong)s * (args.topk - n_win);

    float qreg[DSV41_ATTN_MAX_PER_LANE];
    float acc[DSV41_ATTN_MAX_PER_LANE];
    for (uint i = 0; i < dpl; i++) {
        qreg[i] = qv[i * 32u + lane];
        acc[i] = 0.0f;
    }

    // seeded at a finite sentinel, so a row with no reachable candidate stays well defined
    float m = -1.0e30f;
    float l = 0.0f;
    uint valid = 0u;

    for (uint t = sg; t < args.topk; t += nsg) {
        const int id = t < n_win ? idx[t] : idc[t - n_win];
        if (id < 0) continue;
        // the two caches stay where they are; gathering them into one array cost a blit
        // and a 256 KB copy per block, and a blit ends the compute encoder
        device const float *kvv = (uint)id < args.n_window
                                ? kv + (ulong)id * d
                                : kv_cmp + (ulong)((uint)id - args.n_window) * d;

        float p = 0.0f;
        for (uint i = 0; i < dpl; i++) p = fma(qreg[i], kvv[i * 32u + lane], p);
        p = simd_sum(p) * args.scale;

        const float mnew = max(m, p);
        const float corr = exp(m - mnew);           // 0 on the first valid candidate
        const float e = exp(p - mnew);
        l = fma(l, corr, e);
        for (uint i = 0; i < dpl; i++) acc[i] = fma(acc[i], corr, e * kvv[i * 32u + lane]);
        m = mnew;
        valid++;
    }

    // shared layout: [nsg][d] accumulators, then nsg maxima, denominators and valid counts
    threadgroup float *part = shared;
    threadgroup float *pm = shared + (uint)nsg * d;
    threadgroup float *pl = pm + nsg;
    threadgroup float *pv = pl + nsg;
    for (uint i = 0; i < dpl; i++) part[(uint)sg * d + i * 32u + lane] = acc[i];
    if (lane == 0u) {
        pm[sg] = valid ? m : -1.0e30f;
        pl[sg] = l;
        pv[sg] = (float)valid;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg != 0u) return;

    float M = -1.0e30f;
    float total_valid = 0.0f;
    for (ushort i = 0; i < nsg; i++) {
        M = max(M, pm[i]);
        total_valid += pv[i];
    }
    device float *o = out + ((ulong)s * args.n_head + h) * d;
    if (total_valid == 0.0f) {
        for (uint i = 0; i < dpl; i++) o[i * 32u + lane] = 0.0f;
        return;
    }

    float L = 0.0f;
    float sum[DSV41_ATTN_MAX_PER_LANE];
    for (uint i = 0; i < dpl; i++) sum[i] = 0.0f;
    for (ushort g = 0; g < nsg; g++) {
        if (pv[g] == 0.0f) continue;
        const float w = exp(pm[g] - M);
        L = fma(pl[g], w, L);
        for (uint i = 0; i < dpl; i++) sum[i] = fma(part[(uint)g * d + i * 32u + lane], w, sum[i]);
    }
    // the sink is a logit on the denominator alone
    L += exp(sink[h] - M);
    const float inv = 1.0f / L;
    for (uint i = 0; i < dpl; i++) o[i * 32u + lane] = sum[i] * inv;
}

// CSA2 compressor pooling.  `ratio` consecutive tokens collapse into one KV latent,
// weighted by a softmax taken PER CHANNEL over the group -- not over the channels -- and
// the RMSNorm that follows is fused in so the pooled values never leave registers.
// compress_ratio 1 is a plain projection and never reaches here.

struct dsv41_compress_pool_args {
    uint groups;
    uint ratio;
    uint dim;           // head_dim
    float eps;
};

kernel void kernel_dsv41_compress_pool(
        constant dsv41_compress_pool_args &args,
        device const float                *kv,     // [groups * ratio, dim]
        device const float                *score,  // [groups * ratio, dim]
        device const float                *norm_w, // [dim]
        device float                      *out,    // [groups, dim]
        threadgroup float                 *shared [[threadgroup(0)]],
        uint  tgpig [[threadgroup_position_in_grid]],
        uint  tpitg [[thread_position_in_threadgroup]],
        uint  ntg   [[threads_per_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg   [[simdgroup_index_in_threadgroup]],
        ushort nsg  [[simdgroups_per_threadgroup]]) {
    const uint g = tgpig;
    if (g >= args.groups) return;
    const uint d = args.dim;
    const uint ratio = args.ratio;

    device const float *kvg = kv + (ulong)g * ratio * d;
    device const float *scg = score + (ulong)g * ratio * d;
    device float *o = out + (ulong)g * d;

    // pooled values stay in threadgroup memory only long enough to take their RMS
    threadgroup float *pooled = shared;
    float ss = 0.0f;
    for (uint j = tpitg; j < d; j += ntg) {
        float mx = -1.0e30f;
        for (uint r = 0; r < ratio; r++) mx = max(mx, scg[(ulong)r * d + j]);
        float sum = 0.0f, acc = 0.0f;
        for (uint r = 0; r < ratio; r++) {
            const float e = exp(scg[(ulong)r * d + j] - mx);
            sum += e;
            acc = fma(e, kvg[(ulong)r * d + j], acc);
        }
        const float v = acc / sum;
        pooled[j] = v;
        ss = fma(v, v, ss);
    }

    threadgroup float *red = shared + d;
    ss = simd_sum(ss);
    if (lane == 0u) red[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float rstd_share;
    if (tpitg == 0u) {
        float total = 0.0f;
        for (ushort i = 0; i < nsg; i++) total += red[i];
        rstd_share = 1.0f / sqrt(total / (float)d + args.eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float rstd = rstd_share;
    for (uint j = tpitg; j < d; j += ntg) o[j] = pooled[j] * rstd * norm_w[j];
}

// Indexer level-zero scores: score[j] = sum_h relu(q[h] . k[j]) * weights[h], with the
// softmax scale already folded into `weights` by the caller.  One lane owns one whole
// head, so each candidate costs a single cross-lane reduction rather than one per head,
// and every lane reads the same k[j] element (a broadcast) while striding its own q row.

struct dsv41_index_score_args {
    uint n_index_head;  // <= 32
    uint index_dim;
    uint n_scan;        // positions to score
    uint use_positions; // 0 = score 0..n_scan-1, 1 = score positions[i]
    float scale;        // softmax_scale * n_heads^-0.5, applied to the head weights
    uint rows;          // queries, tgpig.y; q and weights are per row
    uint stride;        // row stride of scores
};

// `scale` folds softmax_scale * n_heads^-0.5 in here so the caller never has to touch the
// weights_proj output.  `positions`, when used, restricts the scan to a candidate set: a layer that reads a
// source's candidate blocks can never pick anything outside it, so scoring the rest is
// wasted work.  Output is COMPACTED -- scores[i] belongs to positions[i].
kernel void kernel_dsv41_index_score(
        constant dsv41_index_score_args &args,
        device const float              *q,       // [n_index_head, index_dim]
        device const float              *index_k, // [., index_dim]
        device const float              *weights, // [n_index_head], scale folded in
        device float                    *scores,  // [n_scan]
        device const int                *positions,
        uint2  tgpig [[threadgroup_position_in_grid]],
        ushort lane  [[thread_index_in_simdgroup]],
        ushort sg    [[simdgroup_index_in_threadgroup]],
        ushort nsg   [[simdgroups_per_threadgroup]]) {
    const uint i = tgpig.x * (uint)nsg + (uint)sg;
    const uint t = tgpig.y;
    if (i >= args.n_scan || t >= args.rows) return;
    const uint j = args.use_positions ? (uint)positions[i] : i;

    const uint idim = args.index_dim;
    q += (ulong)t * args.n_index_head * idim;
    weights += (ulong)t * args.n_index_head;
    scores += (ulong)t * args.stride;
    device const float *k = index_k + (ulong)j * idim;

    float contrib = 0.0f;
    if ((uint)lane < args.n_index_head) {
        device const float *qh = q + (ulong)lane * idim;
        float dot = 0.0f;
        for (uint e = 0; e < idim; e++) dot = fma(qh[e], k[e], dot);
        // rectified before the head weight, which may be negative
        contrib = dot > 0.0f ? dot * weights[lane] * args.scale : 0.0f;
    }
    const float total = simd_sum(contrib);
    if (lane == 0u) scores[i] = total;
}

// Interleaved RoPE.  V4.1 takes ADJACENT ELEMENT PAIRS as one complex number -- not the
// half-split GPT-NeoX form -- and rotates only the last `rope_dim` elements of each head.
// `inverse` conjugates, which is how the attention output gets the query's rotation
// removed so the cache can stay in one shared rotated form.
//
// The frequency table is supplied as cos/sin per (position, pair), which keeps YaRN and
// the per-layer theta entirely on the host.

struct dsv41_rope_args {
    uint n_head;
    uint head_dim;
    uint rope_dim;      // rotated tail, <= head_dim
    uint pos_base;      // absolute position of row 0
    uint pos_stride;    // row r sits at pos_base + r * pos_stride: compressed groups step by ratio
    uint inverse;
};

kernel void kernel_dsv41_rope(
        constant dsv41_rope_args &args,
        device float             *x,     // [rows, n_head, head_dim], rotated in place
        device const float       *cosv,  // [max_pos, rope_dim / 2]
        device const float       *sinv,
        uint2  tgpig [[threadgroup_position_in_grid]],
        uint2  tpitg [[thread_position_in_threadgroup]],
        uint2  ntg   [[threads_per_threadgroup]]) {
    const uint row = tgpig.y;
    const uint h = tgpig.x;
    if (h >= args.n_head) return;

    const uint half_rd = args.rope_dim / 2u;
    const uint pos = args.pos_base + row * args.pos_stride;
    device const float *c = cosv + (ulong)pos * half_rd;
    device const float *s = sinv + (ulong)pos * half_rd;
    device float *v = x + ((ulong)row * args.n_head + h) * args.head_dim
                        + (args.head_dim - args.rope_dim);

    for (uint i = tpitg.x; i < half_rd; i += ntg.x) {
        const float re = v[2u * i];
        const float im = v[2u * i + 1u];
        const float si = args.inverse ? -s[i] : s[i];
        v[2u * i]      = re * c[i] - im * si;
        v[2u * i + 1u] = fma(re, si, im * c[i]);
    }
}

// Grouped output LoRA.  wo_a is block diagonal over o_groups: each group projects only
// its own heads, so output row r reads the input slice belonging to group r / o_lora.
// That group-dependent input offset is the whole reason this is not a plain mat-vec.
//
// Input is [o_groups, in_per_group] (the attention output, heads flattened), output is
// [o_groups * o_lora]; wo_b afterwards is an ordinary mat-vec.

struct dsv41_output_lora_args {
    uint in_per_group;  // head_dim * n_head / o_groups
    uint o_lora;        // rows per group
    uint o_groups;
    uint quantized;     // 0 = f32 weights, 1 = q8_0
};

// tgpig.y is the token of a prefill chunk: the weight row a threadgroup walks is the same
// for every token, so the batch reuses it out of cache instead of re-reading it per token.
kernel void kernel_dsv41_output_lora(
        constant dsv41_output_lora_args &args,
        device const void               *wo_a,   // [o_groups * o_lora, in_per_group]
        device const float              *in,     // [rows, o_groups, in_per_group]
        device float                    *out,    // [rows, o_groups * o_lora]
        uint2  tgpig [[threadgroup_position_in_grid]],
        ushort lane  [[thread_index_in_simdgroup]],
        ushort sg    [[simdgroup_index_in_threadgroup]],
        ushort nsg   [[simdgroups_per_threadgroup]]) {
    const uint r = tgpig.x * (uint)nsg + (uint)sg;
    const uint rows = args.o_groups * args.o_lora;
    if (r >= rows) return;

    const uint n = args.in_per_group;
    const uint t = tgpig.y;
    in += (ulong)t * args.o_groups * n;
    out += (ulong)t * rows;
    device const float *xr = in + (ulong)(r / args.o_lora) * n;   // this row's group

    float sum = 0.0f;
    if (args.quantized == 0u) {
        device const float *w = (device const float *)wo_a + (ulong)r * n;
        for (uint k = lane; k < n; k += 32u) sum = fma(w[k], xr[k], sum);
    } else {
        const uint nb = n / QK8_0;
        device const block_q8_0 *w = (device const block_q8_0 *)wo_a + (ulong)r * nb;
        for (uint b = lane; b < nb; b += 32u) {
            float q = 0.0f;
            device const int8_t *qs = w[b].qs;
            device const float *y = xr + b * QK8_0;
            for (ushort i = 0; i < QK8_0; i++) q = fma((float)qs[i], y[i], q);
            sum = fma(q, (float)w[b].d, sum);
        }
    }
    sum = simd_sum(sum);
    if (lane == 0u) out[r] = sum;
}

// Level one of the two-level indexer: score each block by its best position.  The block
// holding the newest position is pinned with +inf because it is only partly filled and
// would otherwise be outscored by an older, full block.  Positions the query cannot
// reach arrive already at -1e30, so a block of them scores -1e30 and reads as unreachable.

struct dsv41_block_max_args {
    uint width;         // scored positions
    uint block;         // candidate_block_size
    uint n_blocks;
    uint compress_len;  // for the pin
};

kernel void kernel_dsv41_block_max(
        constant dsv41_block_max_args &args,
        device const float            *scores,     // [width]
        device float                  *block_score, // [n_blocks]
        uint   tgpig [[threadgroup_position_in_grid]],
        ushort lane  [[thread_index_in_simdgroup]],
        ushort sg    [[simdgroup_index_in_threadgroup]],
        ushort nsg   [[simdgroups_per_threadgroup]]) {
    const uint b = tgpig * (uint)nsg + (uint)sg;
    if (b >= args.n_blocks) return;

    const uint lo = b * args.block;
    const uint hi = min(lo + args.block, args.width);
    float mx = -1.0e30f;
    for (uint j = lo + lane; j < hi; j += 32u) mx = max(mx, scores[j]);
    mx = simd_max(mx);
    if (lane != 0u) return;
    // the block holding the newest position is pinned in
    block_score[b] = (args.compress_len > 0u && b == (args.compress_len - 1u) / args.block)
                   ? 1.0e30f : mx;
}

// Exact top-k by (score descending, then INDEX DESCENDING among equals).  That tie order
// is not incidental: relu zeroes every head for some positions, so scores are exactly 0.0
// for many candidates at once, and picking differently there changes the attention output.
// A bitonic argsort cannot express it -- it compares on value alone and is not stable.
//
// Element j is selected iff fewer than k elements beat it, so every rank is independent
// and the whole selection is one parallel pass.  That is O(n^2) work spread over n
// simdgroups; the host refuses sizes where that stops being reasonable.

struct dsv41_rank_select_args {
    uint n;
    uint k;
};

kernel void kernel_dsv41_rank_select(
        constant dsv41_rank_select_args &args,
        device const float              *scores, // [n]
        device int                      *keep,   // [n], 0/1
        uint   tgpig [[threadgroup_position_in_grid]],
        ushort lane  [[thread_index_in_simdgroup]],
        ushort sg    [[simdgroup_index_in_threadgroup]],
        ushort nsg   [[simdgroups_per_threadgroup]]) {
    const uint j = tgpig * (uint)nsg + (uint)sg;
    if (j >= args.n) return;

    const float mine = scores[j];
    uint beats = 0u;
    for (uint i = lane; i < args.n; i += 32u) {
        const float other = scores[i];
        // strictly better, or equal and later: exactly the release's repeated-argmax order
        beats += (other > mine || (other == mine && i > j)) ? 1u : 0u;
    }
    beats = simd_sum(beats);
    if (lane == 0u) keep[j] = beats < args.k ? 1 : 0;
}

// Exact top-k for a candidate set too large for the O(n^2) rank pass.
//
// The order is the same one rank_select implements -- score descending, then INDEX
// descending among equals -- and it has to stay exact, because relu zeroes whole heads
// and leaves hundreds of positions at exactly 0.0 where the tie order decides the
// output.  So this is a radix SELECT, not an approximate threshold: four 8-bit passes
// narrow the score to the exact k-th value, then four more narrow the index among the
// elements that tie with it.  Every pass is O(n) and the refinement lives in a device
// buffer, so the whole selection runs without a single readback.
//
// Floats are compared as sortable uints: flipping the sign bit for positives and every
// bit for negatives makes unsigned order match float order.

struct dsv41_radix_state {
    uint prefix;    // the bits settled so far
    uint shift;     // bits still to settle, counting down 24, 16, 8, 0
    uint k_rem;     // how many are still needed from the current prefix class
    uint thresh;    // the settled score key, once the value phase is done
    uint pass;      // 0 on the first pass of a phase, so the prefix check is skipped
};

static inline uint dsv41_sortable(float f) {
    uint b = as_type<uint>(f);
    if (b == 0x80000000u) b = 0u;                   // -0.0 and +0.0 are one value
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

struct dsv41_radix_args {
    uint n;
    uint mode;      // 0: settle the score.  1: settle the index among score ties
    uint k;
};

// Starts a select: prefix 0, 24 bits to go, k still wanted, first pass; the histogram
// clear.  On the device, because a host write into the shared scratch would land while
// the previous select in the same command buffer is still queued against it.
kernel void kernel_dsv41_radix_init(
        constant dsv41_radix_args &args,
        device dsv41_radix_state  *state,
        device uint               *hist,
        uint tpitg [[thread_position_in_threadgroup]],
        uint ntg   [[threads_per_threadgroup]]) {
    if (tpitg == 0u) {
        state->prefix = 0u;
        state->shift = 24u;
        state->k_rem = args.k;
        state->thresh = 0u;
        state->pass = 0u;
    }
    for (uint b = tpitg; b < 256u; b += ntg) hist[b] = 0u;
}

kernel void kernel_dsv41_radix_hist(
        constant dsv41_radix_args    &args,
        device const float           *scores,
        device dsv41_radix_state     *state,
        device atomic_uint           *hist,     // [256]
        uint gid [[thread_position_in_grid]]) {
    if (gid >= args.n) return;
    const uint u = dsv41_sortable(scores[gid]);
    if (args.mode != 0u && u != state->thresh) return;
    const uint key = args.mode == 0u ? u : gid;
    const uint shift = state->shift;
    if (state->pass != 0u && (key >> (shift + 8u)) != state->prefix) return;
    atomic_fetch_add_explicit(&hist[(key >> shift) & 255u], 1u, memory_order_relaxed);
}

// One thread walks the 256 buckets from the top: the first that carries the k-th element
// settles the next 8 bits.  Clearing the histogram here keeps the pass self-contained.
kernel void kernel_dsv41_radix_scan(
        constant dsv41_radix_args &args,
        device dsv41_radix_state  *state,
        device uint               *hist,
        uint tpitg [[thread_position_in_threadgroup]]) {
    if (tpitg != 0u) return;
    uint cum = 0u;
    uint chosen = 0u;
    for (int b = 255; b >= 0; b--) {
        const uint c = hist[b];
        if (cum + c >= state->k_rem) { chosen = (uint)b; break; }
        cum += c;
    }
    state->prefix = state->pass == 0u ? chosen : ((state->prefix << 8u) | chosen);
    state->k_rem -= cum;
    state->pass += 1u;
    if (state->shift == 0u) {
        // phase done: the settled key is the score threshold, or the index threshold
        if (args.mode == 0u) state->thresh = state->prefix;
        state->shift = 24u;
        state->pass = 0u;
    } else {
        state->shift -= 8u;
    }
    for (uint b = 0; b < 256u; b++) hist[b] = 0u;
}

// keep = beats the threshold outright, or ties with it at an index the tie order admits.
kernel void kernel_dsv41_radix_keep(
        constant dsv41_radix_args      &args,
        device const float             *scores,
        device const dsv41_radix_state *state,
        device int                     *keep,
        uint gid [[thread_position_in_grid]]) {
    if (gid >= args.n) return;
    const uint u = dsv41_sortable(scores[gid]);
    keep[gid] = (u > state->thresh || (u == state->thresh && gid >= state->prefix)) ? 1 : 0;
}

// Applies the candidate source's block decision to a consumer layer's scores: a position
// whose block was not selected can never be picked, which is what masking it to -inf
// means to the rank pass below.
struct dsv41_block_mask_args {
    uint n;
    uint block;
};

kernel void kernel_dsv41_block_mask(
        constant dsv41_block_mask_args &args,
        device const int              *block_keep,
        device float                  *scores,
        uint gid [[thread_position_in_grid]]) {
    if (gid >= args.n) return;
    if (!block_keep[gid / args.block]) scores[gid] = -1.0e30f;
}

// ---- Selection over a prefill chunk, one row per query ----------------------------
//
// Query t of a chunk starting at pos0 reaches n_t = min(n_cap, (pos0 + t + 1) / ratio)
// compressed positions -- or ceil(n_t / block) candidate blocks -- and wants
// k_t = min(k_max, n_t) of them.  The kernels below take that rule instead of one fixed
// n and k, so a single dispatch covers every query of the chunk with each keeping its
// own reach.  It is also what gives each query its own candidate blocks: with one block
// decision per layer, every query of a chunk was masked with the last query's.

struct dsv41_rows_args {
    uint rows;
    uint stride;      // row stride of scores / keep, in elements
    uint n_cap;       // positions published so far
    uint pos0;
    uint ratio;
    uint k_max;
    uint blocks;      // 1: extents count blocks of `block`, not positions
    uint block;
    uint offset;      // compact: added to every id
    uint width;       // compact: output row width
    uint nb_stride;   // block_max / block_mask: row stride of the block arrays
    uint mode;        // radix: 0 settles the score, 1 the index among ties
};

static inline uint dsv41_row_reach(constant dsv41_rows_args &a, uint t) {
    return min(a.n_cap, (a.pos0 + t + 1u) / a.ratio);
}

static inline uint dsv41_row_n(constant dsv41_rows_args &a, uint t) {
    const uint n = dsv41_row_reach(a, t);
    return a.blocks ? (n + a.block - 1u) / a.block : n;
}

kernel void kernel_dsv41_rank_select_rows(
        constant dsv41_rows_args &a,
        device const float       *scores,   // [rows, stride]
        device int               *keep,     // [rows, stride]
        uint2  tgpig [[threadgroup_position_in_grid]],
        ushort lane  [[thread_index_in_simdgroup]],
        ushort sg    [[simdgroup_index_in_threadgroup]],
        ushort nsg   [[simdgroups_per_threadgroup]]) {
    const uint t = tgpig.y;
    if (t >= a.rows) return;
    const uint n = dsv41_row_n(a, t);
    const uint k = min(a.k_max, n);
    const uint j = tgpig.x * (uint)nsg + (uint)sg;
    if (j >= n) return;
    scores += (ulong)t * a.stride;
    keep += (ulong)t * a.stride;

    const float mine = scores[j];
    uint beats = 0u;
    for (uint i = lane; i < n; i += 32u) {
        const float other = scores[i];
        beats += (other > mine || (other == mine && i > j)) ? 1u : 0u;
    }
    beats = simd_sum(beats);
    if (lane == 0u) keep[j] = beats < k ? 1 : 0;
}

// One threadgroup per row; a row that reaches nothing comes out all -1.
kernel void kernel_dsv41_compact_rows(
        constant dsv41_rows_args &a,
        device const int         *keep,     // [rows, stride]
        device int               *out,      // [rows, width]
        threadgroup uint         *counts [[threadgroup(0)]],
        uint tgpig [[threadgroup_position_in_grid]],
        uint tpitg [[thread_position_in_threadgroup]],
        uint ntg   [[threads_per_threadgroup]]) {
    const uint t = tgpig;
    if (t >= a.rows) return;
    const uint n = dsv41_row_n(a, t);
    const uint k = min(a.k_max, n);
    keep += (ulong)t * a.stride;
    out += (ulong)t * a.width;

    const uint chunk = (n + ntg - 1u) / ntg;
    const uint lo = tpitg * chunk;
    const uint hi = min(lo + chunk, n);
    uint mine = 0u;
    for (uint j = lo; j < hi; j++) mine += keep[j] ? 1u : 0u;
    counts[tpitg] = mine;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tpitg == 0u) {
        uint running = 0u;
        for (uint i = 0; i < ntg; i++) {
            const uint c = counts[i];
            counts[i] = running;
            running += c;
        }
        counts[ntg] = running;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint w = counts[tpitg];
    for (uint j = lo; j < hi && w < k; j++) {
        if (!keep[j]) continue;
        out[w++] = (int)(j + a.offset);
    }
    const uint filled = min(counts[ntg], k);
    for (uint i = filled + tpitg; i < a.width; i += ntg) out[i] = -1;
}

kernel void kernel_dsv41_block_max_rows(
        constant dsv41_rows_args &a,
        device const float       *scores,       // [rows, stride]
        device float             *block_score,  // [rows, nb_stride]
        uint2  tgpig [[threadgroup_position_in_grid]],
        ushort lane  [[thread_index_in_simdgroup]],
        ushort sg    [[simdgroup_index_in_threadgroup]],
        ushort nsg   [[simdgroups_per_threadgroup]]) {
    const uint t = tgpig.y;
    if (t >= a.rows) return;
    const uint n = dsv41_row_reach(a, t);
    const uint nb = (n + a.block - 1u) / a.block;
    const uint b = tgpig.x * (uint)nsg + (uint)sg;
    if (b >= nb) return;
    scores += (ulong)t * a.stride;
    block_score += (ulong)t * a.nb_stride;

    const uint lo = b * a.block;
    const uint hi = min(lo + a.block, n);
    float mx = -1.0e30f;
    for (uint j = lo + lane; j < hi; j += 32u) mx = max(mx, scores[j]);
    mx = simd_max(mx);
    if (lane != 0u) return;
    // the block holding the query's newest position is pinned in
    block_score[b] = b == (n - 1u) / a.block ? 1.0e30f : mx;
}

kernel void kernel_dsv41_block_mask_rows(
        constant dsv41_rows_args &a,
        device const int         *block_keep,   // [rows, nb_stride]
        device float             *scores,       // [rows, stride]
        uint2 gid [[thread_position_in_grid]]) {
    const uint t = gid.y;
    if (t >= a.rows) return;
    const uint j = gid.x;
    if (j >= dsv41_row_reach(a, t)) return;
    if (!block_keep[(ulong)t * a.nb_stride + j / a.block]) {
        scores[(ulong)t * a.stride + j] = -1.0e30f;
    }
}

// The radix select per row: its refinement state and histogram are per row too.
kernel void kernel_dsv41_radix_init_rows(
        constant dsv41_rows_args &a,
        device dsv41_radix_state *state,    // [rows]
        device uint              *hist,     // [rows, 256]
        uint2 tgpig [[threadgroup_position_in_grid]],
        uint2 tpitg [[thread_position_in_threadgroup]],
        uint2 ntg   [[threads_per_threadgroup]]) {
    const uint t = tgpig.y;
    if (t >= a.rows) return;
    if (tpitg.x == 0u) {
        const uint n = dsv41_row_n(a, t);
        state[t].prefix = 0u;
        state[t].shift = 24u;
        state[t].k_rem = min(a.k_max, n);
        state[t].thresh = 0u;
        state[t].pass = 0u;
    }
    for (uint b = tpitg.x; b < 256u; b += ntg.x) hist[(ulong)t * 256u + b] = 0u;
}

kernel void kernel_dsv41_radix_hist_rows(
        constant dsv41_rows_args &a,
        device const float       *scores,
        device dsv41_radix_state *state,
        device atomic_uint       *hist,
        uint2 gid [[thread_position_in_grid]]) {
    const uint t = gid.y;
    if (t >= a.rows) return;
    const uint j = gid.x;
    if (j >= dsv41_row_n(a, t)) return;
    device dsv41_radix_state *st = state + t;
    const uint u = dsv41_sortable(scores[(ulong)t * a.stride + j]);
    if (a.mode != 0u && u != st->thresh) return;
    const uint key = a.mode == 0u ? u : j;
    const uint shift = st->shift;
    if (st->pass != 0u && (key >> (shift + 8u)) != st->prefix) return;
    atomic_fetch_add_explicit(&hist[(ulong)t * 256u + ((key >> shift) & 255u)], 1u,
                              memory_order_relaxed);
}

kernel void kernel_dsv41_radix_scan_rows(
        constant dsv41_rows_args &a,
        device dsv41_radix_state *state,
        device uint              *hist,
        uint gid [[thread_position_in_grid]]) {
    const uint t = gid;
    if (t >= a.rows) return;
    device dsv41_radix_state *st = state + t;
    hist += (ulong)t * 256u;
    uint cum = 0u;
    uint chosen = 0u;
    for (int b = 255; b >= 0; b--) {
        const uint c = hist[b];
        if (cum + c >= st->k_rem) { chosen = (uint)b; break; }
        cum += c;
    }
    st->prefix = st->pass == 0u ? chosen : ((st->prefix << 8u) | chosen);
    st->k_rem -= cum;
    st->pass += 1u;
    if (st->shift == 0u) {
        if (a.mode == 0u) st->thresh = st->prefix;
        st->shift = 24u;
        st->pass = 0u;
    } else {
        st->shift -= 8u;
    }
    for (uint b = 0; b < 256u; b++) hist[b] = 0u;
}

kernel void kernel_dsv41_radix_keep_rows(
        constant dsv41_rows_args       &a,
        device const float             *scores,
        device const dsv41_radix_state *state,
        device int                     *keep,
        uint2 gid [[thread_position_in_grid]]) {
    const uint t = gid.y;
    if (t >= a.rows) return;
    const uint j = gid.x;
    if (j >= dsv41_row_n(a, t)) return;
    const uint u = dsv41_sortable(scores[(ulong)t * a.stride + j]);
    keep[(ulong)t * a.stride + j] =
        (u > state[t].thresh || (u == state[t].thresh && j >= state[t].prefix)) ? 1 : 0;
}

// Compacts a keep mask into ascending positions, shifted by `offset`, with anything the
// query cannot reach (or any unfilled slot) written as -1.  Ascending order is what
// sparse_attn consumes; one threadgroup keeps the scan trivially sequential.

struct dsv41_compact_args {
    uint n;
    uint k;
    uint offset;
    uint reach;     // positions >= this are unreachable and emit -1
    uint width;     // the whole output row: everything past the picks is -1
};

kernel void kernel_dsv41_compact(
        constant dsv41_compact_args &args,
        device const int            *keep,
        device int                  *out,      // [k]
        threadgroup uint            *counts [[threadgroup(0)]],
        uint tpitg [[thread_position_in_threadgroup]],
        uint ntg   [[threads_per_threadgroup]]) {
    // Each thread owns a contiguous chunk, so an exclusive scan of the per-chunk counts
    // gives every thread its write base and the result stays ascending.  A single thread
    // walking `keep` would serialise n dependent global reads, which at n = 16384 costs
    // more than the whole rank pass.
    const uint chunk = (args.n + ntg - 1u) / ntg;
    const uint lo = tpitg * chunk;
    const uint hi = min(lo + chunk, args.n);

    uint mine = 0u;
    for (uint j = lo; j < hi; j++) mine += keep[j] ? 1u : 0u;
    counts[tpitg] = mine;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tpitg == 0u) {
        uint running = 0u;
        for (uint t = 0; t < ntg; t++) {
            const uint c = counts[t];
            counts[t] = running;
            running += c;
        }
        counts[ntg] = running;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint w = counts[tpitg];
    for (uint j = lo; j < hi && w < args.k; j++) {
        if (!keep[j]) continue;
        out[w++] = j < args.reach ? (int)(j + args.offset) : -1;
    }
    // anything the selection could not fill, out to the end of the row: a reader that
    // takes every row at the same width must find -1 past this query's picks
    const uint filled = min(counts[ntg], args.k);
    for (uint i = filled + tpitg; i < args.width; i += ntg) out[i] = -1;
}

// ---- Routed experts for a prefill chunk, expert-major, in F32 -------------------------
//
// A chunk's (token, slot) pairs are counting-sorted by expert on the device, then each
// expert's tokens go through its weights in groups of up to DSV41_XM_R: a weight block is
// decoded once and applied to every token of the group, so tokens sharing an expert read
// it once.  Every pair's arithmetic is its own -- nothing is shared but the weight loads
// -- so the result does not depend on how the pairs were grouped, and there is no MMA and
// no half staging anywhere: the sums are the one-token kernels' F32 sums in another order.
// The down projection lands per pair in the experts scratch and a last pass folds each
// token's slots and its shared expert.

#define DSV41_XM_R 8u

struct dsv41_moe_xm_args {
    uint n_expert;      // routed pool
    uint topk;
    uint rows;          // tokens
    uint in_dim;
    uint mid_dim;
    uint out_dim;
    uint n_pairs;       // rows * topk
    ulong gate_expert_bytes;
    ulong gate_row_bytes;
    ulong down_expert_bytes;
    ulong down_row_bytes;
    float clamp;
};

static constant float dsv41_mxfp4_lut[16] = {
     0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};

static inline float dsv41_e8m0(uchar e) {
    return as_type<float>(e == 0 ? 0x00400000u : (uint)e << 23);
}

// four nibble bytes of one block: elements i (low nibbles) and i+16 (high) for the lane's
// four i; unscaled, the block scale goes on the sum
static inline void dsv41_mxfp4_quarter(packed_uchar4 nib, threadgroup const float *lut,
                                       thread float4 &lo, thread float4 &hi) {
    lo = float4(lut[nib.x & 15u], lut[nib.y & 15u], lut[nib.z & 15u], lut[nib.w & 15u]);
    hi = float4(lut[nib.x >> 4], lut[nib.y >> 4], lut[nib.z >> 4], lut[nib.w >> 4]);
}

// the nibble table in threadgroup memory, as in the one-token kernel
#define DSV41_XM_LUT(lut, tid)                                                                 \
    threadgroup float lut[16];                                                                \
    if ((tid) < 16u) lut[tid] = dsv41_mxfp4_lut[tid];                                         \
    threadgroup_barrier(mem_flags::mem_threadgroup)

// pair p = (token, slot) counts toward its expert
kernel void kernel_dsv41_moe_hist(
        constant dsv41_moe_xm_args &a,
        device const int           *sel,     // [rows, topk]
        device atomic_uint         *counts,  // [n_expert], zero on entry
        uint gid [[thread_position_in_grid]]) {
    if (gid >= a.n_pairs) return;
    const int e = sel[gid];
    if (e < 0 || (uint)e >= a.n_expert) return;
    atomic_fetch_add_explicit(&counts[e], 1u, memory_order_relaxed);
}

// Offsets, the group list, and the group count; one threadgroup, thread 0 walks the pool.
// Clears the counts behind itself so the next layer's histogram starts at zero.  The
// expert kernels are launched for the most groups there could be and exit past the count.
kernel void kernel_dsv41_moe_scan(
        constant dsv41_moe_xm_args &a,
        device uint                *counts,  // [n_expert]
        device uint                *cursor,  // [n_expert + 1]: scatter cursors, then n_groups
        device uint                *groups,  // [n_pairs * 3]: expert, first sorted pair, count
        uint tpitg [[thread_position_in_threadgroup]]) {
    if (tpitg != 0u) return;
    uint off = 0u, g = 0u;
    for (uint e = 0; e < a.n_expert; e++) {
        const uint c = counts[e];
        cursor[e] = off;
        for (uint k = 0; k < c; k += DSV41_XM_R) {
            groups[3u * g] = e;
            groups[3u * g + 1u] = off + k;
            groups[3u * g + 2u] = min(DSV41_XM_R, c - k);
            g++;
        }
        off += c;
        counts[e] = 0u;
    }
    cursor[a.n_expert] = g;
}

kernel void kernel_dsv41_moe_scatter(
        constant dsv41_moe_xm_args &a,
        device const int           *sel,
        device atomic_uint         *cursor,
        device uint                *sorted,  // [n_pairs]: token << 8 | slot
        uint gid [[thread_position_in_grid]]) {
    if (gid >= a.n_pairs) return;
    const int e = sel[gid];
    if (e < 0 || (uint)e >= a.n_expert) return;
    const uint pos = atomic_fetch_add_explicit(&cursor[e], 1u, memory_order_relaxed);
    sorted[pos] = ((gid / a.topk) << 8) | (gid % a.topk);
}

// Gate and up for one expert group, register-blocked.  A lane owns half of every block
// (16 k) for DSV41_XM_T of the group's tokens and holds that x in registers while it walks
// DSV41_XM_ROWS rows of gate and of up, so a loaded x value meets 2*ROWS weights and a
// decoded weight meets T tokens.  A threadgroup is DSV41_XM_SPLITS token slices by two row
// sets, one simdgroup each, so 2*ROWS rows share a threadgroup and the slices share its
// weight loads through L1.  A slice past the group's tokens exits; a slice's dead token
// recomputes a live one's x, which keeps the loop branch-free, and only live results are
// stored.  Lanes hold partials over different k of the same outputs and a simd_sum closes
// each one.  The nibble table sits in threadgroup memory as in the one-token kernel, and
// the block scale is applied to the block's sum, as there.
//
// Measured on the released dims against the one-token kernels run token-major, per
// (token, expert) pair: 1.3x slower at one token per expert, 1.6x faster at two, 1.8x at
// four and eight -- so the host takes this path from two tokens per expert up.
#define DSV41_XM_T 2u        // tokens per simdgroup
#define DSV41_XM_SPLITS 4u   // token slices per threadgroup; T * SPLITS = DSV41_XM_R
#define DSV41_XM_ROWS 4u
#define DSV41_XM_DOWN_ROWS 8u
#define DSV41_XM_NV 2u       // float4 of x per lane per nibble half
#define DSV41_XM_BPP 16u     // blocks per pass

// the lane's part of a block against T x slices: loads the nibbles once, applies them to
// every token, and folds the block scale into each running sum
#define DSV41_XM_BLOCK(wp, dv, acc, xv) do {                                                  \
    float4 lo_[DSV41_XM_NV], hi_[DSV41_XM_NV];                                               \
    _Pragma("unroll") for (uint i_ = 0; i_ < DSV41_XM_NV; i_++) {                             \
        dsv41_mxfp4_quarter(*(device const packed_uchar4 *)((wp) + boff + 4u * i_), lut,     \
                            lo_[i_], hi_[i_]);                                                \
    }                                                                                         \
    _Pragma("unroll") for (uint t_ = 0; t_ < DSV41_XM_T; t_++) {                              \
        float4 p_ = lo_[0] * (xv)[t_][0];                                                     \
        _Pragma("unroll") for (uint i_ = 1; i_ < DSV41_XM_NV; i_++) {                         \
            p_ = fma(lo_[i_], (xv)[t_][i_], p_);                                              \
        }                                                                                     \
        _Pragma("unroll") for (uint i_ = 0; i_ < DSV41_XM_NV; i_++) {                         \
            p_ = fma(hi_[i_], (xv)[t_][DSV41_XM_NV + i_], p_);                                \
        }                                                                                     \
        (acc)[t_] = fma((dv), (p_.x + p_.y) + (p_.z + p_.w), (acc)[t_]);                      \
    }                                                                                         \
} while (0)

// x for the lane's part of block `blk`, T tokens
#define DSV41_XM_LOAD_X(xv, xr, blk) do {                                                     \
    _Pragma("unroll") for (uint t_ = 0; t_ < DSV41_XM_T; t_++) {                              \
        _Pragma("unroll") for (uint i_ = 0; i_ < DSV41_XM_NV; i_++) {                         \
            (xv)[t_][i_] = *(device const float4 *)((xr)[t_] + (blk) * 32u + 4u * i_);        \
            (xv)[t_][DSV41_XM_NV + i_] =                                                      \
                *(device const float4 *)((xr)[t_] + (blk) * 32u + 16u + 4u * i_);             \
        }                                                                                     \
    }                                                                                         \
} while (0)

kernel void kernel_dsv41_moe_pair_xm_mxfp4(
        constant dsv41_moe_xm_args &a,
        device const uchar         *gate_w,  // all experts
        device const uchar         *up_w,
        device const float         *x,       // [rows, in_dim]
        device const float         *wts,     // [rows, topk]
        device const uint          *groups,
        device const uint          *cursor,
        device const uint          *sorted,
        device float               *mid,     // [rows * topk, mid_dim]
        uint2  tgpig [[threadgroup_position_in_grid]],
        ushort tid   [[thread_index_in_threadgroup]],
        ushort lane  [[thread_index_in_simdgroup]],
        ushort sg    [[simdgroup_index_in_threadgroup]]) {
    DSV41_XM_LUT(lut, tid);

    const uint g = tgpig.y;
    if (g >= cursor[a.n_expert]) return;
    const uint e = groups[3u * g], start = groups[3u * g + 1u], n = groups[3u * g + 2u];
    const uint t0 = (sg % DSV41_XM_SPLITS) * DSV41_XM_T;
    if (t0 >= n) return;
    const uint nt = min(DSV41_XM_T, n - t0);
    const uint row0 = (tgpig.x * 2u + sg / DSV41_XM_SPLITS) * DSV41_XM_ROWS;
    const uint b = lane / (32u / DSV41_XM_BPP), part = lane % (32u / DSV41_XM_BPP);
    const uint koff = part * 4u * DSV41_XM_NV, boff = 1u + koff;

    device const float *xr[DSV41_XM_T];
#pragma unroll
    for (uint t = 0; t < DSV41_XM_T; t++) {
        xr[t] = x + (ulong)(sorted[start + t0 + min(t, nt - 1u)] >> 8) * a.in_dim + koff;
    }
    device const uchar *gw = gate_w + (ulong)e * a.gate_expert_bytes + (ulong)row0 * a.gate_row_bytes;
    device const uchar *uw = up_w + (ulong)e * a.gate_expert_bytes + (ulong)row0 * a.gate_row_bytes;

    float ag[DSV41_XM_ROWS][DSV41_XM_T], au[DSV41_XM_ROWS][DSV41_XM_T];
#pragma unroll
    for (uint r = 0; r < DSV41_XM_ROWS; r++) {
#pragma unroll
        for (uint t = 0; t < DSV41_XM_T; t++) { ag[r][t] = 0.0f; au[r][t] = 0.0f; }
    }
    const uint nb = a.in_dim / 32u;
    for (uint blk = b; blk < nb; blk += DSV41_XM_BPP) {
        float4 xv[DSV41_XM_T][2u * DSV41_XM_NV];
        DSV41_XM_LOAD_X(xv, xr, blk);
        const ulong bo = (ulong)blk * 17u;
#pragma unroll
        for (uint r = 0; r < DSV41_XM_ROWS; r++) {
            device const uchar *gp = gw + (ulong)r * a.gate_row_bytes + bo;
            device const uchar *up = uw + (ulong)r * a.gate_row_bytes + bo;
            DSV41_XM_BLOCK(gp, dsv41_e8m0(gp[0]), ag[r], xv);
            DSV41_XM_BLOCK(up, dsv41_e8m0(up[0]), au[r], xv);
        }
    }
#pragma unroll
    for (uint r = 0; r < DSV41_XM_ROWS; r++) {
#pragma unroll
        for (uint t = 0; t < DSV41_XM_T; t++) {
            float gv = simd_sum(ag[r][t]), uv = simd_sum(au[r][t]);
            if (lane != 0u || t >= nt) continue;
            const uint v = sorted[start + t0 + t];
            const uint pair = (v >> 8) * a.topk + (v & 255u);
            if (a.clamp > 1.0e-6f) {
                gv = min(gv, a.clamp);
                uv = clamp(uv, -a.clamp, a.clamp);
            }
            mid[(ulong)pair * a.mid_dim + row0 + r] = (gv / (1.0f + exp(-gv))) * uv * wts[pair];
        }
    }
}

// the down projection of one expert group, per pair into the experts scratch: the same
// blocking with DSV41_XM_DOWN_ROWS rows of one matrix
kernel void kernel_dsv41_moe_down_xm_mxfp4(
        constant dsv41_moe_xm_args &a,
        device const uchar         *down_w,
        device const float         *mid,     // [rows * topk, mid_dim]
        device const uint          *groups,
        device const uint          *cursor,
        device const uint          *sorted,
        device float               *experts, // [rows * topk, out_dim]
        uint2  tgpig [[threadgroup_position_in_grid]],
        ushort tid   [[thread_index_in_threadgroup]],
        ushort lane  [[thread_index_in_simdgroup]],
        ushort sg    [[simdgroup_index_in_threadgroup]]) {
    DSV41_XM_LUT(lut, tid);

    const uint g = tgpig.y;
    if (g >= cursor[a.n_expert]) return;
    const uint e = groups[3u * g], start = groups[3u * g + 1u], n = groups[3u * g + 2u];
    const uint t0 = (sg % DSV41_XM_SPLITS) * DSV41_XM_T;
    if (t0 >= n) return;
    const uint nt = min(DSV41_XM_T, n - t0);
    const uint row0 = (tgpig.x * 2u + sg / DSV41_XM_SPLITS) * DSV41_XM_DOWN_ROWS;
    const uint b = lane / (32u / DSV41_XM_BPP), part = lane % (32u / DSV41_XM_BPP);
    const uint koff = part * 4u * DSV41_XM_NV, boff = 1u + koff;

    uint pair[DSV41_XM_T];
    device const float *mr[DSV41_XM_T];
#pragma unroll
    for (uint t = 0; t < DSV41_XM_T; t++) {
        const uint v = sorted[start + t0 + min(t, nt - 1u)];
        pair[t] = (v >> 8) * a.topk + (v & 255u);
        mr[t] = mid + (ulong)pair[t] * a.mid_dim + koff;
    }
    device const uchar *dw = down_w + (ulong)e * a.down_expert_bytes + (ulong)row0 * a.down_row_bytes;

    float acc[DSV41_XM_DOWN_ROWS][DSV41_XM_T];
#pragma unroll
    for (uint r = 0; r < DSV41_XM_DOWN_ROWS; r++) {
#pragma unroll
        for (uint t = 0; t < DSV41_XM_T; t++) acc[r][t] = 0.0f;
    }
    const uint nb = a.mid_dim / 32u;
    for (uint blk = b; blk < nb; blk += DSV41_XM_BPP) {
        float4 xv[DSV41_XM_T][2u * DSV41_XM_NV];
        DSV41_XM_LOAD_X(xv, mr, blk);
        const ulong bo = (ulong)blk * 17u;
#pragma unroll
        for (uint r = 0; r < DSV41_XM_DOWN_ROWS; r++) {
            device const uchar *dp = dw + (ulong)r * a.down_row_bytes + bo;
            DSV41_XM_BLOCK(dp, dsv41_e8m0(dp[0]), acc[r], xv);
        }
    }
#pragma unroll
    for (uint r = 0; r < DSV41_XM_DOWN_ROWS; r++) {
#pragma unroll
        for (uint t = 0; t < DSV41_XM_T; t++) {
            const float v = simd_sum(acc[r][t]);
            if (lane == 0u && t < nt) experts[(ulong)pair[t] * a.out_dim + row0 + r] = v;
        }
    }
}

// out[t] = shared[t] + sum over slots of the pair outputs, slots in order; a slot the
// sort skipped (no valid expert) has no output and is skipped here too
kernel void kernel_dsv41_moe_sum(
        constant dsv41_moe_xm_args &a,
        device const float         *experts, // [rows * topk, out_dim]
        device const float         *shared,  // [rows, out_dim]
        device float               *out,     // [rows, out_dim]
        device const int           *sel,     // [rows, topk]
        uint2 gid [[thread_position_in_grid]]) {
    const uint j = gid.x, t = gid.y;
    if (j >= a.out_dim || t >= a.rows) return;
    float acc = shared[(ulong)t * a.out_dim + j];
    for (uint s = 0; s < a.topk; s++) {
        const int e = sel[t * a.topk + s];
        if (e < 0 || (uint)e >= a.n_expert) continue;
        acc += experts[((ulong)t * a.topk + s) * a.out_dim + j];
    }
    out[(ulong)t * a.out_dim + j] = acc;
}

// Single-Pass mHC: one projection of the flattened, normalised residual stream, split
// into the pre / post / comb coefficients, with `comb` made doubly stochastic by Sinkhorn.
//
// The stream is normalised over the WHOLE hc*dim vector -- one statistic per token, not
// per copy -- then hc_fn projects it to (2 + hc) * hc mixes.  The coefficients a sublayer
// computes are consumed by the NEXT one, which is what makes it single pass.
//
// comb is only hc x hc (4 x 4 in the release) but the Sinkhorn is 20 sequential
// iterations, so it is done once by a single thread after the reduction rather than
// spread across threads that would have to barrier on every half-iteration.

#define DSV41_HC_MAX 8u
#define DSV41_MIX_MAX ((2u + DSV41_HC_MAX) * DSV41_HC_MAX)

struct dsv41_hc_mixes_args {
    uint n_tokens;      // one threadgroup each
    uint hc_dim;        // hc * n_embd
    uint mix_hc;        // (2 + hc) * hc
    uint hc;
    uint iters;
    float norm_eps;     // the RMS epsilon, not hc_eps
    float hc_eps;
    uint f16_weights;   // 0 = f32 hc_fn, 1 = f16
};

static inline float dsv41_sigmoid(float x) {
    // matches sigmoid_stable: never exponentiate a positive argument
    if (x >= 0.0f) return 1.0f / (1.0f + exp(-x));
    const float e = exp(x);
    return e / (1.0f + e);
}

// The mHC projection used to be one threadgroup per token: at decode that is a single
// threadgroup for the whole GPU, 256 threads walking 24 rows of 20480 for 797 us a call
// -- over half the decode step.  Splitting it gives every row its own threadgroup, so the
// grid carries mix_hc * n_tokens of them and each thread walks hc_dim/256 elements.
//
// `ss` is the same sum of squares in every row's threadgroup.  Recomputing it there is
// free: the pass already has x in hand, and it saves a second kernel and a round trip.
kernel void kernel_dsv41_hc_mix_proj(
        constant dsv41_hc_mixes_args &args,
        device const float           *stream,   // [n_tokens, hc_dim]
        device const void            *hc_fn,    // [mix_hc, hc_dim]
        device float                 *mixes,    // [n_tokens, mix_hc]
        threadgroup float            *red [[threadgroup(0)]],
        uint2  tgpig [[threadgroup_position_in_grid]],
        uint2  tpitg [[thread_position_in_threadgroup]],
        uint2  ntg   [[threads_per_threadgroup]],
        ushort lane  [[thread_index_in_simdgroup]],
        ushort sg    [[simdgroup_index_in_threadgroup]],
        ushort nsg   [[simdgroups_per_threadgroup]]) {
    const uint m = tgpig.x;
    const uint tok = tgpig.y;
    if (m >= args.mix_hc || tok >= args.n_tokens) return;

    device const float *x = stream + (ulong)tok * args.hc_dim;
    float dot = 0.0f, ss = 0.0f;
    if (args.f16_weights) {
        device const half *w = (device const half *)hc_fn + (ulong)m * args.hc_dim;
        for (uint i = tpitg.x; i < args.hc_dim; i += ntg.x) {
            const float v = x[i];
            dot = fma((float)w[i], v, dot);
            ss = fma(v, v, ss);
        }
    } else {
        device const float *w = (device const float *)hc_fn + (ulong)m * args.hc_dim;
        for (uint i = tpitg.x; i < args.hc_dim; i += ntg.x) {
            const float v = x[i];
            dot = fma(w[i], v, dot);
            ss = fma(v, v, ss);
        }
    }
    dot = simd_sum(dot);
    ss = simd_sum(ss);
    if (lane == 0u) { red[sg] = dot; red[nsg + sg] = ss; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tpitg.x != 0u) return;
    float d = 0.0f, s2 = 0.0f;
    for (ushort i = 0; i < nsg; i++) { d += red[i]; s2 += red[nsg + i]; }
    const float rstd = 1.0f / sqrt(s2 / (float)args.hc_dim + args.norm_eps);
    mixes[(ulong)tok * args.mix_hc + m] = d * rstd;
}

// The split / sigmoid / Sinkhorn tail, one threadgroup per token.  hc is 4, so this is
// tens of operations; it never wanted the 20480-wide projection in front of it.
kernel void kernel_dsv41_hc_mix_split(
        constant dsv41_hc_mixes_args &args,
        device const float           *mixes,   // [n_tokens, mix_hc]
        device const float           *hc_scale,
        device const float           *hc_base,
        device float                 *pre,
        device float                 *post,
        device float                 *comb,
        uint   tgpig [[threadgroup_position_in_grid]],
        ushort lane  [[thread_index_in_simdgroup]]) {
    const uint tok = tgpig;
    if (tok >= args.n_tokens) return;
    device const float *mx = mixes + (ulong)tok * args.mix_hc;
    device float *pre_t = pre + (ulong)tok * args.hc;
    device float *post_t = post + (ulong)tok * args.hc;
    device float *comb_t = comb + (ulong)tok * args.hc * args.hc;

    const uint hc = args.hc;
    const float eps = args.hc_eps;

    if (lane < hc) {
        pre_t[lane] = dsv41_sigmoid(fma(mx[lane], hc_scale[0], hc_base[lane])) + eps;
        post_t[lane] = 2.0f * dsv41_sigmoid(fma(mx[hc + lane], hc_scale[1],
                                                hc_base[hc + lane]));
    }

    // hc is a power of two and hc*hc <= 32 for every shape DS4 carries, so one lane owns
    // one comb element and the Sinkhorn sweeps are butterflies inside a single simdgroup.
    // Walking them serially in thread 0 cost 91 us a call -- 17% of a decode step -- for
    // a 4x4 matrix.  Lane l holds element (l / hc, l % hc): xor masks below hc move along
    // a row, masks from hc up move down a column, and both stay inside the first hc*hc
    // lanes, so the idle lanes above cannot disturb either reduction.
    const uint n = hc * hc;
    if (n > 32u || (hc & (hc - 1u)) != 0u) {
        if (lane != 0u) return;                 // fallback: shapes the butterfly cannot hold
        for (uint j = 0; j < hc * hc; j++) {
            comb_t[j] = fma(mx[2u * hc + j], hc_scale[2], hc_base[2u * hc + j]);
        }
        for (uint j = 0; j < hc; j++) {
            float mxv = comb_t[j * hc];
            for (uint k = 1; k < hc; k++) mxv = max(mxv, comb_t[j * hc + k]);
            float sum = 0.0f;
            for (uint k = 0; k < hc; k++) {
                comb_t[j * hc + k] = exp(comb_t[j * hc + k] - mxv);
                sum += comb_t[j * hc + k];
            }
            for (uint k = 0; k < hc; k++) comb_t[j * hc + k] = comb_t[j * hc + k] / sum + eps;
        }
        for (uint it = 0; it < args.iters; it++) {
            if (it != 0u) {
                for (uint j = 0; j < hc; j++) {
                    float row = 0.0f;
                    for (uint k = 0; k < hc; k++) row += comb_t[j * hc + k];
                    for (uint k = 0; k < hc; k++) comb_t[j * hc + k] /= (row + eps);
                }
            }
            for (uint k = 0; k < hc; k++) {
                float col = 0.0f;
                for (uint j = 0; j < hc; j++) col += comb_t[j * hc + k];
                for (uint j = 0; j < hc; j++) comb_t[j * hc + k] /= (col + eps);
            }
        }
        return;
    }

    const bool live = lane < n;
    float v = live ? fma(mx[2u * hc + lane], hc_scale[2], hc_base[2u * hc + lane]) : 0.0f;

    // row softmax: reduce across the lanes sharing this row
    float rmax = live ? v : -1.0e30f;
    for (uint m = 1u; m < hc; m <<= 1) rmax = max(rmax, simd_shuffle_xor(rmax, m));
    float e = live ? exp(v - rmax) : 0.0f;
    float rsum = e;
    for (uint m = 1u; m < hc; m <<= 1) rsum += simd_shuffle_xor(rsum, m);
    v = e / rsum + eps;

    // then alternating column / row normalisation, the first sweep column-only
    float csum = v;
    for (uint m = hc; m < n; m <<= 1) csum += simd_shuffle_xor(csum, m);
    v /= (csum + eps);
    for (uint it = 1; it < args.iters; it++) {
        rsum = v;
        for (uint m = 1u; m < hc; m <<= 1) rsum += simd_shuffle_xor(rsum, m);
        v /= (rsum + eps);
        csum = v;
        for (uint m = hc; m < n; m <<= 1) csum += simd_shuffle_xor(csum, m);
        v /= (csum + eps);
    }
    if (live) comb_t[lane] = v;
}

// hc_pre collapses the hc copies into the one input the sublayer sees, weighted by the
// mix the PREVIOUS sublayer produced.  hc_post expands the sublayer's output back out and
// folds the residual in through `comb`, which Sinkhorn has made doubly stochastic.

struct dsv41_hc_mix_args {
    uint dim;
    uint hc;
    float norm_eps;
};

// A prefill chunk runs `rows` tokens at once; tgpig.y is the token, and every buffer is
// laid out token-major to match the reference's [t][hc][dim] stream.
kernel void kernel_dsv41_hc_pre(
        constant dsv41_hc_mix_args &args,
        device const float         *stream,  // [rows, hc, dim]
        device const float         *mix,     // [rows, hc]
        device float               *out,     // [rows, dim]
        uint2 tpitg [[thread_position_in_threadgroup]],
        uint2 tgpig [[threadgroup_position_in_grid]],
        uint2 ntg   [[threads_per_threadgroup]]) {
    const uint j = tgpig.x * ntg.x + tpitg.x;
    if (j >= args.dim) return;
    const uint t = tgpig.y;
    device const float *st = stream + (ulong)t * args.hc * args.dim;
    device const float *mx = mix + (ulong)t * args.hc;
    float acc = 0.0f;
    for (uint c = 0; c < args.hc; c++) acc = fma(mx[c], st[c * args.dim + j], acc);
    out[(ulong)t * args.dim + j] = acc;
}

kernel void kernel_dsv41_hc_post(
        constant dsv41_hc_mix_args &args,
        device const float         *sub,      // [rows, dim], the sublayer output
        device const float         *residual, // [rows, hc, dim]
        device const float         *post,     // [rows, hc]
        device const float         *comb,     // [rows, hc, hc]
        device float               *out,      // [rows, hc, dim]
        uint2 tpitg [[thread_position_in_threadgroup]],
        uint2 tgpig [[threadgroup_position_in_grid]],
        uint2 ntg   [[threads_per_threadgroup]]) {
    const uint j = tgpig.x * ntg.x + tpitg.x;
    if (j >= args.dim) return;
    const uint t = tgpig.y;
    const ulong hcd = (ulong)args.hc * args.dim;
    sub += (ulong)t * args.dim;
    residual += (ulong)t * hcd;
    post += (ulong)t * args.hc;
    comb += (ulong)t * args.hc * args.hc;
    out += (ulong)t * hcd;
    const float s = sub[j];
    // this thread's column of the residual is read into registers first, so `out` may
    // alias `residual` -- which it does when a sublayer writes back into the stream
    float r[DSV41_HC_MAX];
    for (uint k = 0; k < args.hc; k++) r[k] = residual[k * args.dim + j];
    for (uint c = 0; c < args.hc; c++) {
        float acc = post[c] * s;
        // comb is indexed [k * hc + c]: the sum runs over the residual copy k
        for (uint k = 0; k < args.hc; k++) acc = fma(comb[k * args.hc + c], r[k], acc);
        out[c * args.dim + j] = acc;
    }
}

// MoE routing.  scores = sqrt(softplus(logits)); the correction bias steers WHICH experts
// are picked but never scales them -- the weights come from the unbiased scores, are
// renormalised, then multiplied by routed_scaling_factor.
//
// The tie rule here is the OPPOSITE of the indexer's: repeated argmax with a strict `>`,
// so equals resolve to the LOWER index.  Getting that backwards silently reroutes tokens.

struct dsv41_route_args {
    uint n_expert;
    uint topk;
    float route_scale;
    uint norm_topk;
};

// One threadgroup per token: routing is tiny next to the experts it selects.
// One threadgroup per token.  The selection used to run as a repeated argmax in a single
// thread -- topk * n_expert * topk iterations, 574 us a call at 384 experts, a fifth of
// the whole decode step.  Ranking is the same order expressed in parallel: expert i is
// taken when fewer than topk experts beat it, and `beats` IS its slot, so the picks land
// in the identical order the argmax produced.
kernel void kernel_dsv41_route(
        constant dsv41_route_args &args,
        device const float        *logits,  // [rows, n_expert]
        device const float        *bias,    // [n_expert]
        device int                *idx_out, // [rows, topk]
        device float              *w_out,   // [rows, topk]
        threadgroup float         *scores [[threadgroup(0)]],
        uint2 tpitg [[thread_position_in_threadgroup]],
        uint2 tgpig [[threadgroup_position_in_grid]],
        uint2 ntg   [[threads_per_threadgroup]]) {
    logits += (ulong)tgpig.x * args.n_expert;
    idx_out += (ulong)tgpig.x * args.topk;
    w_out += (ulong)tgpig.x * args.topk;
    for (uint i = tpitg.x; i < args.n_expert; i += ntg.x) {
        const float x = logits[i];
        // softplus_stable: never exponentiate a large positive argument
        const float sp = x > 20.0f ? x : (x < -20.0f ? exp(x) : log(1.0f + exp(x)));
        scores[i] = sqrt(sp);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // the tie rule is the OPPOSITE of the indexer's: a strict `>` means equals resolve to
    // the LOWER index, so a lower-indexed equal counts as beating a higher-indexed one
    for (uint i = tpitg.x; i < args.n_expert; i += ntg.x) {
        const float v = scores[i] + bias[i];
        uint beats = 0u;
        for (uint j = 0; j < args.n_expert; j++) {
            const float o = scores[j] + bias[j];
            beats += (o > v || (o == v && j < i)) ? 1u : 0u;
        }
        if (beats < args.topk) {
            idx_out[beats] = (int)i;
            w_out[beats] = scores[i];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tpitg.x != 0u) return;
    if (args.norm_topk && args.topk > 1u) {
        float sum = 0.0f;
        for (uint s = 0; s < args.topk; s++) sum += w_out[s];
        // 1e-20, not norm_eps: this one matches training
        for (uint s = 0; s < args.topk; s++) w_out[s] /= (sum + 1.0e-20f);
    }
    for (uint s = 0; s < args.topk; s++) w_out[s] *= args.route_scale;
}
