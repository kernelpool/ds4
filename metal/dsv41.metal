// DeepSeek V4.1 keeps the RoPE tail in the quantized KV vector. Unlike V4,
// indexer Q/K have no Hadamard transform, and compressed KV has E4M3 scales.
// Construct scaled IEEE bits instead of approximating exp2/pow. An exponent
// overflow or discarded significand bit must not silently round source weights.
static uint dsv41_native_bits(uint magnitude, int power, uint scale, uint sign) {
    if (scale == 255u) return 0x7fc00000u;
    if (!magnitude) return sign;
    uint top = 31u - clz(magnitude);
    int exponent = int(top) + power + int(scale);
    uint significand = magnitude << (23u - top);
    if (exponent >= 255) return 0x7fc00000u;
    if (exponent <= 0) {
        uint shift = uint(1 - exponent);
        if (shift >= 32u || (significand & ((1u << shift) - 1u))) return 0x7fc00000u;
        return sign | (significand >> shift);
    }
    return sign | (uint(exponent) << 23u) | (significand & 0x7fffffu);
}

// Widen once per prefill layer. Preserve exact F32 bits, including signed
// zero, and flag any value that BF16 cannot represent without rounding.
kernel void kernel_dsv41_native_expand_mxfp4_bf16(
        constant ulong &count [[buffer(0)]], device const uchar *packed [[buffer(1)]],
        device ushort *out [[buffer(2)]], device atomic_uint *errors [[buffer(3)]],
        uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    const ulong i = (ulong(group.y) * 65535u + group.x) * 256u + tid;
    if (i >= count) return;
    const ulong block = (i / 32u) * 17u;
    const uint code = (packed[block + 1u + (i % 16u)] >> ((i % 32u >= 16u) ? 4u : 0u)) & 15u;
    constexpr uint magnitude[8] = {0u, 1u, 2u, 3u, 4u, 6u, 8u, 12u};
    const uint bits = dsv41_native_bits(magnitude[code & 7u], -1, packed[block], (code & 8u) << 28u);
    if (bits & 0xffffu) atomic_fetch_or_explicit(errors, 1u, memory_order_relaxed);
    out[i] = ushort(bits >> 16u);
}


struct ds4_metal_args_dsv41_quantize {
    uint width;
    uint rows;
    uint mode;
};

static inline float dsv41_bf16(float x) {
    uint bits = as_type<uint>(x);
    if ((bits & 0x7f800000u) != 0x7f800000u)
        bits += 0x7fffu + ((bits >> 16u) & 1u);
    return as_type<float>(bits & 0xffff0000u);
}
static inline float4 dsv41_bf16(float4 v) {
    return float4(dsv41_bf16(v.x), dsv41_bf16(v.y), dsv41_bf16(v.z), dsv41_bf16(v.w));
}

static inline float dsv41_pow2_ceil(float x) {
    const uint bits = as_type<uint>(x);
    return as_type<float>((bits & 0x7f800000u) +
                         ((bits & 0x7fffffu) ? 0x800000u : 0u));
}

kernel void kernel_dsv41_bf16_linear(
        constant ulong &count,
        device uint *x,
        uint gid [[thread_position_in_grid]]) {
    const ulong first = (ulong)gid * 4u;
    if (first + 4u <= count) {
        uint4 bits = *((device uint4 *)(x + first));
        const bool4 finite = (bits & 0x7f800000u) != 0x7f800000u;
        bits += select(uint4(0), uint4(0x7fffu) + ((bits >> 16u) & 1u), finite);
        *((device uint4 *)(x + first)) = bits & 0xffff0000u;
    } else {
        for (ulong i = first; i < count; i++) {
            uint bits = x[i];
            if ((bits & 0x7f800000u) != 0x7f800000u)
                bits += 0x7fffu + ((bits >> 16u) & 1u);
            x[i] = bits & 0xffff0000u;
        }
    }
}

struct ds4_metal_args_dsv41_router {
    uint n_expert, top_k, has_bias;
    float scale;
};

// One-token router for up to 1024 experts in one dispatch: the softplus/sqrt
// probability transform, the biased top-k through the same bitonic network as
// kernel_argsort_f32_i32_desc (padding indices sort last), and the weights
// normalised as the generic chain does: gather, SIMD sum over the top_k lanes,
// clamp, divide, scale. Threadgroup memory holds nth ints and 2 * nth floats.
kernel void kernel_dsv41_router_one(
        constant ds4_metal_args_dsv41_router &args,
        device const float *logits,
        device const float *bias,
        device float *probs,
        device int32_t *selected,
        device float *weights,
        threadgroup int32_t *shmem_i32 [[threadgroup(0)]],
        uint2 tg [[threadgroup_position_in_grid]],
        uint2 tpos [[thread_position_in_threadgroup]],
        uint2 tsize [[threads_per_threadgroup]]) {
    const uint col = tpos.x, ntg = tsize.x;
    const int width = (int)args.n_expert;
    logits += tg.y * args.n_expert;   /* one threadgroup per row */
    probs += tg.y * args.n_expert;
    selected += tg.y * args.top_k;
    weights += tg.y * args.top_k;
    threadgroup float *score = (threadgroup float *)(shmem_i32 + ntg);
    threadgroup float *prob = score + ntg;
    shmem_i32[col] = (int)col;
    if ((int)col < width) {
        const float x = logits[col];
        const float ex = exp(x);
        const float em = min(ex, 0.03125f);
        const float poly = em*(1.0f - em*(0.5f - em*(1.0f/3.0f - 0.25f*em)));
        const float sp = select(select(log(1.0f + ex), poly, ex < 0.03125f), x, x > 20.0f);
        const float p = sqrt(sp);
        probs[col] = p;
        prob[col] = p;
        score[col] = args.has_bias ? p + bias[col] : p;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int k = 2; k <= (int)ntg; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            const int ixj = (int)col ^ j;
            if (ixj > (int)col) {
                const int a = shmem_i32[col], b = shmem_i32[ixj];
                const bool swap = ((int)col & k) == 0 ?
                    (a >= width || (b < width && score[a] < score[b])) :
                    (b >= width || (a < width && score[a] > score[b]));
                if (swap) { shmem_i32[col] = b; shmem_i32[ixj] = a; }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    if (col < args.top_k) {
        const int32_t idx = shmem_i32[col];
        selected[col] = idx;
        const float w = prob[idx];
        const float total = clamp(simd_sum(w), 6.103515625e-5f, INFINITY);
        // two stores as in the generic chain: the quotient is rounded before the scale
        threadgroup volatile float *quotient = (threadgroup volatile float *)(prob + ntg) + col;
        *quotient = w / total;
        weights[col] = *quotient * args.scale;
    }
}

struct ds4_metal_args_dsv41_rope {
    uint width, heads, rows, start, inverse, stride;
    float frequencies[32];
    uint round_all;
};

kernel void kernel_dsv41_rope(
        constant ds4_metal_args_dsv41_rope &args,
        device float *x,
        uint2 group [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_simdgroup]]) {
    const float theta = float(args.start + group.y * args.stride) * args.frequencies[lane];
    const float c = precise::cos(theta);
    const float s = args.inverse ? -precise::sin(theta) : precise::sin(theta);
    const ulong head = ((ulong)group.y * args.heads + group.x) * args.width;
    if (args.round_all) {
        for (uint k = lane; k + 64u < args.width; k += 32u) x[head + k] = dsv41_bf16(x[head + k]);
    }
    const ulong i = head + args.width - 64u + 2u * lane;
    float re = x[i], im = x[i + 1u];
    if (args.round_all) { re = dsv41_bf16(re); im = dsv41_bf16(im); }
    x[i] = dsv41_bf16(re * c - im * s);
    x[i + 1u] = dsv41_bf16(re * s + im * c);
}

// Compare in the scaled domain: Metal division can move an exact midpoint
// by an ULP even in precise mode. Products of E4M3 and E2M1 fit exactly in F32.
static float dsv41_fp4_scaled(float value, float scale) {
    const float ax=abs(value);
    int best=0;float distance=ax;
    for(int i=1;i<8;i++) {
        const float d=abs(ax-scale*dsv4_e2m1fn_values[i]);
        if(d<distance || (d==distance && !(i&1) && (best&1))) {best=i;distance=d;}
    }
    return copysign(scale*dsv4_e2m1fn_values[best],value);
}

static float dsv41_fp4_e4m3_scale(float amax) {
    const float ax=max(amax,0.01171875f);
    int lo=0,hi=126;
    while(lo<hi) {
        const int mid=(lo+hi+1)>>1;
        if(6.f*dsv4_e4m3fn_value(mid)<=ax)lo=mid;else hi=mid-1;
    }
    int best=lo;
    if(best<126) {
        const float d0=abs(ax-6.f*dsv4_e4m3fn_value(best));
        const float d1=abs(ax-6.f*dsv4_e4m3fn_value(best+1));
        if(d1<d0 || (d1==d0 && !((best+1)&1) && (best&1)))best++;
    }
    return dsv4_e4m3fn_value(best);
}

kernel void kernel_dsv41_quantize(
        constant ds4_metal_args_dsv41_quantize &args,
        device float *x,
        uint2 group [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_simdgroup]]) {
    const uint block = args.mode == 3u ? 16u : 32u;
    const uint column = group.x * block + lane;
    const bool valid = lane < block && column < args.width;
    const ulong index = (ulong)group.y * args.width + column;
    const float value = valid ? dsv41_bf16(x[index]) : 0.0f;
    const float amax = simd_max(abs(value));
    float result = value;
    if (args.mode == 1u) {
        const float scale = dsv41_pow2_ceil(max(amax, 1.0e-4f) * (1.0f / 448.0f));
        result = copysign(dsv4_e4m3fn_dequant(abs(value) / scale), value) * scale;
    } else if (args.mode == 2u || args.mode == 3u) {
        const float scale = args.mode == 3u
            ? dsv41_fp4_e4m3_scale(amax)
            : dsv41_pow2_ceil(max(amax, 7.052966104933725e-38f) * (1.0f / 6.0f));
        result = dsv41_fp4_scaled(value, scale);
    }
    if (valid) x[index] = dsv41_bf16(result);
}

struct ds4_metal_args_dsv41_rope_quantize {
    uint width, start, inverse, mode;
    float frequencies[32];
};

// One row's rope, quantization and store into a cache row: the rotation of
// the last 64 columns as kernel_dsv41_rope applies it, then the block
// quantization of kernel_dsv41_quantize, written to dst.
static inline void dsv41_rope_quantize_body(
        constant ds4_metal_args_dsv41_rope_quantize &args,
        device const float *x,
        device float *dst,
        uint group,
        uint lane) {
    const uint block = args.mode == 3u ? 16u : 32u;
    const uint column = group * block + lane;
    const bool valid = lane < block && column < args.width;
    float value = valid ? x[column] : 0.0f;
    const float other = simd_shuffle_xor(value, 1u);
    const uint rotary = args.width - 64u;
    if (valid && column >= rotary) {
        const float theta = float(args.start) * args.frequencies[(column - rotary) >> 1u];
        const float c = precise::cos(theta);
        const float s = args.inverse ? -precise::sin(theta) : precise::sin(theta);
        const float re = (column & 1u) ? other : value;
        const float im = (column & 1u) ? value : other;
        // the contraction kernel_dsv41_rope compiles to
        value = (column & 1u) ? dsv41_bf16(fma(im, c, re * s)) : dsv41_bf16(fma(re, c, -(im * s)));
    }
    value = dsv41_bf16(value);
    const float amax = simd_max(abs(value));
    float result = value;
    if (args.mode == 1u) {
        const float scale = dsv41_pow2_ceil(max(amax, 1.0e-4f) * (1.0f / 448.0f));
        result = copysign(dsv4_e4m3fn_dequant(abs(value) / scale), value) * scale;
    } else if (args.mode == 2u || args.mode == 3u) {
        const float scale = args.mode == 3u
            ? dsv41_fp4_e4m3_scale(amax)
            : dsv41_pow2_ceil(max(amax, 7.052966104933725e-38f) * (1.0f / 6.0f));
        result = dsv41_fp4_scaled(value, scale);
    }
    if (valid) dst[column] = dsv41_bf16(result);
}

kernel void kernel_dsv41_rope_quantize(
        constant ds4_metal_args_dsv41_rope_quantize &args,
        device const float *x,
        device float *dst,
        uint group [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_simdgroup]]) {
    dsv41_rope_quantize_body(args, x, dst, group, lane);
}

/* The decode indexer query's rope and quantization in one dispatch over every head, in place: dsv41_rope_quantize_body
 * with (cos, sin) read from the step's RoPE table (the same theta and contraction, so the same bits). */
kernel void kernel_dsv41_rope_quantize_heads_tab(
        constant ds4_metal_args_dsv41_rope_quantize &args,
        device float *x,
        device const float2 *tab,
        uint2 group [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_simdgroup]]) {
    device float *row = x + (ulong)group.y * args.width;
    const uint block = args.mode == 3u ? 16u : 32u;
    const uint column = group.x * block + lane;
    const bool valid = lane < block && column < args.width;
    float value = valid ? row[column] : 0.0f;
    const float other = simd_shuffle_xor(value, 1u);
    const uint rotary = args.width - 64u;
    if (valid && column >= rotary) {
        const float2 cs = tab[(column - rotary) >> 1u];
        const float c = cs.x;
        const float s = args.inverse ? -cs.y : cs.y;
        const float re = (column & 1u) ? other : value;
        const float im = (column & 1u) ? value : other;
        /* the contraction kernel_dsv41_rope compiles to */
        value = (column & 1u) ? dsv41_bf16(fma(im, c, re * s)) : dsv41_bf16(fma(re, c, -(im * s)));
    }
    value = dsv41_bf16(value);
    const float amax = simd_max(abs(value));
    float result = value;
    if (args.mode == 1u) {
        const float scale = dsv41_pow2_ceil(max(amax, 1.0e-4f) * (1.0f / 448.0f));
        result = copysign(dsv4_e4m3fn_dequant(abs(value) / scale), value) * scale;
    } else if (args.mode == 2u || args.mode == 3u) {
        const float scale = args.mode == 3u
            ? dsv4_e4m3fn_dequant(max(amax, 0.01171875f) / 6.0f)
            : dsv41_pow2_ceil(max(amax, 7.052966104933725e-38f) * (1.0f / 6.0f));
        result = copysign(dsv4_e2m1fn_dequant(abs(value) / scale), value) * scale;
    }
    if (valid) row[column] = dsv41_bf16(result);
}

struct ds4_metal_args_dsv41_norm_pair {
    uint n0, n1, ntg0, ntg1;
    float eps;
};

// Two weighted RMS norms in one dispatch, each row on its own thread count
// (ntg0 threads then ntg1), so every reduction matches the single-row kernel.
kernel void kernel_dsv41_norm_pair(
        constant ds4_metal_args_dsv41_norm_pair &args,
        device const float4 *x0,
        device const float4 *w0,
        device float4 *y0,
        device const float4 *x1,
        device const float4 *w1,
        device float4 *y1,
        threadgroup float *shmem [[threadgroup(0)]],
        uint2 tg [[threadgroup_position_in_grid]],
        ushort2 tpos [[thread_position_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const ushort tid = tpos.x;
    const bool second = tid >= args.ntg0;
    const uint ntg = second ? args.ntg1 : args.ntg0;
    const uint t = second ? tid - args.ntg0 : tid;
    const uint sg = second ? sgitg - args.ntg0 / 32u : sgitg;
    const int n = int(second ? args.n1 : args.n0);
    const int ne00_t = n / 4;
    const uint row = tg.y;   /* grid row = activation row */
    device const float4 *x = (second ? x1 : x0) + (ulong)row * ne00_t;
    device const float4 *w = second ? w1 : w0;
    device float4 *y = (second ? y1 : y0) + (ulong)row * ne00_t;
    threadgroup float *sh = shmem + (second ? 32u : 0u);
    if (sg == 0) {
        sh[tiisg] = 0.0f;
    }
    float sumf = 0.0f;
    for (int i = int(t); i < ne00_t; i += int(ntg)) {
        sumf += dot(x[i], x[i]);
    }
    sumf = simd_sum(sumf);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tiisg == 0) {
        sh[sg] = sumf;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sumf = sh[tiisg];
    sumf = simd_sum(sumf);
    const float mean  = sumf/n;
    const float scale = 1.0f/sqrt(mean + args.eps);
    for (int i = int(t); i < ne00_t; i += int(ntg)) {
        float4 v = x[i]*scale;
        v = v*w[i];
        y[i] = ds4_bf16_round(v);
    }
}

struct ds4_metal_args_dsv41_engram {
    uint width;
    uint rows;
    float eps;
    uint masked;
};

kernel void kernel_dsv41_engram_add(
        constant ds4_metal_args_dsv41_engram &args,
        device float *residual,
        device const float *kv,
        device const float *q_weight,
        device const float *k_weight,
        device const uchar *mask,
        uint2 group [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_simdgroup]]) {
    if (args.masked && !mask[group.x]) return;
    const ulong offset = ((ulong)group.x * 4u + group.y) * args.width;
    const ulong key_offset = ((ulong)group.x * 5u + group.y) * args.width;
    const ulong value_offset = ((ulong)group.x * 5u + 4u) * args.width;
    float h2 = 0.0f, k2 = 0.0f, dot = 0.0f;
    for (uint i = lane; i < args.width; i += 32u) {
        const float h = residual[offset + i];
        const float k = dsv41_bf16(kv[key_offset + i]);
        const uint wi = group.y * args.width + i;
        h2 += h * h;
        k2 += k * k;
        dot += h * (q_weight[wi] * k_weight[wi]) * k;
    }
    h2 = simd_sum(h2);
    k2 = simd_sum(k2);
    dot = simd_sum(dot) * rsqrt(h2 / args.width + args.eps) *
          rsqrt(k2 / args.width + args.eps) * rsqrt(float(args.width));
    const float gate = 1.0f / (1.0f + exp(-copysign(sqrt(max(abs(dot), 1.0e-6f)), dot)));
    for (uint i = lane; i < args.width; i += 32u)
        residual[offset + i] = dsv41_bf16(residual[offset + i] +
            gate * dsv41_bf16(kv[value_offset + i]));
}

/* A single product, rounded as written: fast math may not reassociate it into a neighbouring product. */
static inline float dsv41_engram_mul(float a, float b) {
#pragma METAL fp math_mode(safe)
    return a * b;
}

/* kernel_dsv41_engram_add over eight SIMD groups per stream (width 5120): each lane's chain runs in order, segment by
 * segment, handing its three accumulators on, with the fma contractions the 32-lane kernel compiles to written out. */
kernel void kernel_dsv41_engram_add_tg256(
        constant ds4_metal_args_dsv41_engram &args,
        device float *residual,
        device const float *kv,
        device const float *q_weight,
        device const float *k_weight,
        device const uchar *mask,
        threadgroup float *sh [[threadgroup(0)]],
        uint2 group [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_simdgroup]],
        uint sg [[simdgroup_index_in_threadgroup]]) {
    if (args.masked && !mask[group.x]) return;
    constexpr uint W = 5120u, J = W / 256u;
    const uint e0 = sg * 32u * J + lane;
    device float *hp = residual + ((ulong)group.x * 4u + group.y) * W + e0;
    device const float *kp = kv + ((ulong)group.x * 5u + group.y) * W + e0;
    device const float *vp = kv + ((ulong)group.x * 5u + 4u) * W + e0;
    device const float *qp = q_weight + (ulong)group.y * W + e0;
    device const float *wp = k_weight + (ulong)group.y * W + e0;
    float h[J], k[J], p[J], v[J];
    for (uint j = 0; j < J; j++) {
        h[j] = hp[32u * j];
        k[j] = dsv41_bf16(kp[32u * j]);
        p[j] = dsv41_engram_mul(dsv41_engram_mul(h[j], qp[32u * j]), wp[32u * j]);
        v[j] = vp[32u * j];
    }
    float h2 = 0.0f, k2 = 0.0f, dot = 0.0f;
    for (uint s = 0; s < 8u; s++) {
        if (sg == s) {
            if (s) { h2 = sh[lane]; k2 = sh[32u + lane]; dot = sh[64u + lane]; }
            for (uint j = 0; j < J; j++) {
                h2 = fma(h[j], h[j], h2);
                k2 = fma(k[j], k[j], k2);
                dot = fma(p[j], k[j], dot);
            }
            if (s < 7u) { sh[lane] = h2; sh[32u + lane] = k2; sh[64u + lane] = dot; }
        }
        if (s < 7u) threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (sg == 7u) {
        h2 = simd_sum(h2);
        k2 = simd_sum(k2);
        dot = simd_sum(dot) * rsqrt(h2 / args.width + args.eps) *
              rsqrt(k2 / args.width + args.eps) * rsqrt(float(args.width));
        const float gate = 1.0f / (1.0f + exp(-copysign(sqrt(max(abs(dot), 1.0e-6f)), dot)));
        if (lane == 0u) sh[96] = gate;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float gate = sh[96];
    for (uint j = 0; j < J; j++)
        hp[32u * j] = dsv41_bf16(fma(gate, dsv41_bf16(v[j]), h[j]));
}

struct ds4_metal_args_dsv41_pool {
    uint width;
    uint pairs;
    uint tail;
};

kernel void kernel_dsv41_pool2(
        constant ds4_metal_args_dsv41_pool &args,
        device float *out,
        device const float *kv,
        device const float *scores,
        device const float *previous_kv,
        device const float *previous_scores,
        uint2 index [[thread_position_in_grid]]) {
    if (index.x >= args.width || index.y >= args.pairs) return;
    const long a = (long)index.y * 2 - args.tail;
    const ulong b = (ulong)(a + 1) * args.width + index.x;
    const float ka = a < 0 ? previous_kv[index.x] : kv[(ulong)a * args.width + index.x];
    const float sa = a < 0 ? previous_scores[index.x] : scores[(ulong)a * args.width + index.x];
    const float sb = scores[b], peak = max(sa, sb);
    const float ea = exp(sa - peak), eb = exp(sb - peak);
    out[(ulong)index.y * args.width + index.x] = dsv41_bf16((ka * ea + kv[b] * eb) / (ea + eb));
}

struct ds4_metal_args_dsv41_indexer_all {
    uint rows;
    uint start;
    uint ratio;
    uint top_k;
};

// Rows with at most top_k visible keys attend to all of them.
kernel void kernel_dsv41_indexer_all(
        constant ds4_metal_args_dsv41_indexer_all &args,
        device int *ids,
        uint2 index [[thread_position_in_grid]]) {
    if (index.x >= args.top_k || index.y >= args.rows) return;
    const uint visible = (args.start + index.y + 1u) / args.ratio;
    if (index.x < min(visible, args.top_k))
        ids[(ulong)index.y * args.top_k + index.x] = (int)index.x;
}

struct ds4_metal_args_dsv41_candidates {
    uint width;
    uint rows;
    uint start;
    uint ratio;
    uint mask_stride;
};

kernel void kernel_dsv41_candidate_blocks(
        constant ds4_metal_args_dsv41_candidates &args,
        device const float *scores,
        device float *blocks,
        device const float *unused,
        uint2 index [[thread_position_in_grid]]) {
    (void)unused;
    const uint count = (args.width + 7u) / 8u;
    if (index.x >= count || index.y >= args.rows) return;
    const uint visible = min(args.width, (args.start + index.y + 1u) / args.ratio);
    float best = -INFINITY;
    for (uint i = index.x * 8u; i < min(visible, (index.x + 1u) * 8u); i++)
        best = max(best, scores[(ulong)index.y * args.width + i]);
    if (visible && index.x == (visible - 1u) / 8u) best = INFINITY;
    blocks[(ulong)index.y * count + index.x] = best;
}

kernel void kernel_dsv41_candidate_filter(
        constant ds4_metal_args_dsv41_candidates &args,
        device const float *scores,
        device float *out,
        device const float *block_mask,
        uint2 index [[thread_position_in_grid]]) {
    if (index.x >= args.width || index.y >= args.rows) return;
    const ulong offset = (ulong)index.y * args.width + index.x;
    const uint visible = min(args.width, (args.start + index.y + 1u) / args.ratio);
    out[offset] = index.x < visible &&
        block_mask[(ulong)index.y * args.mask_stride + index.x / 8u] == 0.0f
        ? scores[offset] : -INFINITY;
}

/* The 2048 candidate blocks of a 0/-inf mask in ascending order; one
 * threadgroup of 1024 threads, four mask entries per thread and step. */
kernel void kernel_dsv41_candidate_list(
        constant uint &blocks,
        device const float *mask,
        device uint *list,
        uint tid [[thread_index_in_threadgroup]],
        uint lane [[thread_index_in_simdgroup]],
        uint simd [[simdgroup_index_in_threadgroup]]) {
    threadgroup uint part[33];
    uint base = 0u;
    for (uint b0 = 0u; b0 < blocks; b0 += 4096u) {
        const uint b = b0 + tid * 4u;
        bool keep[4];
        uint count = 0u;
        for (uint j = 0u; j < 4u; j++) {
            keep[j] = b + j < blocks && mask[b + j] == 0.0f;
            count += keep[j] ? 1u : 0u;
        }
        const uint before = simd_prefix_exclusive_sum(count);
        if (lane == 31u) part[simd] = before + count;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd == 0u) {
            const uint c = part[lane], c_before = simd_prefix_exclusive_sum(c);
            part[lane] = c_before;
            if (lane == 31u) part[32] = c_before + c;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint at = base + part[simd] + before;
        for (uint j = 0u; j < 4u; j++) {
            if (keep[j] && at < 2048u) list[at] = b + j;
            at += keep[j] ? 1u : 0u;
        }
        base += part[32];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

/* Positions in the candidate list back to key ids, in place and in order. */
kernel void kernel_dsv41_candidate_ids(
        constant uint &count,
        device const uint *list,
        device int *selected,
        uint i [[thread_position_in_grid]]) {
    if (i >= count) return;
    const uint p = (uint)selected[i];
    selected[i] = (int)(list[p >> 3] * 8u + (p & 7u));
}

struct ds4_metal_args_dsv41_carry {
    uint width, rows, words, format, pack;
};

kernel void kernel_dsv41_carry_copy(
        constant ds4_metal_args_dsv41_carry &args,
        device uint *packed, device float *plain,
        uint2 group [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]]) {
    const uint col = group.x * 128u + tid;
    const ulong row = group.y;
    if (args.format == 0u) {
        if (col >= args.width) return;
        device ushort *p = (device ushort *)(packed + row * args.words);
        if (args.pack) p[col] = ushort(as_type<uint>(plain[row * args.width + col]) >> 16);
        else plain[row * args.width + col] = as_type<float>(uint(p[col]) << 16);
    } else {
        const uint word = col / 32u;
        if (args.pack) {
            const bool allowed = col < args.width && plain[row * args.width + col] == 0.0f;
            const uint bits = simd_sum(allowed ? 1u << lane : 0u);
            if (!lane && word < args.words) packed[row * args.words + word] = bits;
        } else if (col < args.width) {
            const uint bits = packed[row * args.words + word];
            plain[row * args.width + col] = bits & (1u << lane) ? 0.0f : -INFINITY;
        }
    }
}

#ifdef DS4_METAL_HAS_TENSOR
kernel void kernel_dsv41_indexer_pack(
        constant uint4 &args,
        device const float *q, device const float *keys,
        device uint *flags, device bfloat *packed_q, device bfloat *packed_keys,
        threadgroup uint *valid [[threadgroup(0)]],
        uint group [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]]) {
    const bool query = group < args.y;
    const uint count = query ? 32u * 128u : 64u * 128u;
    const ulong offset = query ? (ulong)group * count : (ulong)(group - args.y) * count;
    bool exact = true;
    for (uint i = tid; i < count; i += 128u) {
        const float value = query ? q[offset + i] :
            offset + i < (ulong)args.x * 128u ? keys[offset + i] : 0.0f;
        const bfloat converted = bfloat(value);
        if (query) packed_q[offset + i] = converted;
        else packed_keys[offset + i] = converted;
        exact = exact && float(converted) == value;
    }
    const bool same = simd_all(exact);
    if (!(tid % 32u)) valid[tid / 32u] = same;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!tid) flags[group] = valid[0] && valid[1] && valid[2] && valid[3];
}

kernel void kernel_dsv41_indexer_scores_packed(
        constant uint4 &args, constant uint2 &range,
        device const float *q, device const float *weights,
        device const float *keys, device float *scores,
        device const uint *flags, device const bfloat *packed_q,
        device const bfloat *packed_keys,
        threadgroup float *scratch [[threadgroup(0)]],
        uint2 group [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]]) {
    constexpr int HEADS = 32, KEYS = 64, DIM = 128;
    const uint width = args.x, token = group.y, row0 = group.x * KEYS;
    const uint visible = (args.z + token + 1u) / args.w;
    if (row0 >= visible) {
        if (tid < KEYS && row0 + tid < width)
            scores[(ulong)token * width + row0 + tid] = -INFINITY;
        return;
    }
    matmul2d<matmul2d_descriptor(HEADS, KEYS, DIM, false, true, false,
        matmul2d_descriptor::mode::multiply_accumulate), execution_simdgroups<4>> mm;
    auto result = tensor(scratch, dextents<int32_t, 2>(KEYS, HEADS));
    if (flags[range.y + token] && flags[range.x + group.x]) {
        auto query = tensor((device bfloat *)packed_q + (ulong)(range.y + token) * HEADS * DIM,
                             dextents<int32_t, 2>(DIM, HEADS));
        auto key = tensor((device bfloat *)packed_keys + (ulong)row0 * DIM,
                           dextents<int32_t, 2>(DIM, KEYS));
        auto dots = mm.template get_destination_cooperative_tensor<decltype(query), decltype(key), float>();
        for (uint16_t i = 0; i < dots.get_capacity(); i++)
            if (dots.is_valid_element(i)) dots[i] = 0;
        mm.run(query, key, dots);
        dots.store(result);
    } else {
        auto query = tensor((device float *)q + (ulong)token * HEADS * DIM,
                             dextents<int32_t, 2>(DIM, HEADS));
        auto all_keys = tensor((device float *)keys, dextents<int32_t, 2>(DIM, int(width)));
        auto key = all_keys.slice(0, int(row0));
        auto dots = mm.template get_destination_cooperative_tensor<decltype(query), decltype(key), float>();
        for (uint16_t i = 0; i < dots.get_capacity(); i++)
            if (dots.is_valid_element(i)) dots[i] = 0;
        mm.run(query, key, dots);
        dots.store(result);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < KEYS && row0 + tid < width) {
        float sum = 0;
        for (uint h = 0; h < HEADS; h++)
            sum += max(scratch[h * KEYS + tid] * (1.0f / 64.0f), 0.0f) * weights[token * HEADS + h];
        scores[(ulong)token * width + row0 + tid] = row0 + tid < visible ? sum : -INFINITY;
    }
}
#endif

// Single-Mac decode: the block input's collapse (the weighted sum of the four streams with the previous block's
// weights) and its norm run first, then the block's own hyper-connection weights run inside the query projection's
// grid, each with its original reduction order. The query groups read the first collapse's norm; the weights' groups
// run dsv41_hc_block_input_body whole, including its own copy of the collapse and norm, without which the compiled
// Sinkhorn split's bits change.
struct ds4_deferred_hc_args {
    int32_t n_embd, n_hc, sinkhorn_iters;
    float hc_eps, norm_eps;
};
kernel void kernel_dsv41_deferred_collapse(
        constant ds4_deferred_hc_args &args,
        device const float *x, device const float *pre,
        device float *collapsed, device const float *norm_weight,
        device float *norm_dst, device atomic_uint *counter,
        threadgroup float *shared [[threadgroup(0)]],
        ushort tid [[thread_position_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort ntg [[threads_per_threadgroup]]) {

    const uint n_embd = (uint)args.n_embd;
    const uint n4 = n_embd >> 2;

    threadgroup float4 *row = (threadgroup float4 *)shared;
    threadgroup float  *sum_shmem = shared + n_embd;

    /* kernel_rms_norm_fuse_impl zeroes its 32 accumulator slots from
     * simdgroup 0 before the parallel sum. */
    if (sgitg == 0) {
        sum_shmem[tiisg] = 0.0f;
    }

    /* --- Collapse with the preceding sublayer's pre weights, then its BF16
     * boundary, and the weighted RMS partial sum over the rounded values. --- */
    const float w0 = pre[0], w1 = pre[1], w2 = pre[2], w3 = pre[3];
    device const float4 *x0 = (device const float4 *)(x + 0u * n_embd);
    device const float4 *x1 = (device const float4 *)(x + 1u * n_embd);
    device const float4 *x2 = (device const float4 *)(x + 2u * n_embd);
    device const float4 *x3 = (device const float4 *)(x + 3u * n_embd);
    device float4 *collapsed4 = (device float4 *)collapsed;

    float sumf = 0.0f;
    for (uint i = tid; i < n4; i += ntg) {
        float4 acc = 0.0f;
        acc += x0[i] * w0;
        acc += x1[i] * w1;
        acc += x2[i] * w2;
        acc += x3[i] * w3;
        const float4 v = float4(dsv41_bf16(acc.x), dsv41_bf16(acc.y),
                                dsv41_bf16(acc.z), dsv41_bf16(acc.w));
        collapsed4[i] = v;
        row[i] = v;
        sumf += dot(v, v);
    }

    sumf = simd_sum(sumf);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tiisg == 0) {
        sum_shmem[sgitg] = sumf;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    sumf = sum_shmem[tiisg];
    sumf = simd_sum(sumf);

    const float mean  = sumf / (float)args.n_embd;
    const float rscale = 1.0f / sqrt(mean + args.norm_eps);

    device const float4 *w = (device const float4 *)norm_weight;
    device float4 *y = (device float4 *)norm_dst;
    for (uint i = tid; i < n4; i += ntg) {
        const float4 v = (row[i] * rscale) * w[i];
        y[i] = float4(dsv41_bf16(v.x), dsv41_bf16(v.y),
                      dsv41_bf16(v.z), dsv41_bf16(v.w));
    }

    if (tid == 0) atomic_store_explicit(counter, 0u, memory_order_relaxed);

}

/* kernel_dsv41_deferred_collapse as two dispatches over 32 + 10 groups: group s does the original's simdgroup s (the
 * same fma chain, written out) and the norm repeats its second simd_sum on the same 32 partials, so the same bits. */
kernel void kernel_dsv41_deferred_collapse_part(
        constant ds4_deferred_hc_args &args,
        device const float *x, device const float *pre,
        device float *collapsed, device float *slots,
        uint group [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const uint n_embd = (uint)args.n_embd, n4 = n_embd >> 2;
    const uint i0 = group * 32u + tiisg, i1 = i0 + 1024u;
    const bool h0 = i0 < n4, h1 = i1 < n4;
    device const float4 *x0 = (device const float4 *)(x + 0u * n_embd);
    device const float4 *x1 = (device const float4 *)(x + 1u * n_embd);
    device const float4 *x2 = (device const float4 *)(x + 2u * n_embd);
    device const float4 *x3 = (device const float4 *)(x + 3u * n_embd);
    const float w0 = pre[0], w1 = pre[1], w2 = pre[2], w3 = pre[3];
    float4 a0 = 0.0f, b0 = 0.0f, c0 = 0.0f, d0 = 0.0f, a1 = 0.0f, b1 = 0.0f, c1 = 0.0f, d1 = 0.0f;
    if (h0) { a0 = x0[i0]; b0 = x1[i0]; c0 = x2[i0]; d0 = x3[i0]; }
    if (h1) { a1 = x0[i1]; b1 = x1[i1]; c1 = x2[i1]; d1 = x3[i1]; }
    device float4 *collapsed4 = (device float4 *)collapsed;
    float sumf = 0.0f;
    if (h0) {
        const float4 v = dsv41_bf16(fma(d0, w3, fma(c0, w2, fma(b0, w1, a0 * w0))));
        collapsed4[i0] = v;
        sumf += dot(v, v);
    }
    if (h1) {
        const float4 v = dsv41_bf16(fma(d1, w3, fma(c1, w2, fma(b1, w1, a1 * w0))));
        collapsed4[i1] = v;
        sumf += dot(v, v);
    }
    sumf = simd_sum(sumf);
    if (tiisg == 0) slots[group] = sumf;
}

kernel void kernel_dsv41_deferred_collapse_norm(
        constant ds4_deferred_hc_args &args,
        device const float *collapsed, device const float *slots,
        device const float *norm_weight, device float *norm_dst, device atomic_uint *counter,
        uint gid [[thread_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const uint n4 = (uint)args.n_embd >> 2;
    float sumf = slots[tiisg];
    sumf = simd_sum(sumf);
    const float mean  = sumf / (float)args.n_embd;
    const float rscale = 1.0f / sqrt(mean + args.norm_eps);
    if (gid < n4) {
        const float4 r = ((device const float4 *)collapsed)[gid];
        const float4 v = (r * rscale) * ((device const float4 *)norm_weight)[gid];
        ((device float4 *)norm_dst)[gid] = float4(dsv41_bf16(v.x), dsv41_bf16(v.y), dsv41_bf16(v.z), dsv41_bf16(v.w));
    }
    if (gid == 0) atomic_store_explicit(counter, 0u, memory_order_relaxed);
}

kernel void kernel_dsv41_query_deferred_hc(
        constant ds4_metal_args_hc_norm_mix &hc [[buffer(0)]],
        constant ds4_metal_args_dsv4_hc_split_weighted_sum_norm &split_args [[buffer(1)]],
        device const char *residual [[buffer(2)]], device const char *hc_weight [[buffer(3)]],
        device char *mix [[buffer(4)]], device const float *scale [[buffer(5)]],
        device const float *base [[buffer(6)]], device char *split [[buffer(7)]],
        device atomic_uint *completion [[buffer(8)]],
        constant ds4_metal_args_mul_mv &mv [[buffer(9)]],
        device const char *weight [[buffer(10)]], device const char *x [[buffer(11)]],
        device float *q [[buffer(12)]], constant ds4_metal_args_mv_rope &rope [[buffer(13)]],
        device char *collapsed [[buffer(14)]], device const char *norm_weight [[buffer(15)]],
        device char *norm [[buffer(16)]], device const float *pre [[buffer(17)]],
        constant ds4_metal_args_dsv41_rope_quantize &kv_args [[buffer(18)]],
        device const float *kv [[buffer(19)]], device float *cache [[buffer(20)]],
        threadgroup char *shmem [[threadgroup(0)]], uint3 group [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]], ushort simd [[simdgroup_index_in_threadgroup]]) {
    if (group.x < 16u) {
        dsv41_hc_block_input_body<2>(hc, split_args, residual, hc_weight, mix,
            scale, base, split, collapsed, norm_weight, norm, pre, completion, shmem,
            group, lane, simd);
        return;
    }
    if (group.x < 18u) {
        dsv41_rope_quantize_body(kv_args, kv, cache, (group.x - 16u) * 8u + simd, lane);
        return;
    }
    const uint tile = group.x - 18u;
    const float value = ds4_q8_short_value<40, false>(mv, weight, x, uint3(tile*2u,0,0), lane, simd);
    threadgroup float *pairs = (threadgroup float *)shmem;
    if (lane == 0) pairs[simd] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0 && (simd & 1u) == 0) {
        const uint row = tile * 8u + simd, col = row % rope.width;
        const float re = pairs[simd], im = pairs[simd + 1u];
        if (col >= rope.width - 64u) {
            const float theta = float(rope.start) * rope.frequencies[(col - (rope.width - 64u)) >> 1u];
            const float c = precise::cos(theta);
            const float sn = precise::sin(theta);
            q[row] = ds4_bf16_round(fma(re, c, -(im * sn)));
            q[row + 1u] = ds4_bf16_round(fma(im, c, re * sn));
        } else { q[row] = re; q[row + 1u] = im; }
    }
}

/* The decode reduce of flash_attn.metal with V4.1's transforming writer. */
kernel void kernel_dsv41_flash_attn_vec_reduce_heads(
        constant ds4_metal_args_flash_attn_ext_vec_reduce & args,
        device  const char * htmp,
        device        char * dst,
        device const float2 * rope_tab,
        uint   tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    ds4_flash_attn_vec_reduce_row<true>(args, htmp, dst, rope_tab, tgpig, tiisg, sgitg,
                                        (short)FC_flash_attn_ext_vec_reduce_NWG,
                                        (short)FC_flash_attn_ext_vec_reduce_DV);
}

// The released FP8 GEMM resets its dot accumulator every 32 values, then
// adds scaled block results in K order. Power-of-two scales commute with
// these normal F32 products; dense storage contains their exact values.
kernel void kernel_dsv41_native_fp8_mv(
        constant uint4 &a [[buffer(0)]], device const uchar *w [[buffer(1)]],
        device const float *x [[buffer(2)]], device float *out [[buffer(3)]],
        uint row [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    float sum = 0.f;
    for (uint col = lane; col < a.x; col += 32u) {
        ulong i = ulong(row) * a.x + col;
        float weight = a.z ? as_type<float>(uint(reinterpret_cast<device const ushort *>(w)[i]) << 16u) :
                            reinterpret_cast<device const float *>(w)[i];
        sum += simd_sum(weight * x[col]);
    }
    if (!lane) out[row] = dsv41_bf16(sum);
}

// Four independent tokens reuse each weight load. Each dot still reduces
// 32 products at a time and adds those blocks in the original K order.
kernel void kernel_dsv41_native_fp8_mm(
        constant uint4 &a [[buffer(0)]], device const uchar *w [[buffer(1)]],
        device const float *x [[buffer(2)]], device float *out [[buffer(3)]],
        uint2 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    const uint row = group.x, first = group.y * 4u;
    float sums[4] = {0.f, 0.f, 0.f, 0.f};
    for (uint col = lane; col < a.x; col += 32u) {
        const ulong i = ulong(row) * a.x + col;
        const float weight = a.z ? as_type<float>(uint(reinterpret_cast<device const ushort *>(w)[i]) << 16u) :
                                  reinterpret_cast<device const float *>(w)[i];
        for (uint t = 0; t < 4u; t++) {
            const float value = first + t < a.w ? x[ulong(first + t) * a.x + col] : 0.f;
            sums[t] += simd_sum(weight * value);
        }
    }
    if (!lane) for (uint t = 0; t < 4u && first + t < a.w; t++)
        out[ulong(first + t) * a.y + row] = dsv41_bf16(sums[t]);
}

// Native MXFP4 is decoded into F32 registers. In particular, the weights are
// never converted through a half-precision tile or an integer quantizer.
kernel void kernel_dsv41_native_mxfp4_mv(
        constant uint4 &a [[buffer(0)]], device const uchar *w [[buffer(1)]],
        device const float *x [[buffer(2)]], device const int *selected [[buffer(3)]],
        device float *out [[buffer(4)]], constant uint &separate_inputs [[buffer(5)]],
        constant uint &streamed [[buffer(6)]], constant ulong *addresses [[buffer(7)]],
        uint2 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    uint cols = a.x, rows = a.y, slot = group.y, row = group.x;
    if (row >= rows || slot >= a.z) return;
    uint expert = uint(selected[slot]);
    if (expert >= a.w) { if (!lane) out[slot * rows + row] = as_type<float>(0x7fc00000u); return; }
    if (streamed == 2u && addresses[slot] == 0u) return;
    if (streamed == 2u) w = reinterpret_cast<device const uchar *>(addresses[slot]);
    ulong base = (ulong(streamed == 2u ? 0u : streamed == 1u ? slot : expert) * rows + row) * (cols / 32u) * 17u;
    float sum = 0.f;
    constexpr uint magnitude[8] = {0u, 1u, 2u, 3u, 4u, 6u, 8u, 12u};
    for (uint col = lane; col < cols; col += 32u) {
        float weight;
        if (streamed == 3u) {
            const ulong i = (ulong(expert) * rows + row) * cols + col;
            weight = as_type<float>(uint(reinterpret_cast<device const ushort *>(w)[i]) << 16u);
        } else {
            ulong block = base + ulong(col / 32u) * 17u;
            uint code = (w[block + 1u + (col % 16u)] >> ((col % 32u >= 16u) ? 4u : 0u)) & 15u;
            weight = as_type<float>(dsv41_native_bits(magnitude[code & 7u], -1, w[block], (code & 8u) << 28u));
        }
        sum += simd_sum(weight * x[(separate_inputs ? slot * cols : 0u) + col]);
    }
    if (!lane) out[slot * rows + row] = dsv41_bf16(sum);
}

// Tiles list up to four token/slot pairs selecting the same expert. Results
// go back to their original slots, preserving the later expert-ID sum.
kernel void kernel_dsv41_native_mxfp4_mm(
        constant uint4 &a [[buffer(0)]], device const ushort *w [[buffer(1)]],
        device const float *x [[buffer(2)]], device const int *selected [[buffer(3)]],
        device float *out [[buffer(4)]], constant uint &separate_inputs [[buffer(5)]],
        device const int4 *tiles [[buffer(7)]],
        uint2 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    const int4 slots = tiles[group.y];
    const uint expert = uint(selected[slots[0]]), row = group.x;
    float sums[4] = {0.f, 0.f, 0.f, 0.f};
    for (uint col = lane; col < a.x; col += 32u) {
        const ulong i = (ulong(expert) * a.y + row) * a.x + col;
        const float weight = as_type<float>(uint(w[i]) << 16u);
        for (uint t = 0; t < 4u; t++) {
            const uint input = separate_inputs ? uint(max(slots[t], 0)) : uint(max(slots[t], 0)) / a.z;
            const float value = slots[t] >= 0 ? x[ulong(input) * a.x + col] : 0.f;
            sums[t] += simd_sum(weight * value);
        }
    }
    if (!lane) for (uint t = 0; t < 4u; t++)
        if (slots[t] >= 0) out[ulong(slots[t]) * a.y + row] = dsv41_bf16(sums[t]);
}

struct ds4_native_mid_args { uint width, count; float clamp; };
kernel void kernel_dsv41_native_mid(constant ds4_native_mid_args &a [[buffer(0)]],
        device const float *gate [[buffer(1)]], device const float *up [[buffer(2)]],
        device const float *weights [[buffer(3)]], device float *mid [[buffer(4)]],
        uint i [[thread_position_in_grid]]) {
    if (i >= a.width * a.count) return;
    float g = min(gate[i], a.clamp), u = clamp(up[i], -a.clamp, a.clamp);
    mid[i] = dsv41_bf16(precise::divide(g, 1.f + precise::exp(-g)) * u * weights[i / a.width]);
}

kernel void kernel_dsv41_native_sum(constant uint4 &a [[buffer(0)]],
        device const float *experts [[buffer(1)]], device const int *selected [[buffer(2)]],
        device float *out [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    if (i >= a.x * a.z) return;
    const uint token = i / a.x, column = i % a.x;
    selected += token * a.y;
    experts += ulong(token) * a.y * a.x;
    float sum = 0.f;
    uint used = 0u;
    // The upstream MoE visits experts by ID, rather than router score order.
    for (uint k = 0; k < a.y; k++) {
        int best = 0x7fffffff; uint slot = 0u;
        for (uint j = 0; j < a.y; j++)
            if (!(used & (1u << j)) && selected[j] < best) { best = selected[j]; slot = j; }
        sum += experts[slot * a.x + column]; used |= 1u << slot;
    }
    out[i] = sum;
}

// Match the released sparse_attn kernel's 64-slot online softmax, including
// invalid ring slots and the BF16 probability cast before the value product.
kernel void kernel_dsv41_native_attention(constant uint4 &a [[buffer(0)]],
        device const float *q [[buffer(1)]], device const float *raw [[buffer(2)]],
        device const float *compressed [[buffer(3)]], device const int *selected [[buffer(4)]],
        device const float *sink [[buffer(5)]], device float *out [[buffer(6)]],
        uint head [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]],
        uint sg [[simdgroup_index_in_threadgroup]], uint tid [[thread_position_in_threadgroup]]) {
    const uint pos=a.x, count=a.y, top=a.z, dim=a.w;
    const uint nraw = pos == 0u ? 1u : 128u;
    // Eight SIMD groups keep the original 32-lane dot reduction, but compute
    // different keys concurrently. Value dimensions are independent, so each
    // thread owns two of the 512 dimensions and still visits keys in order.
    threadgroup float scores[64], probabilities[64], state[3];
    threadgroup uint indices[64], valid[64];
    float acc0=0.f, acc1=0.f;
    float maximum=-1e30f, total=0.f;
    for (uint base=0;base<nraw+top;base+=64u) {
        for (uint k=sg;k<64;k+=8u) {
            uint slot=base+k, index=0u; bool good=slot<nraw+top;
            if (slot<nraw) {
                index=pos==0u ? 0u : (pos+1u+slot)%128u;
                good=good && (pos>=127u || index<=pos);
            } else if (good) {
                uint ci=count<=top ? slot-nraw : uint(selected[slot-nraw]);
                good=ci<count; index=128u+ci;
            }
            float dot=0.f;
            if (good) for (uint d=lane;d<dim;d+=32u)
                dot += q[head*dim+d] * (index<128u ? raw[index*dim+d] : compressed[(index-128u)*dim+d]);
            dot=simd_sum(dot)*0.04419417382415922f;
            if (!lane) {
                scores[k]=good ? dot : -INFINITY; indices[k]=index; valid[k]=good;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (!tid) {
            float next=maximum;
            for (uint k=0;k<64;k++) next=max(next,scores[k]);
            state[0]=precise::exp(maximum-next); state[1]=next;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid<64u) {
            float p=precise::exp(scores[tid]-state[1]);
            scores[tid]=p; probabilities[tid]=dsv41_bf16(p);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        acc0*=state[0]; acc1*=state[0];
        for (uint k=0;k<64;k++) {
            if (valid[k]) {
                uint index=indices[k]; float p=probabilities[k];
                device const float *value=index<128u ? raw+index*dim : compressed+(index-128u)*dim;
                acc0 += p*value[tid]; acc1 += p*value[tid+256u];
            }
        }
        // The release reduces each 64-slot block before its online update.
        // A sequential denominator sum can cross a BF16 rounding midpoint.
        for (uint stride=1u;stride<64u;stride*=2u) {
            const uint k=tid*2u*stride;
            if (k<64u) scores[k]+=scores[k+stride];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (!tid) { total=total*state[0]+scores[0]; maximum=state[1]; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (!tid) state[2]=total+precise::exp(sink[head]-maximum);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    out[head*dim+tid]=dsv41_bf16(precise::divide(acc0,state[2]));
    out[head*dim+tid+256u]=dsv41_bf16(precise::divide(acc1,state[2]));
}

// Correctness path: keep the release's softplus and explicit normalization
// boundaries. The expert calculation is sensitive to a last-bit weight change.
kernel void kernel_dsv41_native_router(constant uint2 &a [[buffer(0)]],
        device const float *logits [[buffer(1)]], device const float *bias [[buffer(2)]],
        device float *probs [[buffer(3)]], device int *selected [[buffer(4)]],
        device float *weights [[buffer(5)]], constant float &scale [[buffer(6)]],
        uint tid [[thread_position_in_threadgroup]], uint ntg [[threads_per_threadgroup]],
        uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float scores[1024], best_scores[8];
    threadgroup int best_ids[8];
    for (uint i=tid;i<a.x;i+=ntg) {
        float x=logits[i];
        float ex=precise::exp(x);
        float u=1.f+ex;
        float sp=x>20.f ? x : precise::log(u)-precise::divide((u-1.f)-ex,u);
        probs[i]=precise::sqrt(sp);
        scores[i]=probs[i]+bias[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    for (uint j=0;j<a.y;j++) {
        float best=-INFINITY; int bi=-1;
        for (uint i=tid;i<a.x;i+=ntg) {
            float score=scores[i];
            if (score>best) {best=score;bi=int(i);}
        }
        for (uint offset=16u;offset;offset>>=1u) {
            const float other=simd_shuffle_down(best,offset);
            const int oi=simd_shuffle_down(bi,offset);
            if (oi>=0 && (other>best || (other==best && (bi<0 || oi<bi)))) {
                best=other;bi=oi;
            }
        }
        if (!lane) {best_scores[sg]=best;best_ids[sg]=bi;}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (!tid) {
            for (uint k=1;k<8u;k++) {
                const float other=best_scores[k];const int oi=best_ids[k];
                if (oi>=0 && (other>best || (other==best && (bi<0 || oi<bi)))) {
                    best=other;bi=oi;
                }
            }
            selected[j]=bi;
            if (bi>=0) scores[bi]=-INFINITY;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid) return;
    float total=0.f;
    for (uint j=0;j<a.y;j++) total+=probs[selected[j]];
    for (uint j=0;j<a.y;j++) {
        float quotient=precise::divide(probs[selected[j]],total+1e-20f);
        weights[j]=quotient*scale;
    }
}

kernel void kernel_dsv41_native_hc(constant uint2 &a [[buffer(0)]],
        device const float *fn [[buffer(1)]], device const float *x [[buffer(2)]],
        device float *out [[buffer(3)]], constant float &eps [[buffer(4)]],
        uint2 group [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    const uint row=group.x;
    x += ulong(group.y)*a.x; out += ulong(group.y)*a.y;
    float dot=0.f, squares=0.f;
    for(uint i=lane;i<a.x;i+=32u) { dot+=fn[ulong(row)*a.x+i]*x[i];squares+=x[i]*x[i]; }
    dot=simd_sum(dot);squares=simd_sum(squares);
    if(!lane) out[row]=dot*precise::rsqrt(squares/float(a.x)+eps);
}

// Keep softmax normalization before the two weighted products, as in the
// source Compressor. Dividing their combined numerator changes BF16 ties.
kernel void kernel_dsv41_native_pool2(constant ds4_metal_args_dsv41_pool &args,
        device float *out, device const float *kv, device const float *scores,
        device const float *previous_kv, device const float *previous_scores,
        uint2 index [[thread_position_in_grid]]) {
    if(index.x>=args.width || index.y>=args.pairs)return;
    const long a=long(index.y)*2-args.tail;
    const ulong b=ulong(a+1)*args.width+index.x;
    const float ka=a<0 ? previous_kv[index.x] : kv[ulong(a)*args.width+index.x];
    const float sa=a<0 ? previous_scores[index.x] : scores[ulong(a)*args.width+index.x];
    const float sb=scores[b], peak=max(sa,sb);
    const float ea=precise::exp(sa-peak),eb=precise::exp(sb-peak),den=ea+eb;
    const float wa=precise::divide(ea,den),wb=precise::divide(eb,den);
    const float first=ka*wa,second=kv[b]*wb;
    out[ulong(index.y)*args.width+index.x]=dsv41_bf16(first+second);
}

// Native HC keeps the release's individual division boundaries. Reciprocal
// multiplication in the optimized HC kernel changes the cached BF16 input.
kernel void kernel_dsv41_native_hc_input(
        constant ds4_metal_args_dsv4_hc_split_weighted_sum_norm &a [[buffer(0)]],
        device const float *mix [[buffer(1)]], device const float *scale [[buffer(2)]],
        device const float *base [[buffer(3)]], device const float *x [[buffer(4)]],
        device float *split [[buffer(5)]], device float *collapsed [[buffer(6)]],
        device const float *norm_weight [[buffer(7)]], device float *norm [[buffer(8)]],
        device const float *pre [[buffer(9)]], threadgroup float *shared [[threadgroup(0)]],
        uint tid [[thread_position_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
        uint sg [[simdgroup_index_in_threadgroup]], uint ntg [[threads_per_threadgroup]]) {
    threadgroup float *row=shared+4,*sums=row+a.n_embd;
    if(tid<32u)sums[tid]=0.f;
    if(!tid) {
        for(uint i=0;i<4;i++) {
            float p=mix[i]*scale[0]+base[i];
            float q=mix[i+4]*scale[1]+base[i+4];
            split[i]=precise::divide(1.f,1.f+precise::exp(-p))+a.eps;
            split[i+4]=2.f*precise::divide(1.f,1.f+precise::exp(-q));
        }
        float comb[16];
        for(uint i=0;i<16;i++)comb[i]=mix[i+8]*scale[2]+base[i+8];
        for(uint r=0;r<4;r++) {
            float peak=max(max(comb[r*4],comb[r*4+1]),max(comb[r*4+2],comb[r*4+3]));
            for(uint j=0;j<4;j++)comb[r*4+j]=precise::exp(comb[r*4+j]-peak);
            float sum=(comb[r*4]+comb[r*4+1])+(comb[r*4+2]+comb[r*4+3]);
            for(uint j=0;j<4;j++)comb[r*4+j]=precise::divide(comb[r*4+j],sum)+a.eps;
        }
        for(uint j=0;j<4;j++) {
            float sum=((comb[j]+comb[4+j])+comb[8+j])+comb[12+j]+a.eps;
            for(uint r=0;r<4;r++)comb[r*4+j]=precise::divide(comb[r*4+j],sum);
        }
        for(uint iter=1;iter<uint(a.sinkhorn_iters);iter++) {
            for(uint r=0;r<4;r++) {
                float sum=((comb[r*4]+comb[r*4+1])+comb[r*4+2])+comb[r*4+3]+a.eps;
                for(uint j=0;j<4;j++)comb[r*4+j]=precise::divide(comb[r*4+j],sum);
            }
            for(uint j=0;j<4;j++) {
                float sum=((comb[j]+comb[4+j])+comb[8+j])+comb[12+j]+a.eps;
                for(uint r=0;r<4;r++)comb[r*4+j]=precise::divide(comb[r*4+j],sum);
            }
        }
        for(uint i=0;i<16;i++)split[i+8]=comb[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float square=0.f;
    for(uint i=tid;i<uint(a.n_embd);i+=ntg) {
        #pragma METAL fp contract(off)
        float v=0.f;
        for(uint h=0;h<4;h++)v+=x[h*a.n_embd+i]*pre[h];
        v=dsv41_bf16(v);row[i]=v;collapsed[i]=v;square+=v*v;
    }
    square=simd_sum(square);
    if(!lane)sums[sg]=square;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    square=simd_sum(sums[lane]);
    const float rstd=precise::rsqrt(precise::divide(square,float(a.n_embd))+a.norm_eps);
    for(uint i=tid;i<uint(a.n_embd);i+=ntg)norm[i]=dsv41_bf16((row[i]*rstd)*norm_weight[i]);
}

// hc_pre materializes the four F32 products before reducing them.
kernel void kernel_dsv41_native_hc_weighted_sum(
        constant ds4_metal_args_dsv4_hc_weighted_sum &a,
        device const char *x, device const char *weights, device char *out,
        uint gid [[thread_position_in_grid]]) {
    #pragma METAL fp contract(off)
    if (long(gid) >= a.n_embd * a.n_tokens) return;
    const long d = long(gid) % a.n_embd, t = long(gid) / a.n_embd;
    float sum = 0.f;
    for (long h = 0; h < a.n_hc; h++) {
        float v = *reinterpret_cast<device const float *>(x + d*a.nb_x0 + h*a.nb_x1 + t*a.nb_x2);
        float w = *reinterpret_cast<device const float *>(weights + h*a.nb_w0 + t*a.nb_w1);
        sum += v*w;
    }
    *reinterpret_cast<device float *>(out + d*a.nb0 + t*a.nb1) = a.round_bf16 ? dsv41_bf16(sum) : sum;
}

// The release sums the residual products before adding the post-scaled block.
// Including the block in that reduction changes BF16 midpoint decisions.
kernel void kernel_dsv41_native_hc_expand4(
        constant ds4_metal_args_dsv4_hc_expand &a,
        device const char *block, device const char *residual,
        device const char *post, device const char *comb,
        device const char *add, device char *out,
        uint gid [[thread_position_in_grid]]) {
    // hc_post materializes each F32 product before its sum. Safe math still
    // permits statement-level FMA, which crosses those source boundaries.
    #pragma METAL fp contract(off)
    if (long(gid) >= a.n_embd * a.n_tokens) return;
    const long d = long(gid) % a.n_embd, t = long(gid) / a.n_embd;
    float v = *reinterpret_cast<device const float *>(block + d*a.nb_block0 + t*a.nb_block1);
    if (a.has_add) v += *reinterpret_cast<device const float *>(add + d*a.nb_add0 + t*a.nb_add1);
    if (a.round_block_in) v = dsv41_bf16(v);
    for (long dst = 0; dst < 4; dst++) {
        float sum = 0.f;
        for (long src = 0; src < 4; src++) {
            float c = *reinterpret_cast<device const float *>(comb + dst*a.nb_comb0 + src*a.nb_comb1 + t*a.nb_comb2);
            float r = *reinterpret_cast<device const float *>(residual + d*a.nb_res0 + src*a.nb_res1 + t*a.nb_res2);
            sum += c*r;
        }
        float p = *reinterpret_cast<device const float *>(post + dst*a.nb_post0 + t*a.nb_post1);
        float result = v*p + sum;
        *reinterpret_cast<device float *>(out + d*a.nb0 + dst*a.nb1 + t*a.nb2) = a.round_bf16 ? dsv41_bf16(result) : result;
    }
}
