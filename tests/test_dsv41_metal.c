/* DeepSeek V4.1 Metal kernels vs the CPU reference.
 *
 * The engram block is the first piece on the GPU: wkv through the shared BF16
 * mat-vec, then kernel_dsv41_engram_gate.  The CPU side here is an independent
 * transcription of the same maths, so a match is a real cross-check rather than
 * two calls into one implementation.
 */

#define _DARWIN_C_SOURCE

#include "ds4_gpu.h"

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <stdlib.h>

#define HC  4u
#define EPS 1.0e-20f

bool ds4_log_is_tty(FILE *fp) { (void)fp; return false; }

static uint32_t rng_state = 0x12345678u;
static float rnd(void) {
    rng_state = rng_state * 1664525u + 1013904223u;
    return (float)((int32_t)((rng_state >> 8) % 2001u) - 1000) / 1000.0f;
}

static uint16_t f32_to_bf16(float v) {
    uint32_t bits;
    memcpy(&bits, &v, sizeof(bits));
    /* round-to-nearest-even, so the CPU side sees exactly what the GPU reads */
    const uint32_t rounded = bits + 0x7fffu + ((bits >> 16) & 1u);
    return (uint16_t)(rounded >> 16);
}

static float bf16_to_f32(uint16_t v) {
    const uint32_t bits = (uint32_t)v << 16;
    float out;
    memcpy(&out, &bits, sizeof(out));
    return out;
}

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec / 1e6;
}

/* `label` names the shape: the mini model the CPU reference is scored on, and the
 * released dimensions the kernel actually has to run at. */
static int run_shape(const char *label, uint32_t DIM, uint32_t BUCKET, uint32_t HEADD) {
    const uint32_t IN_DIM = BUCKET * HEADD;
    const uint32_t OUT_DIM = DIM * (HC + 1u);

    float *emb = malloc((size_t)IN_DIM * sizeof(float));
    float *x = malloc((size_t)HC * DIM * sizeof(float));
    float *qw = malloc((size_t)HC * DIM * sizeof(float));
    float *kw = malloc((size_t)HC * DIM * sizeof(float));
    float *kv_ref = malloc((size_t)OUT_DIM * sizeof(float));
    float *out_ref = malloc((size_t)HC * DIM * sizeof(float));
    float *out_gpu = malloc((size_t)HC * DIM * sizeof(float));
    float gate_ref[HC], gate_gpu[HC];
    if (!emb || !x || !qw || !kw || !kv_ref || !out_ref || !out_gpu) return 1;

    /* the model map has to be page aligned, same as a real mapped GGUF */
    const size_t wkv_bytes = (size_t)OUT_DIM * IN_DIM * sizeof(uint16_t);
    void *blob = NULL;
    if (posix_memalign(&blob, (size_t)getpagesize(), wkv_bytes) != 0 || !blob) {
        printf("dsv41 metal: weight allocation failed\n");
        return 1;
    }
    uint16_t *wkv = (uint16_t *)blob;
    for (size_t i = 0; i < (size_t)OUT_DIM * IN_DIM; i++) wkv[i] = f32_to_bf16(rnd() * 0.05f);
    for (size_t i = 0; i < IN_DIM; i++) emb[i] = rnd();
    for (size_t i = 0; i < HC * DIM; i++) { x[i] = rnd(); qw[i] = rnd(); kw[i] = rnd(); }

    /* --- CPU reference --------------------------------------------------- */
    for (uint32_t o = 0; o < OUT_DIM; o++) {
        double acc = 0.0;
        const uint16_t *row = wkv + (size_t)o * IN_DIM;
        for (uint32_t i = 0; i < IN_DIM; i++) acc += (double)bf16_to_f32(row[i]) * emb[i];
        kv_ref[o] = (float)acc;
    }
    const float *val = kv_ref + (size_t)HC * DIM;
    for (uint32_t c = 0; c < HC; c++) {
        const float *xc = x + (size_t)c * DIM, *kc = kv_ref + (size_t)c * DIM;
        const float *qc = qw + (size_t)c * DIM, *wc = kw + (size_t)c * DIM;
        double xs = 0.0, ks = 0.0, dot = 0.0;
        for (uint32_t j = 0; j < DIM; j++) {
            xs += (double)xc[j] * xc[j];
            ks += (double)kc[j] * kc[j];
            dot += (double)xc[j] * qc[j] * wc[j] * kc[j];
        }
        const float rstd = (float)(1.0 / sqrt(xs / DIM + EPS)) * (float)(1.0 / sqrt(ks / DIM + EPS));
        float v = (float)dot * rstd / sqrtf((float)DIM);
        float a = fabsf(v);
        if (a < 1e-6f) a = 1e-6f;
        const float s = copysignf(sqrtf(a), v);
        const float g = 1.0f / (1.0f + expf(-s));
        gate_ref[c] = g;
        for (uint32_t j = 0; j < DIM; j++) out_ref[c * DIM + j] = xc[j] + g * val[j];
    }

    /* --- GPU ------------------------------------------------------------- */
    ds4_gpu_tensor *t_emb = ds4_gpu_tensor_alloc((uint64_t)IN_DIM * sizeof(float));
    ds4_gpu_tensor *t_kv = ds4_gpu_tensor_alloc((uint64_t)OUT_DIM * sizeof(float));
    ds4_gpu_tensor *t_x = ds4_gpu_tensor_alloc((uint64_t)HC * DIM * sizeof(float));
    ds4_gpu_tensor *t_qw = ds4_gpu_tensor_alloc((uint64_t)HC * DIM * sizeof(float));
    ds4_gpu_tensor *t_kw = ds4_gpu_tensor_alloc((uint64_t)HC * DIM * sizeof(float));
    ds4_gpu_tensor *t_gate = ds4_gpu_tensor_alloc(sizeof(gate_ref));
    ds4_gpu_tensor *t_out = ds4_gpu_tensor_alloc((uint64_t)HC * DIM * sizeof(float));
    int ok = t_emb && t_kv && t_x && t_qw && t_kw && t_gate && t_out;
    ok = ok && ds4_gpu_tensor_write(t_emb, 0, emb, (uint64_t)IN_DIM * sizeof(float));
    ok = ok && ds4_gpu_tensor_write(t_x, 0, x, (uint64_t)HC * DIM * sizeof(float));
    ok = ok && ds4_gpu_tensor_write(t_qw, 0, qw, (uint64_t)HC * DIM * sizeof(float));
    ok = ok && ds4_gpu_tensor_write(t_kw, 0, kw, (uint64_t)HC * DIM * sizeof(float));
    if (!ok) { printf("dsv41 metal: buffer setup failed\n"); return 1; }

    /* the weights are read through the model-map path, as a bound tensor would be */
    if (!ds4_gpu_set_model_map(wkv, wkv_bytes)) {
        printf("dsv41 metal: could not map the weight blob\n");
        return 1;
    }
    if (!ds4_gpu_dsv41_engram(wkv, wkv_bytes, 0, IN_DIM, DIM, HC, EPS, 1u, NULL,
                              t_emb, t_kv, t_x, t_qw, t_kw, t_gate, t_out, NULL)) {
        printf("dsv41 metal: engram dispatch failed\n");
        return 1;
    }
    const double t0 = now_ms();
    const int reps = 20;
    for (int r = 0; r < reps; r++) {
        if (!ds4_gpu_dsv41_engram(wkv, wkv_bytes, 0, IN_DIM, DIM, HC, EPS, 1u, NULL,
                                  t_emb, t_kv, t_x, t_qw, t_kw, t_gate, t_out, NULL)) return 1;
    }
    const double per_call = (now_ms() - t0) / reps;
    ok = ds4_gpu_tensor_read(t_gate, 0, gate_gpu, sizeof(gate_gpu));
    ok = ok && ds4_gpu_tensor_read(t_out, 0, out_gpu, (uint64_t)HC * DIM * sizeof(float));
    if (!ok) { printf("dsv41 metal: readback failed\n"); return 1; }

    double gd = 0.0, od = 0.0, oscale = 0.0;
    for (uint32_t c = 0; c < HC; c++) {
        const double d = fabs((double)gate_gpu[c] - gate_ref[c]);
        if (d > gd) gd = d;
    }
    for (uint32_t i = 0; i < HC * DIM; i++) {
        const double d = fabs((double)out_gpu[i] - out_ref[i]);
        if (d > od) od = d;
        if (fabs((double)out_ref[i]) > oscale) oscale = fabs((double)out_ref[i]);
    }
    const double orel = od / (oscale > 0.0 ? oscale : 1.0);

    /* f32 accumulation on the GPU against f64 on the CPU over 6144 and 256 terms */
    const int gate_ok = gd <= 2e-5;
    const int out_ok = orel <= 2e-5;
    printf("  [%s] %-18s %s  max|d|=%.3e   gates %.6f %.6f %.6f %.6f\n", label, "engram gate",
           gate_ok ? "ok  " : "FAIL", gd,
           (double)gate_gpu[0], (double)gate_gpu[1], (double)gate_gpu[2], (double)gate_gpu[3]);
    printf("  [%s] %-18s %s  max|d|=%.3e  rel=%.3e\n", label, "engram output",
           out_ok ? "ok  " : "FAIL", od, orel);

    /* a dead token shuts the gate and the stream passes through untouched */
    int dead_ok = 0;
    const int32_t dead_one = 1;
    ds4_gpu_tensor *t_dead = ds4_gpu_tensor_alloc(sizeof(int32_t));
    if (t_dead && ds4_gpu_tensor_write(t_dead, 0, &dead_one, sizeof(dead_one)) &&
        ds4_gpu_dsv41_engram(wkv, wkv_bytes, 0, IN_DIM, DIM, HC, EPS, 1u, t_dead,
                             t_emb, t_kv, t_x, t_qw, t_kw, t_gate, t_out, NULL) &&
        ds4_gpu_tensor_read(t_gate, 0, gate_gpu, sizeof(gate_gpu)) &&
        ds4_gpu_tensor_read(t_out, 0, out_gpu, (uint64_t)HC * DIM * sizeof(float))) {
        dead_ok = 1;
        for (uint32_t c = 0; c < HC; c++) if (gate_gpu[c] != 0.0f) dead_ok = 0;
        for (uint32_t i = 0; i < HC * DIM; i++) if (out_gpu[i] != x[i]) dead_ok = 0;
    }
    ds4_gpu_tensor_free(t_dead);
    printf("  [%s] %-18s %s  gate 0, stream unchanged\n", label, "engram dead token",
           dead_ok ? "ok  " : "FAIL");

    /* several rows at once against one-row calls on each row's inputs */
    int rows_pass = 0;
    {
        const uint32_t R = 5u;
        float *emb_r = malloc((size_t)R * IN_DIM * sizeof(float));
        float *x_r = malloc((size_t)R * HC * DIM * sizeof(float));
        for (uint32_t r = 0; r < R; r++) {
            for (uint32_t i = 0; i < IN_DIM; i++) emb_r[(size_t)r * IN_DIM + i] = emb[i] * (1.0f + 0.1f * r);
            for (uint32_t i = 0; i < HC * DIM; i++) x_r[(size_t)r * HC * DIM + i] = x[i] * (1.0f - 0.05f * r);
        }
        ds4_gpu_tensor *tr_emb = ds4_gpu_tensor_alloc((uint64_t)R * IN_DIM * sizeof(float));
        ds4_gpu_tensor *tr_kv = ds4_gpu_tensor_alloc((uint64_t)R * OUT_DIM * sizeof(float));
        ds4_gpu_tensor *tr_x = ds4_gpu_tensor_alloc((uint64_t)R * HC * DIM * sizeof(float));
        ds4_gpu_tensor *tr_gate = ds4_gpu_tensor_alloc((uint64_t)R * HC * sizeof(float));
        ds4_gpu_tensor *tr_out = ds4_gpu_tensor_alloc((uint64_t)R * HC * DIM * sizeof(float));
        float *out_r = malloc((size_t)R * HC * DIM * sizeof(float)), *gate_r = malloc((size_t)R * HC * sizeof(float));
        float *out_1 = malloc((size_t)HC * DIM * sizeof(float)), gate_1[HC];
        int rows_ok = tr_emb && tr_kv && tr_x && tr_gate && tr_out &&
            ds4_gpu_tensor_write(tr_emb, 0, emb_r, (uint64_t)R * IN_DIM * sizeof(float)) &&
            ds4_gpu_tensor_write(tr_x, 0, x_r, (uint64_t)R * HC * DIM * sizeof(float)) &&
            ds4_gpu_dsv41_engram(wkv, wkv_bytes, 0, IN_DIM, DIM, HC, EPS, R, NULL,
                                 tr_emb, tr_kv, tr_x, t_qw, t_kw, tr_gate, tr_out, NULL) &&
            ds4_gpu_tensor_read(tr_gate, 0, gate_r, (uint64_t)R * HC * sizeof(float)) &&
            ds4_gpu_tensor_read(tr_out, 0, out_r, (uint64_t)R * HC * DIM * sizeof(float));
        double rd = 0.0;
        for (uint32_t r = 0; rows_ok && r < R; r++) {
            rows_ok = ds4_gpu_tensor_write(t_emb, 0, emb_r + (size_t)r * IN_DIM, (uint64_t)IN_DIM * sizeof(float)) &&
                      ds4_gpu_tensor_write(t_x, 0, x_r + (size_t)r * HC * DIM, (uint64_t)HC * DIM * sizeof(float)) &&
                      ds4_gpu_dsv41_engram(wkv, wkv_bytes, 0, IN_DIM, DIM, HC, EPS, 1u, NULL,
                                           t_emb, t_kv, t_x, t_qw, t_kw, t_gate, t_out, NULL) &&
                      ds4_gpu_tensor_read(t_gate, 0, gate_1, sizeof(gate_1)) &&
                      ds4_gpu_tensor_read(t_out, 0, out_1, (uint64_t)HC * DIM * sizeof(float));
            for (uint32_t c = 0; rows_ok && c < HC; c++) rd = fmax(rd, fabs((double)gate_r[(size_t)r * HC + c] - gate_1[c]));
            for (uint32_t i = 0; rows_ok && i < HC * DIM; i++)
                rd = fmax(rd, fabs((double)out_r[(size_t)r * HC * DIM + i] - out_1[i]) / (oscale > 0.0 ? oscale : 1.0));
        }
        const double tr0 = now_ms();
        for (int r = 0; rows_ok && r < reps; r++) {
            rows_ok = ds4_gpu_dsv41_engram(wkv, wkv_bytes, 0, IN_DIM, DIM, HC, EPS, R, NULL,
                                           tr_emb, tr_kv, tr_x, t_qw, t_kw, tr_gate, tr_out, NULL);
        }
        const double per_rows = (now_ms() - tr0) / reps;
        const int pass = rows_ok && rd <= 1e-5;
        printf("  [%s] %-18s %s  max rel %.3e vs one-row calls  (%u rows %.3f ms/call, 1 row %.3f)\n",
               label, "engram rows", pass ? "ok  " : "FAIL", rd, R, per_rows, per_call);
        rows_pass = pass;
        (void)ds4_gpu_tensor_write(t_emb, 0, emb, (uint64_t)IN_DIM * sizeof(float));
        (void)ds4_gpu_tensor_write(t_x, 0, x, (uint64_t)HC * DIM * sizeof(float));
        ds4_gpu_tensor_free(tr_emb); ds4_gpu_tensor_free(tr_kv); ds4_gpu_tensor_free(tr_x);
        ds4_gpu_tensor_free(tr_gate); ds4_gpu_tensor_free(tr_out);
        free(emb_r); free(x_r); free(out_r); free(gate_r); free(out_1);
    }

    /* a chunk: three tokens, the middle one dead.  Row 0 and row 2 carry the single-token
     * input, so both must reproduce the single-token result and row 1 must pass through. */
    int batch_ok = 0;
    {
        const uint32_t R = 3u;
        const int32_t dead3[3] = { 0, 1, 0 };
        float *emb3 = malloc((size_t)R * IN_DIM * sizeof(float));
        float *x3 = malloc((size_t)R * HC * DIM * sizeof(float));
        float *out3 = malloc((size_t)R * HC * DIM * sizeof(float));
        float gate3[3 * HC];
        for (uint32_t r = 0; r < R; r++) {
            memcpy(emb3 + (size_t)r * IN_DIM, emb, (size_t)IN_DIM * sizeof(float));
            memcpy(x3 + (size_t)r * HC * DIM, x, (size_t)HC * DIM * sizeof(float));
        }
        ds4_gpu_tensor *t_emb3 = ds4_gpu_tensor_alloc((uint64_t)R * IN_DIM * sizeof(float));
        ds4_gpu_tensor *t_kv3 = ds4_gpu_tensor_alloc((uint64_t)R * OUT_DIM * sizeof(float));
        ds4_gpu_tensor *t_x3 = ds4_gpu_tensor_alloc((uint64_t)R * HC * DIM * sizeof(float));
        ds4_gpu_tensor *t_gate3 = ds4_gpu_tensor_alloc(sizeof(gate3));
        ds4_gpu_tensor *t_dead3 = ds4_gpu_tensor_alloc(sizeof(dead3));
        int bok = emb3 && x3 && out3 && t_emb3 && t_kv3 && t_x3 && t_gate3 && t_dead3;
        bok = bok && ds4_gpu_tensor_write(t_emb3, 0, emb3, (uint64_t)R * IN_DIM * sizeof(float));
        bok = bok && ds4_gpu_tensor_write(t_x3, 0, x3, (uint64_t)R * HC * DIM * sizeof(float));
        bok = bok && ds4_gpu_tensor_write(t_dead3, 0, dead3, sizeof(dead3));
        /* in place, as the engine runs it: x and out are the same tensor */
        bok = bok && ds4_gpu_dsv41_engram(wkv, wkv_bytes, 0, IN_DIM, DIM, HC, EPS, R, t_dead3,
                                          t_emb3, t_kv3, t_x3, t_qw, t_kw, t_gate3, t_x3, NULL);
        bok = bok && ds4_gpu_tensor_read(t_gate3, 0, gate3, sizeof(gate3));
        bok = bok && ds4_gpu_tensor_read(t_x3, 0, out3, (uint64_t)R * HC * DIM * sizeof(float));
        if (bok) {
            batch_ok = 1;
            for (uint32_t r = 0; r < R; r += 2) {
                for (uint32_t c = 0; c < HC; c++)
                    if (fabs((double)gate3[r * HC + c] - gate_ref[c]) > 2e-5) batch_ok = 0;
                for (uint32_t i = 0; i < HC * DIM; i++)
                    if (fabs((double)out3[(size_t)r * HC * DIM + i] - out_ref[i]) > 2e-5 * oscale)
                        batch_ok = 0;
            }
            for (uint32_t c = 0; c < HC; c++) if (gate3[HC + c] != 0.0f) batch_ok = 0;
            for (uint32_t i = 0; i < HC * DIM; i++)
                if (out3[(size_t)HC * DIM + i] != x[i]) batch_ok = 0;
        }
        ds4_gpu_tensor_free(t_emb3); ds4_gpu_tensor_free(t_kv3); ds4_gpu_tensor_free(t_x3);
        ds4_gpu_tensor_free(t_gate3); ds4_gpu_tensor_free(t_dead3);
        free(emb3); free(x3); free(out3);
    }
    printf("  [%s] %-18s %s  3 rows in place, dead row passes through\n", label,
           "engram batched", batch_ok ? "ok  " : "FAIL");
    printf("  [%s] %-18s %.3f ms/call (%u x %u mat-vec + gate)\n", label, "engram timing",
           per_call, OUT_DIM, IN_DIM);

    ds4_gpu_tensor_free(t_emb); ds4_gpu_tensor_free(t_kv); ds4_gpu_tensor_free(t_x);
    ds4_gpu_tensor_free(t_qw); ds4_gpu_tensor_free(t_kw);
    ds4_gpu_tensor_free(t_gate); ds4_gpu_tensor_free(t_out);

    free(emb); free(x); free(qw); free(kw); free(kv_ref); free(out_ref); free(out_gpu);
    free(blob);
    return !(gate_ok && out_ok && dead_ok && batch_ok && rows_pass);
}

/* Sparse attention, transcribed independently from the released implementation:
 * the max runs over reachable candidates only, the running max is seeded at a finite
 * -1e30 so an all-invalid row yields zeros rather than NaN, and the per-head sink is a
 * logit added to the denominator after the weighted sum. */
static int run_attn(const char *label, uint32_t s_len, uint32_t n_head,
                    uint32_t d, uint32_t topk, uint32_t n_kv, float invalid_frac) {
    const size_t qn = (size_t)s_len * n_head * d;
    float *q = malloc(qn * sizeof(float));
    float *kv = malloc((size_t)n_kv * d * sizeof(float));
    float *sink = malloc((size_t)n_head * sizeof(float));
    int32_t *idxs = malloc((size_t)s_len * topk * sizeof(int32_t));
    float *ref = malloc(qn * sizeof(float));
    float *gpu = malloc(qn * sizeof(float));
    float *sc = malloc((size_t)topk * sizeof(float));
    if (!q || !kv || !sink || !idxs || !ref || !gpu || !sc) return 1;

    for (size_t i = 0; i < qn; i++) q[i] = rnd();
    for (size_t i = 0; i < (size_t)n_kv * d; i++) kv[i] = rnd();
    for (uint32_t i = 0; i < n_head; i++) sink[i] = rnd();
    for (uint32_t si = 0; si < s_len; si++) {
        for (uint32_t t = 0; t < topk; t++) {
            const float r = (rnd() + 1.0f) * 0.5f;
            idxs[(size_t)si * topk + t] =
                r < invalid_frac ? -1 : (int32_t)(((uint32_t)(r * 100000.0f)) % n_kv);
        }
    }
    /* one row with nothing reachable at all, the case that must not produce NaN */
    if (s_len > 1) for (uint32_t t = 0; t < topk; t++) idxs[(size_t)(s_len - 1) * topk + t] = -1;

    const float scale = 1.0f / sqrtf((float)d);
    for (uint32_t si = 0; si < s_len; si++) {
        const int32_t *idx = idxs + (size_t)si * topk;
        for (uint32_t hi = 0; hi < n_head; hi++) {
            const float *qv = q + ((size_t)si * n_head + hi) * d;
            float mx = -1.0e30f;
            for (uint32_t t = 0; t < topk; t++) {
                if (idx[t] < 0) { sc[t] = -1.0e30f; continue; }
                const float *kvv = kv + (size_t)idx[t] * d;
                double acc = 0.0;
                for (uint32_t j = 0; j < d; j++) acc += (double)qv[j] * kvv[j];
                sc[t] = (float)acc * scale;
                if (sc[t] > mx) mx = sc[t];
            }
            double den = 0.0;
            float *o = ref + ((size_t)si * n_head + hi) * d;
            for (uint32_t j = 0; j < d; j++) o[j] = 0.0f;
            uint32_t nvalid = 0;
            for (uint32_t t = 0; t < topk; t++) {
                if (idx[t] < 0) continue;
                const float e = expf(sc[t] - mx);
                den += e;
                nvalid++;
                const float *kvv = kv + (size_t)idx[t] * d;
                for (uint32_t j = 0; j < d; j++) o[j] += e * kvv[j];
            }
            if (nvalid == 0) continue;
            den += expf(sink[hi] - mx);
            for (uint32_t j = 0; j < d; j++) o[j] = (float)(o[j] / den);
        }
    }

    ds4_gpu_tensor *t_q = ds4_gpu_tensor_alloc(qn * sizeof(float));
    ds4_gpu_tensor *t_kv = ds4_gpu_tensor_alloc((uint64_t)n_kv * d * sizeof(float));
    ds4_gpu_tensor *t_sink = ds4_gpu_tensor_alloc((uint64_t)n_head * sizeof(float));
    ds4_gpu_tensor *t_idx = ds4_gpu_tensor_alloc((uint64_t)s_len * topk * sizeof(int32_t));
    ds4_gpu_tensor *t_out = ds4_gpu_tensor_alloc(qn * sizeof(float));
    int ok = t_q && t_kv && t_sink && t_idx && t_out;
    ok = ok && ds4_gpu_tensor_write(t_q, 0, q, qn * sizeof(float));
    ok = ok && ds4_gpu_tensor_write(t_kv, 0, kv, (uint64_t)n_kv * d * sizeof(float));
    ok = ok && ds4_gpu_tensor_write(t_sink, 0, sink, (uint64_t)n_head * sizeof(float));
    ok = ok && ds4_gpu_tensor_write(t_idx, 0, idxs, (uint64_t)s_len * topk * sizeof(int32_t));
    ok = ok && ds4_gpu_dsv41_sparse_attn(s_len, n_head, d, topk, n_kv, scale,
                                         t_q, t_kv, NULL, t_sink, t_idx, topk, NULL, t_out);
    ok = ok && ds4_gpu_tensor_read(t_out, 0, gpu, qn * sizeof(float));
    if (!ok) { printf("  [%s] sparse attn dispatch failed\n", label); return 1; }

    /* the same candidates split across two arrays, as the engine hands them in: the
     * first `split` of each row from one, the rest from another */
    int split_ok = 0;
    {
        const uint32_t split = topk / 3u;
        int32_t *ia = malloc((size_t)s_len * split * sizeof(int32_t));
        int32_t *ib = malloc((size_t)s_len * (topk - split) * sizeof(int32_t));
        float *g2 = malloc(qn * sizeof(float));
        for (uint32_t si = 0; si < s_len; si++) {
            memcpy(ia + (size_t)si * split, idxs + (size_t)si * topk, split * sizeof(int32_t));
            memcpy(ib + (size_t)si * (topk - split), idxs + (size_t)si * topk + split,
                   (topk - split) * sizeof(int32_t));
        }
        ds4_gpu_tensor *t_ia = ds4_gpu_tensor_alloc((uint64_t)s_len * split * sizeof(int32_t));
        ds4_gpu_tensor *t_ib = ds4_gpu_tensor_alloc((uint64_t)s_len * (topk - split) * sizeof(int32_t));
        int sok = ia && ib && g2 && t_ia && t_ib;
        sok = sok && ds4_gpu_tensor_write(t_ia, 0, ia, (uint64_t)s_len * split * sizeof(int32_t));
        sok = sok && ds4_gpu_tensor_write(t_ib, 0, ib, (uint64_t)s_len * (topk - split) * sizeof(int32_t));
        sok = sok && ds4_gpu_dsv41_sparse_attn(s_len, n_head, d, topk, n_kv, scale,
                                               t_q, t_kv, NULL, t_sink, t_ia, split, t_ib, t_out);
        sok = sok && ds4_gpu_tensor_read(t_out, 0, g2, qn * sizeof(float));
        if (sok) {
            split_ok = 1;
            for (size_t i = 0; i < qn; i++) if (g2[i] != gpu[i]) split_ok = 0;
        }
        ds4_gpu_tensor_free(t_ia); ds4_gpu_tensor_free(t_ib);
        free(ia); free(ib); free(g2);
    }

    /* batched the way the engine runs it: a per-call command buffer would time the
     * commit-and-wait round trip rather than the kernel */
    const int batched = ds4_gpu_begin_commands() != 0;
    const double t0 = now_ms();
    for (int r = 0; r < 10; r++) {
        ds4_gpu_dsv41_sparse_attn(s_len, n_head, d, topk, n_kv, scale, t_q, t_kv, NULL,
                                  t_sink, t_idx, topk, NULL, t_out);
    }
    if (batched) ds4_gpu_end_commands();
    const double per_call = (now_ms() - t0) / 10.0;

    double worst = 0.0, scale_out = 0.0;
    int finite = 1;
    for (size_t i = 0; i < qn; i++) {
        if (!isfinite(gpu[i])) finite = 0;
        const double dd = fabs((double)gpu[i] - ref[i]);
        if (dd > worst) worst = dd;
        if (fabs((double)ref[i]) > scale_out) scale_out = fabs((double)ref[i]);
    }
    const double rel = worst / (scale_out > 0.0 ? scale_out : 1.0);
    int zeros_ok = 1;
    if (s_len > 1) {
        for (uint32_t hi = 0; hi < n_head; hi++)
            for (uint32_t j = 0; j < d; j++)
                if (gpu[((size_t)(s_len - 1) * n_head + hi) * d + j] != 0.0f) zeros_ok = 0;
    }
    const int pass = finite && zeros_ok && split_ok && rel <= 3e-5;
    printf("  [%s] %-18s %s  max|d|=%.3e  rel=%.3e  finite=%d  empty-row zeros=%d  "
           "two-list=%d\n",
           label, "sparse attn", pass ? "ok  " : "FAIL", worst, rel, finite, zeros_ok, split_ok);
    printf("  [%s] %-18s %.3f ms/call (%u queries x %u heads x %u cand x %u dim)\n",
           label, "sparse attn timing", per_call, s_len, n_head, topk, d);

    ds4_gpu_tensor_free(t_q); ds4_gpu_tensor_free(t_kv); ds4_gpu_tensor_free(t_sink);
    ds4_gpu_tensor_free(t_idx); ds4_gpu_tensor_free(t_out);
    free(q); free(kv); free(sink); free(idxs); free(ref); free(gpu); free(sc);
    return !pass;
}

/* Compressor pooling: the softmax is taken PER CHANNEL across the group, then the
 * result is RMSNormed.  Transcribed independently from the released implementation. */
static int run_pool(const char *label, uint32_t groups, uint32_t ratio, uint32_t d) {
    const size_t n = (size_t)groups * ratio * d;
    float *kv = malloc(n * sizeof(float));
    float *score = malloc(n * sizeof(float));
    float *nw = malloc((size_t)d * sizeof(float));
    float *ref = malloc((size_t)groups * d * sizeof(float));
    float *gpu = malloc((size_t)groups * d * sizeof(float));
    if (!kv || !score || !nw || !ref || !gpu) return 1;
    for (size_t i = 0; i < n; i++) { kv[i] = rnd(); score[i] = rnd() * 3.0f; }
    for (uint32_t i = 0; i < d; i++) nw[i] = 1.0f + 0.1f * rnd();

    for (uint32_t g = 0; g < groups; g++) {
        double ss = 0.0;
        float *pooled = malloc((size_t)d * sizeof(float));
        for (uint32_t j = 0; j < d; j++) {
            float mx = -1.0e30f;
            for (uint32_t r = 0; r < ratio; r++) {
                const float v = score[((size_t)g * ratio + r) * d + j];
                if (v > mx) mx = v;
            }
            double sum = 0.0, acc = 0.0;
            for (uint32_t r = 0; r < ratio; r++) {
                const float e = expf(score[((size_t)g * ratio + r) * d + j] - mx);
                sum += e;
                acc += (double)e * kv[((size_t)g * ratio + r) * d + j];
            }
            pooled[j] = (float)(acc / sum);
            ss += (double)pooled[j] * pooled[j];
        }
        const float rstd = (float)(1.0 / sqrt(ss / d + EPS));
        for (uint32_t j = 0; j < d; j++) ref[(size_t)g * d + j] = pooled[j] * rstd * nw[j];
        free(pooled);
    }

    ds4_gpu_tensor *t_kv = ds4_gpu_tensor_alloc(n * sizeof(float));
    ds4_gpu_tensor *t_sc = ds4_gpu_tensor_alloc(n * sizeof(float));
    ds4_gpu_tensor *t_nw = ds4_gpu_tensor_alloc((uint64_t)d * sizeof(float));
    ds4_gpu_tensor *t_out = ds4_gpu_tensor_alloc((uint64_t)groups * d * sizeof(float));
    int ok = t_kv && t_sc && t_nw && t_out;
    ok = ok && ds4_gpu_tensor_write(t_kv, 0, kv, n * sizeof(float));
    ok = ok && ds4_gpu_tensor_write(t_sc, 0, score, n * sizeof(float));
    ok = ok && ds4_gpu_tensor_write(t_nw, 0, nw, (uint64_t)d * sizeof(float));
    ok = ok && ds4_gpu_dsv41_compress_pool(groups, ratio, d, EPS, t_kv, t_sc, t_nw, t_out);
    ok = ok && ds4_gpu_tensor_read(t_out, 0, gpu, (uint64_t)groups * d * sizeof(float));
    if (!ok) { printf("  [%s] compressor dispatch failed\n", label); return 1; }

    double worst = 0.0, sc_out = 0.0;
    for (size_t i = 0; i < (size_t)groups * d; i++) {
        const double dd = fabs((double)gpu[i] - ref[i]);
        if (dd > worst) worst = dd;
        if (fabs((double)ref[i]) > sc_out) sc_out = fabs((double)ref[i]);
    }
    const double rel = worst / (sc_out > 0.0 ? sc_out : 1.0);
    const int pass = rel <= 3e-5;
    printf("  [%s] %-18s %s  max|d|=%.3e  rel=%.3e  (%u groups x ratio %u x %u)\n",
           label, "compress pool", pass ? "ok  " : "FAIL", worst, rel, groups, ratio, d);
    ds4_gpu_tensor_free(t_kv); ds4_gpu_tensor_free(t_sc);
    ds4_gpu_tensor_free(t_nw); ds4_gpu_tensor_free(t_out);
    free(kv); free(score); free(nw); free(ref); free(gpu);
    return !pass;
}

/* Indexer scores: rectified per head BEFORE the head weight, which may be negative. */
static int run_index(const char *label, uint32_t ih, uint32_t idim, uint32_t clen,
                     uint32_t cand_n) {
    float *q = malloc((size_t)ih * idim * sizeof(float));
    float *k = malloc((size_t)clen * idim * sizeof(float));
    float *w = malloc((size_t)ih * sizeof(float));
    float *ref = malloc((size_t)clen * sizeof(float));
    float *gpu = malloc((size_t)clen * sizeof(float));
    if (!q || !k || !w || !ref || !gpu) return 1;
    for (size_t i = 0; i < (size_t)ih * idim; i++) q[i] = rnd();
    for (size_t i = 0; i < (size_t)clen * idim; i++) k[i] = rnd();
    for (uint32_t i = 0; i < ih; i++) w[i] = rnd();

    for (uint32_t j = 0; j < clen; j++) {
        double acc = 0.0;
        for (uint32_t h = 0; h < ih; h++) {
            double dot = 0.0;
            for (uint32_t i = 0; i < idim; i++)
                dot += (double)q[(size_t)h * idim + i] * k[(size_t)j * idim + i];
            if (dot > 0.0) acc += dot * (double)w[h];
        }
        ref[j] = (float)acc;
    }

    ds4_gpu_tensor *t_q = ds4_gpu_tensor_alloc((uint64_t)ih * idim * sizeof(float));
    ds4_gpu_tensor *t_k = ds4_gpu_tensor_alloc((uint64_t)clen * idim * sizeof(float));
    ds4_gpu_tensor *t_w = ds4_gpu_tensor_alloc((uint64_t)ih * sizeof(float));
    ds4_gpu_tensor *t_s = ds4_gpu_tensor_alloc((uint64_t)clen * sizeof(float));
    int ok = t_q && t_k && t_w && t_s;
    ok = ok && ds4_gpu_tensor_write(t_q, 0, q, (uint64_t)ih * idim * sizeof(float));
    ok = ok && ds4_gpu_tensor_write(t_k, 0, k, (uint64_t)clen * idim * sizeof(float));
    ok = ok && ds4_gpu_tensor_write(t_w, 0, w, (uint64_t)ih * sizeof(float));
    ok = ok && ds4_gpu_dsv41_index_score(ih, idim, clen, t_q, t_k, t_w, NULL, 1.0f, t_s);
    ok = ok && ds4_gpu_tensor_read(t_s, 0, gpu, (uint64_t)clen * sizeof(float));
    if (!ok) { printf("  [%s] indexer dispatch failed\n", label); return 1; }

    const double t0 = now_ms();
    for (int r = 0; r < 10; r++) ds4_gpu_dsv41_index_score(ih, idim, clen, t_q, t_k, t_w, NULL, 1.0f, t_s);
    const double per_call = (now_ms() - t0) / 10.0;

    double worst = 0.0, sc_out = 0.0;
    for (uint32_t j = 0; j < clen; j++) {
        const double dd = fabs((double)gpu[j] - ref[j]);
        if (dd > worst) worst = dd;
        if (fabs((double)ref[j]) > sc_out) sc_out = fabs((double)ref[j]);
    }
    const double rel = worst / (sc_out > 0.0 ? sc_out : 1.0);
    const int pass = rel <= 3e-5;
    printf("  [%s] %-18s %s  max|d|=%.3e  rel=%.3e  %.3f ms (%u heads x %u dim x %u cand)\n",
           label, "index score", pass ? "ok  " : "FAIL", worst, rel, per_call, ih, idim, clen);
    /* the candidate-restricted path must give the same score for every position it
     * scores: a consumer layer can never pick anything outside the source's set */
    const uint32_t n_scan = cand_n ? (cand_n < clen ? cand_n : clen)
                                   : (clen / 4u > 0u ? clen / 4u : 1u);
    int32_t *pos = malloc((size_t)n_scan * sizeof(int32_t));
    float *sub = malloc((size_t)n_scan * sizeof(float));
    for (uint32_t i = 0; i < n_scan; i++) pos[i] = (int32_t)(((uint64_t)i * clen / n_scan) % clen);
    ds4_gpu_tensor *t_p = ds4_gpu_tensor_alloc((uint64_t)n_scan * sizeof(int32_t));
    ds4_gpu_tensor *t_s2 = ds4_gpu_tensor_alloc((uint64_t)n_scan * sizeof(float));
    int sub_ok = t_p && t_s2 && pos && sub;
    sub_ok = sub_ok && ds4_gpu_tensor_write(t_p, 0, pos, (uint64_t)n_scan * sizeof(int32_t));
    sub_ok = sub_ok && ds4_gpu_dsv41_index_score(ih, idim, n_scan, t_q, t_k, t_w, t_p, 1.0f, t_s2);
    sub_ok = sub_ok && ds4_gpu_tensor_read(t_s2, 0, sub, (uint64_t)n_scan * sizeof(float));
    double sub_worst = 0.0;
    if (sub_ok) {
        for (uint32_t i = 0; i < n_scan; i++) {
            const double dd = fabs((double)sub[i] - ref[pos[i]]);
            if (dd > sub_worst) sub_worst = dd;
        }
    }
    const double sub_rel = sub_worst / (sc_out > 0.0 ? sc_out : 1.0);
    const int sub_pass = sub_ok && sub_rel <= 3e-5;

    const double t1 = now_ms();
    for (int r = 0; r < 10; r++)
        ds4_gpu_dsv41_index_score(ih, idim, n_scan, t_q, t_k, t_w, t_p, 1.0f, t_s2);
    const double sub_ms = (now_ms() - t1) / 10.0;
    printf("  [%s] %-18s %s  max|d|=%.3e  %.3f ms for %u of %u positions (%.1f%%)\n",
           label, "index restricted", sub_pass ? "ok  " : "FAIL", sub_worst, sub_ms,
           n_scan, clen, 100.0 * (double)n_scan / (double)clen);

    ds4_gpu_tensor_free(t_p); ds4_gpu_tensor_free(t_s2);
    free(pos); free(sub);
    ds4_gpu_tensor_free(t_q); ds4_gpu_tensor_free(t_k);
    ds4_gpu_tensor_free(t_w); ds4_gpu_tensor_free(t_s);
    free(q); free(k); free(w); free(ref); free(gpu);
    return !(pass && sub_pass);
}

/* RoPE: adjacent pairs as one complex number; inverse conjugates.  A forward followed by
 * an inverse at the same position must return the original exactly enough to be a
 * round trip, which is what the attention output relies on. */
static int run_rope(const char *label, uint32_t rows, uint32_t nh, uint32_t d, uint32_t rd) {
    const size_t n = (size_t)rows * nh * d;
    const uint32_t half = rd / 2u;
    float *x = malloc(n * sizeof(float));
    float *orig = malloc(n * sizeof(float));
    float *ref = malloc(n * sizeof(float));
    float *gpu = malloc(n * sizeof(float));
    float *cv = malloc((size_t)rows * half * sizeof(float));
    float *sv = malloc((size_t)rows * half * sizeof(float));
    if (!x || !orig || !ref || !gpu || !cv || !sv) return 1;
    for (size_t i = 0; i < n; i++) { x[i] = rnd(); orig[i] = x[i]; ref[i] = x[i]; }
    for (uint32_t p = 0; p < rows; p++) {
        for (uint32_t i = 0; i < half; i++) {
            const double a = (double)p / pow(10000.0, (double)(2 * i) / (double)rd);
            cv[(size_t)p * half + i] = (float)cos(a);
            sv[(size_t)p * half + i] = (float)sin(a);
        }
    }
    for (uint32_t p = 0; p < rows; p++)
        for (uint32_t h = 0; h < nh; h++) {
            float *v = ref + ((size_t)p * nh + h) * d + (d - rd);
            for (uint32_t i = 0; i < half; i++) {
                const float re = v[2 * i], im = v[2 * i + 1];
                const float c = cv[(size_t)p * half + i], sn = sv[(size_t)p * half + i];
                v[2 * i] = re * c - im * sn;
                v[2 * i + 1] = re * sn + im * c;
            }
        }

    ds4_gpu_tensor *t_x = ds4_gpu_tensor_alloc(n * sizeof(float));
    ds4_gpu_tensor *t_c = ds4_gpu_tensor_alloc((uint64_t)rows * half * sizeof(float));
    ds4_gpu_tensor *t_s = ds4_gpu_tensor_alloc((uint64_t)rows * half * sizeof(float));
    int ok = t_x && t_c && t_s;
    ok = ok && ds4_gpu_tensor_write(t_x, 0, x, n * sizeof(float));
    ok = ok && ds4_gpu_tensor_write(t_c, 0, cv, (uint64_t)rows * half * sizeof(float));
    ok = ok && ds4_gpu_tensor_write(t_s, 0, sv, (uint64_t)rows * half * sizeof(float));
    ok = ok && ds4_gpu_dsv41_rope(rows, nh, d, rd, 0, 0, t_x, t_c, t_s);
    ok = ok && ds4_gpu_tensor_read(t_x, 0, gpu, n * sizeof(float));
    if (!ok) { printf("  [%s] rope dispatch failed\n", label); return 1; }

    double worst = 0.0;
    for (size_t i = 0; i < n; i++) {
        const double dd = fabs((double)gpu[i] - ref[i]);
        if (dd > worst) worst = dd;
    }
    /* the untouched head prefix must be byte-identical: only the tail rotates */
    int prefix_ok = 1;
    for (uint32_t p = 0; p < rows; p++)
        for (uint32_t h = 0; h < nh; h++)
            for (uint32_t j = 0; j < d - rd; j++)
                if (gpu[((size_t)p * nh + h) * d + j] != orig[((size_t)p * nh + h) * d + j])
                    prefix_ok = 0;

    ok = ds4_gpu_dsv41_rope(rows, nh, d, rd, 0, 1, t_x, t_c, t_s);
    ok = ok && ds4_gpu_tensor_read(t_x, 0, gpu, n * sizeof(float));
    double rt = 0.0;
    if (ok) for (size_t i = 0; i < n; i++) {
        const double dd = fabs((double)gpu[i] - orig[i]);
        if (dd > rt) rt = dd;
    }
    const int pass = ok && prefix_ok && worst <= 3e-6 && rt <= 3e-6;
    printf("  [%s] %-18s %s  max|d|=%.3e  round-trip=%.3e  tail-only=%d\n",
           label, "rope", pass ? "ok  " : "FAIL", worst, rt, prefix_ok);
    ds4_gpu_tensor_free(t_x); ds4_gpu_tensor_free(t_c); ds4_gpu_tensor_free(t_s);
    free(x); free(orig); free(ref); free(gpu); free(cv); free(sv);
    return !pass;
}

/* Grouped output LoRA: wo_a is block diagonal, so row r reads group r / o_lora. */
static int run_lora(const char *label, uint32_t in_per_group, uint32_t o_lora,
                    uint32_t o_groups, int quantized, uint32_t nt) {
    const uint32_t rows = o_groups * o_lora;
    const size_t nw = (size_t)rows * in_per_group;
    float *wf = malloc(nw * sizeof(float));
    const size_t in_row = (size_t)o_groups * in_per_group;
    float *in = malloc(nt * in_row * sizeof(float));
    float *ref = malloc((size_t)nt * rows * sizeof(float));
    float *gpu = malloc((size_t)nt * rows * sizeof(float));
    if (!wf || !in || !ref || !gpu) return 1;
    for (size_t i = 0; i < nw; i++) wf[i] = rnd() * 0.05f;
    for (size_t i = 0; i < nt * in_row; i++) in[i] = rnd();

    const size_t wbytes = quantized ? (nw / 32) * 34 : nw * sizeof(float);
    void *blob = NULL;
    if (posix_memalign(&blob, (size_t)getpagesize(), wbytes) != 0 || !blob) return 1;
    if (quantized) {
        /* q8_0 exactly as the converter writes it, and the reference uses the
         * dequantized values so the comparison is not against pre-quantized weights */
        uint8_t *p8 = (uint8_t *)blob;
        for (size_t b = 0; b < nw / 32; b++) {
            float amax = 0.0f;
            for (uint32_t i = 0; i < 32; i++) amax = fmaxf(amax, fabsf(wf[b * 32 + i]));
            const float d = amax / 127.0f;
            const float id = d != 0.0f ? 1.0f / d : 0.0f;
            const _Float16 h = (_Float16)d;
            memcpy(p8 + b * 34, &h, 2);
            for (uint32_t i = 0; i < 32; i++) {
                const int8_t q = (int8_t)lrintf(wf[b * 32 + i] * id);
                p8[b * 34 + 2 + i] = (uint8_t)q;
                wf[b * 32 + i] = (float)q * (float)h;
            }
        }
    } else {
        memcpy(blob, wf, wbytes);
    }

    /* every token is the single-token maths against the same shared weight */
    for (uint32_t t = 0; t < nt; t++)
        for (uint32_t r = 0; r < rows; r++) {
            const float *xr = in + t * in_row + (size_t)(r / o_lora) * in_per_group;
            double acc = 0.0;
            for (uint32_t k = 0; k < in_per_group; k++)
                acc += (double)wf[(size_t)r * in_per_group + k] * xr[k];
            ref[(size_t)t * rows + r] = (float)acc;
        }

    ds4_gpu_tensor *t_in = ds4_gpu_tensor_alloc(nt * in_row * sizeof(float));
    ds4_gpu_tensor *t_out = ds4_gpu_tensor_alloc((uint64_t)nt * rows * sizeof(float));
    int ok = t_in && t_out && ds4_gpu_set_model_map(blob, wbytes);
    ok = ok && ds4_gpu_tensor_write(t_in, 0, in, nt * in_row * sizeof(float));
    ok = ok && ds4_gpu_dsv41_output_lora(blob, wbytes, 0, in_per_group, o_lora, o_groups,
                                         quantized, nt, t_in, t_out);
    ok = ok && ds4_gpu_tensor_read(t_out, 0, gpu, (uint64_t)nt * rows * sizeof(float));
    if (!ok) { printf("  [%s] output lora dispatch failed\n", label); return 1; }

    double worst = 0.0, sc = 0.0;
    for (size_t r = 0; r < (size_t)nt * rows; r++) {
        const double dd = fabs((double)gpu[r] - ref[r]);
        if (dd > worst) worst = dd;
        if (fabs((double)ref[r]) > sc) sc = fabs((double)ref[r]);
    }
    const double rel = worst / (sc > 0.0 ? sc : 1.0);
    const int pass = rel <= 3e-5;
    printf("  [%s] %-18s %s  max|d|=%.3e  rel=%.3e  (%u groups x %u rows x %u in, %s, %u tok)\n",
           label, "output lora", pass ? "ok  " : "FAIL", worst, rel,
           o_groups, o_lora, in_per_group, quantized ? "q8_0" : "f32", nt);
    ds4_gpu_tensor_free(t_in); ds4_gpu_tensor_free(t_out);
    free(wf); free(in); free(ref); free(gpu); free(blob);
    return !pass;
}

/* Selection.  `tie_frac` forces a fraction of the scores to exactly 0.0, which is what
 * relu does in the real indexer when every head's dot goes negative -- the case where the
 * tie order decides the output.  The reference here is the released repeated-argmax with
 * >= (prefer the later index among equals), transcribed independently. */
static int run_select(const char *label, uint32_t n, uint32_t k, uint32_t block,
                      uint32_t reach, float tie_frac) {
    float *sc = malloc((size_t)n * sizeof(float));
    int32_t *ref = malloc((size_t)k * sizeof(int32_t));
    int32_t *gpu = malloc((size_t)k * sizeof(int32_t));
    int32_t *keep = malloc((size_t)n * sizeof(int32_t));
    float *bmax_ref = malloc((size_t)((n + block - 1u) / block) * sizeof(float));
    float *bmax_gpu = malloc((size_t)((n + block - 1u) / block) * sizeof(float));
    if (!sc || !ref || !gpu || !keep || !bmax_ref || !bmax_gpu) return 1;
    for (uint32_t i = 0; i < n; i++) {
        const float r = (rnd() + 1.0f) * 0.5f;
        sc[i] = r < tie_frac ? 0.0f : rnd();
    }
    for (uint32_t i = reach; i < n; i++) sc[i] = -1.0e30f;   /* unreachable */

    /* reference top-k: repeated argmax, >= so equals resolve to the later index */
    const uint32_t offset = 64u;
    for (uint32_t i = 0; i < n; i++) keep[i] = 0;
    for (uint32_t s2 = 0; s2 < k; s2++) {
        int best = -1;
        float bv = 0.0f;
        for (uint32_t j = 0; j < n; j++) {
            if (keep[j]) continue;
            if (best < 0 || sc[j] >= bv) { best = (int)j; bv = sc[j]; }
        }
        if (best >= 0) keep[best] = 1;
    }
    uint32_t no = 0;
    for (uint32_t j = 0; j < n && no < k; j++)
        if (keep[j]) ref[no++] = j < reach ? (int32_t)(j + offset) : -1;
    while (no < k) ref[no++] = -1;

    const uint32_t nb = (n + block - 1u) / block;
    for (uint32_t b = 0; b < nb; b++) {
        float mx = -1.0e30f;
        for (uint32_t j = b * block; j < (b + 1u) * block && j < n; j++) mx = fmaxf(mx, sc[j]);
        bmax_ref[b] = (reach > 0 && b == (reach - 1u) / block) ? 1.0e30f : mx;
    }

    ds4_gpu_tensor *t_sc = ds4_gpu_tensor_alloc((uint64_t)n * sizeof(float));
    ds4_gpu_tensor *t_keep = ds4_gpu_tensor_alloc((uint64_t)n * sizeof(int32_t));
    ds4_gpu_tensor *t_out = ds4_gpu_tensor_alloc((uint64_t)k * sizeof(int32_t));
    ds4_gpu_tensor *t_bm = ds4_gpu_tensor_alloc((uint64_t)nb * sizeof(float));
    int ok = t_sc && t_keep && t_out && t_bm;
    ok = ok && ds4_gpu_tensor_write(t_sc, 0, sc, (uint64_t)n * sizeof(float));
    ok = ok && ds4_gpu_dsv41_block_max(n, block, nb, reach, t_sc, t_bm);
    ok = ok && ds4_gpu_tensor_read(t_bm, 0, bmax_gpu, (uint64_t)nb * sizeof(float));
    ok = ok && ds4_gpu_dsv41_topk_select(n, k, offset, reach, t_sc, t_keep, t_out);
    ok = ok && ds4_gpu_tensor_read(t_out, 0, gpu, (uint64_t)k * sizeof(int32_t));
    if (!ok) { printf("  [%s] selection dispatch failed\n", label); return 1; }

    const double t0 = now_ms();
    for (int r = 0; r < 10; r++)
        ds4_gpu_dsv41_topk_select(n, k, offset, reach, t_sc, t_keep, t_out);
    const double per_call = (now_ms() - t0) / 10.0;

    uint32_t bad_bm = 0, bad_tk = 0;
    for (uint32_t b = 0; b < nb; b++) if (bmax_gpu[b] != bmax_ref[b]) bad_bm++;
    for (uint32_t i = 0; i < k; i++) if (gpu[i] != ref[i]) bad_tk++;
    /* ascending order is what sparse_attn consumes */
    int ascending = 1;
    for (uint32_t i = 1; i < k; i++)
        if (gpu[i] >= 0 && gpu[i - 1] >= 0 && gpu[i] <= gpu[i - 1]) ascending = 0;
    const int pass = !bad_bm && !bad_tk && ascending;
    printf("  [%s] %-18s %s  blocks %u/%u exact, top-k %u/%u exact, ascending=%d, %.3f ms"
           " (n=%u k=%u ties~%.0f%%)\n",
           label, "selection", pass ? "ok  " : "FAIL", nb - bad_bm, nb, k - bad_tk, k,
           ascending, per_call, n, k, 100.0 * tie_frac);

    ds4_gpu_tensor_free(t_sc); ds4_gpu_tensor_free(t_keep);
    ds4_gpu_tensor_free(t_out); ds4_gpu_tensor_free(t_bm);
    free(sc); free(ref); free(gpu); free(keep); free(bmax_ref); free(bmax_gpu);
    return !pass;
}

/* mHC mixes + Sinkhorn, transcribed independently.  The check that matters beyond value
 * agreement is that `comb` comes out DOUBLY STOCHASTIC -- every row and every column
 * summing to 1 -- because that is the property the Sinkhorn sweeps exist to produce. */
static int run_hc(const char *label, uint32_t dim, uint32_t hc, uint32_t iters, int f16w,
                  uint32_t n_tokens) {
    const uint32_t hc_dim = hc * dim;
    const uint32_t mix_hc = (2u + hc) * hc;
    const float norm_eps = 1.0e-20f, hc_eps = 1.0e-6f;
    float *stream = malloc((size_t)n_tokens * hc_dim * sizeof(float));
    float *wf = malloc((size_t)mix_hc * hc_dim * sizeof(float));
    float *scale = malloc(3 * sizeof(float));
    float *base = malloc((size_t)mix_hc * sizeof(float));
    float *pre_r = malloc((size_t)hc * sizeof(float));
    float *post_r = malloc((size_t)hc * sizeof(float));
    float *comb_r = malloc((size_t)hc * hc * sizeof(float));
    float *pre_g = malloc((size_t)n_tokens * hc * sizeof(float));
    float *post_g = malloc((size_t)n_tokens * hc * sizeof(float));
    float *comb_g = malloc((size_t)n_tokens * hc * hc * sizeof(float));
    if (!stream || !wf || !scale || !base) return 1;
    for (uint32_t i = 0; i < hc_dim; i++) stream[i] = rnd();
    for (uint32_t t = 1; t < n_tokens; t++)
        memcpy(stream + (size_t)t * hc_dim, stream, (size_t)hc_dim * sizeof(float));
    for (size_t i = 0; i < (size_t)mix_hc * hc_dim; i++) wf[i] = rnd() * 0.05f;
    for (uint32_t i = 0; i < 3; i++) scale[i] = 0.5f + 0.5f * rnd();
    for (uint32_t i = 0; i < mix_hc; i++) base[i] = rnd();

    const size_t wbytes = (size_t)mix_hc * hc_dim * (f16w ? 2 : 4);
    void *blob = NULL;
    if (posix_memalign(&blob, (size_t)getpagesize(), wbytes) != 0 || !blob) return 1;
    if (f16w) {
        _Float16 *h = (_Float16 *)blob;
        for (size_t i = 0; i < (size_t)mix_hc * hc_dim; i++) {
            h[i] = (_Float16)wf[i];
            wf[i] = (float)h[i];   /* reference uses what the kernel will actually read */
        }
    } else {
        memcpy(blob, wf, wbytes);
    }

    double ss = 0.0;
    for (uint32_t i = 0; i < hc_dim; i++) ss += (double)stream[i] * stream[i];
    const float rstd = (float)(1.0 / sqrt(ss / hc_dim + norm_eps));
    float *mixes = malloc((size_t)mix_hc * sizeof(float));
    for (uint32_t m = 0; m < mix_hc; m++) {
        double acc = 0.0;
        for (uint32_t i = 0; i < hc_dim; i++) acc += (double)wf[(size_t)m * hc_dim + i] * stream[i];
        mixes[m] = (float)acc * rstd;
    }
    for (uint32_t j = 0; j < hc; j++) {
        const float a = mixes[j] * scale[0] + base[j];
        const float b = mixes[hc + j] * scale[1] + base[hc + j];
        pre_r[j] = (a >= 0.0f ? 1.0f / (1.0f + expf(-a)) : expf(a) / (1.0f + expf(a))) + hc_eps;
        post_r[j] = 2.0f * (b >= 0.0f ? 1.0f / (1.0f + expf(-b)) : expf(b) / (1.0f + expf(b)));
    }
    for (uint32_t j = 0; j < hc * hc; j++)
        comb_r[j] = mixes[2 * hc + j] * scale[2] + base[2 * hc + j];
    for (uint32_t j = 0; j < hc; j++) {
        float *row = comb_r + (size_t)j * hc;
        float mx = row[0];
        for (uint32_t k = 1; k < hc; k++) if (row[k] > mx) mx = row[k];
        double sum = 0.0;
        for (uint32_t k = 0; k < hc; k++) { row[k] = expf(row[k] - mx); sum += row[k]; }
        for (uint32_t k = 0; k < hc; k++) row[k] = (float)(row[k] / sum) + hc_eps;
    }
    for (uint32_t k = 0; k < hc; k++) {
        double col = 0.0;
        for (uint32_t j = 0; j < hc; j++) col += comb_r[(size_t)j * hc + k];
        const float inv = (float)(col + hc_eps);
        for (uint32_t j = 0; j < hc; j++) comb_r[(size_t)j * hc + k] /= inv;
    }
    for (uint32_t it = 1; it < iters; it++) {
        for (uint32_t j = 0; j < hc; j++) {
            double row = 0.0;
            for (uint32_t k = 0; k < hc; k++) row += comb_r[(size_t)j * hc + k];
            const float inv = (float)(row + hc_eps);
            for (uint32_t k = 0; k < hc; k++) comb_r[(size_t)j * hc + k] /= inv;
        }
        for (uint32_t k = 0; k < hc; k++) {
            double col = 0.0;
            for (uint32_t j = 0; j < hc; j++) col += comb_r[(size_t)j * hc + k];
            const float inv = (float)(col + hc_eps);
            for (uint32_t j = 0; j < hc; j++) comb_r[(size_t)j * hc + k] /= inv;
        }
    }

    ds4_gpu_tensor *t_s = ds4_gpu_tensor_alloc((uint64_t)n_tokens * hc_dim * sizeof(float));
    ds4_gpu_tensor *t_sc = ds4_gpu_tensor_alloc(3 * sizeof(float));
    ds4_gpu_tensor *t_b = ds4_gpu_tensor_alloc((uint64_t)mix_hc * sizeof(float));
    ds4_gpu_tensor *t_pre = ds4_gpu_tensor_alloc((uint64_t)n_tokens * hc * sizeof(float));
    ds4_gpu_tensor *t_post = ds4_gpu_tensor_alloc((uint64_t)n_tokens * hc * sizeof(float));
    ds4_gpu_tensor *t_comb = ds4_gpu_tensor_alloc((uint64_t)n_tokens * hc * hc * sizeof(float));
    int ok = t_s && t_sc && t_b && t_pre && t_post && t_comb && ds4_gpu_set_model_map(blob, wbytes);
    ok = ok && ds4_gpu_tensor_write(t_s, 0, stream, (uint64_t)n_tokens * hc_dim * sizeof(float));
    ok = ok && ds4_gpu_tensor_write(t_sc, 0, scale, 3 * sizeof(float));
    ok = ok && ds4_gpu_tensor_write(t_b, 0, base, (uint64_t)mix_hc * sizeof(float));
    /* every token gets the same stream here, so all n_tokens results must be identical
     * to the single-token reference -- that is what checks the per-threadgroup indexing */
    ok = ok && ds4_gpu_dsv41_hc_mixes(blob, wbytes, 0, n_tokens, hc_dim, hc, iters,
                                      norm_eps, hc_eps, f16w, t_s, t_sc, t_b,
                                      t_pre, t_post, t_comb);
    ok = ok && ds4_gpu_tensor_read(t_pre, 0, pre_g, (uint64_t)n_tokens * hc * sizeof(float));
    ok = ok && ds4_gpu_tensor_read(t_post, 0, post_g, (uint64_t)n_tokens * hc * sizeof(float));
    ok = ok && ds4_gpu_tensor_read(t_comb, 0, comb_g, (uint64_t)n_tokens * hc * hc * sizeof(float));
    if (!ok) { printf("  [%s] hc mixes dispatch failed\n", label); return 1; }

    double worst = 0.0;
    for (uint32_t t = 0; t < n_tokens; t++) {
        for (uint32_t j = 0; j < hc; j++) {
            worst = fmax(worst, fabs((double)pre_g[t * hc + j] - pre_r[j]));
            worst = fmax(worst, fabs((double)post_g[t * hc + j] - post_r[j]));
        }
        for (uint32_t j = 0; j < hc * hc; j++)
            worst = fmax(worst, fabs((double)comb_g[t * hc * hc + j] - comb_r[j]));
    }

    /* the whole point of the Sinkhorn sweeps */
    double ds_err = 0.0;
    for (uint32_t j = 0; j < hc; j++) {
        double r = 0.0, c = 0.0;
        for (uint32_t k = 0; k < hc; k++) { r += comb_g[j * hc + k]; c += comb_g[k * hc + j]; }
        ds_err = fmax(ds_err, fmax(fabs(r - 1.0), fabs(c - 1.0)));
    }
    const int pass = worst <= 3e-5 && ds_err <= 1e-3;
    printf("  [%s] %-18s %s  max|d|=%.3e  doubly-stochastic err=%.3e  (hc=%u dim=%u %s)\n",
           label, "hc mixes", pass ? "ok  " : "FAIL", worst, ds_err, hc, dim,
           f16w ? "f16" : "f32");

    ds4_gpu_tensor_free(t_s); ds4_gpu_tensor_free(t_sc); ds4_gpu_tensor_free(t_b);
    ds4_gpu_tensor_free(t_pre); ds4_gpu_tensor_free(t_post); ds4_gpu_tensor_free(t_comb);
    free(stream); free(wf); free(scale); free(base); free(mixes);
    free(pre_r); free(post_r); free(comb_r); free(pre_g); free(post_g); free(comb_g);
    free(blob);
    return !pass;
}

/* hc_pre collapses the copies with the previous sublayer's mix; hc_post expands a
 * sublayer output back out folding the residual through comb, which is indexed
 * [k * hc + c] -- the sum runs over the RESIDUAL copy k, not the output copy. */
static int run_hcends(const char *label, uint32_t dim, uint32_t hc, uint32_t rows) {
    const size_t hcd = (size_t)hc * dim;
    float *stream = malloc(rows * hcd * sizeof(float));
    float *mix = malloc((size_t)rows * hc * sizeof(float));
    float *sub = malloc((size_t)rows * dim * sizeof(float));
    float *post = malloc((size_t)rows * hc * sizeof(float));
    float *comb = malloc((size_t)rows * hc * hc * sizeof(float));
    float *pre_r = malloc((size_t)rows * dim * sizeof(float));
    float *pre_g = malloc((size_t)rows * dim * sizeof(float));
    float *out_r = malloc(rows * hcd * sizeof(float));
    float *out_g = malloc(rows * hcd * sizeof(float));
    if (!stream || !mix || !sub || !post || !comb) return 1;
    for (size_t i = 0; i < rows * hcd; i++) stream[i] = rnd();
    for (size_t i = 0; i < (size_t)rows * hc; i++) { mix[i] = rnd(); post[i] = rnd(); }
    for (size_t i = 0; i < (size_t)rows * dim; i++) sub[i] = rnd();
    for (size_t i = 0; i < (size_t)rows * hc * hc; i++) comb[i] = rnd();

    /* every row is independent, so the reference is the single-token maths run per row */
    for (uint32_t t = 0; t < rows; t++) {
        const float *st = stream + (size_t)t * hcd, *mx = mix + (size_t)t * hc;
        const float *sb = sub + (size_t)t * dim, *po = post + (size_t)t * hc;
        const float *cb = comb + (size_t)t * hc * hc;
        for (uint32_t j = 0; j < dim; j++) {
            double acc = 0.0;
            for (uint32_t c = 0; c < hc; c++) acc += (double)mx[c] * st[c * dim + j];
            pre_r[(size_t)t * dim + j] = (float)acc;
        }
        for (uint32_t c = 0; c < hc; c++)
            for (uint32_t j = 0; j < dim; j++) {
                double acc = (double)po[c] * sb[j];
                for (uint32_t k = 0; k < hc; k++) acc += (double)cb[k * hc + c] * st[k * dim + j];
                out_r[(size_t)t * hcd + c * dim + j] = (float)acc;
            }
    }

    ds4_gpu_tensor *t_st = ds4_gpu_tensor_alloc(rows * hcd * sizeof(float));
    ds4_gpu_tensor *t_mix = ds4_gpu_tensor_alloc((uint64_t)rows * hc * sizeof(float));
    ds4_gpu_tensor *t_sub = ds4_gpu_tensor_alloc((uint64_t)rows * dim * sizeof(float));
    ds4_gpu_tensor *t_post = ds4_gpu_tensor_alloc((uint64_t)rows * hc * sizeof(float));
    ds4_gpu_tensor *t_comb = ds4_gpu_tensor_alloc((uint64_t)rows * hc * hc * sizeof(float));
    ds4_gpu_tensor *t_pre = ds4_gpu_tensor_alloc((uint64_t)rows * dim * sizeof(float));
    ds4_gpu_tensor *t_out = ds4_gpu_tensor_alloc(rows * hcd * sizeof(float));
    int ok = t_st && t_mix && t_sub && t_post && t_comb && t_pre && t_out;
    ok = ok && ds4_gpu_tensor_write(t_st, 0, stream, rows * hcd * sizeof(float));
    ok = ok && ds4_gpu_tensor_write(t_mix, 0, mix, (uint64_t)rows * hc * sizeof(float));
    ok = ok && ds4_gpu_tensor_write(t_sub, 0, sub, (uint64_t)rows * dim * sizeof(float));
    ok = ok && ds4_gpu_tensor_write(t_post, 0, post, (uint64_t)rows * hc * sizeof(float));
    ok = ok && ds4_gpu_tensor_write(t_comb, 0, comb, (uint64_t)rows * hc * hc * sizeof(float));
    ok = ok && ds4_gpu_dsv41_hc_pre(rows, dim, hc, t_st, t_mix, t_pre);
    ok = ok && ds4_gpu_tensor_read(t_pre, 0, pre_g, (uint64_t)rows * dim * sizeof(float));
    ok = ok && ds4_gpu_dsv41_hc_post(rows, dim, hc, t_sub, t_st, t_post, t_comb, t_out);
    ok = ok && ds4_gpu_tensor_read(t_out, 0, out_g, rows * hcd * sizeof(float));
    if (!ok) { printf("  [%s] hc ends dispatch failed\n", label); return 1; }

    double wp = 0.0, wo = 0.0, sp = 0.0, so = 0.0;
    for (size_t i = 0; i < (size_t)rows * dim; i++) {
        wp = fmax(wp, fabs((double)pre_g[i] - pre_r[i]));
        sp = fmax(sp, fabs((double)pre_r[i]));
    }
    for (size_t i = 0; i < rows * hcd; i++) {
        wo = fmax(wo, fabs((double)out_g[i] - out_r[i]));
        so = fmax(so, fabs((double)out_r[i]));
    }
    const int pass = wp / fmax(sp, 1e-9) <= 3e-6 && wo / fmax(so, 1e-9) <= 3e-6;
    printf("  [%s] %-18s %s  hc_pre rel=%.3e  hc_post rel=%.3e  (rows=%u)\n", label, "hc ends",
           pass ? "ok  " : "FAIL", wp / fmax(sp, 1e-9), wo / fmax(so, 1e-9), rows);
    ds4_gpu_tensor_free(t_st); ds4_gpu_tensor_free(t_mix); ds4_gpu_tensor_free(t_sub);
    ds4_gpu_tensor_free(t_post); ds4_gpu_tensor_free(t_comb);
    ds4_gpu_tensor_free(t_pre); ds4_gpu_tensor_free(t_out);
    free(stream); free(mix); free(sub); free(post); free(comb);
    free(pre_r); free(pre_g); free(out_r); free(out_g);
    return !pass;
}

/* MoE routing.  The tie rule here is strict `>` -- equals resolve to the LOWER index,
 * the opposite of the indexer's rule -- and the weights come from the UNBIASED scores. */
static int run_route(const char *label, uint32_t n_exp, uint32_t topk, float route_scale,
                     uint32_t nt) {
    float *logits = malloc((size_t)nt * n_exp * sizeof(float));
    float *bias = malloc((size_t)n_exp * sizeof(float));
    float *scores = malloc((size_t)n_exp * sizeof(float));
    int32_t *idx_r = malloc((size_t)nt * topk * sizeof(int32_t));
    int32_t *idx_g = malloc((size_t)nt * topk * sizeof(int32_t));
    float *w_r = malloc((size_t)nt * topk * sizeof(float));
    float *w_g = malloc((size_t)nt * topk * sizeof(float));
    if (!logits || !bias) return 1;
    for (size_t i = 0; i < (size_t)nt * n_exp; i++) logits[i] = rnd() * 3.0f;
    for (uint32_t i = 0; i < n_exp; i++) bias[i] = rnd();

    /* each token routes independently against the same bias */
    for (uint32_t t = 0; t < nt; t++) {
    const float *lg = logits + (size_t)t * n_exp;
    int32_t *idx_rr = idx_r + (size_t)t * topk;
    float *w_rr = w_r + (size_t)t * topk;
    for (uint32_t i = 0; i < n_exp; i++) {
        const float x = lg[i];
        const float sp = x > 20.0f ? x : (x < -20.0f ? expf(x) : log1pf(expf(x)));
        scores[i] = sqrtf(sp);
    }
    for (uint32_t s = 0; s < topk; s++) {
        int best = -1;
        float bv = -1.0e30f;
        for (uint32_t i = 0; i < n_exp; i++) {
            int taken = 0;
            for (uint32_t p = 0; p < s; p++) if (idx_rr[p] == (int32_t)i) { taken = 1; break; }
            if (taken) continue;
            const float v = scores[i] + bias[i];
            if (v > bv) { bv = v; best = (int)i; }
        }
        idx_rr[s] = best;
        w_rr[s] = scores[best];
    }
    if (topk > 1) {
        double sum = 0.0;
        for (uint32_t s = 0; s < topk; s++) sum += w_rr[s];
        for (uint32_t s = 0; s < topk; s++) w_rr[s] = (float)(w_rr[s] / (sum + 1e-20));
    }
    for (uint32_t s = 0; s < topk; s++) w_rr[s] *= route_scale;
    }

    ds4_gpu_tensor *t_l = ds4_gpu_tensor_alloc((uint64_t)nt * n_exp * sizeof(float));
    ds4_gpu_tensor *t_b = ds4_gpu_tensor_alloc((uint64_t)n_exp * sizeof(float));
    ds4_gpu_tensor *t_i = ds4_gpu_tensor_alloc((uint64_t)nt * topk * sizeof(int32_t));
    ds4_gpu_tensor *t_w = ds4_gpu_tensor_alloc((uint64_t)nt * topk * sizeof(float));
    int ok = t_l && t_b && t_i && t_w;
    ok = ok && ds4_gpu_tensor_write(t_l, 0, logits, (uint64_t)nt * n_exp * sizeof(float));
    ok = ok && ds4_gpu_tensor_write(t_b, 0, bias, (uint64_t)n_exp * sizeof(float));
    ok = ok && ds4_gpu_dsv41_route(nt, n_exp, topk, route_scale, 1, t_l, t_b, t_i, t_w);
    ok = ok && ds4_gpu_tensor_read(t_i, 0, idx_g, (uint64_t)nt * topk * sizeof(int32_t));
    ok = ok && ds4_gpu_tensor_read(t_w, 0, w_g, (uint64_t)nt * topk * sizeof(float));
    if (!ok) { printf("  [%s] routing dispatch failed\n", label); return 1; }

    uint32_t bad = 0;
    double wd = 0.0;
    for (size_t s = 0; s < (size_t)nt * topk; s++) {
        if (idx_g[s] != idx_r[s]) bad++;
        wd = fmax(wd, fabs((double)w_g[s] - w_r[s]));
    }
    const int pass = !bad && wd <= 3e-6;
    printf("  [%s] %-18s %s  %u/%u picks, max|dw|=%.3e  (%u experts top-%u, %u tok)\n",
           label, "moe routing", pass ? "ok  " : "FAIL", (uint32_t)(nt * topk) - bad,
           (uint32_t)(nt * topk), wd, n_exp, topk, nt);
    ds4_gpu_tensor_free(t_l); ds4_gpu_tensor_free(t_b);
    ds4_gpu_tensor_free(t_i); ds4_gpu_tensor_free(t_w);
    free(logits); free(bias); free(scores);
    free(idx_r); free(idx_g); free(w_r); free(w_g);
    return !pass;
}

/* The MoE reaches each expert by offsetting into the stacked routed tensor, so the
 * shared matmul has to be correct at a non-zero weight offset, not just at zero. */
static int run_offset_matmul(const char *label, uint32_t in_dim, uint32_t out_dim,
                             uint32_t n_expert, uint32_t pick) {
    const size_t per = (size_t)in_dim * out_dim;
    const size_t nw = per * n_expert;
    float *w = NULL, *x = malloc((size_t)in_dim * sizeof(float));
    float *ref = malloc((size_t)out_dim * sizeof(float));
    float *gpu = malloc((size_t)out_dim * sizeof(float));
    void *blob = NULL;
    if (posix_memalign(&blob, (size_t)getpagesize(), nw * sizeof(float)) != 0) return 1;
    w = (float *)blob;
    for (size_t i = 0; i < nw; i++) w[i] = rnd() * 0.05f;
    for (uint32_t i = 0; i < in_dim; i++) x[i] = rnd();

    const float *we = w + (size_t)pick * per;
    for (uint32_t o = 0; o < out_dim; o++) {
        double acc = 0.0;
        for (uint32_t i = 0; i < in_dim; i++) acc += (double)we[(size_t)o * in_dim + i] * x[i];
        ref[o] = (float)acc;
    }

    ds4_gpu_tensor *t_x = ds4_gpu_tensor_alloc((uint64_t)in_dim * sizeof(float));
    ds4_gpu_tensor *t_o = ds4_gpu_tensor_alloc((uint64_t)out_dim * sizeof(float));
    int ok = t_x && t_o && ds4_gpu_set_model_map(blob, nw * sizeof(float));
    ok = ok && ds4_gpu_tensor_write(t_x, 0, x, (uint64_t)in_dim * sizeof(float));
    ok = ok && ds4_gpu_matmul_f32_tensor(t_o, blob, nw * sizeof(float),
                                         (uint64_t)pick * per * sizeof(float),
                                         in_dim, out_dim, t_x, 1);
    ok = ok && ds4_gpu_tensor_read(t_o, 0, gpu, (uint64_t)out_dim * sizeof(float));
    if (!ok) { printf("  [%s] offset matmul dispatch failed\n", label); return 1; }

    double worst = 0.0, sc = 0.0;
    for (uint32_t o = 0; o < out_dim; o++) {
        worst = fmax(worst, fabs((double)gpu[o] - ref[o]));
        sc = fmax(sc, fabs((double)ref[o]));
    }
    const double rel = worst / fmax(sc, 1e-9);
    const int pass = rel <= 3e-6;
    printf("  [%s] %-18s %s  rel=%.3e  (expert %u of %u, %ux%u)\n", label, "offset matmul",
           pass ? "ok  " : "FAIL", rel, pick, n_expert, out_dim, in_dim);
    ds4_gpu_tensor_free(t_x); ds4_gpu_tensor_free(t_o);
    free(x); free(ref); free(gpu); free(blob);
    return !pass;
}



/* MXFP4 routed experts + a Q8_0 shared expert through DS4's fused routed-MoE kernels,
 * which is how the released checkpoint stores them.  The blob is synthesised straight in
 * the quantized layouts and the reference dequantises the same bytes, so the comparison
 * measures the kernels, not a quantizer.  The clamp is set low enough that the asymmetric
 * V4.1 SwiGLU limit actually bites on both sides. */
static const float MXFP4_LUT[16] = {
     0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};

static double e8m0(uint8_t e) {
    const uint32_t bits = e == 0 ? 0x00400000u : (uint32_t)e << 23;
    float v;
    memcpy(&v, &bits, sizeof(v));
    return (double)v;
}

static double mxfp4_row_dot(const uint8_t *row, uint32_t n, const double *y) {
    double acc = 0.0;
    for (uint32_t ib = 0; ib < n / 32u; ib++) {
        const uint8_t *b = row + (size_t)ib * 17u;
        const double d = e8m0(b[0]);
        for (uint32_t j = 0; j < 16u; j++) {
            acc += d * MXFP4_LUT[b[1 + j] & 15u] * y[ib * 32u + j];
            acc += d * MXFP4_LUT[b[1 + j] >> 4] * y[ib * 32u + 16u + j];
        }
    }
    return acc;
}

static double q8_0_row_dot(const uint8_t *row, uint32_t n, const double *y) {
    double acc = 0.0;
    for (uint32_t ib = 0; ib < n / 32u; ib++) {
        const uint8_t *b = row + (size_t)ib * 34u;
        __fp16 h;
        memcpy(&h, b, sizeof(h));
        const double d = (double)(float)h;
        for (uint32_t j = 0; j < 32u; j++) {
            acc += d * (double)(int8_t)b[2 + j] * y[ib * 32u + j];
        }
    }
    return acc;
}

static void fill_mxfp4(uint8_t *p, size_t blocks) {
    for (size_t b = 0; b < blocks; b++) {
        /* scales around 2^-5 keep a 6.0 code near 0.2, so a row dot lands in clamp range */
        p[b * 17u] = (uint8_t)(120u + (uint32_t)((rnd() + 1.0f) * 2.9f));
        for (uint32_t j = 0; j < 16u; j++) {
            p[b * 17u + 1u + j] = (uint8_t)((uint32_t)((rnd() + 1.0f) * 127.4f) & 0xffu);
        }
    }
}

static void fill_q8_0(uint8_t *p, size_t blocks) {
    for (size_t b = 0; b < blocks; b++) {
        const __fp16 d = (__fp16)(0.002f + 0.001f * (rnd() + 1.0f));
        memcpy(p + b * 34u, &d, sizeof(d));
        for (uint32_t j = 0; j < 32u; j++) {
            p[b * 34u + 2u + j] = (uint8_t)(int8_t)(int)(rnd() * 127.0f);
        }
    }
}

/* A prefill chunk's dense projections go through the matrix kernels; V4.1 takes them with
 * F32 tiles.  Q8_0 and F16 weights, more rows than the matvec path handles, against a
 * double reference; the half-tiled result is reported next to it. */
static int run_rows_matmul(const char *label, uint32_t in_dim, uint32_t out_dim, uint32_t rows, int q8) {
    const size_t wbytes = q8 ? (size_t)out_dim * (in_dim / 32u) * 34u : (size_t)out_dim * in_dim * 2u;
    void *blob = NULL;
    if (posix_memalign(&blob, (size_t)getpagesize(), wbytes) != 0) return 1;
    uint8_t *w = blob;
    if (q8) {
        fill_q8_0(w, (size_t)out_dim * (in_dim / 32u));
    } else {
        for (size_t i = 0; i < (size_t)out_dim * in_dim; i++) {
            const __fp16 h = (__fp16)(rnd() * 0.05f);
            memcpy(w + 2u * i, &h, 2u);
        }
    }
    float *x = malloc((size_t)rows * in_dim * sizeof(float));
    for (size_t i = 0; i < (size_t)rows * in_dim; i++) x[i] = rnd();
    double *xd = malloc((size_t)in_dim * sizeof(double));
    float *ref = malloc((size_t)rows * out_dim * sizeof(float));
    for (uint32_t r = 0; r < rows; r++) {
        for (uint32_t i = 0; i < in_dim; i++) xd[i] = x[(size_t)r * in_dim + i];
        for (uint32_t o = 0; o < out_dim; o++) {
            double acc = 0.0;
            if (q8) {
                acc = q8_0_row_dot(w + (size_t)o * (in_dim / 32u) * 34u, in_dim, xd);
            } else {
                for (uint32_t i = 0; i < in_dim; i++) {
                    __fp16 h; memcpy(&h, w + 2u * ((size_t)o * in_dim + i), 2u);
                    acc += (double)(float)h * xd[i];
                }
            }
            ref[(size_t)r * out_dim + o] = (float)acc;
        }
    }
    ds4_gpu_tensor *t_x = ds4_gpu_tensor_alloc((uint64_t)rows * in_dim * sizeof(float));
    ds4_gpu_tensor *t_o = ds4_gpu_tensor_alloc((uint64_t)rows * out_dim * sizeof(float));
    float *gpu = malloc((size_t)rows * out_dim * sizeof(float));
    int ok = t_x && t_o && ds4_gpu_set_model_map(blob, wbytes) &&
             ds4_gpu_tensor_write(t_x, 0, x, (uint64_t)rows * in_dim * sizeof(float));
    double rel[2] = {0.0, 0.0};
    for (int exact = 1; exact >= 0 && ok; exact--) {
        ds4_gpu_set_exact_mm(exact);
        ok = q8 ? ds4_gpu_matmul_quant_tensor(t_o, blob, wbytes, 0, 8u, in_dim, out_dim, t_x, rows)
                : ds4_gpu_matmul_f16_tensor(t_o, blob, wbytes, 0, in_dim, out_dim, t_x, rows);
        ds4_gpu_set_exact_mm(0);
        ok = ok && ds4_gpu_tensor_read(t_o, 0, gpu, (uint64_t)rows * out_dim * sizeof(float));
        double worst = 0.0, sc = 0.0;
        for (size_t i = 0; ok && i < (size_t)rows * out_dim; i++) {
            worst = fmax(worst, fabs((double)gpu[i] - ref[i]));
            sc = fmax(sc, fabs((double)ref[i]));
        }
        rel[exact] = worst / fmax(sc, 1e-9);
    }
    if (!ok) { printf("  [%s] rows matmul dispatch failed\n", label); return 1; }
    const int pass = rel[1] <= 3e-6;
    printf("  [%s] %-18s %s  rel=%.3e f32 tiles, %.3e half tiles  (%s %ux%u, %u rows)\n", label,
           "rows matmul", pass ? "ok  " : "FAIL", rel[1], rel[0], q8 ? "q8_0" : "f16", out_dim, in_dim, rows);
    ds4_gpu_tensor_free(t_x); ds4_gpu_tensor_free(t_o);
    free(blob); free(x); free(xd); free(ref); free(gpu);
    return !pass;
}

static int run_fused_moe(const char *label, uint32_t in_dim, uint32_t mid_dim,
                         uint32_t n_total, uint32_t topk, float clamp) {
    const uint32_t out_dim = in_dim;
    const uint64_t g_row = (uint64_t)(in_dim / 32u) * 17u;
    const uint64_t d_row = (uint64_t)(mid_dim / 32u) * 17u;
    const uint64_t g_exp = (uint64_t)mid_dim * g_row;
    const uint64_t d_exp = (uint64_t)out_dim * d_row;
    const uint64_t sg_bytes = (uint64_t)mid_dim * (in_dim / 32u) * 34u;
    const uint64_t sd_bytes = (uint64_t)out_dim * (mid_dim / 32u) * 34u;
    /* every region starts 256-aligned so the model-map wrap can bind it */
    const uint64_t o_gate = 0;
    const uint64_t o_up = o_gate + ((n_total * g_exp + 255u) & ~255ull);
    const uint64_t o_down = o_up + ((n_total * g_exp + 255u) & ~255ull);
    const uint64_t o_sg = o_down + ((n_total * d_exp + 255u) & ~255ull);
    const uint64_t o_su = o_sg + ((sg_bytes + 255u) & ~255ull);
    const uint64_t o_sd = o_su + ((sg_bytes + 255u) & ~255ull);
    const uint64_t total = o_sd + ((sd_bytes + 255u) & ~255ull);

    void *blob = NULL;
    if (posix_memalign(&blob, (size_t)getpagesize(), (size_t)total) != 0) return 1;
    memset(blob, 0, (size_t)total);
    uint8_t *base = (uint8_t *)blob;
    fill_mxfp4(base + o_gate, (size_t)(n_total * g_exp / 17u));
    fill_mxfp4(base + o_up, (size_t)(n_total * g_exp / 17u));
    fill_mxfp4(base + o_down, (size_t)(n_total * d_exp / 17u));
    fill_q8_0(base + o_sg, (size_t)(sg_bytes / 34u));
    fill_q8_0(base + o_su, (size_t)(sg_bytes / 34u));
    fill_q8_0(base + o_sd, (size_t)(sd_bytes / 34u));

    float *x = malloc((size_t)in_dim * sizeof(float));
    double *xd = malloc((size_t)in_dim * sizeof(double));
    for (uint32_t i = 0; i < in_dim; i++) { x[i] = rnd(); xd[i] = x[i]; }
    int32_t *sel = malloc((size_t)topk * sizeof(int32_t));
    float *wts = malloc((size_t)topk * sizeof(float));
    for (uint32_t s = 0; s < topk; s++) {
        sel[s] = (int32_t)((s * 3u + 1u) % n_total);
        wts[s] = 0.1f + 0.2f * (rnd() + 1.0f);
    }

    /* reference: the shared expert at weight 1, then the routed set, summed */
    double *act = malloc((size_t)mid_dim * sizeof(double));
    double *ref = calloc((size_t)out_dim, sizeof(double));
    uint64_t clamped = 0, total_act = 0;
    for (uint32_t s = 0; s <= topk; s++) {
        const int shared = (s == topk);
        const uint64_t e = shared ? 0u : (uint64_t)sel[s];
        const double weight = shared ? 1.0 : (double)wts[s];
        for (uint32_t j = 0; j < mid_dim; j++) {
            double g, u;
            if (shared) {
                g = q8_0_row_dot(base + o_sg + (uint64_t)j * (in_dim / 32u) * 34u, in_dim, xd);
                u = q8_0_row_dot(base + o_su + (uint64_t)j * (in_dim / 32u) * 34u, in_dim, xd);
            } else {
                g = mxfp4_row_dot(base + o_gate + e * g_exp + (uint64_t)j * g_row, in_dim, xd);
                u = mxfp4_row_dot(base + o_up + e * g_exp + (uint64_t)j * g_row, in_dim, xd);
            }
            if (clamp > 1.0e-6f) {
                if (g > clamp || fabs(u) > clamp) clamped++;
                if (g > clamp) g = clamp;
                if (u > clamp) u = clamp;
                if (u < -clamp) u = -clamp;
            }
            total_act++;
            act[j] = (g / (1.0 + exp(-g))) * u * weight;
        }
        for (uint32_t o = 0; o < out_dim; o++) {
            ref[o] += shared
                ? q8_0_row_dot(base + o_sd + (uint64_t)o * (mid_dim / 32u) * 34u, mid_dim, act)
                : mxfp4_row_dot(base + o_down + e * d_exp + (uint64_t)o * d_row, mid_dim, act);
        }
    }

    ds4_gpu_tensor *t_x = ds4_gpu_tensor_alloc((uint64_t)in_dim * sizeof(float));
    ds4_gpu_tensor *t_sel = ds4_gpu_tensor_alloc((uint64_t)topk * sizeof(int32_t));
    ds4_gpu_tensor *t_w = ds4_gpu_tensor_alloc((uint64_t)topk * sizeof(float));
    ds4_gpu_tensor *t_g = ds4_gpu_tensor_alloc((uint64_t)topk * mid_dim * sizeof(float));
    ds4_gpu_tensor *t_u = ds4_gpu_tensor_alloc((uint64_t)topk * mid_dim * sizeof(float));
    ds4_gpu_tensor *t_m = ds4_gpu_tensor_alloc((uint64_t)topk * mid_dim * sizeof(float));
    ds4_gpu_tensor *t_e = ds4_gpu_tensor_alloc((uint64_t)topk * out_dim * sizeof(float));
    ds4_gpu_tensor *t_sh = ds4_gpu_tensor_alloc((uint64_t)out_dim * sizeof(float));
    ds4_gpu_tensor *t_sa = ds4_gpu_tensor_alloc((uint64_t)mid_dim * sizeof(float));
    ds4_gpu_tensor *t_o = ds4_gpu_tensor_alloc((uint64_t)out_dim * sizeof(float));
    int ok = t_x && t_sel && t_w && t_g && t_u && t_m && t_e && t_sh && t_sa && t_o &&
             ds4_gpu_set_model_map(blob, total);
    ok = ok && ds4_gpu_tensor_write(t_x, 0, x, (uint64_t)in_dim * sizeof(float));
    ok = ok && ds4_gpu_tensor_write(t_sel, 0, sel, (uint64_t)topk * sizeof(int32_t));
    ok = ok && ds4_gpu_tensor_write(t_w, 0, wts, (uint64_t)topk * sizeof(float));
    /* the shared expert, exactly as the engine composes it */
    ok = ok && ds4_gpu_matmul_quant_tensor(t_g, blob, total, o_sg, 8u, in_dim, mid_dim, t_x, 1);
    ok = ok && ds4_gpu_matmul_quant_tensor(t_u, blob, total, o_su, 8u, in_dim, mid_dim, t_x, 1);
    ok = ok && ds4_gpu_swiglu_tensor(t_sa, t_g, t_u, mid_dim, clamp, 1.0f);
    ok = ok && ds4_gpu_matmul_quant_tensor(t_sh, blob, total, o_sd, 8u, mid_dim, out_dim, t_sa, 1);
    /* mirrors the engine: the addend only reaches the fused pair+sum6 decode path */
    const int fold = (topk == 6u);
    double per_call = 0.0;
    ok = ok && ds4_gpu_routed_moe_one_tensor(t_o, t_g, t_u, t_m, t_e, blob, total,
                                             o_gate, o_up, o_down, 39u, 39u,
                                             g_exp, g_row, d_exp, d_row,
                                             in_dim, mid_dim, out_dim, t_sel, t_w,
                                             n_total, topk, clamp, t_x,
                                             fold ? t_sh : NULL, 0, false);
    ok = ok && (fold || ds4_gpu_add_tensor(t_o, t_o, t_sh, out_dim));
    float *gpu = malloc((size_t)out_dim * sizeof(float));
    ok = ok && ds4_gpu_tensor_read(t_o, 0, gpu, (uint64_t)out_dim * sizeof(float));
    if (ok && getenv("DS4_DSV41_TIME_MOE")) {
        const int reps = 20;
        const double t0 = now_ms();
        for (int r = 0; r < reps; r++) {
            ok = ok && ds4_gpu_routed_moe_one_tensor(t_o, t_g, t_u, t_m, t_e, blob, total,
                                                     o_gate, o_up, o_down, 39u, 39u,
                                                     g_exp, g_row, d_exp, d_row,
                                                     in_dim, mid_dim, out_dim, t_sel, t_w,
                                                     n_total, topk, clamp, t_x,
                                                     fold ? t_sh : NULL, 0, false);
        }
        per_call = (now_ms() - t0) / reps;
    }
    if (!ok) { printf("  [%s] fused moe dispatch failed\n", label); return 1; }

    double worst = 0.0, sc = 0.0;
    for (uint32_t o = 0; o < out_dim; o++) {
        worst = fmax(worst, fabs((double)gpu[o] - ref[o]));
        sc = fmax(sc, fabs(ref[o]));
    }
    const double rel = worst / fmax(sc, 1e-9);

    /* Three tokens in one pass through the same kernels: rows 0 and 2 repeat the token
     * above; row 1 takes the experts in the opposite slot order, which exercises the
     * per-token id and weight indexing.  Every row must land on the routed reference. */
    int rows_ok = -1;
    if (topk == 6u) {
        const uint32_t R = 3u;
        float *x3 = malloc((size_t)R * in_dim * sizeof(float));
        int32_t *sel3 = malloc((size_t)R * topk * sizeof(int32_t));
        float *wts3 = malloc((size_t)R * topk * sizeof(float));
        float *out3 = malloc((size_t)R * out_dim * sizeof(float));
        for (uint32_t r = 0; r < R; r++) {
            memcpy(x3 + (size_t)r * in_dim, x, (size_t)in_dim * sizeof(float));
            for (uint32_t s2 = 0; s2 < topk; s2++) {
                const uint32_t src = r == 1u ? topk - 1u - s2 : s2;
                sel3[r * topk + s2] = sel[src];
                wts3[r * topk + s2] = wts[src];
            }
        }
        ds4_gpu_tensor *t_x3 = ds4_gpu_tensor_alloc((uint64_t)R * in_dim * sizeof(float));
        ds4_gpu_tensor *t_sel3 = ds4_gpu_tensor_alloc((uint64_t)R * topk * sizeof(int32_t));
        ds4_gpu_tensor *t_w3 = ds4_gpu_tensor_alloc((uint64_t)R * topk * sizeof(float));
        ds4_gpu_tensor *t_g3 = ds4_gpu_tensor_alloc((uint64_t)R * topk * mid_dim * sizeof(float));
        ds4_gpu_tensor *t_u3 = ds4_gpu_tensor_alloc((uint64_t)R * topk * mid_dim * sizeof(float));
        ds4_gpu_tensor *t_m3 = ds4_gpu_tensor_alloc((uint64_t)R * topk * mid_dim * sizeof(float));
        ds4_gpu_tensor *t_e3 = ds4_gpu_tensor_alloc((uint64_t)R * topk * out_dim * sizeof(float));
        ds4_gpu_tensor *t_o3 = ds4_gpu_tensor_alloc((uint64_t)R * out_dim * sizeof(float));
        int rok = x3 && sel3 && wts3 && out3 && t_x3 && t_sel3 && t_w3 && t_g3 && t_u3 &&
                  t_m3 && t_e3 && t_o3;
        rok = rok && ds4_gpu_tensor_write(t_x3, 0, x3, (uint64_t)R * in_dim * sizeof(float));
        rok = rok && ds4_gpu_tensor_write(t_sel3, 0, sel3, (uint64_t)R * topk * sizeof(int32_t));
        rok = rok && ds4_gpu_tensor_write(t_w3, 0, wts3, (uint64_t)R * topk * sizeof(float));
        rok = rok && ds4_gpu_routed_moe_tokens_tensor(t_o3, t_g3, t_u3, t_m3, t_e3, blob, total,
                                                      o_gate, o_up, o_down, 39u, 39u,
                                                      g_exp, g_row, d_exp, d_row,
                                                      in_dim, mid_dim, out_dim, t_sel3, t_w3,
                                                      n_total, topk, clamp, t_x3, 0, R);
        rok = rok && ds4_gpu_tensor_read(t_o3, 0, out3, (uint64_t)R * out_dim * sizeof(float));
        if (rok) {
            rows_ok = 1;
            /* the single-token run above folded the shared expert; strip it back off the
             * rows, which carry the routed sum alone */
            float *sh = malloc((size_t)out_dim * sizeof(float));
            rok = ds4_gpu_tensor_read(t_sh, 0, sh, (uint64_t)out_dim * sizeof(float));
            for (uint32_t r = 0; r < R && rok; r++) {
                double w3 = 0.0;
                for (uint32_t o = 0; o < out_dim; o++) {
                    const double got = out3[(size_t)r * out_dim + o];
                    w3 = fmax(w3, fabs(got - (ref[o] - sh[o])));
                }
                if (w3 / fmax(sc, 1e-9) > 3e-5) rows_ok = 0;
            }
            if (!rok) rows_ok = 0;
            free(sh);
        } else {
            rows_ok = 0;
        }
        ds4_gpu_tensor_free(t_x3); ds4_gpu_tensor_free(t_sel3); ds4_gpu_tensor_free(t_w3);
        ds4_gpu_tensor_free(t_g3); ds4_gpu_tensor_free(t_u3); ds4_gpu_tensor_free(t_m3);
        ds4_gpu_tensor_free(t_e3); ds4_gpu_tensor_free(t_o3);
        free(x3); free(sel3); free(wts3); free(out3);
    }
    /* Expert-major: eight tokens, each its own input and selection, chosen so experts are
     * shared by several tokens (groups of 1..4).  Every token is held to its own
     * double-precision reference, which is the routed sum plus the shared expert. */
    int xm_ok = -1;
    {
        const uint32_t R = 8u;
        float *xr = malloc((size_t)R * in_dim * sizeof(float));
        double *xd_r = malloc((size_t)in_dim * sizeof(double));
        int32_t *selr = malloc((size_t)R * topk * sizeof(int32_t));
        float *wr = malloc((size_t)R * topk * sizeof(float));
        double *refr = calloc((size_t)R * out_dim, sizeof(double));
        float *shr = malloc((size_t)R * out_dim * sizeof(float));
        float *outr = malloc((size_t)R * out_dim * sizeof(float));
        for (uint32_t r = 0; r < R; r++) {
            for (uint32_t i = 0; i < in_dim; i++) xr[(size_t)r * in_dim + i] = rnd();
            for (uint32_t s2 = 0; s2 < topk; s2++) {
                /* token r takes experts (r + 2 s) mod n_total: heavy overlap between rows */
                selr[r * topk + s2] = (int32_t)((r + 2u * s2) % n_total);
                wr[r * topk + s2] = 0.1f + 0.2f * (rnd() + 1.0f);
            }
        }
        for (uint32_t r = 0; r < R; r++) {
            for (uint32_t i = 0; i < in_dim; i++) xd_r[i] = xr[(size_t)r * in_dim + i];
            for (uint32_t s2 = 0; s2 <= topk; s2++) {
                const int shared = (s2 == topk);
                const uint64_t e = shared ? 0u : (uint64_t)selr[r * topk + s2];
                const double weight = shared ? 1.0 : (double)wr[r * topk + s2];
                for (uint32_t j = 0; j < mid_dim; j++) {
                    double g, u;
                    if (shared) {
                        g = q8_0_row_dot(base + o_sg + (uint64_t)j * (in_dim / 32u) * 34u, in_dim, xd_r);
                        u = q8_0_row_dot(base + o_su + (uint64_t)j * (in_dim / 32u) * 34u, in_dim, xd_r);
                    } else {
                        g = mxfp4_row_dot(base + o_gate + e * g_exp + (uint64_t)j * g_row, in_dim, xd_r);
                        u = mxfp4_row_dot(base + o_up + e * g_exp + (uint64_t)j * g_row, in_dim, xd_r);
                    }
                    if (clamp > 1.0e-6f) {
                        if (g > clamp) g = clamp;
                        if (u > clamp) u = clamp;
                        if (u < -clamp) u = -clamp;
                    }
                    act[j] = (g / (1.0 + exp(-g))) * u * weight;
                }
                for (uint32_t o = 0; o < out_dim; o++) {
                    const double v = shared
                        ? q8_0_row_dot(base + o_sd + (uint64_t)o * (mid_dim / 32u) * 34u, mid_dim, act)
                        : mxfp4_row_dot(base + o_down + e * d_exp + (uint64_t)o * d_row, mid_dim, act);
                    refr[(size_t)r * out_dim + o] += v;
                    if (shared) shr[(size_t)r * out_dim + o] = (float)v;
                }
            }
        }
        ds4_gpu_tensor *t_xr = ds4_gpu_tensor_alloc((uint64_t)R * in_dim * sizeof(float));
        ds4_gpu_tensor *t_selr = ds4_gpu_tensor_alloc((uint64_t)R * topk * sizeof(int32_t));
        ds4_gpu_tensor *t_wr = ds4_gpu_tensor_alloc((uint64_t)R * topk * sizeof(float));
        ds4_gpu_tensor *t_shr = ds4_gpu_tensor_alloc((uint64_t)R * out_dim * sizeof(float));
        ds4_gpu_tensor *t_cnt = ds4_gpu_tensor_alloc((uint64_t)n_total * sizeof(uint32_t));
        ds4_gpu_tensor *t_cur = ds4_gpu_tensor_alloc(((uint64_t)n_total + 1u) * sizeof(uint32_t));
        ds4_gpu_tensor *t_grp = ds4_gpu_tensor_alloc((uint64_t)R * topk * 3u * sizeof(uint32_t));
        ds4_gpu_tensor *t_srt = ds4_gpu_tensor_alloc((uint64_t)R * topk * sizeof(uint32_t));
        ds4_gpu_tensor *t_midr = ds4_gpu_tensor_alloc((uint64_t)R * topk * mid_dim * sizeof(float));
        ds4_gpu_tensor *t_exr = ds4_gpu_tensor_alloc((uint64_t)R * topk * out_dim * sizeof(float));
        ds4_gpu_tensor *t_outr = ds4_gpu_tensor_alloc((uint64_t)R * out_dim * sizeof(float));
        int xok = xr && xd_r && selr && wr && refr && shr && outr && t_xr && t_selr && t_wr &&
                  t_shr && t_cnt && t_cur && t_grp && t_srt && t_midr && t_exr && t_outr;
        xok = xok && ds4_gpu_tensor_fill_f32(t_cnt, 0.0f, n_total);
        xok = xok && ds4_gpu_tensor_write(t_xr, 0, xr, (uint64_t)R * in_dim * sizeof(float));
        xok = xok && ds4_gpu_tensor_write(t_selr, 0, selr, (uint64_t)R * topk * sizeof(int32_t));
        xok = xok && ds4_gpu_tensor_write(t_wr, 0, wr, (uint64_t)R * topk * sizeof(float));
        xok = xok && ds4_gpu_tensor_write(t_shr, 0, shr, (uint64_t)R * out_dim * sizeof(float));
        /* Two calls queued in one batch, as the engine queues one per layer: a decoy
         * selection with fewer expert groups first, then the one under test.  The second
         * must see its own sort -- counts back at zero, its own group count -- not the
         * decoy's. */
        int32_t *sel_decoy = malloc((size_t)R * topk * sizeof(int32_t));
        ds4_gpu_tensor *t_decoy = ds4_gpu_tensor_alloc((uint64_t)R * topk * sizeof(int32_t));
        for (uint32_t i = 0; i < R * topk; i++) sel_decoy[i] = (int32_t)(i % topk);
        xok = xok && sel_decoy && t_decoy &&
              ds4_gpu_tensor_write(t_decoy, 0, sel_decoy, (uint64_t)R * topk * sizeof(int32_t));
        if (xok) {
            const int batched = ds4_gpu_begin_commands() != 0;
            xok = ds4_gpu_dsv41_moe_expert_major(blob, total, o_gate, o_up, o_down,
                                                 g_exp, g_row, d_exp, d_row, n_total, topk, R,
                                                 in_dim, mid_dim, out_dim, clamp,
                                                 t_xr, t_decoy, t_wr, t_shr, t_cnt, t_cur, t_grp,
                                                 t_srt, t_midr, t_exr, t_outr) &&
                  ds4_gpu_dsv41_moe_expert_major(blob, total, o_gate, o_up, o_down,
                                                 g_exp, g_row, d_exp, d_row, n_total, topk, R,
                                                 in_dim, mid_dim, out_dim, clamp,
                                                 t_xr, t_selr, t_wr, t_shr, t_cnt, t_cur, t_grp,
                                                 t_srt, t_midr, t_exr, t_outr);
            if (batched) ds4_gpu_end_commands();
        }
        free(sel_decoy); ds4_gpu_tensor_free(t_decoy);
        xok = xok && ds4_gpu_tensor_read(t_outr, 0, outr, (uint64_t)R * out_dim * sizeof(float));
        double xm_worst = 0.0, xm_scale = 0.0;
        if (xok) {
            for (size_t i = 0; i < (size_t)R * out_dim; i++) {
                xm_worst = fmax(xm_worst, fabs((double)outr[i] - refr[i]));
                xm_scale = fmax(xm_scale, fabs(refr[i]));
            }
            xm_ok = xm_worst / fmax(xm_scale, 1e-9) <= 3e-5;
        } else {
            xm_ok = 0;
        }
        if (getenv("DS4_DSV41_TIME_MOE") && xok) {
            const int reps = 20;
            const double t0 = now_ms();
            for (int r = 0; r < reps; r++) {
                ds4_gpu_dsv41_moe_expert_major(blob, total, o_gate, o_up, o_down,
                                               g_exp, g_row, d_exp, d_row, n_total, topk, R,
                                               in_dim, mid_dim, out_dim, clamp,
                                               t_xr, t_selr, t_wr, t_shr, t_cnt, t_cur, t_grp,
                                               t_srt, t_midr, t_exr, t_outr);
            }
            printf("  [%s] %-18s      %.3f ms/call for %u tokens expert-major\n", label,
                   "xm moe timing", (now_ms() - t0) / reps, R);
        }
        printf("  [%s] %-18s %s  rel=%.3e  (%u tokens, experts shared across rows)\n", label,
               "expert-major moe", xm_ok ? "ok  " : "FAIL", xm_worst / fmax(xm_scale, 1e-9), R);
        ds4_gpu_tensor_free(t_xr); ds4_gpu_tensor_free(t_selr); ds4_gpu_tensor_free(t_wr);
        ds4_gpu_tensor_free(t_shr); ds4_gpu_tensor_free(t_cnt); ds4_gpu_tensor_free(t_cur);
        ds4_gpu_tensor_free(t_grp); ds4_gpu_tensor_free(t_srt); ds4_gpu_tensor_free(t_midr);
        ds4_gpu_tensor_free(t_exr); ds4_gpu_tensor_free(t_outr);
        free(xr); free(xd_r); free(selr); free(wr); free(refr); free(shr); free(outr);
    }
    const int pass = rel <= 3e-5 && rows_ok != 0 && xm_ok != 0;
    printf("  [%s] %-18s %s  rel=%.3e  (mxfp4 %u experts top-%u, %u->%u->%u, q8_0 shared, "
           "%s, clamped %.0f%%%s)\n",
           label, "fused moe", pass ? "ok  " : "FAIL", rel, n_total, topk,
           in_dim, mid_dim, out_dim, fold ? "shared folded" : "shared added",
           100.0 * (double)clamped / (double)(total_act ? total_act : 1),
           rows_ok < 0 ? "" : rows_ok ? ", 3 tokens in one pass ok" : ", 3-TOKEN PASS FAIL");
    if (per_call > 0.0) {
        /* the routed experts are the traffic: topk * (gate + up + down) rows of mxfp4 */
        const double bytes = (double)topk * ((double)mid_dim * (double)g_row * 2.0 +
                                             (double)out_dim * (double)d_row);
        printf("  [%s] %-18s      %.3f ms/call, %.0f GB/s over %.1f MB of experts\n",
               label, "fused moe timing", per_call, bytes / per_call / 1e6, bytes / 1e6);
    }

    ds4_gpu_tensor_free(t_x); ds4_gpu_tensor_free(t_sel); ds4_gpu_tensor_free(t_w);
    ds4_gpu_tensor_free(t_g); ds4_gpu_tensor_free(t_u); ds4_gpu_tensor_free(t_m);
    ds4_gpu_tensor_free(t_e); ds4_gpu_tensor_free(t_sh); ds4_gpu_tensor_free(t_sa);
    ds4_gpu_tensor_free(t_o);
    free(x); free(xd); free(sel); free(wts); free(act); free(ref); free(gpu); free(blob);
    return !pass;
}

/* dsv41_gpu_matvec dispatches the attention projections by stored type, so each arm has
 * to read its own layout at the released shapes.  The reference decodes exactly what was
 * written, so a layout error shows up as a gross mismatch rather than as quant noise. */
static int run_matvec(const char *label, uint32_t in_dim, uint32_t out_dim, uint32_t type) {
    const uint64_t row_bytes = type == 0u ? (uint64_t)in_dim * 4u
                             : type == 1u ? (uint64_t)in_dim * 2u
                                          : (uint64_t)(in_dim / 32u) * 34u;
    const uint64_t bytes = (uint64_t)out_dim * row_bytes;
    void *blob = NULL;
    if (posix_memalign(&blob, (size_t)getpagesize(), (size_t)bytes) != 0) return 1;
    uint8_t *base = (uint8_t *)blob;

    float *x = malloc((size_t)in_dim * sizeof(float));
    double *xd = malloc((size_t)in_dim * sizeof(double));
    for (uint32_t i = 0; i < in_dim; i++) { x[i] = rnd(); xd[i] = x[i]; }

    double *ref = malloc((size_t)out_dim * sizeof(double));
    float *w = malloc((size_t)in_dim * sizeof(float));
    for (uint32_t o = 0; o < out_dim; o++) {
        for (uint32_t i = 0; i < in_dim; i++) w[i] = rnd() * 0.05f;
        uint8_t *row = base + (uint64_t)o * row_bytes;
        double acc = 0.0;
        if (type == 0u) {
            memcpy(row, w, (size_t)in_dim * sizeof(float));
            for (uint32_t i = 0; i < in_dim; i++) acc += (double)w[i] * xd[i];
        } else if (type == 1u) {
            for (uint32_t i = 0; i < in_dim; i++) {
                const __fp16 h = (__fp16)w[i];
                memcpy(row + (size_t)i * 2u, &h, sizeof(h));
                acc += (double)(float)h * xd[i];
            }
        } else {
            for (uint32_t b = 0; b < in_dim / 32u; b++) {
                float amax = 0.0f;
                for (uint32_t j = 0; j < 32u; j++) amax = fmaxf(amax, fabsf(w[b * 32u + j]));
                const __fp16 hd = (__fp16)(amax / 127.0f);
                memcpy(row + (size_t)b * 34u, &hd, sizeof(hd));
                const double d = (double)(float)hd;
                for (uint32_t j = 0; j < 32u; j++) {
                    int q = d > 0.0 ? (int)lrint((double)w[b * 32u + j] / d) : 0;
                    if (q > 127) q = 127;
                    if (q < -127) q = -127;
                    row[(size_t)b * 34u + 2u + j] = (uint8_t)(int8_t)q;
                    acc += d * (double)q * xd[b * 32u + j];
                }
            }
        }
        ref[o] = acc;
    }

    ds4_gpu_tensor *t_x = ds4_gpu_tensor_alloc((uint64_t)in_dim * sizeof(float));
    ds4_gpu_tensor *t_o = ds4_gpu_tensor_alloc((uint64_t)out_dim * sizeof(float));
    int ok = t_x && t_o && ds4_gpu_set_model_map(blob, bytes);
    ok = ok && ds4_gpu_tensor_write(t_x, 0, x, (uint64_t)in_dim * sizeof(float));
    if (type == 0u) {
        ok = ok && ds4_gpu_matmul_f32_tensor(t_o, blob, bytes, 0, in_dim, out_dim, t_x, 1);
    } else if (type == 1u) {
        ok = ok && ds4_gpu_matmul_f16_tensor(t_o, blob, bytes, 0, in_dim, out_dim, t_x, 1);
    } else {
        ok = ok && ds4_gpu_matmul_quant_tensor(t_o, blob, bytes, 0, type,
                                               in_dim, out_dim, t_x, 1);
    }
    float *gpu = malloc((size_t)out_dim * sizeof(float));
    ok = ok && ds4_gpu_tensor_read(t_o, 0, gpu, (uint64_t)out_dim * sizeof(float));
    if (!ok) { printf("  [%s] matvec dispatch failed\n", label); return 1; }

    double worst = 0.0, sc = 0.0;
    for (uint32_t o = 0; o < out_dim; o++) {
        worst = fmax(worst, fabs((double)gpu[o] - ref[o]));
        sc = fmax(sc, fabs(ref[o]));
    }
    const double rel = worst / fmax(sc, 1e-9);
    const int pass = rel <= 3e-6;
    printf("  [%s] %-18s %s  rel=%.3e  (%s %ux%u)\n", label, "typed matvec",
           pass ? "ok  " : "FAIL", rel,
           type == 0u ? "f32" : type == 1u ? "f16" : "q8_0", out_dim, in_dim);
    ds4_gpu_tensor_free(t_x); ds4_gpu_tensor_free(t_o);
    free(x); free(xd); free(ref); free(w); free(gpu); free(blob);
    return !pass;
}

/* The radix select has to agree with the O(n^2) rank pass EXACTLY, tie order included:
 * relu leaves hundreds of positions at 0.0, and picking different ones there changes the
 * attention output.  The reference is the same independent repeated-argmax transcription
 * run_select uses, so agreement is not two calls into one implementation. */
static int run_radix(const char *label, uint32_t n, uint32_t k, float tie_frac) {
    float *sc = malloc((size_t)n * sizeof(float));
    int32_t *keep_r = malloc((size_t)n * sizeof(int32_t));
    int32_t *keep_g = malloc((size_t)n * sizeof(int32_t));
    if (!sc || !keep_r || !keep_g) return 1;
    for (uint32_t i = 0; i < n; i++) {
        sc[i] = ((float)((i * 2654435761u) % 1000u) / 1000.0f) < tie_frac ? 0.0f : rnd();
    }

    /* repeated argmax with >=: among equals the LATER index wins */
    for (uint32_t i = 0; i < n; i++) keep_r[i] = 0;
    for (uint32_t s = 0; s < k; s++) {
        int best = -1;
        float bv = 0.0f;
        for (uint32_t j = 0; j < n; j++) {
            if (keep_r[j]) continue;
            if (best < 0 || sc[j] >= bv) { best = (int)j; bv = sc[j]; }
        }
        if (best >= 0) keep_r[best] = 1;
    }

    ds4_gpu_tensor *t_sc = ds4_gpu_tensor_alloc((uint64_t)n * sizeof(float));
    ds4_gpu_tensor *t_keep = ds4_gpu_tensor_alloc((uint64_t)n * sizeof(int32_t));
    ds4_gpu_tensor *t_out = ds4_gpu_tensor_alloc((uint64_t)k * sizeof(int32_t));
    ds4_gpu_tensor *t_state = ds4_gpu_tensor_alloc(5 * sizeof(uint32_t));
    ds4_gpu_tensor *t_hist = ds4_gpu_tensor_alloc(256 * sizeof(uint32_t));
    int ok = t_sc && t_keep && t_out && t_state && t_hist;
    ok = ok && ds4_gpu_tensor_write(t_sc, 0, sc, (uint64_t)n * sizeof(float));
    /* below the cap the entry point would pick the rank pass, and this test exists to
     * exercise the radix one at every size */
    setenv("DS4_DSV41_FORCE_RADIX", "1", 1);
    const double t0 = now_ms();
    ok = ok && ds4_gpu_dsv41_topk_select_scratch(n, k, 0, n, t_sc, t_keep, t_out,
                                                 t_state, t_hist);
    unsetenv("DS4_DSV41_FORCE_RADIX");
    const double ms = now_ms() - t0;
    ok = ok && ds4_gpu_tensor_read(t_keep, 0, keep_g, (uint64_t)n * sizeof(int32_t));
    if (!ok) { printf("  [%s] radix select dispatch failed\n", label); return 1; }

    uint32_t bad = 0, kept = 0;
    for (uint32_t i = 0; i < n; i++) {
        if (keep_g[i]) kept++;
        if (keep_g[i] != keep_r[i]) bad++;
    }
    /* the engine queues every layer's select into one command batch over the same
     * scratch; a second select with a different k, encoded behind the first, must not
     * disturb it and must come out right itself */
    uint32_t bad_batch = 0;
    {
        const uint32_t k2 = k / 2u > 0u ? k / 2u : 1u;
        int32_t *keep_r2 = malloc((size_t)n * sizeof(int32_t));
        int32_t *keep_b = malloc((size_t)n * sizeof(int32_t));
        ds4_gpu_tensor *t_keep2 = ds4_gpu_tensor_alloc((uint64_t)n * sizeof(int32_t));
        ds4_gpu_tensor *t_out2 = ds4_gpu_tensor_alloc((uint64_t)k2 * sizeof(int32_t));
        for (uint32_t i = 0; i < n; i++) keep_r2[i] = 0;
        for (uint32_t s = 0; s < k2; s++) {
            int best = -1;
            float bv = 0.0f;
            for (uint32_t j = 0; j < n; j++) {
                if (keep_r2[j]) continue;
                if (best < 0 || sc[j] >= bv) { best = (int)j; bv = sc[j]; }
            }
            if (best >= 0) keep_r2[best] = 1;
        }
        setenv("DS4_DSV41_FORCE_RADIX", "1", 1);
        const int batched = ds4_gpu_begin_commands() != 0;
        int bok = keep_r2 && keep_b && t_keep2 && t_out2 &&
                  ds4_gpu_dsv41_topk_select_scratch(n, k, 0, n, t_sc, t_keep, t_out,
                                                    t_state, t_hist) &&
                  ds4_gpu_dsv41_topk_select_scratch(n, k2, 0, n, t_sc, t_keep2, t_out2,
                                                    t_state, t_hist);
        if (batched) ds4_gpu_end_commands();
        unsetenv("DS4_DSV41_FORCE_RADIX");
        bok = bok && ds4_gpu_tensor_read(t_keep, 0, keep_b, (uint64_t)n * sizeof(int32_t));
        if (bok) for (uint32_t i = 0; i < n; i++) if (keep_b[i] != keep_r[i]) bad_batch++;
        bok = bok && ds4_gpu_tensor_read(t_keep2, 0, keep_b, (uint64_t)n * sizeof(int32_t));
        if (bok) for (uint32_t i = 0; i < n; i++) if (keep_b[i] != keep_r2[i]) bad_batch++;
        if (!bok) bad_batch = n;
        ds4_gpu_tensor_free(t_keep2); ds4_gpu_tensor_free(t_out2);
        free(keep_r2); free(keep_b);
    }
    const int pass = !bad && kept == k && !bad_batch;
    printf("  [%s] %-18s %s  %u/%u kept, %u mismatches, %u batched mismatches, %.2f ms "
           "(n=%u k=%u ties~%.0f%%)\n",
           label, "radix select", pass ? "ok  " : "FAIL", kept, k, bad, bad_batch, ms, n, k,
           100.0 * tie_frac);
    ds4_gpu_tensor_free(t_sc); ds4_gpu_tensor_free(t_keep); ds4_gpu_tensor_free(t_out);
    ds4_gpu_tensor_free(t_state); ds4_gpu_tensor_free(t_hist);
    free(sc); free(keep_r); free(keep_g);
    return !pass;
}

/* The rows entry's radix passes (the engine's path) against the same repeated-argmax
 * reference, over rows of different reach sharing one score array. */
static int run_radix_rows(const char *label, uint32_t n, uint32_t k, float tie_frac, uint32_t rows) {
    float *sc = malloc((size_t)n * sizeof(float));
    int32_t *keep_r = malloc((size_t)n * sizeof(int32_t));
    int32_t *keep_g = malloc((size_t)rows * n * sizeof(int32_t));
    for (uint32_t i = 0; i < n; i++) {
        sc[i] = ((float)((i * 2654435761u) % 1000u) / 1000.0f) < tie_frac ? 0.0f : rnd();
    }
    ds4_gpu_tensor *t_sc = ds4_gpu_tensor_alloc((uint64_t)rows * n * sizeof(float));
    ds4_gpu_tensor *t_keep = ds4_gpu_tensor_alloc((uint64_t)rows * n * sizeof(int32_t));
    ds4_gpu_tensor *t_out = ds4_gpu_tensor_alloc((uint64_t)rows * k * sizeof(int32_t));
    ds4_gpu_tensor *t_state = ds4_gpu_tensor_alloc((uint64_t)rows * 5u * sizeof(uint32_t));
    ds4_gpu_tensor *t_hist = ds4_gpu_tensor_alloc((uint64_t)rows * 256u * sizeof(uint32_t));
    int ok = sc && keep_r && keep_g && t_sc && t_keep && t_out && t_state && t_hist;
    for (uint32_t r = 0; ok && r < rows; r++) {
        ok = ds4_gpu_tensor_write(t_sc, (uint64_t)r * n * sizeof(float), sc, (uint64_t)n * sizeof(float));
    }
    /* row r reaches n - rows + r + 1 positions */
    const uint32_t pos0 = n - rows;
    setenv("DS4_DSV41_FORCE_RADIX", "1", 1);
    const double t0 = now_ms();
    ok = ok && ds4_gpu_dsv41_select_rows(rows, n, n, pos0, 1u, k, 0, k, 0, 0, t_sc, t_keep, t_out,
                                         t_state, t_hist);
    unsetenv("DS4_DSV41_FORCE_RADIX");
    const double ms = now_ms() - t0;
    ok = ok && ds4_gpu_tensor_read(t_keep, 0, keep_g, (uint64_t)rows * n * sizeof(int32_t));
    int32_t *out_g = malloc((size_t)rows * k * sizeof(int32_t));
    ok = ok && out_g && ds4_gpu_tensor_read(t_out, 0, out_g, (uint64_t)rows * k * sizeof(int32_t));
    uint32_t bad = 0, bad_out = 0;
    for (uint32_t r = 0; ok && r < rows; r++) {
        const uint32_t nr = pos0 + r + 1u;
        for (uint32_t i = 0; i < n; i++) keep_r[i] = 0;
        for (uint32_t s2 = 0; s2 < k; s2++) {
            int best = -1;
            float bv = 0.0f;
            for (uint32_t j = 0; j < nr; j++) {
                if (keep_r[j]) continue;
                if (best < 0 || sc[j] >= bv) { best = (int)j; bv = sc[j]; }
            }
            if (best >= 0) keep_r[best] = 1;
        }
        for (uint32_t i = 0; i < nr; i++) if (keep_g[(size_t)r * n + i] != keep_r[i]) bad++;
        /* the compacted list: the kept positions ascending, -1 past the count */
        uint32_t w = 0;
        for (uint32_t i = 0; i < nr; i++) {
            if (!keep_r[i]) continue;
            if (w < k && out_g[(size_t)r * k + w] != (int32_t)i) bad_out++;
            w++;
        }
        for (uint32_t i = w; i < k; i++) if (out_g[(size_t)r * k + i] != -1) bad_out++;
    }
    const int pass = ok && !bad && !bad_out;
    printf("  [%s] %-18s %s  %u keep, %u list mismatches over %u rows, %.2f ms (n=%u k=%u ties~%.0f%%)\n",
           label, "radix select rows", pass ? "ok  " : "FAIL", bad, bad_out, rows, ms, n, k, 100.0 * tie_frac);
    free(out_g);
    ds4_gpu_tensor_free(t_sc); ds4_gpu_tensor_free(t_keep); ds4_gpu_tensor_free(t_out);
    ds4_gpu_tensor_free(t_state); ds4_gpu_tensor_free(t_hist);
    free(sc); free(keep_r); free(keep_g);
    return !pass;
}

int main(void) {
    if (!ds4_gpu_init()) {
        printf("dsv41 metal: no Metal device, skipping\n");
        return 0;
    }
    int fail = 0;
    fail |= run_shape("mini", 256u, 24u, 64u);
    fail |= run_shape("released", 5120u, 24u, 256u);
    /* mini: window 16 + index_topk 8 over 8 heads of 64; released: 128 + 512 over 64 of 512 */
    fail |= run_attn("mini", 24u, 8u, 64u, 24u, 40u, 0.25f);
    fail |= run_attn("released", 8u, 64u, 512u, 640u, 768u, 0.20f);
    /* decode is the shape that matters: one query, so occupancy comes from heads alone */
    fail |= run_attn("decode", 1u, 64u, 512u, 640u, 768u, 0.20f);
    fail |= run_attn("decode-k160", 1u, 64u, 512u, 160u, 768u, 0.20f);
    fail |= run_attn("decode-k40", 1u, 64u, 512u, 40u, 768u, 0.20f);
    fail |= run_attn("decode-h8", 1u, 8u, 512u, 640u, 768u, 0.20f);
    fail |= run_pool("mini", 12u, 2u, 64u);
    fail |= run_pool("released", 512u, 2u, 512u);
    fail |= run_index("mini", 4u, 32u, 12u, 0u);
    fail |= run_index("released", 32u, 128u, 8192u, 0u);
    /* long context: the candidate pool is fixed at 2048 blocks x 8 while clen grows */
    fail |= run_index("128k-ctx", 32u, 128u, 131072u, 16384u);
    fail |= run_rope("mini", 24u, 8u, 64u, 16u);
    fail |= run_rope("released", 8u, 64u, 512u, 64u);
    fail |= run_lora("mini", 128u, 32u, 4u, 0, 1u);
    fail |= run_lora("batch", 128u, 32u, 4u, 0, 5u);
    fail |= run_lora("released", 4096u, 1024u, 8u, 1, 1u);
    fail |= run_lora("batch", 4096u, 1024u, 8u, 1, 32u);
    fail |= run_lora("chunk", 4096u, 1024u, 8u, 1, 65u);
    fail |= run_lora("chunk", 4096u, 1024u, 8u, 1, 160u);
    fail |= run_select("mini", 24u, 8u, 2u, 20u, 0.0f);
    fail |= run_select("ties", 4096u, 512u, 8u, 4000u, 0.40f);
    fail |= run_select("released", 16384u, 512u, 8u, 16000u, 0.20f);
    fail |= run_hc("mini", 256u, 4u, 20u, 0, 24u);
    fail |= run_hc("released", 5120u, 4u, 20u, 1, 8u);
    fail |= run_hcends("mini", 256u, 4u, 1u);
    fail |= run_hcends("batch", 256u, 4u, 7u);
    fail |= run_hcends("released", 5120u, 4u, 1u);
    fail |= run_hcends("batch", 5120u, 4u, 64u);
    fail |= run_route("mini", 8u, 2u, 1.5f, 1u);
    fail |= run_route("batch", 8u, 2u, 1.5f, 9u);
    fail |= run_route("released", 384u, 6u, 1.5f, 1u);
    fail |= run_route("batch", 384u, 6u, 1.5f, 64u);
    fail |= run_offset_matmul("mini", 256u, 128u, 8u, 0u);
    fail |= run_rows_matmul("q8_0 64", 5120u, 1280u, 64u, 1);
    fail |= run_rows_matmul("q8_0 2", 5120u, 1280u, 2u, 1);
    fail |= run_rows_matmul("q8_0 5", 5120u, 1280u, 5u, 1);
    fail |= run_rows_matmul("q8_0 8", 4096u, 1001u, 8u, 1);
    fail |= run_rows_matmul("q8_0 5 wide", 1024u, 8192u, 5u, 1);
    fail |= run_rows_matmul("f16 2", 5120u, 384u, 2u, 0);
    fail |= run_rows_matmul("f16 5", 5120u, 384u, 5u, 0);
    fail |= run_rows_matmul("f16 8", 4104u, 1001u, 8u, 0);
    fail |= run_rows_matmul("q8_0 8 wide", 1024u, 4100u, 8u, 1);
    fail |= run_rows_matmul("q8_0 40", 5120u, 2304u, 40u, 1);
    fail |= run_rows_matmul("f16 64", 5120u, 384u, 64u, 0);
    fail |= run_rows_matmul("f16 9", 2304u, 512u, 9u, 0);
    fail |= run_offset_matmul("mini", 256u, 128u, 8u, 5u);
    /* every precision the released checkpoint stores an attention projection at:
     * q_a/q_b/kv/output_b are Q8_0, compressor and indexer F16, the mini model F32 */
    fail |= run_matvec("released", 5120u, 1280u, 8u);
    fail |= run_matvec("released", 1280u, 32768u, 8u);
    fail |= run_matvec("released", 5120u, 512u, 1u);
    fail |= run_matvec("released", 5120u, 32u, 1u);
    fail |= run_matvec("mini", 256u, 128u, 0u);
    /* the released checkpoint's MoE: MXFP4 experts, Q8_0 shared, top-6 of 384 */
    fail |= run_fused_moe("mini", 512u, 256u, 8u, 6u, 7.0f);
    fail |= run_fused_moe("released", 5120u, 2304u, 8u, 6u, 7.0f);
    /* the shared expert can only ride in the top-6 down step; any other width adds it */
    fail |= run_fused_moe("top4", 512u, 256u, 8u, 4u, 7.0f);
    /* the radix select is what lifts the O(n^2) pass's size cap; it has to match it
     * exactly, including where the scores are all equal */
    fail |= run_radix("small", 4096u, 512u, 0.0f);
    fail |= run_radix("ties", 4096u, 512u, 0.40f);
    fail |= run_radix("all-tied", 4096u, 512u, 1.0f);
    fail |= run_radix("released", 65536u, 512u, 0.20f);
    fail |= run_radix("1M-ctx", 1048576u, 512u, 0.30f);
    fail |= run_radix_rows("rows", 20000u, 512u, 0.30f, 3u);
    fail |= run_radix_rows("rows-64k", 65536u, 512u, 0.20f, 2u);
    /* the O(n^2) rank select must REFUSE sizes it cannot serve, not run them slowly:
     * block selection at long context (clen/8 blocks) lands here and needs a different
     * algorithm, so the guard has to be real */
    {
        const uint32_t big = DS4_GPU_DSV41_RANK_SELECT_MAX + 1u;
        ds4_gpu_tensor *a = ds4_gpu_tensor_alloc((uint64_t)big * sizeof(float));
        ds4_gpu_tensor *b = ds4_gpu_tensor_alloc((uint64_t)big * sizeof(int32_t));
        ds4_gpu_tensor *c = ds4_gpu_tensor_alloc(512u * sizeof(int32_t));
        const int refused = !ds4_gpu_dsv41_topk_select(big, 512u, 0u, big, a, b, c);
        printf("  [guard] %-18s %s  n=%u > %u refused\n", "topk size limit",
               refused ? "ok  " : "FAIL", big, DS4_GPU_DSV41_RANK_SELECT_MAX);
        if (!refused) fail = 1;
        ds4_gpu_tensor_free(a); ds4_gpu_tensor_free(b); ds4_gpu_tensor_free(c);
    }
    printf("dsv41 metal: %s\n", fail ? "FAILED" : "all checks passed");
    return fail;
}
