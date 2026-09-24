/* MiMo prefill attention on random data: the default kernel must equal the
 * per-row kernel bit for bit, rows split over two calls must give the same
 * bits as one call, and the error against an fp64 reference must not exceed
 * the per-row kernel's.  DS4_MIMO_ATTN=tiled checks the tiled kernel (not
 * bit-identical to the per-row one) the same way. */
#define _DARWIN_C_SOURCE
#include "ds4_gpu.h"

#include <dispatch/dispatch.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

enum { DK = 192, DV = 128, MAX_HEADS = 64 };
typedef struct { uint32_t H, Hkv, T, pos0, ring, n_swa, first, sink; } attn_case;
static const attn_case cases[] = {
    {64, 4, 300, 0, 8192, 0, 0, 0},    {64, 4, 333, 700, 8192, 0, 0, 0},
    {64, 8, 300, 1000, 640, 128, 0, 1}, {64, 8, 50, 0, 640, 128, 0, 1},
    {64, 4, 100, 20, 8192, 0, 5, 0},   {64, 4, 257, 3000, 4224, 128, 0, 1},
    {64, 4, 1024, 6000, 16384, 0, 0, 0}, {64, 8, 300, 20000, 640, 128, 0, 1},
};

static uint64_t rng = 0x9E3779B97F4A7C15ull;
static double nrand(void) {
    double u[2];
    for (int i = 0; i < 2; i++) {
        rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
        u[i] = (double)(rng >> 11) / 9007199254740992.0;
    }
    return sqrt(-2.0 * log(u[0] + 1e-12)) * cos(6.283185307179586 * u[1]);
}
static uint16_t f2h(float f) { __fp16 h = (__fp16)f; uint16_t u; memcpy(&u, &h, 2); return u; }
static double h2d(uint16_t u) { __fp16 h; memcpy(&h, &u, 2); return (double)h; }

/* rows [r0, r0 + n) of the case in one call */
static float *attn(const attn_case *c, const float *q, ds4_gpu_tensor *tk, ds4_gpu_tensor *tv,
                   const float *sinks, uint64_t map_size, uint32_t r0, uint32_t n) {
    const uint32_t H = c->H;
    ds4_gpu_tensor *tq = ds4_gpu_tensor_alloc((uint64_t)n * H * DK * 4);
    ds4_gpu_tensor *to = ds4_gpu_tensor_alloc((uint64_t)n * H * DV * 4);
    float *o = malloc((size_t)n * H * DV * sizeof(float));
    ds4_gpu_tensor_write(tq, 0, q + (size_t)r0 * H * DK, (uint64_t)n * H * DK * 4);
    const int ok = ds4_gpu_mimo_attn_tensor(to, tq, tk, tv, sinks, map_size, 0, c->sink != 0, NULL, n, H, c->Hkv, DK, DV,
                                            c->pos0 + r0, c->ring, c->n_swa, c->first, 0, 1.0f / sqrtf((float)DK),
                                            NULL, 64, 10000.0f, 1.0f);
    if (ok) ds4_gpu_tensor_read(to, 0, o, (uint64_t)n * H * DV * 4);
    ds4_gpu_tensor_free(tq); ds4_gpu_tensor_free(to);
    if (!ok) { free(o); return NULL; }
    return o;
}

static int check_case(const attn_case *c, const float *sinks, uint64_t map_size, int exact,
                      double *max_err, double *mean_err) {
    const uint32_t H = c->H, Hkv = c->Hkv, T = c->T, G = H / Hkv, npos = c->pos0 + T;
    const float scale = 1.0f / sqrtf((float)DK);
    float *q = malloc((size_t)T * H * DK * sizeof(float));
    uint16_t *kall = malloc((size_t)npos * Hkv * DK * 2), *vall = malloc((size_t)npos * Hkv * DV * 2);
    uint16_t *kc = calloc((size_t)c->ring * Hkv * DK, 2), *vc = calloc((size_t)c->ring * Hkv * DV, 2);
    for (size_t i = 0; i < (size_t)T * H * DK; i++) q[i] = (float)nrand();
    for (size_t i = 0; i < (size_t)npos * Hkv * DK; i++) kall[i] = f2h((float)nrand());
    for (size_t i = 0; i < (size_t)npos * Hkv * DV; i++) vall[i] = f2h((float)nrand());
    /* every position writes its ring row; later positions overwrite earlier ones */
    for (uint32_t p = 0; p < npos; p++) {
        memcpy(kc + (size_t)(p % c->ring) * Hkv * DK, kall + (size_t)p * Hkv * DK, (size_t)Hkv * DK * 2);
        memcpy(vc + (size_t)(p % c->ring) * Hkv * DV, vall + (size_t)p * Hkv * DV, (size_t)Hkv * DV * 2);
    }
    ds4_gpu_tensor *tk = ds4_gpu_tensor_alloc((uint64_t)c->ring * Hkv * DK * 2);
    ds4_gpu_tensor *tv = ds4_gpu_tensor_alloc((uint64_t)c->ring * Hkv * DV * 2);
    ds4_gpu_tensor_write(tk, 0, kc, (uint64_t)c->ring * Hkv * DK * 2);
    ds4_gpu_tensor_write(tv, 0, vc, (uint64_t)c->ring * Hkv * DV * 2);
    float *o = attn(c, q, tk, tv, sinks, map_size, 0, T);
    int ok = o != NULL;
    /* split at an odd row: the second call starts mid-tile */
    const uint32_t a = T > 2 ? (T / 2) | 1u : 0u;
    if (ok && a) {
        float *o1 = attn(c, q, tk, tv, sinks, map_size, 0, a), *o2 = attn(c, q, tk, tv, sinks, map_size, a, T - a);
        ok = o1 && o2;
        if (ok && (memcmp(o1, o, (size_t)a * H * DV * 4) || memcmp(o2, o + (size_t)a * H * DV, (size_t)(T - a) * H * DV * 4))) {
            fprintf(stderr, "rows split at %u differ from one call\n", a);
            ok = 0;
        }
        free(o1); free(o2);
    }
    if (ok && exact) {
        setenv("DS4_MIMO_ATTN", "rowwise", 1);
        float *ref = attn(c, q, tk, tv, sinks, map_size, 0, T);
        unsetenv("DS4_MIMO_ATTN");
        if (!ref || memcmp(ref, o, (size_t)T * H * DV * 4)) {
            fprintf(stderr, "output differs from the per-row kernel\n");
            ok = 0;
        }
        free(ref);
    }

    double *tmax = calloc(T, sizeof(double)), *tsum = calloc(T, sizeof(double));
    if (ok) dispatch_apply(T, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t t) {
        const uint32_t pos = c->pos0 + (uint32_t)t;
        uint32_t lo = c->n_swa && pos + 1 > c->n_swa ? pos + 1 - c->n_swa : 0;
        if (lo < c->first) lo = c->first;
        double *s = malloc((size_t)(pos + 1) * sizeof(double)), acc[DV];
        for (uint32_t h = 0; h < H; h++) {
            const float *qh = q + ((size_t)t * H + h) * DK;
            double m = c->sink ? (double)sinks[h] : -1e300, l = 0.0;
            for (uint32_t p = lo; p <= pos; p++) {
                const uint16_t *kr = kall + ((size_t)p * Hkv + h / G) * DK;
                double d = 0.0;
                for (uint32_t i = 0; i < DK; i++) d += (double)(qh[i] * scale) * h2d(kr[i]);
                s[p] = d;
                if (d > m) m = d;
            }
            if (c->sink) l = exp((double)sinks[h] - m);
            for (uint32_t i = 0; i < DV; i++) acc[i] = 0.0;
            for (uint32_t p = lo; p <= pos; p++) {
                const double w = exp(s[p] - m);
                const uint16_t *vr = vall + ((size_t)p * Hkv + h / G) * DV;
                l += w;
                for (uint32_t i = 0; i < DV; i++) acc[i] += w * h2d(vr[i]);
            }
            const float *oh = o + ((size_t)t * H + h) * DV;
            for (uint32_t i = 0; i < DV; i++) {
                const double e = fabs(acc[i] / l - (double)oh[i]);
                if (e > tmax[t]) tmax[t] = e;
                tsum[t] += e;
            }
        }
        free(s);
    });
    *max_err = 0.0;
    double sum = 0.0;
    for (uint32_t t = 0; t < T; t++) {
        if (tmax[t] > *max_err) *max_err = tmax[t];
        sum += tsum[t];
    }
    *mean_err = sum / ((double)T * H * DV);
    ds4_gpu_tensor_free(tk); ds4_gpu_tensor_free(tv);
    free(q); free(o); free(kall); free(vall); free(kc); free(vc); free(tmax); free(tsum);
    return ok;
}

int main(void) {
    const char *mode = getenv("DS4_MIMO_ATTN");
    const int exact = mode == NULL;
    if (!ds4_gpu_init()) return 1;
    /* sinks live in a page-aligned buffer registered as the model map */
    const uint64_t page = getpagesize();
    float *sinks = NULL;
    if (posix_memalign((void **)&sinks, page, page)) return 1;
    memset(sinks, 0, page);
    for (uint32_t h = 0; h < MAX_HEADS; h++) sinks[h] = (float)(nrand() * 2.0);
    if (!ds4_gpu_set_model_map_range(sinks, page, 0, page, page)) return 1;

    double worst = 0.0, mean = 0.0;
    const int n = (int)(sizeof(cases) / sizeof(cases[0]));
    for (int i = 0; i < n; i++) {
        const attn_case *c = &cases[i];
        double mx, mn;
        if (!check_case(c, sinks, page, exact, &mx, &mn)) {
            fprintf(stderr, "case %d: FAILED\n", i);
            return 1;
        }
        printf("H=%u Hkv=%u T=%u pos0=%u ring=%u swa=%u first=%u sink=%u: max|err| %.3e mean|err| %.3e\n",
               c->H, c->Hkv, c->T, c->pos0, c->ring, c->n_swa, c->first, c->sink, mx, mn);
        if (mx > worst) worst = mx;
        mean += mn / n;
    }
    /* the per-row kernel's errors on these cases */
    const double max_limit = 1.40e-6, mean_limit = 2.94e-8;
    const int pass = worst <= max_limit && mean <= mean_limit;
    printf("worst max|err| %.3e (limit %.3e), average mean|err| %.3e (limit %.3e): %s\n",
           worst, max_limit, mean, mean_limit, pass ? "ok" : "FAILED");
    return pass ? 0 : 1;
}
