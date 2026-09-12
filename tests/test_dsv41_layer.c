/* The V4.1 block composed from the Metal kernels, scored against the CPU reference on
 * identical inputs.
 *
 * Every kernel is already verified in isolation; what this checks is the composition --
 * that the pieces agree on layouts, offsets, RoPE positions and the index list when they
 * are chained.  The CPU side is dsv41_ref_attn_step / dsv41_ref_moe, themselves scored
 * stage by stage against the released implementation, so a match here reaches all the
 * way back.
 *
 * With no arguments it runs the F32 mini model.  Given the released checkpoint and its
 * engram sidecar it runs the same comparison at released precision, where the weights are
 * MXFP4 / Q8_0 / F16 and the layer topology is the real one:
 *
 *   tests/test_dsv41_layer <backbone.gguf> <engram.gguf>
 */

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int ds4_test_dsv41_attn_gpu(const char *, const int *, unsigned, const unsigned *, unsigned,
                            float *, float *, double *, const char *);
int ds4_test_dsv41_prefill(const char *, const int *, unsigned, float *, float *, float *,
                           float *, int32_t *, float *, const char *);
int ds4_test_dsv41_gpu_picks(const char *, const int *, unsigned, unsigned, unsigned, int32_t *);

#define MAX_CASES 10u

struct layer_case { unsigned il; const char *what; };

/* the mini oracle's prompt; its layer indices cover every V4.1 layer kind in miniature */
static const int MINI_TOKENS[] = {172, 47, 117, 192, 323, 251, 195, 359, 9, 211, 277, 242,
                                  292, 87, 70, 472, 88, 396, 314, 193, 486, 39, 87, 174};
static const struct layer_case MINI_CASES[] = {
    {0, "window only (ratio 0)"},
    {2, "ratio 2, kv+index source"},
    {3, "ratio 2, reads layer 2"},
    {4, "ratio 1, candidate source"},
    {5, "ratio 1, uses candidates"},
};

/* "<bos>The capital of France is", the prompt the CPU reference answers correctly */
static const int REAL_TOKENS[] = {0, 671, 6102, 294, 8760, 344};
static const struct layer_case REAL_CASES[] = {
    {0,  "window only (ratio 0)"},
    {1,  "window only, ENGRAM layer"},
    {2,  "ratio 2, kv+index source"},
    {3,  "ratio 2, reads layer 2"},
    {14, "ratio 2, kv+index source, ENGRAM layer"},
    {20, "ratio 1, kv+index+candidate source"},
    {21, "ratio 1, reads layer 20"},
    {24, "ratio 1, index source, uses candidates"},
};

int main(int argc, char **argv) {
    const char *gguf = argc > 1 ? argv[1] : "tests/deepseek_v41/mini/mini.gguf";
    const char *engram = argc > 2 ? argv[2] : NULL;
    const int real = argc > 1;
    const unsigned dim = real ? 5120u : 256u, hc = 4u, stream = dim * hc;

    const int *tokens = real ? REAL_TOKENS : MINI_TOKENS;
    const unsigned n = real ? (unsigned)(sizeof(REAL_TOKENS) / sizeof(int))
                            : (unsigned)(sizeof(MINI_TOKENS) / sizeof(int));
    const struct layer_case *cases = real ? REAL_CASES : MINI_CASES;
    const unsigned n_cases = real ? (unsigned)(sizeof(REAL_CASES) / sizeof(REAL_CASES[0]))
                                  : (unsigned)(sizeof(MINI_CASES) / sizeof(MINI_CASES[0]));

    unsigned ils[MAX_CASES];
    for (unsigned c = 0; c < n_cases; c++) ils[c] = cases[c].il;
    float *cpu_all = calloc((size_t)MAX_CASES * stream, sizeof(float));
    float *gpu_all = calloc((size_t)MAX_CASES * stream, sizeof(float));
    double errs[MAX_CASES];
    for (unsigned c = 0; c < n_cases; c++) errs[c] = -1.0;
    const int ran = ds4_test_dsv41_attn_gpu(gguf, tokens, n, ils, n_cases,
                                            cpu_all, gpu_all, errs, engram);
    int fail = !ran;
    printf("V4.1 FULL BLOCK (attention sublayer + MoE sublayer, both mHC-wrapped)\n"
           "Metal composition vs the CPU reference, comparing the full hc stream\n"
           "  weights: %s\n", real ? "released (mxfp4 experts, q8_0 attention, f16 mHC)"
                                   : "mini (f32 throughout)");
    /* the same bar at both precisions: MXFP4 and Q8_0 decode to exactly the same values
     * on either side, so all that separates them is summation order */
    const double tol = 5e-5;
    for (unsigned c = 0; c < n_cases; c++) {
        const float *cpu = cpu_all + (size_t)c * stream;
        const float *gpu = gpu_all + (size_t)c * stream;
        const double state_err = errs[c];
        double worst = 0.0, scale = 0.0;
        const unsigned ncmp = getenv("DS4_DSV41_MOE_ONLY") ? dim : stream;
        for (unsigned i = 0; i < ncmp; i++) {
            const double d = fabs((double)gpu[i] - cpu[i]);
            if (d > worst) worst = d;
            if (fabs((double)cpu[i]) > scale) scale = fabs((double)cpu[i]);
        }
        const double rel = worst / (scale > 0.0 ? scale : 1.0);
        const int state_ok = state_err >= 0.0 && state_err <= 1e-5;
        const int ok = rel <= tol && state_ok;
        if (!ok) fail = 1;
        printf("  layer%-2u %s  rel=%.3e  caches match=%s (err %.3e)   %s\n",
               cases[c].il, ok ? "ok  " : "FAIL", rel,
               state_ok ? "yes" : "NO", state_err, cases[c].what);
    }
    printf("v4.1 layer composition: %s\n", fail ? "FAILED" : "all layer kinds match");
    free(cpu_all); free(gpu_all);

    /* The indexer's picks over a CHUNKED prefill against the CPU reference's per-position
     * selection, for every index-source layer of the mini.  The layer harness above runs
     * one position at a time, so it cannot see a query being masked with another query's
     * candidate blocks; this can.  Ids are compared as compressed positions, and the two
     * sides pad differently past a query's reach, so only the reachable prefix is held. */
    if (!real) {
        enum { NL = 6, K = 8 };
        static const unsigned index_layers[] = { 2, 4, 5 };
        static const unsigned chunks[] = { 24, 7, 1 };
        int32_t *ref = calloc((size_t)NL * n * K, sizeof(int32_t));
        int32_t *got = calloc((size_t)n * K, sizeof(int32_t));
        const int nl = ds4_test_dsv41_prefill(gguf, tokens, n, NULL, NULL, NULL, NULL,
                                              ref, NULL, NULL);
        int pfail = nl != NL;
        for (unsigned ci = 0; ci < sizeof(chunks) / sizeof(chunks[0]) && !pfail; ci++) {
            for (unsigned li = 0; li < sizeof(index_layers) / sizeof(index_layers[0]); li++) {
                const unsigned il = index_layers[li];
                unsigned bad = 0, compared = 0;
                if (!ds4_test_dsv41_gpu_picks(gguf, tokens, n, chunks[ci], il, got)) { pfail = 1; break; }
                for (unsigned t = 0; t < n; t++) {
                    const int32_t *r = ref + ((size_t)il * n + t) * K, *g = got + (size_t)t * K;
                    unsigned nr = 0, ng = 0;
                    int32_t rv[K], gv[K];
                    for (unsigned j = 0; j < K; j++) {
                        if (r[j] >= 0) rv[nr++] = r[j] - (int32_t)n;   /* joined-KV offset */
                        if (g[j] >= 0) gv[ng++] = g[j];
                    }
                    compared += nr;
                    if (nr != ng || memcmp(rv, gv, nr * sizeof(int32_t)) != 0) bad++;
                }
                if (bad) pfail = 1;
                printf("  picks layer%u chunk %-2u %s  %u/%u positions agree with the CPU reference (%u ids)\n",
                       il, chunks[ci], bad ? "FAIL" : "ok  ", n - bad, n, compared);
            }
        }
        free(ref); free(got);
        printf("v4.1 chunked selection: %s\n", pfail ? "FAILED" : "every query selects with its own reach");
        if (pfail) fail = 1;
    }
    return fail;
}
