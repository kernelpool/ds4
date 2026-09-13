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
 *   tests/test_dsv41_layer <backbone.gguf> <engram.gguf> [<dspark.gguf>]
 *
 * With the DSpark heads sidecar as well, a draft block from the GPU is held to the CPU
 * reference's: the ring rows it keeps, the head logits, the drafts and the confidences.
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
int ds4_test_dsv41_gpu_rollback(const char *, const int *, unsigned, unsigned, unsigned, float *, float *);
int ds4_test_dsv41_gpu_payload(const char *, const char *, const int *, unsigned, unsigned, unsigned,
                               unsigned, unsigned, int *, int *, float *, float *);
unsigned ds4_test_dsv41_gpu_draft(const char *, const char *, const char *, const int *, unsigned,
                                  int *, float *, float *, int *, float *, float *, float *);

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
    const char *dspark = argc > 3 ? argv[3] : NULL;
    const int real = argc > 1;
    if (!real) {
        FILE *probe = fopen(gguf, "rb");
        if (!probe) {
            fprintf(stderr, "%s: the mini fixture is generated, not tracked; build it with "
                    "tests/deepseek_v41/make_mini_model.py and make_mini_gguf.py "
                    "(see docs/MODELS.md), or pass the released checkpoint\n", gguf);
            return 2;
        }
        fclose(probe);
    }
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
                    if (nr != ng || memcmp(rv, gv, nr * sizeof(int32_t)) != 0) {
                        bad++;
                        if (getenv("DS4_TEST_PICKS_VERBOSE")) {
                            printf("    layer%u chunk %u row %u: cpu", il, chunks[ci], t);
                            for (unsigned j = 0; j < nr; j++) printf(" %d", rv[j]);
                            printf(" | gpu");
                            for (unsigned j = 0; j < ng; j++) printf(" %d", gv[j]);
                            printf("\n");
                        }
                    }
                }
                if (bad) pfail = 1;
                printf("  picks layer%u chunk %-2u %s  %u/%u positions agree with the CPU reference (%u ids)\n",
                       il, chunks[ci], bad ? "FAIL" : "ok  ", n - bad, n, compared);
            }
        }
        free(ref); free(got);
        printf("v4.1 chunked selection: %s\n", pfail ? "FAILED" : "every query selects with its own reach");
        if (pfail) fail = 1;

        /* A verify-style pass of k rows rolled back to its first row must leave the state a
         * one-token pass would have left.  Passes of 2..8 rows differ from one-row passes in
         * summation order, so the bar is F32 noise rather than bit equality. */
        const unsigned splits[][2] = { {8, 3}, {n - 6, 5}, {n - 8, 6}, {n - 3, 2}, {n - 1, 1} };
        const unsigned vocab = 512u;
        float *ra = malloc(vocab * sizeof(float)), *rb = malloc(vocab * sizeof(float));
        int rfail = 0;
        for (unsigned i = 0; i < sizeof(splits) / sizeof(splits[0]); i++) {
            const unsigned split = splits[i][0], k = splits[i][1];
            double worst = 0.0, scale = 0.0;
            const int ok = ds4_test_dsv41_gpu_rollback(gguf, tokens, n, split, k, ra, rb) == 0;
            for (unsigned v = 0; ok && v < vocab; v++) {
                worst = fmax(worst, fabs((double)ra[v] - rb[v]));
                scale = fmax(scale, fabs((double)rb[v]));
            }
            const double rel = ok ? worst / (scale > 0.0 ? scale : 1.0) : 1.0;
            const int pass = ok && rel <= 1e-6;
            if (!pass) rfail = 1;
            printf("  rollback split %-2u rows %u %s  rel=%.3e\n", split, k, pass ? "ok  " : "FAIL", rel);
        }
        free(ra); free(rb);
        printf("v4.1 speculative rollback: %s\n", rfail ? "FAILED" : "a rejected tail leaves no trace");
        if (rfail) fail = 1;

        /* A checkpoint written mid-prompt and read into a window of another width must
         * continue exactly as the original state does. */
        int cfail = 0;
        const unsigned csplits[] = { 5, 11, n - 2 };
        for (unsigned i = 0; i < sizeof(csplits) / sizeof(csplits[0]); i++) {
            enum { NP = 4 };
            int ga[NP], gb[NP];
            float *la = malloc(vocab * sizeof(float)), *lb = malloc(vocab * sizeof(float));
            const int ok = ds4_test_dsv41_gpu_payload(gguf, engram, tokens, n, csplits[i], NP, 7u, 9u,
                                                      ga, gb, la, lb) == 0;
            double worst = 0.0;
            int same = ok;
            for (unsigned v = 0; ok && v < vocab; v++) worst = fmax(worst, fabs((double)la[v] - lb[v]));
            for (unsigned j = 0; ok && j < NP; j++) if (ga[j] != gb[j]) same = 0;
            const int pass = ok && same && worst == 0.0;
            if (!pass) cfail = 1;
            printf("  checkpoint split %-2u %s  tokens %s, max|d|=%.3e\n", csplits[i], pass ? "ok  " : "FAIL",
                   same ? "identical" : "DIFFER", worst);
            free(la); free(lb);
        }
        printf("v4.1 checkpoint round trip: %s\n", cfail ? "FAILED" : "a restored state continues bit for bit");
        if (cfail) fail = 1;
    }

    if (real && dspark) {
        enum { MAXB = 16 };
        const unsigned vocab = 129280u;
        int tg[MAXB], tc[MAXB];
        float cg[MAXB], cc[MAXB], kv_err[8];
        float *lg = malloc((size_t)MAXB * vocab * sizeof(float));
        float *lc = malloc((size_t)MAXB * vocab * sizeof(float));
        const unsigned B = ds4_test_dsv41_gpu_draft(gguf, engram, dspark, tokens, n, tg, cg, lg, tc, cc, lc, kv_err);
        int dfail = B == 0;
        printf("V4.1 DSPARK DRAFT (3 stages, head, Markov chain, confidence) vs the CPU reference\n");
        for (unsigned s = 0; s < 3 && !dfail; s++) {
            const int ok = kv_err[s] <= 5e-5f;
            if (!ok) dfail = 1;
            printf("  stage %u main_kv ring row %s  rel=%.3e\n", s, ok ? "ok  " : "FAIL", (double)kv_err[s]);
        }
        for (unsigned i = 0; i < B; i++) {
            double worst = 0.0, scale = 0.0;
            for (unsigned v = 0; v < vocab; v++) {
                worst = fmax(worst, fabs((double)lg[(size_t)i * vocab + v] - lc[(size_t)i * vocab + v]));
                scale = fmax(scale, fabs((double)lc[(size_t)i * vocab + v]));
            }
            const double rel = worst / (scale > 0.0 ? scale : 1.0);
            const double cd = fabs((double)cg[i] - cc[i]);
            const int ok = rel <= 5e-5 && tg[i] == tc[i] && cd <= 1e-3 * fmax(1.0, fabs((double)cc[i]));
            if (!ok) dfail = 1;
            printf("  position %u %s  logits rel=%.3e  draft gpu=%d cpu=%d  confidence gpu=%.4f cpu=%.4f\n",
                   i, ok ? "ok  " : "FAIL", rel, tg[i], tc[i], (double)cg[i], (double)cc[i]);
        }
        free(lg); free(lc);
        printf("v4.1 dspark draft: %s\n", dfail ? "FAILED" : "the GPU drafts what the reference drafts");
        if (dfail) fail = 1;
    }
    return fail;
}
