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
    return fail;
}
