/* Qwen3.8 speculative verify against plain decoding, bit for bit.
 *
 * Run with:
 *   DS4_TEST_MODEL=/path/to/Qwen3.8.gguf make test-qwen4-verify-exact
 *
 * An engine without MTP decodes a greedy continuation and keeps the logits at
 * every position.  An engine with MTP then decodes the same tokens one at a
 * time, and through speculative cycles at draft depth 2 and 3; after every
 * step its logits must equal the plain logits at that position exactly.  The
 * last prompt is long enough for the sparse attention path. */
#include "ds4.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TEST_CTX 4096
#define N_GEN 128

static const char *prompts[] = {
    "The history of the printing press begins in the fifteenth century, when",
    "def merge_intervals(intervals):\n    \"\"\"Merge overlapping [start, end] pairs.\"\"\"\n",
    NULL,   /* long: built in main */
};
#define N_PROMPTS (int)(sizeof(prompts) / sizeof(prompts[0]))
#define LONG_REPEAT 64

static const char *long_part =
    "Section %d. The harbor records list each ship by name, cargo, tonnage and port of origin, "
    "together with the fees paid on arrival and the date the vessel left again. ";

static void fail(const char *what) {
    fprintf(stderr, "FAIL: %s\n", what);
    exit(1);
}

static ds4_engine *engine_open(const char *model, bool mtp) {
    ds4_engine_options opt = {
        .model_path = model,
        .backend = DS4_BACKEND_METAL,
        .n_threads = 1,
        .context_size = TEST_CTX,
        .glm_mtp = mtp,
    };
    ds4_engine *e = NULL;
    if (ds4_engine_open(&e, &opt) != 0) fail("engine open");
    return e;
}

static ds4_session *session_at(ds4_engine *e, const ds4_tokens *prompt) {
    ds4_session *s = NULL;
    char err[256] = {0};
    if (ds4_session_create(&s, e, TEST_CTX) != 0) fail("session create");
    if (ds4_session_sync(s, prompt, err, sizeof(err)) != 0) fail(err);
    return s;
}

/* 1 when the session's logits differ from want, tracking the worst |diff| */
static int differs(ds4_session *s, const float *want, float *got, int V, float *worst) {
    if (ds4_session_copy_logits(s, got, V) != V) fail("copy logits");
    if (memcmp(got, want, (size_t)V * sizeof(float)) == 0) return 0;
    for (int j = 0; j < V; j++) {
        const float d = got[j] > want[j] ? got[j] - want[j] : want[j] - got[j];
        if (d > *worst) *worst = d;
    }
    return 1;
}

int main(void) {
    const char *model = getenv("DS4_TEST_MODEL");
    if (!model || !model[0]) {
        fprintf(stderr, "SKIP: DS4_TEST_MODEL is not set\n");
        return 0;
    }
    char err[256] = {0};
    const int n_ref = N_GEN + 3;   /* a last cycle may run two tokens past N_GEN */
    ds4_tokens prompt[N_PROMPTS] = {{0}};
    int (*ref)[N_GEN + 3] = calloc(N_PROMPTS, sizeof(*ref));
    float *ref_logits = NULL;
    int V = 0;

    /* plain reference from an engine without the predictor */
    ds4_engine *e = engine_open(model, false);
    V = ds4_engine_vocab_size(e);
    ref_logits = malloc((size_t)N_PROMPTS * n_ref * V * sizeof(float));
    if (!ref || !ref_logits) fail("alloc");
    char long_text[LONG_REPEAT * 256];
    size_t len = 0;
    for (int i = 0; i < LONG_REPEAT; i++) len += snprintf(long_text + len, sizeof(long_text) - len, long_part, i + 1);
    for (int p = 0; p < N_PROMPTS; p++) {
        ds4_tokenize_text(e, prompts[p] ? prompts[p] : long_text, &prompt[p]);
        ds4_session *s = session_at(e, &prompt[p]);
        for (int i = 0; i < n_ref; i++) {
            float *row = ref_logits + ((size_t)p * n_ref + i) * V;
            if (ds4_session_copy_logits(s, row, V) != V) fail("copy logits");
            ref[p][i] = ds4_session_argmax(s);
            if (i + 1 < n_ref && ds4_session_eval(s, ref[p][i], err, sizeof(err)) != 0) fail(err);
        }
        ds4_session_free(s);
    }
    ds4_engine_close(e);

    e = engine_open(model, true);
    float *got = malloc((size_t)V * sizeof(float));
    int failures = 0;
    for (int p = 0; p < N_PROMPTS; p++) {
        const float *want = ref_logits + (size_t)p * n_ref * V;
        /* one token at a time with the predictor loaded */
        ds4_session *s = session_at(e, &prompt[p]);
        int bad = 0;
        float worst = 0.0f;
        for (int i = 0; i < N_GEN; i++) {
            bad += differs(s, want + (size_t)i * V, got, V, &worst);
            if (ds4_session_eval(s, ref[p][i], err, sizeof(err)) != 0) fail(err);
        }
        printf("prompt %d (%d tokens) plain: %d/%d positions with differing logits (worst |diff| %g)\n",
               p, prompt[p].len, bad, N_GEN, (double)worst);
        failures += bad != 0;
        ds4_session_free(s);

        for (int depth = 2; depth <= 3; depth++) {
            setenv("DS4_QWEN4_MTP_DEPTH", depth == 2 ? "2" : "3", 1);
            s = session_at(e, &prompt[p]);
            int produced = 0, cycles = 0, drafts = 0, first_bad = -1;
            bad = 0;
            worst = 0.0f;
            int token = ds4_session_argmax(s);
            while (produced < N_GEN) {
                int accepted[4];
                const int n = ds4_session_eval_speculative_argmax(s, token, N_GEN - produced, -1, accepted, 4,
                                                                  err, sizeof(err));
                if (n <= 0) fail(err[0] ? err : "speculative cycle");
                int diverged = 0;
                for (int i = 0; i < n; i++) diverged |= accepted[i] != ref[p][produced + i];
                if (diverged) {
                    printf("prompt %d depth %d: accepted token differs from plain at +%d\n", p, depth, produced);
                    bad++;
                    break;
                }
                produced += n;
                cycles++;
                drafts += n - 1;
                if (differs(s, want + (size_t)produced * V, got, V, &worst)) {
                    if (first_bad < 0) first_bad = produced;
                    bad++;
                }
                token = ds4_session_argmax(s);
            }
            printf("prompt %d depth %d: %d cycles, %d drafts accepted, %d cycles with differing logits "
                   "(first at +%d, worst |diff| %g)\n", p, depth, cycles, drafts, bad, first_bad, (double)worst);
            failures += bad != 0;
            ds4_session_free(s);
        }
    }
    unsetenv("DS4_QWEN4_MTP_DEPTH");
    free(got);
    free(ref_logits);
    free(ref);
    for (int p = 0; p < N_PROMPTS; p++) ds4_tokens_free(&prompt[p]);
    ds4_engine_close(e);
    if (failures) fail("MTP logits differ from plain decoding");
    printf("OK\n");
    return 0;
}
