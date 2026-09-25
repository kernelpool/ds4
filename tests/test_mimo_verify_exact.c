/* MiMo speculative verify against plain decoding, bit for bit.
 *
 * Run with:
 *   DS4_TEST_MODEL=/path/to/MiMo.gguf make test-mimo-verify-exact
 *
 * A plain session decodes a greedy continuation and keeps the logits at
 * every position.  A second session replays it through speculative cycles
 * whose drafts are that continuation (DS4_MIMO_SPEC_DRAFTS), so every cycle
 * verifies a full block, and a third through drafts corrupted at an
 * irregular stride (rejections at every block offset).  After each cycle the session's logits must equal the plain
 * logits at that position exactly, at every draft depth. */
#include "ds4.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define TEST_CTX 4096
#define N_GEN 96

static void fail(const char *what) {
    fprintf(stderr, "FAIL: %s\n", what);
    exit(1);
}

static ds4_session *session_at(ds4_engine *e, const ds4_tokens *prompt) {
    ds4_session *s = NULL;
    char err[256] = {0};
    if (ds4_session_create(&s, e, TEST_CTX) != 0) fail("session create");
    if (ds4_session_sync(s, prompt, err, sizeof(err)) != 0) fail(err);
    return s;
}

int main(void) {
    const char *model = getenv("DS4_TEST_MODEL");
    if (!model || !model[0]) {
        fprintf(stderr, "SKIP: DS4_TEST_MODEL is not set\n");
        return 0;
    }
    ds4_engine_options opt = {
        .model_path = model,
        .backend = DS4_BACKEND_METAL,
        .n_threads = 1,
        .context_size = TEST_CTX,
        .glm_mtp = true,
    };
    ds4_engine *e = NULL;
    if (ds4_engine_open(&e, &opt) != 0) fail("engine open");
    const int V = ds4_engine_vocab_size(e);
    const char *text = getenv("DS4_TEST_PROMPT");
    ds4_tokens prompt = {0};
    ds4_tokenize_text(e, text && text[0] ? text :
                      "The history of the printing press begins in the fifteenth century, when", &prompt);
    char err[256] = {0};

    /* plain reference: logits at every position, greedy tokens */
    float *ref_logits = malloc((size_t)(N_GEN + 1) * V * sizeof(float));
    int ref[N_GEN + 1];
    ds4_session *s = session_at(e, &prompt);
    for (int i = 0; i <= N_GEN; i++) {
        if (ds4_session_copy_logits(s, ref_logits + (size_t)i * V, V) != V) fail("copy logits");
        ref[i] = ds4_session_argmax(s);
        if (i < N_GEN && ds4_session_eval(s, ref[i], err, sizeof(err)) != 0) fail(err);
    }
    ds4_session_free(s);

    float *got = malloc((size_t)V * sizeof(float));
    char oracle[64];
    int failures = 0;
    for (int depth = 1; depth <= ds4_engine_mtp_draft_tokens(e) - 1; depth++) {
        for (int corrupt = 0; corrupt < 2; corrupt++) {
            /* the drafter caches the oracle file by name */
            snprintf(oracle, sizeof(oracle), "/tmp/ds4-mimo-verify-%d-%d-%d.txt", (int)getpid(), depth, corrupt);
            FILE *f = fopen(oracle, "w");
            if (!f) fail("oracle file");
            for (int i = 0; i < prompt.len; i++) fprintf(f, "%d\n", prompt.v[i]);
            for (int i = 0; i <= N_GEN; i++) fprintf(f, "%d\n", corrupt && i * 7 % 11 == 3 ? ref[i] + 1 : ref[i]);
            fclose(f);
            char d[8];
            snprintf(d, sizeof(d), "%d", depth);
            setenv("DS4_MIMO_MTP_DEPTH", d, 1);
            setenv("DS4_MIMO_SPEC_DRAFTS", oracle, 1);
            s = session_at(e, &prompt);
            int produced = 0, cycles = 0, drafts = 0, bad = 0, first_bad = -1;
            float worst = 0.0f;
            int token = ds4_session_argmax(s);
            while (produced < N_GEN) {
                int accepted[8];
                const int n = ds4_session_eval_speculative_argmax(s, token, N_GEN - produced, -1, accepted, 8,
                                                                  err, sizeof(err));
                if (n <= 0) fail(err[0] ? err : "speculative cycle");
                for (int i = 0; i < n; i++) {
                    if (accepted[i] != ref[produced + i]) fail("accepted token differs from the plain one");
                }
                produced += n;
                cycles++;
                drafts += n - 1;
                if (ds4_session_copy_logits(s, got, V) != V) fail("copy logits");
                const float *want = ref_logits + (size_t)produced * V;
                if (memcmp(got, want, (size_t)V * sizeof(float)) != 0) {
                    float m = 0.0f;
                    for (int j = 0; j < V; j++) {
                        const float dd = got[j] > want[j] ? got[j] - want[j] : want[j] - got[j];
                        if (dd > m) m = dd;
                    }
                    if (m > worst) worst = m;
                    if (first_bad < 0) first_bad = produced;
                    bad++;
                }
                token = ds4_session_argmax(s);
            }
            printf("depth %d%s: %d cycles, %d drafts accepted, %d/%d cycles with differing logits "
                   "(first at +%d, worst |diff| %g)\n", depth, corrupt ? " corrupted" : "", cycles, drafts, bad,
                   cycles, first_bad, (double)worst);
            failures += bad != 0;
            ds4_session_free(s);
            unlink(oracle);
        }
    }
    unsetenv("DS4_MIMO_SPEC_DRAFTS");
    free(got);
    free(ref_logits);
    ds4_tokens_free(&prompt);
    ds4_engine_close(e);
    if (failures) fail("verify logits differ from plain decoding");
    printf("OK\n");
    return 0;
}
