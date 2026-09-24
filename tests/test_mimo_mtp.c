/* MiMo MTP session cycle against plain greedy decoding.
 *
 * Run with:
 *   DS4_TEST_MODEL=/path/to/MiMo.gguf make test-mimo-mtp
 * or, for the DFlash drafter instead of the embedded MTP chain:
 *   DS4_TEST_MODEL=... DS4_TEST_DFLASH=/path/to/DFlash.gguf make test-mimo-mtp
 *
 * The drafts of a synthetic model rarely match, so the accept paths run
 * with DS4_MIMO_SPEC_DRAFTS pointing at the plain continuation (every
 * draft accepted) and at a corrupted copy (mixed accept/reject).  Each run
 * must reproduce the plain tokens exactly, as must a continuation after a
 * rewind into the last verify block, live and from a checkpoint saved there;
 * sampled and exact-sampling cycles must run without error. */
#include "ds4.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define TEST_CTX 512
#define N_GEN 48
#define N_REWIND 12
#define N_REF (N_GEN + N_REWIND)

static void fail(const char *what) {
    fprintf(stderr, "FAIL: %s\n", what);
    exit(1);
}

static ds4_engine *open_engine(const char *model, bool exact) {
    const char *dflash = getenv("DS4_TEST_DFLASH");
    ds4_engine_options opt = {
        .model_path = model,
        .mtp_path = dflash && dflash[0] ? dflash : NULL,
        .backend = DS4_BACKEND_METAL,
        .n_threads = 1,
        .context_size = TEST_CTX,
        .glm_mtp = !(dflash && dflash[0]),
        .dspark_exact_sampling = exact,
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

/* greedy speculative decoding of n tokens into out; returns drafts accepted */
static int decode_spec(ds4_session *s, int n, int *out) {
    char err[256] = {0};
    int accepted[8], produced = 0, drafts = 0;
    int token = ds4_session_argmax(s);
    while (produced < n) {
        const int got = ds4_session_eval_speculative_argmax(s, token, n - produced, -1, accepted, 8,
                                                            err, sizeof(err));
        if (got <= 0) fail(err[0] ? err : "speculative cycle");
        for (int i = 0; i < got && produced < n; i++) out[produced++] = accepted[i];
        drafts += got - 1;
        token = ds4_session_argmax(s);
    }
    return drafts;
}

static void decode_sampled(ds4_session *s, int n, uint64_t *rng) {
    char err[256] = {0};
    int token = ds4_session_sample(s, 0.8f, 0, 0.95f, 0.0f, rng);
    for (int i = 0; i < n;) {
        int accepted[8];
        const int got = ds4_session_eval_speculative(s, token, n - i, -1, 0.8f, 0, 0.95f, 0.0f, rng,
                                                     accepted, 8, err, sizeof(err));
        if (got <= 0) fail(err[0] ? err : "sampled cycle");
        i += got;
        token = ds4_session_sample(s, 0.8f, 0, 0.95f, 0.0f, rng);
    }
}

static void expect_tokens(const int *got, const int *want, int n, const char *what) {
    for (int i = 0; i < n; i++) {
        if (got[i] != want[i]) {
            fprintf(stderr, "FAIL: %s differs at token %d: %d vs %d\n", what, i, got[i], want[i]);
            exit(1);
        }
    }
}

static void write_oracle(const char *path, const ds4_tokens *prompt, const int *gen, int n, int corrupt_every) {
    FILE *f = fopen(path, "w");
    if (!f) fail("oracle file");
    for (int i = 0; i < prompt->len; i++) fprintf(f, "%d\n", prompt->v[i]);
    for (int i = 0; i < n; i++) fprintf(f, "%d\n", corrupt_every && i % corrupt_every == 1 ? gen[i] + 1 : gen[i]);
    fclose(f);
}

int main(void) {
    const char *model = getenv("DS4_TEST_MODEL");
    if (!model || !model[0]) {
        fprintf(stderr, "SKIP: DS4_TEST_MODEL is not set\n");
        return 0;
    }
    ds4_engine *e = open_engine(model, false);
    char depth[8];   /* the oracle drafts cover the whole chain */
    snprintf(depth, sizeof(depth), "%d", ds4_engine_mtp_draft_tokens(e) - 1);
    if (!getenv("DS4_MIMO_MTP_DEPTH")) setenv("DS4_MIMO_MTP_DEPTH", depth, 1);
    ds4_tokens prompt = {0};
    ds4_tokenize_text(e, "The quick brown fox jumps over the lazy dog. Once upon a time", &prompt);
    char err[256] = {0};

    /* plain greedy reference */
    int ref[N_REF];
    ds4_session *s = session_at(e, &prompt);
    int token = ds4_session_argmax(s);
    for (int i = 0; i < N_REF; i++) {
        ref[i] = token;
        if (ds4_session_eval(s, token, err, sizeof(err)) != 0) fail(err);
        token = ds4_session_argmax(s);
    }
    ds4_session_free(s);

    /* KV checkpoint round trip: save after eight generated tokens, restore
     * into fresh sessions, continue plain and speculative; both must follow ref */
    int got[N_REF];
    {
        const int saved = 8;
        s = session_at(e, &prompt);
        for (int i = 0; i < saved; i++) {
            if (ds4_session_eval(s, ref[i], err, sizeof(err)) != 0) fail(err);
        }
        FILE *fp = tmpfile();
        if (!fp || ds4_session_save_payload(s, fp, err, sizeof(err)) != 0) fail(err[0] ? err : "checkpoint save");
        const long bytes = ftell(fp);
        /* DFlash restores its context exactly, so it must draft as if never saved */
        const char *dflash = getenv("DS4_TEST_DFLASH");
        const int want_drafts = dflash && dflash[0] ? decode_spec(s, N_GEN - saved, got) : -1;
        ds4_session_free(s);
        for (int spec = 0; spec < 2; spec++) {
            rewind(fp);
            if (ds4_session_create(&s, e, TEST_CTX) != 0) fail("session create");
            if (ds4_session_load_payload(s, fp, (uint64_t)bytes, err, sizeof(err)) != 0) fail(err);
            if (ds4_session_argmax(s) != ref[saved]) fail("checkpoint logits differ");
            if (spec) {
                const int drafts = decode_spec(s, N_GEN - saved, got);
                if (want_drafts >= 0 && drafts != want_drafts) fail("restored DFlash context drafts differently");
            } else {
                for (int i = saved; i < N_GEN; i++) {
                    got[i - saved] = ds4_session_argmax(s);
                    if (ds4_session_eval(s, got[i - saved], err, sizeof(err)) != 0) fail(err);
                }
            }
            expect_tokens(got, ref + saved, N_GEN - saved, spec ? "checkpoint + drafts" : "checkpoint");
            ds4_session_free(s);
        }
        fclose(fp);
        printf("checkpoint round trip: plain and speculative continuations match\n");
    }
    s = session_at(e, &prompt);
    int drafts = decode_spec(s, N_GEN, got);
    expect_tokens(got, ref, N_GEN, "own drafts");
    printf("own drafts: %d accepted\n", drafts);
    ds4_session_free(s);

    char oracle[2][64];
    for (int c = 0; c < 2; c++) {
        snprintf(oracle[c], sizeof(oracle[c]), "/tmp/ds4-mimo-mtp-%d-%d.txt", (int)getpid(), c);
        write_oracle(oracle[c], &prompt, ref, N_REF, c ? 3 : 0);
        setenv("DS4_MIMO_SPEC_DRAFTS", oracle[c], 1);
        s = session_at(e, &prompt);
        drafts = decode_spec(s, N_GEN, got);
        expect_tokens(got, ref, N_GEN, c ? "corrupted oracle drafts" : "oracle drafts");
        printf("%s: %d accepted\n", c ? "corrupted oracle drafts" : "oracle drafts", drafts);
        if (c == 0 && drafts < N_GEN / 2) fail("oracle drafts were not accepted");
        if (c == 1 && (drafts == 0 || drafts >= N_GEN - N_GEN / 3)) fail("corrupted drafts were not rejected");

        /* rewind into the last block (its last row is always inside), save
         * there as a server eviction does, and continue live and restored */
        const int pos = ds4_session_pos(s);
        ds4_session_rewind(s, pos - 1);
        if (ds4_session_pos(s) != pos - 1) fail("rewind position");
        FILE *fp = tmpfile();
        if (!fp || ds4_session_save_payload(s, fp, err, sizeof(err)) != 0) fail(err[0] ? err : "save after rewind");
        const long bytes = ftell(fp);
        drafts = decode_spec(s, N_REWIND + 1, got);
        expect_tokens(got, ref + N_GEN - 1, N_REWIND + 1, "rewound continuation");
        printf("rewound continuation: %d accepted\n", drafts);
        ds4_session_free(s);
        rewind(fp);
        if (ds4_session_create(&s, e, TEST_CTX) != 0) fail("session create");
        if (ds4_session_load_payload(s, fp, (uint64_t)bytes, err, sizeof(err)) != 0) fail(err);
        fclose(fp);
        decode_spec(s, N_REWIND + 1, got);
        expect_tokens(got, ref + N_GEN - 1, N_REWIND + 1, "restored rewound continuation");
        ds4_session_free(s);
        unlink(oracle[c]);
    }
    unsetenv("DS4_MIMO_SPEC_DRAFTS");

    uint64_t rng = 42;
    s = session_at(e, &prompt);
    decode_sampled(s, 24, &rng);
    ds4_session_free(s);
    ds4_engine_close(e);

    e = open_engine(model, true);
    s = session_at(e, &prompt);
    decode_sampled(s, 24, &rng);
    ds4_session_free(s);
    ds4_tokens_free(&prompt);
    ds4_engine_close(e);
    printf("OK\n");
    return 0;
}
