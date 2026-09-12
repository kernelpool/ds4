/* DeepSeek V4.1 Flash CPU reference tests.
 *
 * Vectors in test_dsv41_ref_vectors.inc come from tests/deepseek_v41/gen_ref_vectors.py, which
 * runs the maths of the released reference implementation (inference/model.py + kernel.py).
 * CPU only, so it runs while the GPU is busy.
 */
#include <math.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "test_dsv41_ref_vectors.inc"

void ds4_test_dsv41_hc_split_sinkhorn(const float *, const float *, const float *, unsigned, unsigned,
                                      float, float *, float *, float *);
void ds4_test_dsv41_route(const float *, const float *, unsigned, unsigned, float, int, int32_t *, float *);
void ds4_test_dsv41_sparse_attn(const float *, const float *, const float *, const int32_t *,
                                unsigned, unsigned, unsigned, unsigned, float, float *);
void ds4_test_dsv41_compress_pool(const float *, const float *, unsigned, unsigned, unsigned, float *);
void ds4_test_dsv41_engram_hash(const int32_t *, const int32_t *, unsigned, unsigned, unsigned,
                                int32_t, const int64_t *, const int64_t *, const int64_t *, int64_t *);
void ds4_test_dsv41_engram_gate(const float *, const float *, const float *, const float *,
                                const float *, unsigned, unsigned, float, float *, float *);
void ds4_test_dsv41_candidate_blocks(const float *, unsigned, unsigned, unsigned, unsigned, int32_t *);
void ds4_test_dsv41_source_layer_map(const uint32_t *, unsigned, unsigned, uint32_t *);
void ds4_test_dsv41_shape(uint32_t *);
void ds4_test_e4m3_dequant_row(const void *, uint64_t, float *);
uint32_t ds4_test_e4m3_type(void);
int ds4_test_tensor_nbytes(uint32_t, uint64_t, uint64_t *);
const char *ds4_test_tensor_type_name(uint32_t);
int ds4_test_engram_open(const char *, const int *, unsigned, unsigned,
                         int64_t *, float *, uint64_t *);
int ds4_test_engram_forward(const char *, const int *, unsigned, unsigned,
                            const float *, float *, float *, int);

static int g_fail = 0;

static void check(const char *what, const float *got, const float *want, size_t n, float tol) {
    double worst = 0.0;
    size_t at = 0;
    for (size_t i = 0; i < n; i++) {
        const double d = fabs((double)got[i] - (double)want[i]);
        if (d > worst) { worst = d; at = i; }
    }
    const int ok = worst <= tol;
    if (!ok) g_fail++;
    printf("  %-26s %s  max|d|=%.3e (tol %.0e)%s\n", what, ok ? "ok  " : "FAIL", worst, (double)tol,
           ok ? "" : " <-- mismatch");
    if (!ok) printf("      first worst at %zu: got %.9g want %.9g\n", at, (double)got[at], (double)want[at]);
}

static void check_i(const char *what, const int64_t *got, const int64_t *want, size_t n) {
    size_t bad = 0;
    for (size_t i = 0; i < n; i++) if (got[i] != want[i]) bad++;
    if (bad) g_fail++;
    printf("  %-26s %s  %zu/%zu exact\n", what, bad ? "FAIL" : "ok  ", n - bad, n);
}

int main(void) {
    printf("DeepSeek V4.1 CPU reference vs the released implementation\n");

    /* 1. mHC split + Sinkhorn */
    {
        float pre[V_HC_N * V_HC], post[V_HC_N * V_HC], comb[V_HC_N * V_HC * V_HC];
        for (unsigned n = 0; n < V_HC_N; n++) {
            ds4_test_dsv41_hc_split_sinkhorn(v_hc_mixes + (size_t)n * (2 + V_HC) * V_HC, v_hc_scale,
                                             v_hc_base, V_HC, V_HC_ITERS, V_HC_EPS,
                                             pre + n * V_HC, post + n * V_HC, comb + (size_t)n * V_HC * V_HC);
        }
        check("hc pre", pre, v_hc_pre, V_HC_N * V_HC, 2e-6f);
        check("hc post", post, v_hc_post, V_HC_N * V_HC, 2e-6f);
        check("hc comb (sinkhorn)", comb, v_hc_comb, (size_t)V_HC_N * V_HC * V_HC, 2e-6f);
    }

    /* 2. sqrtsoftplus + noaux_tc routing */
    {
        int32_t idx[V_HC_N * V_TOPK];
        float w[V_HC_N * V_TOPK];
        int bad = 0;
        for (unsigned n = 0; n < V_HC_N; n++) {
            ds4_test_dsv41_route(v_gate_logits + (size_t)n * V_NE, v_gate_bias, V_NE, V_TOPK,
                                 V_ROUTE_SCALE, 1, idx + n * V_TOPK, w + n * V_TOPK);
            for (unsigned k = 0; k < V_TOPK; k++) {
                if (idx[n * V_TOPK + k] != v_gate_idx[n * V_TOPK + k]) bad++;
            }
        }
        if (bad) g_fail++;
        printf("  %-26s %s  %d/%d expert picks match\n", "route expert selection",
               bad ? "FAIL" : "ok  ", V_HC_N * V_TOPK - bad, V_HC_N * V_TOPK);
        check("route weights", w, v_gate_w, V_HC_N * V_TOPK, 2e-6f);
    }

    /* 3. sparse attention with an attention sink */
    {
        float out[V_S * V_H * V_D];
        ds4_test_dsv41_sparse_attn(v_attn_q, v_attn_kv, v_attn_sink, v_attn_idxs,
                                   V_S, V_H, V_D, V_TK, V_SCALE, out);
        check("sparse attn + sink", out, v_attn_out, (size_t)V_S * V_H * V_D, 3e-6f);
    }

    /* 4. compressor softmax pooling */
    {
        float out[V_NG * V_CD];
        ds4_test_dsv41_compress_pool(v_comp_kv, v_comp_score, V_NG, V_RATIO, V_CD, out);
        check("compressor pooling", out, v_comp_out, (size_t)V_NG * V_CD, 2e-6f);
    }

    /* 5. engram n-gram hash */
    {
        const unsigned cols = (V_NG_MAX - 1) * V_NHEAD;
        int64_t got[V_NPOS * ((V_NG_MAX - 1) * V_NHEAD)];
        for (unsigned p = 0; p < V_NPOS; p++) {
            ds4_test_dsv41_engram_hash(v_eng_ids, v_eng_dead, p, V_NG_MAX, V_NHEAD, V_PAD_ID,
                                       v_eng_mult, v_eng_primes, v_eng_offs, got + (size_t)p * cols);
        }
        check_i("engram hash ids", got, v_eng_hash, (size_t)V_NPOS * cols);
    }

    /* 6. engram gate */
    {
        float gate[V_NPOS * V_EHC], out[V_NPOS * V_EHC * V_EDIM];
        for (unsigned p = 0; p < V_NPOS; p++) {
            ds4_test_dsv41_engram_gate(v_eng_x + (size_t)p * V_EHC * V_EDIM,
                                       v_eng_key + (size_t)p * V_EHC * V_EDIM,
                                       v_eng_val + (size_t)p * V_EDIM, v_eng_qw, v_eng_kw,
                                       V_EHC, V_EDIM, V_EPS_E, gate + p * V_EHC,
                                       out + (size_t)p * V_EHC * V_EDIM);
        }
        check("engram gate", gate, v_eng_gate, (size_t)V_NPOS * V_EHC, 3e-6f);
        check("engram output", out, v_eng_out, (size_t)V_NPOS * V_EHC * V_EDIM, 3e-6f);
    }

    /* 7. candidate block selection */
    {
        int32_t keep[V_CB_Q * V_CB_W];
        int bad = 0;
        for (unsigned q = 0; q < V_CB_Q; q++) {
            ds4_test_dsv41_candidate_blocks(v_cb_logits + (size_t)q * V_CB_W, V_CB_W, V_CB_BLK,
                                            V_CB_TOP, (unsigned)v_cb_lens[q], keep + q * V_CB_W);
            for (unsigned j = 0; j < V_CB_W; j++) {
                if (keep[q * V_CB_W + j] != v_cb_keep[q * V_CB_W + j]) bad++;
            }
        }
        if (bad) g_fail++;
        printf("  %-26s %s  %d/%d positions match\n", "candidate blocks",
               bad ? "FAIL" : "ok  ", V_CB_Q * V_CB_W - bad, V_CB_Q * V_CB_W);
    }

    /* 8. shared KV / indexer source mapping (the released layout: sources at 2, 8, 14, 20) */
    {
        const uint32_t sources[4] = { 2, 8, 14, 20 };
        uint32_t map[43];
        ds4_test_dsv41_source_layer_map(sources, 4, 43, map);
        const uint32_t NONE = 0xffffffffu;
        int bad = 0;
        for (unsigned il = 0; il < 43; il++) {
            uint32_t want = NONE;
            if (il >= 20) want = 20; else if (il >= 14) want = 14;
            else if (il >= 8) want = 8; else if (il >= 2) want = 2;
            if (map[il] != want) bad++;
        }
        if (bad) g_fail++;
        printf("  %-26s %s  %d/43 layers map to the right source\n", "kv/index source layers",
               bad ? "FAIL" : "ok  ", 43 - bad);
        /* a source layer is its own source, and layers before the first have none */
        if (map[2] != 2 || map[19] != 14 || map[0] != NONE) { g_fail++; printf("      boundary case wrong\n"); }
    }

    /* 9. shape entry against the released config.json (guards transcription typos) */
    {
        static const struct { const char *name; uint32_t want; } expect[] = {
            {"n_layer (40 + 3 MTP)", 43}, {"n_embd", 5120},   {"n_vocab", 129280},
            {"n_head", 64},          {"head_dim", 512},       {"rope_head_dim", 64},
            {"q_lora_rank", 1280},   {"o_lora_rank", 1024},   {"o_groups", 8},
            {"n_routed_experts", 384}, {"experts_per_tok", 6}, {"moe_intermediate", 2304},
            {"index_n_heads", 32},   {"index_topk", 512},     {"sliding_window", 128},
            {"hc_mult", 4},          {"nextn layers", 3},     {"hash layers (dropped)", 0},
            {"engram layers", 2},    {"engram ngram sizes", 3}, {"engram heads", 8},
            {"engram head_dim", 256}, {"candidate blocks", 2048}, {"candidate block size", 8},
        };
        uint32_t got[24];
        ds4_test_dsv41_shape(got);
        int bad = 0;
        for (unsigned i = 0; i < sizeof(expect)/sizeof(expect[0]); i++) {
            if (got[i] != expect[i].want) {
                bad++;
                printf("      %s: got %u want %u\n", expect[i].name, got[i], expect[i].want);
            }
        }
        if (bad) g_fail++;
        printf("  %-26s %s  %u/24 fields match config.json\n", "V4.1 shape entry",
               bad ? "FAIL" : "ok  ", 24u - (unsigned)bad);
    }

    /* 10. E4M3 storage type: layout, and a decode of all 256 byte patterns */
    {
        /* one block per E8M0 exponent, qs covering every E4M3 pattern exactly once */
        unsigned char row[8 * 33];
        float got[256];
        double want[256];
        for (unsigned b = 0; b < 8; b++) {
            row[b * 33] = (unsigned char)(120 + b);            /* 2^-7 .. 2^0 */
            for (unsigned j = 0; j < 32; j++) row[b * 33 + 1 + j] = (unsigned char)(b * 32 + j);
        }
        for (unsigned b = 0; b < 8; b++) {
            const double d = ldexp(1.0, (int)row[b * 33] - 127);
            for (unsigned j = 0; j < 32; j++) {
                const unsigned v = row[b * 33 + 1 + j];
                const int exp = (int)((v >> 3) & 0xf), mant = (int)(v & 7);
                /* E4M3FN: subnormals are m*2^-9, normals (1+m/8)*2^(e-7), no infinities.
                 * 0x7f/0xff are the reserved NaN patterns; DS4 decodes them as the
                 * +-480 the fields name so weight decode stays finite under -ffast-math,
                 * and the sidecar converter refuses to emit those bytes. */
                double mag = exp == 0 ? (double)mant * ldexp(1.0, -9)
                                      : (1.0 + (double)mant / 8.0) * ldexp(1.0, exp - 7);
                if ((v >> 7) & 1) mag = -mag;
                want[b * 32 + j] = d * mag;
            }
        }
        ds4_test_e4m3_dequant_row(row, 256, got);
        /* every product is a power of two times an exactly-representable value */
        float wantf[256];
        for (unsigned i = 0; i < 256; i++) wantf[i] = (float)want[i];
        check("e4m3 decode (256 patterns)", got, wantf, 256, 0.0f);

        const uint32_t type = ds4_test_e4m3_type();
        uint64_t bytes = 0;
        const int ok_row = ds4_test_tensor_nbytes(type, 256, &bytes) && bytes == 264;
        if (!ok_row) g_fail++;
        printf("  %-26s %s  256 values -> %" PRIu64 " bytes (want 264), name %s\n",
               "e4m3 row layout", ok_row ? "ok  " : "FAIL", bytes, ds4_test_tensor_type_name(type));

        /* one engram table is [384006168, 256]; check the size maths does not overflow */
        const uint64_t rows = 384006168ull;
        uint64_t table = 0;
        const int ok_tab = ds4_test_tensor_nbytes(type, rows * 256ull, &table) &&
                           table == rows * 264ull;
        if (!ok_tab) g_fail++;
        printf("  %-26s %s  %.2f GB per table, %.2f GB for both\n", "e4m3 engram table size",
               ok_tab ? "ok  " : "FAIL", (double)table / 1e9, 2.0 * (double)table / 1e9);
    }

    /* 11. the engram sidecar, if one is on hand (DS4_ENGRAM_GGUF=<path>) */
    {
        const char *path = getenv("DS4_ENGRAM_GGUF");
        if (!path) {
            printf("  %-26s skip  set DS4_ENGRAM_GGUF to check a sidecar\n", "engram sidecar");
        } else {
            /* "the quick brown fox" style ids; only the last position is hashed */
            static const int tokens[] = {128000, 2, 1820, 4062, 14198, 39935};
            static int64_t rows[64];
            static float values[64 * 256];
            uint64_t table_rows = 0;
            const int n_bucket = ds4_test_engram_open(
                    path, tokens, (unsigned)(sizeof(tokens) / sizeof(tokens[0])), 0,
                    rows, values, &table_rows);
            int bad = 0;
            for (int i = 0; i < n_bucket; i++) {
                if (rows[i] < 0 || (uint64_t)rows[i] >= table_rows) bad++;
            }
            /* distinct buckets live in disjoint ranges, so no two rows may collide */
            for (int i = 0; i < n_bucket && !bad; i++) {
                for (int j = i + 1; j < n_bucket; j++) if (rows[i] == rows[j]) bad++;
            }
            if (n_bucket != 24 || bad) g_fail++;
            printf("  %-26s %s  %d buckets, rows in [0, %" PRIu64 "), all distinct\n",
                   "engram hash + row gather", (n_bucket == 24 && !bad) ? "ok  " : "FAIL",
                   n_bucket, table_rows);

            size_t nonzero = 0;
            float lo = values[0], hi = values[0];
            for (size_t i = 0; i < (size_t)n_bucket * 256; i++) {
                if (values[i] != 0.0f) nonzero++;
                if (values[i] < lo) lo = values[i];
                if (values[i] > hi) hi = values[i];
            }
            const int live = nonzero > (size_t)n_bucket * 64 && lo < 0.0f && hi > 0.0f;
            if (!live) g_fail++;
            printf("  %-26s %s  %zu/%d nonzero, range [%.4g, %.4g]\n", "engram row values",
                   live ? "ok  " : "FAIL", nonzero, n_bucket * 256, (double)lo, (double)hi);
        }
    }

    /* 12. the whole engram block against a real sidecar */
    {
        const char *path = getenv("DS4_ENGRAM_GGUF");
        if (path) {
            enum { DIM = 5120, HC = 4 };
            static const int tokens[] = {128000, 2, 1820, 4062, 14198, 39935};
            static float x[HC * DIM], out[HC * DIM];
            float gate[HC];
            uint32_t seed = 12345u;
            for (size_t i = 0; i < HC * DIM; i++) {
                seed = seed * 1664525u + 1013904223u;
                x[i] = (float)((int32_t)(seed >> 8) % 2001 - 1000) / 1000.0f;
            }
            const char *li_env = getenv("DS4_ENGRAM_LAYER");
            const unsigned li = li_env ? (unsigned)atoi(li_env) : 0u;
            const int hc = ds4_test_engram_forward(path, tokens,
                    (unsigned)(sizeof(tokens) / sizeof(tokens[0])), li, x, gate, out, 0);

            int bad = (hc != HC);
            for (int c = 0; c < hc; c++) {
                if (!(gate[c] > 0.0f && gate[c] < 1.0f)) bad++;
            }
            for (size_t i = 0; i < HC * DIM && !bad; i++) {
                if (!isfinite(out[i])) bad++;
            }
            if (bad) g_fail++;
            printf("  %-26s %s  gates %.4f %.4f %.4f %.4f\n", "engram forward", bad ? "FAIL" : "ok  ",
                   (double)gate[0], (double)gate[1], (double)gate[2], (double)gate[3]);

            /* one value is shared by every copy, so (out - x) / gate must not depend on c */
            double worst = 0.0;
            for (uint32_t c = 1; c < HC; c++) {
                for (uint32_t j = 0; j < DIM; j++) {
                    const double a = (out[j] - x[j]) / gate[0];
                    const double b = (out[c * DIM + j] - x[c * DIM + j]) / gate[c];
                    const double d = fabs(a - b);
                    if (d > worst) worst = d;
                }
            }
            const int shared = worst < 1e-3;
            if (!shared) g_fail++;
            printf("  %-26s %s  max|d|=%.3e across hc copies\n", "engram value is shared",
                   shared ? "ok  " : "FAIL", worst);

            /* a dead token (an image span) shuts the gate and passes the stream through */
            static float dead_out[HC * DIM];
            float dead_gate[HC];
            ds4_test_engram_forward(path, tokens,
                    (unsigned)(sizeof(tokens) / sizeof(tokens[0])), li, x, dead_gate, dead_out, 1);
            int dead_bad = 0;
            for (int c = 0; c < HC; c++) if (dead_gate[c] != 0.0f) dead_bad++;
            for (size_t i = 0; i < HC * DIM; i++) if (dead_out[i] != x[i]) dead_bad++;
            if (dead_bad) g_fail++;
            printf("  %-26s %s  gate 0, stream unchanged\n", "engram dead token",
                   dead_bad ? "FAIL" : "ok  ");

            const char *dump = getenv("DS4_ENGRAM_DUMP");
            if (dump) {
                FILE *f = fopen(dump, "wb");
                if (f) {
                    fwrite(x, sizeof(float), HC * DIM, f);
                    fwrite(gate, sizeof(float), HC, f);
                    fwrite(out, sizeof(float), HC * DIM, f);
                    fclose(f);
                    printf("  %-26s ok    wrote %s\n", "engram forward dump", dump);
                }
            }
        }
    }

    printf(g_fail ? "\ndsv41 reference: %d failure(s)\n" : "\ndsv41 reference: all checks passed\n", g_fail);
    return g_fail ? 1 : 0;
}
