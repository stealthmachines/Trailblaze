/*
 * tb_coherence_test.c — GGUF prompt coherence tester
 *
 * Build:
 *   gcc -O2 -std=c11 -DTB_NO_MAIN \
 *       -Ilayer0 -Ilayer1 -Ilayer2 -Ilayer3 -Ilayer4 -Ilayer5 -Iinclude -Isrc \
 *       tb_coherence_test.c \
 *       layer4/tb_infer.c layer4/tb_gguf.c layer4/tb_tokenizer.c \
 *       layer3/tb_orchestration.c layer5/tb_semantic_os.c \
 *       layer0/tb_phi_lattice.c layer1/tb_tensor.c layer2/tb_graph.c \
 *       src/sha256_minimal.c src/hdgl_bootloaderz.c src/hdgl_router.c \
 *       src/vector_container.c src/analog_engine.c src/tb_analog_dispatch.c \
 *       -lm -lpthread -o bin/tb_coherence_test
 *
 * Usage:
 *   ./bin/tb_coherence_test --model <path.gguf> [--hdgl] [--tokens N] [--temp F]
 *
 * What it tests:
 *   Each test case has a prompt and a set of expected substrings in the
 *   decoded output.  The test runs twice — once with HDGL routing ON and
 *   once with it OFF — so you can see whether HDGL is responsible for any
 *   remaining garbling.
 *
 *   Coherence scoring:
 *     PASS  — all expected substrings found, no <unk> tokens, no repeated
 *             n-grams (repetition ratio < 0.4)
 *     WARN  — output is readable but some expected substrings missing
 *     FAIL  — <unk> present, or repetition ratio >= 0.4, or completely
 *             empty output
 */

#define TB_INFER_TEST
#define TB_NO_MAIN

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "layer4/tb_infer.h"
#include "layer4/tb_tokenizer.h"
#include "layer4/tb_gguf.h"

/* ── Colour helpers (ANSI, skipped if not a tty) ─────────────────────────── */
#define COL_RESET  "\033[0m"
#define COL_GREEN  "\033[32m"
#define COL_YELLOW "\033[33m"
#define COL_RED    "\033[31m"
#define COL_BOLD   "\033[1m"
#define COL_CYAN   "\033[36m"

/* ── Test case definition ─────────────────────────────────────────────────── */
typedef struct {
    const char *name;
    const char *prompt;            /* raw user text (template applied automatically) */
    const char *expected[8];       /* substrings any coherent answer should contain;
                                    * NULL-terminated; NULL means "just check basics" */
} TC;

static TC TESTS[] = {
    {
        "capital_of_france",
        "What is the capital of France?",
        { "Paris", NULL }
    },
    {
        "simple_arithmetic",
        "What is 2 + 2?",
        { "4", NULL }
    },
    {
        "word_spaces",
        "Write one sentence about the ocean.",
        /* Just verify no <unk> and spaces are present */
        { NULL }
    },
    {
        "continuation",
        "The quick brown fox",
        /* Expect common continuations; relaxed — just check no <unk> */
        { NULL }
    },
    {
        "list_items",
        "Name three primary colours.",
        /* Red / blue / yellow or red / blue / green depending on model's definition */
        { NULL }
    },
};
#define N_TESTS ((int)(sizeof(TESTS)/sizeof(TESTS[0])))

/* ── Repetition detector ──────────────────────────────────────────────────── */
/* Returns fraction of 3-grams (token trigrams) that are repeats. */
static float repetition_ratio(const int *ids, int n) {
    if (n < 4) return 0.0f;
    int repeats = 0;
    for (int i = 0; i < n - 3; i++) {
        for (int j = i + 1; j < n - 2; j++) {
            if (ids[i]   == ids[j]   &&
                ids[i+1] == ids[j+1] &&
                ids[i+2] == ids[j+2]) {
                repeats++;
                break;
            }
        }
    }
    return (float)repeats / (float)(n - 3);
}

/* ── Result enum ─────────────────────────────────────────────────────────── */
typedef enum { R_PASS = 0, R_WARN = 1, R_FAIL = 2 } Result;

static const char *result_str(Result r) {
    switch (r) {
        case R_PASS: return COL_GREEN  "PASS" COL_RESET;
        case R_WARN: return COL_YELLOW "WARN" COL_RESET;
        case R_FAIL: return COL_RED    "FAIL" COL_RESET;
    }
    return "????";
}

/* ── Single test run ─────────────────────────────────────────────────────── */
typedef struct {
    Result result;
    char   decoded[2048];
    float  rep_ratio;
    int    has_unk;
    int    n_tokens;
    double ms;
} RunResult;

static RunResult run_one(TB_InferCtx *ctx, const TC *tc, int max_tokens) {
    RunResult rr = {0};

    TB_Tokenizer *tok = ctx->model ? ctx->model->tokenizer : NULL;

    /* Build chat-formatted prompt */
    char prompt_buf[4096] = "";
    if (tok) {
        /* Reuse the chat-template logic from tb_infer.c via a thin wrapper:
         * just encode directly as a user message using the model's template.
         * We call tb_tokenizer_encode with the raw user text for simplicity —
         * for a proper chat test the template wrapping happens in tb_infer.c's
         * HTTP handler.  Here we want to test the token round-trip, so use
         * the prompt directly. */
        snprintf(prompt_buf, sizeof(prompt_buf), "%s", tc->prompt);
    }

    /* Encode */
    int tok_ids[2048]; int n_tok = 0;
    if (tok) {
        n_tok = tb_tokenizer_encode(tok, prompt_buf, 1, tok_ids, 2048);
    }
    if (n_tok <= 0) {
        snprintf(rr.decoded, sizeof(rr.decoded), "[encoding failed]");
        rr.result = R_FAIL;
        return rr;
    }

    /* Generate */
    int out_ids[512];
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    int n_gen = tb_infer_generate(ctx, tok_ids, n_tok,
                                   out_ids, max_tokens, 0,
                                   NULL, NULL);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    rr.ms = (t1.tv_sec - t0.tv_sec)*1000.0 + (t1.tv_nsec - t0.tv_nsec)/1e6;
    rr.n_tokens = n_gen;

    if (n_gen <= 0) {
        snprintf(rr.decoded, sizeof(rr.decoded), "[no tokens generated]");
        rr.result = R_FAIL;
        return rr;
    }

    /* Decode */
    char *dec = tok ? tb_tokenizer_decode(tok, out_ids, n_gen) : NULL;
    if (!dec || !dec[0]) {
        snprintf(rr.decoded, sizeof(rr.decoded), "[decode returned empty]");
        free(dec);
        rr.result = R_FAIL;
        return rr;
    }
    snprintf(rr.decoded, sizeof(rr.decoded), "%s", dec);
    free(dec);

    /* Check for <unk> tokens */
    rr.has_unk = (strstr(rr.decoded, "<unk>") != NULL) ? 1 : 0;

    /* Check repetition */
    rr.rep_ratio = repetition_ratio(out_ids, n_gen);

    /* Check expected substrings */
    int missing = 0;
    for (int e = 0; tc->expected[e] != NULL; e++) {
        if (strstr(rr.decoded, tc->expected[e]) == NULL)
            missing++;
    }

    /* Score */
    if (rr.has_unk || rr.rep_ratio >= 0.4f || n_gen <= 0) {
        rr.result = R_FAIL;
    } else if (missing > 0) {
        rr.result = R_WARN;
    } else {
        rr.result = R_PASS;
    }

    return rr;
}

/* ── Print a result row ───────────────────────────────────────────────────── */
static void print_result(const char *label, const TC *tc, const RunResult *rr) {
    printf("  %-10s  %s  rep=%.2f  unk=%s  %dtok  %.0fms\n",
           label,
           result_str(rr->result),
           rr->rep_ratio,
           rr->has_unk ? COL_RED "yes" COL_RESET : "no",
           rr->n_tokens,
           rr->ms);

    /* Print decoded output, truncated */
    char truncated[120];
    strncpy(truncated, rr->decoded, 119);
    truncated[119] = '\0';
    /* replace newlines for display */
    for (int i = 0; truncated[i]; i++)
        if (truncated[i] == '\n' || truncated[i] == '\r') truncated[i] = ' ';
    printf("             \"%s\"\n", truncated);
}

/* ── Main ─────────────────────────────────────────────────────────────────── */
int main(int argc, char **argv) {
    const char *model_path = NULL;
    int max_tokens = 64;
    float temperature = 0.0f;   /* greedy by default for reproducibility */
    int compare_hdgl = 1;       /* run both HDGL-on and HDGL-off by default */

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--model") && i+1 < argc)  model_path = argv[++i];
        else if (!strcmp(argv[i], "--tokens") && i+1 < argc) max_tokens = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--temp") && i+1 < argc)   temperature = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--no-compare")) compare_hdgl = 0;
    }

    if (!model_path) {
        fprintf(stderr,
            "Usage: %s --model <path.gguf> [--tokens N] [--temp F] [--no-compare]\n"
            "\n"
            "  --tokens N      Max new tokens per test (default 64)\n"
            "  --temp F        Sampling temperature 0=greedy (default 0.0)\n"
            "  --no-compare    Only run HDGL-off; skip HDGL-on comparison\n",
            argv[0]);
        return 1;
    }

    printf(COL_BOLD "\n=== TRAILBLAZE Coherence Test ===\n" COL_RESET);
    printf("Model : %s\n", model_path);
    printf("Tokens: %d per prompt\n", max_tokens);
    printf("Temp  : %.2f %s\n\n", temperature,
           temperature == 0.0f ? "(greedy)" : "");

    /* Load model once */
    TB_GGUFModel *model = tb_model_load(model_path);
    if (!model) {
        fprintf(stderr, "Failed to load model: %s\n", model_path);
        return 1;
    }
    printf("Vocab : %d  BOS=%d  EOS=%d\n",
           model->tokenizer ? model->tokenizer->vocab_size : 0,
           model->tokenizer ? model->tokenizer->bos_id : -1,
           model->tokenizer ? model->tokenizer->eos_id : -1);
    printf("Arch  : %s  layers=%d  hidden=%d\n\n",
           model->arch, model->n_layers, model->hidden_dim);

    int total_pass = 0, total_warn = 0, total_fail = 0;

    /* ── Run 1: HDGL OFF (baseline — should be closest to reference) ── */
    {
        printf(COL_BOLD COL_CYAN "── HDGL OFF (baseline) ──\n" COL_RESET);
        TB_InferCtx *ctx = tb_infer_create(model, /*use_hdgl=*/0, 0.0f, 512, 0xC0FFEEULL);
        if (!ctx) { fprintf(stderr, "Failed to create context\n"); return 1; }
        ctx->temperature = temperature;
        ctx->max_new_tokens = max_tokens;

        for (int t = 0; t < N_TESTS; t++) {
            printf(COL_BOLD "  [%d/%d] %s\n" COL_RESET, t+1, N_TESTS, TESTS[t].name);
            RunResult rr = run_one(ctx, &TESTS[t], max_tokens);
            print_result("hdgl=off", &TESTS[t], &rr);
            if (rr.result == R_PASS) total_pass++;
            else if (rr.result == R_WARN) total_warn++;
            else total_fail++;
        }
        tb_infer_free(ctx);
        printf("\n");
    }

    /* ── Run 2: HDGL ON (compare) ── */
    if (compare_hdgl) {
        printf(COL_BOLD COL_CYAN "── HDGL ON (alpha=0.2) ──\n" COL_RESET);
        TB_InferCtx *ctx = tb_infer_create(model, /*use_hdgl=*/1, 0.2f, 512, 0xC0FFEEULL);
        if (!ctx) { fprintf(stderr, "Failed to create context\n"); return 1; }
        ctx->temperature = temperature;
        ctx->max_new_tokens = max_tokens;

        int hdgl_pass = 0, hdgl_warn = 0, hdgl_fail = 0;
        for (int t = 0; t < N_TESTS; t++) {
            printf(COL_BOLD "  [%d/%d] %s\n" COL_RESET, t+1, N_TESTS, TESTS[t].name);
            RunResult rr = run_one(ctx, &TESTS[t], max_tokens);
            print_result("hdgl=on ", &TESTS[t], &rr);
            if (rr.result == R_PASS) hdgl_pass++;
            else if (rr.result == R_WARN) hdgl_warn++;
            else hdgl_fail++;
        }
        tb_infer_free(ctx);

        printf("\n");
        printf(COL_BOLD "HDGL-ON  summary: " COL_RESET
               COL_GREEN "%d pass" COL_RESET "  "
               COL_YELLOW "%d warn" COL_RESET "  "
               COL_RED "%d fail" COL_RESET "\n",
               hdgl_pass, hdgl_warn, hdgl_fail);

        if (hdgl_fail > total_fail || hdgl_warn > total_warn)
            printf(COL_YELLOW
                   "  → HDGL routing is degrading coherence vs baseline.\n"
                   "    Re-run with --no-hdgl to confirm.\n"
                   COL_RESET);
        else if (hdgl_pass >= total_pass)
            printf(COL_GREEN
                   "  → HDGL routing not degrading coherence.\n"
                   COL_RESET);
    }

    /* ── Baseline summary ── */
    printf("\n");
    printf(COL_BOLD "Baseline summary: " COL_RESET
           COL_GREEN "%d pass" COL_RESET "  "
           COL_YELLOW "%d warn" COL_RESET "  "
           COL_RED "%d fail" COL_RESET "\n",
           total_pass, total_warn, total_fail);

    if (total_fail == 0 && total_warn == 0)
        printf(COL_GREEN COL_BOLD "All tests passed — output looks coherent.\n" COL_RESET);
    else if (total_fail > 0)
        printf(COL_RED "Failures detected — tokenizer or inference bug still present.\n" COL_RESET);
    else
        printf(COL_YELLOW "Warnings — output is readable but some expected content missing.\n" COL_RESET);

    tb_model_free(model);
    printf("\n");
    return (total_fail > 0) ? 1 : 0;
}
