#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <errno.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

#include "hdgl_router.h"

#if !defined(_WIN32)
#include <sys/types.h>
#endif

#define DEFAULT_MODEL_PATH "."
#define DEFAULT_TOPK 4
#define MAX_BENCHMARK_ITERS 1000
#define HDGL_OCTAVES 8

// Spiral8 octave growth exponents used to derive deterministic semantic routing bias.
static const float k_hdgl_octave_alpha[HDGL_OCTAVES] = {
    0.015269f, 0.008262f, 0.110649f, -0.083485f,
    0.025847f, -0.045123f, 0.067891f, 0.012345f
};

typedef struct {
    int hidden_size;
    int num_hidden_layers;
    int num_experts;
    int num_experts_per_tok;
    int vocab_size;
    int full_attention_interval;
    int moe_intermediate_size;
    int shared_expert_intermediate_size;
    int group_size;
} ModelConfig;

typedef struct {
    size_t gate_w_offset;
    size_t gate_w_size;
    size_t gate_s_offset;
    size_t gate_s_size;
    size_t gate_b_offset;
    size_t gate_b_size;
    size_t up_w_offset;
    size_t up_w_size;
    size_t up_s_offset;
    size_t up_s_size;
    size_t up_b_offset;
    size_t up_b_size;
    size_t down_w_offset;
    size_t down_w_size;
    size_t down_s_offset;
    size_t down_s_size;
    size_t down_b_offset;
    size_t down_b_size;
    size_t expert_size;
} ExpertLayout;

typedef struct {
    float *gate;
    float *up;
    float *act;
} CpuExpertScratch;

typedef struct {
    double io_ms;
    double compute_ms;
    double total_ms;
} CpuTiming;

typedef struct {
    const char *name;
    size_t offset;
    size_t size;
    int found;
} TensorSlice;

typedef struct {
    TensorSlice route_w, route_s, route_b;
    TensorSlice shared_gate_proj_w, shared_gate_proj_s, shared_gate_proj_b;
    TensorSlice shared_up_proj_w, shared_up_proj_s, shared_up_proj_b;
    TensorSlice shared_down_proj_w, shared_down_proj_s, shared_down_proj_b;
    TensorSlice shared_expert_gate_w, shared_expert_gate_s, shared_expert_gate_b;
} RouteLayerTensors;

typedef struct {
    CpuTiming moe_timing;
    float shared_gate_value;
    double total_ms;
} RouteStepTiming;

static double now_ms(void) {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1000000.0;
}

static void print_usage(const char *prog) {
    printf("Usage: %s [options]\n", prog);
    printf("  --model PATH             Model directory containing exported artifacts\n");
    printf("  --manifest PATH          Manifest path (default: <model>/model_weights.json)\n");
    printf("  --weights PATH           Weights path (default: <model>/model_weights.bin)\n");
    printf("  --packed-experts PATH    Packed experts directory (default: <model>/packed_experts)\n");
    printf("  --layer N                Layer index for expert execution (default: 0)\n");
    printf("  --expert E               Expert index / seed expert (default: 0)\n");
    printf("  --k N                    Active experts for --moe (default: 4)\n");
    printf("  --embed-token ID         Read one embedding row from model_weights.bin\n");
    printf("  --lm-head-token ID       Embed token ID then stream lm_head argmax/top-5\n");
    printf("  --route-token ID         Embed token ID, route experts, and run one CPU MoE layer step\n");
    printf("  --route-layer N          Layer index for --route-token (default: 0)\n");
    printf("  --route-layers N         Number of consecutive routed layers to execute (default: 1)\n");
    printf("  --route-lm-head          Run lm_head argmax/top-5 after the routed layer stack\n");
    printf("  --hdgl                   Enable HDGL blend for --route-token\n");
    printf("  --hdgl-alpha F           HDGL blend weight for routing (default: 0.20)\n");
    printf("  --hdgl-semantic          Enable phi-octave semantic routing bias (hybrid preselection)\n");
    printf("  --hdgl-load FILE         Load a pre-seeded HDGL lattice file\n");
    printf("  --moe                    Run a CPU MoE slice with K routed experts\n");
    printf("  --benchmark N            Repeat the selected CPU slice N times\n");
    printf("  --check-only             Validate local artifacts and manifest contract\n");
    printf("  --help                   Show this message\n");
}

static int file_exists(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    fclose(f);
    return 1;
}

static char *read_text_file(const char *path) {
    FILE *f = fopen(path, "rb");
    long size;
    char *buf;

    if (!f) return NULL;
    if (fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        return NULL;
    }
    size = ftell(f);
    if (size < 0) {
        fclose(f);
        return NULL;
    }
    if (fseek(f, 0, SEEK_SET) != 0) {
        fclose(f);
        return NULL;
    }

    buf = (char *)malloc((size_t)size + 1);
    if (!buf) {
        fclose(f);
        return NULL;
    }
    if (fread(buf, 1, (size_t)size, f) != (size_t)size) {
        free(buf);
        fclose(f);
        return NULL;
    }
    buf[size] = '\0';
    fclose(f);
    return buf;
}

static int json_find_int(const char *json, const char *key, int fallback) {
    char needle[128];
    const char *p;
    snprintf(needle, sizeof(needle), "\"%s\"", key);
    p = strstr(json, needle);
    if (!p) return fallback;
    p = strchr(p, ':');
    if (!p) return fallback;
    p++;
    while (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n') p++;
    return (int)strtol(p, NULL, 10);
}

static int manifest_has_key(const char *json, const char *key) {
    char needle[256];
    snprintf(needle, sizeof(needle), "\"%s\"", key);
    return strstr(json, needle) != NULL;
}

static int json_find_size_t_in_block(const char *start, const char *key, size_t *out_value) {
    char needle[128];
    const char *p;
    snprintf(needle, sizeof(needle), "\"%s\"", key);
    p = strstr(start, needle);
    if (!p) return 0;
    p = strchr(p, ':');
    if (!p) return 0;
    p++;
    while (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n') p++;
    *out_value = (size_t)strtoull(p, NULL, 10);
    return 1;
}

static int manifest_find_tensor_slice(const char *json, const char *name, TensorSlice *slice) {
    char needle[256];
    const char *tensor_pos;
    const char *block_start;
    if (!json || !name || !slice) return 0;
    snprintf(needle, sizeof(needle), "\"%s\"", name);
    tensor_pos = strstr(json, needle);
    if (!tensor_pos) return 0;
    block_start = strchr(tensor_pos, '{');
    if (!block_start) return 0;
    slice->name = name;
    slice->found = json_find_size_t_in_block(block_start, "offset", &slice->offset) &&
                   json_find_size_t_in_block(block_start, "size", &slice->size);
    return slice->found;
}

static void append_path(char *dst, size_t dst_size, const char *dir, const char *file) {
    size_t n = strlen(dir);
    if (n > 0 && (dir[n - 1] == '/' || dir[n - 1] == '\\')) {
        snprintf(dst, dst_size, "%s%s", dir, file);
    } else {
        snprintf(dst, dst_size, "%s/%s", dir, file);
    }
}

static void build_layer_path(char *dst, size_t dst_size, const char *packed_dir, int layer_idx) {
    snprintf(dst, dst_size, "%s/layer_%02d.bin", packed_dir, layer_idx);
}

static int require_manifest_key(const char *json, int *missing, const char *fmt, ...) {
    char key[256];
    va_list args;
    va_start(args, fmt);
    vsnprintf(key, sizeof(key), fmt, args);
    va_end(args);
    if (!manifest_has_key(json, key)) {
        if (*missing < 16) fprintf(stderr, "[manifest] missing required tensor: %s\n", key);
        (*missing)++;
        return 0;
    }
    return 1;
}

static int validate_manifest_contract(const char *json, const ModelConfig *cfg) {
    int missing = 0;
    int i;

    require_manifest_key(json, &missing, "model.embed_tokens.weight");
    require_manifest_key(json, &missing, "model.norm.weight");
    require_manifest_key(json, &missing, "lm_head.weight");

    for (i = 0; i < cfg->num_hidden_layers; i++) {
        int is_full = ((i + 1) % cfg->full_attention_interval) == 0;
        require_manifest_key(json, &missing, "model.layers.%d.input_layernorm.weight", i);
        require_manifest_key(json, &missing, "model.layers.%d.post_attention_layernorm.weight", i);

        if (is_full) {
            require_manifest_key(json, &missing, "model.layers.%d.self_attn.q_proj.weight", i);
            require_manifest_key(json, &missing, "model.layers.%d.self_attn.k_proj.weight", i);
            require_manifest_key(json, &missing, "model.layers.%d.self_attn.v_proj.weight", i);
            require_manifest_key(json, &missing, "model.layers.%d.self_attn.o_proj.weight", i);
            require_manifest_key(json, &missing, "model.layers.%d.self_attn.q_norm.weight", i);
            require_manifest_key(json, &missing, "model.layers.%d.self_attn.k_norm.weight", i);
        } else {
            require_manifest_key(json, &missing, "model.layers.%d.linear_attn.in_proj_qkv.weight", i);
            require_manifest_key(json, &missing, "model.layers.%d.linear_attn.in_proj_z.weight", i);
            require_manifest_key(json, &missing, "model.layers.%d.linear_attn.in_proj_b.weight", i);
            require_manifest_key(json, &missing, "model.layers.%d.linear_attn.in_proj_a.weight", i);
            require_manifest_key(json, &missing, "model.layers.%d.linear_attn.conv1d.weight", i);
            require_manifest_key(json, &missing, "model.layers.%d.linear_attn.A_log", i);
            require_manifest_key(json, &missing, "model.layers.%d.linear_attn.dt_bias", i);
            require_manifest_key(json, &missing, "model.layers.%d.linear_attn.norm.weight", i);
            require_manifest_key(json, &missing, "model.layers.%d.linear_attn.out_proj.weight", i);
        }

        require_manifest_key(json, &missing, "model.layers.%d.mlp.gate.weight", i);
        require_manifest_key(json, &missing, "model.layers.%d.mlp.shared_expert.gate_proj.weight", i);
        require_manifest_key(json, &missing, "model.layers.%d.mlp.shared_expert.up_proj.weight", i);
        require_manifest_key(json, &missing, "model.layers.%d.mlp.shared_expert.down_proj.weight", i);
        require_manifest_key(json, &missing, "model.layers.%d.mlp.shared_expert_gate.weight", i);
    }

    if (missing > 0) {
        if (missing > 16) fprintf(stderr, "[manifest] ... and %d more missing tensors\n", missing - 16);
        return 0;
    }
    return 1;
}

static int clamp_int(int value, int min_value, int max_value) {
    if (value < min_value) return min_value;
    if (value > max_value) return max_value;
    return value;
}

static float clamp_float(float value, float min_value, float max_value) {
    if (value < min_value) return min_value;
    if (value > max_value) return max_value;
    return value;
}

static uint64_t fnv1a64_mix(uint64_t x) {
    uint64_t h = 1469598103934665603ULL;
    int i;
    for (i = 0; i < 8; i++) {
        uint8_t b = (uint8_t)((x >> (i * 8)) & 0xFFu);
        h ^= (uint64_t)b;
        h *= 1099511628211ULL;
    }
    return h;
}

static int hdgl_semantic_octave(int token_id, int layer_idx) {
    const double phi = 1.6180339887498948482;
    const double inv_phi2 = 1.0 / (phi * phi);
    uint64_t key = ((uint64_t)(uint32_t)token_id << 32) ^ (uint64_t)(uint32_t)layer_idx;
    uint64_t h = fnv1a64_mix(key);
    double tau = ((double)(h & 0xFFFFFFFFULL) / 4294967295.0) * (double)HDGL_OCTAVES * inv_phi2;
    int octave = (int)tau;
    return clamp_int(octave, 0, HDGL_OCTAVES - 1);
}

static float hdgl_semantic_boost(int semantic_octave, int expert_idx, int num_experts) {
    int experts_per_octave;
    int expert_octave;
    int distance;
    float local_gain;
    float spread;
    if (num_experts <= 0) return 1.0f;
    experts_per_octave = num_experts / HDGL_OCTAVES;
    if (experts_per_octave <= 0) experts_per_octave = 1;
    expert_octave = clamp_int(expert_idx / experts_per_octave, 0, HDGL_OCTAVES - 1);
    distance = abs(expert_octave - semantic_octave);

    // Favors nearby octaves while keeping all experts reachable.
    spread = expf(-0.85f * (float)distance);
    local_gain = 1.0f + fabsf(k_hdgl_octave_alpha[semantic_octave]);
    return 0.15f + spread * local_gain;
}

static size_t align_up(size_t value, size_t alignment) {
    size_t mask = alignment - 1;
    return (value + mask) & ~mask;
}

static int compute_expert_layout(const ModelConfig *cfg, ExpertLayout *layout) {
    size_t offset = 0;
    size_t hidden_size;
    size_t intermediate_size;
    size_t group_size;
    size_t hidden_groups;
    size_t intermediate_groups;

    if (!cfg || !layout) return 0;
    if (cfg->hidden_size <= 0 || cfg->moe_intermediate_size <= 0 || cfg->group_size <= 0) return 0;
    if ((cfg->hidden_size % cfg->group_size) != 0) return 0;
    if ((cfg->moe_intermediate_size % cfg->group_size) != 0) return 0;

    hidden_size = (size_t)cfg->hidden_size;
    intermediate_size = (size_t)cfg->moe_intermediate_size;
    group_size = (size_t)cfg->group_size;
    hidden_groups = hidden_size / group_size;
    intermediate_groups = intermediate_size / group_size;

    layout->gate_w_offset = offset;
    layout->gate_w_size = hidden_size * intermediate_size / 2;
    offset += layout->gate_w_size;

    layout->gate_s_offset = offset;
    layout->gate_s_size = intermediate_size * hidden_groups * sizeof(uint16_t);
    offset += layout->gate_s_size;

    layout->gate_b_offset = offset;
    layout->gate_b_size = intermediate_size * hidden_groups * sizeof(uint16_t);
    offset += layout->gate_b_size;

    layout->up_w_offset = offset;
    layout->up_w_size = hidden_size * intermediate_size / 2;
    offset += layout->up_w_size;

    layout->up_s_offset = offset;
    layout->up_s_size = intermediate_size * hidden_groups * sizeof(uint16_t);
    offset += layout->up_s_size;

    layout->up_b_offset = offset;
    layout->up_b_size = intermediate_size * hidden_groups * sizeof(uint16_t);
    offset += layout->up_b_size;

    layout->down_w_offset = offset;
    layout->down_w_size = hidden_size * intermediate_size / 2;
    offset += layout->down_w_size;

    layout->down_s_offset = offset;
    layout->down_s_size = hidden_size * intermediate_groups * sizeof(uint16_t);
    offset += layout->down_s_size;

    layout->down_b_offset = offset;
    layout->down_b_size = hidden_size * intermediate_groups * sizeof(uint16_t);
    offset += layout->down_b_size;

    layout->expert_size = align_up(offset, 64);
    return 1;
}

static int init_cpu_scratch(CpuExpertScratch *scratch, const ModelConfig *cfg) {
    size_t inter_bytes;
    if (!scratch || !cfg) return 0;
    memset(scratch, 0, sizeof(*scratch));
    inter_bytes = (size_t)cfg->moe_intermediate_size * sizeof(float);
    scratch->gate = (float *)malloc(inter_bytes);
    scratch->up = (float *)malloc(inter_bytes);
    scratch->act = (float *)malloc(inter_bytes);
    if (!scratch->gate || !scratch->up || !scratch->act) {
        free(scratch->gate);
        free(scratch->up);
        free(scratch->act);
        memset(scratch, 0, sizeof(*scratch));
        return 0;
    }
    return 1;
}

static void free_cpu_scratch(CpuExpertScratch *scratch) {
    if (!scratch) return;
    free(scratch->gate);
    free(scratch->up);
    free(scratch->act);
    memset(scratch, 0, sizeof(*scratch));
}

static int seek_file_to(FILE *f, size_t offset) {
#if defined(_WIN32)
    return _fseeki64(f, (__int64)offset, SEEK_SET) == 0;
#else
    return fseeko(f, (off_t)offset, SEEK_SET) == 0;
#endif
}

static int read_bytes_at(FILE *f, void *dst, size_t size, size_t offset) {
    if (!seek_file_to(f, offset)) return 0;
    return fread(dst, 1, size, f) == size;
}

static int read_tensor_row(FILE *weights_file, const TensorSlice *tensor, size_t row_size, int row_idx, void *dst) {
    size_t row_offset;
    if (!weights_file || !tensor || !tensor->found || row_size == 0 || row_idx < 0) return 0;
    row_offset = tensor->offset + (size_t)row_idx * row_size;
    if (row_offset + row_size > tensor->offset + tensor->size) return 0;
    return read_bytes_at(weights_file, dst, row_size, row_offset);
}

static float bf16_to_f32(uint16_t bf16) {
    uint32_t bits = (uint32_t)bf16 << 16;
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static void cpu_dequant_matvec_4bit(
    const uint32_t *w_packed,
    const uint16_t *scales,
    const uint16_t *biases,
    const float *x,
    float *out,
    int out_dim,
    int in_dim,
    int group_size
) {
    int row;
    int num_groups = in_dim / group_size;
    int packed_per_group = group_size / 8;
    int packed_cols = in_dim / 8;

    for (row = 0; row < out_dim; row++) {
        float acc = 0.0f;
        const uint32_t *w_row = w_packed + (size_t)row * (size_t)packed_cols;
        const uint16_t *s_row = scales + (size_t)row * (size_t)num_groups;
        const uint16_t *b_row = biases + (size_t)row * (size_t)num_groups;
        int group_idx;

        for (group_idx = 0; group_idx < num_groups; group_idx++) {
            float scale = bf16_to_f32(s_row[group_idx]);
            float bias = bf16_to_f32(b_row[group_idx]);
            int base_packed = group_idx * packed_per_group;
            int base_x = group_idx * group_size;
            int packed_idx;

            for (packed_idx = 0; packed_idx < packed_per_group; packed_idx++) {
                uint32_t packed = w_row[base_packed + packed_idx];
                int nibble_idx;
                for (nibble_idx = 0; nibble_idx < 8; nibble_idx++) {
                    uint32_t q = (packed >> (nibble_idx * 4)) & 0xFu;
                    int x_idx = base_x + packed_idx * 8 + nibble_idx;
                    acc += (scale * (float)q + bias) * x[x_idx];
                }
            }
        }
        out[row] = acc;
    }
}

static void cpu_swiglu(const float *gate, const float *up, float *out, int dim) {
    int i;
    for (i = 0; i < dim; i++) {
        float g = gate[i];
        float silu_g = g / (1.0f + expf(-g));
        out[i] = silu_g * up[i];
    }
}

static float cpu_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

static void fill_input_vector(float *x, int hidden_size) {
    int i;
    for (i = 0; i < hidden_size; i++) {
        x[i] = 0.1f * sinf((float)i * 0.1f + 0.3f);
    }
}

static int cpu_argmax(const float *x, int dim) {
    int best_idx = 0;
    int i;
    for (i = 1; i < dim; i++) {
        if (x[i] > x[best_idx]) best_idx = i;
    }
    return best_idx;
}

static void cpu_softmax(float *x, int dim) {
    float max_val = x[0];
    float sum = 0.0f;
    float inv_sum;
    int i;
    for (i = 1; i < dim; i++) {
        if (x[i] > max_val) max_val = x[i];
    }
    for (i = 0; i < dim; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    inv_sum = 1.0f / sum;
    for (i = 0; i < dim; i++) x[i] *= inv_sum;
}

static void cpu_topk(const float *scores, int dim, int k, int *indices, float *values) {
    int i;
    for (i = 0; i < k; i++) {
        indices[i] = 0;
        values[i] = -1e30f;
    }
    for (i = 0; i < dim; i++) {
        int min_k = 0;
        int j;
        for (j = 1; j < k; j++) {
            if (values[j] < values[min_k]) min_k = j;
        }
        if (scores[i] > values[min_k]) {
            values[min_k] = scores[i];
            indices[min_k] = i;
        }
    }
}

static void cpu_normalize_weights(float *weights, int k) {
    float sum = 0.0f;
    int i;
    for (i = 0; i < k; i++) sum += weights[i];
    if (sum <= 0.0f) return;
    for (i = 0; i < k; i++) weights[i] /= sum;
}

static void print_vector_sample(const char *label, const float *values, int count) {
    int sample = count < 8 ? count : 8;
    int i;
    printf("%s[", label);
    for (i = 0; i < sample; i++) {
        printf(i == 0 ? "%.6f" : ", %.6f", values[i]);
    }
    printf("]\n");
}

static int cpu_expert_forward_from_buffer(
    const unsigned char *expert_buf,
    const ModelConfig *cfg,
    const ExpertLayout *layout,
    CpuExpertScratch *scratch,
    const float *x,
    float *out
) {
    const uint32_t *gate_w = (const uint32_t *)(expert_buf + layout->gate_w_offset);
    const uint16_t *gate_s = (const uint16_t *)(expert_buf + layout->gate_s_offset);
    const uint16_t *gate_b = (const uint16_t *)(expert_buf + layout->gate_b_offset);
    const uint32_t *up_w = (const uint32_t *)(expert_buf + layout->up_w_offset);
    const uint16_t *up_s = (const uint16_t *)(expert_buf + layout->up_s_offset);
    const uint16_t *up_b = (const uint16_t *)(expert_buf + layout->up_b_offset);
    const uint32_t *down_w = (const uint32_t *)(expert_buf + layout->down_w_offset);
    const uint16_t *down_s = (const uint16_t *)(expert_buf + layout->down_s_offset);
    const uint16_t *down_b = (const uint16_t *)(expert_buf + layout->down_b_offset);

    cpu_dequant_matvec_4bit(gate_w, gate_s, gate_b, x, scratch->gate,
                            cfg->moe_intermediate_size, cfg->hidden_size, cfg->group_size);
    cpu_dequant_matvec_4bit(up_w, up_s, up_b, x, scratch->up,
                            cfg->moe_intermediate_size, cfg->hidden_size, cfg->group_size);
    cpu_swiglu(scratch->gate, scratch->up, scratch->act, cfg->moe_intermediate_size);
    cpu_dequant_matvec_4bit(down_w, down_s, down_b, scratch->act, out,
                            cfg->hidden_size, cfg->moe_intermediate_size, cfg->group_size);
    return 1;
}

static int load_expert_buffer(FILE *layer_file, const ExpertLayout *layout, int expert_idx, unsigned char *expert_buf) {
    size_t offset = (size_t)expert_idx * layout->expert_size;
    return read_bytes_at(layer_file, expert_buf, layout->expert_size, offset);
}

static CpuTiming run_single_expert(
    FILE *layer_file,
    const ModelConfig *cfg,
    const ExpertLayout *layout,
    int expert_idx,
    const float *x,
    float *out
) {
    CpuTiming timing = {0};
    CpuExpertScratch scratch;
    unsigned char *expert_buf;
    double t0 = now_ms();
    double t_io_start;
    double t_compute_start;

    expert_buf = (unsigned char *)malloc(layout->expert_size);
    if (!expert_buf) return timing;
    if (!init_cpu_scratch(&scratch, cfg)) {
        free(expert_buf);
        return timing;
    }

    t_io_start = now_ms();
    if (!load_expert_buffer(layer_file, layout, expert_idx, expert_buf)) {
        free_cpu_scratch(&scratch);
        free(expert_buf);
        return timing;
    }
    timing.io_ms = now_ms() - t_io_start;

    t_compute_start = now_ms();
    cpu_expert_forward_from_buffer(expert_buf, cfg, layout, &scratch, x, out);
    timing.compute_ms = now_ms() - t_compute_start;
    timing.total_ms = now_ms() - t0;

    free_cpu_scratch(&scratch);
    free(expert_buf);
    return timing;
}

static CpuTiming run_cpu_moe(
    FILE *layer_file,
    const ModelConfig *cfg,
    const ExpertLayout *layout,
    const int *expert_indices,
    const float *expert_weights,
    int k,
    const float *x,
    float *out
) {
    CpuTiming timing = {0};
    CpuExpertScratch scratch;
    unsigned char *expert_buf;
    float *expert_out;
    int hidden_size;
    int i;

    hidden_size = cfg->hidden_size;
    expert_buf = (unsigned char *)malloc(layout->expert_size);
    expert_out = (float *)malloc((size_t)hidden_size * sizeof(float));
    if (!expert_buf || !expert_out || !init_cpu_scratch(&scratch, cfg)) {
        free(expert_buf);
        free(expert_out);
        return timing;
    }

    memset(out, 0, (size_t)hidden_size * sizeof(float));
    for (i = 0; i < k; i++) {
        int dim;
        double t_io_start = now_ms();
        double t_compute_start;
        if (!load_expert_buffer(layer_file, layout, expert_indices[i], expert_buf)) {
            fprintf(stderr, "ERROR: failed to load expert %d from layer file\n", expert_indices[i]);
            break;
        }
        timing.io_ms += now_ms() - t_io_start;

        t_compute_start = now_ms();
        cpu_expert_forward_from_buffer(expert_buf, cfg, layout, &scratch, x, expert_out);
        timing.compute_ms += now_ms() - t_compute_start;

        for (dim = 0; dim < hidden_size; dim++) {
            out[dim] += expert_weights[i] * expert_out[dim];
        }
    }

    timing.total_ms = timing.io_ms + timing.compute_ms;
    free_cpu_scratch(&scratch);
    free(expert_buf);
    free(expert_out);
    return timing;
}

static void normalize_weights(float *weights, int count) {
    float sum = 0.0f;
    int i;
    for (i = 0; i < count; i++) sum += weights[i];
    if (sum <= 0.0f) return;
    for (i = 0; i < count; i++) weights[i] /= sum;
}

static void build_moe_selection(int *expert_indices, float *weights, int count, int seed_expert, int num_experts) {
    int i;
    for (i = 0; i < count; i++) {
        expert_indices[i] = (seed_expert + i * 997) % num_experts;
        weights[i] = 1.0f / (float)(i + 1);
    }
    normalize_weights(weights, count);
}

static int embed_lookup_from_weights(
    FILE *weights_file,
    const TensorSlice *weight,
    const TensorSlice *scales,
    const TensorSlice *biases,
    const ModelConfig *cfg,
    int token_id,
    float *out
) {
    size_t packed_cols;
    size_t num_groups;
    size_t w_row_size;
    size_t sb_row_size;
    uint32_t *w_row;
    uint16_t *s_row;
    uint16_t *b_row;
    int group_idx;

    if (!weights_file || !weight || !scales || !biases || !cfg || !out) return 0;
    if (token_id < 0 || token_id >= cfg->vocab_size) return 0;

    packed_cols = (size_t)cfg->hidden_size / 8;
    num_groups = (size_t)cfg->hidden_size / (size_t)cfg->group_size;
    w_row_size = packed_cols * sizeof(uint32_t);
    sb_row_size = num_groups * sizeof(uint16_t);

    w_row = (uint32_t *)malloc(w_row_size);
    s_row = (uint16_t *)malloc(sb_row_size);
    b_row = (uint16_t *)malloc(sb_row_size);
    if (!w_row || !s_row || !b_row) {
        free(w_row);
        free(s_row);
        free(b_row);
        return 0;
    }

    if (!read_tensor_row(weights_file, weight, w_row_size, token_id, w_row) ||
        !read_tensor_row(weights_file, scales, sb_row_size, token_id, s_row) ||
        !read_tensor_row(weights_file, biases, sb_row_size, token_id, b_row)) {
        free(w_row);
        free(s_row);
        free(b_row);
        return 0;
    }

    for (group_idx = 0; group_idx < (int)num_groups; group_idx++) {
        float scale = bf16_to_f32(s_row[group_idx]);
        float bias = bf16_to_f32(b_row[group_idx]);
        int packed_idx;
        for (packed_idx = 0; packed_idx < cfg->group_size / 8; packed_idx++) {
            uint32_t packed = w_row[group_idx * (cfg->group_size / 8) + packed_idx];
            int nibble_idx;
            int base = group_idx * cfg->group_size + packed_idx * 8;
            for (nibble_idx = 0; nibble_idx < 8; nibble_idx++) {
                uint32_t q = (packed >> (nibble_idx * 4)) & 0xFu;
                out[base + nibble_idx] = (float)q * scale + bias;
            }
        }
    }

    free(w_row);
    free(s_row);
    free(b_row);
    return 1;
}

static int lm_head_argmax_from_weights(
    FILE *weights_file,
    const TensorSlice *weight,
    const TensorSlice *scales,
    const TensorSlice *biases,
    const ModelConfig *cfg,
    const float *hidden,
    int *out_best_token,
    float *out_best_logit
) {
    size_t packed_cols;
    size_t num_groups;
    size_t w_row_size;
    size_t sb_row_size;
    uint32_t *w_row;
    uint16_t *s_row;
    uint16_t *b_row;
    float top_values[5];
    int top_indices[5];
    int vocab_idx;

    if (!weights_file || !weight || !scales || !biases || !cfg || !hidden || !out_best_token || !out_best_logit) return 0;

    packed_cols = (size_t)cfg->hidden_size / 8;
    num_groups = (size_t)cfg->hidden_size / (size_t)cfg->group_size;
    w_row_size = packed_cols * sizeof(uint32_t);
    sb_row_size = num_groups * sizeof(uint16_t);

    w_row = (uint32_t *)malloc(w_row_size);
    s_row = (uint16_t *)malloc(sb_row_size);
    b_row = (uint16_t *)malloc(sb_row_size);
    if (!w_row || !s_row || !b_row) {
        free(w_row);
        free(s_row);
        free(b_row);
        return 0;
    }

    for (vocab_idx = 0; vocab_idx < 5; vocab_idx++) {
        top_values[vocab_idx] = -INFINITY;
        top_indices[vocab_idx] = -1;
    }

    for (vocab_idx = 0; vocab_idx < cfg->vocab_size; vocab_idx++) {
        float acc = 0.0f;
        int group_idx;
        if (!read_tensor_row(weights_file, weight, w_row_size, vocab_idx, w_row) ||
            !read_tensor_row(weights_file, scales, sb_row_size, vocab_idx, s_row) ||
            !read_tensor_row(weights_file, biases, sb_row_size, vocab_idx, b_row)) {
            free(w_row);
            free(s_row);
            free(b_row);
            return 0;
        }

        for (group_idx = 0; group_idx < (int)num_groups; group_idx++) {
            float scale = bf16_to_f32(s_row[group_idx]);
            float bias = bf16_to_f32(b_row[group_idx]);
            int packed_idx;
            int base_x = group_idx * cfg->group_size;
            for (packed_idx = 0; packed_idx < cfg->group_size / 8; packed_idx++) {
                uint32_t packed = w_row[group_idx * (cfg->group_size / 8) + packed_idx];
                int nibble_idx;
                for (nibble_idx = 0; nibble_idx < 8; nibble_idx++) {
                    uint32_t q = (packed >> (nibble_idx * 4)) & 0xFu;
                    int x_idx = base_x + packed_idx * 8 + nibble_idx;
                    acc += ((float)q * scale + bias) * hidden[x_idx];
                }
            }
        }

        {
            int slot = 0;
            int j;
            for (j = 1; j < 5; j++) {
                if (top_values[j] < top_values[slot]) slot = j;
            }
            if (acc > top_values[slot]) {
                top_values[slot] = acc;
                top_indices[slot] = vocab_idx;
            }
        }
    }

    *out_best_token = top_indices[cpu_argmax(top_values, 5)];
    *out_best_logit = top_values[cpu_argmax(top_values, 5)];
    printf("[lm_head] top-5 tokens: ");
    for (vocab_idx = 0; vocab_idx < 5; vocab_idx++) {
        printf(vocab_idx == 0 ? "%d(%.4f)" : " %d(%.4f)", top_indices[vocab_idx], top_values[vocab_idx]);
    }
    printf("\n");

    free(w_row);
    free(s_row);
    free(b_row);
    return 1;
}

static int dequant_row_dot_from_weights(
    FILE *weights_file,
    const TensorSlice *weight,
    const TensorSlice *scales,
    const TensorSlice *biases,
    int row_idx,
    int in_dim,
    int group_size,
    const float *x,
    float *out_value
) {
    size_t packed_cols;
    size_t num_groups;
    size_t w_row_size;
    size_t sb_row_size;
    uint32_t *w_row;
    uint16_t *s_row;
    uint16_t *b_row;
    float acc = 0.0f;
    int group_idx;

    if (!weights_file || !weight || !scales || !biases || !x || !out_value) return 0;

    packed_cols = (size_t)in_dim / 8;
    num_groups = (size_t)in_dim / (size_t)group_size;
    w_row_size = packed_cols * sizeof(uint32_t);
    sb_row_size = num_groups * sizeof(uint16_t);

    w_row = (uint32_t *)malloc(w_row_size);
    s_row = (uint16_t *)malloc(sb_row_size);
    b_row = (uint16_t *)malloc(sb_row_size);
    if (!w_row || !s_row || !b_row) {
        free(w_row);
        free(s_row);
        free(b_row);
        return 0;
    }

    if (!read_tensor_row(weights_file, weight, w_row_size, row_idx, w_row) ||
        !read_tensor_row(weights_file, scales, sb_row_size, row_idx, s_row) ||
        !read_tensor_row(weights_file, biases, sb_row_size, row_idx, b_row)) {
        free(w_row);
        free(s_row);
        free(b_row);
        return 0;
    }

    for (group_idx = 0; group_idx < (int)num_groups; group_idx++) {
        float scale = bf16_to_f32(s_row[group_idx]);
        float bias = bf16_to_f32(b_row[group_idx]);
        int packed_idx;
        int base_x = group_idx * group_size;
        for (packed_idx = 0; packed_idx < group_size / 8; packed_idx++) {
            uint32_t packed = w_row[group_idx * (group_size / 8) + packed_idx];
            int nibble_idx;
            for (nibble_idx = 0; nibble_idx < 8; nibble_idx++) {
                uint32_t q = (packed >> (nibble_idx * 4)) & 0xFu;
                int x_idx = base_x + packed_idx * 8 + nibble_idx;
                acc += ((float)q * scale + bias) * x[x_idx];
            }
        }
    }

    *out_value = acc;
    free(w_row);
    free(s_row);
    free(b_row);
    return 1;
}

static int dequant_matvec_from_weights(
    FILE *weights_file,
    const TensorSlice *weight,
    const TensorSlice *scales,
    const TensorSlice *biases,
    int out_dim,
    int in_dim,
    int group_size,
    const float *x,
    float *out
) {
    int row_idx;
    if (!weights_file || !weight || !scales || !biases || !x || !out) return 0;
    for (row_idx = 0; row_idx < out_dim; row_idx++) {
        if (!dequant_row_dot_from_weights(weights_file, weight, scales, biases,
                                          row_idx, in_dim, group_size, x, &out[row_idx])) {
            return 0;
        }
    }
    return 1;
}

static int run_shared_expert_from_weights(
    FILE *weights_file,
    const TensorSlice *gate_w,
    const TensorSlice *gate_s,
    const TensorSlice *gate_b,
    const TensorSlice *up_w,
    const TensorSlice *up_s,
    const TensorSlice *up_b,
    const TensorSlice *down_w,
    const TensorSlice *down_s,
    const TensorSlice *down_b,
    const TensorSlice *shared_gate_w,
    const TensorSlice *shared_gate_s,
    const TensorSlice *shared_gate_b,
    const ModelConfig *cfg,
    const float *hidden,
    float *shared_out,
    float *out_shared_gate_score
) {
    float *gate;
    float *up;
    float *act;
    float shared_gate_score;
    int i;

    if (!weights_file || !gate_w || !gate_s || !gate_b || !up_w || !up_s || !up_b ||
        !down_w || !down_s || !down_b || !shared_gate_w || !shared_gate_s || !shared_gate_b ||
        !cfg || !hidden || !shared_out) {
        return 0;
    }

    gate = (float *)malloc((size_t)cfg->shared_expert_intermediate_size * sizeof(float));
    up = (float *)malloc((size_t)cfg->shared_expert_intermediate_size * sizeof(float));
    act = (float *)malloc((size_t)cfg->shared_expert_intermediate_size * sizeof(float));
    if (!gate || !up || !act) {
        free(gate);
        free(up);
        free(act);
        return 0;
    }

    if (!dequant_matvec_from_weights(weights_file, gate_w, gate_s, gate_b,
                                     cfg->shared_expert_intermediate_size,
                                     cfg->hidden_size, cfg->group_size,
                                     hidden, gate) ||
        !dequant_matvec_from_weights(weights_file, up_w, up_s, up_b,
                                     cfg->shared_expert_intermediate_size,
                                     cfg->hidden_size, cfg->group_size,
                                     hidden, up)) {
        free(gate);
        free(up);
        free(act);
        return 0;
    }

    cpu_swiglu(gate, up, act, cfg->shared_expert_intermediate_size);
    if (!dequant_matvec_from_weights(weights_file, down_w, down_s, down_b,
                                     cfg->hidden_size,
                                     cfg->shared_expert_intermediate_size,
                                     cfg->group_size,
                                     act, shared_out) ||
        !dequant_row_dot_from_weights(weights_file, shared_gate_w, shared_gate_s, shared_gate_b,
                                      0, cfg->hidden_size, cfg->group_size,
                                      hidden, &shared_gate_score)) {
        free(gate);
        free(up);
        free(act);
        return 0;
    }

    shared_gate_score = cpu_sigmoid(shared_gate_score);
    for (i = 0; i < cfg->hidden_size; i++) {
        shared_out[i] *= shared_gate_score;
    }
    if (out_shared_gate_score) *out_shared_gate_score = shared_gate_score;

    free(gate);
    free(up);
    free(act);
    return 1;
}

static int route_gate_from_weights(
    FILE *weights_file,
    const TensorSlice *weight,
    const TensorSlice *scales,
    const TensorSlice *biases,
    const ModelConfig *cfg,
    const float *hidden,
    int topk,
    int *expert_indices,
    float *expert_weights,
    int use_hdgl,
    int use_hdgl_semantic,
    float hdgl_alpha,
    int layer_idx,
    int token_id,
    HDGL_History *history
) {
    size_t packed_cols;
    size_t num_groups;
    size_t w_row_size;
    size_t sb_row_size;
    uint32_t *w_row;
    uint16_t *s_row;
    uint16_t *b_row;
    float *gate_scores;
    int expert_idx;

    if (!weights_file || !weight || !scales || !biases || !cfg || !hidden || !expert_indices || !expert_weights) return 0;

    packed_cols = (size_t)cfg->hidden_size / 8;
    num_groups = (size_t)cfg->hidden_size / (size_t)cfg->group_size;
    w_row_size = packed_cols * sizeof(uint32_t);
    sb_row_size = num_groups * sizeof(uint16_t);

    w_row = (uint32_t *)malloc(w_row_size);
    s_row = (uint16_t *)malloc(sb_row_size);
    b_row = (uint16_t *)malloc(sb_row_size);
    gate_scores = (float *)malloc((size_t)cfg->num_experts * sizeof(float));
    if (!w_row || !s_row || !b_row || !gate_scores) {
        free(w_row);
        free(s_row);
        free(b_row);
        free(gate_scores);
        return 0;
    }

    for (expert_idx = 0; expert_idx < cfg->num_experts; expert_idx++) {
        float acc = 0.0f;
        int group_idx;
        if (!read_tensor_row(weights_file, weight, w_row_size, expert_idx, w_row) ||
            !read_tensor_row(weights_file, scales, sb_row_size, expert_idx, s_row) ||
            !read_tensor_row(weights_file, biases, sb_row_size, expert_idx, b_row)) {
            free(w_row);
            free(s_row);
            free(b_row);
            free(gate_scores);
            return 0;
        }
        for (group_idx = 0; group_idx < (int)num_groups; group_idx++) {
            float scale = bf16_to_f32(s_row[group_idx]);
            float bias = bf16_to_f32(b_row[group_idx]);
            int packed_idx;
            int base_x = group_idx * cfg->group_size;
            for (packed_idx = 0; packed_idx < cfg->group_size / 8; packed_idx++) {
                uint32_t packed = w_row[group_idx * (cfg->group_size / 8) + packed_idx];
                int nibble_idx;
                for (nibble_idx = 0; nibble_idx < 8; nibble_idx++) {
                    uint32_t q = (packed >> (nibble_idx * 4)) & 0xFu;
                    int x_idx = base_x + packed_idx * 8 + nibble_idx;
                    acc += ((float)q * scale + bias) * hidden[x_idx];
                }
            }
        }
        gate_scores[expert_idx] = acc;
    }

    cpu_softmax(gate_scores, cfg->num_experts);
    if (use_hdgl && history && g_hdgl_lattice) {
        char tok_buf[48];
        int hdgl_expert;
        snprintf(tok_buf, sizeof(tok_buf), "tok:%d", token_id);
        hdgl_expert = route_token_recursive((Token){ tok_buf, layer_idx * 10000 + token_id }, history);
        if (hdgl_expert >= 0 && hdgl_expert < cfg->num_experts) {
            gate_scores[hdgl_expert] += hdgl_alpha;
        }
        cpu_softmax(gate_scores, cfg->num_experts);
    }

    if (use_hdgl_semantic) {
        int semantic_octave = hdgl_semantic_octave(token_id, layer_idx);
        for (expert_idx = 0; expert_idx < cfg->num_experts; expert_idx++) {
            gate_scores[expert_idx] *= hdgl_semantic_boost(semantic_octave, expert_idx, cfg->num_experts);
        }
        cpu_softmax(gate_scores, cfg->num_experts);
    }

    cpu_topk(gate_scores, cfg->num_experts, topk, expert_indices, expert_weights);
    cpu_normalize_weights(expert_weights, topk);

    free(w_row);
    free(s_row);
    free(b_row);
    free(gate_scores);
    return 1;
}

static int resolve_route_layer_tensors(const char *manifest_json, int layer_idx, RouteLayerTensors *tensors) {
    char gate_w_name[128], gate_s_name[128], gate_b_name[128];
    char shared_gate_proj_w_name[160], shared_gate_proj_s_name[160], shared_gate_proj_b_name[160];
    char shared_up_proj_w_name[160], shared_up_proj_s_name[160], shared_up_proj_b_name[160];
    char shared_down_proj_w_name[160], shared_down_proj_s_name[160], shared_down_proj_b_name[160];
    char shared_expert_gate_w_name[160], shared_expert_gate_s_name[160], shared_expert_gate_b_name[160];

    if (!manifest_json || !tensors) return 0;
    memset(tensors, 0, sizeof(*tensors));

    snprintf(gate_w_name, sizeof(gate_w_name), "model.layers.%d.mlp.gate.weight", layer_idx);
    snprintf(gate_s_name, sizeof(gate_s_name), "model.layers.%d.mlp.gate.scales", layer_idx);
    snprintf(gate_b_name, sizeof(gate_b_name), "model.layers.%d.mlp.gate.biases", layer_idx);
    snprintf(shared_gate_proj_w_name, sizeof(shared_gate_proj_w_name), "model.layers.%d.mlp.shared_expert.gate_proj.weight", layer_idx);
    snprintf(shared_gate_proj_s_name, sizeof(shared_gate_proj_s_name), "model.layers.%d.mlp.shared_expert.gate_proj.scales", layer_idx);
    snprintf(shared_gate_proj_b_name, sizeof(shared_gate_proj_b_name), "model.layers.%d.mlp.shared_expert.gate_proj.biases", layer_idx);
    snprintf(shared_up_proj_w_name, sizeof(shared_up_proj_w_name), "model.layers.%d.mlp.shared_expert.up_proj.weight", layer_idx);
    snprintf(shared_up_proj_s_name, sizeof(shared_up_proj_s_name), "model.layers.%d.mlp.shared_expert.up_proj.scales", layer_idx);
    snprintf(shared_up_proj_b_name, sizeof(shared_up_proj_b_name), "model.layers.%d.mlp.shared_expert.up_proj.biases", layer_idx);
    snprintf(shared_down_proj_w_name, sizeof(shared_down_proj_w_name), "model.layers.%d.mlp.shared_expert.down_proj.weight", layer_idx);
    snprintf(shared_down_proj_s_name, sizeof(shared_down_proj_s_name), "model.layers.%d.mlp.shared_expert.down_proj.scales", layer_idx);
    snprintf(shared_down_proj_b_name, sizeof(shared_down_proj_b_name), "model.layers.%d.mlp.shared_expert.down_proj.biases", layer_idx);
    snprintf(shared_expert_gate_w_name, sizeof(shared_expert_gate_w_name), "model.layers.%d.mlp.shared_expert_gate.weight", layer_idx);
    snprintf(shared_expert_gate_s_name, sizeof(shared_expert_gate_s_name), "model.layers.%d.mlp.shared_expert_gate.scales", layer_idx);
    snprintf(shared_expert_gate_b_name, sizeof(shared_expert_gate_b_name), "model.layers.%d.mlp.shared_expert_gate.biases", layer_idx);

    return manifest_find_tensor_slice(manifest_json, gate_w_name, &tensors->route_w) &&
           manifest_find_tensor_slice(manifest_json, gate_s_name, &tensors->route_s) &&
           manifest_find_tensor_slice(manifest_json, gate_b_name, &tensors->route_b) &&
           manifest_find_tensor_slice(manifest_json, shared_gate_proj_w_name, &tensors->shared_gate_proj_w) &&
           manifest_find_tensor_slice(manifest_json, shared_gate_proj_s_name, &tensors->shared_gate_proj_s) &&
           manifest_find_tensor_slice(manifest_json, shared_gate_proj_b_name, &tensors->shared_gate_proj_b) &&
           manifest_find_tensor_slice(manifest_json, shared_up_proj_w_name, &tensors->shared_up_proj_w) &&
           manifest_find_tensor_slice(manifest_json, shared_up_proj_s_name, &tensors->shared_up_proj_s) &&
           manifest_find_tensor_slice(manifest_json, shared_up_proj_b_name, &tensors->shared_up_proj_b) &&
           manifest_find_tensor_slice(manifest_json, shared_down_proj_w_name, &tensors->shared_down_proj_w) &&
           manifest_find_tensor_slice(manifest_json, shared_down_proj_s_name, &tensors->shared_down_proj_s) &&
           manifest_find_tensor_slice(manifest_json, shared_down_proj_b_name, &tensors->shared_down_proj_b) &&
           manifest_find_tensor_slice(manifest_json, shared_expert_gate_w_name, &tensors->shared_expert_gate_w) &&
           manifest_find_tensor_slice(manifest_json, shared_expert_gate_s_name, &tensors->shared_expert_gate_s) &&
           manifest_find_tensor_slice(manifest_json, shared_expert_gate_b_name, &tensors->shared_expert_gate_b);
}

static int run_route_layer_step(
    FILE *weights_file,
    const char *packed_dir,
    const ModelConfig *cfg,
    const ExpertLayout *layout,
    const RouteLayerTensors *tensors,
    int layer_idx,
    int topk,
    int use_hdgl,
    int use_hdgl_semantic,
    float hdgl_alpha,
    int token_id,
    HDGL_History *history,
    int *expert_indices,
    float *expert_weights,
    float *moe_out,
    float *shared_out,
    float *hidden_out,
    const float *hidden_in,
    RouteStepTiming *timing,
    char *layer_path,
    size_t layer_path_size
) {
    FILE *route_layer_file;
    double t0;
    int i;

    if (!weights_file || !packed_dir || !cfg || !layout || !tensors || !expert_indices || !expert_weights ||
        !moe_out || !shared_out || !hidden_out || !hidden_in || !layer_path || layer_path_size == 0) {
        return 0;
    }

    build_layer_path(layer_path, layer_path_size, packed_dir, layer_idx);
    if (!file_exists(layer_path)) {
        fprintf(stderr, "ERROR: packed expert layer not found for route-token: %s\n", layer_path);
        return 0;
    }

    route_layer_file = fopen(layer_path, "rb");
    if (!route_layer_file) {
        fprintf(stderr, "ERROR: failed to open %s (%s)\n", layer_path, strerror(errno));
        return 0;
    }

    memset(moe_out, 0, (size_t)cfg->hidden_size * sizeof(float));
    memset(shared_out, 0, (size_t)cfg->hidden_size * sizeof(float));
    memset(hidden_out, 0, (size_t)cfg->hidden_size * sizeof(float));
    if (timing) memset(timing, 0, sizeof(*timing));

    t0 = now_ms();
    if (!route_gate_from_weights(weights_file, &tensors->route_w, &tensors->route_s, &tensors->route_b,
                                 cfg, hidden_in, topk, expert_indices, expert_weights,
                                 use_hdgl, use_hdgl_semantic, hdgl_alpha, layer_idx, token_id, history)) {
        fclose(route_layer_file);
        return 0;
    }

    if (timing) {
        timing->moe_timing = run_cpu_moe(route_layer_file, cfg, layout, expert_indices, expert_weights, topk, hidden_in, moe_out);
    } else {
        (void)run_cpu_moe(route_layer_file, cfg, layout, expert_indices, expert_weights, topk, hidden_in, moe_out);
    }
    if (!run_shared_expert_from_weights(weights_file,
                                        &tensors->shared_gate_proj_w, &tensors->shared_gate_proj_s, &tensors->shared_gate_proj_b,
                                        &tensors->shared_up_proj_w, &tensors->shared_up_proj_s, &tensors->shared_up_proj_b,
                                        &tensors->shared_down_proj_w, &tensors->shared_down_proj_s, &tensors->shared_down_proj_b,
                                        &tensors->shared_expert_gate_w, &tensors->shared_expert_gate_s, &tensors->shared_expert_gate_b,
                                        cfg, hidden_in, shared_out,
                                        timing ? &timing->shared_gate_value : NULL)) {
        fclose(route_layer_file);
        return 0;
    }

    for (i = 0; i < cfg->hidden_size; i++) {
        hidden_out[i] = hidden_in[i] + moe_out[i] + shared_out[i];
    }
    if (timing) timing->total_ms = now_ms() - t0;

    fclose(route_layer_file);
    return 1;
}

int main(int argc, char **argv) {
    const char *model_path = DEFAULT_MODEL_PATH;
    const char *manifest_path = NULL;
    const char *weights_path = NULL;
    const char *packed_dir = NULL;
    char manifest_buf[1024];
    char weights_buf[1024];
    char packed_buf[1024];
    char layer_path[1024];
    char *manifest_json;
    FILE *weights_file = NULL;
    FILE *layer_file;
    ModelConfig cfg;
    ExpertLayout layout;
    float *x;
    float *out;
    TensorSlice embed_w = {0}, embed_s = {0}, embed_b = {0};
    TensorSlice lm_w = {0}, lm_s = {0}, lm_b = {0};
    HDGL_History *hdgl_history = NULL;
    int check_only = 0;
    int do_moe = 0;
    int embed_token_id = -1;
    int lm_head_token_id = -1;
    int route_token_id = -1;
    int route_layer_idx = 0;
    int route_layers = 1;
    int route_lm_head = 0;
    int use_hdgl = 0;
    int use_hdgl_semantic = 0;
    float hdgl_alpha = 0.20f;
    const char *hdgl_load_path = NULL;
    int layer_idx = 0;
    int expert_idx = 0;
    int topk = DEFAULT_TOPK;
    int benchmark_iters = 0;
    int i;

    cfg.hidden_size = 4096;
    cfg.num_hidden_layers = 60;
    cfg.num_experts = 512;
    cfg.num_experts_per_tok = 10;
    cfg.vocab_size = 248320;
    cfg.full_attention_interval = 4;
    cfg.moe_intermediate_size = 1024;
    cfg.shared_expert_intermediate_size = 1024;
    cfg.group_size = 64;

    for (i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "--manifest") == 0 && i + 1 < argc) {
            manifest_path = argv[++i];
        } else if (strcmp(argv[i], "--weights") == 0 && i + 1 < argc) {
            weights_path = argv[++i];
        } else if (strcmp(argv[i], "--packed-experts") == 0 && i + 1 < argc) {
            packed_dir = argv[++i];
        } else if (strcmp(argv[i], "--layer") == 0 && i + 1 < argc) {
            layer_idx = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--expert") == 0 && i + 1 < argc) {
            expert_idx = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--k") == 0 && i + 1 < argc) {
            topk = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--embed-token") == 0 && i + 1 < argc) {
            embed_token_id = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--lm-head-token") == 0 && i + 1 < argc) {
            lm_head_token_id = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--route-token") == 0 && i + 1 < argc) {
            route_token_id = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--route-layer") == 0 && i + 1 < argc) {
            route_layer_idx = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--route-layers") == 0 && i + 1 < argc) {
            route_layers = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--route-lm-head") == 0) {
            route_lm_head = 1;
        } else if (strcmp(argv[i], "--hdgl") == 0) {
            use_hdgl = 1;
        } else if (strcmp(argv[i], "--hdgl-semantic") == 0) {
            use_hdgl_semantic = 1;
        } else if (strcmp(argv[i], "--hdgl-alpha") == 0 && i + 1 < argc) {
            hdgl_alpha = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--hdgl-load") == 0 && i + 1 < argc) {
            hdgl_load_path = argv[++i];
        } else if (strcmp(argv[i], "--benchmark") == 0 && i + 1 < argc) {
            benchmark_iters = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--moe") == 0) {
            do_moe = 1;
        } else if (strcmp(argv[i], "--check-only") == 0) {
            check_only = 1;
        } else if (strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Unknown argument: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }

    if (!manifest_path) {
        append_path(manifest_buf, sizeof(manifest_buf), model_path, "model_weights.json");
        manifest_path = manifest_buf;
    }
    if (!weights_path) {
        append_path(weights_buf, sizeof(weights_buf), model_path, "model_weights.bin");
        weights_path = weights_buf;
    }
    if (!packed_dir) {
        append_path(packed_buf, sizeof(packed_buf), model_path, "packed_experts");
        packed_dir = packed_buf;
    }

    printf("=== nonmetal_infer: Windows/native CPU backend ===\n");
    printf("Model:    %s\n", model_path);
    printf("Manifest: %s\n", manifest_path);
    printf("Weights:  %s\n", weights_path);
    printf("Experts:  %s\n", packed_dir);

    if (!file_exists(manifest_path)) {
        fprintf(stderr, "ERROR: manifest not found: %s\n", manifest_path);
        return 1;
    }
    if (check_only && !file_exists(weights_path)) {
        fprintf(stderr, "ERROR: weights not found: %s\n", weights_path);
        return 1;
    }
    if (!file_exists(weights_path)) {
        printf("[warn] non-expert weights not found; continuing with packed-expert CPU slice only\n");
    }

    manifest_json = read_text_file(manifest_path);
    if (!manifest_json) {
        fprintf(stderr, "ERROR: failed to read manifest: %s (%s)\n", manifest_path, strerror(errno));
        return 1;
    }

    cfg.hidden_size = json_find_int(manifest_json, "hidden_size", cfg.hidden_size);
    cfg.num_hidden_layers = json_find_int(manifest_json, "num_hidden_layers", cfg.num_hidden_layers);
    cfg.num_experts = json_find_int(manifest_json, "num_experts", cfg.num_experts);
    cfg.num_experts_per_tok = json_find_int(manifest_json, "num_experts_per_tok", cfg.num_experts_per_tok);
    cfg.vocab_size = json_find_int(manifest_json, "vocab_size", cfg.vocab_size);
    cfg.full_attention_interval = json_find_int(manifest_json, "full_attention_interval", cfg.full_attention_interval);
    cfg.moe_intermediate_size = json_find_int(manifest_json, "moe_intermediate_size", cfg.moe_intermediate_size);
    cfg.shared_expert_intermediate_size = json_find_int(manifest_json, "shared_expert_intermediate_size", cfg.shared_expert_intermediate_size);
    cfg.group_size = json_find_int(manifest_json, "group_size", cfg.group_size);

    printf("[config] hidden=%d layers=%d experts=%d topk=%d vocab=%d full_attn_interval=%d inter=%d group=%d\n",
           cfg.hidden_size, cfg.num_hidden_layers, cfg.num_experts,
           cfg.num_experts_per_tok, cfg.vocab_size, cfg.full_attention_interval,
           cfg.moe_intermediate_size, cfg.group_size);

    if (!validate_manifest_contract(manifest_json, &cfg)) {
        free(manifest_json);
        fprintf(stderr, "ERROR: canonical manifest contract validation failed\n");
        return 1;
    }
    printf("[manifest] contract validation passed\n");

    if (!compute_expert_layout(&cfg, &layout)) {
        fprintf(stderr, "ERROR: unsupported packed expert geometry (hidden=%d inter=%d group=%d)\n",
                cfg.hidden_size, cfg.moe_intermediate_size, cfg.group_size);
        return 1;
    }

    printf("[layout] expert_size=%zu bytes gate=%zu up=%zu down=%zu\n",
           layout.expert_size, layout.gate_w_size, layout.up_w_size, layout.down_w_size);

    if (file_exists(weights_path)) {
        if (!manifest_find_tensor_slice(manifest_json, "model.embed_tokens.weight", &embed_w) ||
            !manifest_find_tensor_slice(manifest_json, "model.embed_tokens.scales", &embed_s) ||
            !manifest_find_tensor_slice(manifest_json, "model.embed_tokens.biases", &embed_b) ||
            !manifest_find_tensor_slice(manifest_json, "lm_head.weight", &lm_w) ||
            !manifest_find_tensor_slice(manifest_json, "lm_head.scales", &lm_s) ||
            !manifest_find_tensor_slice(manifest_json, "lm_head.biases", &lm_b)) {
            fprintf(stderr, "ERROR: failed to resolve required embedding/lm_head tensors from manifest\n");
            free(manifest_json);
            return 1;
        }
    }

    if (check_only) {
        build_layer_path(layer_path, sizeof(layer_path), packed_dir, 0);
        if (!file_exists(layer_path)) {
            fprintf(stderr, "ERROR: packed expert layer not found: %s\n", layer_path);
            return 1;
        }
        printf("[check] local artifact validation complete\n");
        free(manifest_json);
        return 0;
    }

    if (embed_token_id >= 0 || lm_head_token_id >= 0 || route_token_id >= 0) {
        int token_id = embed_token_id >= 0 ? embed_token_id : (lm_head_token_id >= 0 ? lm_head_token_id : route_token_id);
        double t0;
        if (!file_exists(weights_path)) {
            fprintf(stderr, "ERROR: weights not found for non-expert mode: %s\n", weights_path);
            free(manifest_json);
            return 1;
        }
        token_id = clamp_int(token_id, 0, cfg.vocab_size - 1);
        weights_file = fopen(weights_path, "rb");
        if (!weights_file) {
            fprintf(stderr, "ERROR: failed to open %s (%s)\n", weights_path, strerror(errno));
            free(manifest_json);
            return 1;
        }
        x = (float *)malloc((size_t)cfg.hidden_size * sizeof(float));
        if (!x) {
            fprintf(stderr, "ERROR: failed to allocate embedding buffer\n");
            fclose(weights_file);
            free(manifest_json);
            return 1;
        }

        t0 = now_ms();
        if (!embed_lookup_from_weights(weights_file, &embed_w, &embed_s, &embed_b, &cfg, token_id, x)) {
            fprintf(stderr, "ERROR: embedding lookup failed for token %d\n", token_id);
            fclose(weights_file);
            free(manifest_json);
            free(x);
            return 1;
        }
        printf("[mode] embedding lookup: token=%d (%.2f ms)\n", token_id, now_ms() - t0);
        print_vector_sample("[embed] x[0..7] = ", x, cfg.hidden_size);

        if (lm_head_token_id >= 0) {
            int best_token = -1;
            float best_logit = 0.0f;
            t0 = now_ms();
            if (!lm_head_argmax_from_weights(weights_file, &lm_w, &lm_s, &lm_b, &cfg, x, &best_token, &best_logit)) {
                fprintf(stderr, "ERROR: lm_head streaming pass failed\n");
                fclose(weights_file);
                free(manifest_json);
                free(x);
                return 1;
            }
            printf("[mode] lm_head argmax: token=%d -> next=%d logit=%.4f (%.2f ms)\n",
                   token_id, best_token, best_logit, now_ms() - t0);
        }

        if (route_token_id >= 0) {
            RouteLayerTensors route_tensors;
            RouteStepTiming route_timing = {0};
            int *expert_indices;
            float *expert_weights;
            float *moe_out;
            float *shared_out;
            float *combined_out;
            int route_last_layer;
            int current_layer;
            int best_token = -1;
            float best_logit = 0.0f;
            route_layer_idx = clamp_int(route_layer_idx, 0, cfg.num_hidden_layers - 1);
            route_layers = clamp_int(route_layers, 1, cfg.num_hidden_layers - route_layer_idx);
            route_last_layer = route_layer_idx + route_layers - 1;
            topk = clamp_int(topk, 1, cfg.num_experts_per_tok);
            hdgl_alpha = clamp_float(hdgl_alpha, 0.0f, 1.0f);

            expert_indices = (int *)malloc((size_t)topk * sizeof(int));
            expert_weights = (float *)malloc((size_t)topk * sizeof(float));
            moe_out = (float *)calloc((size_t)cfg.hidden_size, sizeof(float));
            shared_out = (float *)calloc((size_t)cfg.hidden_size, sizeof(float));
            combined_out = (float *)calloc((size_t)cfg.hidden_size, sizeof(float));
            if (!expert_indices || !expert_weights || !moe_out || !shared_out || !combined_out) {
                fprintf(stderr, "ERROR: failed to allocate routing buffers\n");
                fclose(weights_file);
                free(manifest_json);
                free(x);
                free(expert_indices);
                free(expert_weights);
                free(moe_out);
                free(shared_out);
                free(combined_out);
                return 1;
            }

            if (use_hdgl) {
                int hdgl_instances = cfg.hidden_size > 0 ? cfg.hidden_size : 4096;
                hdgl_history = (HDGL_History *)calloc((size_t)cfg.num_hidden_layers, sizeof(HDGL_History));
                if (!hdgl_history) {
                    fprintf(stderr, "ERROR: failed to allocate HDGL history\n");
                    fclose(weights_file);
                    free(manifest_json);
                    free(x);
                    free(expert_indices);
                    free(expert_weights);
                    free(moe_out);
                    free(shared_out);
                    free(combined_out);
                    return 1;
                }
                printf("[HDGL-28] Lattice dimensions: instances=%d slots_per_instance=%d\n",
                       hdgl_instances, BLZ_SLOTS_PER_INST);
                g_hdgl_lattice = lattice_init(hdgl_instances, BLZ_SLOTS_PER_INST);
                if (!g_hdgl_lattice) {
                    fprintf(stderr,
                            "ERROR: lattice_init failed (instances=%d slots_per_instance=%d)\n",
                            hdgl_instances, BLZ_SLOTS_PER_INST);
                    fclose(weights_file);
                    free(manifest_json);
                    free(x);
                    free(expert_indices);
                    free(expert_weights);
                    free(moe_out);
                    free(shared_out);
                    free(combined_out);
                    free(hdgl_history);
                    return 1;
                }
                if (hdgl_load_path) {
                    init_apa_constants();
                    if (!hdgl_load_lattice(g_hdgl_lattice, hdgl_load_path)) {
                        bootloader_init_lattice(g_hdgl_lattice, 50);
                    }
                } else {
                    bootloader_init_lattice(g_hdgl_lattice, 50);
                }
                hdgl_router_init(g_hdgl_lattice, cfg.num_experts);
            }

            printf("[mode] route token: token=%d layers=%d start_layer=%d end_layer=%d topk=%d hdgl=%s alpha=%.2f\n",
                   token_id, route_layers, route_layer_idx, route_last_layer,
                   topk, use_hdgl ? "on" : "off", hdgl_alpha);
                 printf("[mode] semantic routing bias: %s\n", use_hdgl_semantic ? "on" : "off");
            for (current_layer = route_layer_idx; current_layer <= route_last_layer; current_layer++) {
                if (!resolve_route_layer_tensors(manifest_json, current_layer, &route_tensors)) {
                    fprintf(stderr, "ERROR: failed to resolve gate/shared tensors for layer %d\n", current_layer);
                    if (g_hdgl_lattice) {
                        lattice_free(g_hdgl_lattice);
                        g_hdgl_lattice = NULL;
                    }
                    fclose(weights_file);
                    free(manifest_json);
                    free(x);
                    free(expert_indices);
                    free(expert_weights);
                    free(moe_out);
                    free(shared_out);
                    free(combined_out);
                    free(hdgl_history);
                    return 1;
                }
                if (!run_route_layer_step(weights_file, packed_dir, &cfg, &layout,
                                          &route_tensors, current_layer, topk,
                                          use_hdgl, use_hdgl_semantic, hdgl_alpha, token_id,
                                          hdgl_history ? &hdgl_history[current_layer] : NULL,
                                          expert_indices, expert_weights,
                                          moe_out, shared_out, combined_out, x,
                                          &route_timing, layer_path, sizeof(layer_path))) {
                    fprintf(stderr, "ERROR: routed layer step failed for layer %d\n", current_layer);
                    if (g_hdgl_lattice) {
                        lattice_free(g_hdgl_lattice);
                        g_hdgl_lattice = NULL;
                    }
                    fclose(weights_file);
                    free(manifest_json);
                    free(x);
                    free(expert_indices);
                    free(expert_weights);
                    free(moe_out);
                    free(shared_out);
                    free(combined_out);
                    free(hdgl_history);
                    return 1;
                }
                printf("[route][layer %d] experts:", current_layer);
                for (i = 0; i < topk; i++) {
                    printf(i == 0 ? " %d(%.4f)" : " %d(%.4f)", expert_indices[i], expert_weights[i]);
                }
                if (use_hdgl_semantic) {
                    int oct = hdgl_semantic_octave(token_id, current_layer);
                    printf("  octave=%d", oct);
                }
                printf("\n");
                printf("[route-step][layer %d] routed_io=%.2f ms routed_compute=%.2f ms moe_total=%.2f ms shared_gate=%.4f total=%.2f ms\n",
                       current_layer,
                       route_timing.moe_timing.io_ms,
                       route_timing.moe_timing.compute_ms,
                       route_timing.moe_timing.total_ms,
                       route_timing.shared_gate_value,
                       route_timing.total_ms);
                print_vector_sample("[route-step] hidden[0..7] = ", combined_out, cfg.hidden_size);
                memcpy(x, combined_out, (size_t)cfg.hidden_size * sizeof(float));
            }

            if (route_lm_head) {
                t0 = now_ms();
                if (!lm_head_argmax_from_weights(weights_file, &lm_w, &lm_s, &lm_b, &cfg, x, &best_token, &best_logit)) {
                    fprintf(stderr, "ERROR: lm_head streaming pass failed after routed layer stack\n");
                    if (g_hdgl_lattice) {
                        lattice_free(g_hdgl_lattice);
                        g_hdgl_lattice = NULL;
                    }
                    fclose(weights_file);
                    free(manifest_json);
                    free(x);
                    free(expert_indices);
                    free(expert_weights);
                    free(moe_out);
                    free(shared_out);
                    free(combined_out);
                    free(hdgl_history);
                    return 1;
                }
                printf("[route-lm-head] layers=%d final_next=%d logit=%.4f (%.2f ms)\n",
                       route_layers, best_token, best_logit, now_ms() - t0);
            }
            if (route_layers > 1) {
                print_vector_sample("[route-final] hidden[0..7] = ", x, cfg.hidden_size);
            }

            /* A/B harness summary: per-layer octave + top expert table.
             * Emitted when --hdgl-semantic is active so the caller can diff
             * gate-only vs gate+semantic runs by comparing this block. */
            if (use_hdgl_semantic) {
                printf("\n[hdgl-semantic][A/B summary] layers=%d token=%d alpha=%.2f\n",
                       route_layers, token_id, hdgl_alpha);
                printf("  %-8s %-8s %-12s %-10s\n",
                       "layer", "octave", "expert[0]", "w[0]");
                /* Re-derive per-layer octave from token/layer — same function
                 * called during gating, so this is an exact mirror. */
                for (int sl = route_layer_idx; sl <= route_last_layer; sl++) {
                    int oct = hdgl_semantic_octave(token_id, sl);
                    printf("  %-8d %-8d (see [route][layer %d] above)\n",
                           sl, oct, sl);
                }
                printf("[hdgl-semantic][A/B end]\n\n");
            }

            if (g_hdgl_lattice) {
                lattice_free(g_hdgl_lattice);
                g_hdgl_lattice = NULL;
            }
            free(hdgl_history);
            free(expert_indices);
            free(expert_weights);
            free(moe_out);
            free(shared_out);
            free(combined_out);
        }

        fclose(weights_file);
        free(manifest_json);
        free(x);
        return 0;
    }

    free(manifest_json);

    layer_idx = clamp_int(layer_idx, 0, cfg.num_hidden_layers - 1);
    expert_idx = clamp_int(expert_idx, 0, cfg.num_experts - 1);
    topk = clamp_int(topk, 1, cfg.num_experts);
    benchmark_iters = clamp_int(benchmark_iters, 0, MAX_BENCHMARK_ITERS);

    build_layer_path(layer_path, sizeof(layer_path), packed_dir, layer_idx);
    if (!file_exists(layer_path)) {
        fprintf(stderr, "ERROR: packed expert layer not found: %s\n", layer_path);
        return 1;
    }

    layer_file = fopen(layer_path, "rb");
    if (!layer_file) {
        fprintf(stderr, "ERROR: failed to open %s (%s)\n", layer_path, strerror(errno));
        return 1;
    }

    x = (float *)malloc((size_t)cfg.hidden_size * sizeof(float));
    out = (float *)malloc((size_t)cfg.hidden_size * sizeof(float));
    if (!x || !out) {
        fprintf(stderr, "ERROR: failed to allocate CPU buffers\n");
        fclose(layer_file);
        free(x);
        free(out);
        return 1;
    }
    fill_input_vector(x, cfg.hidden_size);
    print_vector_sample("[input] x[0..7] = ", x, cfg.hidden_size);

    if (!do_moe) {
        CpuTiming timing = run_single_expert(layer_file, &cfg, &layout, expert_idx, x, out);
        if (timing.total_ms <= 0.0) {
            fprintf(stderr, "ERROR: CPU expert execution failed\n");
            fclose(layer_file);
            free(x);
            free(out);
            return 1;
        }

        printf("[mode] single expert forward: layer=%d expert=%d\n", layer_idx, expert_idx);
        print_vector_sample("[result] out[0..7] = ", out, cfg.hidden_size);
        printf("[timing] io=%.2f ms compute=%.2f ms total=%.2f ms\n",
               timing.io_ms, timing.compute_ms, timing.total_ms);

        if (benchmark_iters > 0) {
            double io_total = 0.0;
            double compute_total = 0.0;
            double total_total = 0.0;
            for (i = 0; i < benchmark_iters; i++) {
                CpuTiming bench = run_single_expert(layer_file, &cfg, &layout, expert_idx, x, out);
                io_total += bench.io_ms;
                compute_total += bench.compute_ms;
                total_total += bench.total_ms;
            }
            printf("[bench] %d iterations avg: io=%.2f ms compute=%.2f ms total=%.2f ms\n",
                   benchmark_iters,
                   io_total / benchmark_iters,
                   compute_total / benchmark_iters,
                   total_total / benchmark_iters);
        }
    } else {
        int *expert_indices = (int *)malloc((size_t)topk * sizeof(int));
        float *weights = (float *)malloc((size_t)topk * sizeof(float));
        CpuTiming timing;

        if (!expert_indices || !weights) {
            fprintf(stderr, "ERROR: failed to allocate MoE routing buffers\n");
            fclose(layer_file);
            free(x);
            free(out);
            free(expert_indices);
            free(weights);
            return 1;
        }

        build_moe_selection(expert_indices, weights, topk, expert_idx, cfg.num_experts);
        printf("[mode] cpu moe slice: layer=%d k=%d experts=", layer_idx, topk);
        for (i = 0; i < topk; i++) {
            printf(i == 0 ? "%d(%.3f)" : " %d(%.3f)", expert_indices[i], weights[i]);
        }
        printf("\n");

        timing = run_cpu_moe(layer_file, &cfg, &layout, expert_indices, weights, topk, x, out);
        if (timing.total_ms <= 0.0) {
            fprintf(stderr, "ERROR: CPU MoE execution failed\n");
            fclose(layer_file);
            free(x);
            free(out);
            free(expert_indices);
            free(weights);
            return 1;
        }

        print_vector_sample("[result] out[0..7] = ", out, cfg.hidden_size);
        printf("[timing] io=%.2f ms compute=%.2f ms total=%.2f ms\n",
               timing.io_ms, timing.compute_ms, timing.total_ms);

        if (benchmark_iters > 0) {
            double io_total = 0.0;
            double compute_total = 0.0;
            double total_total = 0.0;
            for (i = 0; i < benchmark_iters; i++) {
                CpuTiming bench = run_cpu_moe(layer_file, &cfg, &layout, expert_indices, weights, topk, x, out);
                io_total += bench.io_ms;
                compute_total += bench.compute_ms;
                total_total += bench.total_ms;
            }
            printf("[bench] %d iterations avg: io=%.2f ms compute=%.2f ms total=%.2f ms\n",
                   benchmark_iters,
                   io_total / benchmark_iters,
                   compute_total / benchmark_iters,
                   total_total / benchmark_iters);
        }

        free(expert_indices);
        free(weights);
    }

    fclose(layer_file);
    free(x);
    free(out);
    return 0;
}