// hdgl_lattice_generator.c - HDGL-28 v2.0 lattice pre-seeder
//
// Generates hdgl_lattice.bin: a pre-seeded lattice state file that can be
// loaded by infer/chat instead of re-seeding from scratch at startup.

#include "hdgl_bootloaderz.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

#define MAGIC    "HDGL"
#define VERSION  0x00020000u

typedef struct {
    uint64_t mantissa_word0;
    int64_t  exponent;
    double   phase;
    double   freq;
    uint32_t state_flags;
    uint32_t strand_idx;
} SlotRecord;

static void write_u32(FILE *f, uint32_t v) { fwrite(&v, 4, 1, f); }
static void write_f64(FILE *f, double   v) { fwrite(&v, 8, 1, f); }

static char *read_text_file(const char *path) {
    FILE *f;
    long size;
    char *buf;

    if (!path) return NULL;
    f = fopen(path, "rb");
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
    if (!json || !key) return fallback;
    snprintf(needle, sizeof(needle), "\"%s\"", key);
    p = strstr(json, needle);
    if (!p) return fallback;
    p = strchr(p, ':');
    if (!p) return fallback;
    p++;
    while (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n') p++;
    return (int)strtol(p, NULL, 10);
}

int main(int argc, char **argv) {
    int    num_instances   = 0;
    int    slots_per       = BLZ_SLOTS_PER_INST;
    int    steps           = 200;
    const char *outfile    = "hdgl_lattice.bin";
    const char *manifest   = "model_weights.json";
    unsigned long seed     = (unsigned long)time(NULL);
    int    explicit_instances = 0;
    double beta            = 0.1;   /* URF §1 fractional β; default h-domain */

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--instances")   && i+1 < argc) { num_instances = atoi(argv[++i]); explicit_instances = 1; }
        else if (!strcmp(argv[i], "--slots-per") && i+1 < argc) { slots_per = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--steps")   && i+1 < argc) { steps = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--output")  && i+1 < argc) { outfile = argv[++i]; }
        else if (!strcmp(argv[i], "--seed")    && i+1 < argc) { seed = (unsigned long)atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--manifest") && i+1 < argc) { manifest = argv[++i]; }
        else if (!strcmp(argv[i], "--beta")    && i+1 < argc) { beta = atof(argv[++i]); }
        else { fprintf(stderr, "Unknown option: %s\n", argv[i]); return 1; }
    }

    if (!explicit_instances) {
        char *manifest_json = read_text_file(manifest);
        if (manifest_json) {
            num_instances = json_find_int(manifest_json, "hidden_size", 4096);
            free(manifest_json);
            printf("[hdgl_lattice_generator] Using hidden_size=%d from manifest: %s\n",
                   num_instances, manifest);
        } else {
            num_instances = 4096;
            printf("[hdgl_lattice_generator] Manifest not found/readable (%s); using default instances=%d\n",
                   manifest, num_instances);
        }
    }
    if (num_instances <= 0) {
        fprintf(stderr, "ERROR: invalid instance count: %d\n", num_instances);
        return 1;
    }

    srand((unsigned int)seed);
    printf("[hdgl_lattice_generator] v%s\n", HDGL_VERSION_STR);
    printf("  instances=%d  slots_per=%d  steps=%d  seed=%lu  beta=%.4f\n",
           num_instances, slots_per, steps, seed, beta);
    printf("  output=%s\n", outfile);

    lattice_set_beta(beta);
    HDGLLattice *lat = lattice_init(num_instances, slots_per);
    if (!lat) { fprintf(stderr, "ERROR: lattice_init failed.\n"); return 1; }
    g_hdgl_lattice = lat;

    bootloader_init_lattice(lat, steps);

    FILE *f = fopen(outfile, "wb");
    if (!f) { fprintf(stderr, "ERROR: cannot open %s\n", outfile); return 1; }

    fwrite(MAGIC, 1, 4, f);
    write_u32(f, VERSION);
    write_u32(f, (uint32_t)num_instances);
    write_u32(f, (uint32_t)slots_per);
    write_f64(f, lat->time);
    write_f64(f, lat->omega);
    write_f64(f, lat->phase_var);

    int total = num_instances * slots_per;
    int written = 0, skipped = 0;

    for (int i = 0; i < total; i++) {
        Slot4096 *s = lattice_get_slot(lat, i);
        SlotRecord rec = {0};
        if (s && s->mantissa_words) {
            rec.mantissa_word0 = s->mantissa_words[0];
            rec.exponent       = s->exponent;
            rec.phase          = s->phase;
            rec.freq           = s->freq;
            rec.state_flags    = s->state_flags;
            rec.strand_idx     = (uint32_t)(i % SPIRAL8_GEOMETRIES);
            written++;
        } else {
            skipped++;
        }
        fwrite(&rec, sizeof(rec), 1, f);
    }

    fclose(f);

    printf("[hdgl_lattice_generator] Done: %d slots written, %d skipped\n", written, skipped);
    printf("  File: %s  (%.1f MB)\n", outfile,
           (double)(total * sizeof(SlotRecord) + 36) / (1024.0*1024.0));

    g_hdgl_lattice = NULL;
    lattice_free(lat);
    return 0;
}
