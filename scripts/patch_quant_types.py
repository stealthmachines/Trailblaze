#!/usr/bin/env python3
"""
patch_quant_types.py — Add Q2_K and Q3_K dequant to tb_gguf.c

Q2_K (qtype 10): 84 bytes/superblock, 256 weights
  layout: [scales:16][qs:64][d:f16 2][dmin:f16 2]

Q3_K (qtype 11): 110 bytes/superblock, 256 weights
  layout: [hmask:32][qs:64][scales:12][d:f16 2]

Also updates tb_gguf_dequant_row() and tb_gguf_dequant_matvec() switch
statements with the new qtype cases.
"""
import sys, re

TARGET = "v0.3/layer4/tb_gguf.c"

import os
base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
path = os.path.join(base, "layer4", "tb_gguf.c")

with open(path, 'rb') as f:
    raw = f.read()
text = raw.decode('utf-8', errors='replace')

# ── 1. Insert Q2_K and Q3_K block dequant functions after Q6_K ───────────────
Q2K_Q3K_FUNCTIONS = r"""
/* ── Q2_K superblock (84 bytes, 256 weights) ─────────────────────────────── */
/* layout: [scales:16][qs:64][d:f16 2][dmin:f16 2]                            */
/* 16 groups of 16 weights; each group has 4-bit scale + 4-bit min            */
static void dequant_q2_k_block(const uint8_t *sb, float *out) {
    const uint8_t  *scales = sb;            /* 16 bytes: lo4=scale, hi4=min  */
    const uint8_t  *qs     = sb + 16;       /* 64 bytes: 4×2-bit per byte    */
    float d    = f16_to_f32(*(const uint16_t*)(sb + 80));
    float dmin = f16_to_f32(*(const uint16_t*)(sb + 82));
    for (int g = 0; g < 16; g++) {
        float sc = d    * (float)(scales[g] & 0x0F);
        float mn = dmin * (float)(scales[g] >>    4);
        for (int i = 0; i < 16; i++) {
            int wi    = g * 16 + i;
            int shift = (wi & 3) << 1;                  /* (wi%4)*2           */
            int q     = (qs[wi >> 2] >> shift) & 0x03;  /* 2-bit, 0..3        */
            out[wi]   = sc * (float)q - mn;
        }
    }
}

/* ── Q3_K superblock (110 bytes, 256 weights) ────────────────────────────── */
/* layout: [hmask:32][qs:64][scales:12][d:f16 2]                              */
/* 3-bit quants = low2 from qs + high1 from hmask; 16 groups, 6-bit scales   */
static void dequant_q3_k_block(const uint8_t *sb, float *out) {
    const uint8_t  *hmask  = sb;            /* 32 bytes: high bit of quant   */
    const uint8_t  *qs     = sb + 32;       /* 64 bytes: low 2 bits          */
    const uint8_t  *sc_raw = sb + 96;       /* 12 bytes: packed 6-bit scales */
    float d_all = f16_to_f32(*(const uint16_t*)(sb + 108));

    /* Unpack 16 signed 6-bit scales from 12 bytes
     * From ggml dequantize_row_q3_K:
     *   is  = (j < 8) ? j : (j - 8)
     *   lo4 = (sc_raw[is/2] >> (4*(is&1))) & 0xF
     *   hi2 = (sc_raw[8 + is/4] >> (2*(is%4))) & 0x3
     *   scale = d_all * ((lo4 | (hi2 << 4)) - 32)          */
    float scales[16];
    for (int j = 0; j < 16; j++) {
        int is  = (j < 8) ? j : (j - 8);
        int lo4 = (sc_raw[is >> 1] >> ((is & 1) << 2)) & 0xF;
        int hi2 = (sc_raw[8 + (is >> 2)] >> ((is & 3) << 1)) & 0x3;
        scales[j] = d_all * (float)((lo4 | (hi2 << 4)) - 32);
    }

    /* Unpack 3-bit quants and dequantise */
    for (int i = 0; i < 256; i++) {
        int low2  = (qs[i >> 2] >> ((i & 3) << 1)) & 0x03;
        int high  = (hmask[i >> 3] >> (i & 7)) & 0x01;
        int q3    = low2 | (high << 2);     /* 0..7                          */
        out[i]    = scales[i >> 4] * (float)(q3 - 4);  /* centered -4..3    */
    }
}

"""

# Insert after Q6_K block function
Q6K_SENTINEL = "/* ── Public dequantise-one-row"
if Q6K_SENTINEL not in text:
    print(f"ERROR: sentinel '{Q6K_SENTINEL}' not found — aborting")
    sys.exit(1)

if "dequant_q2_k_block" in text:
    print("Q2_K/Q3_K functions already present — skipping function insertion")
else:
    text = text.replace(Q6K_SENTINEL,
                        Q2K_Q3K_FUNCTIONS + Q6K_SENTINEL, 1)
    print("Inserted dequant_q2_k_block() and dequant_q3_k_block()")

# ── 2. Add case 10/11 to tb_gguf_dequant_row() switch ────────────────────────
# Use regex to find and replace the Q6_K case block in dequant_row
# Insert new cases just before the existing Q6_K case
Q2K_INSERT = """    case 10: /* Q2_K - 84 bytes/superblock, 256 weights */
        while (n_out < n_weights) {
            dequant_q2_k_block(p, out + n_out);
            p += 84;
            n_out += 256;
        }
        break;
    case 11: /* Q3_K - 110 bytes/superblock, 256 weights */
        while (n_out < n_weights) {
            dequant_q3_k_block(p, out + n_out);
            p += 110;
            n_out += 256;
        }
        break;
    case 13: /* Q5_K - 176 bytes/superblock, 256 weights (stub) */
        while (n_out < n_weights) {
            memset(out + n_out, 0, 256 * sizeof(float)); /* TODO full Q5_K */
            p += 176;
            n_out += 256;
        }
        break;
    case 15: /* Q8_K - 292 bytes/superblock, 256 weights (stub) */
        while (n_out < n_weights) {
            memset(out + n_out, 0, 256 * sizeof(float)); /* TODO full Q8_K */
            p += 292;
            n_out += 256;
        }
        break;
    """

if "case 10: /* Q2_K" in text:
    print("Q2_K/Q3_K cases in dequant_row already present - skipping")
else:
    # Insert before the case 14 (Q6_K) in the dequant_row switch
    # File has CRLF (\r\n) line endings — use re.DOTALL with flexible newline
    pattern = re.compile(
        r'(    case 14: /\* Q6_K \*/\r?\n'
        r'        while \(n_out < n_weights\) \{\r?\n'
        r'            dequant_q6_k_block\(p, out \+ n_out\);\r?\n'
        r'            p \+= 210;\r?\n'
        r'            n_out \+= 256;\r?\n'
        r'        \}\r?\n'
        r'        break;\r?\n'
        r'    default:)'
    )
    m = pattern.search(text)
    if m:
        text = text[:m.start()] + Q2K_INSERT + text[m.start():]
        print("Added case 10/11/13/15 to tb_gguf_dequant_row() switch")
    else:
        print("ERROR: Q6_K case block not found in dequant_row — check manually")
        sys.exit(1)

# ── 3. Add case 10/11 to tb_gguf_dequant_matvec() switch ─────────────────────
MATVEC_INSERT = """    case 10: block_weights = 256;     block_bytes = 84;  break;  /* Q2_K */
    case 11: block_weights = 256;     block_bytes = 110; break;  /* Q3_K */
    case 13: block_weights = 256;     block_bytes = 176; break;  /* Q5_K */
    case 15: block_weights = 256;     block_bytes = 292; break;  /* Q8_K */
    """

if "case 10: block_weights" in text:
    print("Q2_K/Q3_K matvec cases already present - skipping")
else:
    # Insert before the Q6_K matvec case
    mv_pattern = re.compile(
        r'(    case 14: block_weights = 256;[ ]+block_bytes = 210;[^\n]*\n)'
    )
    m2 = mv_pattern.search(text)
    if m2:
        text = text[:m2.start()] + MATVEC_INSERT + text[m2.start():]
        print("Added case 10/11/13/15 to tb_gguf_dequant_matvec() switch")
    else:
        print("ERROR: Q6_K matvec case not found - check manually")
        sys.exit(1)

# ── Write back ────────────────────────────────────────────────────────────────
with open(path, 'wb') as f:
    f.write(text.encode('utf-8'))
print(f"Wrote {path}")
