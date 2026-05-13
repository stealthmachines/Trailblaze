#!/usr/bin/env python3
"""
patch_q5k_q8k.py — Replace Q5_K and Q8_K stubs with full dequant implementations

Q5_K (176 bytes/superblock):
  d[f16] dmin[f16] scales[12] qh[32] qs[128]
  Same 6-bit scale packing as Q4_K (reuse q4_k_unpack_scales).
  5-bit weight: lo4 from qs nibble | high bit from qh array.

Q8_K (292 bytes/superblock):
  d[f32] qs[256 int8_t] bsums[16 int16_t]
  Trivial: out[i] = d * qs[i]
"""
import os, re, sys

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
path = os.path.join(base, "layer4", "tb_gguf.c")

with open(path, 'rb') as f:
    raw = f.read()
text = raw.decode('utf-8', errors='replace')

changes = 0

# ── 1. Insert dequant_q5_k_block + dequant_q8_k_block before the public row function ──
ANCHOR = '/* ── Public dequantise-one-row'
Q5K_Q8K_FUNCS = '''\
/* ── Q5_K superblock (176 bytes, 256 weights) ───────────────────────────── */
/* layout: [d:f16 2][dmin:f16 2][scales:12][qh:32][qs:128]                   */
/* Same 6-bit scale packing as Q4_K.  5-bit weight = qs_nibble | (qh_bit<<4) */
static void dequant_q5_k_block(const uint8_t *sb, float *out) {
    float d    = f16_to_f32(*(const uint16_t*)(sb + 0));
    float dmin = f16_to_f32(*(const uint16_t*)(sb + 2));
    const uint8_t *sc = sb + 4;   /* scales[12]  */
    const uint8_t *qh = sb + 16;  /* qh[32]      */
    const uint8_t *qs = sb + 48;  /* qs[128]     */

    float scales[8], mins[8];
    q4_k_unpack_scales(sc, d, dmin, scales, mins);

    for (int i = 0; i < 256; i++) {
        int g      = i / 32;
        int qh_bit = (qh[i >> 3] >> (i & 7)) & 1;
        int lo4    = (qs[i >> 1] >> ((i & 1) << 2)) & 0xF;
        int q5     = lo4 | (qh_bit << 4);   /* [0..31] */
        out[i]     = scales[g] * (float)q5 - mins[g];
    }
}

/* ── Q8_K superblock (292 bytes, 256 weights) ───────────────────────────── */
/* layout: [d:f32 4][qs:256 int8_t][bsums:32 int16_t]                        */
static void dequant_q8_k_block(const uint8_t *sb, float *out) {
    float d = *(const float *)(sb + 0);
    const int8_t *qs = (const int8_t *)(sb + 4);
    for (int i = 0; i < 256; i++)
        out[i] = d * (float)qs[i];
}

'''

if 'dequant_q5_k_block' in text:
    print("Q5_K block function already present — skipping insertion")
else:
    # Find anchor (handle CRLF or LF)
    idx = text.find(ANCHOR)
    if idx == -1:
        print(f"ERROR: anchor '{ANCHOR}' not found in tb_gguf.c")
        sys.exit(1)
    text = text[:idx] + Q5K_Q8K_FUNCS + text[idx:]
    changes += 1
    print("Inserted dequant_q5_k_block() and dequant_q8_k_block()")

# ── 2. Replace Q5_K stub in tb_gguf_dequant_row() ────────────────────────────
# Match both CRLF and LF variants
Q5K_STUB_PAT = re.compile(
    r'case 13: /\* Q5_K - 176 bytes/superblock, 256 weights \(stub\) \*/\r?\n'
    r'        while \(n_out < n_weights\) \{\r?\n'
    r'            memset\(out \+ n_out, 0, 256 \* sizeof\(float\)\); /\* TODO full Q5_K \*/\r?\n'
    r'            p \+= 176;\r?\n'
    r'            n_out \+= 256;\r?\n'
    r'        \}\r?\n'
    r'        break;'
)
Q5K_REAL = (
    'case 13: /* Q5_K - 176 bytes/superblock, 256 weights */\n'
    '        while (n_out < n_weights) {\n'
    '            dequant_q5_k_block(p, out + n_out);\n'
    '            p += 176;\n'
    '            n_out += 256;\n'
    '        }\n'
    '        break;'
)

m = Q5K_STUB_PAT.search(text)
if m:
    text = text[:m.start()] + Q5K_REAL + text[m.end():]
    changes += 1
    print("Replaced Q5_K stub with dequant_q5_k_block() call")
elif 'dequant_q5_k_block(p, out' in text:
    print("Q5_K dequant_row already patched — skipping")
else:
    print("WARNING: Q5_K stub not found by regex — check tb_gguf.c manually")

# ── 3. Replace Q8_K stub in tb_gguf_dequant_row() ────────────────────────────
Q8K_STUB_PAT = re.compile(
    r'case 15: /\* Q8_K - 292 bytes/superblock, 256 weights \(stub\) \*/\r?\n'
    r'        while \(n_out < n_weights\) \{\r?\n'
    r'            memset\(out \+ n_out, 0, 256 \* sizeof\(float\)\); /\* TODO full Q8_K \*/\r?\n'
    r'            p \+= 292;\r?\n'
    r'            n_out \+= 256;\r?\n'
    r'        \}\r?\n'
    r'        break;'
)
Q8K_REAL = (
    'case 15: /* Q8_K - 292 bytes/superblock, 256 weights */\n'
    '        while (n_out < n_weights) {\n'
    '            dequant_q8_k_block(p, out + n_out);\n'
    '            p += 292;\n'
    '            n_out += 256;\n'
    '        }\n'
    '        break;'
)

m = Q8K_STUB_PAT.search(text)
if m:
    text = text[:m.start()] + Q8K_REAL + text[m.end():]
    changes += 1
    print("Replaced Q8_K stub with dequant_q8_k_block() call")
elif 'dequant_q8_k_block(p, out' in text:
    print("Q8_K dequant_row already patched — skipping")
else:
    print("WARNING: Q8_K stub not found by regex — check tb_gguf.c manually")

if changes > 0:
    with open(path, 'wb') as f:
        f.write(text.encode('utf-8'))
    print(f"Wrote {path}  ({changes} change(s))")
else:
    print("No changes written.")
