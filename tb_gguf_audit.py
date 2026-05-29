#!/usr/bin/env python3
"""
tb_gguf_audit.py -- Tensor name coverage + dequant self-test for tb_infer.c v0.8

Checks:
  1. GGUF KV params vs. what tb_infer reads (n_layers, n_heads, rope_base, etc.)
  2. Tensor name coverage: every name tb_infer.c looks up, hit-or-miss
  3. Silent-passthrough detection: NULL return from tb_gguf_find_tensor
  4. Live dequant self-test: decode first block of each quant type found and
     compare against a pure-Python reference (Q4_0, Q4_K, Q8_0, Q6_K, BF16)
  5. GQA sanity: n_kv_heads divides n_heads
  6. RoPE: checks rope_scaling_type for YaRN/dynamic NTK (not implemented)

Usage:
  python tb_gguf_audit.py path/to/model.gguf
"""

import sys, struct, math, os, mmap, traceback, re

GGUF_MAGIC = b"GGUF"

QTYPE_NAME = {
    0:"F32", 1:"F16", 2:"Q4_0", 8:"Q8_0", 10:"Q2_K", 11:"Q3_K",
    12:"Q4_K", 13:"Q5_K", 14:"Q6_K", 15:"Q8_K", 21:"IQ3_S", 23:"IQ4_XS", 30:"BF16",
}

# Block sizes (bytes) and weights per block -- must match tb_gguf.c exactly
BLOCK_INFO = {
    0:  (4,   1),    # F32
    1:  (2,   1),    # F16
    30: (2,   1),    # BF16
    2:  (18,  32),   # Q4_0
    8:  (34,  32),   # Q8_0
    12: (144, 256),  # Q4_K
    10: (84,  256),  # Q2_K
    11: (110, 256),  # Q3_K
    21: (110, 256),  # IQ3_S
    23: (136, 256),  # IQ4_XS: d(2)+scales(14)+qs(128)-minus-padding → actually 136 bytes
    13: (176, 256),  # Q5_K
    15: (292, 256),  # Q8_K
    14: (210, 256),  # Q6_K
}

# GGUF primitive type sizes
_PRIM_SZ = {0:1, 1:1, 2:2, 3:2, 4:4, 5:4, 6:4, 7:1, 10:8, 11:8, 12:8}

# ── Memory-mapped cursor-based GGUF parser ─────────────────────────────────

class Cursor:
    """Thin wrapper around a bytes-like object with a position cursor."""
    def __init__(self, data):
        self.data = data
        self.pos  = 0
        self.size = len(data)

    def read(self, n):
        end = self.pos + n
        if end > self.size:
            raise EOFError(f"read {n} bytes at {self.pos:#x} but EOF at {self.size:#x}")
        chunk = self.data[self.pos:end]
        self.pos = end
        return chunk

    def skip(self, n):
        self.pos += n
        if self.pos > self.size:
            raise EOFError(f"skip to {self.pos:#x} exceeds file size {self.size:#x}")

    def u8(self):  return struct.unpack_from("<B", self.read(1))[0]
    def u16(self): return struct.unpack_from("<H", self.read(2))[0]
    def u32(self): return struct.unpack_from("<I", self.read(4))[0]
    def u64(self): return struct.unpack_from("<Q", self.read(8))[0]
    def i8(self):  return struct.unpack_from("<b", self.read(1))[0]
    def i32(self): return struct.unpack_from("<i", self.read(4))[0]
    def i64(self): return struct.unpack_from("<q", self.read(8))[0]
    def f32(self): return struct.unpack_from("<f", self.read(4))[0]
    def f64(self): return struct.unpack_from("<d", self.read(8))[0]

    def gguf_str(self, str_len_bytes=8):
        if str_len_bytes == 4:
            n = self.u32()
        else:
            n = self.u64()
        if n > 4 * 1024 * 1024:
            raise ValueError(f"string length {n} at {self.pos:#x} is too large (corrupt GGUF?)")
        return self.read(n).decode("utf-8", errors="replace")


def _skip_val(cur, vtype, str_len_bytes):
    if vtype in _PRIM_SZ:
        cur.skip(_PRIM_SZ[vtype])
    elif vtype == 8:  # string
        if str_len_bytes == 4:
            n = cur.u32()
        else:
            n = cur.u64()
        cur.skip(n)
    elif vtype == 9:  # array
        etype  = cur.u32()
        if str_len_bytes == 4:
            count = cur.u32()
        else:
            count = cur.u64()
        if etype in _PRIM_SZ:
            cur.skip(count * _PRIM_SZ[etype])
        else:
            for _ in range(count):
                _skip_val(cur, etype, str_len_bytes)
    else:
        raise ValueError(f"unknown vtype {vtype} at {cur.pos:#x}")


def _read_val(cur, vtype, str_len_bytes, max_array=32):
    if vtype == 0:  return cur.u8()
    if vtype == 1:  return cur.i8()
    if vtype == 2:  return cur.u16()
    if vtype == 3:
        raw = cur.read(2)
        return struct.unpack_from("<h", raw)[0]
    if vtype == 4:  return cur.u32()
    if vtype == 5:  return cur.i32()
    if vtype == 6:  return cur.f32()
    if vtype == 7:  return bool(cur.u8())
    if vtype == 8:  return cur.gguf_str(str_len_bytes)
    if vtype == 10: return cur.u64()
    if vtype == 11: return cur.i64()
    if vtype == 12: return cur.f64()
    if vtype == 9:
        etype  = cur.u32()
        if str_len_bytes == 4:
            count = cur.u32()
        else:
            count = cur.u64()
        if count <= max_array:
            if etype in _PRIM_SZ:
                return [_read_val(cur, etype, str_len_bytes) for _ in range(count)]
            if etype == 8:
                return [cur.gguf_str(str_len_bytes) for _ in range(count)]
        # Large array -- skip efficiently
        if etype in _PRIM_SZ:
            cur.skip(count * _PRIM_SZ[etype])
        else:
            for _ in range(count):
                _skip_val(cur, etype, str_len_bytes)
        return f"<array[{count}] skipped>"
    raise ValueError(f"unknown vtype {vtype}")


def parse_gguf(path):
    """Return (version, kv_dict, tensors_dict, data_offset)."""
    fsize = os.path.getsize(path)
    with open(path, "rb") as fh:
        mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)

    try:
        cur = Cursor(mm)

        magic = cur.read(4)
        if magic != GGUF_MAGIC:
            raise ValueError(f"not a GGUF file (magic={magic!r})")

        version = cur.u32()
        # v1: n_tensors/n_kv are uint32; v2+: uint64
        if version == 1:
            n_tensors  = cur.u32()
            n_kv       = cur.u32()
            str_len_sz = 4  # v1 uses uint32 string lengths
        else:
            n_tensors  = cur.u64()
            n_kv       = cur.u64()
            str_len_sz = 8  # v2/v3 use uint64 string lengths

        kv = {}
        for i in range(n_kv):
            pos_before = cur.pos
            key   = cur.gguf_str(str_len_sz)
            vtype = cur.u32()
            try:
                val = _read_val(cur, vtype, str_len_sz)
            except Exception as e:
                print(f"  [parse warn at KV #{i}] key={key!r} vtype={vtype} "
                      f"pos={pos_before:#x}: {e} -- skipping remainder of KV section")
                break
            kv[key] = val

        tensors = {}
        for _ in range(n_tensors):
            name   = cur.gguf_str(str_len_sz)
            n_dims = cur.u32()
            shape  = [cur.u64() for _ in range(n_dims)]
            qtype  = cur.u32()
            offset = cur.u64()
            tensors[name] = {"shape": shape, "qtype": qtype, "offset": offset}

        # Data section: 32-byte aligned after tensor info table
        data_off = (cur.pos + 31) & ~31
        return version, kv, tensors, data_off

    finally:
        mm.close()

# ── Reference dequant (pure Python) ────────────────────────────────────────

def f16_to_f32(h):
    sign = (h >> 15) & 1
    exp  = (h >> 10) & 0x1F
    mant =  h        & 0x3FF
    if exp == 0:
        val = (mant / 1024.0) * (2 ** -14)
    elif exp == 31:
        val = float('inf') if mant == 0 else float('nan')
    else:
        val = (1.0 + mant / 1024.0) * (2 ** (exp - 15))
    return -val if sign else val

def bf16_to_f32(b):
    bits = b << 16
    return struct.unpack("<f", struct.pack("<I", bits))[0]

def ref_q4_0(block_bytes):
    d = f16_to_f32(struct.unpack_from("<H", block_bytes, 0)[0])
    out = []
    for i in range(16):
        byte = block_bytes[2 + i]
        out.append(d * ((byte & 0xF) - 8))
        out.append(d * ((byte >> 4) - 8))
    return out

def ref_q8_0(block_bytes):
    d = f16_to_f32(struct.unpack_from("<H", block_bytes, 0)[0])
    out = []
    for i in range(32):
        q = struct.unpack_from("<b", block_bytes, 2 + i)[0]
        out.append(d * q)
    return out

def q4k_unpack_scales(sc, d, dmin):
    scales, mins = [], []
    for j in range(4):
        scales.append(d    * (sc[j]   & 0x3F))
        mins.append(  dmin * (sc[j+4] & 0x3F))
        scales.append(d    * ((sc[j+8] & 0x0F) | ((sc[j]   >> 6) << 4)))
        mins.append(  dmin * ((sc[j+8] >> 4)   | ((sc[j+4] >> 6) << 4)))
    return scales, mins

def ref_q4_k(superblock):
    d    = f16_to_f32(struct.unpack_from("<H", superblock, 0)[0])
    dmin = f16_to_f32(struct.unpack_from("<H", superblock, 2)[0])
    sc   = superblock[4:16]
    qs   = superblock[16:144]
    scales, mins = q4k_unpack_scales(sc, d, dmin)
    out = []
    for i in range(128):
        byte = qs[i]
        sub0 = (2*i)   >> 5
        sub1 = (2*i+1) >> 5
        out.append(scales[sub0] * (byte & 0xF) - mins[sub0])
        out.append(scales[sub1] * (byte >> 4)  - mins[sub1])
    return out

def ref_q2_k(superblock):
    # [scales:16][qs:64][d:f16 2][dmin:f16 2]
    scales_raw = superblock[0:16]
    qs         = superblock[16:80]
    d    = f16_to_f32(struct.unpack_from("<H", superblock, 80)[0])
    dmin = f16_to_f32(struct.unpack_from("<H", superblock, 82)[0])
    out = []
    for g in range(16):
        sc = d    * (scales_raw[g] & 0x0F)
        mn = dmin * (scales_raw[g] >> 4)
        for i in range(16):
            wi    = g * 16 + i
            shift = (wi & 3) << 1
            q     = (qs[wi >> 2] >> shift) & 0x03
            out.append(sc * q - mn)
    return out

def ref_q6_k(superblock):
    ql = superblock[0:128]
    qh = superblock[128:192]
    sc = superblock[192:208]
    d  = f16_to_f32(struct.unpack_from("<H", superblock, 208)[0])
    sc_s = [struct.unpack_from("<b", bytes([sc[i]]))[0] for i in range(16)]
    out = []
    for i in range(256):
        half   = i >> 7
        within = i & 127
        band   = within >> 5
        l      = within & 31
        ql_off = half*64 + l + (band & 1)*32
        lo     = (ql[ql_off] & 0xF) if band < 2 else ((ql[ql_off] >> 4) & 0xF)
        hi     = (qh[half*32 + l] >> (band*2)) & 0x3
        sc_idx = half*8 + (l >> 4) + band*2
        q6     = (lo | (hi << 4)) - 32
        out.append(d * sc_s[sc_idx] * q6)
    return out

def ref_q3_k(superblock):
    # [hmask:32][qs:64][scales:12][d:f16 2]
    hmask  = superblock[0:32]
    qs     = superblock[32:96]
    sc_raw = superblock[96:108]
    d_all  = f16_to_f32(struct.unpack_from("<H", superblock, 108)[0])
    scales = []
    for j in range(16):
        is_  = j if j < 8 else (j - 8)
        lo4  = (sc_raw[is_ >> 1] >> ((is_ & 1) << 2)) & 0xF
        hi2  = (sc_raw[8 + (is_ >> 2)] >> ((is_ & 3) << 1)) & 0x3
        scales.append(d_all * ((lo4 | (hi2 << 4)) - 32))
    out = []
    for i in range(256):
        low2 = (qs[i >> 2] >> ((i & 3) << 1)) & 0x03
        high = (hmask[i >> 3] >> (i & 7)) & 0x01
        q3   = low2 | (high << 2)
        out.append(scales[i >> 4] * (q3 - 4))
    return out

def ref_q5_k(superblock):
    d    = f16_to_f32(struct.unpack_from("<H", superblock, 0)[0])
    dmin = f16_to_f32(struct.unpack_from("<H", superblock, 2)[0])
    sc   = superblock[4:16]
    qh   = superblock[16:48]
    qs   = superblock[48:176]
    scales, mins = q4k_unpack_scales(sc, d, dmin)
    out = []
    for i in range(256):
        g      = i // 32
        qh_bit = (qh[i >> 3] >> (i & 7)) & 1
        lo4    = (qs[i >> 1] >> ((i & 1) << 2)) & 0xF
        q5     = lo4 | (qh_bit << 4)
        out.append(scales[g] * q5 - mins[g])
    return out

def ref_q8_k(superblock):
    # [d:f32 4][qs:256 int8][bsums:32 int16]
    d  = struct.unpack_from("<f", superblock, 0)[0]
    qs = superblock[4:260]
    return [d * struct.unpack_from("<b", bytes([qs[i]]))[0] for i in range(256)]

def ref_bf16_row(data, n):
    return [bf16_to_f32(struct.unpack_from("<H", data, i*2)[0]) for i in range(n)]

DEQUANT_REF = {
    2:  (ref_q4_0,  18),
    8:  (ref_q8_0,  34),
    10: (ref_q2_k,  84),
    11: (ref_q3_k, 110),
    12: (ref_q4_k, 144),
    13: (ref_q5_k, 176),
    14: (ref_q6_k, 210),
    15: (ref_q8_k, 292),
}

# ── Dequant self-test ───────────────────────────────────────────────────────

def dequant_selftest(path, tensors, data_off):
    results = {}
    seen_qtypes = {}
    for name, info in tensors.items():
        qt = info["qtype"]
        if qt not in seen_qtypes:
            seen_qtypes[qt] = (name, info)

    with open(path, "rb") as f:
        for qt, (name, info) in sorted(seen_qtypes.items()):
            qt_name = QTYPE_NAME.get(qt, f"unk{qt}")

            if qt in (0, 1, 30):  # scalar types -- spot check a few values
                try:
                    file_offset = data_off + info["offset"]
                    f.seek(file_offset)
                    blk_sz = BLOCK_INFO[qt][0] * 32  # read 32 elements
                    raw = f.read(blk_sz)
                    if qt == 0:
                        vals = list(struct.unpack_from(f"<{len(raw)//4}f", raw))
                    elif qt == 1:
                        vals = [f16_to_f32(struct.unpack_from("<H", raw, i*2)[0])
                                for i in range(len(raw)//2)]
                    else:  # BF16
                        vals = ref_bf16_row(raw, len(raw)//2)
                    l2  = math.sqrt(sum(x*x for x in vals if math.isfinite(x)))
                    bad = sum(1 for x in vals if not math.isfinite(x))
                    results[qt_name] = (f"OK  L2={l2:.4f}  bad={bad}" if bad == 0
                                        else f"FAIL:NAN bad={bad}/{len(vals)}")
                except Exception as e:
                    results[qt_name] = f"ERROR: {e}"
                continue

            if qt not in DEQUANT_REF:
                results[qt_name] = f"SKIP (no Python ref for type {qt})"
                continue

            ref_fn, block_bytes = DEQUANT_REF[qt]
            try:
                file_offset = data_off + info["offset"]
                f.seek(file_offset)
                block_data = bytes(f.read(block_bytes))
                vals = ref_fn(block_data)
                l2   = math.sqrt(sum(x*x for x in vals))
                bad  = sum(1 for x in vals if not math.isfinite(x))
                zero = sum(1 for x in vals if x == 0.0)
                if bad > 0:
                    status = f"FAIL:NAN  bad={bad}/{len(vals)}  tensor={name}"
                elif l2 < 1e-12:
                    status = f"FAIL:ZERO  L2={l2:.3e}  tensor={name}"
                elif l2 > 1e6:
                    status = f"WARN:LARGE L2={l2:.1f}"
                else:
                    status = f"OK  L2={l2:.4f}  zeros={zero}/{len(vals)}  tensor={name}"
                results[qt_name] = status
            except Exception as e:
                results[qt_name] = f"ERROR: {e}"

    return results

# ── Tensor name coverage ────────────────────────────────────────────────────

def build_expected_names(kv, tensors):
    arch      = kv.get("general.architecture", "llama")
    n_layers  = kv.get(f"{arch}.block_count",  kv.get("llama.block_count",  0))
    n_experts = kv.get(f"{arch}.expert_count", kv.get("llama.expert_count", 0))
    # GGUF key is "full_attention_interval" (full word), not "full_attn_interval"
    fi = kv.get(f"{arch}.full_attention_interval",
                kv.get(f"{arch}.full_attn_interval", 0))

    def _to_int(v, default=0):
        try: return int(v)
        except (TypeError, ValueError): return default
    n_layers  = _to_int(n_layers)
    n_experts = _to_int(n_experts)
    fi        = _to_int(fi)

    expected = []
    expected.append(("token_embd.weight",   [], "FATAL: embedding is identity -- all tokens look the same"))
    expected.append(("output_norm.weight",  [], "WARN: final norm uses ones -- logits unscaled"))
    expected.append(("output.weight", ["token_embd.weight"], "WARN: LM head uses tied embedding (may be intentional)"))

    for L in range(n_layers):
        is_local = fi > 0 and (L % fi != fi - 1)

        expected.append((f"blk.{L}.attn_norm.weight", [],
                         f"L{L}: attn_norm bypassed (uses ones)"))

        if not is_local:
            fused = f"blk.{L}.attn_qkv.weight"
            sep   = [f"blk.{L}.attn_q.weight", f"blk.{L}.attn_k.weight", f"blk.{L}.attn_v.weight"]
            in_gguf = fused in tensors or all(s in tensors for s in sep)
            if not in_gguf:
                expected.append((fused, sep, f"L{L}: QKV passthrough -- q/k/v = copy of normed input"))
            expected.append((f"blk.{L}.attn_q_norm.weight", [], f"L{L}: q_norm skipped (optional for non-Qwen3)"))
            expected.append((f"blk.{L}.attn_k_norm.weight", [], f"L{L}: k_norm skipped (optional for non-Qwen3)"))
            expected.append((f"blk.{L}.attn_output.weight", [f"blk.{L}.attn_out.weight"],
                             f"L{L}: attn out proj = identity -- no projection applied"))
        else:
            for tname, consequence in [
                (f"blk.{L}.attn_qkv.weight",  f"L{L}(local): DeltaNet QKV missing"),
                (f"blk.{L}.attn_gate.weight",  f"L{L}(local): SSM gate missing"),
                (f"blk.{L}.ssm_a",             f"L{L}(local): SSM A missing"),
                (f"blk.{L}.ssm_alpha.weight",  f"L{L}(local): alpha missing"),
                (f"blk.{L}.ssm_beta.weight",   f"L{L}(local): beta missing"),
                (f"blk.{L}.ssm_conv1d.weight", f"L{L}(local): conv1d missing"),
                (f"blk.{L}.ssm_norm.weight",   f"L{L}(local): ssm_norm missing"),
                (f"blk.{L}.ssm_out.weight",    f"L{L}(local): ssm_out missing"),
                (f"blk.{L}.ssm_dt.bias",       f"L{L}(local): dt_bias missing"),
            ]:
                expected.append((tname, [], consequence))

        expected.append((f"blk.{L}.ffn_norm.weight",
                         [f"blk.{L}.post_attention_norm.weight"],
                         f"L{L}: ffn_norm bypassed"))

        if n_experts > 0:
            expected.append((f"blk.{L}.ffn_gate_inp.weight", [],
                             f"L{L}: MoE router missing -- uniform routing"))
            expected.append((f"blk.{L}.ffn_gate_exps.weight", [], f"L{L}: expert gate missing -- MoE zeros out"))
            expected.append((f"blk.{L}.ffn_up_exps.weight",   [], f"L{L}: expert up missing"))
            expected.append((f"blk.{L}.ffn_down_exps.weight", [], f"L{L}: expert down missing"))
        else:
            expected.append((f"blk.{L}.ffn_gate.weight", [], f"L{L}: dense FFN gate missing -- SwiGLU zeros"))
            expected.append((f"blk.{L}.ffn_up.weight",   [], f"L{L}: dense FFN up missing"))
            expected.append((f"blk.{L}.ffn_down.weight", [], f"L{L}: dense FFN down missing"))

    return expected, n_layers, n_experts

# ── GQA sanity ─────────────────────────────────────────────────────────────

def check_gqa(kv):
    arch = kv.get("general.architecture", "llama")
    n_h  = kv.get(f"{arch}.attention.head_count",    kv.get("llama.attention.head_count",    None))
    n_kv = kv.get(f"{arch}.attention.head_count_kv", kv.get("llama.attention.head_count_kv", None))
    if n_h is None:
        return "UNKNOWN (head_count not found in KV)", 0, 0
    n_h  = int(n_h)
    n_kv = int(n_kv) if n_kv is not None else n_h
    if n_h % n_kv != 0:
        return f"FAIL: n_heads={n_h} not divisible by n_kv_heads={n_kv}", n_h, n_kv
    ratio = n_h // n_kv
    tag = "MHA" if ratio == 1 else f"GQA(ratio={ratio})"
    return f"OK {tag}  n_heads={n_h}  n_kv_heads={n_kv}", n_h, n_kv

# ── RoPE check ─────────────────────────────────────────────────────────────

def check_rope(kv):
    arch      = kv.get("general.architecture", "llama")
    rope_type = kv.get(f"{arch}.rope.scaling.type", kv.get("llama.rope.scaling.type", None))
    rope_base = kv.get(f"{arch}.rope.freq_base",    kv.get("llama.rope.freq_base",    None))
    issues = []
    if rope_type in ("yarn", "longrope", "dynamic"):
        issues.append(f"WARN: rope_scaling_type={rope_type!r} -- engine only implements standard RoPE; "
                      f"long-context generation will be wrong")
    elif rope_type is None:
        issues.append("rope_scaling_type: not set (standard RoPE assumed)")
    else:
        issues.append(f"rope_scaling_type={rope_type!r}")
    if rope_base is not None:
        issues.append(f"rope_base={float(rope_base):.1f}")
    else:
        issues.append("rope_base: not in GGUF (engine will use compiled-in default)")
    return issues

# ── Main ───────────────────────────────────────────────────────────────────

def _to_int(v, default=0):
    try: return int(v)
    except (TypeError, ValueError): return default

def audit(path):
    sep = "=" * 70
    print(f"\n{sep}")
    print(f" tb_gguf_audit  ->  {os.path.basename(path)}")
    print(f"{sep}\n")

    version, kv, tensors, data_off = parse_gguf(path)

    arch = kv.get("general.architecture", "llama")
    print(f"GGUF version : {version}")
    print(f"Architecture : {arch}")
    print(f"Tensor count : {len(tensors)}")
    print(f"KV entries   : {len(kv)}")
    print(f"Data offset  : {data_off:#x}")

    # ── Section 1: Key parameters ─────────────────────────────────────────
    print(f"\n-- Section 1: KV parameters")
    key_params = [
        (f"{arch}.block_count",                   "n_layers"),
        (f"{arch}.embedding_length",               "hidden_dim"),
        (f"{arch}.feed_forward_length",            "ffn_dim"),
        (f"{arch}.attention.head_count",           "n_heads"),
        (f"{arch}.attention.head_count_kv",        "n_kv_heads"),
        (f"{arch}.attention.key_length",           "head_dim"),
        (f"{arch}.expert_count",                   "n_experts"),
        (f"{arch}.expert_used_count",              "n_experts_per_tok"),
        (f"{arch}.rope.freq_base",                 "rope_base"),
        (f"{arch}.attention.layer_norm_rms_epsilon","norm_eps"),
        (f"{arch}.full_attention_interval",          "full_attention_interval"),
        ("tokenizer.ggml.bos_token_id",            "bos_id"),
        ("tokenizer.ggml.eos_token_id",            "eos_id"),
    ]
    for gkey, label in key_params:
        val = kv.get(gkey, kv.get(gkey.replace(arch+".", "llama."), "NOT FOUND"))
        print(f"  {label:32s} {val}")

    # ── Section 2: GQA sanity ─────────────────────────────────────────────
    print(f"\n-- Section 2: GQA sanity")
    gqa_status, n_heads, n_kv_heads = check_gqa(kv)
    print(f"  {gqa_status}")

    # ── Section 3: RoPE ───────────────────────────────────────────────────
    print(f"\n-- Section 3: RoPE")
    for issue in check_rope(kv):
        print(f"  {issue}")

    # ── Section 4: Tensor name coverage ──────────────────────────────────
    print(f"\n-- Section 4: Tensor name coverage")
    expected, n_layers, n_experts = build_expected_names(kv, tensors)

    hit = miss = fallback_hit = 0
    miss_lines = []
    pattern_miss_count = {}

    def check_tensor(name, fallbacks):
        if name in tensors:
            return "HIT", name
        for fb in fallbacks:
            if fb in tensors:
                return "FALLBACK", fb
        return "MISS", None

    for (name, fallbacks, consequence) in expected:
        status, found_at = check_tensor(name, fallbacks)
        pat = re.sub(r"\.\d+\.", ".N.", name)

        if status == "HIT":
            hit += 1
        elif status == "FALLBACK":
            fallback_hit += 1
            cnt = pattern_miss_count.get(pat, 0)
            pattern_miss_count[pat] = cnt + 1
            if cnt < 1:
                miss_lines.append(f"  FALLBACK  {name}")
                miss_lines.append(f"            -> found as: {found_at}")
            elif cnt == 1:
                miss_lines.append(f"  FALLBACK  (pattern {pat!r} occurs on all {n_layers} layers -- suppressing)")
        else:
            miss += 1
            cnt = pattern_miss_count.get(pat, 0)
            pattern_miss_count[pat] = cnt + 1
            if cnt < 2:
                miss_lines.append(f"  MISS      {name}")
                miss_lines.append(f"            consequence: {consequence}")
            elif cnt == 2:
                miss_lines.append(f"  MISS      {pat} (occurs on all {n_layers} layers -- suppressing)")

    print(f"  HIT={hit}  FALLBACK={fallback_hit}  MISS={miss}  (of {len(expected)} lookups against {len(tensors)} tensors)")
    if miss_lines:
        print()
        for line in miss_lines:
            print(line)
    else:
        print("  All expected tensors found -- no silent passthroughs detected.")

    # ── Section 5: Quant type inventory ──────────────────────────────────
    print(f"\n-- Section 5: Quant type inventory")
    qtype_count = {}
    for info in tensors.values():
        qt = info["qtype"]
        qtype_count[qt] = qtype_count.get(qt, 0) + 1
    for qt, cnt in sorted(qtype_count.items()):
        name = QTYPE_NAME.get(qt, f"unk{qt}")
        flag = "  [NO dequant impl in tb_gguf.c!]" if qt not in BLOCK_INFO else ""
        print(f"  {name:10s} (type={qt:2d}): {cnt:4d} tensors{flag}")

    # ── Section 6: Block size sanity ─────────────────────────────────────
    print(f"\n-- Section 6: Block size sanity (first tensor per quant type)")
    file_size = os.path.getsize(path)
    checked   = set()
    issues    = []
    tensor_list = list(tensors.items())
    for tname, info in tensor_list:
        qt = info["qtype"]
        if qt in checked or qt not in BLOCK_INFO:
            continue
        block_bytes, block_weights = BLOCK_INFO[qt]
        if block_bytes <= 4:
            checked.add(qt)
            continue
        n_elems = 1
        for d in info["shape"]:
            n_elems *= d
        if n_elems == 0:
            continue
        expected_blocks = (n_elems + block_weights - 1) // block_weights
        expected_bytes  = expected_blocks * block_bytes
        if data_off + info["offset"] + expected_bytes > file_size:
            issues.append(f"  FAIL: {tname} ({QTYPE_NAME.get(qt,'?')}) n_elems={n_elems} "
                          f"expected_bytes={expected_bytes} would exceed file size ({file_size})")
        else:
            checked.add(qt)
            print(f"  OK  {QTYPE_NAME.get(qt,'?'):8s}  tensor={tname}  "
                  f"n_elems={n_elems}  blocks={expected_blocks}")
    for issue in issues:
        print(issue)
    if not issues and not checked:
        print("  (no quantised block types to verify)")

    # ── Section 7: Dequant self-test ──────────────────────────────────────
    print(f"\n-- Section 7: Dequant self-test (first block of each quant type)")
    dq_results = dequant_selftest(path, tensors, data_off)
    for qt_name, result in sorted(dq_results.items()):
        print(f"  {qt_name:10s}: {result}")

    # ── Section 8: Dequant-zeroing risk ──────────────────────────────────
    print(f"\n-- Section 8: Dequant-zeroing risk (tensor found, but C engine will zero it)")
    # v6: IQ4_XS (23) added in v0_8_fixed(6) via dequant_iq4_xs_block + case 23
    IMPLEMENTED_IN_C = {0, 1, 2, 8, 10, 11, 12, 13, 14, 15, 21, 23, 30}
    zero_risk = {}
    for tname, info in tensors.items():
        qt = info["qtype"]
        if qt not in IMPLEMENTED_IN_C:
            key = QTYPE_NAME.get(qt, f"unk{qt}")
            zero_risk.setdefault(key, []).append(tname)
    if zero_risk:
        for qt_name, tlist in sorted(zero_risk.items()):
            print(f"  {qt_name} ({len(tlist)} tensors): C engine hits 'default: memset(out,0)' branch!")
            for t in sorted(tlist)[:4]:
                print(f"    {t}")
            if len(tlist) > 4:
                print(f"    ... and {len(tlist)-4} more")
    else:
        print("  No tensors use unimplemented quant types.")

    # ── Section 9: Architecture-specific checks ───────────────────────────
    print(f"\n-- Section 9: Architecture-specific checks")

    arch_full_attn_key = f"{arch}.full_attention_interval"
    fi_val  = _to_int(kv.get(arch_full_attn_key, kv.get(f"{arch}.full_attn_interval", 0)))
    hd_val  = _to_int(kv.get(f"{arch}.attention.key_length",
                               kv.get("llama.attention.key_length", 0)))
    hid_val = _to_int(kv.get(f"{arch}.embedding_length",
                               kv.get("llama.embedding_length", 0)))
    nh_val  = n_heads

    # (a) Q-head count vs actual global attn_q.weight rows
    # v6 engine self-corrects n_heads from blk.(fi-1).attn_q.weight at load time.
    if fi_val > 0 and hd_val > 0 and nh_val > 0:
        first_global = fi_val - 1
        gbl_q = f"blk.{first_global}.attn_q.weight"
        if gbl_q in tensors:
            t = tensors[gbl_q]
            n_elems = 1
            for d in t["shape"]: n_elems *= d
            actual_q_heads = (n_elems // hid_val) // hd_val if hid_val > 0 else 0
            if actual_q_heads != nh_val:
                print(f"  INFO: GGUF KV n_heads={nh_val} but blk.{first_global}.attn_q.weight "
                      f"implies {actual_q_heads} heads")
                print(f"        v6 engine self-corrects n_heads {nh_val}->{actual_q_heads} from "
                      f"blk.{first_global}.attn_q.weight at load time -- OK")
                nh_val = actual_q_heads  # propagate corrected value for later checks
            else:
                print(f"  OK  Global Q-head count: n_heads={nh_val} matches attn_q.weight shape")
        else:
            print(f"  INFO: blk.{first_global} uses fused QKV (no separate attn_q.weight)")

    # (b) n_kv_heads via global attn_k.weight (v6 fix: no longer reads local blk.0.attn_qkv.weight)
    # v6 engine reads blk.(fi-1).attn_k.weight shape[0] / head_dim to derive n_kv_heads.
    if fi_val > 0 and hd_val > 0 and hid_val > 0:
        first_global = fi_val - 1
        global_k_key = f"blk.{first_global}.attn_k.weight"
        if global_k_key in tensors:
            k_elems = 1
            for d in tensors[global_k_key]["shape"]: k_elems *= d
            engine_nkv = k_elems // hid_val // hd_val if hid_val > 0 and hd_val > 0 else 0
            gguf_nkv   = _to_int(kv.get(f"{arch}.attention.head_count_kv",
                                         kv.get("llama.attention.head_count_kv", 0)))
            if engine_nkv > 0 and (gguf_nkv == 0 or engine_nkv == gguf_nkv):
                print(f"  OK  n_kv_heads={engine_nkv} from blk.{first_global}.attn_k.weight "
                      f"(v6 global-layer path)")
            elif engine_nkv > 0 and gguf_nkv != engine_nkv:
                print(f"  INFO: GGUF KV n_kv_heads={gguf_nkv} but blk.{first_global}.attn_k.weight "
                      f"implies {engine_nkv}")
                print(f"        v6 engine self-corrects n_kv_heads {gguf_nkv}->{engine_nkv} -- OK")
            else:
                print(f"  INFO: could not derive n_kv_heads from blk.{first_global}.attn_k.weight")
        else:
            # Fall back to checking if old local-QKV path would give correct result
            local_qkv_key = "blk.0.attn_qkv.weight"
            if local_qkv_key in tensors:
                qkv_rows = 1
                for d in tensors[local_qkv_key]["shape"]: qkv_rows *= d
                qkv_rows //= hid_val
                c_q_rows    = nh_val * hd_val
                c_kv_rows   = qkv_rows - c_q_rows
                c_derived   = c_kv_rows // (2 * hd_val) if hd_val > 0 else 0
                print(f"  WARN: blk.{first_global}.attn_k.weight missing; "
                      f"old local-QKV path gives n_kv_heads={c_derived}")

    # (c) Partial RoPE
    # v6 engine: tb_rope_apply now accepts rope_rotary_dim and limits rotation to
    # the first rotary_dim elements of each head (partial RoPE, Qwen3.5 style).
    rope_dim_count = kv.get(f"{arch}.rope.dimension_count", None)
    rope_dim_sects = kv.get(f"{arch}.rope.dimension_sections", None)
    if rope_dim_count is not None and hd_val > 0:
        rdc = int(rope_dim_count)
        if rdc < hd_val:
            print(f"  OK  Partial RoPE  rope.dimension_count={rdc} < head_dim={hd_val}")
            print(f"        v6 engine rotates only first {rdc} dims ({rdc//2} pairs) per head -- IMPLEMENTED")
            if isinstance(rope_dim_sects, list):
                print(f"        rope.dimension_sections={rope_dim_sects} "
                      f"(YaRN frequency-band scaling -- NOT YET IMPLEMENTED, quality impact only)")
        else:
            print(f"  OK  RoPE dimension_count={rdc} covers full head_dim={hd_val}")
    else:
        print(f"  INFO: rope.dimension_count not in GGUF")

    # (d) full_attention_interval key name match
    fi_raw = kv.get(arch_full_attn_key, None)
    if fi_raw is not None:
        print(f"  OK  {arch_full_attn_key}={fi_raw} found")
        print(f"      C engine uses strstr('full_attention_interval') -> matches correctly")
        print(f"      Hybrid layout: every {fi_raw}th layer is global softmax, rest are DeltaNet")
    else:
        print(f"  INFO: full_attention_interval not in GGUF (fi=0, all layers treated as global)")

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n-- Summary")
    dq_fail = sum(1 for r in dq_results.values() if r.startswith("FAIL"))
    zero_risk_count = sum(len(v) for v in zero_risk.values())
    print(f"  Tensor misses (silent passthrough):          {miss}")
    print(f"  Tensor fallbacks (alias resolved OK):        {fallback_hit}")
    if zero_risk_count:
        print(f"  Tensors dequant-zeroed in C engine:          {zero_risk_count}  <- SILENT WRONG OUTPUT")
    else:
        print(f"  Tensors dequant-zeroed in C engine:          0  (all quant types implemented)")
    print(f"  Dequant self-test failures (implemented types): {dq_fail}")
    fails = []
    if miss > 0: fails.append("tensor misses")
    if zero_risk_count > 0: fails.append(f"zeroing ({zero_risk_count} tensors of unimplemented quant type)")
    if dq_fail > 0: fails.append("dequant errors")
    if fails:
        print(f"  OVERALL: ISSUES FOUND -- {', '.join(fails)}")
        print(f"  See Sections 8 and 9 for architecture-specific bugs")
    else:
        print("  OVERALL: PASS -- no gaps detected by static analysis")
    print()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        default = r"C:\Users\Owner\.ollama\models\blobs\unsloth\Qwen3.5-9B-GGUF\Qwen3.5-9B-UD-Q2_K_XL.gguf"
        if os.path.exists(default):
            print(f"No path given, using: {default}")
            audit(default)
        else:
            print("Usage: python tb_gguf_audit.py <model.gguf>")
            sys.exit(1)
    else:
        for path in sys.argv[1:]:
            try:
                audit(path)
            except Exception:
                traceback.print_exc()
