f = open('layer4/tb_infer.c', encoding='utf-8', errors='replace').read()
checks = [
    ('QK-norm',            'attn_q_norm.weight'),
    ('GGUF dequant for QKV', 'Route through tb_gguf_dequant_matvec'),
    ('output proj new',    'tb_gguf_dequant_matvec(tb_gguf_tensor_data(m, wo_t)'),
    ('packed MoE new',     'moe_intermediate_size > 0'),
    ('compat shim removed','4bit_compat(wq, sq, bq'),
]
for name, pat in checks:
    found = pat in f
    status = 'PRESENT' if found else 'MISSING'
    print(name + ': ' + status)
