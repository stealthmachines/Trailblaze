# Trailblaze v0.8b Release Summary

Date: 2026-05-29
Branch: v0.8b

## Commit stack

1. `c8c253c` - v0.8b rapid prototype artifact diffs
2. `8c31771` - minimal source patch: telemetry-coupled runtime controller + strand endpoints
3. `3528d82` - extended source patch: docs + Coqui telemetry bridge utility
4. `ee65505` - integration smoke checks for controller policy behavior

## Source changes (vs v0.7)

- `layer4/tb_infer.h`
  - Added controller state fields to `TB_InferCtx`
  - Added external telemetry control API declarations
- `layer4/tb_infer.c`
  - Added `tb_ctrl_apply_external_metrics`
  - Added `tb_ctrl_export_state`
  - Applied controller scaling in HDGL expert routing path
  - Added state/control HTTP endpoints
- `README.md`
  - Updated v0.8b headline/context
  - Documented `/api/strand/control` and `/api/strand/state`
- `scripts/bridge_coqui_metrics.py`
  - Added trainer telemetry bridge utility for Coqui-style `avg_*` deltas
- `docs/v08b_rapid_notes.md`
  - Added rapid rollout and control-policy notes
- `tb_integration_test.c`
  - Added v0.8b controller policy smoke assertions

## Runtime API additions

- `POST /api/strand/control`
  - Input: metric deltas (`avg_loss_mel`, `avg_loss_1`, `avg_loss_gen`, `avg_loss_disc`, optional `disc_real_vol`)
  - Action: updates runtime controller scales and quality pressure
- `GET /api/strand/state`
  - Output: current controller values and telemetry snapshot

## Validation performed

- VS Code diagnostics:
  - `layer4/tb_infer.c`: no errors
  - `layer4/tb_infer.h`: no errors
  - `README.md`: no errors
  - `tb_integration_test.c`: no errors
- Python syntax validation:
  - `py -3.10 -m py_compile scripts/bridge_coqui_metrics.py`

## Notes

- The two patch artifact files are intentionally retained:
  - `trailblaze_v08b_minimal.diff`
  - `trailblaze_v08b_extended.diff`
- Integration smoke checks currently validate controller-policy invariants in-process and do not require starting the HTTP server.
