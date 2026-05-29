# Trailblaze v0.8b Rapid Notes

## Minimal layer
- Runtime controller state fields in `TB_InferCtx`
- Telemetry-coupled alpha/semantic scaling in `tb_route_experts`
- HTTP endpoints:
  - `POST /api/strand/control`
  - `GET /api/strand/state`

## Extended layer
- `scripts/bridge_coqui_metrics.py` to ingest Coqui-style `avg_*` deltas
- README documentation for strand control API

## Initial control policy
- If `avg_loss_mel` rises strongly: reduce effective HDGL/noisy influence
- If `avg_loss_1` rises: tighten quality pressure
- If discriminator component volatility rises: damp noisy exposure
- If generator/discriminator and mel all improve: cautiously relax pressure

## Rollout
1. Start with minimal layer only.
2. Observe `/api/strand/state` while running representative workloads.
3. Add bridge script in training loops.
4. Tune thresholds only after collecting several rounds of telemetry.
