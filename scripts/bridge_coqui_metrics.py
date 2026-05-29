#!/usr/bin/env python3
"""Bridge Coqui-style metric lines into Trailblaze /api/strand/control."""

from __future__ import annotations

import argparse
import json
import re
import urllib.request

PATTERN = re.compile(
    r"avg_(?P<name>[a-zA-Z0-9_]+):\s*(?P<value>-?\d+(?:\.\d+)?)\s*\((?P<delta>[+-]?\d+(?:\.\d+)?)\)"
)


def parse_metrics(path: str) -> dict[str, float]:
    text = open(path, "r", encoding="utf-8", errors="ignore").read()
    deltas: dict[str, float] = {}
    for m in PATTERN.finditer(text):
        key = m.group("name")
        deltas[f"avg_{key}"] = float(m.group("delta"))

    # Optional convenience field from discriminator component spread.
    disc_real = [v for k, v in deltas.items() if k.startswith("avg_loss_disc_real_")]
    if len(disc_real) >= 2:
        mean = sum(disc_real) / len(disc_real)
        var = sum((x - mean) ** 2 for x in disc_real) / len(disc_real)
        deltas["disc_real_vol"] = var ** 0.5

    return deltas


def post(url: str, payload: dict) -> str:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8", errors="ignore")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="Path to Coqui metrics text/log")
    ap.add_argument("--url", default="http://127.0.0.1:11434/api/strand/control")
    args = ap.parse_args()

    payload = parse_metrics(args.log)
    if not payload:
        raise SystemExit("No avg_* metric deltas found in log")

    print(post(args.url, payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
