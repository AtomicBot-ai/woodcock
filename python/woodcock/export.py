"""Woodcock drafter export format v1 (read by the native runtime, src/core/drafter.cpp).

    <dir>/
      woodcock.json       manifest: format, version, mode, target taps, tensor table, sha256
      config.json         DrafterConfig (python/woodcock/drafter.py)
      model.safetensors   bf16 weights, names as in python/woodcock/drafter.py
      NOTICE.md           provenance of the weights (copied target embedding / head)

The target model is not part of the export; the manifest names it so a runtime
can check it loads the matching target (hidden size, vocabulary, tap layers).
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten

from .drafter import Drafter

FORMAT = "woodcock-drafter"
VERSION = 1


def export_drafter(src: str, out: str, target_name: str | None = None) -> Path:
    src_p, out_p = Path(src), Path(out)
    out_p.mkdir(parents=True, exist_ok=True)
    d = Drafter.load(src_p, dtype=mx.bfloat16)
    d.save(out_p)
    meta = json.loads((src_p / "meta.json").read_text()) if (src_p / "meta.json").exists() else {}
    if target_name is None:
        target_name = meta.get("train", {}).get("target")
    wfile = out_p / "model.safetensors"
    sha = hashlib.sha256(wfile.read_bytes()).hexdigest()
    tensors = {k: {"shape": list(v.shape), "dtype": str(v.dtype).split(".")[-1]} for k, v in tree_flatten(d.parameters())}
    c = d.cfg
    manifest = {
        "format": FORMAT,
        "version": VERSION,
        "mode": "woodcock-v1",
        "target": {"name": target_name, "hidden_size": c.target_hidden_size, "vocab_size": c.vocab_size,
                   "feature_layers": c.feature_layers},
        "drafter": {"hidden_size": c.hidden_size, "num_layers": c.num_layers, "post_norm": c.post_norm,
                    "window": c.window},
        "weights": {"file": "model.safetensors", "sha256": sha, "num_params": d.num_params(), "tensors": tensors},
        "training": meta.get("train"),
    }
    (out_p / "woodcock.json").write_text(json.dumps(manifest, indent=2))
    notice = Path(__file__).resolve().parents[2] / "NOTICE.md"
    if notice.exists():
        shutil.copy(notice, out_p / "NOTICE.md")
    # self-check: the exported file reloads and computes the same thing
    e = Drafter.load(out_p)
    tok = mx.array([[1, 2, 3]])
    g = mx.zeros((1, 3, c.hidden_size), mx.bfloat16)
    diff = mx.abs(d(tok, g)[1].astype(mx.float32) - e(tok, g)[1].astype(mx.float32)).max().item()
    if diff != 0.0:
        raise RuntimeError(f"export round-trip mismatch {diff}")
    print(f"[export] {src} -> {out} ({manifest['mode']}, {d.num_params() / 1e6:.1f}M params, sha256 {sha[:12]})")
    return out_p
