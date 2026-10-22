"""Reference outputs (MLX, float32) for the native runtime's real-model test.

    python scripts/make_cpp_reference.py --target Qwen/Qwen3-0.6B --drafter runs/smoke_adapt/final --out runs/cpp_ref
    WOODCOCK_REF=runs/cpp_ref ./build/woodcock_tests
"""

import argparse
import json
from pathlib import Path

import mlx.core as mx
from huggingface_hub import snapshot_download

from woodcock.data import chat_ids
from woodcock.spec import generate_autoregressive
from woodcock.target import Target

ap = argparse.ArgumentParser()
ap.add_argument("--target", default="Qwen/Qwen3-0.6B")
ap.add_argument("--drafter", default="runs/smoke_adapt/final")
ap.add_argument("--out", default="runs/cpp_ref")
ap.add_argument("--n", type=int, default=48)
a = ap.parse_args()

t = Target.load(a.target)
t.model.set_dtype(mx.float32)
prompt = chat_ids(t.tokenizer, "Write a Python function that checks whether a number is prime.")
o = t(mx.array([prompt]), t.make_cache(), capture=False)
last = o.logits[0, -1].astype(mx.float32).tolist()
ar, _ = generate_autoregressive(t, prompt, a.n)
tdir = Path(a.target) if Path(a.target).exists() else Path(snapshot_download(a.target))
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
(out / "reference.json").write_text(json.dumps({
    "target_dir": str(tdir),
    "drafter_dir": str(Path(a.drafter).resolve()),
    "prompt": prompt,
    "last_logits": last,
    "ar_greedy": ar,
    "n_new": a.n,
}))
print(f"wrote {out / 'reference.json'}: prompt {len(prompt)} tokens, {len(ar)} AR tokens")
