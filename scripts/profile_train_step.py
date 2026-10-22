"""Where does an adaptation step spend its time? Measures each component on one
real batch (B x L tokens) with the real target, so training-throughput fixes are
chosen from measurements rather than guesses."""

import argparse
import time

import mlx.core as mx
import mlx.nn as nn

from woodcock.drafter import config_for_target, init_from_target
from woodcock.target import Target
from woodcock.train import _soft_ce_ckpt, soft_ce_step


def timeit(fn, reps=3):
    fn()
    mx.synchronize()
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        mx.synchronize()
        ts.append(time.perf_counter() - t)
    return 1e3 * sorted(ts)[len(ts) // 2]


ap = argparse.ArgumentParser()
ap.add_argument("--target", default="Qwen/Qwen3-1.7B")
ap.add_argument("--B", type=int, default=2)
ap.add_argument("--L", type=int, default=1024)
a = ap.parse_args()

t = Target.load(a.target)
d = init_from_target(config_for_target(t), t)
ids = mx.random.randint(0, 50000, (a.B, a.L))
mask = mx.ones((a.B, a.L))
o = t(ids)
tl = o.logits
feats = [f[:, :-1] for f in o.features]
mx.eval(tl, feats)
print(f"tokens/batch {a.B * a.L}")
print(f"target fwd (logits+taps)           {timeit(lambda: mx.eval(t(ids).logits)):8.1f} ms")


def make_loss(K, ckpt=True):
    ce_fn = _soft_ce_ckpt if ckpt else soft_ce_step

    def loss(m, toks, feats, tl, mask):
        outs = m.ttt_forward(toks, m.fuse(feats), K)
        tot = 0.0
        for j, x in enumerate(outs):
            n = toks.shape[1] - 1 - j
            ce, _ = ce_fn(x[:, :n], m.lm_head.weight, tl[:, 1 + j : 1 + j + n], mask[:, 2 + j : 2 + j + n])
            tot = tot + ce
        return tot
    return loss


for dtype in (mx.float32, mx.bfloat16):
    d.set_dtype(dtype)
    d.embed_tokens.weight = d.embed_tokens.weight.astype(mx.bfloat16)
    for train_head in (True, False):
        d.unfreeze()
        d.embed_tokens.freeze()
        if not train_head:
            d.lm_head.freeze()
        for K in (1, 2, 4):
            vg = nn.value_and_grad(d, make_loss(K))
            ms = timeit(lambda: mx.eval(vg(d, ids[:, 1:], feats, tl, mask)))
            print(f"drafter TTT K={K} fwd+bwd dtype={str(dtype).split('.')[-1]:8s} head={'train' if train_head else 'frozen'}  {ms:8.1f} ms")

d.set_dtype(mx.bfloat16)
d.unfreeze(); d.embed_tokens.freeze()
x = mx.random.normal((a.B, a.L - 1, d.cfg.hidden_size)).astype(mx.bfloat16)
W = d.lm_head.weight
print(f"head+softCE fwd only (one step)     {timeit(lambda: mx.eval(soft_ce_step(x, W, tl[:, 1:], mask[:, 1:]))):8.1f} ms")
g = mx.grad(lambda W: soft_ce_step(x, W, tl[:, 1:], mask[:, 1:])[0])
print(f"head+softCE fwd+bwd (one step)      {timeit(lambda: mx.eval(g(W))):8.1f} ms")
print(f"peak {mx.get_peak_memory() / 1e9:.1f} GB")
