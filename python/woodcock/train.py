"""Training loops (MLX, single Apple-Silicon device).

* adapt      — train a drafter against a frozen target with training-time test
               (EAGLE-3 §3.2) and a step-discounted soft-label CE (weights lambda^j),
               computed on assistant tokens only (EAGLE-3.1 §5.1). Target forward
               runs online: storing 3 x d_target hidden states per token would cost
               ~12 KB/token for Qwen3-1.7B, more than recomputing them.
All loops checkpoint (weights + optimizer + step + data position) and resume.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten, tree_map, tree_map_with_path, tree_unflatten

from .data import SFTStream, batches
from .drafter import Drafter
from .memory import MemoryReport
from .target import Target


# ---------------------------------------------------------------- shared

def lr_schedule(peak, warmup, total, final_frac=0.1):
    warm = optim.linear_schedule(peak * 1e-3, peak, max(1, warmup))
    cos = optim.cosine_decay(peak, max(1, total - warmup), end=peak * final_frac)
    return optim.join_schedules([warm, cos], [max(1, warmup)])


def clip_grads(grads, max_norm):
    if not max_norm:
        return grads, None
    return optim.clip_grad_norm(grads, max_norm)


def _save_opt(opt, path):
    state = dict(tree_flatten(opt.state))
    mx.save_safetensors(str(path), {k: v for k, v in state.items() if isinstance(v, mx.array)})


def _load_opt(opt, path):
    if Path(path).exists():
        opt.state = tree_unflatten(list(mx.load(str(path)).items()))


def soft_ce_step(out, W, t_logits, mask):
    """Soft-label CE between target and drafter distributions, summed over masked
    positions. W is passed explicitly so mx.checkpoint differentiates it."""
    z = (out @ W.T).astype(mx.float32)
    logq = z - mx.logsumexp(z, axis=-1, keepdims=True)
    tl = t_logits.astype(mx.float32)
    p = mx.softmax(tl, axis=-1)
    ce = -(p * logq).sum(-1)
    hit = (mx.argmax(z, -1) == mx.argmax(tl, -1)).astype(mx.float32)
    return (ce * mask).sum(), (hit * mask).sum()


_soft_ce_ckpt = mx.checkpoint(soft_ce_step)


class MasterWeights:
    """fp32 master copy of the trainable parameters while the model computes in bf16.

    Updating bf16 weights directly would lose Adam steps: bf16 resolves ~0.4% of a
    weight's magnitude, while a step at lr 1e-4 is ~1e-4 of it and rounds away.
    """

    def __init__(self, model: nn.Module, opt: optim.Optimizer):
        self.model, self.opt = model, opt
        self.master = tree_map(lambda p: p.astype(mx.float32), model.trainable_parameters())

    def step(self, grads):
        self.master = self.opt.apply_gradients(grads, self.master)
        self.model.update(tree_map(lambda p: p.astype(mx.bfloat16), self.master))

    def save(self, path):
        mx.save_safetensors(str(path), dict(tree_flatten(self.master)))

    def load(self, path):
        """Restore saved values into the CURRENT trainable-parameter tree. Rebuilding the
        tree from the file (tree_unflatten) would drop the empty subtrees of frozen modules
        ("embed_tokens": {}) that the gradients still contain, and the first optimizer step
        after a resume would fail with KeyError."""
        if Path(path).exists():
            saved = mx.load(str(path))
            missing = [k for k, _ in tree_flatten(self.master) if k not in saved]
            if missing:
                raise ValueError(f"{path}: missing master weights {missing[:5]}")
            self.master = tree_map_with_path(lambda k, p: saved[k], self.master)
            self.model.update(tree_map(lambda p: p.astype(mx.bfloat16), self.master))


# ---------------------------------------------------------------- adaptation

@dataclass
class AdaptConfig:
    target: str
    data: str
    out: str
    init: str = "scratch"  # scratch | layerinit | drafter:<drafter dir>
    ttt_steps: int = 4  # EAGLE-3.1: post-norm allows TTT 4 instead of 8
    step_discount: float = 0.8  # lambda: TTT step j is weighted lambda**j (docs/provenance.md)
    lr: float = 1e-4
    warmup: int = 100
    total_steps: int = 3000
    grad_accum: int = 4
    token_budget: int = 2048
    max_len: int = 1024
    max_batch: int = 8
    clip: float = 0.5
    weight_decay: float = 0.0
    train_head: bool = True
    train_embeddings: bool = False
    window: int | None = None
    holdout: int = 200
    eval_every: int = 250
    save_every: int = 250
    seed: int = 0
    log: str = ""

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=2))


def build_drafter(cfg: AdaptConfig, target: Target) -> Drafter:
    from .drafter import config_for_target, init_from_target

    if cfg.init in ("scratch", "layerinit"):
        dc = config_for_target(target, window=cfg.window)
        layer = target.num_layers - 1 if cfg.init == "layerinit" else None
        return init_from_target(dc, target, layer_from_target=layer)
    if cfg.init.startswith("drafter:"):  # continue training from a saved drafter (new optimizer, new schedule)
        return Drafter.load(cfg.init.split(":", 1)[1])
    raise ValueError(cfg.init)


def adapt(cfg: AdaptConfig):
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg.save(out / "train_config.json")
    mx.random.seed(cfg.seed)
    target = Target.load(cfg.target)
    state_path = out / "state.json"
    if state_path.exists():
        drafter = Drafter.load(out / "last")
        state = json.loads(state_path.read_text())
    else:
        drafter = build_drafter(cfg, target)
        state = {"step": 0, "epoch": 0, "seen": 0, "tokens": 0}
    # compute in bf16 (1.7x faster TTT step, scripts/profile_train_step.py); fp32 master weights
    drafter.set_dtype(mx.bfloat16)
    drafter.unfreeze()
    if not cfg.train_embeddings:
        drafter.embed_tokens.freeze()
    if not cfg.train_head:
        drafter.lm_head.freeze()
    n_train = sum(v.size for _, v in tree_flatten(drafter.trainable_parameters()))
    print(f"[adapt] drafter trainable params {n_train/1e6:.1f}M, total {drafter.num_params()/1e6:.1f}M", flush=True)

    opt = optim.AdamW(lr_schedule(cfg.lr, cfg.warmup, cfg.total_steps), betas=[0.9, 0.95], weight_decay=cfg.weight_decay)
    _load_opt(opt, out / "last" / "optimizer.safetensors")
    master = MasterWeights(drafter, opt)
    master.load(out / "last" / "master.safetensors")
    eos = target.tokenizer.eos_token_id
    stream = SFTStream(Path(cfg.data), cfg.max_len, eos, seed=cfg.seed, holdout=cfg.holdout)
    weights = [cfg.step_discount**j for j in range(cfg.ttt_steps)]

    def loss_fn(model, tokens_in, feats, t_logits, sel):
        return ttt_loss(model, tokens_in, feats, t_logits, sel, cfg.ttt_steps, weights)

    vg = nn.value_and_grad(drafter, lambda m, *a: loss_fn(m, *a)[0])

    def target_pass(ids, mask):
        return target_pass_for(target, ids, mask, cfg.ttt_steps)

    def run_eval(n_batches=30):
        tot = np.zeros((cfg.ttt_steps, 3))
        for k, (ids, mask) in enumerate(batches(stream.examples(train=False), cfg.token_budget, cfg.max_batch)):
            if k >= n_batches:
                break
            tl, feats, sel = target_pass(ids, mask)
            _, met = loss_fn(drafter, mx.array(ids[:, 1:]), feats, tl, sel)
            for j, (ce, hit, cnt) in enumerate(met):
                tot[j] += [ce.item(), hit.item(), cnt.item()]
        return {f"step{j}": {"ce": tot[j, 0] / max(tot[j, 2], 1), "top1": tot[j, 1] / max(tot[j, 2], 1)}
                for j in range(cfg.ttt_steps)}

    mem = MemoryReport()
    log = open(out / "log.jsonl", "a")
    t_start = time.perf_counter()
    acc_grads, acc_n, acc_loss = None, 0, 0.0
    tok_since, t_since = 0, time.perf_counter()
    while state["step"] < cfg.total_steps:
        it = batches(stream.examples(train=True, epoch=state["epoch"]), cfg.token_budget, cfg.max_batch)
        for bi, (ids, mask) in enumerate(it):
            if bi < state["seen"]:
                continue  # resume: skip batches already consumed this epoch
            state["seen"] = bi + 1
            tl, feats, sel = target_pass(ids, mask)
            loss, grads = vg(drafter, mx.array(ids[:, 1:]), feats, tl, sel)
            grads = tree_map(lambda g: g.astype(mx.float32), grads)
            acc_grads = grads if acc_grads is None else tree_map(lambda a, b: a + b, acc_grads, grads)
            acc_loss = acc_loss + loss
            acc_n += 1
            tok_since += int(mask.sum())
            state["tokens"] += int(ids.size)
            if acc_n < cfg.grad_accum:
                mx.eval(acc_grads, acc_loss)
                continue
            acc_grads = tree_map(lambda a: a / acc_n, acc_grads)
            acc_grads, gnorm = clip_grads(acc_grads, cfg.clip)
            master.step(acc_grads)
            mean_loss = acc_loss / acc_n
            mx.eval(master.master, drafter.parameters(), opt.state, mean_loss)
            acc_grads, acc_n, acc_loss = None, 0, 0.0
            state["step"] += 1
            mem.sample()
            if state["step"] % 10 == 0:
                dt = time.perf_counter() - t_since
                rec = {"step": state["step"], "loss": mean_loss.item(), "lr": float(opt.learning_rate),
                       "gnorm": float(gnorm) if gnorm is not None else None,
                       "sup_tok_per_s": tok_since / dt, "tokens": state["tokens"],
                       "peak_gb": mem.peak_gb, "elapsed_h": (time.perf_counter() - t_start) / 3600}
                print("[adapt]", json.dumps(rec), flush=True)
                log.write(json.dumps(rec) + "\n"); log.flush()
                tok_since, t_since = 0, time.perf_counter()
            if state["step"] % cfg.eval_every == 0 or state["step"] == cfg.total_steps:
                ev = run_eval()
                print("[adapt][eval]", json.dumps({"step": state["step"], **ev}), flush=True)
                log.write(json.dumps({"step": state["step"], "eval": ev}) + "\n"); log.flush()
            if state["step"] % cfg.save_every == 0 or state["step"] == cfg.total_steps:
                _checkpoint(drafter, opt, state, out, master)
            if state["step"] >= cfg.total_steps:
                break
        else:
            state["epoch"] += 1
            state["seen"] = 0
    _checkpoint(drafter, opt, state, out, master)
    drafter.save(out / "final", extra={"train": asdict(cfg), "state": state})
    print(f"[adapt] done: {state}, peak {mem.peak_gb:.1f} GB", flush=True)


def ttt_loss(model, tokens_in, feats, t_logits, sel, ttt_steps, weights):
    """Step-discounted soft-label CE of the TTT unroll.

    feats: raw target taps at positions 0..L-2 (fusion is trained, so it runs here).
    t_logits: target logits only at the rows the loss needs; sel[j] = (drafter rows,
    target rows) of the supervised entries of TTT step j. Heads and soft-CE are
    evaluated only on supervised positions; the value equals the masked full loss
    (tests/test_train.py::test_selected_loss_equals_masked_full_loss).
    """
    g = model.fuse(feats)
    outs = model.ttt_forward(tokens_in, g, ttt_steps)
    W = model.lm_head.weight
    d = outs[0].shape[-1]
    total, metrics = 0.0, []
    for j, (rows_d, rows_t) in enumerate(sel):
        if rows_d.size == 0:
            continue
        o = outs[j].reshape(-1, d)[rows_d]
        ce, hit = _soft_ce_ckpt(o, W, t_logits[rows_t], mx.ones((rows_d.size,)))
        total = total + weights[j] * ce / rows_d.size
        metrics.append((ce, hit, mx.array(float(rows_d.size))))
    return total, metrics


def target_pass_for(target, ids: np.ndarray, mask: np.ndarray, ttt_steps: int):
    """Target taps for every position; target logits only where a loss needs them
    (position p supervises the token at p+1, so rows with mask[p+1] = 1)."""
    B, L = ids.shape
    o = target(mx.array(ids), capture=True, logits=False, hidden=True)
    need = np.zeros((B, L), bool)
    need[:, :-1] = mask[:, 1:] > 0
    flat = np.flatnonzero(need.reshape(-1))
    row_of = np.full(B * L, -1, np.int64)
    row_of[flat] = np.arange(len(flat))
    h = o.hidden.reshape(B * L, -1)[mx.array(flat.astype(np.int32))]
    t_logits = target.lm_head(target.model.model.norm(h))
    sel = []
    L1 = L - 1  # drafter entries
    for j in range(ttt_steps):
        n = L1 - 1 - j  # entry i of step j predicts token i+2+j; target row (b, i+1+j)
        if n <= 0:
            break
        b, i = np.nonzero(mask[:, 2 + j : 2 + j + n] > 0)
        rows_t = row_of[b * L + i + 1 + j]
        assert (rows_t >= 0).all()
        sel.append((mx.array((b * L1 + i).astype(np.int32)), mx.array(rows_t.astype(np.int32))))
    feats = [mx.stop_gradient(f[:, :-1]) for f in o.features]
    return mx.stop_gradient(t_logits), feats, sel


def _checkpoint(drafter, opt, state, out, master=None):
    d = out / "last"
    drafter.save(d)
    _save_opt(opt, d / "optimizer.safetensors")
    if master is not None:
        master.save(d / "master.safetensors")
    (out / "state.json").write_text(json.dumps(state))
