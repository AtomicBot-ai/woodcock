import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import pytest

from woodcock.train import MasterWeights


def test_master_weights_keep_updates_bf16_would_drop(tmp_path):
    m = nn.Linear(4, 4, bias=False)
    m.weight = mx.ones((4, 4))
    m.set_dtype(mx.bfloat16)
    opt = optim.SGD(learning_rate=1e-4)
    mw = MasterWeights(m, opt)
    g = {"weight": mx.ones((4, 4), mx.float32)}
    # direct bf16 update: 1 - 1e-4 rounds back to 1.0
    assert (mx.ones((1,), mx.bfloat16) - mx.array(1e-4, mx.bfloat16)).item() == 1.0
    for _ in range(50):
        mw.step(g)
    mx.eval(mw.master)
    np.testing.assert_allclose(np.array(mw.master["weight"]), 1 - 50e-4, rtol=1e-6)
    # the bf16 model follows the master (to bf16 resolution) instead of being stuck at 1.0
    assert float(m.weight[0, 0].astype(mx.float32)) < 1.0
    mw.save(tmp_path / "m.safetensors")
    m2 = nn.Linear(4, 4, bias=False)
    m2.set_dtype(mx.bfloat16)
    mw2 = MasterWeights(m2, optim.SGD(learning_rate=1e-4))
    mw2.load(tmp_path / "m.safetensors")
    np.testing.assert_array_equal(np.array(mw2.master["weight"]), np.array(mw.master["weight"]))


def test_resume_continues_exactly_with_frozen_embeddings(tmp_path):
    """Two optimizer steps in one run == one step, checkpoint, reload into fresh objects,
    one more step. The drafter's embeddings are frozen, as in `adapt`."""
    from mlx.utils import tree_flatten

    from test_drafter import tiny_cfg
    from woodcock.drafter import Drafter
    from woodcock.train import _load_opt, _save_opt

    def fresh():
        mx.random.seed(0)
        d = Drafter(tiny_cfg())
        d.set_dtype(mx.bfloat16)
        d.unfreeze()
        d.embed_tokens.freeze()
        opt = optim.AdamW(learning_rate=1e-3, betas=[0.9, 0.95])
        return d, opt

    def grads(d, k):
        mx.random.seed(10 + k)
        return tree_map(lambda p: mx.random.normal(p.shape), d.trainable_parameters())

    from mlx.utils import tree_map

    d, opt = fresh()
    mw = MasterWeights(d, opt)
    mw.step(grads(d, 0))
    mw.step(grads(d, 1))
    mx.eval(mw.master)
    ref = dict(tree_flatten(mw.master))

    d, opt = fresh()
    mw = MasterWeights(d, opt)
    mw.step(grads(d, 0))
    mx.eval(mw.master, opt.state)
    mw.save(tmp_path / "master.safetensors")
    _save_opt(opt, tmp_path / "opt.safetensors")
    d, opt = fresh()
    _load_opt(opt, tmp_path / "opt.safetensors")
    mw = MasterWeights(d, opt)
    mw.load(tmp_path / "master.safetensors")
    mw.step(grads(d, 1))  # failed with KeyError 'embed_tokens' before the fix
    mx.eval(mw.master)
    out = dict(tree_flatten(mw.master))
    assert out.keys() == ref.keys()
    for k in ref:
        np.testing.assert_array_equal(np.array(out[k]), np.array(ref[k]), err_msg=k)


@pytest.mark.model
def test_selected_loss_equals_masked_full_loss():
    """The supervised-positions-only loss equals the straightforward masked loss
    over all positions (old formulation), on a real target."""
    from woodcock.drafter import config_for_target, init_from_target
    from woodcock.target import Target
    from woodcock.train import soft_ce_step, target_pass_for, ttt_loss

    t = Target.load("Qwen/Qwen3-0.6B")
    t.model.set_dtype(mx.float32)
    mx.random.seed(0)
    d = init_from_target(config_for_target(t), t)
    d.set_dtype(mx.float32)
    rng = np.random.default_rng(0)
    B, L, K = 2, 48, 3
    ids = rng.integers(0, 5000, (B, L)).astype(np.int32)
    mask = np.zeros((B, L), np.float32)
    mask[0, 20:] = 1
    mask[1, 31:44] = 1
    w = [0.8**j for j in range(K)]
    tl, feats, sel = target_pass_for(t, ids, mask, K)
    new, _ = ttt_loss(d, mx.array(ids[:, 1:]), feats, tl, sel, K, w)
    # reference: full logits everywhere, mask applied afterwards
    full = t(mx.array(ids)).logits
    outs = d.ttt_forward(mx.array(ids[:, 1:]), d.fuse(feats), K)
    ref = 0.0
    for j, o in enumerate(outs):
        n = L - 2 - j
        m = mx.array(mask[:, 2 + j : 2 + j + n])
        ce, _ = soft_ce_step(o[:, :n], d.lm_head.weight, full[:, 1 + j : 1 + j + n], m)
        ref = ref + w[j] * ce / m.sum()
    np.testing.assert_allclose(new.item(), ref.item(), rtol=1e-4)
