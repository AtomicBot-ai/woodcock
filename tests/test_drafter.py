import mlx.core as mx
import numpy as np
import pytest

from woodcock.drafter import Drafter, DrafterConfig


def tiny_cfg(**kw):
    base = dict(
        hidden_size=32,
        num_layers=1,
        num_heads=4,
        num_kv_heads=2,
        head_dim=8,
        intermediate_size=48,
        vocab_size=50,
        target_hidden_size=24,
        rope_theta=10000.0,
    )
    base.update(kw)
    return DrafterConfig(**base)


def _close(a, b, tol=2e-4):
    np.testing.assert_allclose(np.array(a), np.array(b), atol=tol, rtol=tol)


@pytest.mark.parametrize(
    "post_norm,window,layers",
    [(True, None, 1), (False, None, 1), (True, None, 2), (True, 5, 1)],
)
def test_ttt_matches_inference_chain(post_norm, window, layers):
    """Training-time-test step j at entry i == real chain drafting from context i."""
    mx.random.seed(0)
    cfg = tiny_cfg(post_norm=post_norm, window=window, num_layers=layers)
    m = Drafter(cfg)
    L, steps = 12, 3
    tokens = mx.random.randint(0, cfg.vocab_size, (1, L))
    g = mx.random.normal((1, L, cfg.hidden_size))
    outs = m.ttt_forward(tokens, g, steps)
    for i in (0, 4, 8):
        cache = m.make_cache()
        hid, out = m(tokens[:, : i + 1], g[:, : i + 1], cache)
        _close(out[:, -1], outs[0][:, i])
        fb = m.feedback(hid[:, -1:], out[:, -1:])
        for j in range(1, steps):
            hid, out = m(tokens[:, i + j : i + j + 1], fb, cache)
            _close(out[:, -1], outs[j][:, i])
            fb = m.feedback(hid, out)


def test_feature_conditioning_matters_in_feature_mode():
    mx.random.seed(2)
    d = Drafter(tiny_cfg())
    tokens = mx.random.randint(0, 50, (1, 6))
    _, o1 = d(tokens, mx.random.normal((1, 6, 32)))
    _, o2 = d(tokens, mx.random.normal((1, 6, 32)))
    assert float(mx.abs(o1 - o2).max()) > 1e-3


def test_save_load_roundtrip(tmp_path):
    mx.random.seed(4)
    d = Drafter(tiny_cfg(window=6))
    d.save(tmp_path / "dr")
    e = Drafter.load(tmp_path / "dr")
    tokens = mx.random.randint(0, 50, (1, 5))
    g = mx.random.normal((1, 5, 32))
    _close(d(tokens, g)[1], e(tokens, g)[1], tol=0)


def test_loads_configs_written_by_the_earlier_code_base(tmp_path):
    """Earlier configs carry residual / lowrank_rank / lowrank_k; plain drafters load,
    anything this code base cannot represent is rejected."""
    import json

    d = Drafter(tiny_cfg())
    d.save(tmp_path / "dr")
    cfg = json.loads((tmp_path / "dr/config.json").read_text())
    cfg.update(residual="feature", lowrank_rank=0, lowrank_k=128)
    (tmp_path / "dr/config.json").write_text(json.dumps(cfg))
    Drafter.load(tmp_path / "dr")
    for bad in ({"residual": "token"}, {"lowrank_rank": 128}):
        (tmp_path / "dr/config.json").write_text(json.dumps({**cfg, **bad}))
        with pytest.raises(ValueError):
            Drafter.load(tmp_path / "dr")
