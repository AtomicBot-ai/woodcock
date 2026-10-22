import json

import mlx.core as mx
import numpy as np
import pytest

from test_drafter import tiny_cfg
from woodcock.data import SFTStream
from woodcock.drafter import Drafter
from woodcock.vocab import load_ids, response_counts, top_ids


def test_top_ids_frequency_order_ties_and_forced():
    counts = np.array([5, 0, 9, 5, 1, 7, 0, 0])
    assert top_ids(counts, 3).tolist() == [0, 2, 5]  # 9, 7, then the tie 5/5 -> smaller id
    assert top_ids(counts, 3, {6}).tolist() == [2, 5, 6]  # forced id replaces the least frequent
    assert top_ids(counts, 2, {2}).tolist() == [2, 5]  # forced id already selected: no duplicate
    with pytest.raises(ValueError):
        top_ids(counts, 0)
    with pytest.raises(ValueError):
        top_ids(counts, 1, {1, 6})


def test_counts_use_the_training_split_of_sftstream(tmp_path):
    p = tmp_path / "r.jsonl"
    recs = [{"id": str(i), "domain": "chat", "prompt_ids": [1], "response_ids": [10 + i] * 9} for i in range(20)]
    p.write_text("".join(json.dumps(r) + "\n" for r in recs))
    counts, hold = response_counts(p, 64, seed=3, holdout=5)
    train_ids = {r["id"] for r in SFTStream(p, 64, 0, seed=3, holdout=5).records(train=True)}
    assert {r["id"] for r in hold} == {r["id"] for r in recs} - train_ids
    assert counts.sum() == 9 * 15 and all(counts[10 + int(i)] == 9 for i in train_ids)
    np.savez(tmp_path / "c.npz", counts=counts, eos=np.array([63]))
    ids = load_ids(f"{tmp_path / 'c.npz'}:4")
    assert 63 in ids.tolist() and len(ids) == 4


def test_restricted_head_equals_gathered_full_logits(tmp_path):
    mx.random.seed(0)
    d = Drafter(tiny_cfg())
    tokens = mx.random.randint(0, 50, (1, 4))
    _, out = d(tokens, mx.random.normal((1, 4, 32)))
    full = d.logits(out)
    ids = [3, 7, 8, 21, 49]
    d.restrict_vocab(ids)
    np.testing.assert_array_equal(np.array(d.logits(out)), np.array(full[..., mx.array(ids)]))
    rows = mx.argmax(d.logits(out), axis=-1)
    assert d.token_ids(rows).tolist() == [[ids[r] for r in rows[0].tolist()]]
    with pytest.raises(ValueError):
        d.restrict_vocab(ids)
    with pytest.raises(ValueError):
        d.save(tmp_path / "dr")
