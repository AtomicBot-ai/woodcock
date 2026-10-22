import json

from woodcock.data import SFTStream


def test_eos_only_for_naturally_ended_responses(tmp_path):
    p = tmp_path / "r.jsonl"
    recs = [
        {"id": "a", "domain": "chat", "prompt_ids": [1, 2], "response_ids": list(range(10, 22)), "max_new_tokens": 64},
        {"id": "b", "domain": "chat", "prompt_ids": [1, 2], "response_ids": list(range(10, 74)), "max_new_tokens": 64},
        {"id": "c", "domain": "chat", "prompt_ids": [1, 2], "response_ids": list(range(10, 522))},  # legacy, 512 cap
    ]
    p.write_text("".join(json.dumps(r) + "\n" for r in recs))
    out = {len(ids): (ids, mask) for ids, mask in SFTStream(p, max_len=2048, eos_id=999).examples()}
    ended = out[2 + 12 + 1][0]
    assert ended[-1] == 999  # stopped before the cap: EOS appended
    assert 2 + 64 in out and 999 not in out[2 + 64][0]  # hit the cap: cut mid-text, no EOS
    assert 2 + 512 in out and 999 not in out[2 + 512][0]  # legacy record without the field
    ids, mask = out[2 + 64]
    assert mask[:2].sum() == 0 and mask[2:].all()  # loss only on response tokens
