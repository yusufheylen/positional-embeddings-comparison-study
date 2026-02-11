from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from src.models.embeddings import rope, yarn


def test_rotate_half_swaps_and_negates_halves():
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    out = rope.rotate_half(x)
    assert torch.equal(out, torch.tensor([[-3.0, -4.0, 1.0, 2.0]]))


def test_apply_rotary_pos_emb_delegates_to_hf(monkeypatch):
    called = {}

    def fake_hf(q, k, cos, sin, position_ids, unsqueeze_dim):
        called["args"] = (q, k, cos, sin, position_ids, unsqueeze_dim)
        return "rq", "rk"

    monkeypatch.setattr(rope, "hf_apply_rotary_pos_emb", fake_hf)
    result = rope.apply_rotary_pos_emb("q", "k", "cos", "sin", position_ids="pid", unsqueeze_dim=3)

    assert result == ("rq", "rk")
    assert called["args"] == ("q", "k", "cos", "sin", "pid", 3)


def test_get_yarn_scaling_dict_includes_optional_attention_factor():
    scaling = yarn.get_yarn_scaling_dict(
        factor=3.0,
        attention_factor=1.5,
        beta_fast=16.0,
        beta_slow=2.0,
    )
    assert scaling == {
        "type": "yarn",
        "factor": 3.0,
        "beta_fast": 16.0,
        "beta_slow": 2.0,
        "attention_factor": 1.5,
    }


def test_get_yarn_config_sets_rope_scaling(monkeypatch):
    cfg = SimpleNamespace(rope_scaling=None)
    monkeypatch.setattr(yarn.AutoConfig, "from_pretrained", lambda *_args, **_kwargs: cfg)

    out = yarn.get_yarn_config("dummy/model", factor=4.0, original_max_position_embeddings=4096)
    assert out is cfg
    assert out.rope_scaling["type"] == "yarn"
    assert out.rope_scaling["factor"] == 4.0
    assert out.rope_scaling["original_max_position_embeddings"] == 4096
