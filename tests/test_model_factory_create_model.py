from types import SimpleNamespace

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from src.models import base


def test_create_model_auto_sets_eager_for_pope(monkeypatch):
    config = SimpleNamespace(_attn_implementation=None)
    dummy_model = object()

    monkeypatch.setattr(base, "get_model_config", lambda *_args, **_kwargs: config)
    monkeypatch.setattr(base.AutoModelForCausalLM, "from_pretrained", lambda *_args, **_kwargs: dummy_model)

    converted = object()
    monkeypatch.setattr("src.models.embeddings.pope.convert_to_pope", lambda model: converted)

    out = base.create_model("dummy", pe_type="pope", attn_implementation="auto")
    assert config._attn_implementation == "eager"
    assert out is converted


def test_create_model_uses_detected_attention_for_rope(monkeypatch):
    config = SimpleNamespace(_attn_implementation=None)
    dummy_model = object()

    monkeypatch.setattr(base, "get_model_config", lambda *_args, **_kwargs: config)
    monkeypatch.setattr(base, "get_best_attn_implementation", lambda: "sdpa")
    monkeypatch.setattr(base.AutoModelForCausalLM, "from_pretrained", lambda *_args, **_kwargs: dummy_model)

    out = base.create_model("dummy", pe_type="rope", attn_implementation="auto")
    assert config._attn_implementation == "sdpa"
    assert out is dummy_model


def test_create_model_nope_applies_conversion(monkeypatch):
    config = SimpleNamespace(_attn_implementation=None)
    dummy_model = object()
    converted = object()

    monkeypatch.setattr(base, "get_model_config", lambda *_args, **_kwargs: config)
    monkeypatch.setattr(base.AutoModelForCausalLM, "from_pretrained", lambda *_args, **_kwargs: dummy_model)
    monkeypatch.setattr("src.models.embeddings.nope.convert_to_nope", lambda model, attention_type: converted)

    out = base.create_model("dummy", pe_type="nope", attention_type="qk_norm_nope", attn_implementation="eager")
    assert config._attn_implementation == "eager"
    assert out is converted
