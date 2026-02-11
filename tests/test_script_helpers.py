import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("datasets")


def _load_script_module(module_name: str, relative_path: str):
    root = Path(__file__).resolve().parents[1]
    script_path = root / relative_path
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_load_pe_type_from_checkpoint_prefers_training_config(tmp_path):
    evaluate_script = _load_script_module("evaluate_script_pe_type", "scripts/evaluate.py")
    (tmp_path / "training_config.yaml").write_text(yaml.safe_dump({"pe_type": "nope"}))

    pe_type = evaluate_script.load_pe_type_from_checkpoint(str(tmp_path))
    assert pe_type == "nope"


def test_load_pe_type_from_checkpoint_fallbacks_to_model_config(tmp_path):
    evaluate_script = _load_script_module("evaluate_script_pe_type_fallback", "scripts/evaluate.py")
    (tmp_path / "config.json").write_text(json.dumps({"_pe_type": "pope"}))

    pe_type = evaluate_script.load_pe_type_from_checkpoint(str(tmp_path))
    assert pe_type == "pope"


def test_load_pe_type_from_checkpoint_returns_none_when_missing(tmp_path):
    evaluate_script = _load_script_module("evaluate_script_pe_none", "scripts/evaluate.py")
    assert evaluate_script.load_pe_type_from_checkpoint(str(tmp_path)) is None


def test_run_needle_eval_stringifies_tuple_keys(monkeypatch):
    evaluate_script = _load_script_module("evaluate_script_needle", "scripts/evaluate.py")

    class FakeNeedleEvaluator:
        def __init__(self, model, tokenizer, device):
            self.model = model
            self.tokenizer = tokenizer
            self.device = device

        def evaluate(self, **kwargs):
            return {(512, 0.5): 0.75}

    monkeypatch.setattr(evaluate_script, "NeedleInHaystackEvaluator", FakeNeedleEvaluator)
    results = evaluate_script.run_needle_eval(
        model=object(),
        tokenizer=object(),
        context_lengths=[512],
        device="cpu",
        num_trials=1,
    )
    assert results == {"512_0.5": 0.75}


def test_run_passkey_eval_uses_lengths_only(monkeypatch):
    evaluate_script = _load_script_module("evaluate_script_passkey", "scripts/evaluate.py")

    class FakePasskeyEvaluator:
        def __init__(self, model, tokenizer, device):
            pass

        def evaluate_lengths_only(self, **kwargs):
            return {1024: 0.6}

    monkeypatch.setattr(evaluate_script, "PasskeyRetrievalEvaluator", FakePasskeyEvaluator)
    results = evaluate_script.run_passkey_eval(
        model=object(),
        tokenizer=object(),
        context_lengths=[1024],
        device="cpu",
        num_trials=2,
    )
    assert results == {1024: 0.6}


def test_log_to_wandb_logs_tables_and_scalars(monkeypatch):
    evaluate_script = _load_script_module("evaluate_script_wandb", "scripts/evaluate.py")
    logged = []

    class FakeTable:
        def __init__(self, columns):
            self.columns = columns
            self.rows = []

        def add_data(self, *row):
            self.rows.append(row)

    fake_wandb = SimpleNamespace(
        Table=FakeTable,
        log=lambda payload: logged.append(payload),
    )
    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)

    evaluate_script.log_to_wandb(
        results={
            "perplexity": {1024: 10.0},
            "needle": {"1024_0.5": 0.4},
            "passkey": {1024: 0.7},
        },
        checkpoint_name="run1",
        pe_type="rope",
    )

    all_keys = {key for entry in logged for key in entry.keys()}
    assert "perplexity_table" in all_keys
    assert "needle_table" in all_keys
    assert "passkey_table" in all_keys
    assert "perplexity/ctx_1024" in all_keys
    assert "passkey/ctx_1024" in all_keys


def test_deep_merge_merges_nested_dicts():
    train_script = _load_script_module("train_script_merge", "scripts/train.py")
    base = {"a": 1, "nested": {"x": 1, "y": 2}}
    override = {"nested": {"y": 99, "z": 3}, "b": 2}

    merged = train_script.deep_merge(base, override)
    assert merged == {"a": 1, "nested": {"x": 1, "y": 99, "z": 3}, "b": 2}


def test_load_config_merges_defaults(tmp_path):
    train_script = _load_script_module("train_script_load_config", "scripts/train.py")

    (tmp_path / "base.yaml").write_text(
        yaml.safe_dump(
            {
                "seed": 42,
                "training": {"max_steps": 100, "logging_steps": 10},
                "model": {"name_or_path": "base-model"},
            }
        )
    )
    child_path = tmp_path / "child.yaml"
    child_path.write_text(
        yaml.safe_dump(
            {
                "defaults": ["base"],
                "training": {"max_steps": 200},
                "model": {"name_or_path": "child-model"},
            }
        )
    )

    merged = train_script.load_config(str(child_path))
    assert merged["seed"] == 42
    assert merged["training"]["max_steps"] == 200
    assert merged["training"]["logging_steps"] == 10
    assert merged["model"]["name_or_path"] == "child-model"
