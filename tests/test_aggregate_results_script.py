import importlib.util
import json
from pathlib import Path

import pytest

pytest.importorskip("pandas")


def _load_script_module(module_name: str, relative_path: str):
    root = Path(__file__).resolve().parents[1]
    script_path = root / relative_path
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_load_results_reads_eval_files(tmp_path):
    aggregate = _load_script_module("aggregate_results_script", "scripts/aggregate_results.py")

    run_dir = tmp_path / "run_a"
    run_dir.mkdir()
    (run_dir / "eval_results.json").write_text(json.dumps({"perplexity": {"1024": 12.3}}))

    ignored = tmp_path / "run_b"
    ignored.mkdir()
    (ignored / "other.json").write_text("{}")

    results = aggregate.load_results(str(tmp_path))
    assert set(results.keys()) == {"run_a"}
    assert results["run_a"]["perplexity"]["1024"] == 12.3


def test_create_tables_shape_and_empty_behavior():
    aggregate = _load_script_module("aggregate_results_tables", "scripts/aggregate_results.py")

    results = {
        "rope": {
            "pe_type": "rope",
            "perplexity": {"2048": 10.0},
            "passkey": {"2048": 0.6},
            "needle": {"2048_0.5": 0.4},
        },
        "nope": {"pe_type": "nope"},
    }

    ppl_df = aggregate.create_perplexity_table(results)
    assert set(ppl_df["method"]) == {"rope"}
    assert float(ppl_df.iloc[0]["ppl_2048"]) == pytest.approx(10.0)

    passkey_df = aggregate.create_passkey_table(results)
    assert set(passkey_df["method"]) == {"rope"}
    assert float(passkey_df.iloc[0]["passkey_2048"]) == pytest.approx(0.6)

    needle_df = aggregate.create_needle_table(results)
    assert len(needle_df) == 1
    assert int(needle_df.iloc[0]["context_length"]) == 2048
    assert float(needle_df.iloc[0]["position"]) == pytest.approx(0.5)
    assert float(needle_df.iloc[0]["accuracy"]) == pytest.approx(0.4)

    assert aggregate.create_perplexity_table({"x": {}}).empty
    assert aggregate.create_passkey_table({"x": {}}).empty
    assert aggregate.create_needle_table({"x": {}}).empty


def test_print_summary_outputs_sections(capsys):
    aggregate = _load_script_module("aggregate_results_print", "scripts/aggregate_results.py")
    results = {
        "rope": {
            "pe_type": "rope",
            "perplexity": {"1024": 9.9},
            "passkey": {"1024": 0.7},
            "needle": {"1024_0.0": 0.4, "1024_1.0": 0.6},
        }
    }

    aggregate.print_summary(results)
    out = capsys.readouterr().out

    assert "EVALUATION RESULTS SUMMARY" in out
    assert "Perplexity" in out
    assert "Passkey Retrieval Accuracy" in out
    assert "Needle-in-Haystack" in out
