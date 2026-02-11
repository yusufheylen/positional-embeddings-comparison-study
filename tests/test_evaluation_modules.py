import math
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from src.evaluation.needle import NeedleInHaystackEvaluator
from src.evaluation.passkey import PasskeyRetrievalEvaluator
from src.evaluation.perplexity import PerplexityEvaluator


class DummyBatch(dict):
    @property
    def input_ids(self):
        return self["input_ids"]

    def to(self, device):
        return self


class DummyTokenizer:
    def __init__(self):
        self.pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return list(range(len(text.split())))

    def decode(self, tokens, skip_special_tokens=True):
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        return " ".join(f"tok{tok}" for tok in tokens)

    def __call__(self, text, return_tensors="pt", truncation=False, add_special_tokens=True):
        ids = torch.arange(max(len(text.split()), 1), dtype=torch.long).unsqueeze(0)
        return DummyBatch({"input_ids": ids})


class DummyModel:
    def __init__(self, loss_value=math.log(2), generated_ids=None):
        self.loss_value = float(loss_value)
        self.generated_ids = generated_ids or [99]

    def to(self, device):
        return self

    def eval(self):
        return self

    def __call__(self, input_ids, labels=None):
        return SimpleNamespace(loss=torch.tensor(self.loss_value, dtype=torch.float32))

    def generate(self, **kwargs):
        input_ids = kwargs["input_ids"]
        suffix = torch.tensor([self.generated_ids], dtype=torch.long)
        return torch.cat([input_ids, suffix], dim=1)


class FakeDataset(list):
    def select(self, idxs):
        return [self[i] for i in idxs]


def test_perplexity_evaluate_at_length_computes_expected_value():
    evaluator = PerplexityEvaluator(DummyModel(loss_value=math.log(2)), DummyTokenizer(), device="cpu")
    ppl = evaluator._evaluate_at_length(
        texts=["a b c d e f", "one two three four five six seven"],
        context_length=4,
        stride=2,
        batch_size=1,
    )
    assert ppl == pytest.approx(2.0, rel=1e-4)


def test_perplexity_evaluate_at_length_returns_inf_when_no_tokens():
    evaluator = PerplexityEvaluator(DummyModel(), DummyTokenizer(), device="cpu")
    ppl = evaluator._evaluate_at_length(
        texts=["short", "tiny text"],
        context_length=10,
        stride=5,
        batch_size=1,
    )
    assert math.isinf(ppl)


def test_perplexity_evaluate_dataset_limits_samples(monkeypatch):
    evaluator = PerplexityEvaluator(DummyModel(), DummyTokenizer(), device="cpu")
    recorded = {}

    def fake_evaluate(texts, context_lengths, **kwargs):
        recorded["texts"] = texts
        recorded["context_lengths"] = context_lengths
        return {16: 1.23}

    monkeypatch.setattr(evaluator, "evaluate", fake_evaluate)
    dataset = FakeDataset([{"text": "a"}, {"text": "b"}, {"text": "c"}])
    result = evaluator.evaluate_dataset(dataset, context_lengths=[16], max_samples=2)

    assert result == {16: 1.23}
    assert recorded["texts"] == ["a", "b"]
    assert recorded["context_lengths"] == [16]


def test_needle_generate_haystack_hits_target_token_length():
    evaluator = NeedleInHaystackEvaluator(DummyModel(), DummyTokenizer(), device="cpu")
    haystack = evaluator.generate_haystack(target_length=12, filler_text="alpha beta")
    assert len(evaluator.tokenizer.encode(haystack, add_special_tokens=False)) == 12


def test_needle_insert_needle_by_fraction():
    evaluator = NeedleInHaystackEvaluator(DummyModel(), DummyTokenizer(), device="cpu")
    out = evaluator.insert_needle("one two three four", "NEEDLE", position_fraction=0.5)
    words = out.split()
    assert words[2] == "NEEDLE"


def test_needle_evaluate_single_checks_expected_answer(monkeypatch):
    tokenizer = DummyTokenizer()
    monkeypatch.setattr(tokenizer, "decode", lambda *_args, **_kwargs: "the answer is 1234")
    evaluator = NeedleInHaystackEvaluator(DummyModel(generated_ids=[7, 8]), tokenizer, device="cpu")

    success, response = evaluator.evaluate_single("hay", "needle", "query", "1234")
    assert success is True
    assert "1234" in response


def test_needle_evaluate_aggregates_trials(monkeypatch):
    evaluator = NeedleInHaystackEvaluator(DummyModel(), DummyTokenizer(), device="cpu")
    monkeypatch.setattr(evaluator, "generate_haystack", lambda *_args, **_kwargs: "hay")
    monkeypatch.setattr(evaluator, "insert_needle", lambda *_args, **_kwargs: "full")

    outcomes = iter([(True, "ok"), (False, "no"), (True, "ok"), (True, "ok")])
    monkeypatch.setattr(evaluator, "evaluate_single", lambda *_args, **_kwargs: next(outcomes))

    results = evaluator.evaluate(context_lengths=[32], position_fractions=[0.0, 1.0], num_trials=2)
    assert results[(32, 0.0)] == pytest.approx(0.5)
    assert results[(32, 1.0)] == pytest.approx(1.0)


def test_needle_results_to_heatmap_data_builds_matrix():
    evaluator = NeedleInHaystackEvaluator(DummyModel(), DummyTokenizer(), device="cpu")
    contexts, positions, matrix = evaluator.results_to_heatmap_data(
        {(128, 0.0): 0.1, (128, 0.5): 0.2, (256, 0.5): 0.9}
    )
    assert contexts == [128, 256]
    assert positions == [0.0, 0.5]
    assert matrix == [[0.1, 0.2], [0.0, 0.9]]


def test_passkey_generate_passkey_digits_only():
    evaluator = PasskeyRetrievalEvaluator(DummyModel(), DummyTokenizer(), device="cpu")
    passkey = evaluator.generate_passkey(length=6)
    assert len(passkey) == 6
    assert passkey.isdigit()


def test_passkey_create_prompt_contains_key_and_query():
    evaluator = PasskeyRetrievalEvaluator(DummyModel(), DummyTokenizer(), device="cpu")
    prompt = evaluator.create_prompt("65432", target_length=128, position_fraction=0.25)
    assert "65432" in prompt
    assert evaluator.QUERY in prompt


def test_passkey_evaluate_single_detects_success(monkeypatch):
    tokenizer = DummyTokenizer()
    monkeypatch.setattr(tokenizer, "decode", lambda *_args, **_kwargs: "passkey is 56789")
    evaluator = PasskeyRetrievalEvaluator(DummyModel(generated_ids=[5, 6]), tokenizer, device="cpu")

    success, response = evaluator.evaluate_single("prompt", "56789")
    assert success is True
    assert "56789" in response


def test_passkey_evaluate_and_lengths_only_aggregate(monkeypatch):
    evaluator = PasskeyRetrievalEvaluator(DummyModel(), DummyTokenizer(), device="cpu")
    monkeypatch.setattr(evaluator, "generate_passkey", lambda length=5: "11111")
    monkeypatch.setattr(evaluator, "create_prompt", lambda *_args, **_kwargs: "prompt")

    sequence = iter([(True, "ok"), (False, "no"), (True, "ok"), (True, "ok")])
    monkeypatch.setattr(evaluator, "evaluate_single", lambda *_args, **_kwargs: next(sequence))

    pair_results = evaluator.evaluate(context_lengths=[64], position_fractions=[0.0, 1.0], num_trials=2)
    assert pair_results[(64, 0.0)] == pytest.approx(0.5)
    assert pair_results[(64, 1.0)] == pytest.approx(1.0)

    lengths_sequence = iter([(True, "ok"), (False, "no"), (True, "ok"), (True, "ok")])
    monkeypatch.setattr(evaluator, "evaluate_single", lambda *_args, **_kwargs: next(lengths_sequence))
    length_results = evaluator.evaluate_lengths_only(context_lengths=[128, 256], num_trials=2)
    assert length_results[128] == pytest.approx(0.5)
    assert length_results[256] == pytest.approx(1.0)
