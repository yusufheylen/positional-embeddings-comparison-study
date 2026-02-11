from itertools import islice

import pytest

pytest.importorskip("torch")
datasets_mod = pytest.importorskip("datasets")
pytest.importorskip("transformers")

from src.data.dataset import load_dataset_for_training, tokenize_and_chunk_dataset


class DummyTokenizer:
    eos_token_id = 2

    def __call__(self, text, add_special_tokens=False, return_attention_mask=True):
        token_count = len(text.split())
        return {
            "input_ids": list(range(token_count)),
            "attention_mask": [1] * token_count,
        }


def test_load_dataset_for_training_applies_num_samples_when_not_streaming(monkeypatch):
    class FakeDataset:
        def __len__(self):
            return 10

        def select(self, idxs):
            return list(idxs)

    monkeypatch.setattr("src.data.dataset.load_dataset", lambda *args, **kwargs: FakeDataset())
    selected = load_dataset_for_training(
        dataset_name="dummy",
        split="train",
        streaming=False,
        num_samples=3,
    )
    assert selected == [0, 1, 2]


def test_tokenize_and_chunk_dataset_non_streaming_path():
    dataset = datasets_mod.Dataset.from_dict({"text": ["a b c d", "e f g h"]})
    out = tokenize_and_chunk_dataset(
        dataset=dataset,
        tokenizer=DummyTokenizer(),
        text_column="text",
        max_length=3,
        add_eos=True,
        streaming=False,
        num_proc=1,
    )

    rows = list(out)
    assert len(rows) > 0
    assert all(len(row["input_ids"]) == 3 for row in rows)
    assert all(row["labels"] == row["input_ids"] for row in rows)


def test_tokenize_and_chunk_dataset_streaming_path():
    stream = datasets_mod.IterableDataset.from_generator(
        lambda: iter([{"text": "a b c d"}, {"text": "e f g h"}])
    )
    out = tokenize_and_chunk_dataset(
        dataset=stream,
        tokenizer=DummyTokenizer(),
        text_column="text",
        max_length=3,
        add_eos=True,
        streaming=True,
    )

    first_two = list(islice(iter(out), 2))
    assert len(first_two) == 2
    assert all(len(row["input_ids"]) == 3 for row in first_two)
    assert all(row["labels"] == row["input_ids"] for row in first_two)
