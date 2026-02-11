import json

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from src.training.trainer import CheckpointResumptionCallback, PETrainer, create_training_args


def test_create_training_args_sets_min_lr_for_cosine():
    args = create_training_args(
        output_dir="tmp-out",
        min_learning_rate=1e-5,
        report_to="none",
        bf16=False,
        tf32=False,
        gradient_checkpointing=False,
    )
    assert args.lr_scheduler_kwargs["min_lr"] == pytest.approx(1e-5)


def test_create_training_args_uses_none_when_no_scheduler_kwargs():
    args = create_training_args(
        output_dir="tmp-out",
        min_learning_rate=None,
        report_to="none",
        bf16=False,
        tf32=False,
        gradient_checkpointing=False,
    )
    assert args.lr_scheduler_kwargs is None


def test_get_resumption_state_reads_metadata(tmp_path):
    metadata = {"global_step": 123, "samples_seen": 456, "epoch": 1.5}
    state_file = tmp_path / "pe_trainer_state.json"
    state_file.write_text(json.dumps(metadata))

    state = PETrainer.get_resumption_state(str(tmp_path))
    assert state == metadata


def test_get_resumption_state_missing_file_returns_empty_dict(tmp_path):
    state = PETrainer.get_resumption_state(str(tmp_path))
    assert state == {}


def test_checkpoint_resumption_callback_loads_state_on_train_begin(tmp_path):
    metadata = {"global_step": 10, "samples_seen": 88}
    (tmp_path / "pe_trainer_state.json").write_text(json.dumps(metadata))

    callback = CheckpointResumptionCallback(checkpoint_path=str(tmp_path))
    callback.on_train_begin(args=None, state=None, control=None)

    assert callback.resumption_state == metadata
