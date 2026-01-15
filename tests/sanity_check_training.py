"""Training sanity check - verify training infrastructure works end-to-end.

Runs 1 training iteration with a small model to verify:
- Model loading with different PE types
- Data collation
- Forward/backward pass
- Logging
"""

import sys
import torch
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.base import create_model, get_best_attn_implementation
from src.data.dataset import get_data_collator
from src.training.trainer import PETrainer, create_training_args


def run_sanity_check(pe_type: str = "rope"):
    """Run a single training iteration sanity check.

    Args:
        pe_type: Positional embedding type to test.
    """
    print(f"\n{'='*60}")
    print(f"Sanity Check: {pe_type.upper()}")
    print(f"{'='*60}")

    # Use smallest available model
    model_name = "HuggingFaceTB/SmolLM2-135M"

    print(f"\n1. Loading model: {model_name}")
    print(f"   PE type: {pe_type}")
    print(f"   Attention: {get_best_attn_implementation()}")

    # Load model
    model = create_model(
        model_name,
        pe_type=pe_type,
        trust_remote_code=True,
        dtype=torch.float32,  # Use float32 for CPU/MPS
    )

    # Get tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"   Model loaded: {model.config.num_hidden_layers} layers, "
          f"{sum(p.numel() for p in model.parameters())/1e6:.1f}M params")

    # Create dummy dataset
    print("\n2. Creating dummy dataset")
    dummy_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Python is a popular programming language for data science.",
        "Transformers have revolutionized natural language processing.",
    ]

    # Tokenize
    encodings = tokenizer(
        dummy_texts,
        padding="max_length",
        truncation=True,
        max_length=64,
        return_tensors="pt",
    )

    # Create simple dataset
    from torch.utils.data import Dataset

    class SimpleDataset(Dataset):
        def __init__(self, encodings):
            self.encodings = encodings

        def __len__(self):
            return len(self.encodings["input_ids"])

        def __getitem__(self, idx):
            return {
                "input_ids": self.encodings["input_ids"][idx].tolist(),
                "attention_mask": self.encodings["attention_mask"][idx].tolist(),
                "labels": self.encodings["input_ids"][idx].tolist(),
            }

    dataset = SimpleDataset(encodings)
    print(f"   Dataset size: {len(dataset)} samples")

    # Create data collator
    print("\n3. Setting up training")
    collator = get_data_collator(
        tokenizer,
        attn_implementation=get_best_attn_implementation(),
        mask_past_sequences=True,
    )

    # Create training args (1 step only)
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        training_args = create_training_args(
            output_dir=tmpdir,
            max_steps=1,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=1,
            learning_rate=1e-4,
            logging_steps=1,
            save_steps=1000,  # Don't save
            bf16=False,  # Use float32 for compatibility
            tf32=False,
            gradient_checkpointing=False,
            report_to="none",  # No wandb
            dataloader_num_workers=0,
        )

        # Create trainer
        trainer = PETrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            data_collator=collator,
            processing_class=tokenizer,
        )

        print("   Trainer created")

        # Run training
        print("\n4. Running 1 training step...")
        try:
            result = trainer.train()
            print(f"   ✓ Training step completed!")
            print(f"   Loss: {result.training_loss:.4f}")
            print(f"   Global step: {trainer.state.global_step}")
        except Exception as e:
            print(f"   ✗ Training failed: {e}")
            raise

    print(f"\n{'='*60}")
    print(f"✓ Sanity check PASSED for {pe_type.upper()}")
    print(f"{'='*60}\n")

    return True


def main():
    """Run sanity checks for all PE types."""
    print("\n" + "="*60)
    print("PE Training Infrastructure Sanity Check")
    print("="*60)

    # Test each PE type
    pe_types = ["rope", "nope"]  # Skip pope/yarn for quick check

    results = {}
    for pe_type in pe_types:
        try:
            results[pe_type] = run_sanity_check(pe_type)
        except Exception as e:
            print(f"✗ {pe_type.upper()} failed: {e}")
            results[pe_type] = False

    # Summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    for pe_type, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {pe_type.upper()}: {status}")

    all_passed = all(results.values())
    print("\n" + ("✓ All sanity checks passed!" if all_passed else "✗ Some checks failed"))

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
