# Positional Embeddings Comparison Study

Comparing positional embedding strategies (RoPE, PoPE, DroPE, NoPE, YaRN) for context length generalization in transformers.

## Overview

This project empirically compares how different positional embedding (PE) methods affect a transformer's ability to generalize to longer context lengths than seen during training. We train language models **from scratch** with various PE strategies and evaluate their performance across different sequence lengths.

**Key questions:**
1. Which PE method enables best length generalization?
2. Does the "scaffold hypothesis" hold - can PE be dropped after initial training?
3. How do DroPE variants (dropping PE mid-training) compare across different initial PE methods?

## Methods Compared

### Baseline Runs (from scratch, 16k steps)

| Run | Method | Description | Reference |
|-----|--------|-------------|-----------|
| 0 | **NoPE** | No positional embeddings | Baseline |
| 1 | **RoPE** | Rotary Position Embeddings | [Su et al., 2021](https://arxiv.org/abs/2104.09864) |
| 2 | **RoPE + YaRN** | RoPE with Yet another RoPE extensioN | [Peng et al., 2023](https://arxiv.org/abs/2309.00071) |
| 3 | **PoPE** | Polar Positional Embeddings | [arXiv:2509.10534](https://arxiv.org/abs/2509.10534) |

### Scaffold Runs (load PE checkpoint, convert to NoPE, continue training)

| Run | Method | Source | Remaining Steps |
|-----|--------|--------|-----------------|
| scaffold_rope_10k | **RoPE → NoPE** | RoPE @ 10k steps | 6k |
| scaffold_rope_15k | **RoPE → NoPE** | RoPE @ 15k steps | 1k |
| scaffold_yarn_10k | **YaRN → NoPE** | YaRN @ 10k steps | 6k |
| scaffold_yarn_15k | **YaRN → NoPE** | YaRN @ 15k steps | 1k |
| scaffold_pope_10k | **PoPE → NoPE** | PoPE @ 10k steps | 6k |
| scaffold_pope_15k | **PoPE → NoPE** | PoPE @ 15k steps | 1k |

Inspired by [DroPE (Sakana AI, 2025)](https://arxiv.org/abs/2512.12167). Scaffold runs use a fresh LR schedule after conversion.

## Experimental Setup

### Training Approach: From-Scratch

All models are trained **from random initialization** (not continued pretraining) to ensure:
- Fair comparison: all methods see identical data from identical starting point
- PoPE compatibility: PoPE cannot be retrofitted to RoPE models
- Clear demonstration of PE effects during training

### Model & Data
- **Architecture:** LLaMA-style decoder-only transformer (~360M-494M parameters)
- **Dataset:** FineWeb (HuggingFaceFW/fineweb)
- **Training context:** 1024 tokens
- **Evaluation contexts:** 1024, 2048, 4096, 8192, 16384 tokens

### Training Parameters (based on DroPE paper)
- **Total steps (S):** 16,000 (baselines)
- **Scaffold switch points:** 10,000 and 15,000 steps
- **Batch size:** 64
- **Learning rate (baselines):** 3e-4 → 3e-5 (cosine)
- **Learning rate (scaffold from 10k):** 1e-4 → 1e-5 (fresh cosine, 180 step warmup)
- **Learning rate (scaffold from 15k):** 5e-5 → 5e-6 (fresh cosine, 30 step warmup)
- **Warmup (baselines):** 3% of steps (~480 steps)

### Evaluation
- Perplexity at varying context lengths
- Needle-in-haystack retrieval accuracy
- Passkey retrieval

## Results

*Coming soon after experiments complete*

## Quick Start

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/positional-embeddings-comparison.git
cd positional-embeddings-comparison

# Setup environment
conda create -n pe-study python=3.11 -y
conda activate pe-study
pip install -r requirements.txt
./scripts/install.sh

# Login for datasets
huggingface-cli login
wandb login

# Run a training experiment (after config update)
python scripts/train.py --config configs/rope_scratch.yaml

# Evaluate a checkpoint
python scripts/evaluate.py --checkpoint outputs/rope --eval all --context-lengths 1024 2048 4096 8192 16384
```

## Project Structure

```
positional-embeddings-comparison/
├── configs/                       # Training configurations
│   ├── base.yaml                  # Shared defaults (from-scratch)
│   ├── rope_scratch.yaml          # RoPE from scratch (run 1)
│   ├── nope_scratch.yaml          # NoPE from scratch (run 0)
│   ├── pope_scratch.yaml          # PoPE from scratch (run 3)
│   ├── yarn_scratch.yaml          # YaRN from scratch (run 2)
│   ├── scaffold_base.yaml         # Shared scaffold defaults
│   ├── scaffold_rope_10k.yaml     # RoPE→NoPE from 10k checkpoint
│   ├── scaffold_rope_15k.yaml     # RoPE→NoPE from 15k checkpoint
│   ├── scaffold_yarn_10k.yaml     # YaRN→NoPE from 10k checkpoint
│   ├── scaffold_yarn_15k.yaml     # YaRN→NoPE from 15k checkpoint
│   ├── scaffold_pope_10k.yaml     # PoPE→NoPE from 10k checkpoint
│   └── scaffold_pope_15k.yaml     # PoPE→NoPE from 15k checkpoint
├── src/
│   ├── models/
│   │   ├── base.py          # Model factory
│   │   └── embeddings/      # PE implementations
│   │       ├── rope.py
│   │       ├── pope.py
│   │       ├── yarn.py
│   │       └── nope.py
│   ├── training/
│   │   ├── trainer.py       # Enhanced trainer with grad_norm logging
│   │   └── drope_callback.py # Mid-training PE switch
│   ├── evaluation/
│   │   ├── perplexity.py
│   │   ├── needle.py        # Needle-in-haystack
│   │   └── passkey.py       # Passkey retrieval
│   └── data/
│       └── dataset.py       # Data loading and collators
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   └── run_all.sh
├── tests/
│   ├── test_pe_implementations.py
│   └── sanity_check_training.py
└── notebooks/
    └── analysis.ipynb       # Results analysis
```

## Reproducing Results

### Prerequisites
- Python 3.11+
- PyTorch 2.0+
- GPU with at least 24GB VRAM (RTX 4090 or A100 recommended)
- ~20-30 GPU hours for all 10 experiments (4 baselines + 6 scaffold)

### Full reproduction

```bash
# Step 1: Run 4 baseline experiments (16k steps each)
python scripts/train.py --config configs/nope_scratch.yaml --seed 42
python scripts/train.py --config configs/rope_scratch.yaml --seed 42
python scripts/train.py --config configs/yarn_scratch.yaml --seed 42
python scripts/train.py --config configs/pope_scratch.yaml --seed 42

# Step 2: Run 6 scaffold experiments (load baseline checkpoints, convert to NoPE)
# Requires baseline checkpoints from step 1
bash scripts/run_scaffold_experiments.sh

# Step 3: Evaluate all checkpoints
bash scripts/run_evals.sh
```

## Key Findings

*Coming soon*

## Citation

If you find this work useful, please cite:

```bibtex
@misc{heylen2026pecomparison,
  author = {Yusuf Heylen},
  title = {Comparing Positional Embedding Strategies for Context Length Generalization},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/yusufheylen/positional-embeddings-comparison}
}
```

## Acknowledgments & References

This project builds on:

- **DroPE:** [Sakana AI, 2025](https://arxiv.org/abs/2512.12167) — Dropping positional embeddings mid-training
  - Reference: https://github.com/SakanaAI/DroPE
- **PoPE:** [arXiv:2509.10534](https://arxiv.org/abs/2509.10534) — Polar positional embeddings
- **RoPE:** [Su et al., 2021](https://arxiv.org/abs/2104.09864) — Rotary position embeddings
- **YaRN:** [Peng et al., 2023](https://arxiv.org/abs/2309.00071) — Context extension for RoPE

## License

MIT License — see [LICENSE](LICENSE) for details.
