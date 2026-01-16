# Positional Embeddings Comparison Study

Comparing positional embedding strategies (RoPE, PoPE, DroPE, NoPE, YaRN) for context length generalization in transformers.

## Overview

This project empirically compares how different positional embedding (PE) methods affect a transformer's ability to generalize to longer context lengths than seen during training. We train language models **from scratch** with various PE strategies and evaluate their performance across different sequence lengths.

**Key questions:**
1. Which PE method enables best length generalization?
2. Does the "scaffold hypothesis" hold - can PE be dropped after initial training?
3. How do DroPE variants (dropping PE mid-training) compare across different initial PE methods?

## Methods Compared

| Run | Method | Description | Reference |
|-----|--------|-------------|-----------|
| 0 | **NoPE** | No positional embeddings | Baseline |
| 1 | **RoPE** | Rotary Position Embeddings | [Su et al., 2021](https://arxiv.org/abs/2104.09864) |
| 2 | **RoPE + YaRN** | RoPE with Yet another RoPE extensioN | [Peng et al., 2023](https://arxiv.org/abs/2309.00071) |
| 3 | **PoPE** | Polar Positional Embeddings | [arXiv:2509.10534](https://arxiv.org/abs/2509.10534) |
| 4a | **RoPE → NoPE** | RoPE then drop PE (DroPE) | [Sakana AI, 2025](https://arxiv.org/abs/2512.12167) |
| 4b | **YaRN → NoPE** | YaRN then drop PE | Novel combination |
| 4c | **PoPE → NoPE** | PoPE then drop PE | Novel combination |

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
- **Total steps (S):** 16,000
- **DroPE switch point (T):** 14,000 (87.5% of training)
- **Batch size:** 64
- **Learning rate:** 3e-4
- **Warmup:** 3% of steps (~480 steps)

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
├── configs/                 # Training configurations
│   ├── base.yaml            # Shared defaults
│   ├── rope_scratch.yaml    # RoPE from scratch
│   ├── nope_scratch.yaml    # NoPE from scratch
│   ├── pope_scratch.yaml    # PoPE from scratch
│   ├── yarn_scratch.yaml    # YaRN from scratch
│   ├── drope_rope.yaml      # RoPE → NoPE at step 14k
│   ├── drope_yarn.yaml      # YaRN → NoPE at step 14k
│   └── drope_pope.yaml      # PoPE → NoPE at step 14k
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
- Python 3.10+
- PyTorch 2.0+
- GPU with at least 40GB VRAM (A100 recommended)
- ~35-50 GPU hours for all 7 experiments

### Full reproduction

```bash
# Run all 7 experiments sequentially
./scripts/run_all_experiments.sh

# Or run individually
python scripts/train.py --config configs/rope_scratch.yaml --seed 42
python scripts/train.py --config configs/nope_scratch.yaml --seed 42
python scripts/train.py --config configs/yarn_scratch.yaml --seed 42
python scripts/train.py --config configs/pope_scratch.yaml --seed 42
python scripts/train.py --config configs/drope_rope.yaml --seed 42
python scripts/train.py --config configs/drope_yarn.yaml --seed 42
python scripts/train.py --config configs/drope_pope.yaml --seed 42

# Evaluate all checkpoints
./scripts/run_evals.sh
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
