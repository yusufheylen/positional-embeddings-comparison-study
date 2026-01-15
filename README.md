# positional-embeddings-comparison-study
Comparing positional embedding strategies (RoPE, PoPE, DroPE, NoPE, YaRN) for context length generalization in transformers.

## Overview

This project empirically compares how different positional embedding (PE) methods affect a transformer's ability to generalize to longer context lengths than seen during training. We train small language models with various PE strategies and evaluate their performance across different sequence lengths.

**Key question:** How do recent approaches like PoPE (polar positional embeddings) and DroPE (dropping positional embeddings mid-training) compare for length generalization?

## Methods Compared

| Method | Description | Reference |
|--------|-------------|-----------|
| **NoPE** | No positional embeddings | Baseline |
| **RoPE** | Rotary Position Embeddings | [Su et al., 2021](https://arxiv.org/abs/2104.09864) |
| **RoPE + YaRN** | RoPE with Yet another RoPE extensioN | [Peng et al., 2023](https://arxiv.org/abs/2309.00071) |
| **PoPE** | Polar Positional Embeddings | [Paper, 2025](https://arxiv.org/abs/2509.10534) |
| **RoPE → DroPE** | RoPE then ablate and continue training | [Sakana AI, 2025](https://arxiv.org/abs/2512.12167) |
| **PoPE → DroPE** | PoPE then ablate and continue training | Novel combination |

## Results

*Coming soon*

<!-- 
TODO: Add key findings here
- Perplexity vs context length plot
- Summary table of best performing methods
- Key takeaways
-->

## Quick Start

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/positional-embeddings-comparison.git
cd positional-embeddings-comparison

# Install dependencies
pip install -r requirements.txt

# Run a training experiment
python scripts/train.py --config configs/rope.yaml

# Evaluate a checkpoint
python scripts/evaluate.py --checkpoint outputs/rope/checkpoint-final --max-length 16384
```

## Project Structure

```
positional-embeddings-comparison/
├── configs/                 # Training configurations for each PE variant
│   ├── nope.yaml
│   ├── rope.yaml
│   ├── rope_yarn.yaml
│   ├── pope.yaml
│   └── drope_rope.yaml
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   ├── transformer.py   # Base transformer implementation
│   │   └── embeddings/      # PE implementations
│   │       ├── __init__.py
│   │       ├── rope.py
│   │       ├── pope.py
│   │       ├── yarn.py
│   │       └── nope.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py       # Training loop
│   │   └── drope.py         # DroPE ablation logic
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── perplexity.py
│   │   └── needle.py        # Needle-in-haystack evaluation
│   └── data/
│       ├── __init__.py
│       └── dataset.py
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   └── run_all.sh           # Launch full experiment suite
├── notebooks/
│   └── analysis.ipynb       # Results analysis and plotting
├── results/                 # Experiment outputs (not tracked in git)
├── requirements.txt
└── README.md
```

## Experimental Setup

### Model
- **Architecture:** GPT-style decoder-only transformer
- **Size:** ~160M parameters (based on Pythia-160M configuration)
- **Training context:** 2048 tokens
- **Evaluation contexts:** 2048, 4096, 8192, 16384 tokens

### Training
- **Dataset:** SlimPajama (subset)
- **Steps:** 100,000
- **Batch size:** TBD
- **Learning rate:** TBD
- **DroPE ablation point:** Step 70,000 (70% of training)

### Evaluation
- Perplexity at varying context lengths
- Needle-in-haystack retrieval accuracy
- Passkey retrieval

## Reproducing Results

### Prerequisites
- Python 3.10+
- PyTorch 2.0+
- GPU with at least 24GB VRAM (A100 recommended) or adjust batch size

### Full reproduction

```bash
# Run all experiments (requires GPU cluster or patience)
bash scripts/run_all.sh

# Or run individual experiments
python scripts/train.py --config configs/rope.yaml --seed 42
python scripts/train.py --config configs/rope.yaml --seed 43
python scripts/train.py --config configs/rope.yaml --seed 44
```

### Using pre-trained checkpoints

*Coming soon: Links to trained checkpoints*

## Key Findings

*Coming soon*

<!--
TODO: Fill in after experiments
1. Finding 1
2. Finding 2
3. Finding 3
-->

## Blog Post

For a detailed writeup of this work, see: [Link to blog post]

## Citation

If you find this work useful, please cite:

```bibtex
@misc{yourname2025pecomparison,
  author = {Your Name},
  title = {Comparing Positional Embedding Strategies for Context Length Generalization},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/YOUR_USERNAME/positional-embeddings-comparison}
}
```

## Acknowledgments & References

This project builds on the following work:

- **DroPE:** [Sakana AI, 2025](https://arxiv.org/abs/2512.12167) — Dropping positional embeddings mid-training
  - Reference implementation: https://github.com/SakanaAI/DroPE
- **PoPE:** [Authors, 2025](https://arxiv.org/abs/2509.10534) — Polar positional embeddings
- **RoPE:** [Su et al., 2021](https://arxiv.org/abs/2104.09864) — Rotary position embeddings
- **YaRN:** [Peng et al., 2023](https://arxiv.org/abs/2309.00071) — Context extension for RoPE
- **Pythia:** [Biderman et al., 2023](https://arxiv.org/abs/2304.01373) — Model architecture reference

## License

MIT License — see [LICENSE](LICENSE) for details.