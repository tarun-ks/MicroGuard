# MicroGuard

**Free, real-time hallucination detection for any RAG pipeline.**
A 270M model outperforms a 1.1B model. All models run locally in <100ms at $0/eval.

[![License](https://img.shields.io/badge/License-Apache%202.0-green)](LICENSE)
[![Models on HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Models-HuggingFace-yellow)](https://huggingface.co/tarun5986)
[![Demo](https://img.shields.io/badge/%F0%9F%8E%AE%20Live-Demo-blue)](https://huggingface.co/spaces/tarun5986/MicroGuard)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue)]()

---

> **Key finding:** Architecture quality matters more than parameter count.
> Gemma-270M (270M params) outperforms TinyLlama (1.1B params) by **2 percentage points** in balanced accuracy and **8.6 points** in F1-score — with 4x fewer parameters.

---

## Try it now — no install needed

**[Launch Live Demo on HuggingFace Spaces](https://huggingface.co/spaces/tarun5986/MicroGuard)**

Paste any context + question + answer and get a FAITHFUL / UNFAITHFUL verdict instantly.

## What is this?

If you're running a RAG pipeline, you need to know whether the generated answers actually match the retrieved context. The standard approach is to use GPT-4 as a judge, but that costs ~$0.002 per call and adds 500-2000ms latency. For production systems handling thousands of requests, that adds up fast.

MicroGuard takes a different approach: fine-tune small language models (135M-1B parameters) to do the same job locally, for free, in under 100ms.

## Available Models

| Model | Params | Bal. Acc. | F1 | Latency | HuggingFace |
|-------|--------|-----------|-----|---------|-------------|
| **Gemma-1B** | 1.0B | **69.4%** | **0.721** | 88ms | [Download](https://huggingface.co/tarun5986/MicroGuard-Gemma-1B) |
| Qwen-0.5B | 500M | 67.6% | 0.698 | 56ms | [Download](https://huggingface.co/tarun5986/MicroGuard-Qwen-0.5B) |
| Gemma-270M | 270M | 67.0% | 0.688 | 60ms | [Download](https://huggingface.co/tarun5986/MicroGuard-Gemma-270M) |
| TinyLlama-1.1B | 1.1B | 64.7% | 0.589 | 53ms | [Download](https://huggingface.co/tarun5986/MicroGuard-TinyLlama-1.1B) |
| SmolLM-135M | 135M | 64.3% | 0.661 | 72ms | [Download](https://huggingface.co/tarun5986/MicroGuard-SmolLM-135M) |

All models are LoRA adapters — small, fast to download, and easy to swap.

### Baselines (for reference)

| Model | Params | Bal. Acc. | F1 | Notes |
|-------|--------|-----------|-----|-------|
| RoBERTa-large (fine-tuned) | 355M | 68.8% | 0.720 | Encoder baseline |
| RoBERTa-base (fine-tuned) | 125M | 68.4% | 0.716 | Encoder baseline |
| DeBERTa-v3 NLI (zero-shot) | 184M | 50.7% | 0.485 | No training needed |

A few things stood out:

- Gemma-1B edges past both RoBERTa baselines, which is interesting since generative models don't usually beat encoders on classification tasks
- Architecture matters more than raw size — Gemma-270M beats TinyLlama-1.1B despite having 4x fewer parameters
- All fine-tuned models dramatically beat the zero-shot NLI approach (+13-19 points), confirming that task-specific training is essential

## Quick Start

```bash
pip install torch transformers peft accelerate
```

```python
from microguard import MicroGuard

guard = MicroGuard(model="gemma-270m")  # also: "qwen-0.5b", "gemma-1b"
result = guard.check(
    context="The Eiffel Tower was built in 1889 by Gustave Eiffel in Paris.",
    question="Who built the Eiffel Tower?",
    answer="The Eiffel Tower was built by Gustave Eiffel in 1889."
)
print(result)
# {'verdict': 'FAITHFUL', 'confidence': 51.2, 'latency_ms': 64.0}
```

You can also point it at a local adapter if you've trained your own:

```python
guard = MicroGuard(
    model="path/to/your/adapter",
    base_model="google/gemma-3-270m-it"
)
```

## How it works

The approach is straightforward:

1. Take an off-the-shelf small LM (Gemma, Qwen, etc.)
2. Fine-tune it with LoRA on faithfulness-labeled data
3. At inference, format the (context, question, answer) as a prompt and compare logits for "FAITHFUL" vs "UNFAITHFUL" tokens

The logit comparison (what we call "constrained decoding") is important. When we tried standard text generation, about 13% of outputs were garbage (partial words, random text, etc.). Comparing logits directly gives a clean binary decision every time.

## Training your own

### Data setup

```bash
python scripts/download_datasets.py
python scripts/preprocess_datasets.py
```

This pulls RAGBench (95K examples), RAGTruth (18K), and HaluBench (15K), converts them to a unified format, and creates train/val/test splits.

### Training

```bash
# Train Gemma-270M (smallest, good for experimentation)
python scripts/train_slm.py --model gemma3_270m --max_train_samples 40000 --epochs 3

# Or train the best model
python scripts/train_slm.py --model gemma3_1b --max_train_samples 40000 --epochs 3
```

On a T4 GPU (Google Colab free tier), Gemma-270M takes about 2 hours. On an A100, it's around 20 minutes.

### Colab notebooks

If you don't have a local GPU:

- [MicroGuard_A100.ipynb](notebooks/MicroGuard_A100.ipynb) — runs everything end-to-end on a paid A100 (~5 hours)
- [MicroGuard_Colab_Resume.ipynb](notebooks/MicroGuard_Colab_Resume.ipynb) — designed for the free tier, saves checkpoints to Drive so you don't lose progress on disconnects

## Project layout

```
MicroGuard/
├── microguard/           # pip-installable package
│   ├── __init__.py
│   └── classifier.py     # MicroGuard class
├── scripts/              # training and data prep
│   ├── download_datasets.py
│   ├── preprocess_datasets.py
│   └── train_slm.py
├── demo/                 # Gradio app for HuggingFace Spaces
├── notebooks/            # Colab notebooks
├── results/              # experiment result JSONs
└── figures/              # plots from the paper
```

## Limitations

Worth being upfront about:

- 69% balanced accuracy means roughly 1 in 3 unfaithful answers slip through. This is useful as a pre-filter but not a standalone quality gate.
- Trained on English data only. Multilingual RAG would need separate fine-tuning.
- The binary faithful/unfaithful split doesn't capture severity. An answer that gets one date wrong and an answer that fabricates an entire paragraph both get the same label.
- Longer contexts get truncated to ~900 characters. If the relevant evidence is buried deep in the retrieved passage, the model might miss it.

## Citation

```bibtex
@article{sharma2026microguard,
  title={MicroGuard: Sub-Billion Parameter Faithfulness Classification
         for Real-Time Retrieval-Augmented Generation Quality Assurance},
  author={Sharma, Tarun},
  journal={IEEE Access},
  year={2026},
  note={Under review. DOI pending.}
}
```

## License

Apache 2.0
