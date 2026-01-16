# TreatyBench

Benchmarking LLM Capabilities in Treaty Language Understanding and Generation

**Paper:** AI for Peace Workshop @ ICLR 2026

## Overview

TreatyBench is the first systematic benchmark for evaluating LLM capabilities in understanding and generating international treaty language. Built on the UK AISI Inspect framework for reproducibility.

## Tasks

### Core Tasks (132 items)

| Task | Type | Items | Metric |
|------|------|-------|--------|
| **A1: Classification** | Understanding | 25 | Accuracy |
| **A2: Extraction** | Understanding | 22 | Semantic F1 |
| **A3: Ambiguity Detection** | Understanding | 20 | F1 |
| **A4: Cross-Reference Resolution** | Understanding | 15 | Accuracy |
| **B1: Completion** | Generation | 15 | LLM Judge (1-5) |
| **B2: Generation** | Generation | 20 | LLM Judge (1-5) |
| **B3: Adversarial Revision** | Generation | 15 | LLM Judge (1-5) |

### Hard Variants

| Task | Description |
|------|-------------|
| **A1-Hard** | Multi-domain provisions (trade + environment, etc.) |
| **A2-Hard** | Nested conditionals and complex obligations |
| **B2-Hard** | Novel scenarios (AI weapons treaties) |

## Domains

- Arms Control (NPT, CWC, CTBT)
- Trade (GATT, WTO, TRIPS)
- Environment (Paris Agreement, CBD, Montreal Protocol)
- Human Rights (UDHR, ICCPR, CAT)
- AI Governance (EU AI Act, UNESCO, OECD)

## Quick Start

```bash
# Install dependencies
pip install -e .

# Run a single task
python -m inspect_ai eval src/tasks/classification.py@treaty_classification --model anthropic/claude-sonnet-4-20250514

# Run all experiments on a model
python scripts/run_experiments.py --model opus

# Run all experiments on all models
python scripts/run_experiments.py --all

# List available tasks and models
python scripts/run_experiments.py --list
```

## Model Configurations

```bash
# Claude Opus 4.5
--model anthropic/claude-opus-4-5-20251101

# GPT-5.2
--model openai/gpt-5.2

# Gemini 3.0 Pro
--model google/gemini-3.0-pro

# Grok 4.1
--model grok/grok-4-1-fast-reasoning
```

## Results

| Model | A1 | A2 | A3 | A4 | B1 | B2 | B3 |
|-------|----|----|----|----|----|----|-----|
| Opus 4.5 | 92 (87) | **64** (65) | 93 | **93** | **5.0** | 4.95 | **4.73** |
| GPT-5.2 | **100** (67) | 48 (65) | 88 | 80 | **5.0** | **5.0** | 4.33 |
| Grok 4.1 | **100** (80) | 30 (**90**) | **98** | 77 | 4.73 | 4.85 | **4.73** |

*Understanding tasks (A1-A4): accuracy (%). Generation tasks (B1-B3): LLM judge score (1-5). Hard variants in parentheses.*

### Key Findings

1. **Extraction is the hardest understanding task** (30-64%), while classification approaches ceiling (92-100%)
2. **Hard variants reveal unexpected patterns**: Grok 4.1 improves 60 points on nested conditional extraction (A2-Hard)
3. **Generation approaches ceiling** (4.3-5.0/5), suggesting need for more challenging specifications

## Scoring Methods

- **Classification/Cross-Reference**: Standard accuracy
- **Extraction**: Semantic F1 using Sentence-BERT embeddings (cosine similarity threshold 0.7)
- **Ambiguity Detection**: F1 on identified terms + reasoning quality
- **Generation**: LLM-as-judge with explanation-first prompting

## Project Structure

```
treatybench/
├── src/
│   ├── tasks/
│   │   ├── classification.py      # A1: Domain classification
│   │   ├── extraction.py          # A2: Obligation extraction
│   │   ├── ambiguity.py           # A3: Ambiguity detection
│   │   ├── cross_reference.py     # A4: Cross-reference resolution
│   │   ├── generation.py          # B1, B2: Completion & generation
│   │   ├── revision.py            # B3: Adversarial revision
│   │   └── hard_variants.py       # A1H, A2H, B2H
│   ├── scorers/
│   │   ├── f1_scorer.py           # Extraction scoring
│   │   ├── semantic_scorer.py     # Sentence-BERT similarity
│   │   └── multi_judge.py         # Multi-model LLM judging
│   └── data/
│       ├── classification.jsonl
│       ├── extraction.jsonl
│       ├── completion.jsonl
│       ├── generation.jsonl
│       ├── ambiguity.jsonl
│       ├── cross_reference.jsonl
│       ├── revision.jsonl
│       └── *_hard.jsonl           # Hard variant data
├── scripts/
│   ├── run_experiments.py         # Experiment runner
│   └── analyze_experiments.py     # Results analysis
├── logs/                          # Inspect evaluation logs
├── pyproject.toml
└── README.md
```

## Citation

```bibtex
@inproceedings{treatybench2026,
  title={TreatyBench: Benchmarking LLM Capabilities in Treaty Language},
  author={Prasad, Amritanshu},
  booktitle={AI for Peace Workshop, ICLR 2026},
  year={2026}
}
```

## License

MIT
