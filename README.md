# TreatyBench

Benchmarking LLM Capabilities in Treaty Language Understanding and Generation

**Paper:** Targeting NeurIPS 2026 Datasets and Benchmarks Track

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

# Claude Sonnet 4
--model anthropic/claude-sonnet-4-20250514

# GPT-4o
--model openai/gpt-4o

# GPT-5.2
--model openai/gpt-5.2

# o3-mini
--model openai/o3-mini

# Gemini 2.5 Pro
--model google/gemini-2.5-pro

# Grok 4.1
--model grok/grok-4-1-fast-reasoning
```

## Results

### Main Results

| Model | A1 | A2 | A3 | A4 | B1 | B2 | B3 |
|-------|----|----|----|----|----|----|-----|
| Claude Opus 4.5 | 92 | **64** | 92 | **93** | 100 | 95 | **4.73** |
| Claude Sonnet 4 | 96 | 59 | **95** | 90 | 100 | 95 | 4.07 |
| GPT-4o | **100** | 59 | 88 | 77 | 87 | 98 | 4.47 |
| GPT-5.2 | **100** | 48 | 88 | 80 | **100** | **100** | 4.33 |
| Gemini 2.5 Pro | **100** | 41 | **98** | 87 | **100** | **100** | **4.93** |
| Grok 4.1 | **100** | 30 | **98** | 77 | 93 | **100** | **4.73** |
| o3-mini | 92 | 32 | 75 | 67 | 63 | 60 | 2.40 |

*Understanding tasks (A1-A4): accuracy (%). Generation tasks: B1/B2 = multijudge accuracy (% items rated >= 4/5), B3 = mean judge score (1-5).*

### Hard Variants

| Model | A1-H | A2-H | B2-H |
|-------|------|------|------|
| Claude Opus 4.5 | 87 | 65 | 100 |
| Claude Sonnet 4 | 80 | 80 | 100 |
| GPT-4o | **93** | **85** | 73 |
| GPT-5.2 | 67 | 65 | 100 |
| Gemini 2.5 Pro | 73 | **85** | 100 |
| Grok 4.1 | 80 | **90** | 100 |
| o3-mini | 60 | 35 | 80 |

### Key Findings

1. **Striking capability gap**: Classification approaches ceiling (92-100%) while extraction remains challenging (30-64%)
2. **Gemini 2.5 Pro leads on generation**: Highest B3 score (4.93) and perfect B1/B2 multijudge accuracy, despite moderate extraction (41%)
3. **Reasoning model paradox**: o3-mini achieves lowest extraction (32%) and dramatically lower multijudge generation scores (63% B1, 60% B2), revealing self-enhancement bias in single-judge evaluation
4. **Hard variants reveal unexpected patterns**: Extraction improves on hard variants; Grok 4.1 jumps from 30% to 90% on A2-Hard
5. **Adversarial revision is most discriminating**: Scores range from 2.40 (o3-mini) to 4.93 (Gemini 2.5 Pro)
6. **Domain effects**: Arms control provisions are hardest to classify; AI governance easiest

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
│   ├── analyze_experiments.py     # Results analysis
│   └── generate_figures.py        # Paper figure generation
├── paper/
│   ├── treatybench_full.tex       # Full NeurIPS-format paper
│   ├── treatybench.tex            # Original workshop paper
│   ├── figures/                   # Generated figures (radar, heatmap, etc.)
│   └── neurips_2025.sty           # NeurIPS style file
├── logs/                          # Inspect evaluation logs
├── pyproject.toml
└── README.md
```

## Citation

```bibtex
@inproceedings{treatybench2026,
  title={TreatyBench: A Comprehensive Benchmark for Evaluating LLM Capabilities in International Treaty Language Understanding and Generation},
  author={Prasad, Amritanshu},
  booktitle={NeurIPS Datasets and Benchmarks Track},
  year={2026}
}
```

## License

MIT
