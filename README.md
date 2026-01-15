# TreatyBench

Benchmarking LLM Capabilities in Treaty Language Understanding and Generation

**Paper:** AI for Peace Workshop @ ICLR 2026

## Overview

TreatyBench is the first systematic benchmark for evaluating LLM capabilities in understanding and generating international treaty language. Built on the UK AISI Inspect framework for reproducibility.

## Tasks

| Task | Type | Items | Metric |
|------|------|-------|--------|
| **A1: Classification** | Understanding | 25 | Accuracy |
| **A2: Extraction** | Understanding | 22 | F1 |
| **B1: Completion** | Generation | 15 | LLM Judge (1-5) |
| **B2: Generation** | Generation | 20 | LLM Judge (1-5) |

## Domains

- Arms Control (NPT, CWC, CTBT)
- Trade (GATT, WTO, TRIPS)
- Environment (Paris Agreement, CBD, Montreal Protocol)
- Human Rights (UDHR, ICCPR, CAT)
- AI Governance (EU AI Act, UNESCO, OECD)

## Quick Start

```bash
# Install dependencies
pip install inspect-ai

# Run classification task
python -m inspect_ai eval src/tasks/classification.py --model anthropic/claude-sonnet-4-20250514

# Run all tasks
python -m inspect_ai eval src/tasks/classification.py src/tasks/extraction.py src/tasks/generation.py --model anthropic/claude-sonnet-4-20250514

# Limit samples for quick test
python -m inspect_ai eval src/tasks/classification.py --model anthropic/claude-sonnet-4-20250514 --limit 5
```

## Model Configurations

### Standard Models
```bash
# Claude Sonnet 4
--model anthropic/claude-sonnet-4-20250514

# Claude Opus 4.5
--model anthropic/claude-opus-4-5-20251101

# GPT-5.2
--model openai/gpt-5.2

# Gemini 3.0 Pro
--model google/gemini-3.0-pro

# Grok 4.1
--model xai/grok-4.1
```

### Thinking/Reasoning Variants
```bash
# Claude with extended thinking
--model anthropic/claude-opus-4-5-20251101 -T thinking=enabled

# OpenAI o-series
--model openai/o3

# Gemini with reasoning
--model google/gemini-3.0-pro-reasoning
```

## Results

| Model | A1: Classification | A2: Extraction | B1: Completion | B2: Generation |
|-------|-------------------|----------------|----------------|----------------|
| Claude Opus 4.5 | 92% | 50% | 4.73/5 | 4.80/5 |
| Claude Sonnet 4 | 96% | 50% | 4.80/5 | 4.20/5 |
| GPT-4o | **100%** | **59%** | 4.87/5 | 4.80/5 |
| o3-mini | 92% | 32% | **5.00/5** | **5.00/5** |

### Key Findings

1. **Classification is near-ceiling** (92-100%), while extraction is harder (32-59%)
2. **Reasoning model paradox**: o3-mini achieves worst extraction but perfect generation
3. **Extended thinking provides marginal benefit** for generation tasks

## Project Structure

```
treatybench/
├── src/
│   ├── tasks/
│   │   ├── classification.py    # A1: Domain classification
│   │   ├── extraction.py        # A2: Obligation extraction
│   │   ├── generation.py        # B1, B2: Generation tasks
│   │   └── generation_thinking.py  # B1, B2 with extended thinking
│   └── data/
│       ├── classification.jsonl
│       ├── extraction.jsonl
│       ├── completion.jsonl
│       └── generation.jsonl
├── scripts/
│   └── analyze_results.py       # Results analysis script
├── paper/
│   └── treatybench.tex          # Paper draft
├── logs/                        # Inspect evaluation logs
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
