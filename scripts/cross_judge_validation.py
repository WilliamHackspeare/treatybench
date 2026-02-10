"""
Cross-Judge Validation: Re-score all generation task outputs using GPT-5.2
as a second judge (in addition to the primary Opus 4.5 judge).

Extracts model answers from existing eval logs, scores them with GPT-5.2,
and computes correlation with original Opus 4.5 scores.
"""

import json
import zipfile
import asyncio
import os
import sys
from pathlib import Path
from statistics import mean, stdev
from collections import defaultdict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from inspect_ai.model import get_model, GenerateConfig

BASE_DIR = Path(__file__).parent.parent
LOGS_DIR = BASE_DIR / "logs"

# Judge models
PRIMARY_JUDGE = "anthropic/claude-opus-4-5-20251101"
CROSS_JUDGE = "openai/gpt-5.2"

# Models and generation tasks to cross-validate
MODELS_TO_CHECK = ["opus", "sonnet", "gpt5", "gemini25pro", "grok41", "o3mini"]
GEN_TASKS = ["B1_completion", "B2_generation", "B3_revision"]

JUDGE_TEMPLATE = """You are evaluating the quality of a generated treaty provision.

TASK SPECIFICATION:
{question}

GENERATED RESPONSE:
{answer}

REFERENCE (for context, not exact match required):
{target}

First, analyze the response by considering:
1. Legal Soundness: Is the language legally valid and enforceable?
2. Specificity: Are terms and obligations clearly defined?
3. Implementability: Could this be practically implemented?
4. Completeness: Does it address all aspects of the specification?
5. Treaty Conventions: Does it follow standard treaty drafting conventions?

After your analysis, provide a final rating on a scale of 1-5:
- 5: Excellent - Could serve as viable treaty language
- 4: Good - Addresses requirements with minor gaps
- 3: Adequate - Meets basic requirements but lacks specificity
- 2: Poor - Misses key requirements or contains significant issues
- 1: Failing - Incoherent, legally invalid, or misses the specification

Format your response as:
ANALYSIS: [Your detailed analysis]
GRADE: [1-5]
"""

import re

def find_eval_log(model_key, task_key):
    """Find eval log for a model/task."""
    dirs_to_check = [
        LOGS_DIR / model_key / task_key,
        LOGS_DIR / "gemini25pro" / task_key if model_key == "gemini25pro" else None,
        LOGS_DIR / "gemini" / task_key if model_key == "gemini25pro" else None,
    ]
    for d in [d for d in dirs_to_check if d]:
        if d.exists():
            evals = sorted(d.glob("*.eval"), key=lambda p: p.name, reverse=True)
            if evals:
                return evals[0]
    return None


def extract_samples_from_eval(eval_path):
    """Extract sample questions, answers, targets, and original scores from eval ZIP."""
    samples = []
    try:
        with zipfile.ZipFile(eval_path, 'r') as z:
            # Get reductions for original scores
            reductions = {}
            if 'reductions.json' in z.namelist():
                red_data = json.loads(z.read('reductions.json'))
                if red_data:
                    for s in red_data[0].get("samples", []):
                        reductions[s.get("sample_id", "")] = s.get("value")

            # Get individual samples for question/answer/target
            sample_files = [n for n in z.namelist() if n.startswith("samples/")]
            for sf in sample_files:
                data = json.loads(z.read(sf))
                sample_id = data.get("id", "")

                # Extract input text
                question = ""
                if isinstance(data.get("input"), str):
                    question = data["input"]
                elif isinstance(data.get("input"), list):
                    for msg in data["input"]:
                        if msg.get("role") == "user":
                            question = msg.get("content", "")
                            break

                # Extract model answer
                answer = ""
                output = data.get("output", {})
                if isinstance(output, dict):
                    choices = output.get("choices", [])
                    if choices:
                        content = choices[0].get("message", {}).get("content", "")
                        # Handle content that may be a list of content blocks
                        if isinstance(content, list):
                            text_parts = []
                            for block in content:
                                if isinstance(block, dict) and block.get("type") == "text":
                                    text_parts.append(block.get("text", ""))
                                elif isinstance(block, str):
                                    text_parts.append(block)
                            answer = "\n".join(text_parts)
                        elif isinstance(content, str):
                            answer = content

                # Extract target
                target = data.get("target", "")
                if isinstance(target, list):
                    target = target[0] if target else ""

                original_score = reductions.get(sample_id)

                if question and answer:
                    samples.append({
                        "id": sample_id,
                        "question": question,
                        "answer": answer,
                        "target": target,
                        "original_score": original_score,
                    })
    except Exception as e:
        print(f"  Error extracting from {eval_path.name}: {e}")

    return samples


async def judge_single(model, question, answer, target):
    """Get a score from a judge model."""
    prompt = JUDGE_TEMPLATE.format(question=question, answer=answer, target=target)
    try:
        response = await model.generate(
            prompt,
            config=GenerateConfig(temperature=0.0),
        )
        text = response.completion
        match = re.search(r"GRADE:\s*([1-5])", text)
        if match:
            return float(match.group(1))
    except Exception as e:
        print(f"    Judge error: {e}")
    return None


async def main():
    print("=" * 70)
    print("Cross-Judge Validation: GPT-5.2 vs Opus 4.5")
    print("=" * 70)

    cross_judge_model = get_model(CROSS_JUDGE)

    all_results = {}
    all_pairs = []  # (original_score, cross_score) pairs for correlation

    for model_key in MODELS_TO_CHECK:
        all_results[model_key] = {}
        print(f"\n--- {model_key.upper()} ---")

        for task_key in GEN_TASKS:
            eval_path = find_eval_log(model_key, task_key)
            if not eval_path:
                print(f"  {task_key}: NO LOG FOUND")
                continue

            samples = extract_samples_from_eval(eval_path)
            if not samples:
                print(f"  {task_key}: NO SAMPLES EXTRACTED")
                continue

            print(f"  {task_key}: Scoring {len(samples)} items with GPT-5.2...")

            # Score each sample with GPT-5.2
            cross_scores = []
            original_scores = []
            for i, s in enumerate(samples):
                score = await judge_single(
                    cross_judge_model,
                    s["question"],
                    s["answer"],
                    s["target"],
                )
                if score is not None:
                    cross_scores.append(score)
                    orig = s["original_score"]
                    if orig is not None:
                        original_scores.append(float(orig))
                        all_pairs.append((float(orig), score))
                    print(f"    {s['id']}: original={orig}, cross={score}")

            if cross_scores:
                cross_mean = mean(cross_scores)
                orig_mean = mean(original_scores) if original_scores else None

                # Compute agreement (within 1 point)
                agreement = 0
                if original_scores and len(original_scores) == len(cross_scores):
                    agreement = sum(1 for o, c in zip(original_scores, cross_scores) if abs(o - c) <= 1) / len(cross_scores)

                all_results[model_key][task_key] = {
                    "n": len(cross_scores),
                    "cross_judge_mean": round(cross_mean, 2),
                    "original_judge_mean": round(orig_mean, 2) if orig_mean else None,
                    "agreement_within_1": round(agreement, 2),
                    "cross_scores": cross_scores,
                    "original_scores": original_scores,
                }

                orig_str = f"{orig_mean:.2f}" if orig_mean else "N/A"
                print(f"    Mean: original={orig_str}, cross={cross_mean:.2f}, agreement={agreement:.0%}")

    # Compute overall correlation
    if all_pairs:
        n = len(all_pairs)
        orig_vals = [p[0] for p in all_pairs]
        cross_vals = [p[1] for p in all_pairs]

        # Pearson correlation
        mean_o = mean(orig_vals)
        mean_c = mean(cross_vals)
        cov = sum((o - mean_o) * (c - mean_c) for o, c in all_pairs) / n
        std_o = (sum((o - mean_o)**2 for o in orig_vals) / n) ** 0.5
        std_c = (sum((c - mean_c)**2 for c in cross_vals) / n) ** 0.5

        if std_o > 0 and std_c > 0:
            pearson_r = cov / (std_o * std_c)
        else:
            pearson_r = float('nan')

        overall_agreement = sum(1 for o, c in all_pairs if abs(o - c) <= 1) / n

        print(f"\n{'=' * 70}")
        print(f"OVERALL CROSS-JUDGE STATISTICS")
        print(f"{'=' * 70}")
        print(f"Total scored pairs: {n}")
        print(f"Pearson r: {pearson_r:.3f}")
        print(f"Agreement (within 1 point): {overall_agreement:.0%}")
        print(f"Mean original (Opus 4.5): {mean_o:.2f}")
        print(f"Mean cross (GPT-5.2): {mean_c:.2f}")

        all_results["_overall"] = {
            "n_pairs": n,
            "pearson_r": round(pearson_r, 3),
            "agreement_within_1": round(overall_agreement, 2),
            "mean_original": round(mean_o, 2),
            "mean_cross": round(mean_c, 2),
        }

    # Save results
    output_path = BASE_DIR / "scripts" / "cross_judge_results.json"
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
