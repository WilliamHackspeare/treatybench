"""
TreatyBench Benchmark Runner

Main entry point for evaluating LLMs on treaty language tasks.
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional
import re

sys.path.insert(0, str(Path(__file__).parent))
from llm_interface import call_llm, get_available_models

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"


def load_task_data(task_type: str) -> list[dict]:
    """Load task items from JSON file."""
    # Check both understanding and generation directories
    for subdir in ["understanding", "generation"]:
        task_file = DATA_DIR / "tasks" / subdir / f"{task_type}.json"
        if task_file.exists():
            with open(task_file, 'r', encoding='utf-8') as f:
                return json.load(f)

    raise FileNotFoundError(f"Task data not found for: {task_type}")


def load_schema() -> dict:
    """Load benchmark schema."""
    schema_file = DATA_DIR / "schema.json"
    with open(schema_file, 'r', encoding='utf-8') as f:
        return json.load(f)


# =============================================================================
# Task-Specific Prompts
# =============================================================================

def get_task_prompt(task_type: str, item: dict) -> str:
    """Generate prompt for a specific task item."""

    if task_type == "A1_classification":
        return f"""Classify the following treaty provision into exactly one of these categories:
- arms_control
- trade
- environment
- human_rights
- ai_governance

Respond with ONLY the category name, nothing else.

Treaty provision:
\"\"\"{item['input']}\"\"\"

Category:"""

    elif task_type == "A2_obligation":
        return f"""Extract all obligations, rights, and prohibitions from the following treaty text.

For each item, identify:
1. Type: "obligation", "right", or "prohibition"
2. Actor: Who bears this obligation/right
3. Action: What they must/may/must not do

Respond in JSON format as a list of objects.

Treaty text:
\"\"\"{item['input']}\"\"\"

JSON response:"""

    elif task_type == "B2_generation":
        return f"""You are drafting treaty language. Generate a formal treaty provision based on the following specification.

The provision should:
- Use formal treaty language (e.g., "shall", "undertakes to", "State Party")
- Be specific and implementable
- Address all requirements in the specification

Specification:
{item['input']}

Draft treaty provision:"""

    elif task_type == "B1_completion":
        return f"""Complete the following partial treaty provision. Fill in the [BLANK] with appropriate treaty language.

Partial provision:
{item['input']}

Complete provision:"""

    else:
        # Generic prompt
        return f"""Task: {task_type}

Input:
{item['input']}

Response:"""


# =============================================================================
# Evaluation Logic
# =============================================================================

def evaluate_classification(prediction: str, gold: str) -> dict:
    """Evaluate A1 classification task."""
    # Clean prediction
    pred_clean = prediction.strip().lower()
    gold_clean = gold.strip().lower()

    # Check for exact match or substring match
    correct = pred_clean == gold_clean or gold_clean in pred_clean

    return {
        "correct": correct,
        "prediction": pred_clean,
        "gold": gold_clean
    }


def evaluate_obligation_extraction(prediction: str, gold: list) -> dict:
    """Evaluate A2 obligation extraction task."""
    try:
        # Parse prediction JSON
        pred_list = json.loads(prediction)
        if not isinstance(pred_list, list):
            pred_list = [pred_list]
    except json.JSONDecodeError:
        # Try to extract JSON from response
        json_match = re.search(r'\[.*\]', prediction, re.DOTALL)
        if json_match:
            try:
                pred_list = json.loads(json_match.group())
            except:
                pred_list = []
        else:
            pred_list = []

    # Simple F1 based on count match (simplified)
    pred_count = len(pred_list)
    gold_count = len(gold)

    # Calculate overlap based on action keywords
    gold_actions = set()
    for g in gold:
        if isinstance(g, dict) and 'action' in g:
            gold_actions.add(g['action'].lower()[:30])  # First 30 chars

    pred_actions = set()
    for p in pred_list:
        if isinstance(p, dict) and 'action' in p:
            pred_actions.add(p['action'].lower()[:30])

    overlap = len(gold_actions & pred_actions)
    precision = overlap / max(len(pred_actions), 1)
    recall = overlap / max(len(gold_actions), 1)
    f1 = 2 * precision * recall / max(precision + recall, 0.001)

    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "pred_count": pred_count,
        "gold_count": gold_count
    }


def evaluate_generation_with_judge(
    task_type: str,
    specification: str,
    generated: str,
    judge_model: str = "claude-sonnet"
) -> dict:
    """Evaluate generation task using LLM judge."""

    judge_prompt = f"""You are evaluating the quality of a generated treaty provision.

SPECIFICATION (what was requested):
{specification}

GENERATED PROVISION:
{generated}

Rate the generated provision on a scale of 1-5:
1 = Incoherent or legally invalid
2 = Coherent but misses key requirements
3 = Adequate but generic/vague
4 = Good, addresses requirements with minor gaps
5 = Excellent, could serve as viable treaty language

Provide your rating as a single number (1-5) followed by a brief justification.

Format: [SCORE]: [JUSTIFICATION]

Rating:"""

    response = call_llm(
        judge_model,
        [{"role": "user", "content": judge_prompt}],
        temperature=0.3
    )

    # Parse score from response
    response_text = response["content"]
    score_match = re.search(r'(\d)[:/\s]', response_text)
    if score_match:
        score = int(score_match.group(1))
    else:
        # Try to find any digit 1-5
        for char in response_text:
            if char.isdigit() and 1 <= int(char) <= 5:
                score = int(char)
                break
        else:
            score = 3  # Default to middle

    return {
        "score": score,
        "justification": response_text,
        "judge_model": judge_model,
        "judge_usage": response["usage"]
    }


# =============================================================================
# Main Runner
# =============================================================================

def run_task(
    task_type: str,
    model: str,
    items: Optional[list] = None,
    sample_size: Optional[int] = None,
    judge_model: str = "claude-sonnet",
    verbose: bool = True
) -> dict:
    """Run a single task type on a model."""

    if items is None:
        items = load_task_data(task_type)

    if sample_size and sample_size < len(items):
        items = items[:sample_size]

    results = []
    total_usage = {"input_tokens": 0, "output_tokens": 0}

    schema = load_schema()
    task_info = schema["task_types"].get(task_type, {})

    if verbose:
        print(f"\n{'='*60}")
        print(f"Task: {task_type} | Model: {model} | Items: {len(items)}")
        print(f"{'='*60}")

    for i, item in enumerate(items):
        if verbose:
            print(f"\n--- Item {i+1}/{len(items)} ---")

        # Get prompt and call model
        prompt = get_task_prompt(task_type, item)

        response = call_llm(
            model,
            [{"role": "user", "content": prompt}],
            temperature=0.3  # Lower temperature for consistency
        )

        total_usage["input_tokens"] += response["usage"]["input_tokens"]
        total_usage["output_tokens"] += response["usage"]["output_tokens"]

        prediction = response["content"]

        if verbose:
            print(f"Input: {item['input'][:100]}...")
            print(f"Prediction: {prediction[:200]}...")

        # Evaluate based on task type
        if task_type == "A1_classification":
            eval_result = evaluate_classification(prediction, item["gold_answer"])
        elif task_type == "A2_obligation":
            eval_result = evaluate_obligation_extraction(prediction, item["gold_answer"])
        elif task_type.startswith("B"):
            # Generation task - use judge
            eval_result = evaluate_generation_with_judge(
                task_type, item["input"], prediction, judge_model
            )
            total_usage["input_tokens"] += eval_result.get("judge_usage", {}).get("input_tokens", 0)
            total_usage["output_tokens"] += eval_result.get("judge_usage", {}).get("output_tokens", 0)
        else:
            eval_result = {"raw_prediction": prediction}

        results.append({
            "item_id": item.get("task_id", f"item_{i}"),
            "input": item["input"],
            "prediction": prediction,
            "evaluation": eval_result
        })

        if verbose and task_type == "A1_classification":
            print(f"Correct: {eval_result['correct']}")

    # Compute aggregate metrics
    metrics = compute_metrics(task_type, results)

    if verbose:
        print(f"\n{'='*60}")
        print(f"RESULTS: {task_type}")
        print(f"{'='*60}")
        for k, v in metrics.items():
            print(f"{k}: {v:.3f}" if isinstance(v, float) else f"{k}: {v}")

    return {
        "task_type": task_type,
        "model": model,
        "judge_model": judge_model,
        "n_items": len(items),
        "results": results,
        "metrics": metrics,
        "usage": total_usage,
        "timestamp": datetime.now().isoformat()
    }


def compute_metrics(task_type: str, results: list) -> dict:
    """Compute aggregate metrics for a task."""

    if task_type == "A1_classification":
        correct = sum(1 for r in results if r["evaluation"].get("correct", False))
        return {
            "accuracy": correct / len(results) if results else 0,
            "correct": correct,
            "total": len(results)
        }

    elif task_type == "A2_obligation":
        f1_scores = [r["evaluation"].get("f1", 0) for r in results]
        return {
            "mean_f1": sum(f1_scores) / len(f1_scores) if f1_scores else 0,
            "total": len(results)
        }

    elif task_type.startswith("B"):
        scores = [r["evaluation"].get("score", 3) for r in results]
        return {
            "mean_score": sum(scores) / len(scores) if scores else 0,
            "score_distribution": {
                str(i): sum(1 for s in scores if s == i)
                for i in range(1, 6)
            },
            "total": len(results)
        }

    return {"total": len(results)}


def save_results(results: dict, task_type: str, model: str):
    """Save results to JSON file."""
    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{task_type}_{model}_{timestamp}.json"
    filepath = RESULTS_DIR / filename

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {filepath}")
    return filepath


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run TreatyBench evaluation")
    parser.add_argument("--task", required=True, help="Task type (e.g., A1_classification)")
    parser.add_argument("--model", default="claude-sonnet", help="Model to evaluate")
    parser.add_argument("--judge", default="claude-sonnet", help="Judge model for generation tasks")
    parser.add_argument("--sample", type=int, help="Number of items to sample")
    parser.add_argument("--quiet", action="store_true", help="Less verbose output")
    parser.add_argument("--no-save", action="store_true", help="Don't save results")

    args = parser.parse_args()

    results = run_task(
        task_type=args.task,
        model=args.model,
        sample_size=args.sample,
        judge_model=args.judge,
        verbose=not args.quiet
    )

    if not args.no_save:
        save_results(results, args.task, args.model)
