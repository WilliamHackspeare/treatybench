#!/usr/bin/env python3
"""
TreatyBench Results Analysis Script

Parses Inspect log files and generates summary tables.
Uses inspect_ai.log API to read binary .eval format.
"""

import json
from pathlib import Path
from collections import defaultdict

# Use Inspect's log reading API for .eval files
from inspect_ai.log import list_eval_logs, read_eval_log

LOGS_DIR = Path(__file__).parent.parent / "logs"


def load_eval_results(log_info) -> dict:
    """Load results from an eval log file using Inspect API."""
    try:
        # Read just the header (metadata + results, no samples)
        log = read_eval_log(log_info, header_only=True)

        # Extract key metrics
        model = log.eval.model
        task = log.eval.task

        # Get scorer results
        accuracy = None
        if log.results and log.results.scores:
            main_score = log.results.scores[0]
            if hasattr(main_score, 'metrics') and 'accuracy' in main_score.metrics:
                accuracy = main_score.metrics['accuracy'].value

        return {
            "model": model,
            "task": task,
            "accuracy": accuracy,
            "samples": log.eval.dataset.samples if log.eval.dataset else 0,
            "timestamp": str(log.eval.created) if log.eval.created else "",
        }
    except Exception as e:
        print(f"Error loading {log_info}: {e}")
        return None


def collect_all_results() -> list[dict]:
    """Collect results from all log files using Inspect API."""
    results = []

    if not LOGS_DIR.exists():
        print(f"Logs directory not found: {LOGS_DIR}")
        return results

    # Use Inspect's list_eval_logs to get all logs
    try:
        log_list = list_eval_logs(str(LOGS_DIR))
        for log_info in log_list:
            result = load_eval_results(log_info)
            if result:
                results.append(result)
    except Exception as e:
        print(f"Error listing logs: {e}")

    return results


def format_model_name(model: str) -> str:
    """Format model name for display."""
    mappings = {
        "anthropic/claude-opus-4-5-20251101": "Claude Opus 4.5",
        "anthropic/claude-sonnet-4-20250514": "Claude Sonnet 4",
        "openai/gpt-4o": "GPT-4o",
        "openai/o3-mini": "o3-mini",
        "openai/o3": "o3",
        "google/gemini-2.0-flash": "Gemini 2.0 Flash",
        "google/gemini-3.0-pro": "Gemini 3.0 Pro",
    }
    return mappings.get(model, model)


def format_task_name(task: str) -> str:
    """Format task name for display."""
    mappings = {
        "treaty_classification": "A1: Classification",
        "obligation_extraction": "A2: Extraction",
        "treaty_completion": "B1: Completion",
        "treaty_generation": "B2: Generation",
        "treaty_completion_thinking": "B1: Completion (Think)",
        "treaty_generation_thinking": "B2: Generation (Think)",
    }
    return mappings.get(task, task)


def get_latest_results(results: list[dict]) -> list[dict]:
    """Filter to keep only the most recent result for each model/task pair."""
    # Group by model and task, keeping the most recent by timestamp
    latest = {}
    for r in results:
        key = (r["model"], r["task"])
        if key not in latest or r["timestamp"] > latest[key]["timestamp"]:
            latest[key] = r
    return list(latest.values())


def generate_results_table(results: list[dict]) -> str:
    """Generate a markdown results table."""
    # Filter to latest results only
    results = get_latest_results(results)

    # Group by model and task
    by_model = defaultdict(dict)

    for r in results:
        model = format_model_name(r["model"])
        task = format_task_name(r["task"])
        by_model[model][task] = r["accuracy"]

    # Define task order
    task_order = [
        "A1: Classification",
        "A2: Extraction",
        "B1: Completion",
        "B2: Generation",
        "B1: Completion (Think)",
        "B2: Generation (Think)",
    ]

    # Build table
    lines = []
    lines.append("| Model | " + " | ".join(task_order) + " |")
    lines.append("|" + "------|" * (len(task_order) + 1))

    for model in sorted(by_model.keys()):
        row = [model]
        for task in task_order:
            score = by_model[model].get(task)
            if score is not None:
                if task.startswith("A1") or task.startswith("A2"):
                    row.append(f"{score:.0%}" if score <= 1 else f"{score:.1%}")
                else:
                    row.append(f"{score:.2f}")
            else:
                row.append("-")
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def generate_latex_table(results: list[dict]) -> str:
    """Generate a LaTeX results table."""
    # Filter to latest results only
    results = get_latest_results(results)

    # Group by model and task
    by_model = defaultdict(dict)

    for r in results:
        model = format_model_name(r["model"])
        task = r["task"]
        by_model[model][task] = r["accuracy"]

    lines = [
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{TreatyBench Results}",
        "\\label{tab:results}",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Model & Classification & Extraction & Completion & Generation \\\\",
        "\\midrule",
    ]

    for model in sorted(by_model.keys()):
        scores = by_model[model]
        cls = scores.get("treaty_classification", "-")
        ext = scores.get("obligation_extraction", "-")
        comp = scores.get("treaty_completion", "-")
        gen = scores.get("treaty_generation", "-")

        cls_str = f"{cls:.0%}" if isinstance(cls, float) else "-"
        ext_str = f"{ext:.0%}" if isinstance(ext, float) else "-"
        comp_str = f"{comp:.2f}" if isinstance(comp, float) else "-"
        gen_str = f"{gen:.2f}" if isinstance(gen, float) else "-"

        lines.append(f"{model} & {cls_str} & {ext_str} & {comp_str} & {gen_str} \\\\")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])

    return "\n".join(lines)


def main():
    print("TreatyBench Results Analysis")
    print("=" * 50)

    results = collect_all_results()
    print(f"Found {len(results)} evaluation logs\n")

    if not results:
        print("No results found. Run evaluations first.")
        return

    # Filter to latest results for display
    latest_results = get_latest_results(results)
    print(f"Using {len(latest_results)} latest results (filtered from {len(results)} total logs)\n")

    # Print summary by model
    print("Results by Model:")
    print("-" * 50)

    by_model = defaultdict(list)
    for r in latest_results:
        by_model[r["model"]].append(r)

    for model, model_results in sorted(by_model.items()):
        print(f"\n{format_model_name(model)}:")
        for r in sorted(model_results, key=lambda x: x["task"]):
            task = format_task_name(r["task"])
            score = r["accuracy"]
            if score is not None:
                if "Classification" in task or "Extraction" in task:
                    print(f"  {task}: {score:.1%}")
                else:
                    print(f"  {task}: {score:.2f}/5")

    print("\n" + "=" * 50)
    print("\nMarkdown Table:")
    print("-" * 50)
    print(generate_results_table(results))

    print("\n" + "=" * 50)
    print("\nLaTeX Table:")
    print("-" * 50)
    print(generate_latex_table(results))

    # Save results summary (latest only)
    output_file = Path(__file__).parent.parent / "results_summary.json"
    with open(output_file, 'w') as f:
        json.dump(latest_results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
