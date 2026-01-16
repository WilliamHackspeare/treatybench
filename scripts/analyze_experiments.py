"""
TreatyBench Results Analysis

Analyzes experiment results from Inspect AI logs and generates
tables/figures for the paper.

Usage:
    python scripts/analyze_experiments.py --logs logs/
    python scripts/analyze_experiments.py --latex   # Generate LaTeX tables
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import Optional
import statistics

LOGS_DIR = Path(__file__).parent.parent / "logs"


def load_inspect_results(log_dir: Path) -> dict:
    """Load results from Inspect AI log directory.

    Inspect AI saves results as eval.json in the log directory.
    """
    results = {}

    # Look for eval.json files in subdirectories
    for model_dir in log_dir.iterdir():
        if not model_dir.is_dir():
            continue

        model_name = model_dir.name
        results[model_name] = {}

        for task_dir in model_dir.iterdir():
            if not task_dir.is_dir():
                continue

            task_name = task_dir.name

            # Find the most recent eval.json
            eval_files = list(task_dir.glob("**/eval.json"))
            if not eval_files:
                continue

            eval_file = max(eval_files, key=lambda p: p.stat().st_mtime)

            try:
                with open(eval_file) as f:
                    eval_data = json.load(f)
                    results[model_name][task_name] = eval_data
            except Exception as e:
                print(f"Error loading {eval_file}: {e}")

    return results


def compute_task_metrics(results: dict) -> dict:
    """Compute aggregate metrics per task per model."""
    metrics = {}

    for model_name, model_results in results.items():
        metrics[model_name] = {}

        for task_name, eval_data in model_results.items():
            # Extract metrics from eval data
            scores = eval_data.get("results", {}).get("scores", [])

            if not scores:
                continue

            # Aggregate depending on task type
            if "classification" in task_name.lower():
                # Accuracy for classification
                correct = sum(1 for s in scores if s.get("value") == "C")
                total = len(scores)
                metrics[model_name][task_name] = {
                    "metric": "accuracy",
                    "value": correct / total if total > 0 else 0,
                    "n": total,
                }

            elif "extraction" in task_name.lower():
                # F1 for extraction
                f1_scores = [
                    s.get("metadata", {}).get("f1", 0)
                    for s in scores
                    if s.get("metadata", {}).get("f1") is not None
                ]
                metrics[model_name][task_name] = {
                    "metric": "mean_f1",
                    "value": statistics.mean(f1_scores) if f1_scores else 0,
                    "n": len(f1_scores),
                }

            elif any(x in task_name.lower() for x in ["generation", "completion", "revision"]):
                # Mean score for generation tasks
                grade_scores = []
                for s in scores:
                    # Try different locations for score
                    score = s.get("metadata", {}).get("mean_score")
                    if score is None:
                        # Single judge score
                        match = s.get("value")
                        if match and match not in ["C", "I", "P"]:
                            try:
                                score = float(match)
                            except:
                                pass
                    if score is not None:
                        grade_scores.append(score)

                metrics[model_name][task_name] = {
                    "metric": "mean_score",
                    "value": statistics.mean(grade_scores) if grade_scores else 0,
                    "n": len(grade_scores),
                }

            elif "ambiguity" in task_name.lower() or "crossref" in task_name.lower():
                # F1 for structured tasks
                f1_scores = [
                    s.get("metadata", {}).get("f1", 0)
                    for s in scores
                    if s.get("metadata", {}).get("f1") is not None
                ]
                metrics[model_name][task_name] = {
                    "metric": "mean_f1",
                    "value": statistics.mean(f1_scores) if f1_scores else 0,
                    "n": len(f1_scores),
                }

    return metrics


def generate_main_table(metrics: dict) -> str:
    """Generate main results table in LaTeX."""

    # Order tasks
    task_order = [
        "A1_classification",
        "A2_extraction",
        "A3_ambiguity",
        "A4_crossref",
        "B1_completion",
        "B2_generation",
        "B3_revision",
    ]

    model_order = ["opus", "gpt5", "gemini", "grok"]

    # Header
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Main Results: Model Performance Across Tasks}",
        r"\label{tab:main-results}",
        r"\small",
        r"\begin{tabular}{l" + "c" * len(model_order) + "}",
        r"\toprule",
        "Task & " + " & ".join([m.upper() for m in model_order]) + r" \\",
        r"\midrule",
    ]

    for task in task_order:
        row = [task.replace("_", " ")]
        for model in model_order:
            if model in metrics and task in metrics[model]:
                val = metrics[model][task]["value"]
                metric_type = metrics[model][task]["metric"]
                if metric_type == "accuracy":
                    row.append(f"{val*100:.1f}\\%")
                elif metric_type == "mean_f1":
                    row.append(f"{val:.2f}")
                else:
                    row.append(f"{val:.2f}")
            else:
                row.append("--")
        lines.append(" & ".join(row) + r" \\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines)


def generate_hard_table(metrics: dict) -> str:
    """Generate hard variants comparison table."""

    pairs = [
        ("A1_classification", "A1H_classification"),
        ("A2_extraction", "A2H_extraction"),
        ("B2_generation", "B2H_generation"),
    ]

    model_order = ["opus", "gpt5", "gemini", "grok"]

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Standard vs. Hard Task Variants}",
        r"\label{tab:hard-comparison}",
        r"\small",
        r"\begin{tabular}{ll" + "c" * len(model_order) + "}",
        r"\toprule",
        "Task & Variant & " + " & ".join([m.upper() for m in model_order]) + r" \\",
        r"\midrule",
    ]

    for std, hard in pairs:
        task_name = std.split("_")[0]
        for variant, label in [(std, "Standard"), (hard, "Hard")]:
            row = [task_name if label == "Standard" else "", label]
            for model in model_order:
                if model in metrics and variant in metrics[model]:
                    val = metrics[model][variant]["value"]
                    metric_type = metrics[model][variant]["metric"]
                    if metric_type == "accuracy":
                        row.append(f"{val*100:.1f}\\%")
                    else:
                        row.append(f"{val:.2f}")
                else:
                    row.append("--")
            lines.append(" & ".join(row) + r" \\")
        lines.append(r"\midrule")

    # Remove last midrule
    lines[-1] = r"\bottomrule"

    lines.extend([
        r"\end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines)


def print_summary(metrics: dict):
    """Print human-readable summary."""
    print("\n" + "="*70)
    print("EXPERIMENT RESULTS SUMMARY")
    print("="*70)

    for model_name, model_metrics in sorted(metrics.items()):
        print(f"\n{model_name.upper()}")
        print("-" * 40)
        for task_name, task_data in sorted(model_metrics.items()):
            metric_type = task_data["metric"]
            val = task_data["value"]
            n = task_data["n"]
            if metric_type == "accuracy":
                print(f"  {task_name}: {val*100:.1f}% (n={n})")
            else:
                print(f"  {task_name}: {val:.3f} (n={n})")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze TreatyBench experiment results")
    parser.add_argument("--logs", default=str(LOGS_DIR), help="Log directory")
    parser.add_argument("--latex", action="store_true", help="Generate LaTeX tables")
    parser.add_argument("--output", help="Output file for tables")

    args = parser.parse_args()

    log_dir = Path(args.logs)
    if not log_dir.exists():
        print(f"Log directory not found: {log_dir}")
        sys.exit(1)

    # Load results
    results = load_inspect_results(log_dir)

    if not results:
        print("No results found in log directory")
        sys.exit(1)

    # Compute metrics
    metrics = compute_task_metrics(results)

    # Print summary
    print_summary(metrics)

    # Generate LaTeX if requested
    if args.latex:
        print("\n" + "="*70)
        print("LATEX TABLES")
        print("="*70)

        main_table = generate_main_table(metrics)
        hard_table = generate_hard_table(metrics)

        print("\n% Main Results Table")
        print(main_table)

        print("\n% Hard Variants Comparison")
        print(hard_table)

        if args.output:
            with open(args.output, 'w') as f:
                f.write("% Auto-generated by analyze_experiments.py\n\n")
                f.write("% Main Results Table\n")
                f.write(main_table)
                f.write("\n\n% Hard Variants Comparison\n")
                f.write(hard_table)
            print(f"\nTables saved to: {args.output}")
