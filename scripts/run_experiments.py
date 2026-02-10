"""
TreatyBench Experiment Runner

Runs all tasks on 7 frontier models using Inspect AI framework:
- Claude Opus 4.5 (anthropic/claude-opus-4-5-20251101)
- Claude Sonnet 4 (anthropic/claude-sonnet-4-20250514)
- GPT-4o (openai/gpt-4o)
- GPT-5.2 (openai/gpt-5.2)
- o3-mini (openai/o3-mini)
- Gemini 2.5 Pro (google/gemini-2.5-pro)
- Grok 4.1 (xai/grok-4.1)

Usage:
    python scripts/run_experiments.py --all           # Run everything
    python scripts/run_experiments.py --model opus   # Run single model
    python scripts/run_experiments.py --task A1      # Run single task
"""

import subprocess
import sys
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Optional

# Load .env and set up API key aliases
from dotenv import load_dotenv
load_dotenv()

# Map GROK_API_KEY to XAI_API_KEY if needed
if os.getenv("GROK_API_KEY") and not os.getenv("XAI_API_KEY"):
    os.environ["XAI_API_KEY"] = os.getenv("GROK_API_KEY")

# Model configurations
MODELS = {
    "opus": "anthropic/claude-opus-4-5-20251101",
    "sonnet": "anthropic/claude-sonnet-4-20250514",
    "gpt4o": "openai/gpt-4o",
    "gpt5": "openai/gpt-5.2",
    "o3mini": "openai/o3-mini",
    "gemini25pro": "google/gemini-2.5-pro",
    "grok41": "grok/grok-4-1-fast-reasoning",  # Grok 4.1 with reasoning
}

# Task configurations (task_name: file_path@task_function)
# Using file paths since module import has issues
TASKS = {
    # Core tasks
    "A1_classification": "src/tasks/classification.py@treaty_classification",
    "A2_extraction": "src/tasks/extraction.py@obligation_extraction",
    "B1_completion": "src/tasks/generation.py@treaty_completion_multijudge",
    "B2_generation": "src/tasks/generation.py@treaty_generation_multijudge",
    # New tasks
    "A3_ambiguity": "src/tasks/ambiguity.py@ambiguity_detection",
    "A4_crossref": "src/tasks/cross_reference.py@cross_reference_resolution",
    "B3_revision": "src/tasks/revision.py@adversarial_revision",
    # Hard variants
    "A1H_classification": "src/tasks/hard_variants.py@classification_hard",
    "A2H_extraction": "src/tasks/hard_variants.py@extraction_hard",
    "B2H_generation": "src/tasks/hard_variants.py@generation_hard_multijudge",
}

BASE_DIR = Path(__file__).parent.parent
LOGS_DIR = BASE_DIR / "logs"


def run_task(task_name: str, task_path: str, model_id: str, model_name: str) -> dict:
    """Run a single task on a single model using Inspect AI.

    Returns dict with status and timing info.
    """
    print(f"\n{'='*60}", flush=True)
    print(f"Running: {task_name} on {model_name}", flush=True)
    print(f"Model ID: {model_id}", flush=True)
    print(f"{'='*60}", flush=True)

    start_time = datetime.now()

    # Construct inspect eval command
    cmd = [
        sys.executable, "-m", "inspect_ai", "eval",
        task_path,
        "--model", model_id,
        "--log-dir", str(LOGS_DIR / model_name / task_name),
    ]

    try:
        result = subprocess.run(
            cmd,
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout per task
        )

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        success = result.returncode == 0

        if not success:
            print(f"ERROR: Task failed with return code {result.returncode}")
            print(f"STDERR: {result.stderr[:500]}")

        return {
            "task": task_name,
            "model": model_name,
            "model_id": model_id,
            "success": success,
            "duration_seconds": duration,
            "returncode": result.returncode,
            "stderr": result.stderr[:500] if not success else "",
            "timestamp": start_time.isoformat(),
        }

    except subprocess.TimeoutExpired:
        return {
            "task": task_name,
            "model": model_name,
            "model_id": model_id,
            "success": False,
            "duration_seconds": 3600,
            "error": "timeout",
            "timestamp": start_time.isoformat(),
        }
    except Exception as e:
        return {
            "task": task_name,
            "model": model_name,
            "model_id": model_id,
            "success": False,
            "error": str(e),
            "timestamp": start_time.isoformat(),
        }


def run_model(model_name: str, tasks: Optional[list[str]] = None) -> list[dict]:
    """Run all tasks for a single model."""
    if model_name not in MODELS:
        print(f"Unknown model: {model_name}")
        print(f"Available: {list(MODELS.keys())}")
        return []

    model_id = MODELS[model_name]
    tasks_to_run = tasks or list(TASKS.keys())

    results = []
    for task_name in tasks_to_run:
        if task_name not in TASKS:
            print(f"Unknown task: {task_name}, skipping")
            continue

        task_path = TASKS[task_name]
        result = run_task(task_name, task_path, model_id, model_name)
        results.append(result)

        # Save intermediate results
        save_results(results, f"partial_{model_name}")

    return results


def run_all(tasks: Optional[list[str]] = None) -> dict:
    """Run all tasks on all models."""
    all_results = {}

    for model_name in MODELS.keys():
        print(f"\n{'#'*60}")
        print(f"# MODEL: {model_name.upper()}")
        print(f"{'#'*60}")

        results = run_model(model_name, tasks)
        all_results[model_name] = results

    return all_results


def save_results(results: dict | list, name: str):
    """Save results to JSON file."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = LOGS_DIR / f"experiment_{name}_{timestamp}.json"

    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {filepath}")
    return filepath


def summarize_results(results: dict):
    """Print summary of experiment results."""
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)

    for model_name, model_results in results.items():
        print(f"\n{model_name.upper()}:")
        success_count = sum(1 for r in model_results if r.get("success", False))
        total = len(model_results)
        print(f"  Tasks completed: {success_count}/{total}")

        for r in model_results:
            status = "✓" if r.get("success") else "✗"
            duration = r.get("duration_seconds", 0)
            print(f"  {status} {r['task']}: {duration:.1f}s")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run TreatyBench experiments")
    parser.add_argument("--all", action="store_true", help="Run all tasks on all models")
    parser.add_argument("--model", choices=list(MODELS.keys()), help="Run specific model")
    parser.add_argument("--task", help="Run specific task(s), comma-separated")
    parser.add_argument("--list", action="store_true", help="List available tasks and models")

    args = parser.parse_args()

    if args.list:
        print("Available models:")
        for name, model_id in MODELS.items():
            print(f"  {name}: {model_id}")
        print("\nAvailable tasks:")
        for name, path in TASKS.items():
            print(f"  {name}: {path}")
        sys.exit(0)

    # Parse task list if provided
    tasks = None
    if args.task:
        tasks = [t.strip() for t in args.task.split(",")]

    if args.all:
        results = run_all(tasks)
        save_results(results, "full")
        summarize_results(results)
    elif args.model:
        results = run_model(args.model, tasks)
        save_results(results, args.model)
    else:
        parser.print_help()
        print("\nExamples:")
        print("  python scripts/run_experiments.py --all")
        print("  python scripts/run_experiments.py --model opus")
        print("  python scripts/run_experiments.py --model gpt5 --task A1_classification,A2_extraction")
