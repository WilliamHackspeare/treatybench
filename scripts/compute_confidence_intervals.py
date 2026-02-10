"""
Compute bootstrap confidence intervals and per-item difficulty analysis
for all TreatyBench results from existing Inspect eval logs.

Inspect .eval files are ZIP archives containing:
  - reductions.json: per-sample scores with numeric values
  - header.json: experiment metadata
  - samples/*.json: individual sample details
"""

import json
import zipfile
import random
import math
from pathlib import Path
from collections import defaultdict

N_BOOTSTRAP = 10000
CI_LEVEL = 0.95
random.seed(42)

BASE_DIR = Path(__file__).parent.parent
LOGS_DIR = BASE_DIR / "logs"

MODEL_DIRS = {
    "opus": "opus",
    "sonnet": "sonnet",
    "gpt5": "gpt5",
    "gemini25pro": "gemini25pro",
    "grok41": "grok41",
    "o3mini": "o3mini",
}

# Fallback directories for models with older log locations
ALT_DIRS = {
    "gemini25pro": ["gemini"],
}

TASK_KEYS = [
    "A1_classification", "A2_extraction", "A3_ambiguity", "A4_crossref",
    "B1_completion", "B2_generation", "B3_revision",
    "A1H_classification", "A2H_extraction", "B2H_generation",
]

# Tasks where value is a judge score (1-5) vs binary (0/1)
JUDGE_SCORE_TASKS = {"B3_revision"}
# Tasks where value is binary accuracy (multijudge: >=4 = CORRECT)
MULTIJUDGE_ACCURACY_TASKS = {"B1_completion", "B2_generation", "B2H_generation"}


def find_eval_log(model_key, task_key):
    """Find the most recent eval log for a model/task combination."""
    model_dir = MODEL_DIRS.get(model_key)
    dirs_to_check = []

    if model_dir:
        dirs_to_check.append(LOGS_DIR / model_dir / task_key)

    if model_key in ALT_DIRS:
        for alt in ALT_DIRS[model_key]:
            dirs_to_check.append(LOGS_DIR / alt / task_key)

    for d in dirs_to_check:
        if d.exists():
            evals = sorted(d.glob("*.eval"), key=lambda p: p.name, reverse=True)
            if evals:
                return evals[0]

    return None


def load_reductions(eval_path):
    """Load per-sample scores from an Inspect eval ZIP file's reductions.json."""
    try:
        with zipfile.ZipFile(eval_path, 'r') as z:
            if 'reductions.json' in z.namelist():
                data = json.loads(z.read('reductions.json'))
                if data and len(data) > 0:
                    return data[0].get("samples", [])
        return []
    except Exception as e:
        print(f"  Error loading {eval_path.name}: {e}")
        return []


def load_header(eval_path):
    """Load header metadata from eval ZIP."""
    try:
        with zipfile.ZipFile(eval_path, 'r') as z:
            if 'header.json' in z.namelist():
                return json.loads(z.read('header.json'))
        return {}
    except:
        return {}


def bootstrap_ci(values, n_bootstrap=N_BOOTSTRAP, ci_level=CI_LEVEL):
    """Compute bootstrap confidence interval for the mean."""
    if not values:
        return None, None, None
    n = len(values)
    observed_mean = sum(values) / n

    boot_means = []
    for _ in range(n_bootstrap):
        sample = [values[random.randint(0, n-1)] for _ in range(n)]
        boot_means.append(sum(sample) / n)

    boot_means.sort()
    alpha = (1 - ci_level) / 2
    lower_idx = max(0, int(alpha * n_bootstrap))
    upper_idx = min(n_bootstrap - 1, int((1 - alpha) * n_bootstrap) - 1)

    return observed_mean, boot_means[lower_idx], boot_means[upper_idx]


def wilson_ci(successes, total, z=1.96):
    """Wilson score interval for a proportion."""
    if total == 0:
        return 0, 0, 0
    p = successes / total
    denom = 1 + z**2 / total
    centre = (p + z**2 / (2*total)) / denom
    margin = z * math.sqrt((p*(1-p) + z**2/(4*total)) / total) / denom
    return p, max(0, centre - margin), min(1, centre + margin)


def main():
    all_results = {}
    item_difficulty = defaultdict(lambda: {"correct": 0, "total": 0, "models_correct": []})
    cost_data = {}

    print("=" * 70)
    print("TreatyBench: Bootstrap CIs & Per-Item Difficulty Analysis")
    print("=" * 70)

    for model_key in MODEL_DIRS:
        all_results[model_key] = {}
        cost_data[model_key] = {"total_tokens": 0, "total_time": 0}
        print(f"\n--- {model_key.upper()} ---")

        for task_key in TASK_KEYS:
            eval_path = find_eval_log(model_key, task_key)
            if not eval_path:
                print(f"  {task_key}: NO LOG FOUND")
                continue

            samples = load_reductions(eval_path)
            if not samples:
                print(f"  {task_key}: NO SAMPLES")
                continue

            # Extract numeric values
            values = []
            for s in samples:
                v = s.get("value")
                if v is not None:
                    try:
                        values.append(float(v))
                    except (TypeError, ValueError):
                        pass

            if not values:
                print(f"  {task_key}: NO NUMERIC VALUES")
                continue

            n = len(values)

            if task_key in JUDGE_SCORE_TASKS:
                # B3: mean judge score on 1-5 scale
                mean_val, ci_low, ci_high = bootstrap_ci(values)
                display = f"{mean_val:.2f} [{ci_low:.2f}, {ci_high:.2f}]"
                result_type = "judge_score"
            else:
                # All other tasks: accuracy (proportion of items scoring >= threshold)
                # For multijudge tasks, value=1.0 means CORRECT (score >= 4)
                # For understanding tasks, value=1.0 means exact match
                mean_val, ci_low, ci_high = bootstrap_ci(values)

                # Also compute Wilson CI (better for small proportions)
                successes = sum(1 for v in values if v >= 0.99)
                w_p, w_low, w_high = wilson_ci(successes, n)

                pct = mean_val * 100
                pct_low = ci_low * 100
                pct_high = ci_high * 100
                w_pct_low = w_low * 100
                w_pct_high = w_high * 100

                display = f"{pct:.0f}% [{pct_low:.0f}, {pct_high:.0f}] (Wilson: [{w_pct_low:.0f}, {w_pct_high:.0f}])"
                result_type = "accuracy"

            all_results[model_key][task_key] = {
                "type": result_type,
                "mean": mean_val,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "n": n,
                "values": values,
                "display": display,
            }

            print(f"  {task_key} (n={n}): {display}")

            # Track per-item difficulty
            for s in samples:
                sample_id = s.get("sample_id", "")
                if sample_id:
                    v = s.get("value", 0)
                    is_correct = (float(v) >= 0.99) if v is not None else False
                    item_difficulty[sample_id]["total"] += 1
                    if is_correct:
                        item_difficulty[sample_id]["correct"] += 1
                        item_difficulty[sample_id]["models_correct"].append(model_key)

            # Collect cost/timing data from header
            header = load_header(eval_path)
            stats = header.get("stats", {})
            if stats:
                cost_data[model_key]["total_tokens"] += stats.get("model_usage", {}).get("total_tokens", 0)
                cost_data[model_key]["total_time"] += stats.get("elapsed_time", 0)

    # Per-item difficulty analysis
    print("\n" + "=" * 70)
    print("PER-ITEM DIFFICULTY ANALYSIS")
    print("=" * 70)

    difficulty_data = {}
    floor_items = []
    ceiling_items = []
    discriminating_items = []

    for item_id, info in sorted(item_difficulty.items()):
        if info["total"] == 0:
            continue
        difficulty = info["correct"] / info["total"]
        difficulty_data[item_id] = {
            "correct": info["correct"],
            "total": info["total"],
            "difficulty": round(difficulty, 3),
            "models_correct": info["models_correct"],
        }
        if info["correct"] == info["total"]:
            floor_items.append(item_id)
        elif info["correct"] == 0:
            ceiling_items.append(item_id)
        else:
            discriminating_items.append(item_id)

    total_items = len(difficulty_data)
    print(f"\nTotal items with results: {total_items}")
    print(f"Floor items (all models correct): {len(floor_items)} ({len(floor_items)/max(total_items,1)*100:.0f}%)")
    print(f"Ceiling items (no model correct): {len(ceiling_items)} ({len(ceiling_items)/max(total_items,1)*100:.0f}%)")
    print(f"Discriminating items: {len(discriminating_items)} ({len(discriminating_items)/max(total_items,1)*100:.0f}%)")

    # Difficulty distribution by task prefix
    task_prefixes = defaultdict(list)
    for item_id, info in difficulty_data.items():
        prefix = item_id.split("_")[0]  # A1, A2, B1, etc.
        task_prefixes[prefix].append(info["difficulty"])

    print("\nDifficulty by task (mean % of models correct):")
    for prefix in sorted(task_prefixes.keys()):
        vals = task_prefixes[prefix]
        mean_d = sum(vals) / len(vals) * 100
        print(f"  {prefix}: {mean_d:.0f}% mean difficulty, {len(vals)} items")

    # Hardest items
    print("\nTop 15 hardest items:")
    sorted_items = sorted(difficulty_data.items(), key=lambda x: x[1]["difficulty"])
    for item_id, info in sorted_items[:15]:
        print(f"  {item_id}: {info['correct']}/{info['total']} ({info['difficulty']*100:.0f}%) correct by: {', '.join(info['models_correct']) or 'none'}")

    # Floor items
    print(f"\nFloor items ({len(floor_items)} items, all models correct):")
    for item_id in sorted(floor_items)[:15]:
        print(f"  {item_id}")

    # Save all outputs
    output = {
        "confidence_intervals": {
            model: {
                task: {k: v for k, v in data.items() if k != "values"}
                for task, data in tasks.items()
            }
            for model, tasks in all_results.items()
        },
        "difficulty_analysis": {
            "total_items": total_items,
            "floor_items": len(floor_items),
            "ceiling_items": len(ceiling_items),
            "discriminating_items": len(discriminating_items),
            "floor_item_ids": sorted(floor_items),
            "ceiling_item_ids": sorted(ceiling_items),
            "by_task_prefix": {
                prefix: {
                    "mean_difficulty": round(sum(vals)/len(vals), 3),
                    "n_items": len(vals),
                }
                for prefix, vals in task_prefixes.items()
            },
            "hardest_15": [{"id": k, **v} for k, v in sorted_items[:15]],
        },
        "per_item": difficulty_data,
        "cost_data": cost_data,
    }

    output_path = BASE_DIR / "scripts" / "ci_analysis_results.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
