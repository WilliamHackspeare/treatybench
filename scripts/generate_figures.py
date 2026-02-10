"""
Generate figures for TreatyBench paper.

Creates:
1. Performance radar plot (Figure 1) - comparing all models across tasks
2. Results heatmap (Figure 2) - task x model performance matrix
3. Difficulty analysis (Figure 3) - standard vs hard variant comparison
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import matplotlib.patches as mpatches

# Output directory
PAPER_DIR = Path(__file__).parent.parent / "paper" / "figures"
PAPER_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# Data: All results from experiments
# ============================================================================

# Order: Opus 4.5, Sonnet 4, GPT-4o, GPT-5.2, Gemini 2.5 Pro, Grok 4.1, o3-mini
MODELS = ["Opus 4.5", "Sonnet 4", "GPT-4o", "GPT-5.2", "Gemini 2.5", "Grok 4.1", "o3-mini"]
MODEL_SHORT = ["Opus 4.5", "Sonnet 4", "GPT-4o", "GPT-5.2", "Gemini 2.5", "Grok 4.1", "o3-mini"]

# Understanding tasks (accuracy %)
UNDERSTANDING = {
    "A1\nClassification": [92, 96, 100, 100, 100, 100, 92],
    "A2\nExtraction":     [64, 59, 59, 48, 41, 30, 32],
    "A3\nAmbiguity":      [92, 95, 88, 88, 98, 98, 75],
    "A4\nCross-Ref":      [93, 90, 77, 80, 87, 77, 67],
}

# Generation tasks: B1/B2 as multijudge accuracy (0-100), B3 as judge score (1-5)
GENERATION = {
    "B1\nCompletion": [100, 100, 87, 100, 100, 93, 63],  # multijudge accuracy %
    "B2\nGeneration":  [95, 95, 98, 100, 100, 100, 60],   # multijudge accuracy %
    "B3\nRevision":    [4.73, 4.07, 4.47, 4.33, 4.93, 4.73, 2.40],  # judge score 1-5
}

# Hard variants
HARD = {
    "A1-Hard": [87, 80, 93, 67, 73, 80, 60],
    "A2-Hard": [65, 80, 85, 65, 85, 90, 35],
    "B2-Hard": [100, 100, 73, 100, 100, 100, 80],
}

STANDARD_FOR_HARD = {
    "A1": [92, 96, 100, 100, 100, 100, 92],
    "A2": [64, 59, 59, 48, 41, 30, 32],
    "B2": [95, 95, 98, 100, 100, 100, 60],  # multijudge accuracy %
}

# Colors for models
COLORS = {
    "Opus 4.5": "#6B4C9A",     # Purple (Anthropic)
    "Sonnet 4": "#9B7FCB",     # Light purple
    "GPT-4o": "#2E7D32",       # Green (OpenAI)
    "GPT-5.2": "#4CAF50",      # Light green
    "Gemini 2.5": "#E65100",   # Orange (Google)
    "Grok 4.1": "#1565C0",     # Blue (xAI)
    "o3-mini": "#81C784",      # Pale green (OpenAI reasoning)
}


def fig1_radar_plot():
    """Performance radar plot comparing all models across all 7 tasks."""

    # Combine all tasks, normalizing generation to 0-100
    task_names = list(UNDERSTANDING.keys()) + list(GENERATION.keys())

    # Build data matrix
    n_tasks = len(task_names)
    n_models = len(MODELS)

    data = np.zeros((n_models, n_tasks))
    for i, task in enumerate(UNDERSTANDING.keys()):
        for j in range(n_models):
            data[j, i] = UNDERSTANDING[task][j]

    for i, task in enumerate(GENERATION.keys()):
        for j in range(n_models):
            val = GENERATION[task][j]
            if task == "B3\nRevision":
                data[j, i + len(UNDERSTANDING)] = val * 20  # scale 1-5 to 20-100
            else:
                data[j, i + len(UNDERSTANDING)] = val  # B1/B2 already 0-100 %

    # Radar plot
    angles = np.linspace(0, 2 * np.pi, n_tasks, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for j, model in enumerate(MODELS):
        values = data[j].tolist()
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=1.8, markersize=5,
                label=model, color=COLORS[model], alpha=0.85)
        ax.fill(angles, values, alpha=0.05, color=COLORS[model])

    # Clean task labels
    task_labels = [t.replace("\n", " ") for t in task_names]
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(task_labels, fontsize=10)

    # Set radial limits
    ax.set_ylim(0, 105)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20", "40", "60", "80", "100"], fontsize=8, color="grey")

    # Add note about generation scaling
    ax.set_title("Model Performance Across All Tasks\n(Generation scores scaled: 1-5 $\\rightarrow$ 20-100)",
                 fontsize=13, fontweight='bold', pad=25)

    ax.legend(loc='upper right', bbox_to_anchor=(1.4, 1.1), fontsize=9, frameon=True)

    plt.tight_layout()
    plt.savefig(PAPER_DIR / "radar_plot.pdf", bbox_inches='tight', dpi=300)
    plt.savefig(PAPER_DIR / "radar_plot.png", bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved: {PAPER_DIR / 'radar_plot.pdf'}")


def fig2_heatmap():
    """Results heatmap: task x model performance matrix."""

    # All tasks with raw values
    tasks_u = ["A1: Classification", "A2: Extraction", "A3: Ambiguity", "A4: Cross-Ref"]
    tasks_g = ["B1: Completion", "B2: Generation", "B3: Revision"]
    tasks_h = ["A1-Hard", "A2-Hard", "B2-Hard"]

    all_tasks = tasks_u + tasks_g + tasks_h

    # Build data matrix (all normalized to 0-100)
    # Order: Opus 4.5, Sonnet 4, GPT-4o, GPT-5.2, Gemini 2.5, Grok 4.1, o3-mini
    data = np.array([
        # Understanding
        [92, 96, 100, 100, 100, 100, 92],    # A1
        [64, 59, 59, 48, 41, 30, 32],        # A2
        [92, 95, 88, 88, 98, 98, 75],        # A3
        [93, 90, 77, 80, 87, 77, 67],        # A4
        # Generation (B1/B2 as multijudge accuracy %, B3 scaled from 1-5)
        [100, 100, 87, 100, 100, 93, 63],    # B1 multijudge acc
        [95, 95, 98, 100, 100, 100, 60],     # B2 multijudge acc
        [94.6, 81.4, 89.4, 86.6, 98.6, 94.6, 48.0],  # B3 scaled
        # Hard variants
        [87, 80, 93, 67, 73, 80, 60],        # A1H
        [65, 80, 85, 65, 85, 90, 35],        # A2H
        [100, 100, 73, 100, 100, 100, 80],   # B2H
    ])

    fig, ax = plt.subplots(figsize=(10, 7))

    # Custom colormap: red (low) -> white (mid) -> green (high)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("rg", ["#FFCDD2", "#FFFFFF", "#C8E6C9", "#2E7D32"], N=256)

    im = ax.imshow(data, cmap=cmap, aspect='auto', vmin=25, vmax=100)

    # Axes
    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels(MODEL_SHORT, fontsize=11, rotation=30, ha='right')
    ax.set_yticks(range(len(all_tasks)))
    ax.set_yticklabels(all_tasks, fontsize=11)

    # Add text annotations
    for i in range(len(all_tasks)):
        for j in range(len(MODELS)):
            val = data[i, j]
            # For generation tasks, show original 1-5 score
            if i in [4, 5]:
                # B1/B2: show multijudge accuracy %
                text = f"{val:.0f}"
            elif i == 6:
                # B3: show original 1-5 score
                gen_b3 = [4.73, 4.07, 4.47, 4.33, 4.93, 4.73, 2.40]
                text = f"{gen_b3[j]:.2f}"
            else:
                text = f"{val:.0f}"

            text_color = "white" if val < 45 or val > 92 else "black"
            ax.text(j, i, text, ha="center", va="center", fontsize=9.5,
                    color=text_color, fontweight='bold')

    # Divider lines between sections
    ax.axhline(y=3.5, color='black', linewidth=2)
    ax.axhline(y=6.5, color='black', linewidth=2)

    # Section labels
    ax.text(-1.8, 1.5, "Understanding", fontsize=11, fontweight='bold',
            rotation=90, va='center', ha='center')
    ax.text(-1.8, 5.0, "Generation", fontsize=11, fontweight='bold',
            rotation=90, va='center', ha='center')
    ax.text(-1.8, 8.0, "Hard", fontsize=11, fontweight='bold',
            rotation=90, va='center', ha='center')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Performance (%)", fontsize=11)

    ax.set_title("TreatyBench Performance Heatmap", fontsize=14, fontweight='bold', pad=15)

    plt.tight_layout()
    plt.savefig(PAPER_DIR / "heatmap.pdf", bbox_inches='tight', dpi=300)
    plt.savefig(PAPER_DIR / "heatmap.png", bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved: {PAPER_DIR / 'heatmap.pdf'}")


def fig3_difficulty_analysis():
    """Standard vs Hard variant comparison (grouped bar chart)."""

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)

    x = np.arange(len(MODELS))
    width = 0.35

    tasks = [
        ("Classification", "A1", "A1-Hard"),
        ("Extraction", "A2", "A2-Hard"),
        ("Generation", "B2", "B2-Hard"),
    ]

    standard_data = {
        "A1": [92, 96, 100, 100, 100, 100, 92],
        "A2": [64, 59, 59, 48, 41, 30, 32],
        "B2": [95, 95, 98, 100, 100, 100, 60],  # multijudge accuracy %
    }

    hard_data = {
        "A1-Hard": [87, 80, 93, 67, 73, 80, 60],
        "A2-Hard": [65, 80, 85, 65, 85, 90, 35],
        "B2-Hard": [100, 100, 73, 100, 100, 100, 80],
    }

    for idx, (title, std_key, hard_key) in enumerate(tasks):
        ax = axes[idx]
        std_vals = standard_data[std_key]
        hard_vals = hard_data[hard_key]

        bars1 = ax.bar(x - width/2, std_vals, width, label='Standard',
                       color='#42A5F5', edgecolor='white', linewidth=0.5)
        bars2 = ax.bar(x + width/2, hard_vals, width, label='Hard',
                       color='#EF5350', edgecolor='white', linewidth=0.5)

        # Add delta annotations
        for i in range(len(MODELS)):
            delta = hard_vals[i] - std_vals[i]
            sign = "+" if delta > 0 else ""
            y_pos = max(std_vals[i], hard_vals[i]) + 2
            color = '#2E7D32' if delta > 0 else '#C62828'
            ax.text(x[i], y_pos, f"{sign}{delta}", ha='center', va='bottom',
                    fontsize=8, color=color, fontweight='bold')

        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(MODEL_SHORT, fontsize=9, rotation=45, ha='right')
        ax.set_ylim(0, 115)
        ax.set_ylabel("Accuracy (%)" if idx == 0 else "")
        ax.grid(axis='y', alpha=0.3)

        if idx == 0:
            ax.legend(fontsize=10, loc='lower left')

    fig.suptitle("Standard vs. Hard Variant Performance", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(PAPER_DIR / "difficulty_analysis.pdf", bbox_inches='tight', dpi=300)
    plt.savefig(PAPER_DIR / "difficulty_analysis.png", bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved: {PAPER_DIR / 'difficulty_analysis.pdf'}")


def fig4_task_variance():
    """Box plot showing score variance across models for each task."""

    all_data = {
        "A1": [92, 96, 100, 100, 100, 100, 92],
        "A2": [64, 59, 59, 48, 41, 30, 32],
        "A3": [92, 95, 88, 88, 98, 98, 75],
        "A4": [93, 90, 77, 80, 87, 77, 67],
        "B1": [100, 100, 87, 100, 100, 93, 63],
        "B2": [95, 95, 98, 100, 100, 100, 60],
        "B3": [94.6, 81.4, 89.4, 86.6, 98.6, 94.6, 48.0],
    }

    fig, ax = plt.subplots(figsize=(10, 5))

    tasks = list(all_data.keys())
    data_list = [all_data[t] for t in tasks]

    bp = ax.boxplot(data_list, labels=tasks, patch_artist=True, widths=0.5,
                    medianprops=dict(color='black', linewidth=2))

    understanding_color = '#BBDEFB'
    generation_color = '#C8E6C9'

    for i, patch in enumerate(bp['boxes']):
        if i < 4:
            patch.set_facecolor(understanding_color)
        else:
            patch.set_facecolor(generation_color)

    # Overlay individual model points
    for i, task in enumerate(tasks):
        vals = all_data[task]
        jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(vals))
        for j, (v, jit) in enumerate(zip(vals, jitter)):
            ax.plot(i + 1 + jit, v, 'o', color=COLORS[MODELS[j]],
                    markersize=7, alpha=0.8, zorder=5)

    # Legend for model colors
    legend_elements = [mpatches.Patch(facecolor=understanding_color, label='Understanding'),
                       mpatches.Patch(facecolor=generation_color, label='Generation')]
    for m in MODELS:
        legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                               markerfacecolor=COLORS[m], markersize=8, label=m))

    ax.legend(handles=legend_elements, loc='lower left', fontsize=9, ncol=2)

    ax.set_ylabel("Performance (%)", fontsize=12)
    ax.set_title("Score Distribution Across Models by Task", fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(25, 110)

    plt.tight_layout()
    plt.savefig(PAPER_DIR / "task_variance.pdf", bbox_inches='tight', dpi=300)
    plt.savefig(PAPER_DIR / "task_variance.png", bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved: {PAPER_DIR / 'task_variance.pdf'}")


if __name__ == "__main__":
    print("Generating TreatyBench figures...")
    fig1_radar_plot()
    fig2_heatmap()
    fig3_difficulty_analysis()
    fig4_task_variance()
    print(f"\nAll figures saved to: {PAPER_DIR}")
