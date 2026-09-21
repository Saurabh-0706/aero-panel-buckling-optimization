"""
compare_optimizers.py

Comparison plot contrasting the two optimizers against each other and
against the raw DOE data -- same role as the earlier COMSOL crush-tube
project's optimizer-comparison plot. Deliberately designed to still
produce a complete, useful plot with ONLY the classical optimizer's
result present (optimizers/llm_optimizer.py is optional and may never
have been run) -- it does not error out or block on the LLM result being
absent, it just leaves that series off and says so.

Usage:
    python optimizers/compare_optimizers.py
"""

import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ls_dyna_model"))

from deck_builder import load_config  # noqa: E402


def load_doe_points(csv_path):
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    return {
        "N_cr_per_mm": [float(r["N_cr_per_mm"]) for r in rows],
        "mass_kg": [float(r["mass_kg"]) for r in rows],
    }


def load_optimizer_result(path):
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def main():
    config = load_config()
    target = config["targets"]["design_running_load_N_per_mm"]

    doe_path = ROOT / "results" / "doe_results_panel.csv"
    doe = load_doe_points(doe_path)

    classical_path = ROOT / "optimizers" / "classical_optimizer_result.json"
    classical = load_optimizer_result(classical_path)
    if classical is None:
        print(f"No classical optimizer result at {classical_path} -- run "
              f"`python optimizers/classical_optimizer.py` first.")
        sys.exit(1)

    llm_path = ROOT / "optimizers" / "llm_optimizer_result.json"
    llm = load_optimizer_result(llm_path)
    if llm is None:
        print(f"No LLM optimizer result at {llm_path} -- plotting the classical "
              f"optimizer's result alone (this is expected if you haven't set up an "
              f"ANTHROPIC_API_KEY / run optimizers/llm_optimizer.py yet; see that "
              f"script's module docstring for setup instructions).")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # --- Left panel: design space scatter (mass vs. buckling capacity) ---
    ax = axes[0]
    ax.scatter(doe["N_cr_per_mm"], doe["mass_kg"], s=18, alpha=0.5,
               color="#7f8c8d", label=f"DOE points (n={len(doe['N_cr_per_mm'])})")
    ax.axvline(target, color="#c0392b", linestyle="--", linewidth=1.3,
               label=f"target = {target:.0f} N/mm")

    cb = classical["best"]
    ax.scatter([cb["surrogate_predicted_N_cr_per_mm"]], [cb["mass_kg"]],
               marker="*", s=380, color="#2980b9", edgecolor="black", linewidth=0.8,
               zorder=5, label=f"Classical (DE) optimum: {cb['mass_kg']:.3f} kg")

    if llm is not None:
        lb = llm["best"]
        ax.scatter([lb["predicted_N_cr_per_mm"]], [lb["mass_kg"]],
                   marker="D", s=170, color="#27ae60", edgecolor="black", linewidth=0.8,
                   zorder=5, label=f"LLM optimum: {lb['mass_kg']:.3f} kg")

    ax.set_xlabel("Surrogate-predicted N_cr (N/mm)")
    ax.set_ylabel("Panel mass (kg)")
    ax.set_title("Design space: mass vs. buckling capacity")
    ax.legend(loc="upper left", fontsize=8.5)
    ax.grid(alpha=0.25)

    # --- Right panel: bar comparison of best masses found ---
    ax2 = axes[1]
    labels, masses, colors = [], [], []
    labels.append("Best DOE\npoint meeting\ntarget")
    feasible_doe_masses = [m for m, n in zip(doe["mass_kg"], doe["N_cr_per_mm"]) if n >= target]
    masses.append(min(feasible_doe_masses) if feasible_doe_masses else float("nan"))
    colors.append("#7f8c8d")

    labels.append("Classical\noptimizer\n(DE)")
    masses.append(cb["mass_kg"])
    colors.append("#2980b9")

    if llm is not None:
        labels.append(f"LLM\noptimizer\n({llm['model']})")
        masses.append(lb["mass_kg"])
        colors.append("#27ae60")

    bars = ax2.bar(labels, masses, color=colors, edgecolor="black", linewidth=0.7)
    for bar, m in zip(bars, masses):
        if m == m:  # not NaN
            ax2.text(bar.get_x() + bar.get_width() / 2, m + 0.01, f"{m:.3f}",
                      ha="center", va="bottom", fontsize=9)
    ax2.set_ylabel("Panel mass (kg)")
    ax2.set_title("Best feasible mass found, by method")
    ax2.grid(alpha=0.25, axis="y")

    fig.suptitle(
        "Stiffened-panel buckling optimization: classical DOE vs. metaheuristic vs. LLM search",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out_path = ROOT / "results" / "optimizer_comparison.png"
    fig.savefig(out_path, dpi=160)
    print(f"Saved comparison plot to {out_path}")

    # Text summary
    print("\nSummary:")
    if feasible_doe_masses:
        print(f"  Best raw DOE point meeting target:  {min(feasible_doe_masses):.4f} kg")
    else:
        print(f"  No raw DOE point meets the target directly (expected -- DOE is a "
              f"random sample, not an optimization).")
    print(f"  Classical (DE) optimizer:           {cb['mass_kg']:.4f} kg")
    if llm is not None:
        print(f"  LLM optimizer:                      {lb['mass_kg']:.4f} kg")
        diff_pct = abs(lb["mass_kg"] - cb["mass_kg"]) / min(lb["mass_kg"], cb["mass_kg"]) * 100.0
        if lb["mass_kg"] < cb["mass_kg"]:
            print(f"  -> LLM found the lighter design: {diff_pct:.1f}% lighter than "
                  f"the classical (DE) result.")
        else:
            print(f"  -> Classical (DE) found the lighter design: the LLM's result is "
                  f"{diff_pct:.1f}% heavier.")
    else:
        print(f"  LLM optimizer:                      not run (optional, see "
              f"optimizers/llm_optimizer.py)")


if __name__ == "__main__":
    main()
