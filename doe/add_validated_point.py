"""
add_validated_point.py

Adds ONE already-solved point to the DOE dataset: specifically, the real
LS-DYNA result from optimizers/validate_optimum.py (which re-solves an
optimizer's winning design for real, outside the normal DOE sweep). That
point is genuine ground-truth data -- solved by the same LS-DYNA pipeline,
just triggered from a different script -- so it belongs in the training
set precisely because it sits exactly where the surrogate was least
certain (the design-space edge an optimizer converged to), which is the
most valuable place to have real data.

This does NOT call LS-DYNA again. It reads
optimizers/optimum_validation_result.json (already on disk from a prior
validate_optimum.py run), recomputes lambda_1 from the real N_cr_per_mm
via the same reference_load_Pref_N() function the rest of the pipeline
uses (so it stays internally consistent), and appends one row to both
results/doe_points_panel.csv and results/doe_results_panel.csv --
refusing if the design already exists in either file, same
duplicate-guard spirit as everywhere else in this project.

Usage:
    python optimizers/validate_optimum.py     # produces the input this needs
    python doe/add_validated_point.py
"""

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ls_dyna_model"))

from classical_buckling import reference_load_Pref_N, load_config  # noqa: E402

POINTS_FIELDNAMES = ["point_id", "t_skin", "n_str", "h_str", "t_str"]
RESULTS_FIELDNAMES = ["point_id", "t_skin", "n_str", "h_str", "t_str",
                       "mass_kg", "lambda_1", "N_cr_per_mm", "meets_target"]


def main():
    validation_path = ROOT / "optimizers" / "optimum_validation_result.json"
    if not validation_path.exists():
        print(f"No {validation_path} -- run optimizers/validate_optimum.py first.")
        sys.exit(1)
    with open(validation_path) as f:
        val = json.load(f)

    design = val["design"]
    t_skin, n_str = design["t_skin"], int(design["n_str"])
    h_str, t_str = design["h_str"], design["t_str"]
    real_N_cr = val["real_lsdyna_N_cr_per_mm"]
    mass_kg = val["mass_kg"]

    points_path = ROOT / "results" / "doe_points_panel.csv"
    results_path = ROOT / "results" / "doe_results_panel.csv"
    if not points_path.exists() or not results_path.exists():
        print(f"Expected both {points_path} and {results_path} to already exist.")
        sys.exit(1)

    with open(points_path) as f:
        points_rows = list(csv.DictReader(f))
    with open(results_path) as f:
        results_rows = list(csv.DictReader(f))

    def same_design(row):
        return (abs(float(row["t_skin"]) - t_skin) < 1e-6
                and int(row["n_str"]) == n_str
                and abs(float(row["h_str"]) - h_str) < 1e-6
                and abs(float(row["t_str"]) - t_str) < 1e-6)

    if any(same_design(r) for r in points_rows):
        print("This exact design is already in doe_points_panel.csv -- nothing to add "
              "(if you meant to re-validate, that's fine, just not something this script "
              "needs to append again).")
        sys.exit(0)

    next_id = max(int(r["point_id"]) for r in points_rows) + 1 if points_rows else 0

    config = load_config()
    Wb = config["fixed_parameters"]["panel_width_Wb_mm"]
    Pref = reference_load_Pref_N(t_skin, n_str, config)
    lambda_1 = (real_N_cr * Wb) / Pref
    target = config["targets"]["design_running_load_N_per_mm"]
    meets = real_N_cr >= target

    with open(points_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=POINTS_FIELDNAMES)
        writer.writerow({
            "point_id": next_id, "t_skin": t_skin, "n_str": n_str,
            "h_str": h_str, "t_str": t_str,
        })

    with open(results_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULTS_FIELDNAMES)
        writer.writerow({
            "point_id": next_id, "t_skin": t_skin, "n_str": n_str,
            "h_str": h_str, "t_str": t_str,
            "mass_kg": f"{mass_kg:.5f}",
            "lambda_1": f"{lambda_1:.6f}",
            "N_cr_per_mm": f"{real_N_cr:.4f}",
            "meets_target": meets,
        })

    print(f"Added point_id {next_id} (the validated optimum) to both "
          f"doe_points_panel.csv and doe_results_panel.csv.")
    print(f"  t_skin={t_skin:.4f} n_str={n_str} h_str={h_str:.4f} t_str={t_str:.4f}")
    print(f"  N_cr_per_mm={real_N_cr:.4f} (real LS-DYNA), lambda_1={lambda_1:.6f}, "
          f"mass={mass_kg:.4f}kg, meets_target={meets}")
    print(f"\nNext: retrain the surrogate on the now-{len(results_rows)+1}-point "
          f"dataset with `python surrogate/train_surrogate.py`, then re-run "
          f"`python optimizers/classical_optimizer.py`.")


if __name__ == "__main__":
    main()
