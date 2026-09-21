"""
validate_optimum.py

Closes the loop on the whole pipeline: the classical optimizer's design
(optimizers/classical_optimizer_result.json) only exists inside the
surrogate model so far -- differential evolution never touches LS-DYNA
directly, it only ever queries the trained Gaussian Process. This script
takes that exact design point and runs it through a REAL LS-DYNA solve,
then reports a three-way comparison: surrogate prediction vs. real solve
vs. the independent classical hand-calc.

Why this matters, honestly: a surrogate can be an excellent fit on its
training data (this one cross-validates at R^2=0.987) and still be wrong
at a specific point it extrapolated to reach -- especially right at the
edge of the design space, which is exactly where an optimizer's result
tends to land (n_str=6 is the DOE's upper bound). This is the check that
would catch that, rather than taking the optimizer's word for it.

Usage:
    python optimizers/validate_optimum.py
    python optimizers/validate_optimum.py --keep-work-dir
"""

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ls_dyna_model"))
sys.path.insert(0, str(ROOT / "doe"))

from deck_builder import load_config  # noqa: E402
from classical_buckling import subpanel_running_load_N_per_mm  # noqa: E402
from run_doe_lsdyna import solve_one_point_lsdyna  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--keep-work-dir", action="store_true")
    parser.add_argument(
        "--result-file",
        default=str(ROOT / "optimizers" / "classical_optimizer_result.json"),
        help="Which optimizer result to validate (defaults to the classical "
             "optimizer's). Point optimizers/llm_optimizer_result.json here to "
             "validate the LLM optimizer's design instead.",
    )
    args = parser.parse_args()

    result_path = Path(args.result_file)
    if not result_path.exists():
        print(f"No result file at {result_path} -- run the optimizer first.")
        sys.exit(1)
    with open(result_path) as f:
        result = json.load(f)
    best = result["best"]
    t_skin, n_str = best["t_skin"], best["n_str"]
    h_str, t_str = best["h_str"], best["t_str"]

    # Field name differs slightly between the two optimizers' result JSON
    # (classical_optimizer.py writes surrogate_predicted_N_cr_per_mm,
    # llm_optimizer.py writes predicted_N_cr_per_mm) -- handle both.
    surrogate_pred = best.get("surrogate_predicted_N_cr_per_mm", best.get("predicted_N_cr_per_mm"))

    config = load_config()
    target = config["targets"]["design_running_load_N_per_mm"]

    print(f"Validating optimum from {result_path.name}:")
    print(f"  t_skin={t_skin:.4f}mm  n_str={n_str}  h_str={h_str:.4f}mm  t_str={t_str:.4f}mm")
    print(f"  mass = {best['mass_kg']:.4f} kg\n")
    print("Running a real LS-DYNA solve on this exact design point "
          "(seconds to a couple minutes for a linear eigenvalue extraction)...\n")

    work_root = ROOT / "results" / "lsdyna_work_validate_optimum"
    work_root.mkdir(parents=True, exist_ok=True)

    sol = solve_one_point_lsdyna("optimum_validation", t_skin, n_str, h_str, t_str,
                                  config, work_root)
    real_N_cr = sol["N_cr_per_mm"]

    hand_calc_N_cr = subpanel_running_load_N_per_mm(t_skin, n_str, config)

    print("--- Three-way comparison ---")
    print(f"  Surrogate predicted:  {surrogate_pred:7.1f} N/mm")
    print(f"  REAL LS-DYNA solve:   {real_N_cr:7.1f} N/mm   <-- ground truth")
    print(f"  Classical hand-calc:  {hand_calc_N_cr:7.1f} N/mm   (local sub-panel only, sanity floor)")
    print(f"  Design target:        {target:7.1f} N/mm\n")

    surrogate_err_pct = 100 * (surrogate_pred - real_N_cr) / real_N_cr
    print(f"  Surrogate vs. real:   {surrogate_err_pct:+.1f}% "
          f"({'surrogate over-predicted' if surrogate_err_pct > 0 else 'surrogate under-predicted'} "
          f"the real capacity)")

    real_feasible = real_N_cr >= target
    print(f"\n  Real solve {'MEETS' if real_feasible else 'FALLS SHORT OF'} the {target} N/mm target.")
    if not real_feasible:
        print(f"  This means the optimizer's design is NOT actually feasible in reality -- "
              f"the surrogate over-predicted its buckling capacity by "
              f"{abs(surrogate_err_pct):.1f}%. Worth adding this point (and nearby ones) to "
              f"the DOE and retraining the surrogate before trusting this optimum further.")
    elif abs(surrogate_err_pct) > 10:
        print(f"  It meets the target, but the surrogate was off by more than 10% at this "
              f"point -- likely because n_str=6 sits right at the DOE's upper bound, where "
              f"the GP has less data to interpolate from. Worth treating this specific "
              f"optimum with some caution and considering a few more DOE points near it.")
    else:
        print(f"  Surrogate prediction was within 10% of the real solve -- the optimizer's "
              f"result holds up against ground truth.")

    out = {
        "validated_result_file": str(result_path),
        "design": {"t_skin": t_skin, "n_str": n_str, "h_str": h_str, "t_str": t_str},
        "mass_kg": best["mass_kg"],
        "surrogate_predicted_N_cr_per_mm": surrogate_pred,
        "real_lsdyna_N_cr_per_mm": real_N_cr,
        "hand_calc_N_cr_per_mm": hand_calc_N_cr,
        "target_N_per_mm": target,
        "surrogate_error_pct": surrogate_err_pct,
        "real_meets_target": real_feasible,
    }
    out_path = ROOT / "optimizers" / "optimum_validation_result.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved full comparison to {out_path}")

    if not args.keep_work_dir:
        import shutil
        shutil.rmtree(work_root / "point_optimum_validation", ignore_errors=True)


if __name__ == "__main__":
    main()
