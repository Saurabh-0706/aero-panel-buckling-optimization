"""
test_single_point_lsdyna.py

Solves exactly one design point (defaulting to a mid-range baseline:
t_skin=2.0mm, n_str=3, h_str=20mm, t_str=2.0mm) against a real LS-DYNA
install and cross-checks it against the classical-plate-theory hand-calc
(ls_dyna_model/classical_buckling.py). Run this BEFORE attempting a full
DOE sweep with run_doe_lsdyna.py -- same reasoning as the crush-tube
project's test_single_point_comsol.py: fail fast on one point, not after
burning through 30 of them.

The classical hand-calc only captures LOCAL sub-panel buckling (it ignores
overall-panel and stringer-flexural modes), so don't expect an exact
match -- but it should be the same order of magnitude and same direction of
trend. A wildly different result (10x off, wrong sign of trend when you
vary a parameter) points at a deck/BC/unit problem rather than "the
hand-calc is just approximate."

Usage:
    python doe/test_single_point_lsdyna.py
    python doe/test_single_point_lsdyna.py --t-skin 1.5 --n-str 4 --h-str 15 --t-str 1.5
    python doe/test_single_point_lsdyna.py --keep-work-dir
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ls_dyna_model"))

from deck_builder import load_config, mass_kg  # noqa: E402
from classical_buckling import subpanel_running_load_N_per_mm  # noqa: E402
from run_doe_lsdyna import solve_one_point_lsdyna  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--t-skin", type=float, default=2.0)
    parser.add_argument("--n-str", type=int, default=3)
    parser.add_argument("--h-str", type=float, default=20.0)
    parser.add_argument("--t-str", type=float, default=2.0)
    parser.add_argument("--keep-work-dir", action="store_true")
    args = parser.parse_args()

    config = load_config()
    print(f"Solving one point: t_skin={args.t_skin}mm, n_str={args.n_str}, "
          f"h_str={args.h_str}mm, t_str={args.t_str}mm")
    print("(this should take seconds to at most a couple minutes -- a linear "
          "eigenvalue extraction, not a nonlinear transient collapse)\n")

    work_root = ROOT / "results" / "lsdyna_work_test"
    work_root.mkdir(parents=True, exist_ok=True)

    sol = solve_one_point_lsdyna("test", args.t_skin, args.n_str, args.h_str, args.t_str,
                                  config, work_root)
    m = mass_kg(args.t_skin, args.n_str, args.h_str, args.t_str, config)

    print("\n--- LS-DYNA result ---")
    print(f"Lowest eigenvalue (buckling load factor): {sol['lambda_1']:.4f}")
    print(f"Critical running load: {sol['N_cr_per_mm']:.1f} N/mm")
    print(f"Panel mass: {m:.4f} kg")

    hand_calc = subpanel_running_load_N_per_mm(args.t_skin, args.n_str, config)
    pct_diff = 100 * (sol["N_cr_per_mm"] - hand_calc) / hand_calc
    print("\n--- Cross-check against classical sub-panel buckling hand-calc ---")
    print(f"Hand-calc (local sub-panel buckling only): {hand_calc:.1f} N/mm")
    print(f"LS-DYNA vs. hand-calc: {pct_diff:+.1f}%")
    print("Expect LS-DYNA's result to sit in the same ballpark, likely somewhat "
          "different since the hand-calc ignores overall-panel/stringer-flexural "
          "modes -- a match within roughly 2x is a reasonable first sanity check. "
          "A wildly different order of magnitude points at a deck/BC/unit problem; "
          "paste this whole output back and we'll debug it together.")

    if not args.keep_work_dir:
        import shutil
        shutil.rmtree(work_root / "point_test", ignore_errors=True)


if __name__ == "__main__":
    main()
