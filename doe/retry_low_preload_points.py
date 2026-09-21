"""
retry_low_preload_points.py

A second, OPPOSITE failure mode from the one PRELOAD_SAFETY_FACTOR was
originally tuned against. That constant was lowered over the life of this
project (0.15 -> 0.06 -> 0.02) specifically to stop the preload step from
exceeding a design's true critical load ("too high"). Running the infill
batch concentrated at thin skin (t_skin ~1.2-1.3mm) with n_str=6 -- a
corner of the design space not well represented in the original 200-point
DOE -- surfaced the mirror-image problem: LS-DYNA's own eigensolver
refused to trust its result with an explicit message ("Error 60419 ...
Numerical problems may be caused by too low initial loading for
buckling"), because at that corner the hand-calc estimate is low enough
that 0.02x of it sits near the solver's numerical noise floor. Exactly
the risk PRELOAD_SAFETY_FACTOR's own docstring flagged as worth watching
for, now confirmed for real.

This does NOT change the global PRELOAD_SAFETY_FACTOR (that would re-
litigate the 199/200 points already solved correctly at 0.02, and would
also sweep point 85 -- a separate, already-settled anomaly on the "too
high" side -- into a retry it doesn't need and isn't expected to help).
Instead it temporarily overrides the safety factor in-process, for only
the specific failed point_ids given, leaving everything else on disk
untouched.

Usage:
    python doe/retry_low_preload_points.py --point-ids 201 203 214 --safety-factor 0.08
"""

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ls_dyna_model"))
sys.path.insert(0, str(ROOT))

import classical_buckling  # noqa: E402 -- import the MODULE (not the function) so the
                             # override below is visible to solve_one_point_lsdyna,
                             # which reads classical_buckling.PRELOAD_SAFETY_FACTOR
                             # indirectly via reference_load_Pref_N() at call time.
from deck_builder import load_config, mass_kg  # noqa: E402
from run_doe_lsdyna import solve_one_point_lsdyna  # noqa: E402

RESULTS_FIELDNAMES = ["point_id", "t_skin", "n_str", "h_str", "t_str",
                       "mass_kg", "lambda_1", "N_cr_per_mm", "meets_target"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--point-ids", type=str, nargs="+", required=True,
                         help="point_id(s) to retry, as they appear in doe_points_panel.csv "
                              "and doe_failed_points_panel.json")
    parser.add_argument("--safety-factor", type=float, required=True,
                         help="Temporary PRELOAD_SAFETY_FACTOR override for just this retry "
                              "-- does not modify classical_buckling.py on disk.")
    parser.add_argument("--keep-work-dirs", action="store_true")
    args = parser.parse_args()

    print(f"Overriding PRELOAD_SAFETY_FACTOR: {classical_buckling.PRELOAD_SAFETY_FACTOR} "
          f"-> {args.safety_factor} (in-process only, for point_ids {args.point_ids})\n")
    classical_buckling.PRELOAD_SAFETY_FACTOR = args.safety_factor

    config = load_config()
    points_path = ROOT / "results" / "doe_points_panel.csv"
    results_path = ROOT / "results" / "doe_results_panel.csv"
    failed_path = ROOT / "results" / "doe_failed_points_panel.json"

    with open(points_path) as f:
        points_by_id = {r["point_id"]: r for r in csv.DictReader(f)}

    missing = [pid for pid in args.point_ids if pid not in points_by_id]
    if missing:
        print(f"point_id(s) not found in {points_path.name}: {missing}")
        sys.exit(1)

    if failed_path.exists():
        with open(failed_path) as f:
            failed_records = json.load(f)
    else:
        # No failed-points file yet (e.g. this is the first failure ever hit, or a
        # previous retry already cleared it) -- treat as empty rather than crash.
        failed_records = []

    work_root = ROOT / "results" / "lsdyna_work"
    work_root.mkdir(parents=True, exist_ok=True)

    new_result_rows = []
    still_failed_ids = set()
    for pid in args.point_ids:
        row = points_by_id[pid]
        t_skin, n_str = float(row["t_skin"]), int(row["n_str"])
        h_str, t_str = float(row["h_str"]), float(row["t_str"])
        print(f"[{pid}] t_skin={t_skin:.3f} n_str={n_str} h_str={h_str:.2f} "
              f"t_str={t_str:.3f} ...", end=" ")
        try:
            sol = solve_one_point_lsdyna(pid, t_skin, n_str, h_str, t_str, config, work_root)
            m = mass_kg(t_skin, n_str, h_str, t_str, config)
            meets = sol["N_cr_per_mm"] >= config["targets"]["design_running_load_N_per_mm"]
            print(f"N_cr={sol['N_cr_per_mm']:.1f} N/mm, mass={m:.3f}kg, "
                  f"{'MEETS' if meets else 'below'} target")
            new_result_rows.append({
                "point_id": pid, "t_skin": t_skin, "n_str": n_str,
                "h_str": h_str, "t_str": t_str,
                "mass_kg": f"{m:.5f}",
                "lambda_1": f"{sol['lambda_1']:.6f}",
                "N_cr_per_mm": f"{sol['N_cr_per_mm']:.4f}",
                "meets_target": meets,
            })
            if not args.keep_work_dirs:
                import shutil
                shutil.rmtree(work_root / f"point_{pid}", ignore_errors=True)
        except Exception as exc:  # noqa: BLE001
            print(f"FAILED -- {str(exc).splitlines()[0].strip()}")
            still_failed_ids.add(pid)

    if new_result_rows:
        file_mode = "a" if results_path.exists() else "w"
        with open(results_path, file_mode, newline="") as f:
            writer = csv.DictWriter(f, fieldnames=RESULTS_FIELDNAMES)
            if file_mode == "w":
                writer.writeheader()
            writer.writerows(new_result_rows)
        print(f"\nAppended {len(new_result_rows)} newly-solved point(s) to {results_path.name}")

    solved_ids = {r["point_id"] for r in new_result_rows}
    remaining_failed = [r for r in failed_records
                         if not (r["point_id"] in solved_ids and r["point_id"] not in still_failed_ids)]
    if len(remaining_failed) != len(failed_records):
        with open(failed_path, "w") as f:
            json.dump(remaining_failed, f, indent=2)
        print(f"Removed {len(failed_records) - len(remaining_failed)} now-solved point(s) "
              f"from {failed_path.name}")

    if still_failed_ids:
        print(f"\n{len(still_failed_ids)} point(s) still failed at safety_factor="
              f"{args.safety_factor}: {sorted(still_failed_ids)} -- still in {failed_path.name}, "
              f"unchanged.")
    print(f"\nReminder: PRELOAD_SAFETY_FACTOR override was in-process only -- "
          f"classical_buckling.py on disk is untouched. If this value works well, update its "
          f"docstring and consider whether it should apply more broadly.")


if __name__ == "__main__":
    main()
