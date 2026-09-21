"""
generate_doe_panel.py

Latin-hypercube DOE over the 4 panel design parameters (t_skin, n_str,
h_str, t_str), bounds pulled from ls_dyna_model/panel_config.json. n_str
is sampled continuously like the others and then rounded to the nearest
integer -- LHS on the continuous box, rounded at the end, is simpler than
mixed-integer LHS and fine at this sample size.

Usage:
    python doe/generate_doe_panel.py --n 60 --seed 43        # first generation
    python doe/generate_doe_panel.py --add 140 --seed 44     # extend an existing DOE

IMPORTANT about --n on an existing DOE: point_id is just a row index
(0..n-1), and LHS produces a DIFFERENT point set for a different n even
with the same seed -- so re-running with a bigger --n does NOT extend the
existing points file, it silently reassigns point_ids 0..59 to different
design parameters than whatever is already sitting in doe_results_panel.csv
under those same IDs (run_doe_lsdyna.py's resumability matches by point_id
only, not by re-checking the parameters). That would corrupt already-solved
data rather than error out -- exactly the kind of silent-bad-data risk this
project has been careful to design against elsewhere. --n now refuses to
overwrite a points file that has a results file referencing it; use --add
to extend the DOE instead, which keeps existing points/results untouched
and appends a genuinely new, independently-seeded batch with continuing
point_ids.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

from scipy.stats import qmc

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SEEDS_LOG_PATH = ROOT / "results" / "doe_points_panel.seeds.json"


def load_config():
    with open(ROOT / "ls_dyna_model" / "panel_config.json") as f:
        return json.load(f)


def generate(n, seed, start_id=0):
    config = load_config()
    dp = config["design_parameters"]
    names = ["t_skin", "n_str", "h_str", "t_str"]
    lowers = [dp[n_]["bounds"][0] for n_ in names]
    uppers = [dp[n_]["bounds"][1] for n_ in names]

    sampler = qmc.LatinHypercube(d=len(names), seed=seed)
    unit_sample = sampler.random(n=n)
    sample = qmc.scale(unit_sample, lowers, uppers)

    rows = []
    for i, point in enumerate(sample):
        t_skin, n_str, h_str, t_str = point
        rows.append({
            "point_id": start_id + i,
            "t_skin": round(float(t_skin), 5),
            "n_str": int(round(n_str)),
            "h_str": round(float(h_str), 5),
            "t_str": round(float(t_str), 5),
        })
    return rows


def _load_seeds_log():
    if SEEDS_LOG_PATH.exists():
        with open(SEEDS_LOG_PATH) as f:
            return json.load(f)
    return []


def _save_seeds_log(seeds):
    with open(SEEDS_LOG_PATH, "w") as f:
        json.dump(seeds, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=60,
                         help="Fresh generation (overwrites the points file). Refuses to run if a "
                              "results file already references the existing points file -- see "
                              "--add.")
    parser.add_argument("--add", type=int, default=None,
                         help="Append this many NEW points to the existing points file instead of "
                              "starting over -- keeps all existing points/results untouched, "
                              "continues point_id numbering, and requires --seed to be one that "
                              "hasn't been used for a previous batch of this DOE.")
    parser.add_argument("--seed", type=int, default=43)
    args = parser.parse_args()

    out_path = ROOT / "results" / "doe_points_panel.csv"
    out_path.parent.mkdir(exist_ok=True)
    fieldnames = ["point_id", "t_skin", "n_str", "h_str", "t_str"]

    if args.add is not None:
        if not out_path.exists():
            print(f"{out_path} doesn't exist yet -- use --n for a first generation, not --add.")
            sys.exit(1)
        with open(out_path) as f:
            existing_rows = list(csv.DictReader(f))
        used_seeds = _load_seeds_log()
        if args.seed in used_seeds:
            print(f"Seed {args.seed} was already used for a previous batch of this DOE "
                  f"(used so far: {used_seeds}) -- pick a new, unused --seed so the appended "
                  f"points are a genuinely independent LHS batch, not a repeat of one already in "
                  f"the file.")
            sys.exit(1)
        next_id = (max(int(r["point_id"]) for r in existing_rows) + 1) if existing_rows else 0
        new_rows = generate(args.add, args.seed, start_id=next_id)
        with open(out_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerows(new_rows)
        used_seeds.append(args.seed)
        _save_seeds_log(used_seeds)
        print(f"Appended {len(new_rows)} new points (ids {next_id}-{next_id + len(new_rows) - 1}) "
              f"to {out_path} -- total now {len(existing_rows) + len(new_rows)} points.")
        return

    if out_path.exists():
        results_paths = [ROOT / "results" / "doe_results_panel.csv",
                          ROOT / "results" / "doe_results_panel_mock.csv"]
        existing_results = [p for p in results_paths if p.exists()]
        if existing_results:
            print(
                f"REFUSING to overwrite {out_path.name}: {[p.name for p in existing_results]} "
                "already reference point_ids from the CURRENT points file. Regenerating from "
                "scratch would silently reassign those same point_ids to DIFFERENT design "
                "parameters, corrupting the already-solved data (run_doe_lsdyna.py's "
                "resumability matches by point_id only, not by re-checking the parameters). Use "
                "--add N --seed <new seed> to extend the DOE instead, or delete the results "
                "file(s) first if you genuinely want to start over."
            )
            sys.exit(1)

    rows = generate(args.n, args.seed)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    _save_seeds_log([args.seed])
    print(f"Wrote {len(rows)} points to {out_path}")


if __name__ == "__main__":
    main()
