"""
generate_infill_points.py

Targeted "infill" sampling: unlike generate_doe_panel.py's global
Latin-hypercube sweep, this samples a small batch of NEW points concentrated
in the neighborhood the classical optimizer actually converged to --
t_skin/h_str/t_str near the current optimum, n_str restricted to {5, 6} (the
narrow-bay, thin-skin corner that turned out to be mass-efficient).

Why this exists: optimizers/validate_optimum.py found the surrogate was
~1.7% optimistic exactly at the optimizer's winning design point, which
sits right at n_str's upper bound (6) -- the edge of the DOE's design
space, where a Gaussian Process has the least data to interpolate from
and is most likely to extrapolate poorly. Adding a handful of real solves
right in that neighborhood (rather than more points spread uniformly
across the whole space, which is what another global --add batch would
give you) directly targets the region the surrogate needs to get right,
the same "infill" idea used in sequential/Bayesian design optimization:
solve near where the optimizer keeps landing, retrain, re-optimize,
repeat until the surrogate and the optimizer agree.

This only appends to results/doe_points_panel.csv -- exactly like
generate_doe_panel.py's --add mode (same seed-tracking safety, same
point_id continuation), so run_doe_lsdyna.py picks these up as new,
unsolved points on the next run. It does NOT touch doe_results_panel.csv;
that only gets rows once these are actually solved.

Usage:
    python doe/generate_infill_points.py --n 15 --seed 50
    python doe/generate_infill_points.py --n 15 --seed 50 \\
        --t-skin-center 1.546 --t-skin-halfwidth 0.35 \\
        --h-str-center 29.67 --h-str-halfwidth 6 \\
        --t-str-center 1.05 --t-str-halfwidth 0.3
"""

import argparse
import csv
import json
import random
import sys
from pathlib import Path

from scipy.stats import qmc

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SEEDS_LOG_PATH = ROOT / "results" / "doe_points_panel.seeds.json"


def load_config():
    with open(ROOT / "ls_dyna_model" / "panel_config.json") as f:
        return json.load(f)


def clamp_band(center, halfwidth, global_bounds):
    lo = max(center - halfwidth, global_bounds[0])
    hi = min(center + halfwidth, global_bounds[1])
    if lo >= hi:
        raise ValueError(f"Infill band [{lo}, {hi}] for center={center}, "
                          f"halfwidth={halfwidth} is empty after clamping to global "
                          f"bounds {global_bounds} -- widen the band or move the center.")
    return lo, hi


def generate(n, seed, start_id, t_skin_band, h_str_band, t_str_band, n_str_choices):
    sampler = qmc.LatinHypercube(d=3, seed=seed)  # t_skin, h_str, t_str -- n_str sampled separately (discrete)
    unit_sample = sampler.random(n=n)
    lowers = [t_skin_band[0], h_str_band[0], t_str_band[0]]
    uppers = [t_skin_band[1], h_str_band[1], t_str_band[1]]
    sample = qmc.scale(unit_sample, lowers, uppers)

    rng = random.Random(seed)
    rows = []
    for i, (t_skin, h_str, t_str) in enumerate(sample):
        rows.append({
            "point_id": start_id + i,
            "t_skin": round(float(t_skin), 5),
            "n_str": rng.choice(n_str_choices),
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
    parser.add_argument("--n", type=int, default=15)
    parser.add_argument("--seed", type=int, required=True,
                         help="Must not collide with any seed already used for this DOE "
                              "(main sweep or a previous infill batch) -- shared seeds log "
                              "with generate_doe_panel.py catches this.")
    parser.add_argument("--t-skin-center", type=float, default=1.546)
    parser.add_argument("--t-skin-halfwidth", type=float, default=0.35)
    parser.add_argument("--h-str-center", type=float, default=29.67)
    parser.add_argument("--h-str-halfwidth", type=float, default=6.0)
    parser.add_argument("--t-str-center", type=float, default=1.05)
    parser.add_argument("--t-str-halfwidth", type=float, default=0.3)
    parser.add_argument("--n-str-choices", type=int, nargs="+", default=[5, 6],
                         help="n_str values to sample from -- restricted by default to the "
                              "narrow-bay corner the optimizer converged to.")
    args = parser.parse_args()

    config = load_config()
    dp = config["design_parameters"]

    t_skin_band = clamp_band(args.t_skin_center, args.t_skin_halfwidth, dp["t_skin"]["bounds"])
    h_str_band = clamp_band(args.h_str_center, args.h_str_halfwidth, dp["h_str"]["bounds"])
    t_str_band = clamp_band(args.t_str_center, args.t_str_halfwidth, dp["t_str"]["bounds"])
    for choice in args.n_str_choices:
        if not (dp["n_str"]["bounds"][0] <= choice <= dp["n_str"]["bounds"][1]):
            print(f"--n-str-choices value {choice} is outside the global n_str bounds "
                  f"{dp['n_str']['bounds']}.")
            sys.exit(1)

    out_path = ROOT / "results" / "doe_points_panel.csv"
    if not out_path.exists():
        print(f"{out_path} doesn't exist yet -- run generate_doe_panel.py first.")
        sys.exit(1)
    with open(out_path) as f:
        existing_rows = list(csv.DictReader(f))

    used_seeds = _load_seeds_log()
    if args.seed in used_seeds:
        print(f"Seed {args.seed} was already used for a previous batch of this DOE "
              f"(used so far: {used_seeds}) -- pick a new, unused --seed.")
        sys.exit(1)

    next_id = (max(int(r["point_id"]) for r in existing_rows) + 1) if existing_rows else 0
    new_rows = generate(args.n, args.seed, next_id, t_skin_band, h_str_band, t_str_band,
                         args.n_str_choices)

    fieldnames = ["point_id", "t_skin", "n_str", "h_str", "t_str"]
    with open(out_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerows(new_rows)
    used_seeds.append(args.seed)
    _save_seeds_log(used_seeds)

    print(f"Appended {len(new_rows)} infill points (ids {next_id}-{next_id + len(new_rows) - 1}) "
          f"to {out_path} -- total now {len(existing_rows) + len(new_rows)} points.")
    print(f"  t_skin band: [{t_skin_band[0]:.3f}, {t_skin_band[1]:.3f}] mm")
    print(f"  h_str band:  [{h_str_band[0]:.3f}, {h_str_band[1]:.3f}] mm")
    print(f"  t_str band:  [{t_str_band[0]:.3f}, {t_str_band[1]:.3f}] mm")
    print(f"  n_str choices: {args.n_str_choices}")
    print(f"\nNext: python doe/run_doe_lsdyna.py   (resumable -- solves just these new points)")


if __name__ == "__main__":
    main()
