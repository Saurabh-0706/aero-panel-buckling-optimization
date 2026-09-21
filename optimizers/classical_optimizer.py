"""
classical_optimizer.py

The primary, guaranteed-runnable optimizer: no API key, no external
service, no cost -- just scipy running locally. Minimizes panel mass
subject to the surrogate-predicted buckling capacity meeting the design
target, using differential evolution (a population-based metaheuristic --
the classical/non-LLM counterpart to optimizers/llm_optimizer.py).

Problem:
    minimize    mass_kg(t_skin, n_str, h_str, t_str)
    subject to  surrogate.predict(t_skin, n_str, h_str, t_str) >= design_running_load_N_per_mm
    bounds      from ls_dyna_model/panel_config.json's design_parameters
    n_str       integer (2..6), others continuous

Why differential evolution specifically: it's a global metaheuristic (no
gradient needed -- the surrogate is a GP, not something conveniently
differentiable-by-hand here), it natively supports the mixed
integer/continuous search space via scipy's `integrality` argument (n_str
stays a true integer throughout the search, not rounded post-hoc, which
would risk landing on an infeasible or non-optimal rounded point), and it
natively supports nonlinear constraints via `constraints=`, so the
buckling-capacity requirement is enforced during the search rather than
bolted on as an ad-hoc penalty term.

Because DE is stochastic, this runs it from several independent seeds and
reports the best feasible result found, plus the spread across seeds, so
the reported optimum isn't a one-off lucky draw.

Usage:
    python optimizers/classical_optimizer.py
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
from scipy.optimize import NonlinearConstraint, differential_evolution

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ls_dyna_model"))

from deck_builder import load_config, mass_kg  # noqa: E402
from classical_buckling import subpanel_running_load_N_per_mm  # noqa: E402

FEATURE_NAMES = ["t_skin", "n_str", "h_str", "t_str"]
N_SEEDS = 6


def load_surrogate(model_path):
    if not model_path.exists():
        raise RuntimeError(
            f"No surrogate model found at {model_path} -- run "
            f"`python surrogate/train_surrogate.py` first."
        )
    bundle = joblib.load(model_path)
    return bundle["pipeline"]


def decode(x):
    """x = [t_skin, n_str, h_str, t_str] with n_str snapped to the nearest
    integer -- DE's `integrality` flag already keeps it exactly integer-
    valued during the search, this round() is just defensive (float
    representation of an integer, e.g. 4.0 vs 3.9999999999998)."""
    t_skin, n_str, h_str, t_str = x
    return float(t_skin), int(round(n_str)), float(h_str), float(t_str)


def make_objective(config):
    def objective(x):
        t_skin, n_str, h_str, t_str = decode(x)
        return mass_kg(t_skin, n_str, h_str, t_str, config)
    return objective


def make_constraint_fn(surrogate):
    def predicted_N_cr(x):
        t_skin, n_str, h_str, t_str = decode(x)
        X = np.array([[t_skin, n_str, h_str, t_str]])
        return surrogate.predict(X)[0]
    return predicted_N_cr


def run_optimization(config, surrogate, target_N_per_mm, seeds=range(N_SEEDS)):
    dp = config["design_parameters"]
    bounds = [tuple(dp[name]["bounds"]) for name in FEATURE_NAMES]
    integrality = [False, True, False, False]  # only n_str is integer

    objective = make_objective(config)
    predicted_N_cr = make_constraint_fn(surrogate)
    constraint = NonlinearConstraint(predicted_N_cr, target_N_per_mm, np.inf)

    results = []
    for seed in seeds:
        res = differential_evolution(
            objective,
            bounds,
            integrality=integrality,
            constraints=(constraint,),
            seed=seed,
            popsize=30,
            maxiter=400,
            tol=1e-8,
            polish=True,
            mutation=(0.4, 1.0),
            recombination=0.8,
        )
        t_skin, n_str, h_str, t_str = decode(res.x)
        feasible = bool(predicted_N_cr(res.x) >= target_N_per_mm - 1e-6)
        results.append({
            "seed": seed,
            "success": bool(res.success),
            "feasible": feasible,
            "mass_kg": float(res.fun),
            "t_skin": t_skin, "n_str": n_str, "h_str": h_str, "t_str": t_str,
            "predicted_N_cr_per_mm": float(predicted_N_cr(res.x)),
        })
    return results


def summarize(results, target_N_per_mm, config):
    feasible_results = [r for r in results if r["feasible"]]
    if not feasible_results:
        return None, results

    best = min(feasible_results, key=lambda r: r["mass_kg"])

    # Cross-check against the same skin-only classical hand-calc used to
    # size Pref in the LS-DYNA pipeline -- a genuinely independent sanity
    # floor, not just a re-statement of the surrogate's own prediction. See
    # classical_buckling.py's module docstring: this hand-calc ignores
    # stringer-flexural/overall-panel modes, so it's expected to be in the
    # same ballpark as (not identical to) the surrogate/LS-DYNA number.
    hand_calc_N_cr = subpanel_running_load_N_per_mm(best["t_skin"], best["n_str"], config)

    # GP predictive std at the optimum -- how much the surrogate itself
    # trusts this prediction. A high value here would be a flag that DE
    # found its optimum by exploiting a region the surrogate is
    # extrapolating in, e.g. right at a bounds corner sparse in DOE data.
    return best, results, hand_calc_N_cr


def main():
    config = load_config()
    target = config["targets"]["design_running_load_N_per_mm"]
    model_path = ROOT / "surrogate" / "surrogate_model.joblib"
    surrogate = load_surrogate(model_path)

    print(f"Optimizing for minimum mass subject to surrogate-predicted "
          f"N_cr_per_mm >= {target} N/mm (running {N_SEEDS} independent DE seeds)...\n")

    results = run_optimization(config, surrogate, target)

    for r in results:
        status = "OK " if r["feasible"] else "INFEASIBLE"
        print(f"  seed {r['seed']}: [{status}] mass={r['mass_kg']:.4f} kg  "
              f"t_skin={r['t_skin']:.4f} n_str={r['n_str']} h_str={r['h_str']:.3f} "
              f"t_str={r['t_str']:.4f}  pred_N_cr={r['predicted_N_cr_per_mm']:.1f} N/mm")

    feasible = [r for r in results if r["feasible"]]
    if not feasible:
        print("\nNo seed found a feasible design meeting the target within the "
              "bounds -- either the target is infeasible for this design space, "
              "or DE needs more iterations/popsize. Not writing a result file.")
        sys.exit(1)

    best = min(feasible, key=lambda r: r["mass_kg"])
    hand_calc_N_cr = subpanel_running_load_N_per_mm(best["t_skin"], best["n_str"], config)

    # GP predictive std, computed directly (best["predicted_N_cr_per_mm"]
    # only carries the mean prediction from the constraint function above).
    X_best = np.array([[best["t_skin"], best["n_str"], best["h_str"], best["t_str"]]])
    _, std = surrogate.predict(X_best, return_std=True)
    std = float(std[0])

    print(f"\nBest feasible design across {len(results)} seeds "
          f"({len(feasible)}/{len(results)} seeds found a feasible result):")
    print(f"  mass       = {best['mass_kg']:.4f} kg")
    print(f"  t_skin     = {best['t_skin']:.4f} mm")
    print(f"  n_str      = {best['n_str']}")
    print(f"  h_str      = {best['h_str']:.4f} mm")
    print(f"  t_str      = {best['t_str']:.4f} mm")
    print(f"  surrogate-predicted N_cr = {best['predicted_N_cr_per_mm']:.1f} N/mm "
          f"(+/- {std:.1f} N/mm GP std)  vs target {target} N/mm")
    print(f"  independent skin-only hand-calc N_cr = {hand_calc_N_cr:.1f} N/mm "
          f"(sanity floor only -- ignores stringer/overall-panel modes, see "
          f"classical_buckling.py)")

    if std > 0.15 * best["predicted_N_cr_per_mm"]:
        print(f"  NOTE: GP predictive std is >15% of the predicted value at this "
              f"optimum -- the surrogate has relatively low confidence here "
              f"(likely a sparsely-sampled corner of the design space). Consider "
              f"adding DOE points near this design before trusting the result "
              f"for anything beyond a portfolio demo.")

    out = {
        "target_N_per_mm": target,
        "n_seeds": N_SEEDS,
        "n_feasible_seeds": len(feasible),
        "best": {
            "mass_kg": best["mass_kg"],
            "t_skin": best["t_skin"],
            "n_str": best["n_str"],
            "h_str": best["h_str"],
            "t_str": best["t_str"],
            "surrogate_predicted_N_cr_per_mm": best["predicted_N_cr_per_mm"],
            "surrogate_std_N_cr_per_mm": std,
            "hand_calc_N_cr_per_mm": hand_calc_N_cr,
        },
        "all_seed_results": results,
    }
    out_path = ROOT / "optimizers" / "classical_optimizer_result.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved full result to {out_path}")


if __name__ == "__main__":
    main()
