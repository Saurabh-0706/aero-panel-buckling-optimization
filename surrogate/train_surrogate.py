"""
train_surrogate.py

Trains a surrogate model for N_cr_per_mm (the LS-DYNA-computed buckling
critical running load) as a function of the 4 design parameters
(t_skin, n_str, h_str, t_str), so the optimizers don't need to call
LS-DYNA for every candidate design they evaluate.

Deliberately NOT surrogating mass -- mass_kg() (in ls_dyna_model/
deck_builder.py) is a closed-form analytical function of the same 4
parameters, exact and free to evaluate, so surrogating it would only add
approximation error for no benefit. Only the expensive-to-evaluate
quantity (N_cr_per_mm, which requires an actual LS-DYNA eigenvalue solve)
gets a surrogate.

Model choice: Gaussian Process Regression (scikit-learn). Reasons for this
over e.g. a plain polynomial/RBF-network or a random forest:
  - Only 199 training points in a 4D space -- GPR is a strong choice at
    this sample size (a random forest / gradient-boosted tree needs more
    data to interpolate smoothly; GPR's kernel bakes in a smoothness
    assumption that matches the underlying physics, which genuinely is a
    smooth function of the 4 inputs).
  - GPR gives a predictive standard deviation for free (not used by the
    classical optimizer downstream, since it isn't Bayesian-optimization-
    based, but produced and reported here anyway as an honest quality
    signal -- and available if a future optimizer wants to use it, e.g.
    to penalize designs the surrogate is unsure about).
  - Inputs are standardized and n_str (integer, small range 2-6) is fed in
    as a continuous feature like the others -- the GP kernel handles
    mixed-scale continuous features fine after standardization; treating
    n_str as one-hot/categorical was considered but rejected because the
    relationship (more stringers -> narrower skin bays -> higher local
    buckling load, roughly monotonic) is fundamentally ordinal/continuous,
    matching how it's actually varied in the DOE and by the optimizers.

Validation: since there are only 199 points, a single train/test split
would waste data and give a noisy quality estimate. Uses 5-fold
cross-validation instead, reporting R^2 and RMSE (in N/mm, same units as
the target) both per-fold and pooled, so the reported quality number is
honest about the actual data available -- not an artifact of one lucky
split. The final shipped model is then refit on ALL 199 points (standard
practice: CV is only used to estimate quality, not to select the
deployed model's training set).

Usage:
    python surrogate/train_surrogate.py
    python surrogate/train_surrogate.py --data results/doe_results_panel.csv

Output:
    surrogate/surrogate_model.joblib   -- fitted Pipeline(StandardScaler, GPR)
    surrogate/surrogate_report.json    -- CV metrics + basic dataset stats
"""

import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
FEATURE_NAMES = ["t_skin", "n_str", "h_str", "t_str"]
TARGET_NAME = "N_cr_per_mm"


def load_dataset(csv_path):
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    X = np.array([[float(r[name]) for name in FEATURE_NAMES] for r in rows])
    y = np.array([float(r[TARGET_NAME]) for r in rows])
    point_ids = [r["point_id"] for r in rows]

    # Same non-physical-data guard used throughout this project (see
    # run_doe_lsdyna.py's load_existing() purge logic) -- the results file
    # should already be clean of these (the hard-failure check upstream
    # sends them to failed_points.json instead), but a surrogate trained on
    # even one bad row would silently corrupt everything built on top of
    # it, so this is checked again here rather than assumed.
    bad = [(pid, val) for pid, val in zip(point_ids, y) if val <= 0]
    if bad:
        raise RuntimeError(
            f"Refusing to train on non-physical data: {len(bad)} row(s) have "
            f"N_cr_per_mm <= 0 (point_ids: {[b[0] for b in bad]}). These "
            f"should have been caught upstream (see run_doe_lsdyna.py's "
            f"hard-failure check) -- fix the results file before retraining."
        )
    return X, y, point_ids


def build_pipeline():
    # ConstantKernel * RBF: standard smooth-function GP kernel: RBF encodes
    # "similar inputs -> similar outputs", ConstantKernel lets the GP learn
    # the output's overall variance scale rather than assuming it's ~1
    # (which it isn't -- N_cr_per_mm ranges from ~15 to ~2200+ N/mm across
    # the DOE, see classical_buckling.py's __main__ block). WhiteKernel adds
    # a learned noise floor -- appropriate here because N_cr_per_mm isn't
    # noise-free: it depends on the mesh discretization and the per-point
    # PRELOAD_SAFETY_FACTOR-scaled Pref (a purely numerical solve-setup
    # choice, see classical_buckling.py), both of which introduce small
    # point-to-point variation that isn't part of the "true" underlying
    # design->buckling-load function. Letting the GP fit a noise term
    # instead of forcing an exact interpolation avoids overfitting to that
    # solve-to-solve numerical noise.
    kernel = ConstantKernel(1.0, (1e-2, 1e4)) * RBF(
        length_scale=[1.0, 1.0, 1.0, 1.0], length_scale_bounds=(1e-2, 1e3)
    ) + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-3, 1e3))

    gpr = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,  # target also standardized internally (in addition to X below)
        n_restarts_optimizer=8,  # multiple restarts: kernel hyperparameter optimization is
                                   # non-convex, guards against a bad local optimum with only
                                   # one restart
        random_state=43,
    )
    return Pipeline([
        ("scaler", StandardScaler()),  # standardize the 4 inputs -- puts n_str (range 2-6)
                                         # and h_str (range 10-40mm) on comparable footing for
                                         # a single isotropic-ish length_scale-per-dimension kernel
        ("gpr", gpr),
    ])


def evaluate_cv(X, y, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=43)
    pipe = build_pipeline()
    y_pred = cross_val_predict(pipe, X, y, cv=kf)

    residuals = y - y_pred
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    mae = float(np.mean(np.abs(residuals)))
    mape = float(np.mean(np.abs(residuals / y)) * 100.0)

    # Per-fold R^2 too -- a pooled R^2 alone can hide one bad fold; report
    # the spread so the number isn't misleadingly tidy.
    fold_r2 = []
    for train_idx, test_idx in kf.split(X):
        pipe_fold = build_pipeline()
        pipe_fold.fit(X[train_idx], y[train_idx])
        pred_fold = pipe_fold.predict(X[test_idx])
        res_fold = y[test_idx] - pred_fold
        ss_res_f = float(np.sum(res_fold ** 2))
        ss_tot_f = float(np.sum((y[test_idx] - y[test_idx].mean()) ** 2))
        fold_r2.append(1.0 - ss_res_f / ss_tot_f if ss_tot_f > 0 else float("nan"))

    return {
        "n_splits": n_splits,
        "pooled_r2": r2,
        "pooled_rmse_N_per_mm": rmse,
        "pooled_mae_N_per_mm": mae,
        "pooled_mape_pct": mape,
        "per_fold_r2": fold_r2,
        "worst_fold_r2": min(fold_r2),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(ROOT / "results" / "doe_results_panel.csv"))
    parser.add_argument("--out-dir", default=str(ROOT / "surrogate"))
    args = parser.parse_args()

    data_path = Path(args.data)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    X, y, point_ids = load_dataset(data_path)
    print(f"Loaded {len(y)} clean data points from {data_path}")
    print(f"N_cr_per_mm range: [{y.min():.1f}, {y.max():.1f}] N/mm "
          f"(mean {y.mean():.1f}, std {y.std():.1f})\n")

    print("Running 5-fold cross-validation...")
    cv_metrics = evaluate_cv(X, y, n_splits=5)
    print(f"  Pooled R^2:   {cv_metrics['pooled_r2']:.4f}")
    print(f"  Pooled RMSE:  {cv_metrics['pooled_rmse_N_per_mm']:.2f} N/mm")
    print(f"  Pooled MAE:   {cv_metrics['pooled_mae_N_per_mm']:.2f} N/mm")
    print(f"  Pooled MAPE:  {cv_metrics['pooled_mape_pct']:.2f}%")
    print(f"  Per-fold R^2: {[round(r, 4) for r in cv_metrics['per_fold_r2']]}")
    print(f"  Worst fold R^2: {cv_metrics['worst_fold_r2']:.4f}\n")

    if cv_metrics["worst_fold_r2"] < 0.5:
        print("WARNING: at least one CV fold has R^2 < 0.5 -- the surrogate is "
              "not reliably capturing the design->buckling-load relationship "
              "in some region of the design space. Treat optimizer results "
              "with caution and consider adding more DOE points before "
              "trusting them for anything beyond a portfolio demo.\n")
    else:
        print("CV quality looks solid across all folds.\n")

    print(f"Refitting final model on all {len(y)} points...")
    final_pipe = build_pipeline()
    final_pipe.fit(X, y)

    fitted_kernel = final_pipe.named_steps["gpr"].kernel_
    print(f"Fitted kernel: {fitted_kernel}\n")

    model_path = out_dir / "surrogate_model.joblib"
    joblib.dump({
        "pipeline": final_pipe,
        "feature_names": FEATURE_NAMES,
        "target_name": TARGET_NAME,
    }, model_path)
    print(f"Saved fitted model to {model_path}")

    report = {
        "data_path": str(data_path),
        "n_points": len(y),
        "feature_names": FEATURE_NAMES,
        "target_name": TARGET_NAME,
        "target_stats": {
            "min": float(y.min()), "max": float(y.max()),
            "mean": float(y.mean()), "std": float(y.std()),
        },
        "cv_metrics": cv_metrics,
        "fitted_kernel": str(fitted_kernel),
    }
    report_path = out_dir / "surrogate_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved report to {report_path}")


if __name__ == "__main__":
    main()
