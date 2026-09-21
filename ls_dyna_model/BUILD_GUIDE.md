# Stiffened Panel Buckling Optimization -- Build Guide

## The problem

A flat aluminum skin panel with `n_str` equally-spaced longitudinal blade
stringers, loaded in axial (in-plane) compression -- representative of a
wing-lower-cover or fuselage-skin structural bay. Goal: find the
minimum-mass panel (skin thickness, stringer count/height/thickness) that
still meets a target buckling load per unit width.

Full geometry, material, BC, and load-case details live in
`panel_config.json` -- this file is the narrative version of the same
information, plus the reasoning behind the key choices.

## Setup

1. `pip install ansys-dyna-core` (PyDYNA -- this is what builds the
   keyword deck and drives the solver; verified locally against
   `ansys-dyna-core==0.12.1`).
2. Find your Student install's solver executable. It's a pure
   command-line batch program (`lsdyna_sp.exe` on Windows,
   `lsdyna_sp` on Linux) -- no GUI involved to run a solve. Easiest way to
   find the exact path: right-click the Start Menu shortcut LS-DYNA
   installed → Properties → look at the "Target"/command line, or just
   browse to the install folder for a `bin\winx64` (Windows) or
   `bin/linx64` (Linux) subfolder containing it.
3. Paste that full path into `panel_config.json`'s `solver.executable`
   field, replacing the `"CONFIRM -- ..."` placeholder.

That's it -- no separate registration step. `doe/run_doe_lsdyna.py` reads
`solver.executable` straight out of `panel_config.json` and passes it to
PyDYNA's `run_dyna(deck, executable=that_path, ...)` on every solve. This
was a deliberate choice over PyDYNA's `save-ansys-path` /
ansys-tools-path auto-discovery mechanism: that mechanism is real, but by
reading PyDYNA's own source (`ansys/dyna/core/run/windows_runner.py`) it
turns out only the *Linux* runner ever consults it --
`WindowsRunner._find_solver()` never calls `get_dyna_path()` at all, it
only accepts an explicit `executable=` kwarg or hunts for a full Ansys
Unified install (which a Student LS-DYNA-only install typically is not).
Passing the path explicitly sidesteps that split entirely and behaves
identically on both platforms.

LS-PrePost (the GUI) is untouched by any of this and remains available any
time you want to manually open a `.k` deck or a results file for visual
inspection -- this pipeline just doesn't need it to run a solve.

## Why linear buckling, not a nonlinear collapse simulation

The previous project (crush-tube energy absorption) used a nonlinear
transient solve because the physics genuinely required it -- a real
snap-through buckling/plastic-fold collapse. That's also exactly what made
it slow (30-45+ min/point) and numerically fragile (self-contact,
elastoplastic convergence failures, a two-stage rate profile needed just to
coax the solver through).

A stiffened panel's *initial* buckling load, by contrast, is a **linear
eigenvalue problem**: given a reference in-plane stress state, what
multiple of that load causes the panel to become unstable? No plasticity,
no self-contact, no time-stepping through a collapse event -- just an
eigenvalue extraction on top of one linear-ish static solve. This is
standard practice for first-pass aerospace panel sizing (a panel is then
often re-checked for post-buckled strength separately, which is out of
scope here) and should solve in seconds per point rather than tens of
minutes, with none of the convergence drama the crush-tube model had.

## Workflow (per design point)

1. Generate nodes + shell elements for the skin (`La x Wb`) and `n_str`
   blade stringers from `(t_skin, n_str, h_str, t_str)`.
2. `*BOUNDARY_SPC_SET`: simply-supported edges (see `panel_config.json`'s
   `boundary_conditions` block for the exact DOF set per edge).
3. Nonlinear implicit static pre-load step (`*CONTROL_IMPLICIT_GENERAL` +
   `*CONTROL_IMPLICIT_STATIC`) applying `reference_load_Pref_N` at the
   loaded edge.
4. `*CONTROL_IMPLICIT_BUCKLE` eigenvalue extraction referencing that stress
   state -- lowest 3 eigenvalues (buckling load factors).
5. Python side (not LS-DYNA expressions, same "keep the arithmetic
   independently checkable" philosophy as before): read the lowest
   eigenvalue `lambda_1`, compute `critical_load = lambda_1 * Pref`,
   `N_cr = critical_load / Wb` (running load), and `mass` analytically from
   the four design parameters.

## Confirmed against a real solve (2026-09-21, Student R16.1, Windows)

Two first-draft guesses turned out wrong on the very first real run, both
now fixed:

- **Eigenvalue output location.** `d3hsp` does NOT contain the eigenvalues
  for this version -- it only logs a line like `write eigout file` as the
  real results go to a separate text file, `eigout`, with a
  `MODE    EIGENVALUE  ...` table. `run_doe_lsdyna.py`'s
  `_parse_eigenvalue_from_eigout()` now reads that file directly.
- **Load direction sign.** The first-draft deck applied the reference load
  in +X on the free edge while the opposite edge was fixed in X -- that
  pulls the panel apart (tension) rather than pushing it together
  (compression). The real solve came back with all-negative eigenvalues,
  which is the textbook symptom of exactly this sign error (buckling only
  under a load reversed from what was applied) -- not a guess, read
  straight off real output. `deck_builder.py`'s `build_deck()` now applies
  the load with a negative `sf` so it pushes the free edge back toward the
  fixed edge.

## Confirmed against a full 60-point DOE sweep (2026-09-21)

Single-point tests came back positive and sane (within ~30-55% of the
classical hand-calc, expected given the hand-calc's known blind spots --
see its own docstring). But the full sweep surfaced one more real issue:
5 of 60 points (8.3%) came back with a **negative** `lambda_1` that
"succeeded" with no exception, writing nonphysical data straight into the
results CSV. Checked several single-variable theories for what
distinguished those 5 points (stringer slenderness, stringer thickness,
stringer-to-skin area ratio) against the full result set -- none cleanly
separate them, which makes sense: `Pref` is sized purely off a SKIN-only
local-buckling hand-calc, so it has no way to account for a
stringer-flexural or overall-panel mode governing at a lower load instead,
and which geometries trigger that isn't reducible to one simple ratio.

Fixed two ways, not one: `PRELOAD_SAFETY_FACTOR` (classical_buckling.py)
raised from 0.15 to 0.06 for more headroom, AND (the actual correctness
fix, not just a tuning knob) `solve_one_point_lsdyna()` now treats any
non-positive `lambda_1` as a hard failure -- it raises, which routes the
point to `failed_points.json` for retry via `--retry-failed`, instead of
silently recording bad data. `run_doe_lsdyna.py`'s `load_existing()` also
self-heals: any negative-`N_cr_per_mm` row already sitting in an existing
results CSV (from before this check existed) is automatically stripped out
and re-queued the next time `run_doe_lsdyna.py` runs -- no manual CSV
editing needed.

## What is still NOT yet verified

- Whether the lower safety factor actually eliminates the negative-lambda
  failure mode, or just reduces its frequency -- the hard-failure check
  above is the real safety net either way.
- `*BOUNDARY_SPC_SET` node-group assignments -- if the lowest eigenvalue
  still comes back near zero or clearly nonphysical, a rigid-body mode may
  be sneaking through.
- Mesh density -- start coarse (fast), refine only if a mesh-convergence
  check shows the eigenvalue is still moving significantly.

Run `doe/test_single_point_lsdyna.py` first, on one point, before ever
attempting a full DOE sweep -- exactly the same reasoning as
`test_single_point_comsol.py` in the last project: fail fast on one point,
not after burning through 30 of them.

## Final DOE dataset (closed out 2026-09-21)

Extended from the initial 60-point sweep to 200 points via
`generate_doe_panel.py --add 140 --seed 44`. Of the 3 points that failed
even after tightening `PRELOAD_SAFETY_FACTOR` to 0.06 (points 85, 95,
117), 2 recovered cleanly at `PRELOAD_SAFETY_FACTOR = 0.02` via
`--retry-failed`; point 85 got *worse* (lambda_1 went from -7.18 to
-29.79) rather than better, which is the opposite of what "Pref still too
high" predicts -- flagged as a genuine anomaly worth a closer look
someday, not just a matter of pushing the safety factor lower, and
accepted as a known gap rather than chased further. **Final dataset:
199/200 points, zero non-physical (negative) results, point 85 excluded**
-- `results/doe_results_panel.csv`. This is the dataset the surrogate
model in `surrogate/train_surrogate.py` is trained on.

## Surrogate model + optimizers

See `surrogate/train_surrogate.py` (Gaussian Process Regression on
`N_cr_per_mm`, 5-fold cross-validated: pooled R^2 = 0.987, worst single
fold R^2 = 0.976, RMSE ~26 N/mm against a target of 300 N/mm -- solid for
a 199-point, 4D dataset) and `optimizers/` (a guaranteed-free
differential-evolution optimizer plus an optional LLM-driven one, see
`optimizers/classical_optimizer.py` and `optimizers/llm_optimizer.py`'s
own docstrings for the full reasoning). `optimizers/compare_optimizers.py`
produces the final comparison plot.
