# Stiffened Panel Buckling Optimization

An AI-driven optimization pipeline for a stiffened aircraft panel (skin +
blade stringers under axial compression), built around real LS-DYNA
Student Edition buckling eigenvalue solves. Built as portfolio material
for FEM/structures-adjacent roles, mirroring an earlier COMSOL-based
crush-tube energy-absorption project but with a full LS-DYNA
DOE -> surrogate -> optimizer -> comparison pipeline end to end.

## Pipeline

1. **Design of experiments** (`doe/`) -- Latin-hypercube sampling over 4
   design variables (`t_skin`, `n_str`, `h_str`, `t_str`), each point
   solved as a real LS-DYNA linear buckling eigenvalue analysis
   (nonlinear implicit preload step + `*CONTROL_IMPLICIT_BUCKLE`
   eigenvalue extraction), driven end-to-end through PyDYNA
   (`ansys-dyna-core`) rather than hand-written keyword text. See
   `ls_dyna_model/BUILD_GUIDE.md` for the full build/debug history,
   including every real-solver issue hit and how each was diagnosed and
   fixed (not guessed at).
   **Final dataset: 215 points** (199 from the original sweep, 1 real
   validated optimum, and 15 further real solves from a targeted infill
   batch concentrated at the optimizer's converged design -- see step 6/7;
   the original sweep's 1 anomaly, and a second, opposite preload failure
   mode surfaced and fixed in the infill batch, are both documented in
   `BUILD_GUIDE.md`) -- `results/doe_results_panel.csv`.
2. **Surrogate model** (`surrogate/train_surrogate.py`) -- Gaussian
   Process Regression on `N_cr_per_mm` (the expensive-to-evaluate
   quantity), trained on the dataset above. Mass is NOT surrogated --
   it's exact and free to compute analytically
   (`ls_dyna_model/deck_builder.py`'s `mass_kg()`). 5-fold
   cross-validated: **pooled R² = 0.988, worst single fold R² = 0.983**,
   RMSE ≈ 24 N/mm against a 300 N/mm design target.
3. **Classical optimizer** (`optimizers/classical_optimizer.py`) -- the
   primary, guaranteed-free optimizer: differential evolution (a
   population-based metaheuristic) minimizing mass subject to the
   surrogate-predicted buckling capacity meeting the target, with `n_str`
   held to a true integer throughout the search via scipy's
   `integrality` support. Run from 6 independent seeds for a robustness
   check, not just a single stochastic run.
4. **LLM-driven optimizer** (`optimizers/llm_optimizer.py`, optional) --
   an LLM (Google Gemini, chosen specifically because its free tier needs
   no credit card) iteratively proposes candidate designs, sees the
   running history and best-so-far result, and refines its guesses,
   scored against the same surrogate + analytical mass function as the
   classical optimizer. Requires a free `GEMINI_API_KEY`; degrades
   gracefully with clear setup instructions (not a traceback) if one
   isn't configured, and nothing else in the pipeline depends on it
   having been run.
5. **Comparison** (`optimizers/compare_optimizers.py`) -- plots the DOE
   points as background context alongside both optimizers' results (or
   just the classical one, if the LLM optimizer hasn't been run) and
   prints a numeric summary. Output: `results/optimizer_comparison.png`.
6. **Validation** (`optimizers/validate_optimum.py`) -- closes the loop:
   an optimizer's result only exists inside the surrogate model until
   this runs the exact winning design point through a real LS-DYNA solve
   and reports a three-way comparison (surrogate prediction vs. real
   solve vs. the independent hand-calc), rather than trusting the
   surrogate's word for it. This caught a real gap: the first optimum
   (1.103 kg, `n_str` at its upper bound) came in 1.7% short of target in
   the real solve. `doe/add_validated_point.py` folded that real result
   into the dataset and the surrogate/optimizer were re-run.
7. **Infill sampling + a second failure mode** (`doe/generate_infill_points.py`)
   -- rather than stopping at one patched point, added 15 further real
   solves concentrated right where the optimizer converges (same
   sequential/Bayesian "infill" idea used in step 6). 3 of those 15 hit a
   **new, opposite** solver failure from every earlier one in this project
   -- preload set too *low* rather than too high, at a thin-skin/narrow-bay
   corner -- diagnosed from the real LS-DYNA log (not guessed) and fixed
   with a targeted retry (`doe/retry_low_preload_points.py`) rather than
   changing the global tuning constant. Full story in `BUILD_GUIDE.md`.
   Re-running the surrogate/optimizer on the final 215-point dataset and
   re-validating gave the result below -- this time the real solve
   *clears* the target, closing the loop for good.

## Results so far

| Method | Best feasible mass | Notes |
|---|---|---|
| Best raw DOE point meeting target | 1.200 kg | Random sample, not optimized |
| Classical optimizer (differential evolution) | **1.122 kg** | Consistent across all 6 seeds; validated against a real LS-DYNA solve (see below) |
| LLM optimizer (Gemini, 12 rounds) | 1.361 kg | 21% heavier than the classical result -- see `ls_dyna_model` case study for why |

**Validation result:** the first classical optimum (1.103 kg) was re-solved
through real LS-DYNA and came in 1.7% short of the 300 N/mm target -- the
surrogate was slightly optimistic right at `n_str`'s upper bound, the edge
of the design space. That real point was added to the dataset, and a
further 15-point infill batch concentrated at that same corner (3 of
which needed a targeted preload-tuning fix for a new, opposite failure
mode -- see `BUILD_GUIDE.md`) brought the dataset to 215 points. Re-running
the surrogate and optimizer gave the 1.122 kg result above, with a
tighter predictive uncertainty at the optimum (±13.6 vs. ±21.8 N/mm
originally). Re-validated against a real LS-DYNA solve: **309.0 N/mm real
vs. 300.0 N/mm surrogate-predicted vs. 316.3 N/mm hand-calc sanity
floor** -- the surrogate under-predicted by 2.9% this time (the safe
direction), and the real solve clears the target. See
`optimizers/optimum_validation_result.json`.

## Running it

```bash
pip install -r requirements.txt

# 1. DOE (already solved -- results/doe_results_panel.csv is the final dataset).
#    To extend it further or start over, see doe/generate_doe_panel.py and
#    doe/run_doe_lsdyna.py's own docstrings.

# 2. Surrogate model
python surrogate/train_surrogate.py

# 3. Classical optimizer (no API key needed)
python optimizers/classical_optimizer.py

# 4. LLM optimizer (optional -- needs a free GEMINI_API_KEY, see script docstring)
python optimizers/llm_optimizer.py

# 5. Comparison plot (works with just step 3, or steps 3+4)
python optimizers/compare_optimizers.py

# 6. Validate the classical optimizer's result against a real LS-DYNA solve
python optimizers/validate_optimum.py

# 7. If validation finds a gap: fold the real point into the dataset and re-run
python doe/add_validated_point.py
python surrogate/train_surrogate.py
python optimizers/classical_optimizer.py

# 8. A further batch of real solves concentrated near the optimum, for
#    closing an edge-of-design-space gap more thoroughly than one point can
python doe/generate_infill_points.py --n 15 --seed 50
python doe/run_doe_lsdyna.py

# 8b. If any infill points fail with "too low initial loading for buckling"
#     (a different failure mode than #7's -- see BUILD_GUIDE.md), retry just
#     those point_ids with a higher, per-point preload safety factor rather
#     than changing the global default
python doe/retry_low_preload_points.py --point-ids <ids> --safety-factor 0.06

# 9. Re-run steps 2-3 (and 5-6) on the fully closed-out dataset
python surrogate/train_surrogate.py
python optimizers/classical_optimizer.py
python optimizers/compare_optimizers.py
python optimizers/validate_optimum.py
```

## Project structure

```
ls_dyna_model/          Deck builder (PyDYNA), classical hand-calc cross-check,
                         panel_config.json, BUILD_GUIDE.md (full build/debug narrative)
doe/                     DOE generation + resumable LS-DYNA-driving pipeline
results/                 DOE points/results CSVs, trained surrogate's validation report,
                         final comparison plot
surrogate/               Surrogate model training script + fitted model
optimizers/              Classical (DE) + optional LLM optimizers, comparison script
```

## Design philosophy

Carried over from the earlier crush-tube project: never silently record a
nonphysical result. Every stage that can fail (a solver crash, a
negative/nonphysical eigenvalue, non-physical surrogate training data) is
checked explicitly and either raises loudly or self-heals a previously
corrupted file -- see `ls_dyna_model/BUILD_GUIDE.md` and the module
docstrings in `doe/run_doe_lsdyna.py` and `surrogate/train_surrogate.py`
for the specific real issues this caught.
