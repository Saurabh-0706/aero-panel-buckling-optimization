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
   **Final dataset: 199/200 points solved cleanly** (1 excluded as an
   unresolved anomaly, documented in `BUILD_GUIDE.md`) --
   `results/doe_results_panel.csv`.
2. **Surrogate model** (`surrogate/train_surrogate.py`) -- Gaussian
   Process Regression on `N_cr_per_mm` (the expensive-to-evaluate
   quantity), trained on the 199-point dataset. Mass is NOT surrogated --
   it's exact and free to compute analytically
   (`ls_dyna_model/deck_builder.py`'s `mass_kg()`). 5-fold
   cross-validated: **pooled R² = 0.987, worst single fold R² = 0.976**,
   RMSE ≈ 26 N/mm against a 300 N/mm design target.
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
   surrogate's word for it.

## Results so far

| Method | Best feasible mass | Notes |
|---|---|---|
| Best raw DOE point meeting target | 1.296 kg | Random sample, not optimized |
| Classical optimizer (differential evolution) | **1.103 kg** | ~15% lighter than the best DOE point; consistent across all 6 seeds |
| LLM optimizer | *(run `optimizers/llm_optimizer.py` with an API key)* | Optional |

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
