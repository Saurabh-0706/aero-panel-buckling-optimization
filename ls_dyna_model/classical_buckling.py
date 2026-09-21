"""
classical_buckling.py

Closed-form classical-plate-theory estimate of stiffened-panel buckling,
used two ways:
  1. As the independent hand-calc cross-check for the LS-DYNA pipeline
     (same role yield_stress*2*pi*R*t played for the crush-tube baseline).
  2. As the physics behind --mock synthetic data in run_doe_lsdyna.py, so
     the rest of the pipeline (resumable CSV writing, surrogate training,
     optimizers) can be smoke-tested without a real LS-DYNA install.

Model: initial buckling of a stiffened panel is very often governed by
LOCAL buckling of the skin sub-panel between adjacent stringers, not by
overall panel buckling -- so this treats each stringer bay as an
independent simply-supported flat plate of width b = Wb/(n_str+1) under
uniaxial compression:

    sigma_cr = k * pi^2 * E / (12*(1-nu^2)) * (t/b)^2      [classical plate buckling]
    N_cr     = sigma_cr * t                                 [running load, N per mm width]

with k = 4.0, the standard buckling coefficient for a long (aspect ratio
>~3) simply-supported plate loaded on two opposite edges. This deliberately
ignores overall-panel and stringer-flexural buckling modes (which a real
eigenvalue extraction captures and this hand-calc doesn't) -- it's a sanity
floor, not a replacement for the real analysis. See BUILD_GUIDE.md.
"""

import math

from deck_builder import load_config, mass_kg  # noqa: F401 (mass_kg re-exported for convenience)


K_BUCKLING_COEFF = 4.0  # simply-supported long plate, uniaxial compression

PRELOAD_SAFETY_FACTOR = 0.02
"""
How far below the hand-calc-estimated critical load to keep the LS-DYNA
nonlinear implicit *preload* step's reference load Pref.

Why this exists at all (found from a real solve, not anticipated up
front): a single fixed Pref across the whole DOE cannot work, because the
design space spans roughly a 150x range of actual critical loads (see this
file's own __main__ block -- N_cr from ~15 to ~2200+ N/mm depending on
t_skin/n_str). A fixed Pref that's comfortably subcritical for the
strongest designs is wildly supercritical for the weakest ones, and vice
versa. Confirmed concretely: with a fixed Pref=100,000N, a mid-range point
whose hand-calc estimate was ~215.9 N/mm (running load) x 400mm width =
~86,360N total -- LESS than Pref -- came back with a buckling eigenvalue
of -0.80 instead of a sane positive value. The preload step is a NONLINEAR
implicit static solve; asking it to equilibrate at a load already past (or
very near) the panel's actual buckling point pushes it into large-
deflection/near-bifurcation behavior, which breaks the small-deformation
"elastic reference state" assumption the subsequent linear eigenvalue
extraction depends on -- hence the sign flip and off-magnitude result,
independent of the load-direction fix.

Fix: scale Pref per design point off this file's own classical hand-calc
estimate (subpanel_running_load_N_per_mm), at a safety factor comfortably
below 1.0, so the preload step stays elastic/subcritical for every point
in the DOE automatically.

Tuning history, each step driven by a real result, not guessed ahead of
time:
  - 0.15 (first draft): a 60-point sweep came back with 5 points (8.3%)
    still negative.
  - 0.06: reduced the 140-point follow-on sweep's failure rate to 3/140
    (2.1%) -- confirms this lever works, just needed more margin. Checked
    several single-variable theories for what distinguishes failing points
    (h_str/t_str slenderness, t_str alone, stringer-to-skin
    cross-sectional area ratio) against the full result set each time --
    none cleanly separate them. Expected, not a dead end:
    `subpanel_running_load_N_per_mm()` is a SKIN-ONLY local-buckling
    estimate -- it doesn't depend on h_str/t_str at all -- so it can't
    "see" a stringer-flexural or overall-panel mode governing instead, and
    which geometries trigger that isn't reducible to one simple ratio
    (out of scope for a hand-calc whose only job is sizing Pref, not
    replacing the eigenvalue extraction).
  - 0.02: the 3 points still failing at 0.06 came back with lambda_1
    around -7 to -16 -- deeply negative, not just barely below zero,
    meaning Pref was still several times past their true critical load.
    Dropped further for a retry via --retry-failed.

On mixing safety factors across a dataset (this constant has already
changed twice over the life of this DOE): that's fine, not a
data-integrity problem, because PRELOAD_SAFETY_FACTOR is a purely
NUMERICAL knob controlling how the preload step is set up -- it has no
bearing on the physical meaning of a result once solved. Two points solved
under different factor values are still directly comparable, AS LONG AS
each one's own preload stayed subcritical (which is exactly what
solve_one_point_lsdyna()'s non-positive-lambda_1 check enforces per point,
retrying under a note in failed_points.json otherwise).

The "opposite failure mode" flagged above as a hypothetical -- a preload
so small its forces sit near the solver's numerical noise floor -- did
show up for real, just not from lowering this constant further. It hit an
untested CORNER of the design space instead: an infill batch concentrated
at thin skin (t_skin ~1.2-1.3mm) with n_str=6 (narrow bay, low hand-calc
N_cr ~150-190 N/mm) produced 3 points (201, 203, 214) where LS-DYNA's own
eigensolver refused to trust its result and errored explicitly: "Error
60419: No trusted eigenvalues were computed... Numerical problems may be
caused by too low initial loading for buckling." At those points, 0.02x
the (already low) hand-calc estimate gave a total preload of only
~1200-1450 N -- too small in absolute terms for the solver, independent of
whether it's "correctly" subcritical. Retried with a temporary, per-point
override (doe/retry_low_preload_points.py; does NOT change this global
default) at 0.06 -- the same value already validated across the other
137/140 points of the original sweep -- and all 3 solved cleanly on the
first try (N_cr = 194.7 / 187.5 / 159.9 N/mm respectively), confirming
0.06 as the fix for this corner without needing to explore all the way up
to 0.10.

Net effect: PRELOAD_SAFETY_FACTOR stays 0.02 as the global default (it's
correct for the 211 already-solved points at that value), but 0.02 is now
known to be too aggressive specifically for thin-skin + high-n_str
(narrow-bay) designs -- anyone extending the DOE further into that corner
should expect to need something closer to 0.06 there, and
doe/retry_low_preload_points.py is the documented, tested way to apply
that per-point rather than by editing this constant.
"""


def subpanel_running_load_N_per_mm(t_skin_mm, n_str, config):
    n_str = int(round(n_str))
    fixed = config["fixed_parameters"]
    mat = config["material"]
    Wb = fixed["panel_width_Wb_mm"]
    E_MPa = mat["youngs_modulus_GPa"] * 1000.0
    nu = mat["poissons_ratio"]

    b_mm = Wb / (n_str + 1)
    sigma_cr_MPa = (K_BUCKLING_COEFF * math.pi ** 2 * E_MPa
                     / (12 * (1 - nu ** 2)) * (t_skin_mm / b_mm) ** 2)
    N_cr_per_mm = sigma_cr_MPa * t_skin_mm
    return N_cr_per_mm


def reference_load_Pref_N(t_skin_mm, n_str, config):
    """Per-design-point LS-DYNA preload reference load (total force, N),
    scaled off the classical hand-calc so the nonlinear preload step stays
    safely subcritical across the whole DOE. See PRELOAD_SAFETY_FACTOR's
    docstring above for why a single fixed Pref doesn't work. Used by both
    deck_builder.build_deck() (to actually apply the load) and
    run_doe_lsdyna.py (to convert the returned eigenvalue back into a
    physical critical load) -- both must call this same function so the
    Pref that was actually solved against always matches the Pref used to
    interpret the result."""
    Wb = config["fixed_parameters"]["panel_width_Wb_mm"]
    estimate_N_cr_per_mm = subpanel_running_load_N_per_mm(t_skin_mm, n_str, config)
    return PRELOAD_SAFETY_FACTOR * estimate_N_cr_per_mm * Wb


if __name__ == "__main__":
    config = load_config()
    print("Sub-panel classical buckling estimate vs. design target "
          f"({config['targets']['design_running_load_N_per_mm']} N/mm):\n")
    for t_skin, n_str, h_str, t_str in [
        (1.0, 2, 10.0, 1.0),
        (2.0, 4, 25.0, 2.0),
        (3.0, 6, 40.0, 3.0),
        (1.6, 3, 20.0, 1.8),
    ]:
        N_cr = subpanel_running_load_N_per_mm(t_skin, n_str, config)
        m = mass_kg(t_skin, n_str, h_str, t_str, config)
        meets = "MEETS" if N_cr >= config["targets"]["design_running_load_N_per_mm"] else "below"
        print(f"t_skin={t_skin:.2f}mm n_str={n_str} h_str={h_str:.1f}mm t_str={t_str:.2f}mm  "
              f"-> N_cr~={N_cr:7.1f} N/mm ({meets} target)  mass={m:.3f}kg")
