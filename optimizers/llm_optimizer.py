"""
llm_optimizer.py

OPTIONAL second optimizer, for comparison against the guaranteed-runnable
classical/metaheuristic one (optimizers/classical_optimizer.py). Uses an
LLM (Google Gemini) as an iterative design-search agent instead of
differential evolution: each round, the model sees the full history of
designs tried so far (parameters, mass, surrogate-predicted buckling
capacity, feasibility) and proposes the next candidate to try, in plain
JSON. This script scores that candidate with the same surrogate model and
analytical mass function the classical optimizer uses -- the LLM never
sees or touches LS-DYNA directly, it's playing the same "propose a design
point, get scored" game an engineer iterating by hand would.

Why Gemini, not Claude, for this script specifically: the Gemini API
(ai.google.dev) has a genuinely free tier -- no credit card required to
get a key -- unlike the Claude API's pay-as-you-go billing, and this
optimizer's whole point (an *optional*, zero-friction extra data point
for the comparison plot) is undermined if running it costs money. Default
model here, `gemini-3.5-flash-lite`, is free-tier eligible and plenty
capable for proposing 4 numbers in bounds each round. (An earlier draft
defaulted to `gemini-2.5-flash-lite` -- Google retired that model for new
API users partway through this project; if this script ever starts
failing with a 404 NOT_FOUND naming a specific model, that's what's
happening again, and the fix is just passing --model with whatever
current model the error message names, e.g.
`python optimizers/llm_optimizer.py --model gemini-<current>-flash-lite`.)

One tradeoff worth knowing about before using this: Google's free-tier
terms permit using submitted prompts/responses to improve their products
(this doesn't apply to their paid tier). The data here is just 4 generic
geometry numbers per round, nothing sensitive -- but worth being aware of
as a general policy if you reuse this pattern elsewhere.

This requires a Gemini API key and is kept separate from the guaranteed-
free classical optimizer; it degrades gracefully -- with clear setup
instructions, not a traceback -- if no key is configured. Skipping this
script entirely does not block the rest of the pipeline: compare_optimizers.py
works fine with only the classical optimizer's result.

Setup (only needed to run this specific script, and it's free):
    1. Go to https://aistudio.google.com/apikey and sign in with a Google
       account, then click "Create API key". No credit card needed for
       the free tier. (Not available in every region -- e.g. the UK/EEA/
       Switzerland currently have restrictions; check Google's "available
       regions" docs if key creation is blocked for your account.)
    2. export GEMINI_API_KEY=...
    3. pip install google-genai --break-system-packages   (if not already installed)

Usage:
    python optimizers/llm_optimizer.py
    python optimizers/llm_optimizer.py --rounds 15 --model gemini-3.1-flash-lite
"""

# NOTE (see module docstring): Google's model lineup moves fast and older
# model IDs get retired for new API users without much notice. If this
# default 404s, check https://ai.google.dev/gemini-api/docs/models for the
# current cheap/fast "flash-lite"-class model name and pass it via --model.

import argparse
import json
import os
import re
import sys
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ls_dyna_model"))

from deck_builder import load_config, mass_kg  # noqa: E402

DEFAULT_MODEL = "gemini-3.5-flash-lite"
DEFAULT_ROUNDS = 12


def check_prerequisites():
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print(
            "No GEMINI_API_KEY (or GOOGLE_API_KEY) found in the environment -- skipping "
            "the LLM-driven optimizer (this is expected/optional, not an error).\n\n"
            "This script is not required for the rest of the project: the "
            "classical/metaheuristic optimizer (optimizers/classical_optimizer.py) "
            "already produces a complete, guaranteed-free result, and "
            "compare_optimizers.py works fine with just that.\n\n"
            "To run this optimizer later (free, no credit card needed):\n"
            "  1. Go to https://aistudio.google.com/apikey, sign in with a Google\n"
            "     account, and click 'Create API key'.\n"
            "  2. export GEMINI_API_KEY=...\n"
            "  3. pip install google-genai --break-system-packages   (if not already installed)\n"
            "  4. python optimizers/llm_optimizer.py\n"
        )
        return None

    try:
        from google import genai  # noqa: F401
    except ImportError:
        print(
            "GEMINI_API_KEY is set, but the `google-genai` package isn't installed.\n"
            "Install it with:\n"
            "  pip install google-genai --break-system-packages\n"
        )
        return None

    return api_key


def load_surrogate(model_path):
    if not model_path.exists():
        raise RuntimeError(
            f"No surrogate model found at {model_path} -- run "
            f"`python surrogate/train_surrogate.py` first."
        )
    bundle = joblib.load(model_path)
    return bundle["pipeline"]


def evaluate_design(design, config, surrogate):
    t_skin, n_str, h_str, t_str = (
        float(design["t_skin"]), int(round(design["n_str"])),
        float(design["h_str"]), float(design["t_str"]),
    )
    m = mass_kg(t_skin, n_str, h_str, t_str, config)
    X = np.array([[t_skin, n_str, h_str, t_str]])
    pred, std = surrogate.predict(X, return_std=True)
    return {
        "t_skin": t_skin, "n_str": n_str, "h_str": h_str, "t_str": t_str,
        "mass_kg": float(m),
        "predicted_N_cr_per_mm": float(pred[0]),
        "predicted_std_N_cr_per_mm": float(std[0]),
    }


def clamp_to_bounds(design, dp):
    for name in ["t_skin", "n_str", "h_str", "t_str"]:
        lo, hi = dp[name]["bounds"]
        design[name] = min(max(design[name], lo), hi)
    return design


SYSTEM_PROMPT = """You are optimizing a stiffened aircraft panel design to MINIMIZE MASS \
subject to a minimum buckling load constraint. You will propose one candidate design per \
turn; after each proposal you will be told its exact mass and its predicted buckling \
capacity so you can refine your next guess. This is a real engineering optimization loop, \
not a one-shot guess -- use the growing history to reason about trends (e.g. "increasing \
n_str let me reduce t_skin while staying feasible") the way an engineer iterating by hand \
would.

Respond with ONLY a JSON object of the form:
{"t_skin": <float>, "n_str": <int>, "h_str": <float>, "t_str": <float>, "reasoning": "<one \
short sentence>"}

No other text, no markdown code fences."""


def build_user_prompt(config, history, best_feasible):
    dp = config["design_parameters"]
    target = config["targets"]["design_running_load_N_per_mm"]
    lines = [
        f"Design variables and bounds:",
        f"  t_skin (mm): [{dp['t_skin']['bounds'][0]}, {dp['t_skin']['bounds'][1]}]",
        f"  n_str (integer count): [{dp['n_str']['bounds'][0]}, {dp['n_str']['bounds'][1]}]",
        f"  h_str (mm): [{dp['h_str']['bounds'][0]}, {dp['h_str']['bounds'][1]}]",
        f"  t_str (mm): [{dp['t_str']['bounds'][0]}, {dp['t_str']['bounds'][1]}]",
        f"",
        f"Constraint: predicted_N_cr_per_mm >= {target} (this is a surrogate model's "
        f"prediction of the panel's buckling critical running load, trained on real "
        f"LS-DYNA eigenvalue-buckling solves).",
        f"Objective: minimize mass_kg.",
        f"",
    ]
    if not history:
        lines.append(
            "No designs tried yet. Propose a reasonable first guess anywhere in bounds."
        )
    else:
        lines.append(f"History so far ({len(history)} designs tried):")
        for i, h in enumerate(history):
            feasible = "FEASIBLE" if h["predicted_N_cr_per_mm"] >= target else "infeasible"
            lines.append(
                f"  {i+1}. t_skin={h['t_skin']:.4f} n_str={h['n_str']} "
                f"h_str={h['h_str']:.3f} t_str={h['t_str']:.4f}  "
                f"-> mass={h['mass_kg']:.4f}kg, predicted_N_cr={h['predicted_N_cr_per_mm']:.1f} "
                f"N/mm [{feasible}]"
            )
        if best_feasible:
            lines.append(
                f"\nBest FEASIBLE so far: mass={best_feasible['mass_kg']:.4f}kg "
                f"(t_skin={best_feasible['t_skin']:.4f}, n_str={best_feasible['n_str']}, "
                f"h_str={best_feasible['h_str']:.3f}, t_str={best_feasible['t_str']:.4f})"
            )
        else:
            lines.append("\nNo feasible design found yet -- prioritize finding one.")
    lines.append(
        "\nPropose your next candidate design as JSON now, trying to beat the best "
        "feasible mass found so far (or find a first feasible design if none exists yet)."
    )
    return "\n".join(lines)


def parse_llm_json(text):
    text = text.strip()
    # Strip markdown code fences if the model added them despite instructions.
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    api_key = check_prerequisites()
    if api_key is None:
        sys.exit(0)  # graceful no-op, not an error -- see check_prerequisites()

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)

    config = load_config()
    dp = config["design_parameters"]
    target = config["targets"]["design_running_load_N_per_mm"]
    surrogate = load_surrogate(ROOT / "surrogate" / "surrogate_model.joblib")

    print(f"Running LLM-driven optimization: {args.rounds} rounds, model={args.model}\n")

    history = []
    best_feasible = None

    for round_i in range(args.rounds):
        user_prompt = build_user_prompt(config, history, best_feasible)
        try:
            response = client.models.generate_content(
                model=args.model,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    max_output_tokens=300,
                ),
            )
        except Exception as e:
            # Deliberately broad: this is a network I/O boundary (API auth errors,
            # rate limits, proxy/connection failures, timeouts all land here), and
            # the whole point of this being an *optional* script is that any of
            # those should end with a clear message and whatever partial history
            # was collected, not a stack trace.
            print(f"Round {round_i+1}: Gemini API call failed ({type(e).__name__}: {e}) "
                  f"-- stopping early with whatever history has been collected so far.")
            break

        raw_text = (response.text or "").strip()
        try:
            proposal = parse_llm_json(raw_text)
            design = {k: proposal[k] for k in ["t_skin", "n_str", "h_str", "t_str"]}
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"Round {round_i+1}: couldn't parse a valid design from the model's "
                  f"response ({e}) -- raw response was: {raw_text!r} -- skipping this round.")
            continue

        design = clamp_to_bounds(design, dp)
        result = evaluate_design(design, config, surrogate)
        history.append(result)

        feasible = result["predicted_N_cr_per_mm"] >= target
        if feasible and (best_feasible is None or result["mass_kg"] < best_feasible["mass_kg"]):
            best_feasible = result

        reasoning = proposal.get("reasoning", "")
        status = "FEASIBLE" if feasible else "infeasible"
        print(f"Round {round_i+1}/{args.rounds}: t_skin={result['t_skin']:.4f} "
              f"n_str={result['n_str']} h_str={result['h_str']:.3f} "
              f"t_str={result['t_str']:.4f} -> mass={result['mass_kg']:.4f}kg "
              f"pred_N_cr={result['predicted_N_cr_per_mm']:.1f} [{status}]"
              + (f"  ({reasoning})" if reasoning else ""))

    print()
    if best_feasible is None:
        print("No feasible design found across all rounds -- not writing a result file.")
        sys.exit(1)

    print(f"Best feasible design found by the LLM optimizer across {len(history)} rounds:")
    print(f"  mass       = {best_feasible['mass_kg']:.4f} kg")
    print(f"  t_skin     = {best_feasible['t_skin']:.4f} mm")
    print(f"  n_str      = {best_feasible['n_str']}")
    print(f"  h_str      = {best_feasible['h_str']:.4f} mm")
    print(f"  t_str      = {best_feasible['t_str']:.4f} mm")
    print(f"  surrogate-predicted N_cr = {best_feasible['predicted_N_cr_per_mm']:.1f} N/mm "
          f"vs target {target} N/mm")

    out = {
        "target_N_per_mm": target,
        "model": args.model,
        "n_rounds_requested": args.rounds,
        "n_rounds_completed": len(history),
        "best": best_feasible,
        "history": history,
    }
    out_path = ROOT / "optimizers" / "llm_optimizer_result.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved full result to {out_path}")


if __name__ == "__main__":
    main()
