"""
run_doe_lsdyna.py

For each row of results/doe_points_panel.csv: build a parametrized LS-DYNA
deck (ls_dyna_model/deck_builder.py, now on PyDYNA's Keywords API), run it
via PyDYNA's `run_dyna()` (ansys.dyna.core.run), parse the buckling
eigenvalue out of the solver's text output, and combine with an
analytically-computed mass to get running-load-vs-mass data for the
optimizer stage.

Why PyDYNA for building/running but plain text parsing for eigenvalues:
building the deck and running the solver are both purely local operations
in PyDYNA (confirmed against an installed ansys-dyna-core -- no server, no
Docker). `run_dyna()` is called with an explicit `executable=` kwarg
pointing at panel_config.json's solver.executable, rather than relying on
`save-ansys-path`/ansys-tools-path auto-discovery -- confirmed by reading
PyDYNA's own source (ansys/dyna/core/run/windows_runner.py) that on
Windows the auto-discovery path is never consulted at all (only Linux's
runner checks it), so passing the path explicitly is both simpler and the
only option that actually works on both platforms. Ansys's official result
reader (PyDPF) exists too, but it works by talking to a small local DPF
server process it starts for you -- for pulling out a single scalar (the
buckling load factor) that's an extra moving part not worth adding yet, so
this still greps it straight out of the solver's own text printout, same
as the first draft.

Structural note vs. the crush-tube project's COMSOL pipeline: there is NO
persistent server/client connection here -- each point is one independent,
self-contained solve. That structurally eliminates the entire class of
"connection died, cascaded through the rest of the DOE" failures the
COMSOL pipeline had; a failure on one point is fully isolated from every
other point.

PREREQUISITE not yet confirmed as of this writing: exactly where your
LS-DYNA version writes the eigenvalue results (see
_parse_eigenvalue_from_d3hsp below and BUILD_GUIDE.md). Run
doe/test_single_point_lsdyna.py FIRST, on one point, before attempting a
full sweep -- paste back whatever the real output looks like (or whatever
error run_dyna raises) and we'll fix the parser/deck together.

Usage:
    python doe/run_doe_lsdyna.py                  # real run, needs LS-DYNA
    python doe/run_doe_lsdyna.py --mock            # synthetic data (classical_buckling.py), no LS-DYNA needed
    python doe/run_doe_lsdyna.py --mock --n 5      # quick pipeline smoke test
    python doe/run_doe_lsdyna.py --retry-failed    # re-attempt previously-failed points only
"""

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ls_dyna_model"))

from deck_builder import load_config, build_deck, mass_kg  # noqa: E402
from classical_buckling import subpanel_running_load_N_per_mm, reference_load_Pref_N  # noqa: E402


SOLVER_TIMEOUT_S = 600  # generous vs. the expected few-seconds-to-tens-of-seconds
                        # eigenvalue solve -- a real hang (not just "slow") should
                        # still trip this rather than block the whole DOE forever.
                        # NOTE: run_dyna() is a blocking call with no timeout
                        # parameter of its own, so this is enforced from the Python
                        # side via a worker thread -- if it fires, the underlying
                        # LS-DYNA process may still be running in the background;
                        # check Task Manager if timeouts start happening repeatedly.


def mock_solve_one_point(t_skin, n_str, h_str, t_str, config):
    """Synthetic stand-in for a real LS-DYNA solve, used only to smoke-test
    the rest of the pipeline (CSV writing, resumability, surrogate/optimizer
    scripts) before real DOE data exists. Built on the same classical-plate
    formula used as the independent hand-calc cross-check -- NOT a
    replacement for a real eigenvalue extraction (it ignores overall-panel
    and stringer-flexural buckling modes), do not use these numbers for
    anything except pipeline verification."""
    import random
    N_cr = subpanel_running_load_N_per_mm(t_skin, n_str, config)
    # small synthetic scatter so mock data isn't perfectly deterministic-smooth
    rng = random.Random(hash((round(t_skin, 4), n_str, round(h_str, 4), round(t_str, 4))))
    N_cr *= rng.uniform(0.95, 1.05)
    Pref = reference_load_Pref_N(t_skin, n_str, config)
    lambda_1 = N_cr * config["fixed_parameters"]["panel_width_Wb_mm"] / Pref
    return {"lambda_1": lambda_1, "N_cr_per_mm": N_cr}


def _parse_eigenvalue_from_eigout(eigout_path):
    """
    Confirmed against a real solve (Student R16.1, Windows): LS-DYNA does
    NOT write the buckling eigenvalues into d3hsp at all for this version --
    they go to a separate, dedicated text file called `eigout` (d3hsp only
    logs the line "write eigout file" as it's produced). eigout's real
    format, copied verbatim from an actual run:

        r e s u l t s   o f   e i g e n v a l u e   a n a l y s i s:
        problem time =  1.00000E+00
        (all frequencies de-shifted)
                                     |------ frequency -----|
                MODE    EIGENVALUE       RADIANS        CYCLES        PERIOD
                   1  -5.385213E-01  -8.570833E-02
                   2  -8.161411E-01  -1.298929E-01
                   3  -9.005137E-01  -1.433212E-01

         EIGENVECTOR SCALING FACTOR
                   1   1.853499E-03
                   ...

    Mode 1 (first row after the "MODE  EIGENVALUE ..." header) is the
    lowest/first-extracted buckling mode -- that's lambda_1.
    """
    text = Path(eigout_path).read_text(errors="replace")
    header_match = re.search(r"MODE\s+EIGENVALUE", text)
    if not header_match:
        raise RuntimeError(
            f"Could not find the 'MODE    EIGENVALUE' results header in {eigout_path} -- "
            "the file format may differ from what was confirmed on 2026-09-21. Paste the "
            "file's contents back so we can adjust the parser."
        )
    table_text = text[header_match.end():]
    rows = []
    for line in table_text.splitlines():
        line = line.strip()
        if not line:
            if rows:
                break  # blank line after we've already collected rows ends the table
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            mode = int(parts[0])
            eigenvalue = float(parts[1])
        except ValueError:
            continue
        rows.append((mode, eigenvalue))
    if not rows:
        raise RuntimeError(
            f"Found the results header in {eigout_path} but no parseable mode/eigenvalue "
            "rows under it -- paste the file's contents back so we can adjust the parser."
        )
    rows.sort(key=lambda r: r[0])
    return rows[0][1]


def _run_lsdyna_windows_direct(executable, deck_path, point_dir, ncpu=1, memory_mb=256):
    """Launch the solver directly with subprocess, bypassing PyDYNA's own
    WindowsRunner.

    PyDYNA's WindowsRunner._get_env_script() (ansys/dyna/core/run/
    windows_runner.py) does `os.listdir(solver_folder)` hunting for a
    sibling entry with "lsprepost" in its name, to `call` an MPI env-setup
    .bat script before launching the solver -- it assumes a full Ansys
    Unified install layout where the solver and an `lsprepost` folder sit
    side by side under the same parent. A standalone LS-DYNA Student Suite
    install doesn't have that layout (`lsdyna/` and `lspp/` are siblings of
    each other, not nested), so that lookup throws
    `IndexError: list index out of range` before the solver is ever
    launched -- confirmed against a real Student R16.1 install. This
    happens on every Windows Student install with this folder layout, not
    just this one, so it's a real upstream limitation, not a config
    mistake.

    Workaround: skip run_dyna()/WindowsRunner for the actual "launch the
    exe" step and invoke the solver directly with the same i=/ncpu=/
    memory= command-line convention run_dyna() itself uses -- the SMP
    solver doesn't need that env-setup batch script at all for a plain
    single-machine solve. Deck building still goes entirely through
    PyDYNA (build_deck() above); this only replaces the OS-level launch.
    """
    log_path = point_dir / "lsrun.out.txt"
    cmd = [executable, f"i={deck_path.name}", f"ncpu={ncpu}", f"memory={memory_mb}m"]
    with open(log_path, "w") as log_f:
        result = subprocess.run(
            cmd, cwd=str(point_dir), stdout=log_f, stderr=subprocess.STDOUT,
        )
    if result.returncode != 0:
        tail = log_path.read_text(errors="replace")[-1500:]
        raise RuntimeError(
            f"LS-DYNA exited with code {result.returncode}. Log tail "
            f"({log_path}):\n{tail}"
        )


def solve_one_point_lsdyna(point_id, t_skin, n_str, h_str, t_str, config, work_root):
    """Build this point's deck with PyDYNA and solve it.

    NOTE on solver discovery: run_dyna()'s auto-discovery path
    (ansys-tools-path / `save-ansys-path`) is only consulted by PyDYNA's
    *Linux* runner -- on Windows, WindowsRunner._find_solver() never calls
    get_dyna_path() at all, it only honors an explicit `executable=` kwarg
    (or falls back to hunting for a full Ansys Unified install, which a
    Student LS-DYNA-only install is not). Confirmed by reading
    ansys/dyna/core/run/windows_runner.py directly rather than assuming
    the two platforms behave the same. So this always passes
    config["solver"]["executable"] through explicitly -- simpler than
    save-ansys-path and works the same on both platforms.

    NOTE on Windows specifically: see _run_lsdyna_windows_direct() above --
    PyDYNA's own run_dyna() cannot launch the solver at all on a Student
    Suite install (a confirmed upstream bug against this folder layout), so
    on Windows this bypasses run_dyna() for just the launch step and calls
    the solver directly. Linux is unaffected by that bug and still goes
    through run_dyna() normally.
    """
    executable = config["solver"]["executable"]
    if executable.startswith("CONFIRM"):
        raise RuntimeError(
            "panel_config.json's solver.executable is still a placeholder -- fill in "
            "the full path to your Student install's lsdyna_sp.exe (or lsdyna_dp.exe) "
            "first (see BUILD_GUIDE.md)."
        )
    if not Path(executable).is_file():
        raise RuntimeError(
            f"panel_config.json's solver.executable ({executable!r}) does not point to "
            "a real file. Double check the path (Start Menu shortcut properties usually "
            "show the real command line) and update panel_config.json."
        )

    point_dir = work_root / f"point_{point_id}"
    point_dir.mkdir(parents=True, exist_ok=True)
    deck_path = point_dir / "panel.k"
    deck = build_deck(point_id, t_skin, n_str, h_str, t_str, config)
    deck.export_file(str(deck_path))

    def _run():
        if sys.platform.startswith("win"):
            return _run_lsdyna_windows_direct(executable, deck_path, point_dir)
        from ansys.dyna.core.run import run_dyna
        return run_dyna(
            str(deck_path),
            working_directory=str(point_dir),
            executable=executable,
        )

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(_run)
        try:
            future.result(timeout=SOLVER_TIMEOUT_S)
        except FutureTimeoutError:
            raise RuntimeError(
                f"LS-DYNA did not finish within {SOLVER_TIMEOUT_S}s -- the process may "
                f"still be running in {point_dir}; check Task Manager. If solves are "
                f"routinely this slow, raise SOLVER_TIMEOUT_S in this file."
            )
        # run_dyna() itself raises RuntimeError on a nonzero solver exit code
        # (see linux_runner.py / windows_runner.py) and FileNotFoundError if
        # the executable path is bad -- both propagate out of future.result()
        # above and are caught by run()'s try/except, so nothing further to
        # translate here.

    eigout_path = point_dir / "eigout"
    if not eigout_path.exists():
        raise RuntimeError(
            f"Solver returned success but {eigout_path} was not produced -- "
            f"check panel_config.json's output.likely_location note."
        )
    lambda_1 = _parse_eigenvalue_from_eigout(eigout_path)

    if lambda_1 <= 0:
        # NOT a parsing bug -- confirmed on a real 60-point DOE sweep, 5/60
        # points (8.3%) came back with a negative lambda_1 that "succeeded"
        # (no exception, wrote a nonphysical negative N_cr straight into the
        # results CSV as if it were valid data) before this check existed.
        # A negative eigenvalue here means the nonlinear preload step
        # crossed the panel's TRUE critical load before the eigenvalue
        # extraction ran -- almost certainly because the true critical load
        # is governed by a stringer-flexural or overall-panel mode that
        # classical_buckling.py's skin-only hand-calc (and therefore Pref,
        # which is sized off of it) can't see. See
        # classical_buckling.PRELOAD_SAFETY_FACTOR's docstring for the full
        # story and why this is treated as a hard failure (goes to
        # failed_points.json for retry) rather than silently recorded --
        # letting bad data into the results CSV unflagged is exactly the
        # class of bug that burned the crush-tube/COMSOL project earlier.
        raise RuntimeError(
            f"Non-positive lambda_1 ({lambda_1:.4f}) -- the preload step likely "
            f"crossed this design's true critical load before the eigenvalue "
            f"extraction ran (see PRELOAD_SAFETY_FACTOR's docstring in "
            f"classical_buckling.py). Treating as a failed point rather than "
            f"recording nonphysical data."
        )

    # Must be the exact same Pref build_deck() actually applied for this
    # point (per-point now, not a fixed config value -- see
    # classical_buckling.reference_load_Pref_N()'s docstring), otherwise
    # lambda_1 gets converted back with the wrong reference load.
    Pref = reference_load_Pref_N(t_skin, n_str, config)
    Wb = config["fixed_parameters"]["panel_width_Wb_mm"]
    critical_load_N = lambda_1 * Pref
    N_cr_per_mm = critical_load_N / Wb
    return {"lambda_1": lambda_1, "N_cr_per_mm": N_cr_per_mm}


def load_existing(out_path, failed_path, fieldnames):
    """Loads already-recorded results/failures, and self-heals the results
    CSV: solve_one_point_lsdyna() now treats a non-positive N_cr_per_mm as a
    hard failure (see its docstring for why), but any *older* results file
    written before that check existed can still have nonphysical negative
    rows sitting in it as if they were valid data (this happened for real --
    5/60 points on the first full DOE sweep). Any such row found here is
    stripped out of out_path and excluded from done_ids, so it's picked
    back up and re-solved automatically on this run -- no manual CSV
    editing needed."""
    done_ids = set()
    failed_records = []
    purged_point_ids = []
    if out_path.exists():
        with open(out_path) as f:
            rows = list(csv.DictReader(f))
        clean_rows = []
        for row in rows:
            if float(row["N_cr_per_mm"]) <= 0:
                purged_point_ids.append(row["point_id"])
                continue
            clean_rows.append(row)
            done_ids.add(row["point_id"])
        if purged_point_ids:
            with open(out_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(clean_rows)
    if failed_path.exists():
        with open(failed_path) as f:
            failed_records = json.load(f)
        for rec in failed_records:
            done_ids.add(rec["point_id"])
    return done_ids, failed_records, purged_point_ids


def run(mock, n_limit, retry_failed, keep_work_dirs):
    config = load_config()
    doe_path = ROOT / "results" / "doe_points_panel.csv"
    with open(doe_path) as f:
        rows = list(csv.DictReader(f))
    if n_limit is not None:
        rows = rows[:n_limit]

    out_path = ROOT / "results" / ("doe_results_panel_mock.csv" if mock else "doe_results_panel.csv")
    failed_path = ROOT / "results" / ("doe_failed_points_panel_mock.json" if mock else "doe_failed_points_panel.json")
    fieldnames = ["point_id", "t_skin", "n_str", "h_str", "t_str",
                  "mass_kg", "lambda_1", "N_cr_per_mm", "meets_target"]

    done_ids, failed_records, purged_point_ids = load_existing(out_path, failed_path, fieldnames)
    if purged_point_ids:
        print(f"Purged {len(purged_point_ids)} nonphysical (non-positive N_cr) "
              f"row(s) from {out_path.name} recorded before this check existed -- "
              f"will re-solve: {', '.join(purged_point_ids)}")

    def write_failed_records():
        with open(failed_path, "w") as jf:
            json.dump(failed_records, jf, indent=2)

    if retry_failed:
        previously_failed_ids = {r["point_id"] for r in failed_records}
        done_ids -= previously_failed_ids
        failed_records = [r for r in failed_records if r["point_id"] not in previously_failed_ids]
        write_failed_records()
    rows_to_run = [r for r in rows if r["point_id"] not in done_ids]

    skipped = len(rows) - len(rows_to_run)
    if skipped:
        print(f"Resuming: skipping {skipped} point(s) already recorded.")
    if not rows_to_run:
        print("Nothing to do -- every point already has a recorded result. "
              "Pass --retry-failed to re-attempt previously-failed points.")
        return

    work_root = ROOT / "results" / ("lsdyna_work_mock" if mock else "lsdyna_work")
    work_root.mkdir(exist_ok=True)

    file_mode = "a" if out_path.exists() else "w"
    with open(out_path, file_mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if file_mode == "w":
            writer.writeheader()
        f.flush()

        for row in rows_to_run:
            point_id = row["point_id"]
            t_skin = float(row["t_skin"])
            n_str = int(row["n_str"])
            h_str = float(row["h_str"])
            t_str = float(row["t_str"])
            print(f"[{point_id}] t_skin={t_skin:.3f} n_str={n_str} h_str={h_str:.2f} "
                  f"t_str={t_str:.3f} ...", end=" ")

            try:
                if mock:
                    sol = mock_solve_one_point(t_skin, n_str, h_str, t_str, config)
                else:
                    sol = solve_one_point_lsdyna(point_id, t_skin, n_str, h_str, t_str,
                                                  config, work_root)
                m = mass_kg(t_skin, n_str, h_str, t_str, config)
                meets = sol["N_cr_per_mm"] >= config["targets"]["design_running_load_N_per_mm"]
                print(f"N_cr={sol['N_cr_per_mm']:.1f} N/mm, mass={m:.3f}kg, "
                      f"{'MEETS' if meets else 'below'} target")
                writer.writerow({
                    "point_id": point_id, "t_skin": t_skin, "n_str": n_str,
                    "h_str": h_str, "t_str": t_str,
                    "mass_kg": f"{m:.5f}",
                    "lambda_1": f"{sol['lambda_1']:.6f}",
                    "N_cr_per_mm": f"{sol['N_cr_per_mm']:.4f}",
                    "meets_target": meets,
                })
                f.flush()
                if not mock and not keep_work_dirs:
                    shutil.rmtree(work_root / f"point_{point_id}", ignore_errors=True)
            except Exception as exc:  # noqa: BLE001 -- log whatever the solver/parser raises
                print(f"FAILED -- {str(exc).splitlines()[0].strip()}")
                failed_records.append({
                    "point_id": point_id, "t_skin": t_skin, "n_str": n_str,
                    "h_str": h_str, "t_str": t_str, "error": str(exc),
                })
                write_failed_records()

    write_failed_records()
    print(f"\nWrote {out_path}")
    if failed_records:
        print(f"{len(failed_records)} point(s) failed this run -- see {failed_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--keep-work-dirs", action="store_true",
                         help="Don't delete each point's LS-DYNA working directory after a "
                              "successful solve (useful while debugging the deck/parser).")
    args = parser.parse_args()
    run(mock=args.mock, n_limit=args.n, retry_failed=args.retry_failed,
        keep_work_dirs=args.keep_work_dirs)


if __name__ == "__main__":
    main()
