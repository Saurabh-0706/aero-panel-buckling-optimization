"""
deck_builder.py

Builds the mesh (nodes + shell elements for skin and blade stringers) and
renders a full LS-DYNA deck for one stiffened-panel design point, from
(t_skin, n_str, h_str, t_str) -- now built on PyDYNA (`ansys-dyna-core`)'s
structured Keywords API instead of hand-written text lines, so card syntax
comes from Ansys's own validated keyword classes rather than a guess.

Field names below (dofx/dofy/dofz on BoundarySpcSet, .nodes as a SeriesCard
on SetNodeList, .curves as a DataFrame on DefineCurve, etc.) were confirmed
directly against an installed ansys-dyna-core, not guessed from
documentation -- see the project chat history for the interactive session
that verified each one. What is still NOT verified is the actual LS-DYNA
solve itself (this has never touched a real solver) and exactly where the
buckling eigenvalue shows up in your version's output -- see BUILD_GUIDE.md.

The mesh-generation logic (node/element numbering, geometry, boundary node
sets) is unchanged from the first draft and still independently checked --
see `python ls_dyna_model/deck_builder.py --selftest`.
"""

import json
from pathlib import Path

import pandas as pd
from ansys.dyna.core import Deck, keywords

ROOT = Path(__file__).resolve().parent.parent


def load_config():
    with open(ROOT / "ls_dyna_model" / "panel_config.json") as f:
        return json.load(f)


def build_mesh(t_skin_mm, n_str, h_str_mm, t_str_mm, config):
    """Unchanged from the first draft -- see module docstring. Returns node
    coordinates, shell element connectivity (tagged skin=1 / stringer=2),
    and the boundary node-id sets needed for BCs and load application."""
    n_str = int(round(n_str))
    fixed = config["fixed_parameters"]
    La = fixed["panel_length_La_mm"]
    Wb = fixed["panel_width_Wb_mm"]

    target_elem_size_mm = 12.5  # ~ La/40; first-draft mesh density, see BUILD_GUIDE.md
    nx = max(4, round(La / target_elem_size_mm))
    ny_base = max(2, round(Wb / target_elem_size_mm))
    n_bays = n_str + 1
    ny = ny_base + (-ny_base % n_bays)
    if ny == 0:
        ny = n_bays

    dx = La / nx
    dy = Wb / ny

    nodes = {}
    skin_node_of = {}
    node_id_counter = 1
    for j in range(ny + 1):
        for i in range(nx + 1):
            nodes[node_id_counter] = (i * dx, j * dy, 0.0)
            skin_node_of[(i, j)] = node_id_counter
            node_id_counter += 1

    elements = []
    elem_id_counter = 1
    for j in range(ny):
        for i in range(nx):
            n1 = skin_node_of[(i, j)]
            n2 = skin_node_of[(i + 1, j)]
            n3 = skin_node_of[(i + 1, j + 1)]
            n4 = skin_node_of[(i, j + 1)]
            elements.append((elem_id_counter, 1, n1, n2, n3, n4))
            elem_id_counter += 1

    stringer_j_lines = [k * ny // n_bays for k in range(1, n_str + 1)]
    stringer_tip_node_of = {}
    for k_idx, j_line in enumerate(stringer_j_lines):
        y = j_line * dy
        for i in range(nx + 1):
            x = i * dx
            nodes[node_id_counter] = (x, y, h_str_mm)
            stringer_tip_node_of[(k_idx, i)] = node_id_counter
            node_id_counter += 1
        for i in range(nx):
            base1 = skin_node_of[(i, j_line)]
            base2 = skin_node_of[(i + 1, j_line)]
            tip2 = stringer_tip_node_of[(k_idx, i + 1)]
            tip1 = stringer_tip_node_of[(k_idx, i)]
            elements.append((elem_id_counter, 2, base1, base2, tip2, tip1))
            elem_id_counter += 1

    loaded_edge_x0 = [skin_node_of[(0, j)] for j in range(ny + 1)]
    loaded_edge_xLa = [skin_node_of[(nx, j)] for j in range(ny + 1)]
    unloaded_edge_y0 = [skin_node_of[(i, 0)] for i in range(nx + 1)]
    unloaded_edge_yWb = [skin_node_of[(i, ny)] for i in range(nx + 1)]

    return {
        "nodes": nodes, "elements": elements,
        "nx": nx, "ny": ny, "dx": dx, "dy": dy,
        "loaded_edge_x0": loaded_edge_x0, "loaded_edge_xLa": loaded_edge_xLa,
        "unloaded_edge_y0": unloaded_edge_y0, "unloaded_edge_yWb": unloaded_edge_yWb,
        "n_str_actual": n_str,
    }


def mass_kg(t_skin_mm, n_str, h_str_mm, t_str_mm, config):
    """Unchanged -- analytical mass, independent of the LS-DYNA mesh/solve."""
    n_str = int(round(n_str))
    fixed = config["fixed_parameters"]
    La_m = fixed["panel_length_La_mm"] / 1000.0
    Wb_m = fixed["panel_width_Wb_mm"] / 1000.0
    rho = config["material"]["density_kg_m3"]
    skin_volume_m3 = La_m * Wb_m * (t_skin_mm / 1000.0)
    stringer_volume_m3 = n_str * La_m * (h_str_mm / 1000.0) * (t_str_mm / 1000.0)
    return rho * (skin_volume_m3 + stringer_volume_m3)


def build_deck(point_id, t_skin_mm, n_str, h_str_mm, t_str_mm, config):
    """Builds a full ansys.dyna.core.Deck for one design point. Units: mm /
    N / tonne / s (density in tonne/mm^3 = kg/m^3 * 1e-12, E and stress in
    N/mm^2 = MPa) -- confirm this matches what your install expects; a unit
    mismatch is a classic first-run LS-DYNA gotcha, see BUILD_GUIDE.md."""
    mesh = build_mesh(t_skin_mm, n_str, h_str_mm, t_str_mm, config)
    mat_cfg = config["material"]
    fixed = config["fixed_parameters"]
    study = config["study"]
    n_str = mesh["n_str_actual"]

    deck = Deck(title=f"Panel point {point_id}: t_skin={t_skin_mm:.4f}mm n_str={n_str} "
                       f"h_str={h_str_mm:.4f}mm t_str={t_str_mm:.4f}mm")

    node_ids = sorted(mesh["nodes"].keys())
    node_kw = keywords.Node()
    node_kw.nodes = pd.DataFrame({
        "nid": node_ids,
        "x": [mesh["nodes"][n][0] for n in node_ids],
        "y": [mesh["nodes"][n][1] for n in node_ids],
        "z": [mesh["nodes"][n][2] for n in node_ids],
    })
    deck.append(node_kw)

    elem_kw = keywords.ElementShell()
    elem_kw.elements = pd.DataFrame({
        "eid": [e[0] for e in mesh["elements"]],
        "pid": [e[1] for e in mesh["elements"]],
        "n1": [e[2] for e in mesh["elements"]],
        "n2": [e[3] for e in mesh["elements"]],
        "n3": [e[4] for e in mesh["elements"]],
        "n4": [e[5] for e in mesh["elements"]],
    })
    deck.append(elem_kw)

    mat = keywords.Mat001(mid=1, ro=mat_cfg["density_kg_m3"] / 1e12,
                           e=mat_cfg["youngs_modulus_GPa"] * 1000.0, pr=mat_cfg["poissons_ratio"])
    deck.append(mat)

    sec_skin = keywords.SectionShell(secid=1, elform=16, t1=t_skin_mm, t2=t_skin_mm,
                                      t3=t_skin_mm, t4=t_skin_mm)
    deck.append(sec_skin)
    sec_str = keywords.SectionShell(secid=2, elform=16, t1=t_str_mm, t2=t_str_mm,
                                     t3=t_str_mm, t4=t_str_mm)
    deck.append(sec_str)

    part_kw = keywords.Part()
    part_kw.parts = pd.DataFrame({"pid": [1, 2], "secid": [1, 2], "mid": [1, 1]})
    deck.append(part_kw)

    # Boundary conditions -- see panel_config.json's boundary_conditions block.
    edge_sets = [
        (1, mesh["loaded_edge_x0"], dict(dofx=1, dofy=1, dofz=1)),   # reacting edge, fully fixed in-plane
        (2, mesh["loaded_edge_xLa"], dict(dofx=0, dofy=1, dofz=1)),  # loaded edge, X free for load
        (3, mesh["unloaded_edge_y0"], dict(dofx=0, dofy=0, dofz=1)),  # simply supported, in-plane free
        (4, mesh["unloaded_edge_yWb"], dict(dofx=0, dofy=0, dofz=1)),
    ]
    for sid, node_list, dofs in edge_sets:
        set_kw = keywords.SetNodeList()
        set_kw.sid = sid
        set_kw.nodes = list(node_list)
        deck.append(set_kw)

        spc = keywords.BoundarySpcSet()
        spc.nsid = sid
        spc.dofx, spc.dofy, spc.dofz = dofs["dofx"], dofs["dofy"], dofs["dofz"]
        deck.append(spc)

    # Reference load, distributed equally among the loaded edge's nodes.
    # Pref is computed PER DESIGN POINT (not a single fixed config value) --
    # see classical_buckling.PRELOAD_SAFETY_FACTOR's docstring for why a
    # fixed Pref doesn't work across the DOE's full range of critical loads.
    # Deferred import: classical_buckling.py itself imports from this module
    # at its top level (for mass_kg), so importing it back at deck_builder's
    # own module level would be circular; importing here, inside the
    # function, is safe because by the time build_deck() is ever called this
    # module has already finished its own top-level import.
    from classical_buckling import reference_load_Pref_N
    Pref = reference_load_Pref_N(t_skin_mm, n_str, config)
    n_loaded_nodes = len(mesh["loaded_edge_xLa"])
    force_per_node = Pref / n_loaded_nodes

    curve = keywords.DefineCurve(lcid=1)
    curve.curves = pd.DataFrame({"a1": [0.0, 1.0], "o1": [0.0, 1.0]})
    deck.append(curve)

    # NOTE on sign: node set 2 (loaded_edge_xLa) is free in +X (dofx=0) while
    # the opposite edge (loaded_edge_x0) is fixed in X -- so a *positive*
    # force in the dof=1 (+X) direction pulls the free edge AWAY from the
    # fixed edge, i.e. TENSION, not the intended axial compression. This
    # was not a guess -- a real solve against this exact deck came back
    # with negative eigenvalues (buckling only under a load reversed from
    # what was applied), which is the textbook symptom of exactly this
    # sign error. Compression means pushing the free edge back toward the
    # fixed edge, i.e. a force in -X, hence the minus sign below.
    load = keywords.LoadNodeSet()
    load.nsid, load.dof, load.lcid, load.sf = 2, 1, 1, -force_per_node
    deck.append(load)

    # Two-step linear buckling: nonlinear implicit pre-load, then eigenvalue
    # extraction -- first-draft solver control values beyond the library's
    # own defaults, see BUILD_GUIDE.md.
    deck.append(keywords.ControlImplicitGeneral(imflag=1, dt0=1.0))
    deck.append(keywords.ControlImplicitSolution())  # library defaults
    deck.append(keywords.ControlImplicitBuckle(nmode=study["n_eigenvalues_to_extract"]))
    deck.append(keywords.ControlTermination(endtim=1.0))
    deck.append(keywords.DatabaseGlstat(dt=1.0))

    return deck


def render_keyword_deck(point_id, t_skin_mm, n_str, h_str_mm, t_str_mm, config):
    """Back-compat text-returning wrapper around build_deck(), used by the
    self-test below."""
    deck = build_deck(point_id, t_skin_mm, n_str, h_str_mm, t_str_mm, config)
    return deck.write()


def _selftest():
    """Checks the mesh-generation and deck-building logic without needing a
    real LS-DYNA install: no duplicate node/element IDs, non-degenerate
    element connectivity, node counts match expectations, and the PyDYNA
    Deck round-trips (builds, writes text, starts with *KEYWORD / ends with
    *END) without raising."""
    config = load_config()
    cases = [(1.0, 2, 10.0, 1.0), (2.0, 4, 25.0, 2.0), (3.0, 6, 40.0, 3.0)]
    for t_skin, n_str, h_str, t_str in cases:
        mesh = build_mesh(t_skin, n_str, h_str, t_str, config)
        node_ids = list(mesh["nodes"].keys())
        assert len(node_ids) == len(set(node_ids)), "duplicate node IDs"
        elem_ids = [e[0] for e in mesh["elements"]]
        assert len(elem_ids) == len(set(elem_ids)), "duplicate element IDs"
        for eid, pid, n1, n2, n3, n4 in mesh["elements"]:
            corner_ids = {n1, n2, n3, n4}
            assert len(corner_ids) == 4, f"degenerate element {eid}"
            for nid in corner_ids:
                assert nid in mesh["nodes"], f"element {eid} references missing node {nid}"
        expected_skin_nodes = (mesh["nx"] + 1) * (mesh["ny"] + 1)
        expected_stringer_nodes = n_str * (mesh["nx"] + 1)
        assert len(mesh["nodes"]) == expected_skin_nodes + expected_stringer_nodes
        assert len(mesh["loaded_edge_x0"]) == mesh["ny"] + 1
        assert len(mesh["loaded_edge_xLa"]) == mesh["ny"] + 1
        assert len(mesh["unloaded_edge_y0"]) == mesh["nx"] + 1
        assert len(mesh["unloaded_edge_yWb"]) == mesh["nx"] + 1
        m = mass_kg(t_skin, n_str, h_str, t_str, config)
        assert m > 0
        print(f"OK  t_skin={t_skin} n_str={n_str} h_str={h_str} t_str={t_str}  "
              f"nodes={len(mesh['nodes'])} elements={len(mesh['elements'])} "
              f"mass={m:.4f}kg  (nx={mesh['nx']}, ny={mesh['ny']})")

        deck = build_deck(0, t_skin, n_str, h_str, t_str, config)
        text = deck.write()
        assert text.strip().startswith("*KEYWORD")
        assert text.strip().endswith("*END")
        assert text.count("*NODE") == 1
        assert text.count("*ELEMENT_SHELL") == 1
        assert text.count("*BOUNDARY_SPC_SET") == 4
        validation = deck.validate()
        assert validation.is_valid, f"deck.validate() found errors: {validation.errors}"
    print("\nAll self-tests passed.")


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        _selftest()
    else:
        print(__doc__)
