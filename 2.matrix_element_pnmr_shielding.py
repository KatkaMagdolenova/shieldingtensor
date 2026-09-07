r"""
CURIE CONTRIBUTION TO THE PARAMAGNETIC NMR SHIELDING TENSOR
============================================================

WHAT THIS SCRIPT CALCULATES
---------------------------
The input file contains zero-field electronic energies, matrix elements of

    D_u = 1/2 (g_e S_u + L_u),

and hyperfine matrix elements separated into FC, SD and LI contributions.
The script uses these quantities to calculate ONLY the temperature-dependent
Curie part of the pNMR shielding tensor.

For a degenerate zero-field level lambda, with states |lambda,a>,

    beta = 1/(k_B T),
    w(lambda,a) = exp(-beta E_lambda) / Q.

For Cartesian directions u,v = x,y,z,

    Lambda_Curie(lambda,a;u,v)
      = -beta sum_(a' in the same level)
          <lambda,a|H_B(u)|lambda,a'>
          <lambda,a'|H_mu(v)|lambda,a>,

and the final thermal shielding is

    sigma_Curie(u,v)
      = sum_(lambda,a) w(lambda,a) Re[Lambda_Curie(lambda,a;u,v)].

The tensor is dimensionless; the reports multiply it by 10^6 and print ppm.

All generated results are written to a REPORT_DIR subfolder next to the
input data file, for example:

    report/
        *.curie_report.txt
        *.curie_summary.txt
        *.curie_pcs_field.cube

HOW THE INPUT MATRIX ELEMENTS ARE USED
--------------------------------------
The electronic-field derivative is

    H_B = D/c.

For the nuclear-moment derivative, the input file supplies the hyperfine block
A and the same prefactor P used to build that block.  Since

    P = g_e mu_B gamma_M,        H_mu = A/gamma_M,

we use the equivalent expression

    H_mu = g_e mu_B (A/P).

This is important: the input labels the A block as atomic units and this script
DOES NOT convert A to MHz or reinterpret its unit.  A and P are used exactly in
the common numerical convention of the input file; their common scale cancels
in A/P.

DECOMPOSITION PRINTED IN THE REPORT
-----------------------------------
Because A_TOTAL = A_FC + A_SD + A_LI, the Curie term is linear:

    sigma_TOTAL = sigma_FC + sigma_SD + sigma_LI.

For every thermally relevant level, the report also resolves the answer into
individual state paths a -> a'.  For the ground Kramers doublet this means
1->1, 1->2, 2->1 and 2->2.  Every number in that table is already the final
Boltzmann-weighted contribution of that single path in ppm.

LI means the LI part of the HYPERFINE operator; it is not the complete orbital
NMR shielding contribution.

CHECKS
------
The report verifies that FC+SD+LI equals TOTAL.  If the populated ground level
is an isolated doublet, it also evaluates the same Curie term in the equivalent
effective-doublet representation and checks the expected 1/T scaling.  These
are internal consistency checks, not independent experimental validation.

SCOPE
-----
The complete pNMR shielding is generally

    sigma_total = sigma_orbital + sigma_Curie.

The orbital contribution requires additional response information that is not
present in this data file.  No experimental chemical shift is reported because
that would additionally require a reference shielding.

USE
---
Place this script and "h2o.casscf.rel.pnmr.data" in the same directory:

    python pnmr_curie_supervisor.py

or supply another input file:

    python pnmr_curie_supervisor.py other.data --temperature 298.15

Normally only the settings immediately below need to be changed.

================================================================================
LONG-DISTANCE (POINT-DIPOLE) CURIE-SHIFT CUBE  --  NEW SECTION
================================================================================

WHERE THE GEOMETRY COMES FROM
-------------------------------
If the .data file contains a block

    # number of atoms and geometry (in a.u.)
        <natoms>
        El   x   y   z          (repeated natoms times, x,y,z in BOHR)

it is read directly -- no extra file is needed, and this is now the normal
case. The atom order in this block is used as-is (atom #1 = first line),
independent of the "nucleus N" numbering used later for the FC/SD/LI blocks
(some nuclei, e.g. symmetry-equivalent hydrogens, may simply not have a
magnetic-data block even though they appear in the geometry).

If the .data file has NO such block (older files), the script falls back to
looking for a small, separate, plain-text XYZ file (element, x, y, z in
ANGSTROM, standard format) named

    <data file stem>.xyz

next to the input data file (Settings.geometry_file can also give an
explicit path). If neither is found, the cube step is skipped with a clear
message; every other part of the script runs exactly as before.

Either way, the geometry is used only to (a) place atoms in the cube header
so Avogadro draws the molecule, and (b) fix the paramagnetic-centre position
(Settings.center_atom, 1-based index into the geometry list) from which the
field distance R is measured.

WHAT THE CUBE ACTUALLY CONTAINS
--------------------------------
The cube stores the isotropic CURIE-ONLY point-dipole shielding one would obtain
for a point-dipole probe nucleus placed at each grid point r, i.e. the same
physical quantity as "isotropic" in section [2] of the normal report, but
evaluated at an arbitrary field point instead of at a real, ab-initio
computed nucleus.

Derivation (kept fully inside the atomic-units framework already used by
this script -- no additional SI constants such as mu0 or muB are needed):

For a nucleus far from the paramagnetic centre, the exact hyperfine-moment
derivative operator reduces, to leading order in 1/R, to the ordinary
Zeeman-derivative operator reshaped by the point-dipole tensor T(R), plus a
gauge-commutator term:

    (dH/dM_l)_LD  ~=  alpha^2 * sum_m T_ml(R) * (dH/dB_m)  +  i[H0, f_l]

    T_ml(R) = (3 R_m R_l - R^2 delta_ml) / R^5,   alpha = 1/c.

The gauge-commutator term vanishes EXACTLY for matrix elements taken
between two states of the same energy (same thermal level lambda), because
its off-diagonal form is proportional to (E_m - E_n), which is zero within
a degenerate level. This is exactly the class of matrix elements entering
the Curie sum above, so within this script the point-dipole relation

    H_mu,l^LD[a,a'](r)  =  (1/c^2) * sum_m T_ml(R) * H_B,m[a,a']

holds without any further approximation beyond the 1/R expansion itself.

Substituting this into the Curie double sum and using T_ml(R) = D_ml(Rhat)/R^3
with D(Rhat) = 3 Rhat Rhat^T - I gives the compact closed form actually used
by the code:

    chi0(u,v)      = curie(H_B, H_B, info, beta)      [computed ONCE]
    sigma_LD(r)    = (1/c^2) * (1/R^3) * chi0 @ D(Rhat)
    isotropic(r)   = Tr(sigma_LD(r)) / 3

chi0 is the atomic-units Curie B-B response tensor built
from the SAME H_B operator already used everywhere else in this script. It is
not the complete macroscopic susceptibility; it contains only the Curie part.
No isolated-spin-doublet mapping is required, so the same exact-state level
structure already represented by "info" is retained.

SIGN CONVENTION -- IMPORTANT
-----------------------------
This script reports sigma_Curie directly as a SHIELDING (matching Table [2]
above), with no extra minus sign. The cube value is therefore also a
shielding, in the same sign convention as the rest of THIS script. Some
other tools (e.g. a ReSpect-based post-processor using g/A tensors in MHz)
report the NMR SHIFT delta = -sigma instead. Settings.cube_sign lets you
flip the sign if you need to compare the two conventions directly.

WHAT IS "NEW PHYSICS" VS. "ALREADY VALIDATED"
-----------------------------------------------
Everything up to and including "sigma_Curie(u,v)" above is exactly what the
rest of the script already computes and checks (FC+SD+LI=TOTAL etc.). The
point-dipole extrapolation to a field point (the T(R)/chi0 formula) is the
one genuinely NEW piece of physics added for the cube; it is the standard
long-distance/point-dipole limit, but it is only as good as the 1/R
expansion itself, and it is a diagnostic/visualisation aid, not a
recomputation of the ab-initio Curie shift at the real nuclear positions
(those keep coming from section [2]/[3] exactly as before).
"""

import argparse
import math
import re
from pathlib import Path

# ------------------------------- SETTINGS ---------------------------------
HERE = Path(__file__).resolve().parent
DEFAULT_DATA_FILE = HERE / "h2o.casscf.rel.pnmr.data"
TEMPERATURE_K = 298.15
# States closer than this are treated as one numerically degenerate level.
# 1e-8 Ha = 2.19e-3 cm^-1; change only if the state solver has larger noise.
DEGENERACY_TOL_HA = 1e-8
THERMAL_REPORT_CUTOFF = 1e-8
DISPLAY_TOL_PPM = 1e-6
DOUBLET_POP_CUTOFF = 0.999999

KB_HA_PER_K = 3.166811563e-6
HARTREE_TO_CM = 219474.6313705
C_AU = 137.035999084
G_E = 2.00231930436256
MU_B_AU = 1.0 / (2.0 * C_AU)
BOHR_PER_A = 1.889726124565062

# --- NEW: long-distance (point-dipole) Curie-shift cube settings ----------
GEOMETRY_FILE = None          # None -> auto: "<data file stem>.xyz"
CENTER_ATOM = 1               # 1-based index into the XYZ geometry file
WRITE_CUBE = True
REPORT_DIR = "report"
CUBE_FILE_SUFFIX = ".curie_pcs_field.cube"
CUBE_SPACING_A = 0.4
CUBE_PADDING_A = 4.0
CUBE_CUTOFF_A = 1.0
CUBE_SIGN = "shielding"       # "shielding" (this script's own convention)
                               # or "shift" (delta = -sigma, NMR convention)

# ------------------------------- INPUT ------------------------------------
NUM = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
PAIR = re.compile(r"\(\s*(" + NUM + r")\s*,\s*(" + NUM + r")\s*\)")


def matrix_block(lines, i, n):
    """Read an n x n block with three complex values (x,y,z) per line."""
    M = {u: {} for u in "xyz"}
    count = 0
    while count < n*n:
        if i >= len(lines):
            raise ValueError(f"Unexpected end of file while reading a {n}x{n} matrix block")
        line = lines[i]
        i += 1
        if not line.strip():
            continue
        p, z = line.split(None, 2), PAIR.findall(line)
        if len(p) < 2 or len(z) != 3:
            raise ValueError(f"Cannot read matrix line: {line}")
        a, b = int(p[0]), int(p[1])
        for u, (x, y) in zip("xyz", z):
            M[u][(a, b)] = complex(float(x), float(y))
        count += 1
    return M, i


def _next_nonblank(lines, i, what):
    while i < len(lines) and not lines[i].strip():
        i += 1
    if i >= len(lines):
        raise ValueError(f"Unexpected end of file while reading {what}")
    return i


def read_data(path):
    """Read geometry, energies, magnetic moments and FC/SD/LI hyperfine blocks."""
    L = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    d = {"energies": {}, "nuclei": []}
    i = 0

    while i < len(L):
        line = L[i]

        if "number of atoms and geometry" in line:
            i = _next_nonblank(L, i + 1, "number of atoms")
            n_atoms = int(L[i].split()[0])
            i += 1
            geom = []
            while len(geom) < n_atoms:
                if i >= len(L):
                    raise ValueError("Unexpected end of file while reading geometry")
                if L[i].strip():
                    p = L[i].split()
                    if len(p) < 4:
                        raise ValueError(f"Cannot read geometry line: {L[i]!r}")
                    el = p[0]
                    x_bohr, y_bohr, z_bohr = (float(v) for v in p[1:4])
                    geom.append((el, (x_bohr / BOHR_PER_A,
                                       y_bohr / BOHR_PER_A,
                                       z_bohr / BOHR_PER_A)))
                i += 1
            d["geometry"] = geom
            continue

        if "number of quasi-relativistic states" in line:
            i = _next_nonblank(L, i + 1, "number of quasi-relativistic states")
            n = int(L[i].split()[0])
            i += 1
            while len(d["energies"]) < n:
                if i >= len(L):
                    raise ValueError("Unexpected end of file while reading state energies")
                if L[i].strip():
                    p = L[i].split()
                    if len(p) < 2:
                        raise ValueError(f"Cannot read energy line: {L[i]!r}")
                    d["energies"][int(p[0])] = float(p[1])
                i += 1
            d["n_states"] = n
            continue

        if "magnetic moment matrix elements" in line:
            if "n_states" not in d:
                raise ValueError("Magnetic-moment block appears before the number of states")
            i = _next_nonblank(L, i + 1, "magnetic moment matrix elements")
            d["D"], i = matrix_block(L, i, d["n_states"])
            continue

        m = re.search(r"nucleus\s+(\d+)\s*\(([^;]+);\s*P[^=]*=\s*("+NUM+r")", line)
        if m:
            compact = line.replace(" ", "")
            N = {
                "index": int(m.group(1)),
                "label": m.group(2).strip(),
                "P": float(m.group(3)),
                # The present input explicitly states:
                # P = g_e * g_N * beta_e * beta_N.
                "P_formula_ok": "P(g_e*g_N*beta_e*beta_N)=" in compact,
            }
            i += 1

            for name in ("FC", "SD", "LI"):
                marker = f"# {name} contribution"
                while i < len(L) and marker not in L[i]:
                    # Do not silently steal a block from the next nucleus.
                    if re.search(r"#\s*nucleus\s+\d+", L[i], re.I):
                        raise ValueError(
                            f"Missing {name} block for nucleus {N['index']} before next nucleus"
                        )
                    i += 1
                if i >= len(L):
                    raise ValueError(f"Missing {name} block for nucleus {N['index']}")
                i = _next_nonblank(L, i + 1, f"{name} block for nucleus {N['index']}")
                N[name], i = matrix_block(L, i, d["n_states"])

            d["nuclei"].append(N)
            continue

        i += 1

    if not all(k in d for k in ("n_states", "D")) or not d["energies"]:
        raise ValueError("Input does not contain the required energy/magnetic blocks.")
    if not d["nuclei"]:
        raise ValueError("Input does not contain any nucleus FC/SD/LI blocks.")
    return d


def read_xyz(path):
    """Read a plain XYZ file: line1=natoms, line2=comment, then
    'Element  x  y  z' (Angstrom) for each atom. Returns a list of
    (element, (x, y, z)) tuples, 0-indexed in list order (atom #1 = index 0).
    """
    lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    n = int(lines[0].split()[0])
    atoms = []
    for line in lines[2:2 + n]:
        p = line.split()
        if len(p) < 4:
            raise ValueError(f"Cannot read XYZ line: {line!r}")
        atoms.append((p[0], (float(p[1]), float(p[2]), float(p[3]))))
    if len(atoms) != n:
        raise ValueError(f"XYZ header says {n} atoms, found {len(atoms)}")
    return atoms

# ------------------------------ CALCULATION -------------------------------
def make_levels(E):
    """Group numerically degenerate states without merging genuine small splittings."""
    levels = []
    for s, e in sorted(E.items(), key=lambda x: x[1]):
        if not levels or abs(e - levels[-1]["E_ref"]) >= DEGENERACY_TOL_HA:
            levels.append({"E_ref": e, "E_values": [e], "states": [s]})
        else:
            levels[-1]["E_values"].append(e)
            levels[-1]["states"].append(s)

    for x in levels:
        vals = x.pop("E_values")
        x["E"] = sum(vals) / len(vals)
        x["spread"] = max(vals) - min(vals)
        x.pop("E_ref")
    return levels


def thermal(levels, T):
    if T <= 0:
        raise ValueError("Temperature must be positive.")
    beta = 1/(KB_HA_PER_K*T)
    E0 = levels[0]["E"]
    b = [math.exp(-beta*(x["E"]-E0)) for x in levels]
    Q = sum(len(x["states"])*q for x, q in zip(levels, b))
    info = [{**x, "w": q/Q, "population": len(x["states"])*q/Q}
            for x, q in zip(levels, b)]
    return beta, info


def add_ops(*ops):
    return {u: {ij: sum(op[u].get(ij, 0j) for op in ops)
                for ij in set().union(*(op[u] for op in ops))} for u in "xyz"}


def field_operator(D):
    return {u: {ij: z/C_AU for ij, z in D[u].items()} for u in "xyz"}


def nuclear_operator(A, P):
    if P == 0: raise ValueError("P=0: cannot form A/P")
    f = G_E*MU_B_AU/P
    return {u: {ij: f*z for ij, z in A[u].items()} for u in "xyz"}


def curie(HB, HM, info, beta, paths=False):
    """Direct state sum; returned path values are final weighted contributions."""
    S = {(u,v): 0.0 for u in "xyz" for v in "xyz"}
    P = []
    for lev, x in enumerate(info):
        for a in x["states"]:
            for ap in x["states"]:
                for u in "xyz":
                    for v in "xyz":
                        q = x["w"]*(-beta*HB[u][(a,ap)]*HM[v][(ap,a)]).real
                        S[(u,v)] += q
                        if paths: P.append((lev,a,ap,u,v,q))
    return S, P

# Equivalent effective-doublet representation used only as an internal check.
def block(M, u, a, b):
    return M[u][(a,a)], M[u][(a,b)], M[u][(b,a)], M[u][(b,b)]


def pauli_coeffs(M):
    a,b,c,d = M
    return {"x": (b+c)/2, "y": 0.5j*(b-c), "z": (a-d)/2, "0": (a+d)/2}


def doublet_tensor(HB, HM, a, b, beta):
    g, K = {}, {}  # K = A/gamma
    for u in "xyz":
        q = pauli_coeffs(block(HB,u,a,b))
        for s in "xyz": g[(u,s)] = (4*C_AU*q[s]).real
    for v in "xyz":
        q = pauli_coeffs(block(HM,v,a,b))
        for s in "xyz": K[(s,v)] = (2*q[s]).real
    return {(u,v): -beta/(8*C_AU)*sum(g[(u,s)]*K[(s,v)] for s in "xyz")
            for u in "xyz" for v in "xyz"}


def herm_error(M):
    return max(abs(z-M[u][(j,i)].conjugate()) for u in "xyz" for (i,j),z in M[u].items())


# --- NEW: point-dipole geometry helpers ------------------------------------
def outer3(u, v):
    return {(a, b): u[a]*v[b] for a in "xyz" for b in "xyz"}


def dipolar_D(Rhat):
    """D(Rhat) = 3 Rhat Rhat^T - I, Rhat given as a dict keyed by 'x','y','z'."""
    O = outer3(Rhat, Rhat)
    return {(a, b): 3*O[(a, b)] - (1.0 if a == b else 0.0) for a in "xyz" for b in "xyz"}


def mat_trace_product(M, D):
    """Tr(M @ D) for two dicts keyed by (row, col) in 'xyz'."""
    return sum(M[(a, b)]*D[(b, a)] for a in "xyz" for b in "xyz")


def chi0_curie(HB, info, beta):
    """Curie B-B response Y_Curie(B,B) = curie(H_B,H_B), in the script's a.u.

    It is the position-independent tensor needed for the Curie-only
    long-distance field.  It is NOT the complete macroscopic susceptibility,
    because temperature-independent/Van-Vleck-like pieces are outside the
    Curie-only scope of this script.
    """
    S, _ = curie(HB, HB, info, beta)
    return S


def pcs_ld_shielding_ppm(chi0, R_bohr):
    """Isotropic point-dipole Curie shielding [ppm] at displacement
    R_bohr = (x, y, z) (atomic units, i.e. bohr) from the paramagnetic centre.
    """
    Rn = math.sqrt(sum(x*x for x in R_bohr))
    if Rn < 1e-9:
        return float("nan")
    Rhat = {u: x/Rn for u, x in zip("xyz", R_bohr)}
    D = dipolar_D(Rhat)
    tr = mat_trace_product(chi0, D)
    return 1e6 * tr / (3.0 * C_AU**2 * Rn**3)


_ELEMENTS = ("X H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn "
             "Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe").split()
_Z = {e: i for i, e in enumerate(_ELEMENTS)}


def write_pcs_cube(cube_path, atoms, center_idx, chi0,
                    spacing_A=CUBE_SPACING_A, padding_A=CUBE_PADDING_A,
                    cutoff_A=CUBE_CUTOFF_A, sign=CUBE_SIGN):
    """Write a Gaussian-cube file with the long-distance point-dipole Curie
    shielding (or shift, see 'sign') field, on a regular grid, for loading
    into Avogadro or any other cube viewer. 'atoms' is the list returned by
    read_xyz(); 'center_idx' is 1-based, matching Settings.center_atom.
    """
    if sign not in ("shielding", "shift"):
        raise ValueError("CUBE_SIGN must be 'shielding' or 'shift'")
    if spacing_A <= 0:
        raise ValueError("CUBE_SPACING_A must be positive")
    if padding_A < 0 or cutoff_A < 0:
        raise ValueError("CUBE_PADDING_A and CUBE_CUTOFF_A must be non-negative")
    if not (1 <= center_idx <= len(atoms)):
        raise ValueError(f"center_idx={center_idx} outside 1..{len(atoms)}")

    xyz_A = [a[1] for a in atoms]
    center_A = xyz_A[center_idx - 1]

    xs = [p[0] for p in xyz_A]; ys = [p[1] for p in xyz_A]; zs = [p[2] for p in xyz_A]
    lo = (min(xs)-padding_A, min(ys)-padding_A, min(zs)-padding_A)
    hi = (max(xs)+padding_A, max(ys)+padding_A, max(zs)+padding_A)
    nx = int(math.ceil((hi[0]-lo[0])/spacing_A)) + 1
    ny = int(math.ceil((hi[1]-lo[1])/spacing_A)) + 1
    nz = int(math.ceil((hi[2]-lo[2])/spacing_A)) + 1

    sign_factor = -1.0 if sign == "shift" else 1.0

    values = []
    vmin, vmax = float("inf"), float("-inf")
    for ix in range(nx):
        x_A = lo[0] + ix*spacing_A
        for iy in range(ny):
            y_A = lo[1] + iy*spacing_A
            for iz in range(nz):
                z_A = lo[2] + iz*spacing_A
                R_A = (x_A-center_A[0], y_A-center_A[1], z_A-center_A[2])
                Rn_A = math.sqrt(sum(c*c for c in R_A))
                if Rn_A < cutoff_A:
                    v = 0.0
                else:
                    R_bohr = tuple(c*BOHR_PER_A for c in R_A)
                    v = sign_factor * pcs_ld_shielding_ppm(chi0, R_bohr)
                values.append(v)
                vmin, vmax = min(vmin, v), max(vmax, v)

    origin_bohr = tuple(c*BOHR_PER_A for c in lo)
    step_bohr = spacing_A*BOHR_PER_A

    label = "shift delta_pc" if sign == "shift" else "shielding sigma_Curie (point-dipole, LD)"
    with open(cube_path, "w") as f:
        f.write(f"Long-distance point-dipole Curie {label} field [ppm]\n")
        f.write(f"generated by pnmr_curie_supervisor.py; centre = atom #{center_idx}\n")
        f.write(f"{len(atoms):5d}{origin_bohr[0]:13.6f}{origin_bohr[1]:13.6f}{origin_bohr[2]:13.6f}\n")
        f.write(f"{nx:5d}{step_bohr:13.6f}{0.0:13.6f}{0.0:13.6f}\n")
        f.write(f"{ny:5d}{0.0:13.6f}{step_bohr:13.6f}{0.0:13.6f}\n")
        f.write(f"{nz:5d}{0.0:13.6f}{0.0:13.6f}{step_bohr:13.6f}\n")
        for el, pos in atoms:
            z = _Z.get(el, 0)
            p_bohr = tuple(c*BOHR_PER_A for c in pos)
            f.write(f"{z:5d}{float(z):13.6f}{p_bohr[0]:13.6f}{p_bohr[1]:13.6f}{p_bohr[2]:13.6f}\n")
        for i in range(0, len(values), 6):
            f.write("".join(f"{v:13.5e}" for v in values[i:i+6]) + "\n")

    return {"path": cube_path, "nx": nx, "ny": ny, "nz": nz, "min": vmin, "max": vmax}


# -------------------------------- OUTPUT ----------------------------------
def ppm(S): return {k: 1e6*v for k,v in S.items()}
def isotropic(S): return sum(S[(u,u)] for u in "xyz")/3


def print_matrix(out, S):
    out("                 x             y             z")
    for u in "xyz": out(f"    {u}  " + " ".join(f"{S[(u,v)]:+13.4f}" for v in "xyz"))



def write_summary(summary_path, path, T, info, nuclei_summary, cube_summary):
    """Write a compact human-readable results summary."""
    L = []
    add = L.append
    add("="*78)
    add("pNMR CURIE-SHIELDING RESULTS SUMMARY")
    add("="*78)
    add(f"Input       : {path}")
    add(f"Temperature : {T:.2f} K")
    add("Scope       : Curie shielding only; delta_Curie = -sigma_Curie")
    add(f"Ground level: states={info[0]['states']}, population={info[0]['population']:.9f}")
    add("")

    for r in nuclei_summary:
        add("-"*78)
        add(f"NUCLEUS {r['index']}: {r['label']}    P={r['P']:+.7g}")
        add("-"*78)
        add("Curie shielding tensor sigma_Curie [ppm]")
        add("                 x             y             z")
        for u in "xyz":
            add(f"    {u}  " + " ".join(f"{r['tensor'][(u,v)]:+13.4f}" for v in "xyz"))
        add(f"  sigma_iso = {r['sigma_iso']:+.4f} ppm")
        add(f"  delta_Curie = {-r['sigma_iso']:+.4f} ppm")
        add("")
        add("Isotropic decomposition [ppm]")
        add(f"  FC={r['iso_FC']:+.4f}   SD={r['iso_SD']:+.4f}   "
            f"LI={r['iso_LI']:+.4f}   TOTAL={r['sigma_iso']:+.4f}")
        add("")
        add("Checks")
        add(f"  FC+SD+LI -> TOTAL : {r['closure']:.3e} ppm")
        if r['doublet_err'] is not None:
            add(f"  direct sum vs doublet : {r['doublet_err']:.3e} ppm")
        if r['invT_err'] is not None:
            add(f"  1/T check            : {r['invT_err']:.3e} ppm")
        elif r['invT_note']:
            add(f"  1/T check            : {r['invT_note']}")
        add(f"  Hermiticity          : {r['herm']:.3e}")
        add("")

    add("="*78)
    add("LONG-DISTANCE CURIE-ONLY FIELD")
    add("="*78)
    if cube_summary is None:
        add("Cube not written.")
    else:
        add(f"Centre atom : #{cube_summary['center']} ({cube_summary['element']})")
        add(f"Sign        : {cube_summary['sign']}")
        add(f"isotropic(Y_BB^Curie) = {cube_summary['chi_iso']:+.6e} a.u.")
        add(f"Cube        : {cube_summary['path']}")
        add(f"Grid        : {cube_summary['nx']}x{cube_summary['ny']}x{cube_summary['nz']}")
        add(f"Range       : {cube_summary['min']:.6g} .. {cube_summary['max']:.6g} ppm")
    add("")
    add("NOTE: sigma_Curie is a shielding contribution. The corresponding Curie")
    add("chemical-shift contribution is delta_Curie = -sigma_Curie. This is not")
    add("the complete experimental pNMR shift.")
    summary_path.write_text("\n".join(L) + "\n", encoding="utf-8")
    return summary_path


def run(path=DEFAULT_DATA_FILE, T=TEMPERATURE_K):
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    result_dir = path.parent / REPORT_DIR
    result_dir.mkdir(parents=True, exist_ok=True)

    data = read_data(path)
    levels = make_levels(data["energies"])
    beta, info = thermal(levels, T)
    HB = field_operator(data["D"])
    text = []
    nuclei_summary = []
    cube_summary = None
    def out(s=""): print(s); text.append(s)

    out("="*78); out("CURIE CONTRIBUTION TO THE pNMR SHIELDING TENSOR"); out("="*78)
    out(f"Input       : {path}")
    out(f"Temperature : {T:.2f} K")
    out("Operators   : H_B=D/c ; H_mu=g_e*mu_B*(A/P) ")
    out("Scope       : Curie shielding only; not the complete pNMR shielding/shift\n")

    out("[1] THERMAL LEVELS")
    out(f"  degeneracy tolerance = {DEGENERACY_TOL_HA:.1e} Ha "
        f"({DEGENERACY_TOL_HA*HARTREE_TO_CM:.3e} cm^-1)")
    for n,x in enumerate(info):
        out(f"  {n}: states={x['states']}  E={x['E']:.12f} Ha  "
            f"spread={x['spread']:.2e} Ha  population={x['population']:.6e}")
    out(f"  D Hermiticity max error = {herm_error(data['D']):.3e}\n")

    for N in data["nuclei"]:
        out("="*78); out(f"NUCLEUS {N['index']}: {N['label']}    P={N['P']:+.7g} (input scale)"); out("="*78)
        out("  P convention: " + ("g_e*g_N*beta_e*beta_N [recognized]"
                                 if N.get("P_formula_ok") else
                                 "header formula not recognized; verify A/P normalization"))
        A = {m:N[m] for m in ("FC","SD","LI")}
        A["TOTAL"] = add_ops(A["FC"],A["SD"],A["LI"])
        S, paths = {}, {}
        for m in ("FC","SD","LI","TOTAL"):
            S[m], paths[m] = curie(HB,nuclear_operator(A[m],N["P"]),info,beta,True)
        R = {m:ppm(S[m]) for m in S}

        out("[2] FINAL CURIE SHIELDING [ppm]"); print_matrix(out,R["TOTAL"])
        out(f"    isotropic = {isotropic(R['TOTAL']):+.4f} ppm\n")

        comps = [(u,v) for u in "xyz" for v in "xyz"
                 if u==v or max(abs(R[m][(u,v)]) for m in R)>DISPLAY_TOL_PPM]
        out("[3] FC / SD / LI DECOMPOSITION [ppm]")
        out("  component          FC             SD             LI          TOTAL")
        for u,v in comps:
            out(f"    {u}{v}      " + " ".join(f"{R[m][(u,v)]:+14.4f}" for m in ("FC","SD","LI","TOTAL")))
        out("  isotropic   " + " ".join(f"{isotropic(R[m]):+14.4f}" for m in ("FC","SD","LI","TOTAL")))
        closure = max(abs(R["FC"][k]+R["SD"][k]+R["LI"][k]-R["TOTAL"][k]) for k in R["TOTAL"])
        out()

        out("[4] INDIVIDUAL STATE-PATH CONTRIBUTIONS [ppm]")
        for lev,x in enumerate(info):
            if x["population"] < THERMAL_REPORT_CUTOFF: continue
            states, cols = x["states"], [(a,ap) for a in x["states"] for ap in x["states"]]
            out(f"  level {lev}, states={states}, population={x['population']:.9f}")
            for u,v in comps:
                out(f"    sigma_{u}{v}")
                out("      mechanism " + " ".join(f"{a}->{ap}".rjust(13) for a,ap in cols) + "           SUM")
                for m in ("FC","SD","LI","TOTAL"):
                    D = {(a,ap):q*1e6 for L,a,ap,uu,vv,q in paths[m] if L==lev and uu==u and vv==v}
                    vals = [D[c] for c in cols]
                    out(f"      {m:9s}" + " ".join(f"{q:+13.4f}" for q in vals) + f" {sum(vals):+13.4f}")
            out()

        out("[5] INTERNAL CONSISTENCY CHECKS")
        out(f"  FC + SD + LI = TOTAL: max error = {closure:.3e} ppm")
        doublet_err, invT_err, invT_note = None, None, ""
        if len(info[0]["states"]) == 2 and info[0]["population"] >= DOUBLET_POP_CUTOFF:
            a,b = info[0]["states"]
            M = doublet_tensor(HB,nuclear_operator(A["TOTAL"],N["P"]),a,b,beta)
            doublet_err = max(abs(R["TOTAL"][k]-1e6*M[k]) for k in M)
            out(f"  direct state sum vs equivalent doublet form: {doublet_err:.3e} ppm")

            beta2,info2 = thermal(levels,2*T)
            if info2[0]["population"] >= DOUBLET_POP_CUTOFF:
                S2,_ = curie(HB,nuclear_operator(A["TOTAL"],N["P"]),info2,beta2)
                invT_err = max(abs(1e6*S2[k]-0.5*R["TOTAL"][k]) for k in S2)
                out(f"  isolated-doublet 1/T check at {2*T:.2f} K: {invT_err:.3e} ppm")
            else:
                invT_note = (f"skipped at {2*T:.2f} K; ground-level population="
                             f"{info2[0]['population']:.6f}")
                out(f"  1/T check at {2*T:.2f} K skipped: excited levels are thermally populated "
                    f"(ground-level population={info2[0]['population']:.6f})")
        else:
            invT_note = "skipped; ground level is not an isolated populated doublet"
            out("  doublet checks skipped: ground level is not an isolated populated doublet")
        hf_herm = max(herm_error(A[m]) for m in ("FC","SD","LI"))
        out(f"  hyperfine Hermiticity max error = {hf_herm:.3e}\n")

        nuclei_summary.append({
            "index": N["index"], "label": N["label"], "P": N["P"],
            "tensor": R["TOTAL"], "sigma_iso": isotropic(R["TOTAL"]),
            "iso_FC": isotropic(R["FC"]), "iso_SD": isotropic(R["SD"]),
            "iso_LI": isotropic(R["LI"]), "closure": closure,
            "doublet_err": doublet_err, "invT_err": invT_err,
            "invT_note": invT_note, "herm": hf_herm,
        })

    # --- Long-distance point-dipole field from the Curie contribution only ---
    out("="*78); out("[6] LONG-DISTANCE (POINT-DIPOLE) CURIE-ONLY FIELD"); out("="*78)
    if not WRITE_CUBE:
        out("  Cube output disabled (WRITE_CUBE = False).\n")
    else:
        atoms, geom_source = None, None
        if "geometry" in data:
            atoms, geom_source = data["geometry"], f"{path} (embedded, converted from bohr)"
        else:
            if GEOMETRY_FILE:
                geom_path = Path(GEOMETRY_FILE)
                if not geom_path.is_absolute():
                    geom_path = path.parent / geom_path
            else:
                geom_path = path.with_suffix(".xyz")
            if geom_path.exists():
                atoms, geom_source = read_xyz(geom_path), str(geom_path)

        if atoms is None:
            out("  No geometry found: the .data file has no '# number of atoms and")
            out("  geometry' block, and no external XYZ file was found either")
            out(f"  (looked for: {Path(GEOMETRY_FILE) if GEOMETRY_FILE else Path(path).with_suffix('.xyz')}).")
            out("  Skipping cube output.\n")
        else:
            if not (1 <= CENTER_ATOM <= len(atoms)):
                out(f"  CENTER_ATOM={CENTER_ATOM} is out of range for {len(atoms)} atoms "
                    f"in {geom_source}; skipping cube output.\n")
            else:
                chi0 = chi0_curie(HB, info, beta)
                out(f"  Geometry    : {geom_source}  ({len(atoms)} atoms)")
                out(f"  Centre atom : #{CENTER_ATOM} ({atoms[CENTER_ATOM-1][0]})")
                out(f"  Sign        : {CUBE_SIGN}  "
                    f"({'delta = -sigma_Curie' if CUBE_SIGN == 'shift' else 'sigma_Curie, this script''s own convention'})")
                out("  Y_BB^Curie = curie(H_B,H_B) [a.u.; not full susceptibility]:")
                print_matrix(out, {k: v for k, v in chi0.items()})
                out(f"    isotropic(Y_BB^Curie) = {isotropic(chi0):+.6e} a.u.\n")

                cube_path = result_dir / (path.stem + CUBE_FILE_SUFFIX)
                info_cube = write_pcs_cube(cube_path, atoms, CENTER_ATOM, chi0)
                cube_summary = {
                    **info_cube,
                    "center": CENTER_ATOM,
                    "element": atoms[CENTER_ATOM-1][0],
                    "sign": CUBE_SIGN,
                    "chi_iso": isotropic(chi0),
                }
                out(f"  CUBE: {info_cube['path']}  grid={info_cube['nx']}x{info_cube['ny']}x{info_cube['nz']}  "
                    f"range={info_cube['min']:.3g}..{info_cube['max']:.3g} ppm")
                out("  (open in Avogadro; this is the Curie-only point-dipole field built")
                out("   directly from the H_B matrix elements, without a g/A-tensor mapping.)\n")

    report = result_dir / (path.stem + ".curie_report.txt")
    report.write_text("\n".join(text)+"\n", encoding="utf-8")
    summary = result_dir / (path.stem + ".curie_summary.txt")
    write_summary(summary, path, T, info, nuclei_summary, cube_summary)
    print(f"Report saved to : {report}")
    print(f"Summary saved to: {summary}")
    return report


if __name__ == "__main__":
    p=argparse.ArgumentParser(description="Curie contribution to the pNMR shielding tensor")
    p.add_argument("data_file",nargs="?",type=Path,default=DEFAULT_DATA_FILE)
    p.add_argument("--temperature",type=float,default=TEMPERATURE_K)
    a=p.parse_args(); run(a.data_file,a.temperature)
