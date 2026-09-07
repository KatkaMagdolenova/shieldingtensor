"""
===============================================================================
Soncini Curie Shielding and RKB Long-Distance Analysis
===============================================================================

PURPOSE
-------
Reads ReSpect g-tensor + HFCC output and calculates the Curie contribution
to the isotropic paramagnetic NMR shift.

Important:
    sigma_Curie = shielding tensor
    d_*          = delta_* = scalar shift [ppm]

This is NOT the complete experimental NMR shift.

INPUT
-----
Place this script in the same folder as:

    2c-gt.out_gt
    2c-hfs.out_hfcc

The script reads:
    g_total, g_SZ, g_OZ, g_Rel
    A_total, FC, PSO, SD, R1, R2
    gN, geometry, gauge origin, c-scaling

OUTPUT
------
Terminal:
    metadata + checks
    Table 1
    Table 2
    g @ A_total.T matrices
    Curie shielding tensor sigma_Curie [ppm]
    Table 3
    Table 4

Files:
    pnmr_output/pnmr_results.txt   human-readable result report
    pnmr_output/pnmr_results.csv   machine-readable result table
    pnmr_output/pcs_field.cube     3D long-distance PCS field

MAIN EQUATIONS
--------------
Curie shielding tensor:

    sigma_Curie =
        -(muB/(gN*muN)) * beta * S(S+1)/3 * g @ A.T

    beta = 1/(kB*T)

ReSpect A is in MHz:

    A[J] = h * 1e6 * A[MHz]

Isotropic Curie shift:

    sigma_iso = Tr(sigma_Curie)/3
    delta[ppm] = -1e6 * sigma_iso

In the code, delta is named d:
    d_total  = total Curie shift
    d_nonFC  = non-Fermi-contact Curie shift
    d_PDA    = point-dipole shift
    d_LD     = long-distance PCS

DECOMPOSITIONS
--------------
g:
    g = SZ + OZ + Rel

A:
    A = FC + PSO + SD + R1 + R2

nonFC:
    A_nonFC = A - FC
    d_nonFC = d_PSO + d_SD + d_R1 + d_R2

iso/ani:
    d_total = d_iso + d_ani

LONG-DISTANCE / PDA
-------------------
Curie susceptibility:

    chi = mu0 * muB^2 * beta * S(S+1)/3 * g @ g.T

Dipolar tensor:

    D = 3*u*u.T - I,   u = R/|R|

Long-distance PCS:

    d_LD = 1e6 * iso(chi @ D) / (4*pi*R^3)

Point-dipole HFC:

    A_PDA ~ D @ g / R^3
    d_PDA = shift_ppm(g, A_PDA, gN)

d_PDA and d_LD should agree numerically; this is a unit/sign/algebra check.

HOW TO READ THE TABLES
----------------------
Table 1:
    main scalar results per nucleus
    Aiso, d_total, d_nonFC, d_iso, d_ani, d_PDA, d_LD

Table 2:
    Curie shift split by ReSpect A-parts and g-parts

g @ A.T:
    raw 3x3 tensor product before the Curie prefactor
    Tr(g @ A.T)/3 is the scalar entering the isotropic shift

sigma_Curie:
    full 3x3 Curie shielding tensor, printed in ppm
    delta_Curie = -Tr(sigma_Curie)/3

Table 3:
    long-distance tensor residuals
    rel = ||A_actual - A_PDA|| / ||A_actual||
    use mainly as a diagnostic

Table 4:
    all 15 products shift_ppm(g_part, A_part)
    3 g-parts x 5 A-parts
    sum of all 15 = d_total
    no matched/mismatched interpretation is imposed

SETTINGS TO CHANGE
------------------
Usually only edit Settings:

    gt_file, hfs_file     input file names
    output_dir            output folder
    temperature_K         temperature
    spin_S                spin
    center_atom           paramagnetic centre for R-dependent quantities
    near_A, far_A         distance labels only
    write_txt             write text result report: True/False
    write_csv             write CSV: True/False
    write_cube            write cube: True/False
    cube_spacing_A        cube grid spacing
    cube_padding_A        cube margin
    cube_cutoff_A         removes the 1/R^3 singular region

NOTES
-----
- center_atom affects R, PDA, LD, Table 3 and cube, but not d_total itself.
- gauge origin is printed only as metadata; it is not used as the R origin.
- adjusted c is printed only; no simple post-processing rescaling is applied.
- detailed theory and function-by-function explanation belong in the README.

===============================================================================
"""

from __future__ import annotations
import csv, io, math, re
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Dict, List, Optional, Tuple
import numpy as np


# ---------------------------- SETTINGS ---------------------------------------
@dataclass(frozen=True)
class Settings:
    # Input files: same folder as this Python script.
    gt_file: str = "2c-gt.out_gt"
    hfs_file: str = "2c-hfs.out_hfcc"

    # All generated files go into one subfolder next to the script.
    output_dir: str = "pnmr_output"

    temperature_K: float = 298.15
    spin_S: float = 0.5
    center_atom: int = 1
    near_A: float = 4.0
    far_A: float = 8.0

    write_txt: bool = True
    txt_file: str = "pnmr_results.txt"

    write_csv: bool = True
    csv_file: str = "pnmr_results.csv"

    write_cube: bool = True
    cube_file: str = "pcs_field.cube"
    cube_spacing_A: float = 0.4
    cube_padding_A: float = 4.0
    cube_cutoff_A: float = 1.0


S = Settings()

# SI constants + a.u. constants used only by the SI/a.u. check.
mu0 = 4 * math.pi * 1e-7
muB = 9.2740100783e-24
muN = 5.0507837461e-27
kB = 1.380649e-23
h = 6.62607015e-34
Eh = 4.3597447222071e-18
c_au = 137.035999084
mp_me = 1836.15267343
A_per_m = 1e-10
bohr_per_A = 1.889726124565062

beta_si = 1 / (kB * S.temperature_K)
beta_au = Eh / (kB * S.temperature_K)
spin_factor = S.spin_S * (S.spin_S + 1) / 3
_FLOAT = r"[-+]?\d+(?:\.\d*)?(?:[Ee][-+]?\d+)?"


# ----------------------------- PARSING ---------------------------------------
def script_dir() -> Path:
    return Path(__file__).resolve().parent


def resolve_input_file(name: str) -> Path:
    p = Path(name)
    if p.is_absolute() and p.exists():
        return p

    candidate = script_dir() / PureWindowsPath(name).name
    if candidate.exists():
        return candidate

    raise FileNotFoundError(
        f"Input file not found: {candidate}\n"
        f"Put '{PureWindowsPath(name).name}' in the same folder as this script."
    )


def output_path(name: str) -> Path:
    folder = script_dir() / S.output_dir
    folder.mkdir(parents=True, exist_ok=True)
    return folder / Path(name).name


def matrix3(lines: List[str]) -> np.ndarray:
    rows = []
    for line in lines:
        x = [float(v) for v in re.findall(_FLOAT, line)]
        if len(x) >= 3:
            rows.append(x[:3])
        if len(rows) == 3:
            return np.array(rows)
    raise RuntimeError("3x3 matrix not found")


def parse_geometry(text: str) -> Dict[int, Tuple[str, np.ndarray]]:
    lines = text.splitlines()
    start = next(i for i, s in enumerate(lines) if "Molecular geometry [A]" in s)
    pat = re.compile(rf"^\s*([A-Za-z]{{1,3}})\s+(\d+)\s+{_FLOAT}\s+({_FLOAT})\s+({_FLOAT})\s+({_FLOAT})\s*$")
    out = {}
    for line in lines[start + 1:]:
        if out and not line.strip():
            break
        if m := pat.match(line):
            out[int(m.group(2))] = (m.group(1), np.array([float(m.group(i)) for i in (3, 4, 5)]))
    if not out:
        raise RuntimeError("Geometry not found")
    return out


def parse_c_adjust(text: str) -> Tuple[Optional[float], Optional[float]]:
    a = re.search(r"Speed of light is adjusted\s+([-+0-9.Ee]+)\s+times", text)
    c = re.search(r"Speed of light is now:\s+([-+0-9.Ee]+)", text)
    return (float(a.group(1)) if a else None, float(c.group(1)) if c else None)


def parse_gauge_origin(text: str) -> Optional[np.ndarray]:
    m = re.search(r"GO\s*=\s*(" + _FLOAT + r")\s+(" + _FLOAT + r")\s+(" + _FLOAT + r")", text)
    return np.array([float(m.group(i)) for i in (1, 2, 3)]) if m else None


def parse_g_tensor(text: str) -> Dict[str, np.ndarray]:
    keys = {"Spin-Zeeman term": "SZ", "Orbital-Zeeman term": "OZ",
            "Relativistic term": "Rel", "g(B,S)": "total"}
    lines, out = text.splitlines(), {}
    for i, line in enumerate(lines):
        label = line.strip().rstrip(":")
        if label in keys:
            out[keys[label]] = matrix3(lines[i + 1:i + 20])
    if "total" not in out:
        raise RuntimeError("g tensor not found")
    return out


def parse_hfcc(text: str) -> Dict[int, Dict]:
    lines = text.splitlines()
    starts = [(i, int(m.group(1))) for i, line in enumerate(lines)
              if (m := re.search(r"HFCC FOR NUCLEUS\s*#\s*(\d+)\s*=", line))]
    tags = {"A": r"a\(I,S\) \[MHz\]:", "FC": r"Fermi-contact term \(FC\):",
            "PSO": r"Paramagnetic spin orbit term \(PSO\):",
            "SD": r"Spin dipolar term \(SD\):",
            "R1": r"First relativistic term \(R1\):",
            "R2": r"Second relativistic term \(R2\):"}
    if not starts:
        raise RuntimeError("HFCC blocks not found")

    out = {}
    for b, (i0, idx) in enumerate(starts):
        block = lines[i0:starts[b + 1][0] if b + 1 < len(starts) else len(lines)]
        txt = "\n".join(block)
        rec = {}
        if m := re.search(r"g-factor\s*=\s*([-+0-9.Ee]+)", txt):
            rec["gN"] = float(m.group(1))
        for name, pat in tags.items():
            for j, line in enumerate(block):
                if re.match(r"\s*" + pat + r"\s*$", line):
                    rec[name] = matrix3(block[j + 1:j + 15])
                    break
        out[idx] = rec
    return out


# ----------------------------- PHYSICS ---------------------------------------
def iso(M: np.ndarray) -> float:
    return float(np.trace(M) / 3)


def shift_ppm(g: np.ndarray, A_MHz: np.ndarray, gN: float) -> float:
    A_J = h * 1e6 * A_MHz
    sigma_iso = -(muB / (gN * muN)) * beta_si * spin_factor * iso(g @ A_J.T)
    return -1e6 * sigma_iso


def shielding_tensor_ppm(g: np.ndarray, A_MHz: np.ndarray, gN: float) -> np.ndarray:
    """Full Curie shielding tensor multiplied by 1e6, i.e. printed in ppm."""
    A_J = h * 1e6 * A_MHz
    sigma = -(muB / (gN * muN)) * beta_si * spin_factor * (g @ A_J.T)
    return 1e6 * sigma


def split_iso_ani(M: np.ndarray) -> Tuple[float, np.ndarray]:
    x = iso(M)
    return x, M - x * np.eye(3)


def shift_iso_ani(g: np.ndarray, A: np.ndarray, gN: float) -> Tuple[float, float]:
    gi, ga = split_iso_ani(g)
    ai, aa = split_iso_ani(A)
    return shift_ppm(gi * np.eye(3), ai * np.eye(3), gN), shift_ppm(ga, aa, gN)


def shift_ppm_au(g: np.ndarray, A_MHz: np.ndarray, gN: float) -> float:
    A_Eh = h * 1e6 * A_MHz / Eh
    gamma_N = gN / (2 * c_au * mp_me)
    sigma_iso = -beta_au / (2 * c_au * gamma_N) * spin_factor * iso(g @ A_Eh.T)
    return -1e6 * sigma_iso


def susceptibility(g: np.ndarray) -> np.ndarray:
    return mu0 * muB**2 * beta_si * spin_factor * (g @ g.T)


def ld_shift_ppm(chi: np.ndarray, R_A: np.ndarray) -> float:
    R = np.linalg.norm(R_A)
    if R < 1e-9:
        return float("nan")
    u = R_A / R
    D = 3 * np.outer(u, u) - np.eye(3)
    return 1e6 * iso(chi @ D) / (4 * math.pi * (R * A_per_m)**3)


def pda_A_MHz(g: np.ndarray, R_A: np.ndarray, gN: float) -> np.ndarray:
    R = np.linalg.norm(R_A)
    if R < 1e-9:
        return np.full((3, 3), np.nan)
    u = R_A / R
    D = 3 * np.outer(u, u) - np.eye(3)
    pref = mu0 * (gN * muN) * muB / (4 * math.pi * (R * A_per_m)**3)
    return pref * (D @ g) / (h * 1e6)


def principal_g(g: np.ndarray) -> np.ndarray:
    return np.sort(np.sqrt(np.clip(np.linalg.eigvalsh(g @ g.T), 0, None)))


def cross_terms(gp: Dict[str, np.ndarray], rec: Dict, gN: float) -> Dict[Tuple[str, str], float]:
    return {(gn, an): shift_ppm(gp[gn], rec[an], gN)
            for gn in ("SZ", "OZ", "Rel") if gn in gp
            for an in ("FC", "PSO", "SD", "R1", "R2") if an in rec}


def rkb_check(gp: Dict[str, np.ndarray], rec: Dict, R: np.ndarray, gN: float) -> Optional[Dict]:
    if np.linalg.norm(R) < 1e-9 or any(k not in rec for k in ("FC", "PSO", "SD", "R1", "R2")) \
       or any(k not in gp for k in ("SZ", "OZ", "Rel")):
        return None

    Arel = rec["R1"] + rec["R2"]
    pairs = {
        "SD": (rec["SD"], pda_A_MHz(gp["SZ"], R, gN)),
        "PSO": (rec["PSO"], pda_A_MHz(gp["OZ"], R, gN)),
        "REL": (Arel, pda_A_MHz(gp["Rel"], R, gN)),
        "COMB": (rec["PSO"] + Arel,
                 pda_A_MHz(gp["OZ"], R, gN) + pda_A_MHz(gp["Rel"], R, gN)),
    }
    out = {"R_A": float(np.linalg.norm(R)), "FC_norm": float(np.linalg.norm(rec["FC"]))}
    for name, (a, p) in pairs.items():
        n = np.linalg.norm(a)
        out[f"{name}_rel"] = float(np.linalg.norm(a - p) / n) if n > 1e-12 else float("nan")
    return out


# ----------------------------- OUTPUT ----------------------------------------
def print_tables12(rows: List[Dict]) -> None:
    meta = f"{'At':>3} {'El':>2} {'R/A':>6} {'lbl':>6}"

    print("\nTABLE 1 [ppm]")
    head = meta + f" {'Aiso':>9} {'d_total':>11} {'d_nonFC':>10} {'d_iso':>10} {'d_ani':>10} {'d_PDA':>10} {'d_LD':>10}"
    print(head); print("-" * len(head))
    for r in rows:
        print(f"{r['Atom']:3d} {r['El']:>2} {r['R_A']:6.3f} {r['label']:>6} {r['Aiso_MHz']:9.3f} "
              f"{r['d_total']:11.4g} {r['d_nonFC']:10.3g} {r['d_iso']:10.3g} {r['d_ani']:10.3g} "
              f"{r['d_PDA']:10.3g} {r['d_LD']:10.3g}")

    ac = [c for c in ("d_FC", "d_PSO", "d_SD", "d_R1", "d_R2") if any(c in r for r in rows)]
    gc = [c for c in ("d_gSZ", "d_gOZ", "d_gRel") if any(c in r for r in rows)]
    print("\nTABLE 2 [ppm]")
    head = meta + "".join(f" {c[2:]:>9}" for c in ac + gc)
    print(head); print("-" * len(head))
    for r in rows:
        print(f"{r['Atom']:3d} {r['El']:>2} {r['R_A']:6.3f} {r['label']:>6}" +
              "".join(f" {r.get(c, float('nan')):9.3g}" for c in ac + gc))


def print_gAT(rows: List[Dict], mats: Dict[int, np.ndarray],
              sigmas: Dict[int, np.ndarray]) -> None:
    print("\ng @ A_total^T [MHz-like]  +  Curie shielding sigma [ppm]")
    for r in rows:
        idx = r["Atom"]
        M = mats[idx]
        sigma = sigmas[idx]

        print(f"\nAtom #{idx} {r['El']}  R={r['R_A']:.4f} A")

        print("  g @ A_total^T:")
        for x in M:
            print("    " + "  ".join(f"{v:14.6e}" for v in x))
        print(f"    Tr/3={iso(M):.6e}")

        print("  sigma_Curie [ppm]:")
        for x in sigma:
            print("    " + "  ".join(f"{v:14.6e}" for v in x))
        print(f"    sigma_iso={iso(sigma):.6e} ppm  -> delta={r['d_total']:.4g} ppm")


def print_rkb(rows: List[Dict]) -> None:
    if not rows:
        return
    rows = sorted(rows, key=lambda r: r["R_A"])
    print("\nTABLE 3 -- LD tensor residuals  rel=||A-PDA||/||A||")
    head = f"{'At':>3} {'El':>2} {'R/A':>6} {'lbl':>6} {'||FC||':>9} {'SD/SZ':>9} {'PSO/OZ':>9} {'R1R2/Rel':>10} {'COMB':>9}"
    print(head); print("-" * len(head))
    for r in rows:
        print(f"{r['Atom']:3d} {r['El']:>2} {r['R_A']:6.3f} {r['label']:>6} {r['FC_norm']:9.3g} "
              f"{r['SD_rel']:9.3g} {r['PSO_rel']:9.3g} {r['REL_rel']:10.3g} {r['COMB_rel']:9.3g}")
    far = [r for r in rows if r["label"] == "far"] or rows
    av = lambda k: sum(r[k] for r in far) / len(far)
    print(f"far avg: SD={av('SD_rel'):.3g}  PSO={av('PSO_rel'):.3g}  R1R2={av('REL_rel'):.3g}  COMB={av('COMB_rel'):.3g}")


def print_cross(rows: List[Dict], crosses: Dict[int, Dict[Tuple[str, str], float]]) -> None:
    print("\nTABLE 4 -- shift(g_part, A_part) [ppm]")
    anames, gnames = ("FC", "PSO", "SD", "R1", "R2"), ("SZ", "OZ", "Rel")
    for r in rows:
        c = crosses.get(r["Atom"], {})
        if not c:
            continue
        print(f"\nAtom #{r['Atom']} {r['El']}  R={r['R_A']:.4f} A")
        print("        " + "".join(f"{a:>12}" for a in anames))
        for g in gnames:
            print(f"{g:>6}  " + "".join(f"{c.get((g, a), float('nan')):12.4g}" for a in anames))
        print(f"sum={sum(c.values()):.6g}  d_total={r['d_total']:.6g}")



def write_text_report(rows, mats, sigmas, rkb_rows, crosses,
                      g, go, c_scale, c_now, checks) -> Path:
    """Write the same results as a compact ReSpect-like text report."""
    path, buf = output_path(S.txt_file), io.StringIO()
    with redirect_stdout(buf):
        print("=" * 79)
        print(" pNMR CURIE-SHIFT POST-PROCESSING RESULTS")
        print("=" * 79)
        print(f"T={S.temperature_K:g} K   S={S.spin_S:g}   center_atom={S.center_atom}")
        pg = principal_g(g)
        print(f"g_iso={iso(g):.6f}   g_principal={pg[0]:.6f},{pg[1]:.6f},{pg[2]:.6f}")
        if go is not None:
            print(f"GO[A]={go[0]:.4f},{go[1]:.4f},{go[2]:.4f}")
        if c_scale is not None:
            print(f"c_scale={c_scale:g}   c={c_now:g} a.u.")
        print("checks[ppm]: " + "  ".join(f"{k}={v:.2e}" for k, v in checks.items()))
        print_tables12(rows)
        print_gAT(rows, mats, sigmas)
        print_rkb(rkb_rows)
        print_cross(rows, crosses)
        print("\n" + "=" * 79)
        print(" END OF pNMR CURIE-SHIFT RESULTS")
        print("=" * 79)
    path.write_text(buf.getvalue(), encoding="utf-8")
    return path

def write_csv(rows: List[Dict]) -> Path:
    path = output_path(S.csv_file)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return path


def write_cube(atoms: Dict[int, Tuple[str, np.ndarray]], center: np.ndarray, chi: np.ndarray) -> Dict:
    symbols = ("X H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn "
               "Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe").split()
    Z = {e: i for i, e in enumerate(symbols)}

    xyz = np.array([v[1] for v in atoms.values()])
    origin = xyz.min(0) - S.cube_padding_A
    dims = np.ceil((xyz.max(0) + S.cube_padding_A - origin) / S.cube_spacing_A).astype(int) + 1
    nx, ny, nz = map(int, dims)

    xs = origin[0] + S.cube_spacing_A * np.arange(nx)
    ys = origin[1] + S.cube_spacing_A * np.arange(ny)
    zs = origin[2] + S.cube_spacing_A * np.arange(nz)
    X, Y, Zc = np.meshgrid(xs, ys, zs, indexing="ij")
    R = np.stack((X-center[0], Y-center[1], Zc-center[2]), axis=-1)
    rn = np.linalg.norm(R, axis=-1)
    rs = np.where(rn < S.cube_cutoff_A, np.inf, rn)
    u = R / rs[..., None]
    q = np.einsum("...i,ij,...j->...", u, chi, u)
    values = (3*q - np.trace(chi)) / (12*math.pi*(rs*A_per_m)**3) * 1e6

    ob, step = origin * bohr_per_A, S.cube_spacing_A * bohr_per_A
    path = output_path(S.cube_file)
    with open(path, "w") as f:
        f.write("Long-distance pseudocontact shift field delta_pc(r) [ppm]\n\n")
        f.write(f"{len(atoms):5d}{ob[0]:13.6f}{ob[1]:13.6f}{ob[2]:13.6f}\n")
        f.write(f"{nx:5d}{step:13.6f}{0:13.6f}{0:13.6f}\n")
        f.write(f"{ny:5d}{0:13.6f}{step:13.6f}{0:13.6f}\n")
        f.write(f"{nz:5d}{0:13.6f}{0:13.6f}{step:13.6f}\n")
        for el, pos in atoms.values():
            z, p = Z[el], pos * bohr_per_A
            f.write(f"{z:5d}{float(z):13.6f}{p[0]:13.6f}{p[1]:13.6f}{p[2]:13.6f}\n")
        flat = values.ravel()
        for i in range(0, len(flat), 6):
            f.write("".join(f"{v:13.5e}" for v in flat[i:i+6]) + "\n")
    return {"path": path, "nx": nx, "ny": ny, "nz": nz,
            "min": float(values.min()), "max": float(values.max())}


# ------------------------------ MAIN -----------------------------------------
def main() -> None:
    gt = resolve_input_file(S.gt_file).read_text(errors="ignore")
    hfs = resolve_input_file(S.hfs_file).read_text(errors="ignore")
    atoms, gp, hfcc = parse_geometry(gt), parse_g_tensor(gt), parse_hfcc(hfs)
    c_scale, c_now = parse_c_adjust(gt)
    go = parse_gauge_origin(gt)

    g = gp["total"]
    chi = susceptibility(g)
    center = atoms[S.center_atom][1]

    pg = principal_g(g)
    print(f"g_iso={iso(g):.6f}   g_principal={pg[0]:.6f},{pg[1]:.6f},{pg[2]:.6f}")
    if go is not None:
        print(f"GO[A]={go[0]:.4f},{go[1]:.4f},{go[2]:.4f}")
    if c_scale is not None:
        print(f"c_scale={c_scale:g}   c={c_now:g} a.u." + ("  [adjusted]" if abs(c_scale-1) > 1e-12 else ""))

    biggest = max(hfcc, key=lambda i: abs(iso(hfcc[i]["A"])) if "A" in hfcc[i] else 0)
    if biggest != S.center_atom:
        print(f"WARNING: largest |Aiso| is atom #{biggest}; center_atom={S.center_atom}")

    rows, mats, sigmas, rkb_rows, crosses = [], {}, {}, [], {}
    for idx in sorted(hfcc):
        if idx not in atoms or "A" not in hfcc[idx] or "gN" not in hfcc[idx]:
            continue

        el, xyz = atoms[idx]
        rec, gN = hfcc[idx], hfcc[idx]["gN"]
        R = xyz - center
        rn = float(np.linalg.norm(R))
        label = "center" if rn < 1e-9 else "near" if rn < S.near_A else "mid" if rn < S.far_A else "far"

        A = rec["A"]
        dA = {k: shift_ppm(g, rec[k], gN) for k in ("FC", "PSO", "SD", "R1", "R2") if k in rec}
        dg = {k: shift_ppm(gp[k], A, gN) for k in ("SZ", "OZ", "Rel") if k in gp}
        d_total = shift_ppm(g, A, gN)
        d_nonFC = shift_ppm(g, A-rec["FC"], gN) if "FC" in rec else float("nan")
        d_iso, d_ani = shift_iso_ani(g, A, gN)
        A_pda = pda_A_MHz(g, R, gN)
        d_pda = shift_ppm(g, A_pda, gN) if rn > 1e-9 else float("nan")
        d_ld = ld_shift_ppm(chi, R)

        M = g @ A.T
        mats[idx] = M
        sigmas[idx] = shielding_tensor_ppm(g, A, gN)
        mflat = {f"gAT_{a}{b}": float(M[i, j]) for i, a in enumerate("xyz") for j, b in enumerate("xyz")}

        rk = rkb_check(gp, rec, R, gN)
        rkflat = {f"rkb_{k}_rel": (rk[f"{k}_rel"] if rk else float("nan")) for k in ("SD", "PSO", "REL", "COMB")}
        if rk:
            rk.update(Atom=idx, El=el, label=label)
            rkb_rows.append(rk)

        cr = cross_terms(gp, rec, gN)
        crosses[idx] = cr
        crflat = {f"cross_{gn}_{an}_ppm": v for (gn, an), v in cr.items()}

        rows.append({
            "Atom": idx, "El": el, "R_A": rn, "label": label, "Aiso_MHz": iso(A),
            "d_total": d_total, "d_nonFC": d_nonFC, "d_iso": d_iso, "d_ani": d_ani,
            **{f"d_{k}": v for k, v in dA.items()},
            **{f"d_g{k}": v for k, v in dg.items()},
            "d_PDA": d_pda, "d_LD": d_ld,
            **mflat, **rkflat, **crflat
        })

    center_row = next((r for r in rows if r["label"] == "center"), rows[0])
    i0 = center_row["Atom"]
    checks = {
        "SI/AU": abs(shift_ppm_au(g, hfcc[i0]["A"], hfcc[i0]["gN"]) - shift_ppm(g, hfcc[i0]["A"], hfcc[i0]["gN"])),
        "PDA/LD": max(abs(r["d_PDA"]-r["d_LD"]) for r in rows if r["R_A"] > 1e-9),
        "iso+ani": max(abs(r["d_iso"]+r["d_ani"]-r["d_total"]) for r in rows),
        "total=FC+nonFC": max(abs(r["d_total"]-r["d_FC"]-r["d_nonFC"]) for r in rows if "d_FC" in r),
        "nonFC=sum(parts)": max(abs(r["d_nonFC"]-sum(r[f"d_{k}"] for k in ("PSO","SD","R1","R2"))) for r in rows
                                if all(f"d_{k}" in r for k in ("PSO","SD","R1","R2"))),
        "sum(15)": max(abs(sum(crosses[r["Atom"]].values())-r["d_total"]) for r in rows if crosses[r["Atom"]]),
    }
    print("checks[ppm]: " + "  ".join(f"{k}={v:.2e}" for k, v in checks.items()))

    print_tables12(rows)
    print_gAT(rows, mats, sigmas)
    print_rkb(rkb_rows)
    print_cross(rows, crosses)

    if S.write_txt:
        txt_path = write_text_report(rows, mats, sigmas, rkb_rows, crosses,
                                     g, go, c_scale, c_now, checks)
        print(f"\nTXT:  {txt_path}")

    if S.write_csv:
        csv_path = write_csv(rows)
        print(f"\nCSV:  {csv_path}")
    if S.write_cube:
        info = write_cube(atoms, center, chi)
        print(f"CUBE: {info['path']}  grid={info['nx']}x{info['ny']}x{info['nz']}  "
              f"range={info['min']:.3g}..{info['max']:.3g} ppm")


if __name__ == "__main__":
    main()
