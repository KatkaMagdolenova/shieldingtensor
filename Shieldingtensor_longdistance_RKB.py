"""ReSpect S=1/2 pNMR shielding + RKB long-distance diagnostics.
Soncini, AU/Stano-Komorovsky, KM/PDA, gA^T, iso/aniso, cube,
and  the gauge-aware FC/SD/PSO/REL long-distance mapping.
The script also writes a detailed self-explanatory Markdown report.
"""

from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass, field
from pathlib import Path, PureWindowsPath
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

# =============================================================================
# 1. USER SETTINGS 
# =============================================================================

@dataclass(frozen=True)
class Config:
    # ReSpect files
    gt_file: str = r"C:\Users\katka\Desktop\Python\programy\__pycache__\teoretickafyzika\2c-gt.out_gt"
    hfs_file: str = r"C:\Users\katka\Desktop\Python\programy\__pycache__\teoretickafyzika\2c-hfs.out_hfcc"

    
    temperature_K: float = 298.15
    spin_S: float = 0.5

    # Python uses zero-based indexing: 0 = atom #1 in ReSpect.
    paramagnetic_center_index: int = 0

    ld_near_A: float = 4.0
    ld_far_A: float = 8.0

    # RKB derivation assumes the expansion centre O is also the Zeeman gauge origin.
    # Allowed values: "gauge_origin", "paramagnetic_center".
    rkb_origin_mode: str = "gauge_origin"

    detailed_atoms: Tuple[int, ...] = ()

    # Outputs
    output_dir: str = "."
    summary_csv: str = "pnmr_summary.csv"
    matrices_csv: str = "pnmr_matrices.csv"
    decomposition_csv: str = "pnmr_decomposition.csv"
    rkb_csv: str = "rkb_mapping.csv"
    report_md: str = "pnmr_detailed_report.md"

    write_pcs_cube: bool = True
    pcs_cube_file: str = "pcs_long_distance.cube"
    pcs_cube_spacing_A: float = 0.40
    pcs_cube_padding_A: float = 4.0
    pcs_cube_cutoff_A: float = 1.0
    pcs_cube_clip_abs_ppm: Optional[float] = None

# =============================================================================
# 2. CONSTANTS AND SMALL DATA CONTAINERS
# =============================================================================

@dataclass(frozen=True)
class Constants:
    mu0: float = 4.0 * math.pi * 1e-7
    muB: float = 9.2740100783e-24       # J/T
    muN: float = 5.0507837461e-27       # J/T
    kB: float = 1.380649e-23            # J/K
    h: float = 6.62607015e-34           # J s
    Eh: float = 4.3597447222071e-18     # J
    c_au: float = 137.035999084
    mp_me: float = 1836.15267343
    angstrom_m: float = 1e-10
    bohr_per_angstrom: float = 1.889726124565062
    ge_free: float = 2.0023193044

C = Constants()
ZERO3 = np.zeros((3, 3), dtype=float)
_FLOAT_RE = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[Ee][-+]?\d+)?")

@dataclass
class Atom:
    index: int
    symbol: str
    mass: float
    xyz_A: np.ndarray

@dataclass
class NucleusHFCC:
    index: int
    symbol: str
    gN: Optional[float] = None
    mass_number: Optional[int] = None
    tensors_MHz: Dict[str, np.ndarray] = field(default_factory=dict)

@dataclass
class RunData:
    atoms: List[Atom]
    hfcc: Dict[int, NucleusHFCC]
    g_parts: Dict[str, np.ndarray]
    gauge_origin_A: Optional[np.ndarray]
    c_adjust: Optional[float]
    c_now: Optional[float]

# =============================================================================
# 3. GENERIC HELPERS
# =============================================================================

def beta_si(cfg: Config) -> float:
    return 1.0 / (C.kB * cfg.temperature_K)

def beta_au(cfg: Config) -> float:
    return C.Eh / (C.kB * cfg.temperature_K)

def resolve_file(name: str) -> Path:
    p = Path(name)
    if p.exists():
        return p
    base = PureWindowsPath(name).name if ("\\" in name or ":" in name) else p.name
    for candidate in (Path(base), Path("/mnt/data") / base):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(name)

def numbers(line: str) -> List[float]:
    return [float(x) for x in _FLOAT_RE.findall(line)]

def matrix3(lines: Iterable[str]) -> np.ndarray:
    rows: List[List[float]] = []
    for line in lines:
        vals = numbers(line)
        if len(vals) >= 3:
            rows.append(vals[:3])
        if len(rows) == 3:
            return np.array(rows, dtype=float)
    raise RuntimeError("Expected a 3x3 matrix but could not parse one.")

def iso(M: np.ndarray) -> float:
    return float(np.trace(M) / 3.0)

def norm(M: np.ndarray) -> float:
    return float(np.linalg.norm(M))

def antisym_norm(M: np.ndarray) -> float:
    return norm(0.5 * (M - M.T))

def rel_error(actual: np.ndarray, predicted: np.ndarray) -> float:
    den = norm(predicted)
    num = norm(actual - predicted)
    if den < 1e-30:
        return float("nan") if num > 1e-30 else 0.0
    return num / den

def fmt_matrix(M: np.ndarray, digits: int = 8) -> str:
    return "\n".join(" ".join(f"{x:15.{digits}e}" for x in row) for row in M)

def matrix_fields(M: np.ndarray) -> Dict[str, float]:
    return {f"{a}{b}": float(M[i, j]) for i, a in enumerate("xyz") for j, b in enumerate("xyz")}

def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

def sigma_result(sigma: np.ndarray) -> Tuple[np.ndarray, float, float]:
    sigma_iso = iso(sigma)
    return sigma, sigma_iso, -sigma_iso * 1e6

def principal_g(g: np.ndarray) -> np.ndarray:
    vals = np.linalg.eigvalsh(g @ g.T)
    return np.sqrt(np.maximum(vals, 0.0))

def distance_label(cfg: Config, R: float) -> str:
    if R < 1e-12:
        return "center"
    if R < cfg.ld_near_A:
        return "near"
    if R < cfg.ld_far_A:
        return "mid"
    return "far"

# =============================================================================
# 4. RESPECT PARSERS
# =============================================================================

def parse_geometry(text: str) -> List[Atom]:
    lines = text.splitlines()
    start = next((i for i, line in enumerate(lines) if "Molecular geometry [A]" in line), None)
    if start is None:
        raise RuntimeError("Molecular geometry [A] block not found.")

    row_re = re.compile(
        r"^\s*([A-Za-z]{1,3})\s+(\d+)\s+([-+0-9.Ee]+)\s+"
        r"([-+0-9.Ee]+)\s+([-+0-9.Ee]+)\s+([-+0-9.Ee]+)\s*$"
    )
    atoms: List[Atom] = []
    for line in lines[start + 1:]:
        if atoms and not line.strip():
            break
        if m := row_re.match(line):
            atoms.append(Atom(
                index=int(m.group(2)),
                symbol=m.group(1),
                mass=float(m.group(3)),
                xyz_A=np.array([float(m.group(4)), float(m.group(5)), float(m.group(6))]),
            ))
    if not atoms:
        raise RuntimeError("Geometry block found, but no atoms were parsed.")
    return atoms

def parse_gauge_origin(text: str) -> Optional[np.ndarray]:
    m = re.search(r"GO\s*=\s*([-+0-9.Ee]+)\s+([-+0-9.Ee]+)\s+([-+0-9.Ee]+)", text)
    return None if m is None else np.array([float(m.group(i)) for i in (1, 2, 3)])

def parse_c_info(text: str) -> Tuple[Optional[float], Optional[float]]:
    m1 = re.search(r"Speed of light is adjusted\s+([-+0-9.Ee]+)\s+times", text)
    m2 = re.search(r"Speed of light is now:\s+([-+0-9.Ee]+)", text)
    return (float(m1.group(1)) if m1 else None, float(m2.group(1)) if m2 else None)

def parse_g_parts(text: str) -> Dict[str, np.ndarray]:
    labels = {
        "Spin-Zeeman term": "SZ",
        "Orbital-Zeeman term": "OZ",
        "Relativistic term": "REL",
        "g(B,S)": "total",
    }
    lines = text.splitlines()
    out: Dict[str, np.ndarray] = {}
    for i, line in enumerate(lines):
        label = line.strip().rstrip(":")
        if label in labels:
            out[labels[label]] = matrix3(lines[i + 1:i + 20])
    if "total" not in out:
        raise RuntimeError("g(B,S) tensor not found.")
    return out

def parse_hfcc(text: str) -> Dict[int, NucleusHFCC]:
    lines = text.splitlines()
    start_re = re.compile(r"HFCC FOR NUCLEUS\s*#\s*(\d+)\s*=\s*([A-Za-z]+)")
    starts = [(i, int(m.group(1)), m.group(2)) for i, line in enumerate(lines) if (m := start_re.search(line))]
    if not starts:
        raise RuntimeError("No HFCC blocks found.")

    patterns = {
        "A": r"\s*a\(I,S\) \[MHz\]:\s*$",
        "FC": r"\s*Fermi-contact term \(FC\):\s*$",
        "PSO": r"\s*Paramagnetic spin orbit term \(PSO\):\s*$",
        "SD": r"\s*Spin dipolar term \(SD\):\s*$",
        "R1": r"\s*First relativistic term \(R1\):\s*$",
        "R2": r"\s*Second relativistic term \(R2\):\s*$",
    }

    out: Dict[int, NucleusHFCC] = {}
    for b, (start, idx, symbol) in enumerate(starts):
        end = starts[b + 1][0] if b + 1 < len(starts) else len(lines)
        block = lines[start:end]
        block_text = "\n".join(block)
        nuc = NucleusHFCC(index=idx, symbol=symbol)

        if m := re.search(r"g-factor\s*=\s*([-+0-9.Ee]+)", block_text):
            nuc.gN = float(m.group(1))
        if m := re.search(r"Atomic mass number\s*=\s*(\d+)", block_text):
            nuc.mass_number = int(m.group(1))

        for name, pattern in patterns.items():
            for j, line in enumerate(block):
                if re.match(pattern, line):
                    nuc.tensors_MHz[name] = matrix3(block[j + 1:j + 15])
                    break
        out[idx] = nuc
    return out

def load_data(cfg: Config) -> Tuple[RunData, Path, Path]:
    gt = resolve_file(cfg.gt_file)
    hfs = resolve_file(cfg.hfs_file)
    gt_text, hfs_text = gt.read_text(errors="ignore"), hfs.read_text(errors="ignore")
    c_adjust, c_now = parse_c_info(gt_text)
    return RunData(
        atoms=parse_geometry(gt_text),
        hfcc=parse_hfcc(hfs_text),
        g_parts=parse_g_parts(gt_text),
        gauge_origin_A=parse_gauge_origin(gt_text),
        c_adjust=c_adjust,
        c_now=c_now,
    ), gt, hfs

# =============================================================================
# 5. SHIELDING
# =============================================================================

def raw_gAT(g: np.ndarray, A_MHz: np.ndarray) -> np.ndarray:
    return g @ A_MHz.T

def soncini_doublet(cfg: Config, g: np.ndarray, A_MHz: np.ndarray, gN: float) -> Tuple[np.ndarray, float, float]:
    A_J = C.h * 1e6 * A_MHz
    pref = -(C.muB / (gN * C.muN)) * beta_si(cfg) * 0.25
    return sigma_result(pref * (g @ A_J.T))

def wrong_gA_diagnostic(cfg: Config, g: np.ndarray, A_MHz: np.ndarray, gN: float) -> float:
    A_J = C.h * 1e6 * A_MHz
    pref = -(C.muB / (gN * C.muN)) * beta_si(cfg) * 0.25
    return sigma_result(pref * (g @ A_J))[2]

def stano_komorovsky_au(cfg: Config, g: np.ndarray, A_MHz: np.ndarray, gN: float) -> Tuple[np.ndarray, float, float]:
    A_Eh = C.h * 1e6 * A_MHz / C.Eh
    gamma_N_au = gN / (2.0 * C.c_au * C.mp_me)
    pref = -beta_au(cfg) / (8.0 * C.c_au * gamma_N_au)
    return sigma_result(pref * (g @ A_Eh.T))

def susceptibility(cfg: Config, g: np.ndarray) -> np.ndarray:
    spin_factor = cfg.spin_S * (cfg.spin_S + 1.0) / 3.0
    return C.mu0 * C.muB**2 * beta_si(cfg) * spin_factor * (g @ g.T)

def dipole_D(R_A: np.ndarray) -> Tuple[np.ndarray, float]:
    R = float(np.linalg.norm(R_A))
    if R < 1e-12:
        return ZERO3.copy(), R
    u = R_A / R
    return 3.0 * np.outer(u, u) - np.eye(3), R

def km_long_distance(chi: np.ndarray, R_A: np.ndarray) -> Tuple[np.ndarray, float, float]:
    D, R = dipole_D(R_A)
    if R < 1e-12:
        return sigma_result(ZERO3.copy())
    sigma = -(chi @ D) / (4.0 * math.pi * (R * C.angstrom_m) ** 3)
    return sigma_result(sigma)

def pda_hyperfine_MHz(g: np.ndarray, R_A: np.ndarray, gN: float) -> np.ndarray:
    D, R = dipole_D(R_A)
    if R < 1e-12:
        return ZERO3.copy()
    pref_J = C.mu0 / (4.0 * math.pi * (R * C.angstrom_m) ** 3) * (gN * C.muN) * C.muB
    return pref_J * (D @ g) / (C.h * 1e6)

# =============================================================================
# 6. ISO / ANISOTROPIC AND PRINTED-PART DECOMPOSITIONS
# =============================================================================

def split_iso_ani(M: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    x = iso(M)
    Miso = x * np.eye(3)
    return x, Miso, M - Miso

def trace_gA(g: np.ndarray, A_MHz: np.ndarray) -> float:
    return iso(raw_gAT(g, A_MHz))

def trace_decomposition(g: np.ndarray, A_MHz: np.ndarray) -> Dict[str, float]:
    giso, gI, gani = split_iso_ani(g)
    Aiso, AI, Aani = split_iso_ani(A_MHz)
    ii, ia = trace_gA(gI, AI), trace_gA(gI, Aani)
    ai, aa = trace_gA(gani, AI), trace_gA(gani, Aani)
    return {
        "g_iso": giso,
        "A_iso_MHz": Aiso,
        "total_MHzlike": trace_gA(g, A_MHz),
        "iso_iso_MHzlike": ii,
        "iso_ani_MHzlike": ia,
        "ani_iso_MHzlike": ai,
        "ani_ani_MHzlike": aa,
        "reconstructed_MHzlike": ii + ia + ai + aa,
    }

def delta_from_trace(cfg: Config, value_MHzlike: float, gN: float) -> float:
    pref_per_MHz = -(C.muB / (gN * C.muN)) * beta_si(cfg) * 0.25 * C.h * 1e6
    return -1e6 * pref_per_MHz * value_MHzlike

def g_analysis_parts(gp: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    g = gp["total"]
    _, gI, gani = split_iso_ani(g)
    out = {
        "g_total": g,
        "g_total_isoI": gI,
        "g_total_ani": gani,
        "g_free_e_isoI": C.ge_free * np.eye(3),
        "g_shift_total_minus_free_e": g - C.ge_free * np.eye(3),
    }
    for key in ("SZ", "OZ", "REL"):
        if key in gp:
            out[f"g_{key}"] = gp[key]
    if all(k in gp for k in ("SZ", "OZ", "REL")):
        out["g_sum_printed_parts"] = gp["SZ"] + gp["OZ"] + gp["REL"]
    return out

def A_parts(T: Dict[str, np.ndarray], A_pda: np.ndarray) -> Dict[str, np.ndarray]:
    A, FC = T["A"], T.get("FC", ZERO3)
    REL = T.get("R1", ZERO3) + T.get("R2", ZERO3)
    out = {k: T[k] for k in ("A", "FC", "PSO", "SD", "R1", "R2") if k in T}
    out.update({"REL": REL, "nonFC": A - FC, "PDA": A_pda, "nonFC_minus_PDA": (A - FC) - A_pda})
    if all(k in T for k in ("PSO", "SD", "R1", "R2")):
        out["printed_nonFC_sum"] = T["PSO"] + T["SD"] + REL
    return out

# =============================================================================
# 7. RKB LONG-DISTANCE DECOMPOSITION DIAGNOSTICS
# =============================================================================

def rkb_origin(cfg: Config, data: RunData, center_atom: Atom) -> Tuple[np.ndarray, str]:
    mode = cfg.rkb_origin_mode.strip().lower()
    if mode == "gauge_origin":
        if data.gauge_origin_A is not None:
            return np.array(data.gauge_origin_A), "ReSpect gauge origin"
        return np.array(center_atom.xyz_A), "paramagnetic center fallback"
    if mode == "paramagnetic_center":
        return np.array(center_atom.xyz_A), "paramagnetic center"
    raise ValueError("rkb_origin_mode must be 'gauge_origin' or 'paramagnetic_center'.")

def rkb_mapping_for_nucleus(
    cfg: Config, data: RunData, center_atom: Atom, atom: Atom, nuc: NucleusHFCC
) -> Tuple[List[Dict[str, object]], Dict[str, np.ndarray]]:
    if nuc.gN is None:
        raise RuntimeError(f"Missing gN for nucleus #{nuc.index}")

    O, O_label = rkb_origin(cfg, data, center_atom)
    R = atom.xyz_A - O
    T, gp = nuc.tensors_MHz, data.g_parts

    A = T["A"]
    FC = T.get("FC", ZERO3)
    PSO = T.get("PSO", ZERO3)
    SD = T.get("SD", ZERO3)
    REL = T.get("R1", ZERO3) + T.get("R2", ZERO3)

    g, gSZ = gp["total"], gp.get("SZ", ZERO3)
    gOZ, gREL = gp.get("OZ", ZERO3), gp.get("REL", ZERO3)

    LD = {
        "total": pda_hyperfine_MHz(g, R, nuc.gN),
        "SZ": pda_hyperfine_MHz(gSZ, R, nuc.gN),
        "OZ": pda_hyperfine_MHz(gOZ, R, nuc.gN),
        "REL": pda_hyperfine_MHz(gREL, R, nuc.gN),
        "orbrel": pda_hyperfine_MHz(gOZ + gREL, R, nuc.gN),
    }

    R_PSO = PSO - LD["OZ"]
    R_REL = REL - LD["REL"]
    R_orbrel = (PSO + REL) - LD["orbrel"]

    tests = [
        ("FC -> 0", "DIRECT", FC, ZERO3,
         "No delta distribution remains in the smooth retained LD kernel."),
        ("SD -> LD[SZ]", "DIRECT", SD, LD["SZ"],
         "Clean term-by-term SD to spin-Zeeman mapping."),
        ("PSO -> LD[OZ] + gauge", "GAUGE-SENSITIVE", PSO, LD["OZ"],
         "Residual need not vanish separately and is not a pure gauge tensor."),
        ("R1+R2 -> LD[REL] + gauge", "GAUGE-SENSITIVE", REL, LD["REL"],
         "R1+R2 is the available printed proxy for the analytic aREL sector."),
        ("PSO+R1+R2 -> LD[OZ+REL]", "COMBINED", PSO + REL, LD["orbrel"],
         "Relevant orbital+relativistic gauge-cancelling diagnostic."),
        ("A_total-FC -> LD[g_total]", "PRIMARY", A - FC, LD["total"],
         "Strongest practical complete non-contact LD comparison."),
        ("PSO+SD+R1+R2 -> LD[g_total]", "PRIMARY/PARTS", PSO + SD + REL, LD["total"],
         "Same primary test built only from printed HFC parts."),
        ("A_total -> LD[g_total]", "PHYSICAL-LD", A, LD["total"],
         "Expected only when FC/overlap and higher multipoles are negligible."),
    ]

    rows: List[Dict[str, object]] = []
    for name, status, actual, predicted, note in tests:
        da = soncini_doublet(cfg, g, actual, nuc.gN)[2]
        dp = soncini_doublet(cfg, g, predicted, nuc.gN)[2]
        rows.append({
            "Atom": nuc.index,
            "Element": atom.symbol,
            "R_RKB_A": float(np.linalg.norm(R)),
            "RKB_origin": O_label,
            "mapping": name,
            "status": status,
            "actual_norm_MHz": norm(actual),
            "predicted_norm_MHz": norm(predicted),
            "difference_norm_MHz": norm(actual - predicted),
            "relative_difference": rel_error(actual, predicted),
            "actual_Aiso_MHz": iso(actual),
            "predicted_Aiso_MHz": iso(predicted),
            "actual_delta_ppm": da,
            "predicted_delta_ppm": dp,
            "delta_difference_ppm": da - dp,
            "note": note,
        })

    return rows, {
        "R": R,
        "A": A, "FC": FC, "PSO": PSO, "SD": SD, "REL": REL,
        "LD_total": LD["total"], "LD_SZ": LD["SZ"], "LD_OZ": LD["OZ"],
        "LD_REL": LD["REL"], "LD_orbrel": LD["orbrel"],
        "R_PSO": R_PSO, "R_REL": R_REL, "R_orbrel": R_orbrel,
        "R_sum_check": (R_PSO + R_REL) - R_orbrel,
        "A_closure": A - (FC + PSO + SD + REL),
        "nonFC_closure": (A - FC) - (PSO + SD + REL),
    }

# =============================================================================
# 8. ANALYZE ONE NUCLEUS
# =============================================================================

def add_matrix(rows: List[Dict[str, object]], atom: Atom, R: float, family: str, name: str, units: str, M: np.ndarray) -> None:
    row: Dict[str, object] = {
        "Atom": atom.index, "Element": atom.symbol, "Distance_A": R,
        "family": family, "name": name, "units": units, "iso": iso(M),
    }
    row.update(matrix_fields(M))
    rows.append(row)

def analyze_nucleus(
    cfg: Config, atom: Atom, nuc: NucleusHFCC, center_A: np.ndarray, g: np.ndarray, chi: np.ndarray
) -> Tuple[Dict[str, object], List[Dict[str, object]], Dict[str, np.ndarray]]:
    if nuc.gN is None or "A" not in nuc.tensors_MHz:
        raise RuntimeError(f"Nucleus #{nuc.index}: missing gN or total A tensor")

    T = nuc.tensors_MHz
    A = T["A"]
    FC, PSO, SD = T.get("FC", ZERO3), T.get("PSO", ZERO3), T.get("SD", ZERO3)
    REL = T.get("R1", ZERO3) + T.get("R2", ZERO3)
    nonFC = A - FC
    Rvec = atom.xyz_A - center_A
    R = float(np.linalg.norm(Rvec))
    A_pda = pda_hyperfine_MHz(g, Rvec, nuc.gN)

    A_named = {"total": A, "FC": FC, "PSO": PSO, "SD": SD, "REL": REL, "nonFC": nonFC, "PDA": A_pda}
    routes = {
        "Soncini_total": soncini_doublet(cfg, g, A, nuc.gN),
        "AU_total": stano_komorovsky_au(cfg, g, A, nuc.gN),
        "Soncini_FC": soncini_doublet(cfg, g, FC, nuc.gN),
        "Soncini_PSO": soncini_doublet(cfg, g, PSO, nuc.gN),
        "Soncini_SD": soncini_doublet(cfg, g, SD, nuc.gN),
        "Soncini_REL": soncini_doublet(cfg, g, REL, nuc.gN),
        "Soncini_nonFC": soncini_doublet(cfg, g, nonFC, nuc.gN),
        "Soncini_PDA": soncini_doublet(cfg, g, A_pda, nuc.gN),
        "AU_PDA": stano_komorovsky_au(cfg, g, A_pda, nuc.gN),
        "KM_long_distance": km_long_distance(chi, Rvec),
    }

    sig_sonc, _, d_sonc = routes["Soncini_total"]
    sig_au, _, d_au = routes["AU_total"]
    sig_pda, _, d_pda = routes["Soncini_PDA"]
    sig_km, _, d_km = routes["KM_long_distance"]
    d_wrong = wrong_gA_diagnostic(cfg, g, A, nuc.gN)

    summary: Dict[str, object] = {
        "Atom": nuc.index, "Element": atom.symbol, "IsotopeMassNumber": nuc.mass_number, "gN": nuc.gN,
        "Distance_A": R, "LD_label": distance_label(cfg, R),
        "Aiso_MHz": iso(A), "FCiso_MHz": iso(FC), "nonFCiso_MHz": iso(nonFC), "PDAiso_MHz": iso(A_pda),
        "A_nonFC_minus_PDA_norm_MHz": norm(nonFC - A_pda),
        "gAT_iso_MHzlike": iso(raw_gAT(g, A)),
        "Soncini_total_ppm": d_sonc, "AU_total_ppm": d_au, "AU_minus_Soncini_ppm": d_au - d_sonc,
        "sigma_AU_minus_Soncini_norm": norm(sig_au - sig_sonc),
        "wrong_gA_ppm": d_wrong, "wrong_gA_minus_correct_ppm": d_wrong - d_sonc,
        "Soncini_FC_ppm": routes["Soncini_FC"][2], "Soncini_PSO_ppm": routes["Soncini_PSO"][2],
        "Soncini_SD_ppm": routes["Soncini_SD"][2], "Soncini_REL_ppm": routes["Soncini_REL"][2],
        "Soncini_nonFC_ppm": routes["Soncini_nonFC"][2],
        "Soncini_PDA_ppm": d_pda, "AU_PDA_ppm": routes["AU_PDA"][2], "KM_long_distance_ppm": d_km,
        "PDA_minus_KM_ppm": d_pda - d_km, "sigma_PDA_minus_KM_norm": norm(sig_pda - sig_km),
        "Soncini_total_minus_KM_ppm": d_sonc - d_km,
    }

    matrix_rows: List[Dict[str, object]] = []
    # Actual/predicted A tensors.
    for name, M in {**A_named, "R1": T.get("R1", ZERO3), "R2": T.get("R2", ZERO3)}.items():
        add_matrix(matrix_rows, atom, R, "A", name, "MHz", M)
    # Raw g A^T before constants.
    for name, M in A_named.items():
        add_matrix(matrix_rows, atom, R, "gAT", name, "MHz-like", raw_gAT(g, M))
    # Full shielding tensors.
    for name, (sigma, _, _) in routes.items():
        add_matrix(matrix_rows, atom, R, "sigma", name, "dimensionless", sigma)

    details = {
        **A_named,
        "Rvec_center": Rvec,
        "gAT_total": raw_gAT(g, A),
        "gAT_FC": raw_gAT(g, FC),
        "gAT_nonFC": raw_gAT(g, nonFC),
        "sigma_Soncini": sig_sonc,
        "sigma_AU": sig_au,
        "sigma_PDA": sig_pda,
        "sigma_KM": sig_km,
    }
    return summary, matrix_rows, details

# =============================================================================
# 9. OPTIONAL PCS CUBE
# =============================================================================

_ATOMIC_NUMBERS = {
    "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8,
    "F": 9, "Ne": 10, "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15,
    "S": 16, "Cl": 17, "Ar": 18, "K": 19, "Ca": 20, "Sc": 21, "Ti": 22,
    "V": 23, "Cr": 24, "Mn": 25, "Fe": 26, "Co": 27, "Ni": 28, "Cu": 29, "Zn": 30,
}

def write_pcs_cube(path: Path, cfg: Config, atoms: List[Atom], center_A: np.ndarray, chi: np.ndarray) -> Dict[str, object]:
    xyz = np.array([a.xyz_A for a in atoms])
    origin_A = xyz.min(axis=0) - cfg.pcs_cube_padding_A
    top_A = xyz.max(axis=0) + cfg.pcs_cube_padding_A
    nx, ny, nz = map(int, np.ceil((top_A - origin_A) / cfg.pcs_cube_spacing_A).astype(int) + 1)
    values: List[float] = []

    origin_bohr = origin_A * C.bohr_per_angstrom
    step_bohr = cfg.pcs_cube_spacing_A * C.bohr_per_angstrom
    with path.open("w", encoding="utf-8") as f:
        f.write("Long-distance pseudocontact shift field delta_pc(r) [ppm]\n")
        f.write("KM/PDA field only; not full Soncini/contact shift\n")
        f.write(f"{len(atoms):5d}{origin_bohr[0]:13.6f}{origin_bohr[1]:13.6f}{origin_bohr[2]:13.6f}\n")
        f.write(f"{nx:5d}{step_bohr:13.6f}{0.0:13.6f}{0.0:13.6f}\n")
        f.write(f"{ny:5d}{0.0:13.6f}{step_bohr:13.6f}{0.0:13.6f}\n")
        f.write(f"{nz:5d}{0.0:13.6f}{0.0:13.6f}{step_bohr:13.6f}\n")
        for a in atoms:
            Z = _ATOMIC_NUMBERS.get(a.symbol, 0)
            r = a.xyz_A * C.bohr_per_angstrom
            f.write(f"{Z:5d}{float(Z):13.6f}{r[0]:13.6f}{r[1]:13.6f}{r[2]:13.6f}\n")

        count = 0
        for ix in range(nx):
            for iy in range(ny):
                for iz in range(nz):
                    point = origin_A + cfg.pcs_cube_spacing_A * np.array([ix, iy, iz])
                    Rvec = point - center_A
                    R = float(np.linalg.norm(Rvec))
                    delta = 0.0 if R < cfg.pcs_cube_cutoff_A else km_long_distance(chi, Rvec)[2]
                    if cfg.pcs_cube_clip_abs_ppm is not None and cfg.pcs_cube_clip_abs_ppm > 0:
                        delta = float(np.clip(delta, -cfg.pcs_cube_clip_abs_ppm, cfg.pcs_cube_clip_abs_ppm))
                    values.append(delta)
                    f.write(f"{delta:13.5e}")
                    count += 1
                    if count % 6 == 0:
                        f.write("\n")
        if count % 6:
            f.write("\n")

    arr = np.array(values)
    return {
        "nx": nx, "ny": ny, "nz": nz, "npoints": nx * ny * nz,
        "min_ppm": float(arr.min()), "max_ppm": float(arr.max()),
        "spacing_A": cfg.pcs_cube_spacing_A, "padding_A": cfg.pcs_cube_padding_A,
        "cutoff_A": cfg.pcs_cube_cutoff_A, "clip_ppm": cfg.pcs_cube_clip_abs_ppm,
    }

# =============================================================================
# 10. READABLE CONSOLE AND MARKDOWN REPORT
# =============================================================================

def print_summary(rows: List[Dict[str, object]]) -> None:
    print("\n" + "=" * 134)
    print("pNMR SHIELDING / DELTA COMPARISON")
    print("=" * 134)
    print("AU = Stano/Komorovsky a.u. check.  PDA-KM and AU-Soncini should be numerically ~0.")
    print("-" * 134)
    print(f"{'At':>3} {'El':>2} {'R/A':>7} {'Soncini':>12} {'AU':>12} {'AU-Sonc':>11} {'FC':>11} {'nonFC':>11} {'PDA':>11} {'KM':>11} {'PDA-KM':>11}")
    print("-" * 134)
    for r in rows:
        print(
            f"{int(r['Atom']):3d} {str(r['Element']):>2} {float(r['Distance_A']):7.3f} "
            f"{float(r['Soncini_total_ppm']):12.4e} {float(r['AU_total_ppm']):12.4e} "
            f"{float(r['AU_minus_Soncini_ppm']):11.2e} {float(r['Soncini_FC_ppm']):11.4e} "
            f"{float(r['Soncini_nonFC_ppm']):11.4e} {float(r['Soncini_PDA_ppm']):11.4e} "
            f"{float(r['KM_long_distance_ppm']):11.4e} {float(r['PDA_minus_KM_ppm']):11.2e}"
        )
    if rows:
        print("-" * 134)
        print(f"max |AU-Soncini| delta       = {max(abs(float(r['AU_minus_Soncini_ppm'])) for r in rows):.3e} ppm")
        print(f"max ||sigma_AU-sigma_Sonc|| = {max(float(r['sigma_AU_minus_Soncini_norm']) for r in rows):.3e}")
        print(f"max |PDA-KM| delta           = {max(abs(float(r['PDA_minus_KM_ppm'])) for r in rows):.3e} ppm")
        print(f"max ||sigma_PDA-sigma_KM||  = {max(float(r['sigma_PDA_minus_KM_norm']) for r in rows):.3e}")

def print_gAT(rows: List[Dict[str, object]], details: Dict[int, Dict[str, np.ndarray]]) -> None:
    print("\n" + "=" * 88)
    print("RAW g @ A_total.T MATRICES - BEFORE THE SHIELDING PREFACTOR")
    print("=" * 88)
    for r in rows:
        idx = int(r["Atom"])
        print(f"\nAtom #{idx} {r['Element']}   R = {float(r['Distance_A']):.4f} A")
        print(fmt_matrix(details[idx]["gAT_total"]))
        print(f"Tr(g @ A.T)/3 = {float(r['gAT_iso_MHzlike']):.10e} MHz-like")

def print_rkb(rows: List[Dict[str, object]]) -> None:
    print("\n" + "=" * 142)
    print("RKB LONG-DISTANCE DECOMPOSITION")
    print("=" * 142)
    print("DIRECT may converge termwise; GAUGE-SENSITIVE residuals do not have to vanish separately.")
    print("COMBINED tests orbital+relativistic cancellation; PRIMARY is the main non-contact comparison.")
    print("-" * 142)
    print(f"{'At':>3} {'El':>2} {'R/A':>7} {'mapping':<34} {'status':<16} {'||A||':>11} {'||LD||':>11} {'||diff||':>11} {'rel.diff':>11}")
    print("-" * 142)
    for r in rows:
        rel = float(r["relative_difference"])
        rels = "nan" if not np.isfinite(rel) else f"{rel:.3e}"
        print(
            f"{int(r['Atom']):3d} {str(r['Element']):>2} {float(r['R_RKB_A']):7.3f} "
            f"{str(r['mapping']):<34.34} {str(r['status']):<16.16} "
            f"{float(r['actual_norm_MHz']):11.3e} {float(r['predicted_norm_MHz']):11.3e} "
            f"{float(r['difference_norm_MHz']):11.3e} {rels:>11}"
        )

def md_matrix(title: str, M: np.ndarray, units: str = "") -> List[str]:
    suffix = f" [{units}]" if units else ""
    return [f"**{title}{suffix}**", "", "```text", fmt_matrix(M), "```", ""]

def make_report(
    cfg: Config, data: RunData, gt: Path, hfs: Path, center: Atom, chi: np.ndarray,
    summary: List[Dict[str, object]], details: Dict[int, Dict[str, np.ndarray]],
    decomp: List[Dict[str, object]], rkb_rows: List[Dict[str, object]],
    rkb_tensors: Dict[int, Dict[str, np.ndarray]], cube_info: Optional[Dict[str, object]],
) -> str:
    g = data.g_parts["total"]
    O, Olabel = rkb_origin(cfg, data, center)
    gsum = sum((data.g_parts[k] for k in ("SZ", "OZ", "REL") if k in data.g_parts), ZERO3)
    gclosure = norm(g - gsum) if all(k in data.g_parts for k in ("SZ", "OZ", "REL")) else float("nan")

    max_au = max(abs(float(r["AU_minus_Soncini_ppm"])) for r in summary) if summary else float("nan")
    max_au_tensor = max(float(r["sigma_AU_minus_Soncini_norm"]) for r in summary) if summary else float("nan")
    max_pda = max(abs(float(r["PDA_minus_KM_ppm"])) for r in summary) if summary else float("nan")
    max_pda_tensor = max(float(r["sigma_PDA_minus_KM_norm"]) for r in summary) if summary else float("nan")

    lines: List[str] = [
        "# Detailed ReSpect pNMR shielding + RKB long-distance report", "",
        "This report is generated by the same script that produces the CSV files. It separates the original pNMR/shielding analysis from the added RKB long-distance decomposition so that the gauge-sensitive pieces are not over-interpreted.", "",
        "## 1. Inputs and scope", "",
        f"- GT file: `{gt}`",
        f"- HFCC file: `{hfs}`",
        f"- Temperature: `{cfg.temperature_K} K`",
        f"- Effective spin: `S = {cfg.spin_S}` (isolated-doublet formula)",
        f"- Paramagnetic center: atom #{center.index} {center.symbol}, `{center.xyz_A}` A",
        f"- ReSpect gauge origin: `{data.gauge_origin_A}` A",
        f"- RKB expansion origin: `{O}` A ({Olabel})",
        f"- ReSpect c adjustment: `{data.c_adjust}`, c_now=`{data.c_now}`", "",
    ]
    if data.c_adjust is not None and abs(data.c_adjust - 1.0) > 1e-12:
        lines += [
            "> **Important c warning.** The Python post-processing is internally consistent for the tensors ReSpect actually produced, but it cannot convert `g(c_adjust), A(c_adjust)` into physical-c tensors by a simple rescaling. A physical-c comparison requires a ReSpect calculation at the physical c.", "",
        ]

    lines += [
        "## 2. Parsed g tensor and parser checks", "",
        "ReSpect convention is `g(B,S)`. For the printed decomposition the script reads spin-Zeeman, orbital-Zeeman and relativistic blocks.", "",
    ]
    lines += md_matrix("g_total", g)
    for key, label in (("SZ", "g_SZ"), ("OZ", "g_OZ"), ("REL", "g_REL")):
        if key in data.g_parts:
            lines += md_matrix(label, data.g_parts[key])
    lines += [
        f"- `g_iso = {iso(g):.10f}`",
        f"- principal g values from `sqrt(eig(g g^T)) = {principal_g(g)}`",
        f"- antisymmetric norm of g = `{antisym_norm(g):.6e}`",
        f"- decomposition closure `||g_total-(g_SZ+g_OZ+g_REL)|| = {gclosure:.6e}`", "",
        "## 3. Index convention and the raw matrix g A^T", "",
        "ReSpect prints `g(B,S)` and `A(I,S)`. The common effective-spin index is therefore contracted as", "",
        r"\[K_{BI}=\sum_S g_{BS}A_{IS}=(gA^T)_{BI}.\]", "",
        "`K = g @ A.T` is not yet the shielding tensor. Because g is dimensionless and A is in MHz, K has MHz-like units. The physical prefactor is applied only afterwards.", "",
        "For every nucleus, the total raw K matrix is:", "",
    ]
    for r in summary:
        idx = int(r["Atom"])
        lines += [f"### Atom #{idx} {r['Element']} - raw total g A^T", ""]
        lines += md_matrix("g @ A_total.T", details[idx]["gAT_total"], "MHz-like")
        lines += [f"`Tr(g A^T)/3 = {r['gAT_iso_MHzlike']:.10e} MHz-like`", ""]

    lines += [
        "## 4. Soncini / Moon-Patchkovskii doublet shielding", "",
        r"\[\sigma^p=-\frac{\mu_B\beta}{4g_N\mu_N}\,gA^T,\qquad \beta=(k_BT)^{-1}.\]", "",
        "Since ReSpect A is in MHz, the code first converts it to energy: `A[J] = h 10^6 A[MHz]`. The resulting shielding tensor is dimensionless. The reported chemical shift is", "",
        r"\[\sigma_{iso}=\frac13\operatorname{Tr}\sigma,\qquad \delta_{ppm}=-10^6\sigma_{iso}.\]", "",
        "The script also evaluates FC, PSO, SD, printed relativistic (R1+R2), nonFC and PDA contributions through the same linear Soncini map.", "",
        "## 5. Atomic-unit Stano/Komorovsky check", "",
        r"\[\sigma^p_{au}=-\frac{\beta_{au}}{8c\gamma_N^{au}}gA^T.\]", "",
        "This is not an independent physical model; it is the same doublet Curie physics written in Hartree atomic units. Therefore both the isotropic delta and the full 3x3 shielding tensor must agree with the SI implementation.", "",
        f"- max `|delta_AU-delta_Soncini| = {max_au:.6e} ppm`",
        f"- max `||sigma_AU-sigma_Soncini||_F = {max_au_tensor:.6e}`", "",
        "The summary also stores a deliberately wrong `g @ A` diagnostic. If that differs only slightly from `g @ A.T`, it means the input tensors happen to be nearly symmetric; it does not make `g @ A` the correct contraction.", "",
        "## 6. Susceptibility and Kurland-McGarvey long-distance PCS", "",
        r"\[\chi=\mu_0\mu_B^2\beta\frac{S(S+1)}{3}gg^T.\]", "",
        "For S=1/2 the spin factor is 1/4. The long-distance point-dipole shielding is", "",
        r"\[\sigma_{KM}(\mathbf R)=-\frac{1}{4\pi R^3}\chi\left(3\hat R\hat R^T-I\right).\]", "",
    ]
    lines += md_matrix("chi", chi, "m^3")
    lines += [
        f"`chi_iso = {iso(chi):.10e} m^3`", "",
        "This expression contains only the point-dipole long-distance PCS. It does not contain the full local hyperfine tensor or the Fermi-contact contribution.", "",
        "## 7. PDA bridge", "",
        r"\[A_{PDA}[J]=\frac{\mu_0}{4\pi R^3}(g_N\mu_N)\mu_B\left(3\hat R\hat R^T-I\right)g.\]", "",
        "After conversion to MHz and insertion into the Soncini formula, the factors reorganize into the KM expression. Therefore `Soncini[g,A_PDA] == KM` is an algebra/unit/sign/transpose check, not evidence that the real quantum-chemical A tensor equals A_PDA.", "",
        f"- max `|delta_PDA-delta_KM| = {max_pda:.6e} ppm`",
        f"- max `||sigma_PDA-sigma_KM||_F = {max_pda_tensor:.6e}`", "",
        "## 8. Main scalar results", "",
        "|Atom|El|R/A|Soncini|AU|FC|nonFC|SD|PDA|KM|AU-Sonc|PDA-KM|",
        "|---:|:--:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in summary:
        lines.append(
            f"|{r['Atom']}|{r['Element']}|{r['Distance_A']:.3f}|{r['Soncini_total_ppm']:.6e}|{r['AU_total_ppm']:.6e}|"
            f"{r['Soncini_FC_ppm']:.6e}|{r['Soncini_nonFC_ppm']:.6e}|{r['Soncini_SD_ppm']:.6e}|"
            f"{r['Soncini_PDA_ppm']:.6e}|{r['KM_long_distance_ppm']:.6e}|{r['AU_minus_Soncini_ppm']:.2e}|{r['PDA_minus_KM_ppm']:.2e}|"
        )

    lines += [
        "", "## 9. Why nonFC is not automatically LongDistance", "",
        r"\[A_{nonFC}=A_{total}-A_{FC}.\]", "",
        "This removes only the explicit Fermi-contact part. The remaining tensor can still contain local/near-field orbital, spin-dipolar and relativistic effects. A true point-dipole limit requires the real non-contact tensor to approach A_PDA. Therefore `Soncini_nonFC != KM` at finite distance is not by itself a bug.", "",
        "## 10. Iso/anisotropic decomposition", "",
        r"\[T=T_{iso}I+T_{ani},\qquad T_{iso}=\frac13\operatorname{Tr}T,\qquad \operatorname{Tr}T_{ani}=0.\]", "",
        r"\[\frac13\operatorname{Tr}(gA^T)=g_{iso}A_{iso}+\frac13\operatorname{Tr}(g_{ani}A_{ani}^T).\]", "",
        "The iso-ani and ani-iso cross terms vanish analytically because the anisotropic pieces are traceless. The complete cross-product table is written to `pnmr_decomposition.csv`.", "",
        "## 11. Added RKB long-distance decomposition", "",
        "The operator derivation gives the schematic stationary mapping", "",
        r"\[FC\to0,\qquad SD\to LD[g_{SZ}],\]",
        r"\[PSO\to LD[g_{OZ}]+G_{orb},\qquad REL\to LD[g_{REL}]+G_{rel}.\]", "",
        "The complete gauge operator is nonzero as an operator; only its complete stationary diagonal expectation vanishes. Therefore the isolated PSO and REL residuals must not be called pure gauge tensors.", "",
        r"\[R_{PSO}=A_{PSO}-LD[g_{OZ}],\qquad R_{REL}=(A_{R1}+A_{R2})-LD[g_{REL}].\]", "",
        "At finite R these can contain gauge rearrangement plus higher multipoles, finite-distance effects, printed-decomposition mismatch and numerical error. The more meaningful combined residual is", "",
        r"\[R_{orbrel}=R_{PSO}+R_{REL}=(A_{PSO}+A_{R1}+A_{R2})-LD[g_{OZ}+g_{REL}].\]", "",
        "The strongest practical test is", "",
        r"\[(A_{total}-A_{FC})\to LD[g_{total}].\]", "",
        "### RKB numerical table", "",
        "|Atom|El|R_RKB/A|mapping|status|diff norm MHz|rel.diff|delta diff ppm|",
        "|---:|:--:|---:|:--|:--|---:|---:|---:|",
    ]
    for r in rkb_rows:
        rel = r["relative_difference"]
        rel_s = "nan" if not np.isfinite(rel) else f"{rel:.5e}"
        lines.append(
            f"|{r['Atom']}|{r['Element']}|{r['R_RKB_A']:.3f}|{r['mapping']}|{r['status']}|"
            f"{r['difference_norm_MHz']:.5e}|{rel_s}|{r['delta_difference_ppm']:.5e}|"
        )

    chosen = set(cfg.detailed_atoms)
    if summary:
        chosen.add(int(summary[0]["Atom"]))
        far = next((int(r["Atom"]) for r in summary if r["LD_label"] == "far"), None)
        if far is not None:
            chosen.add(far)

    lines += ["", "## 12. Detailed selected nuclei", ""]
    for idx in sorted(chosen):
        if idx not in details or idx not in rkb_tensors:
            continue
        d, rt = details[idx], rkb_tensors[idx]
        r = next(x for x in summary if int(x["Atom"]) == idx)
        lines += [f"### Atom #{idx} {r['Element']}", "", f"Distance from paramagnetic center: `{r['Distance_A']:.6f} A`", ""]
        for title, key, units in (
            ("A_total", "total", "MHz"), ("A_FC", "FC", "MHz"), ("A_nonFC", "nonFC", "MHz"),
            ("A_PDA", "PDA", "MHz"), ("g A_total^T", "gAT_total", "MHz-like"),
            ("sigma_Soncini", "sigma_Soncini", "dimensionless"), ("sigma_AU", "sigma_AU", "dimensionless"),
            ("sigma_PDA", "sigma_PDA", "dimensionless"), ("sigma_KM", "sigma_KM", "dimensionless"),
        ):
            lines += md_matrix(title, d[key], units)
        lines += [
            f"- `||A_nonFC-A_PDA|| = {r['A_nonFC_minus_PDA_norm_MHz']:.6e} MHz`",
            f"- `delta_Soncini = {r['Soncini_total_ppm']:.10e} ppm`",
            f"- `delta_nonFC = {r['Soncini_nonFC_ppm']:.10e} ppm`",
            f"- `delta_KM = {r['KM_long_distance_ppm']:.10e} ppm`", "",
        ]
        lines += md_matrix("R_PSO = PSO-LD[OZ] (gauge-sensitive residual)", rt["R_PSO"], "MHz")
        lines += md_matrix("R_REL = (R1+R2)-LD[REL] (gauge-sensitive residual)", rt["R_REL"], "MHz")
        lines += md_matrix("R_orbrel combined residual", rt["R_orbrel"], "MHz")
        lines += md_matrix("Primary residual (A_total-FC)-LD[g_total]", (rt["A"] - rt["FC"]) - rt["LD_total"], "MHz")
        lines += [
            f"- algebraic residual-sum check `||(R_PSO+R_REL)-R_orbrel|| = {norm(rt['R_sum_check']):.6e} MHz`",
            f"- A decomposition closure `||A-(FC+PSO+SD+R1+R2)|| = {norm(rt['A_closure']):.6e} MHz`",
            f"- nonFC closure `||(A-FC)-(PSO+SD+R1+R2)|| = {norm(rt['nonFC_closure']):.6e} MHz`", "",
        ]

    lines += [
        "## 13. Units used throughout", "",
        "|Quantity|Units|Meaning|",
        "|:--|:--|:--|",
        "|g|dimensionless|ReSpect EPR tensor|",
        "|A, FC, PSO, SD, R1, R2, PDA|MHz|ReSpect / constructed hyperfine tensors|",
        "|g A^T|MHz-like|raw tensor contraction before physical constants|",
        "|A energy in Soncini|J|`h 10^6 A[MHz]`|",
        "|A energy in AU check|Hartree|`h 10^6 A[MHz]/E_h`|",
        "|R in report|Angstrom|geometry|",
        "|R in KM/PDA prefactors|m|Angstrom x 1e-10|",
        "|chi|m^3|single-molecule Curie susceptibility|",
        "|sigma|dimensionless|shielding tensor|",
        "|delta|ppm|`-Tr(sigma) 10^6 / 3`|",
        "|cube values|ppm|KM/PDA pseudocontact field only|", "",
        "## 14. Exact-state formulation: what is and is not implemented", "",
        "The accompanying notes discuss a future exact-state backend based on electronic energies and matrix elements of `dH/dB`, `dH/dmu` and possibly `d2H/dB dmu`. The present ReSpect GT/HFCC files do not provide those state-resolved matrices, so this script intentionally does **not** invent an exact-state calculation. Its implemented backend is the isolated S=1/2 spin-Hamiltonian route `g,A -> sigma`.", "",
        "## 15. What to edit for a new calculation", "",
        "- New ReSpect calculation: change only `gt_file` and `hfs_file` first.",
        "- Temperature: change `temperature_K`; all Curie terms scale through beta.",
        "- Paramagnetic center for KM/PDA/cube: change `paramagnetic_center_index` (zero-based).",
        "- RKB expansion origin: normally keep `rkb_origin_mode='gauge_origin'` for the termwise derivation; use `'paramagnetic_center'` only deliberately.",
        "- Detailed report nuclei: set `detailed_atoms=(8,20,21)` using ReSpect atom numbers.",
        "- Distance labels: change `ld_near_A` / `ld_far_A`; these labels do not change physics.",
        "- Cube: set `write_pcs_cube=False` to disable it, or change spacing/padding/cutoff.",
        "- Do not change `g @ A.T` to `g @ A`; that changes the contracted index convention.",
        "- Do not set `spin_S` to another value and assume the Soncini backend is now general: this script validates S=1/2 because its main formula contains the doublet factor 1/4.", "",
        "## 16. Output files", "",
        "- `pnmr_summary.csv`: one scalar summary row per nucleus.",
        "- `pnmr_matrices.csv`: one 3x3 matrix per row; includes A tensors, raw gA^T matrices and shielding tensors.",
        "- `pnmr_decomposition.csv`: complete g-part x A-part iso/anisotropic decomposition.",
        "- `rkb_mapping.csv`: direct, gauge-sensitive, combined and primary long-distance diagnostics.",
        "- `pnmr_detailed_report.md`: this report.",
        "- `pcs_long_distance.cube`: optional KM/PDA scalar field.", "",
    ]
    if cube_info is not None:
        lines += [
            "## 17. Cube generated in this run", "",
            f"Grid: `{cube_info['nx']} x {cube_info['ny']} x {cube_info['nz']} = {cube_info['npoints']}` points",
            f"Spacing: `{cube_info['spacing_A']} A`; padding: `{cube_info['padding_A']} A`; cutoff: `{cube_info['cutoff_A']} A`",
            f"Range: `{cube_info['min_ppm']:.6e} .. {cube_info['max_ppm']:.6e} ppm`",
            "The cube is only the long-distance pseudocontact field. It is not spin density, FC shift or full Soncini shift.", "",
        ]
    return "\n".join(lines)

# =============================================================================
# 11. MAIN PROGRAM
# =============================================================================

def main() -> None:
    cfg = Config()
    if abs(cfg.spin_S - 0.5) > 1e-12:
        raise ValueError("This script implements the isolated S=1/2 doublet shielding formula; keep spin_S=0.5.")

    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    data, gt, hfs = load_data(cfg)
    if not (0 <= cfg.paramagnetic_center_index < len(data.atoms)):
        raise IndexError("paramagnetic_center_index is outside the atom list")

    center = data.atoms[cfg.paramagnetic_center_index]
    g = data.g_parts["total"]
    chi = susceptibility(cfg, g)

    print(f"Parsed atoms: {len(data.atoms)}; HFCC blocks: {len(data.hfcc)}")
    print(f"Paramagnetic center: atom #{center.index} {center.symbol} at {center.xyz_A} A")
    print(f"g_iso = {iso(g):.10f}; principal g = {principal_g(g)}; chi_iso = {iso(chi):.10e} m^3")
    if data.gauge_origin_A is not None:
        print(f"Gauge origin = {data.gauge_origin_A} A; |GO-center| = {np.linalg.norm(data.gauge_origin_A-center.xyz_A):.6f} A")
    if data.c_adjust is not None and abs(data.c_adjust - 1.0) > 1e-12:
        print(f"WARNING: ReSpect used c_adjust={data.c_adjust}; c_now={data.c_now}")

    summary: List[Dict[str, object]] = []
    matrices: List[Dict[str, object]] = []
    decomp: List[Dict[str, object]] = []
    rkb_rows: List[Dict[str, object]] = []
    details: Dict[int, Dict[str, np.ndarray]] = {}
    rkb_tensors: Dict[int, Dict[str, np.ndarray]] = {}
    gparts = g_analysis_parts(data.g_parts)

    for idx in sorted(data.hfcc):
        nuc = data.hfcc[idx]
        atom_i = idx - 1
        if not (0 <= atom_i < len(data.atoms)) or "A" not in nuc.tensors_MHz:
            continue
        atom = data.atoms[atom_i]

        row, mrows, det = analyze_nucleus(cfg, atom, nuc, center.xyz_A, g, chi)
        summary.append(row)
        matrices.extend(mrows)
        details[idx] = det

        A_pda = det["PDA"]
        for g_name, gpart in gparts.items():
            for A_name, Apart in A_parts(nuc.tensors_MHz, A_pda).items():
                d = trace_decomposition(gpart, Apart)
                decomp.append({
                    "Atom": idx, "Element": atom.symbol, "Distance_A": row["Distance_A"],
                    "g_part": g_name, "A_part": A_name, **d,
                    "delta_total_ppm": delta_from_trace(cfg, d["total_MHzlike"], nuc.gN),
                    "delta_iso_iso_ppm": delta_from_trace(cfg, d["iso_iso_MHzlike"], nuc.gN),
                    "delta_ani_ani_ppm": delta_from_trace(cfg, d["ani_ani_MHzlike"], nuc.gN),
                })

        rr, rt = rkb_mapping_for_nucleus(cfg, data, center, atom, nuc)
        rkb_rows.extend(rr)
        rkb_tensors[idx] = rt

    cube_info = None
    if cfg.write_pcs_cube:
        cube_info = write_pcs_cube(out / cfg.pcs_cube_file, cfg, data.atoms, center.xyz_A, chi)

    print_summary(summary)
    print_gAT(summary, details)
    print_rkb(rkb_rows)

    write_csv(out / cfg.summary_csv, summary)
    write_csv(out / cfg.matrices_csv, matrices)
    write_csv(out / cfg.decomposition_csv, decomp)
    write_csv(out / cfg.rkb_csv, rkb_rows)
    report = make_report(cfg, data, gt, hfs, center, chi, summary, details, decomp, rkb_rows, rkb_tensors, cube_info)
    (out / cfg.report_md).write_text(report, encoding="utf-8")

    print("\nFiles written:")
    for name in (cfg.summary_csv, cfg.matrices_csv, cfg.decomposition_csv, cfg.rkb_csv, cfg.report_md):
        print(" ", (out / name).resolve())
    if cfg.write_pcs_cube:
        print(" ", (out / cfg.pcs_cube_file).resolve())

if __name__ == "__main__":
    main()
