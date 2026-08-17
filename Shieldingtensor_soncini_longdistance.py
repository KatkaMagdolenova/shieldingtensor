"""
PURPOSE
=======
The script post-processes ReSpect output files for an S = 1/2 paramagnetic
system. It reads:

    1. the EPR g tensor and its ReSpect decomposition from the GT output,
    2. the hyperfine A tensor and its FC/PSO/SD/R1/R2 decomposition from the
       HFS/HFCC output,
    3. the molecular geometry, gauge origin, and the speed-of-light setting.

Then it calculates, for every nucleus, the paramagnetic NMR shielding tensor
and the chemical shift in several equivalent or diagnostic ways.

THEORY ROADMAP USED IN THE CODE
===============================
1. Soncini / Moon--Patchkovskii doublet Curie term

       sigma^p = -(mu_B * beta)/(4 * g_N * mu_N)  g A^T

   This is the S = 1/2 limit of the finite-temperature pNMR theory based on
   the Helmholtz free energy. beta = 1/(k_B T). The factor 1/4 is
   S(S+1)/3 for S = 1/2.

2. ReSpect tensor convention

       g(B,S): first index = external magnetic-field component B,
               second index = effective-spin component S
       A(I,S): first index = nuclear magnetic moment component I,
               second index = effective-spin component S

   The common contracted index is S, therefore the product is

       sigma(B,I) proportional to g(B,S) A(I,S)

   which is matrix multiplication

       g @ A.T

   and not g @ A.

3. Komorovsky/Stano atomic-unit check

   The same S = 1/2 Curie formula is evaluated in Hartree atomic units. It is
   not a separate physical model. It is a unit-consistency check: if the SI
   Soncini implementation and the a.u. implementation agree, the MHz -> J,
   MHz -> Hartree, beta, gamma_N, and sign conventions are consistent.

4. Kurland--McGarvey / long-distance pseudocontact limit

       sigma_pc(r) = - chi [3 Rhat Rhat^T - I] / (4*pi*R^3)

   where R is the vector from the paramagnetic center to the observed nucleus
   or grid point. This is only the long-distance point-dipole PCS field, not
   the full Soncini/contact shift.

5. Magnetic susceptibility used in the long-distance term

       chi = mu0 * muB^2 * beta * S(S+1)/3 * g g^T

   For S = 1/2, S(S+1)/3 = 1/4.

6. PDA bridge

   The point-dipole approximation to the hyperfine tensor is built as

       A_PDA[J] = mu0/(4*pi*R^3) * (g_N mu_N) * mu_B
                  * [3 Rhat Rhat^T - I] @ g

   After converting A_PDA to MHz and inserting it into the Soncini formula,
   the result must be identical to the Kurland--McGarvey long-distance result:

       Soncini[g, A_PDA] == LongDistance/KM

   This equality is a sanity check of units, signs, transposes, and the R^-3
   prefactor. It does not mean that the real quantum-chemical A tensor equals
   the point-dipole A_PDA.

7. Decompositions

   ReSpect gives

       g = g_spin_zeeman + g_orbital_zeeman + g_relativistic
       A = A_FC + A_PSO + A_SD + A_R1 + A_R2

   The script evaluates how every part contributes to the isotropic shift.
   It also splits tensors into isotropic and anisotropic parts:

       T = T_iso I + T_ani,
       T_iso = Tr(T)/3,
       Tr(T_ani) = 0.

   Therefore

       (1/3) Tr(g A^T) = g_iso A_iso
                         + (1/3) Tr(g_ani A_ani^T)

   because the iso--ani and ani--iso cross terms have zero trace.

"""

from __future__ import annotations

import csv
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path, PureWindowsPath
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


# =============================================================================
# 1. USER SETTINGS
# =============================================================================

@dataclass(frozen=True)
class Config:
    """All settings that the user is expected to change."""

    gt_file: str = r"C:\Users\katka\Desktop\Python\programy\__pycache__\teoretickafyzika\2c-gt.out_gt"
    hfs_file: str = r"C:\Users\katka\Desktop\Python\programy\__pycache__\teoretickafyzika\2c-hfs.out_hfcc"

    temperature_K: float = 298.15
    spin_S: float = 0.5

    # First atom as paramagnetic center. Python uses zero-based indexing.
    paramagnetic_center_index: int = 0

    # Labels are only practical diagnostics, not strict physical cutoffs.
    ld_near_A: float = 4.0
    ld_far_A: float = 8.0

    output_dir: str = "."
    main_csv: str = "shielding_results_explained.csv"
    decomp_csv: str = "shielding_iso_decomposition_explained.csv"
    report_md: str = "shielding_report_explained.md"

    # The console stays compact. Detailed blocks are written to the report.
    print_console_table: bool = True

    # Optional visualization of the long-distance pseudocontact shift field.
    # The cube contains delta_pc(r) in ppm, not the full Soncini/contact shift.
    write_pcs_cube: bool = True
    pcs_cube_file: str = "pcs_long_distance_field_ppm_explained.cube"
    pcs_cube_spacing_A: float = 0.40
    pcs_cube_padding_A: float = 4.0
    pcs_cube_cutoff_A: float = 1.0
    pcs_cube_clip_abs_ppm: Optional[float] = None


# =============================================================================
# 2. PHYSICAL CONSTANTS
# =============================================================================

@dataclass(frozen=True)
class Constants:
    """Physical constants. SI units unless stated otherwise."""

    mu0: float = 4.0 * math.pi * 1e-7          # N A^-2
    muB: float = 9.2740100783e-24              # J T^-1
    muN: float = 5.0507837461e-27              # J T^-1
    kB: float = 1.380649e-23                   # J K^-1
    h: float = 6.62607015e-34                  # J s
    Eh: float = 4.3597447222071e-18            # J
    c_au: float = 137.035999084                # speed of light in Hartree a.u.
    mp_me: float = 1836.15267343               # proton/electron mass ratio
    angstrom_m: float = 1e-10                  # m
    bohr_per_angstrom: float = 1.889726124565062 # 1 Angstrom in Bohr
    ge_free: float = 2.0023193044              # diagnostic free-electron g value


C = Constants()


def beta_si(cfg: Config) -> float:
    """Thermal beta in SI units, beta = 1/(k_B T), in J^-1."""
    return 1.0 / (C.kB * cfg.temperature_K)


def beta_au(cfg: Config) -> float:
    """Thermal beta in Hartree atomic units, beta = E_h/(k_B T)."""
    return C.Eh / (C.kB * cfg.temperature_K)


# =============================================================================
# 3. DATA CONTAINERS
# =============================================================================

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
    mass: Optional[float] = None
    mass_number: Optional[int] = None
    position_A: Optional[np.ndarray] = None
    tensors_MHz: Dict[str, np.ndarray] = field(default_factory=dict)


@dataclass
class RunData:
    atoms: List[Atom]
    hfcc: Dict[int, NucleusHFCC]
    g_parts: Dict[str, np.ndarray]
    gauge_origin_A: Optional[np.ndarray]
    c_adjust: Optional[float]
    c_now: Optional[float]


@dataclass
class NucleusResult:
    """One compact, readable result row for the main CSV and console."""

    Atom: int
    Element: str
    IsotopeMassNumber: Optional[int]
    gN: float
    Distance_A: float
    LD_label: str

    Aiso_MHz: float
    FCiso_MHz: float
    nonFCiso_MHz: float
    PDAiso_MHz: float
    A_nonFC_minus_PDA_norm_MHz: float

    Soncini_total_ppm: float
    Soncini_FC_ppm: float
    Soncini_nonFC_ppm: float
    Soncini_SD_only_ppm: float
    Soncini_PDA_from_g_ppm: float

    Stano_checked_total_ppm: float
    Stano_checked_PDA_ppm: float
    Stano_minus_Soncini_ppm: float

    LongDistance_ppm: float
    SonciniPDA_minus_LongDistance_ppm: float
    Soncini_minus_LongDistance_ppm: float


# =============================================================================
# 4. BASIC UTILITIES
# =============================================================================

_FLOAT_RE = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[Ee][-+]?\d+)?")


def resolve_input_file(filename: str) -> Path:
    """
    Resolve hard-coded Windows paths, local basenames, and /mnt/data files.

    This allows the same script to run both on the local Windows machine and
    inside a notebook/sandbox where only the basename exists.
    """
    p = Path(filename)
    if p.exists():
        return p

    basename = PureWindowsPath(filename).name if ("\\" in filename or ":" in filename) else p.name

    for candidate in [Path(basename), Path("/mnt/data") / basename]:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"Input file not found: {filename}")


def read_text(path: Path) -> str:
    return path.read_text(errors="ignore")


def numbers_from_line(line: str) -> List[float]:
    return [float(x) for x in _FLOAT_RE.findall(line)]


def matrix3(lines: Iterable[str]) -> np.ndarray:
    """Read the first numerical 3x3 matrix from a block of text lines."""
    rows: List[List[float]] = []
    for line in lines:
        nums = numbers_from_line(line)
        if len(nums) >= 3:
            rows.append(nums[:3])
        if len(rows) == 3:
            return np.array(rows, dtype=float)
    raise RuntimeError("Could not read a 3x3 matrix from the expected block.")


def fmt_matrix(M: np.ndarray, precision: int = 8) -> str:
    return "\n".join(" ".join(f"{x:16.{precision}e}" for x in row) for row in M)


def tensor_iso(M: np.ndarray) -> float:
    return float(np.trace(M) / 3.0)


def antisymmetric_norm(M: np.ndarray) -> float:
    return float(np.linalg.norm(0.5 * (M - M.T)))


def shielding_from_sigma(sigma: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """Return sigma tensor, sigma_iso, and delta_ppm = -sigma_iso * 1e6."""
    sigma_iso = tensor_iso(sigma)
    delta_ppm = -sigma_iso * 1e6
    return sigma, sigma_iso, delta_ppm


# =============================================================================
# 5. RESPECT PARSERS
# =============================================================================


def parse_geometry(text: str) -> List[Atom]:
    """Parse the 'Molecular geometry [A]' table from ReSpect output."""
    lines = text.splitlines()
    start = next((i for i, line in enumerate(lines) if "Molecular geometry [A]" in line), None)
    if start is None:
        raise RuntimeError("Molecular geometry [A] block not found.")

    row_re = re.compile(
        r"^\s*([A-Za-z]{1,3})\s+(\d+)\s+"
        r"([-+]?\d+(?:\.\d*)?(?:[Ee][-+]?\d+)?)\s+"
        r"([-+]?\d+(?:\.\d*)?(?:[Ee][-+]?\d+)?)\s+"
        r"([-+]?\d+(?:\.\d*)?(?:[Ee][-+]?\d+)?)\s+"
        r"([-+]?\d+(?:\.\d*)?(?:[Ee][-+]?\d+)?)\s*$"
    )

    atoms: List[Atom] = []
    for line in lines[start + 1:]:
        if atoms and line.strip() == "":
            break
        m = row_re.match(line)
        if not m:
            continue
        atoms.append(
            Atom(
                index=int(m.group(2)),
                symbol=m.group(1),
                mass=float(m.group(3)),
                xyz_A=np.array([float(m.group(4)), float(m.group(5)), float(m.group(6))]),
            )
        )

    if not atoms:
        raise RuntimeError("Geometry block found, but no atoms were parsed.")
    return atoms


def parse_common_gauge_origin(text: str) -> Optional[np.ndarray]:
    m = re.search(r"GO\s*=\s*([-+0-9.Ee]+)\s+([-+0-9.Ee]+)\s+([-+0-9.Ee]+)", text)
    if not m:
        return None
    return np.array([float(m.group(1)), float(m.group(2)), float(m.group(3))])


def parse_speed_of_light_info(text: str) -> Tuple[Optional[float], Optional[float]]:
    adjusted = None
    actual = None
    m = re.search(r"Speed of light is adjusted\s+([-+0-9.Ee]+)\s+times", text)
    if m:
        adjusted = float(m.group(1))
    m = re.search(r"Speed of light is now:\s+([-+0-9.Ee]+)", text)
    if m:
        actual = float(m.group(1))
    return adjusted, actual


def parse_g_tensor_parts(text: str) -> Dict[str, np.ndarray]:
    """Parse total g(B,S) and ReSpect's printed g decomposition."""
    label_to_key = {
        "Spin-Zeeman term": "g_spin_zeeman",
        "Orbital-Zeeman term": "g_orbital_zeeman",
        "Relativistic term": "g_relativistic",
        "g(B,S)": "g_total",
    }

    parts: Dict[str, np.ndarray] = {}
    lines = text.splitlines()
    for i, line in enumerate(lines):
        label = line.strip().rstrip(":")
        if label in label_to_key:
            parts[label_to_key[label]] = matrix3(lines[i + 1:i + 20])

    if "g_total" not in parts:
        raise RuntimeError("g(B,S) tensor not found in ReSpect GT output.")
    return parts


def parse_hfcc_blocks(text: str) -> Dict[int, NucleusHFCC]:
    """
    Parse all HFCC blocks.

    Important: the total A tensor is matched only by the exact line
        a(I,S) [MHz]:
    so that Gordon decomposition blocks do not overwrite the real total A tensor.
    """
    lines = text.splitlines()
    block_re = re.compile(r"HFCC FOR NUCLEUS\s*#\s*(\d+)\s*=\s*([A-Za-z]+)")

    starts: List[Tuple[int, int, str]] = []
    for i, line in enumerate(lines):
        m = block_re.search(line)
        if m:
            starts.append((i, int(m.group(1)), m.group(2)))
    if not starts:
        raise RuntimeError("No HFCC FOR NUCLEUS blocks found.")

    tensor_patterns = {
        "A": r"\s*a\(I,S\) \[MHz\]:\s*$",
        "FC": r"\s*Fermi-contact term \(FC\):\s*$",
        "PSO": r"\s*Paramagnetic spin orbit term \(PSO\):\s*$",
        "SD": r"\s*Spin dipolar term \(SD\):\s*$",
        "R1": r"\s*First relativistic term \(R1\):\s*$",
        "R2": r"\s*Second relativistic term \(R2\):\s*$",
    }

    nuclei: Dict[int, NucleusHFCC] = {}
    for b, (start, index, symbol) in enumerate(starts):
        end = starts[b + 1][0] if b + 1 < len(starts) else len(lines)
        block = lines[start:end]
        block_text = "\n".join(block)
        nuc = NucleusHFCC(index=index, symbol=symbol)

        if m := re.search(r"g-factor\s*=\s*([-+0-9.Ee]+)", block_text):
            nuc.gN = float(m.group(1))
        if m := re.search(r"Atomic mass\s*=\s*([-+0-9.Ee]+)", block_text):
            nuc.mass = float(m.group(1))
        if m := re.search(r"Atomic mass number\s*=\s*(\d+)", block_text):
            nuc.mass_number = int(m.group(1))

        for j, line in enumerate(block):
            if "POSITION OF THE NUCLEUS" in line and j + 1 < len(block):
                nums = numbers_from_line(block[j + 1])
                if len(nums) >= 3:
                    nuc.position_A = np.array(nums[-3:])
                break

        for name, pattern in tensor_patterns.items():
            for j, line in enumerate(block):
                if re.match(pattern, line):
                    nuc.tensors_MHz[name] = matrix3(block[j + 1:j + 15])
                    break

        nuclei[index] = nuc

    return nuclei


def load_run_data(cfg: Config) -> Tuple[RunData, Path, Path]:
    """Read both ReSpect output files and parse all data needed later."""
    gt_path = resolve_input_file(cfg.gt_file)
    hfs_path = resolve_input_file(cfg.hfs_file)
    gt_text = read_text(gt_path)
    hfs_text = read_text(hfs_path)

    return (
        RunData(
            atoms=parse_geometry(gt_text),
            hfcc=parse_hfcc_blocks(hfs_text),
            g_parts=parse_g_tensor_parts(gt_text),
            gauge_origin_A=parse_common_gauge_origin(gt_text),
            c_adjust=parse_speed_of_light_info(gt_text)[0],
            c_now=parse_speed_of_light_info(gt_text)[1],
        ),
        gt_path,
        hfs_path,
    )


# =============================================================================
# 6. CORE PHYSICAL FORMULAS
# =============================================================================


def soncini_doublet_si(cfg: Config, g_BS: np.ndarray, A_IS_MHz: np.ndarray, gN: float) -> Tuple[np.ndarray, float, float]:
    """
    Soncini / Moon-Patchkovskii S = 1/2 Curie shielding in SI units.

    A[MHz] is converted to energy by A[J] = h * 1e6 * A[MHz].
    ReSpect convention gives sigma(B,I) proportional to g(B,S) @ A(I,S).T.
    """
    A_J = C.h * 1.0e6 * A_IS_MHz
    prefactor = -(C.muB / (gN * C.muN)) * beta_si(cfg) * 0.25
    sigma = prefactor * (g_BS @ A_J.T)
    return shielding_from_sigma(sigma)


def stano_komorovsky_au_check(cfg: Config, g_BS: np.ndarray, A_IS_MHz: np.ndarray, gN: float) -> Tuple[np.ndarray, float, float]:
    """
    Atomic-unit version of the same S = 1/2 Curie term.

    This should agree with soncini_doublet_si up to numerical roundoff. It is a
    unit-consistency check, not a separate physical model.
    """
    A_Eh = (C.h * 1.0e6 * A_IS_MHz) / C.Eh
    gamma_N_au = gN / (2.0 * C.c_au * C.mp_me)
    prefactor = -beta_au(cfg) / (8.0 * C.c_au * gamma_N_au)
    sigma = prefactor * (g_BS @ A_Eh.T)
    return shielding_from_sigma(sigma)


def old_gA_product_diagnostic(cfg: Config, g_BS: np.ndarray, A_IS_MHz: np.ndarray, gN: float) -> Tuple[np.ndarray, float, float]:
    """Diagnostic only: old product g @ A instead of recommended g @ A.T."""
    A_J = C.h * 1.0e6 * A_IS_MHz
    prefactor = -(C.muB / (gN * C.muN)) * beta_si(cfg) * 0.25
    sigma = prefactor * (g_BS @ A_J)
    return shielding_from_sigma(sigma)


def magnetic_susceptibility_si(cfg: Config, g_BS: np.ndarray) -> np.ndarray:
    """Curie susceptibility, single-molecule SI volume [m^3]."""
    return (
        C.mu0
        * C.muB**2
        * beta_si(cfg)
        * cfg.spin_S
        * (cfg.spin_S + 1.0)
        / 3.0
        * (g_BS @ g_BS.T)
    )


def long_distance_km(chi_si_m3: np.ndarray, R_A: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """
    Kurland--McGarvey point-dipole shielding tensor.

        sigma_pc = - chi @ [3 Rhat Rhat^T - I] / (4*pi*R^3)

    R is converted from Angstrom to meter in the prefactor.
    """
    R_norm_A = float(np.linalg.norm(R_A))
    if R_norm_A < 1e-12:
        return shielding_from_sigma(np.zeros((3, 3)))

    Rhat = R_A / R_norm_A
    D = 3.0 * np.outer(Rhat, Rhat) - np.eye(3)
    R_m = R_norm_A * C.angstrom_m
    sigma = -(chi_si_m3 @ D) / (4.0 * math.pi * R_m**3)
    return shielding_from_sigma(sigma)


def hyperfine_pda_from_g_MHz(g_BS: np.ndarray, R_A: np.ndarray, gN: float) -> np.ndarray:
    """
    Long-distance point-dipole approximation to the hyperfine tensor A.

        A_PDA[J] = mu0/(4*pi*R^3) * (gN*muN) * muB * D(R) @ g

    The returned tensor has ReSpect A(I,S) convention and units MHz. At the
    paramagnetic center, R = 0, the PDA formula is undefined; a zero matrix is
    returned only as a safe placeholder.
    """
    R_norm_A = float(np.linalg.norm(R_A))
    if R_norm_A < 1e-12:
        return np.zeros((3, 3))

    Rhat = R_A / R_norm_A
    D = 3.0 * np.outer(Rhat, Rhat) - np.eye(3)
    R_m = R_norm_A * C.angstrom_m

    pref_J = C.mu0 / (4.0 * math.pi * R_m**3) * (gN * C.muN) * C.muB
    A_J = pref_J * (D @ g_BS)
    return A_J / (C.h * 1.0e6)


# =============================================================================
# 7. TENSOR DECOMPOSITIONS
# =============================================================================


def split_iso_ani(M: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """Return scalar iso value, iso*I matrix, and traceless anisotropic matrix."""
    iso = tensor_iso(M)
    iso_matrix = iso * np.eye(3)
    return iso, iso_matrix, M - iso_matrix


def trace_product_iso(g_part: np.ndarray, A_part_MHz: np.ndarray) -> float:
    """Scalar entering isotropic pNMR: (1/3)Tr(g_part @ A_part.T)."""
    return float(np.trace(g_part @ A_part_MHz.T) / 3.0)


def decompose_trace_product(g_part: np.ndarray, A_part_MHz: np.ndarray) -> Dict[str, float]:
    """Decompose (1/3)Tr(g A.T) into iso--iso and ani--ani pieces."""
    g_iso, g_iso_M, g_ani = split_iso_ani(g_part)
    A_iso, A_iso_M, A_ani = split_iso_ani(A_part_MHz)

    iso_iso = trace_product_iso(g_iso_M, A_iso_M)
    iso_ani = trace_product_iso(g_iso_M, A_ani)
    ani_iso = trace_product_iso(g_ani, A_iso_M)
    ani_ani = trace_product_iso(g_ani, A_ani)

    return {
        "g_iso": g_iso,
        "A_iso_MHz": A_iso,
        "trace_gA_total_MHzlike": trace_product_iso(g_part, A_part_MHz),
        "trace_iso_iso_MHzlike": iso_iso,
        "trace_iso_ani_MHzlike": iso_ani,
        "trace_ani_iso_MHzlike": ani_iso,
        "trace_ani_ani_MHzlike": ani_ani,
        "trace_reconstructed_MHzlike": iso_iso + iso_ani + ani_iso + ani_ani,
    }


def soncini_prefactor_per_MHz(cfg: Config, gN: float) -> float:
    """Convert (1/3)Tr(gA.T) in MHz-like units to sigma_iso."""
    return -(C.muB / (gN * C.muN)) * beta_si(cfg) * 0.25 * C.h * 1.0e6


def delta_from_trace_scalar(cfg: Config, trace_scalar_MHzlike: float, gN: float) -> float:
    sigma_iso = soncini_prefactor_per_MHz(cfg, gN) * trace_scalar_MHzlike
    return -1.0e6 * sigma_iso


def build_g_analysis_parts(g_parts: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Return physical and diagnostic g tensors for decomposition tables."""
    g = g_parts["g_total"]
    _, g_iso_M, g_ani = split_iso_ani(g)

    out = {"g_total": g}
    for key in ["g_spin_zeeman", "g_orbital_zeeman", "g_relativistic"]:
        if key in g_parts:
            out[key] = g_parts[key]

    if all(k in g_parts for k in ["g_spin_zeeman", "g_orbital_zeeman", "g_relativistic"]):
        out["g_sum_printed_parts"] = g_parts["g_spin_zeeman"] + g_parts["g_orbital_zeeman"] + g_parts["g_relativistic"]

    out["g_free_e_isoI"] = C.ge_free * np.eye(3)
    out["g_shift_total_minus_free_e"] = g - C.ge_free * np.eye(3)
    out["g_total_isoI"] = g_iso_M
    out["g_total_ani"] = g_ani
    return out


def build_A_analysis_parts(tensors_MHz: Dict[str, np.ndarray], A_nonFC: np.ndarray, A_PDA: np.ndarray) -> Dict[str, np.ndarray]:
    """Return physical, grouped, and PDA-related A tensors for decomposition."""
    out: Dict[str, np.ndarray] = {}
    for key in ["A", "FC", "PSO", "SD", "R1", "R2"]:
        if key in tensors_MHz:
            out[f"A_{key}"] = tensors_MHz[key]

    if "A" in tensors_MHz and "FC" in tensors_MHz:
        out["A_total_minus_FC"] = tensors_MHz["A"] - tensors_MHz["FC"]
    if all(k in tensors_MHz for k in ["PSO", "SD", "R1", "R2"]):
        out["A_PSO_SD_R1_R2_sum"] = tensors_MHz["PSO"] + tensors_MHz["SD"] + tensors_MHz["R1"] + tensors_MHz["R2"]

    out["A_nonFC"] = A_nonFC
    out["A_PDA_from_g"] = A_PDA
    out["A_nonFC_minus_PDA"] = A_nonFC - A_PDA
    return out


# =============================================================================
# 8. ANALYSIS OF ONE NUCLEUS
# =============================================================================


def long_distance_label(cfg: Config, distance_A: float) -> str:
    if distance_A < 1e-12:
        return "center"
    if distance_A < cfg.ld_near_A:
        return "near"
    if distance_A < cfg.ld_far_A:
        return "mid"
    return "far"


def principal_g_values(g_BS: np.ndarray) -> np.ndarray:
    vals = np.linalg.eigvalsh(g_BS @ g_BS.T)
    return np.sqrt(np.maximum(vals, 0.0))


def analyze_nucleus(
    cfg: Config,
    atom: Atom,
    nucleus: NucleusHFCC,
    center_A: np.ndarray,
    g: np.ndarray,
    chi: np.ndarray,
) -> Tuple[NucleusResult, Dict[str, np.ndarray], Dict[str, float]]:
    """Calculate all main quantities for one nucleus."""
    if nucleus.gN is None:
        raise RuntimeError(f"Missing nuclear g-factor for nucleus #{nucleus.index}.")
    if "A" not in nucleus.tensors_MHz:
        raise RuntimeError(f"Missing total A tensor for nucleus #{nucleus.index}.")

    A = nucleus.tensors_MHz["A"]
    A_FC = nucleus.tensors_MHz.get("FC", np.zeros((3, 3)))
    A_SD = nucleus.tensors_MHz.get("SD", np.zeros((3, 3)))
    A_nonFC = A - A_FC

    R_A = atom.xyz_A - center_A
    distance_A = float(np.linalg.norm(R_A))
    A_PDA = hyperfine_pda_from_g_MHz(g, R_A, nucleus.gN)

    _, _, d_total = soncini_doublet_si(cfg, g, A, nucleus.gN)
    _, _, d_fc = soncini_doublet_si(cfg, g, A_FC, nucleus.gN)
    _, _, d_nonfc = soncini_doublet_si(cfg, g, A_nonFC, nucleus.gN)
    _, _, d_sd = soncini_doublet_si(cfg, g, A_SD, nucleus.gN)
    _, _, d_pda = soncini_doublet_si(cfg, g, A_PDA, nucleus.gN)

    _, _, d_stano_total = stano_komorovsky_au_check(cfg, g, A, nucleus.gN)
    _, _, d_stano_pda = stano_komorovsky_au_check(cfg, g, A_PDA, nucleus.gN)
    sigma_ld, _, d_ld = long_distance_km(chi, R_A)

    result = NucleusResult(
        Atom=nucleus.index,
        Element=atom.symbol,
        IsotopeMassNumber=nucleus.mass_number,
        gN=nucleus.gN,
        Distance_A=distance_A,
        LD_label=long_distance_label(cfg, distance_A),
        Aiso_MHz=tensor_iso(A),
        FCiso_MHz=tensor_iso(A_FC),
        nonFCiso_MHz=tensor_iso(A_nonFC),
        PDAiso_MHz=tensor_iso(A_PDA),
        A_nonFC_minus_PDA_norm_MHz=float(np.linalg.norm(A_nonFC - A_PDA)),
        Soncini_total_ppm=d_total,
        Soncini_FC_ppm=d_fc,
        Soncini_nonFC_ppm=d_nonfc,
        Soncini_SD_only_ppm=d_sd,
        Soncini_PDA_from_g_ppm=d_pda,
        Stano_checked_total_ppm=d_stano_total,
        Stano_checked_PDA_ppm=d_stano_pda,
        Stano_minus_Soncini_ppm=d_stano_total - d_total,
        LongDistance_ppm=d_ld,
        SonciniPDA_minus_LongDistance_ppm=d_pda - d_ld,
        Soncini_minus_LongDistance_ppm=d_total - d_ld,
    )

    tensors = {
        "A": A,
        "A_FC": A_FC,
        "A_nonFC": A_nonFC,
        "A_SD": A_SD,
        "A_PDA": A_PDA,
        "R_A": R_A,
        "sigma_ld": sigma_ld,
    }
    diagnostics = {
        "old_gA_minus_recommended_ppm": old_gA_product_diagnostic(cfg, g, A, nucleus.gN)[2] - d_total,
        "A_antisym_norm_MHz": antisymmetric_norm(A),
    }
    return result, tensors, diagnostics


def make_decomposition_rows(
    cfg: Config,
    result: NucleusResult,
    nucleus: NucleusHFCC,
    g_analysis_parts: Dict[str, np.ndarray],
    A_analysis_parts: Dict[str, np.ndarray],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for g_name, g_part in g_analysis_parts.items():
        for A_name, A_part in A_analysis_parts.items():
            d = decompose_trace_product(g_part, A_part)
            rows.append(
                {
                    "Atom": result.Atom,
                    "Element": result.Element,
                    "IsotopeMassNumber": result.IsotopeMassNumber,
                    "gN": result.gN,
                    "Distance_A": result.Distance_A,
                    "g_part": g_name,
                    "A_part": A_name,
                    **d,
                    "delta_total_ppm": delta_from_trace_scalar(cfg, d["trace_gA_total_MHzlike"], result.gN),
                    "delta_iso_iso_ppm": delta_from_trace_scalar(cfg, d["trace_iso_iso_MHzlike"], result.gN),
                    "delta_ani_ani_ppm": delta_from_trace_scalar(cfg, d["trace_ani_ani_MHzlike"], result.gN),
                }
            )
    return rows



# =============================================================================
# 9. PCS FIELD / CUBE VISUALIZATION
# =============================================================================

_ATOMIC_NUMBERS = {
    "H": 1, "He": 2,
    "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Ne": 10,
    "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17, "Ar": 18,
    "K": 19, "Ca": 20, "Sc": 21, "Ti": 22, "V": 23, "Cr": 24, "Mn": 25, "Fe": 26,
    "Co": 27, "Ni": 28, "Cu": 29, "Zn": 30,
}


def atomic_number(symbol: str) -> int:
    """Return nuclear charge for Gaussian cube atom records."""
    if symbol not in _ATOMIC_NUMBERS:
        raise KeyError(f"Atomic number for element symbol {symbol!r} is not in the small lookup table.")
    return _ATOMIC_NUMBERS[symbol]


def pcs_delta_ppm_at_point(
    chi_si_m3: np.ndarray,
    R_A: np.ndarray,
    cutoff_A: float,
    clip_abs_ppm: Optional[float] = None,
) -> float:
    """
    Long-distance pseudocontact shift field at one point in space.

    The scalar field is

        delta_pc(r) = -1e6 * Tr[sigma_pc(r)]/3

    with

        sigma_pc(r) = - chi @ [3 Rhat Rhat^T - I] / (4*pi*R^3).

    R is measured from the paramagnetic center to the grid point. The formula is
    singular at R = 0, so points closer than cutoff_A are set to zero. Optional
    clipping is only for visualization stability; it is not used in the tabular
    numerical results.
    """
    R_norm_A = float(np.linalg.norm(R_A))
    if R_norm_A < cutoff_A:
        return 0.0

    Rhat = R_A / R_norm_A
    D = 3.0 * np.outer(Rhat, Rhat) - np.eye(3)
    R_m = R_norm_A * C.angstrom_m

    # delta = -sigma_iso*1e6, sigma_iso = -Tr(chi @ D)/(12*pi*R^3)
    delta_ppm = float(np.trace(chi_si_m3 @ D) / (12.0 * math.pi * R_m**3) * 1.0e6)

    if clip_abs_ppm is not None and clip_abs_ppm > 0.0:
        delta_ppm = max(-clip_abs_ppm, min(clip_abs_ppm, delta_ppm))
    return delta_ppm


def write_pcs_cube(
    path: Path,
    cfg: Config,
    atoms: List[Atom],
    center_A: np.ndarray,
    chi_si_m3: np.ndarray,
) -> Dict[str, object]:
    """
    Write a Gaussian cube file with the long-distance PCS field delta_pc(r) [ppm].

    This is a visualization of the Kurland--McGarvey / PDA scalar field only.
    It is not a full Soncini/contact-shift field because a full Soncini field
    would require A(r), i.e. the hyperfine tensor or spin density on every grid
    point rather than only at the nuclei.
    """
    xyz = np.array([atom.xyz_A for atom in atoms])
    origin_A = xyz.min(axis=0) - cfg.pcs_cube_padding_A
    max_A = xyz.max(axis=0) + cfg.pcs_cube_padding_A
    lengths_A = max_A - origin_A
    dims = np.ceil(lengths_A / cfg.pcs_cube_spacing_A).astype(int) + 1
    nx, ny, nz = [int(x) for x in dims]

    values = np.empty((nx, ny, nz), dtype=float)
    for ix in range(nx):
        x = origin_A[0] + ix * cfg.pcs_cube_spacing_A
        for iy in range(ny):
            y = origin_A[1] + iy * cfg.pcs_cube_spacing_A
            for iz in range(nz):
                z = origin_A[2] + iz * cfg.pcs_cube_spacing_A
                R_A = np.array([x, y, z]) - center_A
                values[ix, iy, iz] = pcs_delta_ppm_at_point(
                    chi_si_m3,
                    R_A,
                    cutoff_A=cfg.pcs_cube_cutoff_A,
                    clip_abs_ppm=cfg.pcs_cube_clip_abs_ppm,
                )

    origin_bohr = origin_A * C.bohr_per_angstrom
    step_bohr = cfg.pcs_cube_spacing_A * C.bohr_per_angstrom

    with path.open("w", encoding="utf-8") as f:
        f.write("Long-distance pseudocontact shift field delta_pc(r) [ppm]\n")
        f.write(
            f"spacing_A={cfg.pcs_cube_spacing_A}; padding_A={cfg.pcs_cube_padding_A}; "
            f"cutoff_A={cfg.pcs_cube_cutoff_A}; clip_abs_ppm={cfg.pcs_cube_clip_abs_ppm}\n"
        )
        f.write(f"{len(atoms):5d}{origin_bohr[0]:13.6f}{origin_bohr[1]:13.6f}{origin_bohr[2]:13.6f}\n")
        f.write(f"{nx:5d}{step_bohr:13.6f}{0.0:13.6f}{0.0:13.6f}\n")
        f.write(f"{ny:5d}{0.0:13.6f}{step_bohr:13.6f}{0.0:13.6f}\n")
        f.write(f"{nz:5d}{0.0:13.6f}{0.0:13.6f}{step_bohr:13.6f}\n")

        for atom in atoms:
            Z = atomic_number(atom.symbol)
            xyz_bohr = atom.xyz_A * C.bohr_per_angstrom
            f.write(f"{Z:5d}{float(Z):13.6f}{xyz_bohr[0]:13.6f}{xyz_bohr[1]:13.6f}{xyz_bohr[2]:13.6f}\n")

        n_on_line = 0
        for ix in range(nx):
            for iy in range(ny):
                for iz in range(nz):
                    f.write(f"{values[ix, iy, iz]:13.5e}")
                    n_on_line += 1
                    if n_on_line == 6:
                        f.write("\n")
                        n_on_line = 0
        if n_on_line:
            f.write("\n")

    return {
        "path": str(path),
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "n_points": int(nx * ny * nz),
        "origin_A": origin_A,
        "spacing_A": cfg.pcs_cube_spacing_A,
        "padding_A": cfg.pcs_cube_padding_A,
        "cutoff_A": cfg.pcs_cube_cutoff_A,
        "clip_abs_ppm": cfg.pcs_cube_clip_abs_ppm,
        "min_ppm": float(values.min()),
        "max_ppm": float(values.max()),
        "mean_ppm": float(values.mean()),
        "absmax_ppm": float(np.max(np.abs(values))),
    }


# =============================================================================
# 10. OUTPUT HELPERS
# =============================================================================


def output_path(cfg: Config, filename: str) -> Path:
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / filename


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_compact_table(results: List[NucleusResult]) -> None:
    """
    Print a compact but self-explanatory console table.

    The detailed theory, tensor matrices, and long explanations are written to
    the Markdown report. The console output is intended as a quick run check:
    it shows whether the important equalities are numerically satisfied.
    """
    print("\n" + "=" * 126)
    print("MAIN pNMR RESULTS AND INTERNAL CHECKS")
    print("=" * 126)
    print("All shifts below are chemical shifts delta [ppm], not shielding sigma.")
    print("Definitions used in the columns:")
    print("  SoncTot  = Soncini doublet shift from full ReSpect A tensor: delta[g, A_total]")
    print("  FC       = Fermi-contact part: delta[g, A_FC]")
    print("  nonFC    = everything except FC: delta[g, A_total - A_FC]")
    print("  PDA      = Soncini with point-dipole A_PDA(g,R): delta[g, A_PDA]")
    print("  LongDist = Kurland--McGarvey long-distance PCS from chi and geometry")
    print("  PDA-LD   = PDA - LongDist; should be near zero if units/signs/transposes are correct")
    print("-" * 126)
    print(
        f"{'At':>3} {'El':>2} {'Iso':>4} {'R/A':>7} {'Aiso':>9} {'FCiso':>9} "
        f"{'SoncTot':>11} {'FC':>11} {'nonFC':>11} {'PDA':>11} {'LongDist':>11} {'PDA-LD':>10} {'LD':>6}"
    )
    print("-" * 126)
    for r in results:
        print(
            f"{r.Atom:3d} {r.Element:>2} {str(r.IsotopeMassNumber or ''):>4} "
            f"{r.Distance_A:7.3f} {r.Aiso_MHz:9.3f} {r.FCiso_MHz:9.3f} "
            f"{r.Soncini_total_ppm:11.4e} {r.Soncini_FC_ppm:11.4e} "
            f"{r.Soncini_nonFC_ppm:11.4e} {r.Soncini_PDA_from_g_ppm:11.4e} "
            f"{r.LongDistance_ppm:11.4e} {r.SonciniPDA_minus_LongDistance_ppm:10.2e} {r.LD_label:>6}"
        )
    print("-" * 126)

    if results:
        max_unit = max(abs(r.Stano_minus_Soncini_ppm) for r in results)
        max_pda = max(abs(r.SonciniPDA_minus_LongDistance_ppm) for r in results)
        print("Run-level sanity checks:")
        print(f"  max |Stano - Soncini| over nuclei = {max_unit:.3e} ppm")
        print(f"  max |Soncini_PDA - LongDist| over nuclei = {max_pda:.3e} ppm")
        print("  If both are small, the main unit/sign/transposition checks passed.")


def make_report(
    cfg: Config,
    gt_path: Path,
    hfs_path: Path,
    data: RunData,
    g: np.ndarray,
    chi: np.ndarray,
    center_atom: Atom,
    results: List[NucleusResult],
    detail_blocks: List[str],
    cube_info: Optional[Dict[str, object]] = None,
) -> str:
    """
    Create a single self-contained Markdown report.

    The goal of this report is not just to list numbers, but to explain exactly
    which theoretical expression each number comes from and which comparisons
    are intended as physical results versus diagnostic sanity checks.
    """
    g_sum_message = "not available"
    g_sum_block: List[str] = []
    if all(k in data.g_parts for k in ["g_spin_zeeman", "g_orbital_zeeman", "g_relativistic"]):
        g_sum = data.g_parts["g_spin_zeeman"] + data.g_parts["g_orbital_zeeman"] + data.g_parts["g_relativistic"]
        g_diff = g_sum - g
        g_sum_message = f"max abs(sum printed parts - g_total) = {np.max(np.abs(g_diff)):.3e}"
        g_sum_block = [
            "### Check of ReSpect g decomposition",
            "",
            "ReSpect printed the total tensor and the parts",
            "",
            r"\[",
            r"g = g^{\mathrm{SZ}} + g^{\mathrm{OZ}} + g^{\mathrm{R}}.",
            r"\]",
            "",
            f"Numerical check: `{g_sum_message}`.",
            "This should be close to zero. It checks that the parser read the correct three matrices.",
            "",
        ]

    warning_lines: List[str] = []
    if data.c_adjust is not None and abs(data.c_adjust - 1.0) > 1e-12:
        warning_lines = [
            "> **Important limitation:** ReSpect used an adjusted speed of light.",
            f"> `c_adjust = {data.c_adjust}`, `c_now = {data.c_now}`.",
            "> The Python post-processing is internally consistent for the tensors in the files,",
            "> but the numerical values are `delta[g(c_adjust), A(c_adjust)]`, not corrected physical-c values.",
            "> A physical comparison requires a new ReSpect calculation with physical `c`.",
            "",
        ]

    max_stano = max(abs(r.Stano_minus_Soncini_ppm) for r in results) if results else float('nan')
    max_pda_ld = max(abs(r.SonciniPDA_minus_LongDistance_ppm) for r in results) if results else float('nan')

    lines: List[str] = [
        "# pNMR shielding calculation report",
        "",
        "This report is intentionally verbose. It explains what every main quantity means,",
        "which tensor was read from which ReSpect block, which formula was used, and which",
        "comparisons are physical interpretations versus internal checks.",
        "",
        "---",
        "",
        "## 1. Input files and global settings",
        "",
        f"- GT file: `{gt_path}`",
        f"- HFS/HFCC file: `{hfs_path}`",
        f"- Temperature: `{cfg.temperature_K}` K",
        f"- Effective spin: `S = {cfg.spin_S}`",
        f"- Paramagnetic center: atom #{center_atom.index} {center_atom.symbol}",
        f"- Paramagnetic center coordinates: `{center_atom.xyz_A}` A",
        f"- ReSpect gauge origin: `{data.gauge_origin_A}` A",
        f"- ReSpect speed-of-light setting: `adjusted = {data.c_adjust}`, `c_now = {data.c_now}`",
        "",
        *warning_lines,
        "---",
        "",
        "## 2. Tensor convention used everywhere",
        "",
        "ReSpect prints the EPR tensors with the following index order:",
        "",
        r"\[",
        r"g(B,S), \qquad A(I,S).",
        r"\]",
        "",
        "Here `B` is the external magnetic-field component, `I` is the nuclear magnetic moment component,",
        "and `S` is the effective electron-spin component. The common contracted index is `S`, therefore",
        "the shielding tensor has to be formed as",
        "",
        r"\[",
        r"\sigma(B,I) \propto \sum_S g(B,S)A(I,S) = gA^T.",
        r"\]",
        "",
        "In the code this is `g @ A.T`. The old diagnostic `g @ A` is printed only to show how much the",
        "wrong product would change the result.",
        "",
        "---",
        "",
        "## 3. Theory implemented in the script",
        "",
        "### 3.1 Soncini / Moon--Patchkovskii doublet Curie shielding",
        "",
        "For the present `S = 1/2` doublet, the Curie part of the pNMR shielding is evaluated as",
        "",
        r"\[",
        r"\sigma^{p} = -\frac{\mu_B\beta}{4g_N\mu_N}\, gA^T, \qquad \beta=\frac{1}{k_B T}.",
        r"\]",
        "",
        "The hyperfine tensor is read from ReSpect in MHz and converted to energy as",
        "",
        r"\[",
        r"A[\mathrm{J}] = h\,10^6\,A[\mathrm{MHz}].",
        r"\]",
        "",
        "The reported chemical shift is",
        "",
        r"\[",
        r"\delta_{\mathrm{ppm}} = -\frac{1}{3}\mathrm{Tr}(\sigma)\,10^6.",
        r"\]",
        "",
        "### 3.2 Komorovsky/Stano atomic-unit check",
        "",
        "The same doublet expression is evaluated again in Hartree atomic units:",
        "",
        r"\[",
        r"\sigma^{p}_{\mathrm{a.u.}} = -\frac{\beta_{\mathrm{au}}}{8c\gamma_N^{\mathrm{au}}}\,gA^T.",
        r"\]",
        "",
        "This is not a second physical model. It is a unit check. The important comparison is",
        "",
        r"\[",
        r"\Delta_{\mathrm{unit}} = \delta_{\mathrm{Stano/Komorovsky}}-\delta_{\mathrm{Soncini}}.",
        r"\]",
        "",
        f"For this run, `max |Stano - Soncini| = {max_stano:.3e} ppm`.",
        "",
        "### 3.3 Magnetic susceptibility and long-distance/Kurland--McGarvey limit",
        "",
        "The single-molecule Curie susceptibility used for the long-distance PCS is",
        "",
        r"\[",
        r"\chi = \mu_0\mu_B^2\beta\frac{S(S+1)}{3}\,gg^T.",
        r"\]",
        "",
        "For `S = 1/2`, the spin factor is `S(S+1)/3 = 1/4`. The long-distance pseudocontact shielding at a nucleus or grid point is",
        "",
        r"\[",
        r"\sigma_{\mathrm{pc}}(\mathbf R) = -\frac{1}{4\pi R^3}\,\chi\left(3\hat{\mathbf R}\hat{\mathbf R}^T-I\right).",
        r"\]",
        "",
        "This is only the point-dipole pseudocontact limit. It is not the full Soncini/contact shift.",
        "",
        "### 3.4 PDA bridge: why `Soncini_PDA = LongDistance` should hold",
        "",
        "The point-dipole approximation to the hyperfine tensor is constructed as",
        "",
        r"\[",
        r"A_{\mathrm{PDA}}[\mathrm{J}] = \frac{\mu_0}{4\pi R^3}\,(g_N\mu_N)\mu_B\left(3\hat{\mathbf R}\hat{\mathbf R}^T-I\right)g.",
        r"\]",
        "",
        "After conversion to MHz, this artificial `A_PDA` is inserted into the Soncini formula. Algebraically,",
        "the factors `g_N mu_N` cancel, and the result becomes the Kurland--McGarvey expression above. Therefore",
        "",
        r"\[",
        r"\delta[g,A_{\mathrm{PDA}}] \equiv \delta_{\mathrm{LongDistance}}.",
        r"\]",
        "",
        f"For this run, `max |Soncini_PDA - LongDistance| = {max_pda_ld:.3e} ppm`.",
        "",
        "If this value is close to zero, the PDA bridge, units, signs, and transposes are consistent. It does **not**",
        "mean that the real ReSpect `A_nonFC` tensor is equal to `A_PDA`.",
        "",
        "### 3.5 Tensor decomposition used for interpretation",
        "",
        "The script also evaluates the contributions from the ReSpect decompositions",
        "",
        r"\[",
        r"g = g^{\mathrm{SZ}} + g^{\mathrm{OZ}} + g^{\mathrm{R}},",
        r"\]",
        "",
        r"\[",
        r"A = A^{\mathrm{FC}} + A^{\mathrm{PSO}} + A^{\mathrm{SD}} + A^{\mathrm{R1}} + A^{\mathrm{R2}}.",
        r"\]",
        "",
        "For the isotropic shift, the scalar contraction is",
        "",
        r"\[",
        r"\frac{1}{3}\mathrm{Tr}(gA^T).",
        r"\]",
        "",
        "Each tensor is split into isotropic and anisotropic pieces:",
        "",
        r"\[",
        r"T=T_{\mathrm{iso}}I+T_{\mathrm{ani}},\qquad T_{\mathrm{iso}}=\frac{1}{3}\mathrm{Tr}(T).",
        r"\]",
        "",
        "Then",
        "",
        r"\[",
        r"\frac{1}{3}\mathrm{Tr}(gA^T)=g_{\mathrm{iso}}A_{\mathrm{iso}}+\frac{1}{3}\mathrm{Tr}(g_{\mathrm{ani}}A_{\mathrm{ani}}^T).",
        r"\]",
        "",
        "The iso--ani and ani--iso cross terms should be zero apart from numerical noise.",
        "",
        "---",
        "",
        "## 4. Tensors read from the ReSpect output",
        "",
        "### 4.1 Total g tensor `g(B,S)`",
        "```text",
        fmt_matrix(g),
        "```",
        f"- `g_iso = {tensor_iso(g):.10f}`",
        f"- principal values from `G = g g^T`: `{principal_g_values(g)}`",
        f"- antisymmetric norm of `g`: `{antisymmetric_norm(g):.3e}`",
        "",
        *g_sum_block,
        "### 4.2 Printed g parts",
        "",
    ]

    for key, label in [
        ("g_spin_zeeman", "Spin-Zeeman part"),
        ("g_orbital_zeeman", "Orbital-Zeeman part"),
        ("g_relativistic", "Relativistic part"),
        ("g_total", "Total g(B,S)"),
    ]:
        if key in data.g_parts:
            lines.extend([
                f"#### {label} (`{key}`)",
                "```text",
                fmt_matrix(data.g_parts[key]),
                "```",
                f"- isotropic value = `{tensor_iso(data.g_parts[key]):.10f}`",
                "",
            ])

    lines.extend([
        "### 4.3 Magnetic susceptibility `chi` [m^3]",
        "```text",
        fmt_matrix(chi, precision=10),
        "```",
        f"- `chi_iso = {tensor_iso(chi):.10e} m^3`",
        "",
        "### 4.4 Hyperfine A tensor summary per nucleus",
        "",
        "The total `A` tensor is read only from the exact ReSpect line `a(I,S) [MHz]:`, so the Gordon",
        "decomposition block does not overwrite the real total tensor.",
        "",
        "| Atom | El | isotope | R/A | gN | Aiso | FCiso | nonFCiso | PDAiso | ||nonFC-PDA|| |",
        "|---:|:--:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])

    for r in results:
        lines.append(
            f"| {r.Atom} | {r.Element} | {r.IsotopeMassNumber or ''} | {r.Distance_A:.3f} | {r.gN:.6f} | "
            f"{r.Aiso_MHz:.6f} | {r.FCiso_MHz:.6f} | {r.nonFCiso_MHz:.6f} | "
            f"{r.PDAiso_MHz:.6e} | {r.A_nonFC_minus_PDA_norm_MHz:.6e} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 5. Main numerical results and comparisons",
        "",
        "Read this table as follows:",
        "",
        "- `Soncini total` is the actual pNMR Curie shift computed from the full ReSpect `A` tensor.",
        "- `FC` is the Fermi-contact contribution; in these data it dominates many isotropic shifts.",
        "- `nonFC` is `A_total - A_FC`; it is useful, but it is not automatically the long-distance PCS.",
        "- `PDA` is the Soncini formula with the artificial point-dipole `A_PDA(g,R)` tensor.",
        "- `LongDist` is the same point-dipole limit computed directly from susceptibility and geometry.",
        "- `PDA-LD` should be close to zero; this is an internal algebra/unit check.",
        "",
        "| Atom | El | R/A | Soncini total | FC | nonFC | SD-only | PDA | Stano total | Stano-Soncini | LongDist | PDA-LD | Soncini-LongDist | LD label |",
        "|---:|:--:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:--:|",
    ])

    for r in results:
        lines.append(
            f"| {r.Atom} | {r.Element} | {r.Distance_A:.3f} | "
            f"{r.Soncini_total_ppm:.6e} | {r.Soncini_FC_ppm:.6e} | {r.Soncini_nonFC_ppm:.6e} | "
            f"{r.Soncini_SD_only_ppm:.6e} | {r.Soncini_PDA_from_g_ppm:.6e} | "
            f"{r.Stano_checked_total_ppm:.6e} | {r.Stano_minus_Soncini_ppm:.2e} | "
            f"{r.LongDistance_ppm:.6e} | {r.SonciniPDA_minus_LongDistance_ppm:.2e} | "
            f"{r.Soncini_minus_LongDistance_ppm:.6e} | {r.LD_label} |"
        )

    lines.extend([
        "",
        "### 5.1 What the comparisons mean",
        "",
        "1. `Stano-Soncini` close to zero means that the SI and atomic-unit implementations of the same doublet formula agree.",
        "2. `PDA-LD` close to zero means that `A_PDA` inserted into Soncini is equivalent to the KM long-distance formula.",
        "3. `Soncini total - LongDist` is not expected to be small. The full ReSpect `A` tensor contains contact and local hyperfine effects.",
        "4. `nonFC` is not equal to `LongDist` unless the real non-contact `A` tensor has reached the point-dipole limit.",
        "",
        "---",
        "",
        "## 6. PCS scalar-field cube visualization",
        "",
    ])

    if cube_info is not None:
        lines.extend([
            f"- Cube file: `{cube_info['path']}`",
            "- Scalar field in the cube: `delta_pc(r)` in ppm.",
            "- This field is calculated from the KM/PDA long-distance formula on a 3D grid.",
            f"- Grid dimensions: `{cube_info['nx']} x {cube_info['ny']} x {cube_info['nz']}` = `{cube_info['n_points']}` points.",
            f"- Grid spacing: `{cube_info['spacing_A']}` A.",
            f"- Padding around molecule: `{cube_info['padding_A']}` A.",
            f"- Cutoff around center: `{cube_info['cutoff_A']}` A, because the formula has a `1/R^3` singularity.",
            f"- Clipping used in this cube: `{cube_info['clip_abs_ppm']}` ppm.",
            f"- Value range: min `{cube_info['min_ppm']:.6e}` ppm, max `{cube_info['max_ppm']:.6e}` ppm, absmax `{cube_info['absmax_ppm']:.6e}` ppm.",
            "",
            "Recommended first isosurface values in Avogadro/VMD/ChimeraX: `0.005`, `0.010`, `0.020`, `0.030` ppm.",
            "The red and blue surfaces show opposite signs of the long-distance pseudocontact field. They are not spin density and not contact shift.",
        ])
    else:
        lines.append("Cube output was disabled in the configuration.")

    lines.extend([
        "",
        "---",
        "",
        "## 7. Detailed diagnostics for selected nuclei",
        "",
        "The following blocks show the raw tensors and decompositions for the first nucleus and the first far nucleus.",
        "They are meant to make clear exactly what was read from ReSpect and how each comparison was formed.",
        "",
    ])
    for block in detail_blocks:
        lines.extend(["```text", block, "```", ""])

    lines.extend([
        "---",
        "",
        "## 8. Short interpretation of this run",
        "",
        "- The dominant part of the full Soncini shift is usually the Fermi-contact contribution `FC`.",
        "- The spin-Zeeman part of `g` gives almost all of the total `g x A` shift; orbital-Zeeman and relativistic parts are corrections.",
        "- The long-distance PCS field is small because the `g` tensor, and therefore `chi`, is nearly isotropic.",
        "- `Soncini[g,A_PDA] = LongDistance` is expected by construction and validates the implementation.",
        "- `Soncini_nonFC != LongDistance` is a physical result: the real non-contact hyperfine tensor has not simply reduced to the point-dipole tensor.",
    ])

    return "\n".join(lines)


def detail_block_for_result(
    cfg: Config,
    result: NucleusResult,
    nucleus: NucleusHFCC,
    g: np.ndarray,
    g_parts: Dict[str, np.ndarray],
    tensors: Dict[str, np.ndarray],
    diagnostics: Dict[str, float],
) -> str:
    """
    Make a detailed human-readable block for one representative nucleus.

    This block deliberately repeats the important definitions so that the report
    can be read without constantly looking at the source code.
    """
    title = "First nucleus detailed diagnostics" if result.Atom == 1 else "First far nucleus detailed diagnostics"
    A = tensors["A"]
    A_PDA = tensors["A_PDA"]
    A_nonFC = tensors["A_nonFC"]

    lines = [
        title,
        "-" * 88,
        f"Atom #{result.Atom} {result.Element}, distance from paramagnetic center = {result.Distance_A:.6f} A",
        f"nuclear g-factor gN = {result.gN:.10f}",
        "",
        "THEORY USED FOR THIS BLOCK",
        "  sigma_Soncini = -(muB*beta)/(4*gN*muN) * g @ A.T",
        "  delta_ppm     = -Tr(sigma)/3 * 1e6",
        "  LongDist      = KM/PDA point-dipole PCS from chi and R",
        "  A_nonFC       = A_total - A_FC",
        "  A_PDA         = point-dipole approximation to A built from g and R",
        "",
        "TOTAL A TENSOR read from exact line 'a(I,S) [MHz]:'",
        fmt_matrix(A),
        f"Aiso = {result.Aiso_MHz:.10f} MHz",
        f"A antisymmetric norm = {diagnostics['A_antisym_norm_MHz']:.6e} MHz",
        "",
        "A TENSOR PARTS read from ReSpect decomposition [MHz]",
    ]

    for key in ["FC", "PSO", "SD", "R1", "R2"]:
        if key in nucleus.tensors_MHz:
            lines.extend([
                f"A_{key}, iso = {tensor_iso(nucleus.tensors_MHz[key]):.10f} MHz",
                fmt_matrix(nucleus.tensors_MHz[key]),
                "",
            ])

    lines.extend([
        "A_nonFC = A_total - A_FC [MHz]",
        fmt_matrix(A_nonFC),
        f"A_nonFC_iso = {tensor_iso(A_nonFC):.10f} MHz",
        "",
        "A_PDA_from_g [MHz]; this is the long-distance point-dipole approximation, not the real QC A tensor",
        fmt_matrix(A_PDA),
        f"A_PDA_iso = {tensor_iso(A_PDA):.10f} MHz",
        f"||A_nonFC - A_PDA|| = {result.A_nonFC_minus_PDA_norm_MHz:.6e} MHz",
        "",
        "TOTAL g(B,S) TENSOR",
        fmt_matrix(g),
        f"g_iso = {tensor_iso(g):.10f}",
        "",
    ])

    if all(k in g_parts for k in ["g_spin_zeeman", "g_orbital_zeeman", "g_relativistic"]):
        lines.extend(["g decomposition printed by ReSpect"])
        for name in ["g_spin_zeeman", "g_orbital_zeeman", "g_relativistic"]:
            lines.extend([
                f"{name}, iso = {tensor_iso(g_parts[name]):.10f}",
                fmt_matrix(g_parts[name]),
                "",
            ])
        g_sum = g_parts["g_spin_zeeman"] + g_parts["g_orbital_zeeman"] + g_parts["g_relativistic"]
        lines.append(f"max abs(g_spin_zeeman + g_orbital_zeeman + g_relativistic - g_total) = {np.max(np.abs(g_sum-g)):.6e}")
        lines.append("")

    lines.extend([
        "MATRIX PRODUCTS used for checking the ReSpect index convention",
        "Recommended product: g @ A.T [MHz-like]",
        fmt_matrix(g @ A.T),
        "Old diagnostic product: g @ A [MHz-like]",
        fmt_matrix(g @ A),
        "",
        "SHIFT RESULTS AND COMPARISONS [ppm]",
        f"Soncini/ReSpect total delta      = {result.Soncini_total_ppm:.10e}",
        f"Stano/Komorovsky checked delta   = {result.Stano_checked_total_ppm:.10e}",
        f"Stano - Soncini                  = {result.Stano_minus_Soncini_ppm:.10e}",
        f"Old g@A - recommended            = {diagnostics['old_gA_minus_recommended_ppm']:.10e}",
        f"Soncini FC delta                 = {result.Soncini_FC_ppm:.10e}",
        f"Soncini nonFC delta              = {result.Soncini_nonFC_ppm:.10e}",
        f"Soncini SD-only delta            = {result.Soncini_SD_only_ppm:.10e}",
        f"Soncini PDA-from-g delta         = {result.Soncini_PDA_from_g_ppm:.10e}",
        f"Komorovsky/Stano PDA delta       = {result.Stano_checked_PDA_ppm:.10e}",
        f"Long-distance delta              = {result.LongDistance_ppm:.10e}",
        f"PDA - LongDist                   = {result.SonciniPDA_minus_LongDistance_ppm:.10e}",
        "",
        "Interpretation of the comparisons:",
        "  Stano - Soncini ~ 0  -> SI and atomic-unit formulas agree.",
        "  PDA - LongDist ~ 0   -> A_PDA bridge and KM long-distance formula agree.",
        "  nonFC != LongDist    -> real non-contact A is not simply the point-dipole limit.",
    ])

    if all(k in g_parts for k in ["g_spin_zeeman", "g_orbital_zeeman", "g_relativistic"]):
        lines.extend(["", "PHYSICAL g-PART CONTRIBUTIONS WITH TOTAL A"])
        lines.append(f"{'g_part':>26} {'giso':>12} {'1/3Tr(gA.T)':>15} {'iso_iso':>15} {'ani_ani':>15} {'delta_ppm':>15}")
        for g_name in ["g_spin_zeeman", "g_orbital_zeeman", "g_relativistic", "g_total"]:
            d = decompose_trace_product(g_parts[g_name], A)
            lines.append(
                f"{g_name:>26} {d['g_iso']:12.6f} {d['trace_gA_total_MHzlike']:15.6f} "
                f"{d['trace_iso_iso_MHzlike']:15.6f} {d['trace_ani_ani_MHzlike']:15.6f} "
                f"{delta_from_trace_scalar(cfg, d['trace_gA_total_MHzlike'], result.gN):15.6e}"
            )

    A_parts = build_A_analysis_parts(nucleus.tensors_MHz, A_nonFC, A_PDA)
    lines.extend(["", "ISO/ANISO DECOMPOSITION OF g_total x A_parts"])
    lines.append("Meaning: (1/3)Tr(g A.T) = iso-iso + ani-ani; cross terms should be approximately zero")
    lines.append(f"{'A_part':>18} {'Aiso':>12} {'1/3Tr(gA.T)':>15} {'iso_iso':>15} {'ani_ani':>15} {'delta_ppm':>15}")
    for name, Apart in A_parts.items():
        d = decompose_trace_product(g, Apart)
        lines.append(
            f"{name:>18} {d['A_iso_MHz']:12.6f} {d['trace_gA_total_MHzlike']:15.6f} "
            f"{d['trace_iso_iso_MHzlike']:15.6f} {d['trace_ani_ani_MHzlike']:15.6f} "
            f"{delta_from_trace_scalar(cfg, d['trace_gA_total_MHzlike'], result.gN):15.6e}"
        )

    return "\n".join(lines)


# =============================================================================
# 11. MAIN PROGRAM
# =============================================================================


def main() -> None:
    cfg = Config(output_dir=".")

    print("Reading and parsing ReSpect outputs...")
    print("This run will write an explanatory Markdown report with formulas, tensor sources, and all comparisons.")
    data, gt_path, hfs_path = load_run_data(cfg)

    if not (0 <= cfg.paramagnetic_center_index < len(data.atoms)):
        raise IndexError("PARAMAGNETIC_CENTER_INDEX is outside the atom list.")

    center_atom = data.atoms[cfg.paramagnetic_center_index]
    center_A = center_atom.xyz_A
    g = data.g_parts["g_total"]
    chi = magnetic_susceptibility_si(cfg, g)
    g_analysis_parts = build_g_analysis_parts(data.g_parts)

    print(f"Parsed atoms       : {len(data.atoms)}")
    print(f"Parsed HFCC blocks : {len(data.hfcc)}")
    print(f"Paramagnetic center: atom #{center_atom.index} {center_atom.symbol}, xyz = {center_A} A")
    print(f"g_iso              : {tensor_iso(g):.10f}")
    print(f"principal g        : {principal_g_values(g)}")
    print(f"chi_iso            : {tensor_iso(chi):.10e} m^3")

    if data.c_adjust is not None and abs(data.c_adjust - 1.0) > 1e-12:
        print(f"WARNING: ReSpect used adjusted speed of light: factor={data.c_adjust}, c_now={data.c_now}")

    results: List[NucleusResult] = []
    main_rows: List[Dict[str, object]] = []
    decomp_rows: List[Dict[str, object]] = []
    detail_blocks: List[str] = []
    first_far_detail_done = False

    for nucleus_index in sorted(data.hfcc):
        nucleus = data.hfcc[nucleus_index]
        atom_index0 = nucleus_index - 1
        if atom_index0 < 0 or atom_index0 >= len(data.atoms):
            print(f"Skipping nucleus #{nucleus_index}: no matching atom in geometry.")
            continue
        if "A" not in nucleus.tensors_MHz:
            print(f"Skipping nucleus #{nucleus_index}: no total A tensor.")
            continue

        atom = data.atoms[atom_index0]
        result, tensors, diagnostics = analyze_nucleus(cfg, atom, nucleus, center_A, g, chi)
        results.append(result)
        main_rows.append(asdict(result))

        A_parts = build_A_analysis_parts(nucleus.tensors_MHz, tensors["A_nonFC"], tensors["A_PDA"])
        decomp_rows.extend(make_decomposition_rows(cfg, result, nucleus, g_analysis_parts, A_parts))

        wants_detail = result.Atom == 1 or (result.LD_label == "far" and not first_far_detail_done)
        if wants_detail:
            detail_blocks.append(detail_block_for_result(cfg, result, nucleus, g, data.g_parts, tensors, diagnostics))
            if result.LD_label == "far":
                first_far_detail_done = True

    cube_info: Optional[Dict[str, object]] = None
    if cfg.write_pcs_cube:
        cube_path = output_path(cfg, cfg.pcs_cube_file)
        cube_info = write_pcs_cube(cube_path, cfg, data.atoms, center_A, chi)

    if cfg.print_console_table:
        print_compact_table(results)

    main_csv_path = output_path(cfg, cfg.main_csv)
    decomp_csv_path = output_path(cfg, cfg.decomp_csv)
    report_path = output_path(cfg, cfg.report_md)

    write_csv(main_csv_path, main_rows)
    write_csv(decomp_csv_path, decomp_rows)
    report_path.write_text(
        make_report(cfg, gt_path, hfs_path, data, g, chi, center_atom, results, detail_blocks, cube_info),
        encoding="utf-8",
    )

    print("\nFiles written:")
    print(f"  main CSV           : {main_csv_path.resolve()}")
    print(f"  decomposition CSV  : {decomp_csv_path.resolve()}")
    print(f"  readable report    : {report_path.resolve()}")
    if cube_info is not None:
        print(f"  PCS cube [ppm]     : {Path(cube_info['path']).resolve()}")
        print(
            f"    grid {cube_info['nx']}x{cube_info['ny']}x{cube_info['nz']}, "
            f"range {cube_info['min_ppm']:.3e} .. {cube_info['max_ppm']:.3e} ppm"
        )


if __name__ == "__main__":
    main()
