"""
Idealized 3D peptide backbone + C-beta PDB generator from amino-acid sequences.

Uses standard Engh & Huber covalent geometry (N-CA = 1.458 Å, CA-C = 1.525 Å,
C-N = 1.329 Å, C=O = 1.231 Å, CA-CB = 1.521 Å) in a canonical alpha-helical
conformation (phi = -57°, psi = -47°, omega = 180°) via NeRF (Natural Extension
Reference Frame) placement.
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np

AA1_TO_AA3: dict[str, str] = {
    "A": "ALA",
    "C": "CYS",
    "D": "ASP",
    "E": "GLU",
    "F": "PHE",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "K": "LYS",
    "L": "LEU",
    "M": "MET",
    "N": "ASN",
    "P": "PRO",
    "Q": "GLN",
    "R": "ARG",
    "S": "SER",
    "T": "THR",
    "V": "VAL",
    "W": "TRP",
    "Y": "TYR",
}


def _place_atom(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    bond: float,
    angle_deg: float,
    dihedral_deg: float,
) -> np.ndarray:
    """Place atom D given prior three atoms A-B-C, C-D bond length, B-C-D angle, and A-B-C-D dihedral."""
    bc = c - b
    bc_hat = bc / np.linalg.norm(bc)
    n = np.cross(b - a, bc_hat)
    norm_n = np.linalg.norm(n)
    if norm_n < 1e-8:
        n = np.array([0.0, 0.0, 1.0])
        norm_n = 1.0
    n_hat = n / norm_n
    m = np.cross(n_hat, bc_hat)

    theta = math.radians(180.0 - angle_deg)
    phi = math.radians(dihedral_deg)
    return c + bond * (
        bc_hat * math.cos(theta)
        + m * math.sin(theta) * math.cos(phi)
        - n_hat * math.sin(theta) * math.sin(phi)
    )


def build_peptide_pdb_string(sequence: str) -> str:
    """Generate a standard PDB string for a natural amino-acid peptide sequence."""
    seq = sequence.strip().upper()
    if not seq:
        raise ValueError("Peptide sequence must not be empty.")
    invalid = sorted({ch for ch in seq if ch not in AA1_TO_AA3})
    if invalid:
        raise ValueError(
            f"Unsupported non-standard amino acid(s) for 3D PDB generation: {''.join(invalid)}"
        )

    phi, psi, omega = -57.0, -47.0, 180.0
    lines = ["HEADER    AUTO-GENERATED PEPTIDE 3D STRUCTURE"]
    atom_idx = 1

    n_coord = np.array([0.0, 0.0, 0.0])
    ca_coord = np.array([1.458, 0.0, 0.0])
    c_coord = _place_atom(np.array([-1.0, 1.0, 0.0]), n_coord, ca_coord, 1.525, 111.2, phi)
    prev_n, prev_ca, prev_c = n_coord, ca_coord, c_coord

    for res_i, aa in enumerate(seq, start=1):
        resname = AA1_TO_AA3[aa]
        if res_i > 1:
            n_coord = _place_atom(prev_n, prev_ca, prev_c, 1.329, 116.2, psi)
            ca_coord = _place_atom(prev_ca, prev_c, n_coord, 1.458, 121.7, omega)
            c_coord = _place_atom(prev_c, n_coord, ca_coord, 1.525, 111.2, phi)

        o_coord = _place_atom(n_coord, ca_coord, c_coord, 1.231, 120.8, psi + 180.0)
        atoms: list[tuple[str, np.ndarray, str]] = [
            ("N", n_coord, "N"),
            ("CA", ca_coord, "C"),
            ("C", c_coord, "C"),
            ("O", o_coord, "O"),
        ]
        if aa != "G":
            cb_coord = _place_atom(c_coord, n_coord, ca_coord, 1.521, 110.5, -122.6)
            atoms.append(("CB", cb_coord, "C"))

        for aname, coord, elem in atoms:
            lines.append(
                f"ATOM  {atom_idx:5d}  {aname:<3s} {resname:>3s} A{res_i:4d}    "
                f"{coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}  1.00 20.00           {elem:>2s}"
            )
            atom_idx += 1

        prev_n, prev_ca, prev_c = n_coord, ca_coord, c_coord

    last_res = AA1_TO_AA3[seq[-1]]
    lines.append(f"TER   {atom_idx:5d}      {last_res:>3s} A{len(seq):4d}")
    lines.append("END")
    return "\n".join(lines) + "\n"


def write_temp_peptide_pdb(sequence: str) -> Path:
    """Write an auto-generated PDB file to a temporary path (caller must unlink)."""
    pdb_text = build_peptide_pdb_string(sequence)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix="pep_",
        suffix=".pdb",
        delete=False,
    ) as tf:
        tf.write(pdb_text)
        return Path(tf.name)
