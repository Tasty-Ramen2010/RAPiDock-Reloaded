"""
RAPiDock-Reloaded — cross-platform PeptideBuilder.so
"""

from __future__ import annotations

import io
import math
import warnings
import numpy as np
from Bio.PDB import Structure, Model, Chain, Residue, Atom, PDBIO


# mapping for amino acids from 1 letter code to 3 letter code
AA3 = {
    "A": "ALA", "C": "CYS", "D": "ASP", "E": "GLU", "F": "PHE",
    "G": "GLY", "H": "HIS", "I": "ILE", "K": "LYS", "L": "LEU",
    "M": "MET", "N": "ASN", "P": "PRO", "Q": "GLN", "R": "ARG",
    "S": "SER", "T": "THR", "V": "VAL", "W": "TRP", "Y": "TYR",
}

# Standard AAs
_STANDARD = set("ACDEFGHIKLMNPQRSTVWY")


# ── optional: PeptideBuilder package (pip install PeptideBuilder) ──────────────
try:
    import PeptideBuilder as _PB
    import PeptideBuilder.Geometry as _GEO
    _HAS_PB = True
except ImportError:
    _HAS_PB = False
    warnings.warn(
        "PeptideBuilder package not found.  Full-atom initial structures require it:\n"
        "  pip install PeptideBuilder\n"
        "Falling back to backbone-only (N-CA-C-O) geometry.  The diffusion "
        "model randomises positions immediately, so docking accuracy is unaffected,\n"
        "but sidechain torsion updates will be backbone-only for sequence inputs.",
        stacklevel=2,
    )

# Take sequence, phi and psi angles in degrees, and add oxygen when True to return a Bio.PDB structure
def make_structure_from_sequence(seq: str, phi, psi_im1, oxt: bool = True) -> "Bio.PDB.Structure.Structure":
    if _HAS_PB:
        return _make_structure_peptidebuilder(seq, phi, psi_im1, oxt)
    return _make_structure_fallback(seq, phi, psi_im1, oxt)

# Return int32 array of heavy-atom bond edges - RDKit availability only
def get_edges_from_sequence(seq: str, oxt: bool = True) -> np.ndarray:
    """
    Return heavy-atom bond indices for *seq* as an (N_bonds, 3) int32 array.

    Columns: [src_1based, dst_1based, bond_type]

    The atom ordering matches the PDB produced by make_structure_from_sequence.
    peptide_feature.py slices [:, :2] and subtracts 1 to get 0-based indices.

    Bond graph is extracted via RDKit for correctness.  Falls back to
    backbone-only edges if RDKit parsing fails.
    """
    n = len(seq)
    phi_tmp  = [-57.0] * max(n - 1, 0)   # alpha-helix; only ordering matters
    psi_tmp  = [-47.0] * max(n - 1, 0)
    structure = make_structure_from_sequence(seq, phi_tmp, psi_tmp, oxt=oxt)

    # Serialise to an in-memory PDB string
    pdbio = PDBIO()
    pdbio.set_structure(structure)
    buf = io.StringIO()
    pdbio.save(buf)
    pdb_str = buf.getvalue()

    # Extract bonds with RDKit
    try:
        from rdkit import Chem
        from rdkit.Chem import RemoveHs as _RemoveHs

        mol = Chem.MolFromPDBBlock(pdb_str, sanitize=False, removeHs=False)
        if mol is not None:
            mol = _RemoveHs(mol, sanitize=False)
            edges: list[tuple[int, int, int]] = []
            for bond in mol.GetBonds():
                i  = bond.GetBeginAtomIdx() + 1   # → 1-based
                j  = bond.GetEndAtomIdx()   + 1
                bt = max(1, int(round(bond.GetBondTypeAsDouble())))
                edges.append((i, j, bt))
            if edges:
                return np.array(edges, dtype=np.int32)
    except Exception as exc:
        warnings.warn(
            f"get_edges_from_sequence: RDKit bond extraction failed ({exc}); "
            "using backbone-only fallback.",
            stacklevel=2,
        )

    return _backbone_edges_fallback(seq, oxt)

# PeptideBuilder usage
def _make_structure_peptidebuilder(seq, phi, psi_im1, oxt):
    phi     = list(phi)
    psi_im1 = list(psi_im1)

    aa0 = seq[0] if seq[0] in _STANDARD else "G"
    geo = _GEO.geometry(aa0)
    structure = _PB.initialize_res(geo)

    for i in range(1, len(seq)):
        aa = seq[i] if seq[i] in _STANDARD else "G"
        geo = _GEO.geometry(aa)
        geo.phi     = float(phi[i - 1])
        geo.psi_im1 = float(psi_im1[i - 1])
        _PB.add_residue(structure, geo)

    if oxt:
        _PB.add_terminal_OXT(structure)

    return structure

# PeptideBuilder fallback - when not installed
def _make_structure_fallback(seq, phi, psi_im1, oxt):
    phi     = list(phi)
    psi_im1 = list(psi_im1)
    n       = len(seq)

    CA_CA = 3.8   # Å — approximate Cα–Cα virtual bond length

    # Place Cα atoms along a torsion-driven chain
    coords_ca = np.zeros((n, 3), dtype=np.float32)
    for i in range(1, n):
        ph = math.radians(float(phi[i - 1]))     # ← degrees → radians (was the bug)
        ps = math.radians(float(psi_im1[i - 1]))
        angle = ph + ps
        c, s  = math.cos(angle), math.sin(angle)
        rot   = np.array([[c, -s, 0.0],
                           [s,  c, 0.0],
                           [0.0, 0.0, 1.0]], dtype=np.float32)
        coords_ca[i] = coords_ca[i - 1] + rot @ np.array([CA_CA, 0.0, 0.0],
                                                           dtype=np.float32)

    # Build Bio.PDB structure
    structure = Structure.Structure("peptide")
    model_obj = Model.Model(0)
    chain_obj = Chain.Chain("A")

    serial = 1
    for i, (aa, ca) in enumerate(zip(seq, coords_ca)):
        resname = AA3.get(aa, "UNK")
        res = Residue.Residue((" ", i + 1, " "), resname, " ")

        # Idealized backbone offsets from Cα (Å)
        backbone = {
            "N" : ca + np.array([-0.55,  0.80, 0.00], dtype=np.float32),
            "CA": ca.copy(),
            "C" : ca + np.array([ 1.20,  0.50, 0.00], dtype=np.float32),
            "O" : ca + np.array([ 1.50,  1.55, 0.00], dtype=np.float32),
        }
        for name, pos in backbone.items():
            res.add(Atom.Atom(
                name=name, coord=pos, bfactor=0.0, occupancy=1.0,
                altloc=" ", fullname=f" {name:<3}", serial_number=serial,
            ))
            serial += 1

        chain_obj.add(res)

    # Add C-terminal OXT to the last residue
    if oxt and n > 0:
        last_res = list(chain_obj.get_residues())[-1]
        c_pos   = last_res["C"].get_vector().get_array()
        oxt_pos = c_pos + np.array([1.25, -1.00, 0.00], dtype=np.float32)
        last_res.add(Atom.Atom(
            name="OXT", coord=oxt_pos, bfactor=0.0, occupancy=1.0,
            altloc=" ", fullname=" OXT", serial_number=serial,
        ))

    model_obj.add(chain_obj)
    structure.add(model_obj)
    return structure

# backbone-only bond edges for a 4-atom-per-residue structure; RDKit fallback
def _backbone_edges_fallback(seq: str, oxt: bool) -> np.ndarray:
    edges: list[tuple[int, int, int]] = []
    n = len(seq)

    for i in range(n):
        base = i * 4 + 1
        N_i, CA_i, C_i, O_i = base, base + 1, base + 2, base + 3
        edges += [(N_i, CA_i, 1), (CA_i, C_i, 1), (C_i, O_i, 2)]
        if i < n - 1:
            N_next = (i + 1) * 4 + 1
            edges.append((C_i, N_next, 1))

    if oxt and n > 0:
        last_C   = (n - 1) * 4 + 3
        oxt_atom = n * 4 + 1
        edges.append((last_C, oxt_atom, 1))

    # float64 not support by most OS, using float32
    return np.array(edges, dtype=np.int32)


# old function to convert an array of coordinates and a sequence into a Bio.PDB structure (old)
def coords_to_structure(coords, seq):
    structure = Structure.Structure("peptide")
    model_obj = Model.Model(0)
    chain_obj = Chain.Chain("A")

    serial = 1
    for i, (aa, pos) in enumerate(zip(seq, coords)):
        resname = AA3.get(aa, "UNK")
        res = Residue.Residue((" ", i + 1, " "), resname, " ")

        offsets = {
            "N" : np.array([-0.55,  0.80, 0.00], dtype=np.float32),
            "CA": np.zeros(3, dtype=np.float32),
            "C" : np.array([ 1.20,  0.50, 0.00], dtype=np.float32),
            "O" : np.array([ 1.50,  1.55, 0.00], dtype=np.float32),
        }
        for name, off in offsets.items():
            atom_pos = np.asarray(pos, dtype=np.float32) + off
            res.add(Atom.Atom(
                name=name, coord=atom_pos, bfactor=0.0, occupancy=1.0,
                altloc=" ", fullname=f" {name:<3}", serial_number=serial,
            ))
            serial += 1

        chain_obj.add(res)

    model_obj.add(chain_obj)
    structure.add(model_obj)
    return structure
