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

    Bug fix: the original code used an alpha-helical conformation (-57/-47) which
    causes RDKit's proximity-based bond perception to generate spurious long-range
    bonds between atoms from different residues that happen to be close in 3D space.
    We now use an extended conformation (-120/120) where atoms are well-separated,
    and additionally filter any cross-residue bonds that aren't valid covalent
    connections (peptide N-C bonds or the PRO ring N-CD bond).
    """
    n = len(seq)
    # Extended / beta-strand conformation: backbone atoms are far apart across
    # residues, preventing spurious proximity-based inter-residue bonds.
    phi_tmp = [-120.0] * max(n - 1, 0)
    psi_tmp = [ 120.0] * max(n - 1, 0)
    structure = make_structure_from_sequence(seq, phi_tmp, psi_tmp, oxt=oxt)

    # Serialise to an in-memory PDB string
    pdbio = PDBIO()
    pdbio.set_structure(structure)
    buf = io.StringIO()
    pdbio.save(buf)
    pdb_str = buf.getvalue()

    # Build atom-index → (residue_serial, atom_name) map from the PDB ATOM records.
    # This lets us filter out spurious cross-residue bonds below.
    atom_res: list[int]  = []   # residue serial per atom (0-based atom index)
    atom_name: list[str] = []   # atom name per atom (0-based atom index)
    for line in pdb_str.splitlines():
        if line.startswith("ATOM") or line.startswith("HETATM"):
            try:
                res_seq  = int(line[22:26].strip())
                aname    = line[12:16].strip()
                atom_res.append(res_seq)
                atom_name.append(aname)
            except ValueError:
                pass

    # Extract bonds with RDKit
    try:
        from rdkit import Chem
        from rdkit.Chem import RemoveHs as _RemoveHs

        mol = Chem.MolFromPDBBlock(pdb_str, sanitize=False, removeHs=False)
        if mol is not None:
            mol = _RemoveHs(mol, sanitize=False)
            edges: list[tuple[int, int, int]] = []
            for bond in mol.GetBonds():
                i0 = bond.GetBeginAtomIdx()   # 0-based
                j0 = bond.GetEndAtomIdx()
                bt = max(1, int(round(bond.GetBondTypeAsDouble())))

                # Guard against index overruns (H removal shifts indices)
                if i0 >= len(atom_res) or j0 >= len(atom_res):
                    continue

                ri, rj = atom_res[i0], atom_res[j0]
                ni, nj = atom_name[i0], atom_name[j0]

                if ri == rj:
                    # Intra-residue bond — always keep
                    pass
                else:
                    # Cross-residue bond: only keep if it is a known valid
                    # covalent connection between adjacent residues.
                    # (1) Peptide bond: C of residue k  →  N of residue k+1
                    # (2) PRO ring:     N of residue k  →  CD of same residue
                    #     (this appears as N in one residue and CD in another
                    #      only in PDB CONECT records; with template perception
                    #      it's intra-residue, so this branch is a safety net)
                    is_peptide = (abs(ri - rj) == 1 and
                                  {ni, nj} == {"C", "N"})
                    is_pro_ring = (abs(ri - rj) == 0 and
                                   {ni, nj} == {"N", "CD"})
                    if not (is_peptide or is_pro_ring):
                        continue  # drop spurious proximity bond

                edges.append((i0 + 1, j0 + 1, bt))  # → 1-based

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
