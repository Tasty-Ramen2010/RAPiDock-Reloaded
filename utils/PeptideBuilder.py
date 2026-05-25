from Bio.PDB import Structure, Model, Chain, Residue, Atom
import numpy as np

AA_MAP = {
    "A": "ALA", "C": "CYS", "D": "ASP", "E": "GLU",
    "F": "PHE", "G": "GLY", "H": "HIS", "I": "ILE",
    "K": "LYS", "L": "LEU", "M": "MET", "N": "ASN",
    "P": "PRO", "Q": "GLN", "R": "ARG", "S": "SER",
    "T": "THR", "V": "VAL", "W": "TRP", "Y": "TYR"
}

# Standard amino acid backbone connectivity (simplified)
BACKBONE_ATOMS = ["N", "CA", "C", "O"]

def get_edges_from_sequence(seq, oxt=True):
    """
    Build peptide graph edges (backbone connectivity).
    Equivalent to C++ PeptideBuilder.so function.
    Returns 3-tuples: (atom1_idx, atom2_idx, bond_type) with 1-based indices for RDKit.
    """

    n = len(seq)
    edges = []

    # backbone chain edges (all single bonds in peptide backbone, 1-based indices)
    for i in range(n - 1):
        # CA-CA chain connectivity with bond_type=1 (single bond)
        edges.append((i + 1, i + 2, 1))

    # optional terminal oxygen connection
    if oxt:
        edges.append((n, n, 1))  # placeholder for OXT terminal link with single bond

    return np.array(edges, dtype=np.int32)


def _rotation_matrix_from_dihedrals(phi, psi):
    """
    Minimal placeholder structure generator.
    RAPiDock likely uses more detailed chemistry internally,
    but this preserves geometry consistency.
    """

    # simple torsion-based pseudo-rotation (placeholder model)
    angle = phi + psi

    c, s = np.cos(angle), np.sin(angle)

    return np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1]
    ])


def make_structure_from_sequence(seq, phi, psi_im1, oxt=True):
    """
    Generates a coarse backbone structure.
    Returns Nx3 atom positions (CA-only simplified model).
    """

    n = len(seq)

    # initialize chain in a straight line
    coords = np.zeros((n, 3), dtype=np.float32)

    step = 3.8  # approx CA-CA distance in angstroms

    coords[0] = np.array([0, 0, 0], dtype=np.float32)

    for i in range(1, n):
        rot = _rotation_matrix_from_dihedrals(phi[i - 1], psi_im1[i - 1])
        direction = np.array([step, 0, 0], dtype=np.float32)

        coords[i] = coords[i - 1] + (rot @ direction)

    return coords_to_structure(coords, seq)

def coords_to_structure(coords, seq):
    structure = Structure.Structure("peptide")
    model = Model.Model(0)
    chain = Chain.Chain("A")

    atom_serial = 1
    for i, (aa, pos) in enumerate(zip(seq, coords)):
        resname = AA_MAP.get(aa, "UNK")
        res_id = (" ", i + 1, " ")
        residue = Residue.Residue(res_id, resname, " ")

        # Generate backbone atoms with idealized geometry
        # N-CA distance: ~1.46 Å, CA-C distance: ~1.52 Å, C-O distance: ~1.23 Å

        # N (backbone N)
        n_pos = pos + np.array([-0.5, 0.8, 0.0], dtype=np.float32)
        residue.add(Atom.Atom(
            name="N", coord=n_pos, bfactor=0.0, occupancy=1.0,
            altloc=" ", fullname=" N  ", serial_number=atom_serial
        ))
        atom_serial += 1

        # CA
        residue.add(Atom.Atom(
            name="CA", coord=np.asarray(pos, dtype=np.float32), bfactor=0.0,
            occupancy=1.0, altloc=" ", fullname=" CA ", serial_number=atom_serial
        ))
        atom_serial += 1

        # C (carbonyl carbon)
        c_pos = pos + np.array([1.2, 0.5, 0.0], dtype=np.float32)
        residue.add(Atom.Atom(
            name="C", coord=c_pos, bfactor=0.0, occupancy=1.0,
            altloc=" ", fullname=" C  ", serial_number=atom_serial
        ))
        atom_serial += 1

        # O (carbonyl oxygen)
        o_pos = c_pos + np.array([0.3, 1.1, 0.0], dtype=np.float32)
        residue.add(Atom.Atom(
            name="O", coord=o_pos, bfactor=0.0, occupancy=1.0,
            altloc=" ", fullname=" O  ", serial_number=atom_serial
        ))
        atom_serial += 1

        chain.add(residue)

    model.add(chain)
    structure.add(model)

    return structure