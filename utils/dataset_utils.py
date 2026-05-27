##########################################################################
# File Name: dataset_utils.py
# Author: huifeng
# mail: huifengzhao@zju.edu.cn
# Created Time: Tue 24 Oct 2023 01:47:39 PM CST
#########################################################################


from rdkit import Chem
import MDAnalysis
from utils.PeptideBuilder import get_edges_from_sequence, make_structure_from_sequence
import re
from Bio.PDB import PDBParser


def standard_residue_sort(item):
    # convert to str
    if isinstance(item, int):
        return item, 0
    else:
        s = str(item)
        # extract the digital part
        num = "".join([i for i in s if i.isdigit()])

        # extract the non digital part
        non_num = "".join([i for i in s if not i.isdigit()])
        code = ord(non_num)
        if num == "1":
            return (int(num) if num else 0, -code)
        else:
            return (int(num) if num else 0, code)


three_to_one = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}

three_to_one_esm = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "MSE": "M",  # this is almost the same AA as MET. The sulfur is just replaced by Selen
    "PHE": "F",
    "PRO": "P",
    "PYL": "O",  #
    "SER": "S",
    "SEC": "U",  #
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "ASX": "B",  #
    "GLX": "Z",  #
    "XAA": "X",  #
    "XLE": "J",
}  #


def read_pdb_with_connect_labels(
    pdbfile: str, sanitize: bool = True, addHs: bool = False
):
    mol = Chem.MolFromPDBFile(pdbfile, sanitize=False)

    rw_mol = Chem.RWMol(mol)
    while rw_mol.GetNumBonds() > 0:
        bond = rw_mol.GetBondWithIdx(0)
        rw_mol.RemoveBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())

    edges = set()
    for line in open(pdbfile, "r").readlines():
        if line.startswith("CONECT"):
            content = line.strip().split()[1:]
            start = content[0]
            ends = content[1:]
            for end in set(ends):
                edges.add((tuple(sorted((int(start), int(end)))) + (ends.count(end),)))

    for edge in edges:
        atom1_idx, atom2_idx, bond_type = edge
        if bond_type == 1:
            rw_mol.AddBond(atom1_idx - 1, atom2_idx - 1, Chem.BondType.SINGLE)
        elif bond_type == 2:
            rw_mol.AddBond(atom1_idx - 1, atom2_idx - 1, Chem.BondType.DOUBLE)
        elif bond_type == 3:
            rw_mol.AddBond(atom1_idx - 1, atom2_idx - 1, Chem.BondType.TRIPLE)
        else:
            raise RuntimeError

    if sanitize:
        Chem.SanitizeMol(rw_mol)

    if addHs:
        mol = Chem.AddHs(rw_mol, addCoords=True)
    else:
        mol = rw_mol

    return mol


def read_pdb_with_seq(pdbfile: str, sanitize: bool = True, addHs: bool = False):
    mol = Chem.MolFromPDBFile(pdbfile, sanitize=False)

    rw_mol = Chem.RWMol(mol)
    while rw_mol.GetNumBonds() > 0:
        bond = rw_mol.GetBondWithIdx(0)
        rw_mol.RemoveBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())

    u = MDAnalysis.Universe(pdbfile)
    trans = {}
    seq = []
    for res_idx, residue in enumerate(u.residues):
        res_name = residue.resname.strip()
        seq.append(
            three_to_one[res_name]
            if res_name in three_to_one.keys()
            else f"[{res_name}]"
        )
        trans[
            (
                int(residue.resid)
                if residue.icode == ""
                else f"{residue.resid}{residue.icode.strip()}"
            )
        ] = res_idx
    sorted_res_keys = sorted(set(trans.keys()), key=standard_residue_sort)
    real_idx = [trans[idx] for idx in sorted_res_keys]
    seq = [seq[i] for i in real_idx]
    seq = "".join(seq)
    oxt = len(u.atoms.select_atoms("name OXT")) == 1

    # get_edges_from_sequence returns 1-based indices into the PeptideBuilder
    # template atom ordering (backbone-only: N,CA,C,O per residue + optional OXT).
    # The actual pose PDB may have Cβ and/or side-chain atoms at different positions,
    # making direct index application wrong (carbon ends up with valence 5).
    # Fix: use return_atom_keys=True to get (res_serial, atom_name) per template atom,
    # then remap to actual mol atom indices via sequential residue numbering.
    edges_arr, template_atom_keys = get_edges_from_sequence(seq, oxt, return_atom_keys=True)
    edges = edges_arr.tolist()

    # Build (sequential_1based_residue_idx, atom_name) -> 0-based RDKit atom index.
    # Sequential numbering follows sorted_res_keys order (same as the seq we extracted).
    # We parse the PDB file directly for atom ordering because Chem.MolFromPDBFile
    # uses the same ATOM-record order, and direct PDB parsing is safer than relying on
    # the MonomerInfo API (which may return a base AtomMonomerInfo without resid methods).
    pdb_res_to_seq: dict = {k: i + 1 for i, k in enumerate(sorted_res_keys)}
    actual_idx_map: dict = {}
    rdkit_atom_idx: int = 0
    with open(pdbfile) as _pf:
        for _line in _pf:
            if not (_line.startswith("ATOM") or _line.startswith("HETATM")):
                continue
            try:
                _resnum = int(_line[22:26].strip())
                _icode  = _line[26].strip() if len(_line) > 26 else ""
                _aname  = _line[12:16].strip()
            except (ValueError, IndexError):
                rdkit_atom_idx += 1
                continue
            _res_key = _resnum if not _icode else f"{_resnum}{_icode}"
            _seq_idx = pdb_res_to_seq.get(_res_key)
            if _seq_idx is not None:
                actual_idx_map[(_seq_idx, _aname)] = rdkit_atom_idx
            rdkit_atom_idx += 1

    for edge in edges:
        t_i1, t_j1, bond_type = edge          # 1-based template indices
        key_i = template_atom_keys[t_i1 - 1]  # (seq_res, atom_name) from template
        key_j = template_atom_keys[t_j1 - 1]
        ai = actual_idx_map.get(key_i)
        aj = actual_idx_map.get(key_j)
        if ai is None or aj is None:
            continue  # atom absent in actual PDB (shouldn't happen for backbone)
        if bond_type == 1:
            rw_mol.AddBond(ai, aj, Chem.BondType.SINGLE)
        elif bond_type == 2:
            rw_mol.AddBond(ai, aj, Chem.BondType.DOUBLE)
        elif bond_type == 3:
            rw_mol.AddBond(ai, aj, Chem.BondType.TRIPLE)
        else:
            raise RuntimeError

    if sanitize:
        Chem.SanitizeMol(rw_mol)

    if addHs:
        mol = Chem.AddHs(rw_mol, addCoords=True)
    else:
        mol = rw_mol

    return mol


def get_sequences_from_pdbfile(file_path):
    biopython_parser = PDBParser()
    structure = biopython_parser.get_structure("random_id", file_path)
    structure = structure[0]
    sequence = None
    for i, chain in enumerate(structure):
        seq_pro = ""
        seq_dic = {}
        for res_idx, residue in enumerate(chain):
            if residue.get_resname() == "HOH":
                continue
            c_alpha, n, c, o = None, None, None, None
            for atom in residue:
                if atom.name == "CA":
                    c_alpha = list(atom.get_vector())
                if atom.name == "N":
                    n = list(atom.get_vector())
                if atom.name == "C":
                    c = list(atom.get_vector())
                if atom.name == "O":
                    o = list(atom.get_vector())
            if (
                c_alpha != None and n != None and c != None and o != None
            ):  # only append residue if it is an amino acid
                try:
                    seq_dic[
                        (
                            int(residue.id[1])
                            if residue.id[2] == " "
                            else f"{residue.id[1]}{residue.id[2].strip()}"
                        )
                    ] = three_to_one_esm[residue.get_resname()]
                except Exception as e:
                    seq_dic[
                        (
                            int(residue.id[1])
                            if residue.id[2] == " "
                            else f"{residue.id[1]}{residue.id[2].strip()}"
                        )
                    ] = "-"
                    print(
                        "encountered unknown AA: ",
                        residue.get_resname(),
                        " in the complex. Replacing it with a dash - .",
                    )

        try:
            digit_list = [i for i in seq_dic.keys() if isinstance(i, int)]
            for idx in sorted(
                (
                    set(seq_dic.keys())
                    | set(range(min(digit_list), max(digit_list) + 1))
                    if len(digit_list) > 0
                    else set(seq_dic.keys())
                ),
                key=standard_residue_sort,
            ):
                try:
                    seq_pro += seq_dic[idx]
                except:
                    seq_pro += "-"
                    print(
                        "missed AA: ",
                        idx,
                        " in the complex ",
                        file_path,
                        ". Add it with a dash - .",
                    )
        except:
            print("=========================================" + file_path)

        if sequence is None:
            sequence = seq_pro
        else:
            sequence += ":" + seq_pro

    return sequence


def get_sequences(descriptions):
    new_sequences = []
    for description in descriptions:
        if "pdb" in description:
            new_sequences.append(get_sequences_from_pdbfile(description))
        else:
            new_sequences.append(re.sub(r"\[.*?\]", "-", description))
    return new_sequences
