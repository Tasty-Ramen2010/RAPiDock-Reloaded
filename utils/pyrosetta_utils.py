##########################################################################
# File Name: pyrosetta_opt_per.py
# Author: huifeng
# mail: huifengzhao@zju.edu.cn
# Created Time: Mon 29 Apr 2024 11:10:07 AM CST
#########################################################################
#!/bin/bash

import psutil
import os

# Gate: PyRosetta requires ≥32 GB RAM and must be installed.
_ram_gb = psutil.virtual_memory().total / (1024 ** 3)
PYROSETTA_AVAILABLE = False

try:
    if _ram_gb <= 32:
        raise RuntimeError(
            f"PyRosetta requires at least 32 GB of RAM "
            f"(this system has {_ram_gb:.1f} GB)."
        )
    import pyrosetta
    from pyrosetta.rosetta.protocols.relax import FastRelax
    from pyrosetta.rosetta.core.pack.task import TaskFactory
    from pyrosetta.rosetta.core.pack.task import operation
    from pyrosetta.rosetta.core.select import residue_selector as selections
    from pyrosetta.rosetta.core.select.movemap import MoveMapFactory, move_map_action

    pyrosetta.init(
        " ".join(
            [
                "-mute",
                "all",
                "-use_input_sc",
                "-ignore_unrecognized_res",
                "-ignore_zero_occupancy",
                "false",
                "-load_PDB_components",
                "false",
                "-relax:default_repeats",
                "2",
                "-no_fconfig",
                "-use_terminal_residues",
                "true",
                "-in:file:silent_struct_type",
                "binary",
            ]
        ),
        silent=True,
    )
    scorefxn = pyrosetta.create_score_function("ref2015")
    PYROSETTA_AVAILABLE = True
    print("PyRosetta successfully imported.")

except Exception as _pr_err:
    print(f"PyRosetta unavailable: {_pr_err}  — ref2015 scoring disabled.")


if PYROSETTA_AVAILABLE:
    class RelaxRegion(object):  # Fast Relax
        def __init__(self, max_iter=20):
            super().__init__()
            self.scorefxn = pyrosetta.create_score_function("ref2015")
            self.fast_relax = FastRelax()
            self.fast_relax.set_scorefxn(self.scorefxn)
            self.fast_relax.max_iter(max_iter)

        def __call__(self, protein_file, peptide_file):
            protein_pose = pyrosetta.pose_from_pdb(protein_file)
            peptide_pose = pyrosetta.pose_from_pdb(peptide_file)
            peptide_pose.pdb_info().set_chains("p")
            protein_pose.append_pose_by_jump(peptide_pose, protein_pose.total_residue())
            tf = TaskFactory()
            tf.push_back(operation.InitializeFromCommandline())
            tf.push_back(
                operation.RestrictToRepacking()
            )  # Only allow residues to repack. No design at any position.
            gen_selector = selections.ChainSelector("p")
            jr = pyrosetta.rosetta.core.select.jump_selector.JumpForResidue(
                protein_pose.total_residue()
            )
            all_selector = selections.TrueResidueSelector()
            prevent_repacking_rlt = operation.PreventRepackingRLT()
            tf.push_back(
                operation.OperateOnResidueSubset(prevent_repacking_rlt, all_selector)
            )
            fr = self.fast_relax
            pose = protein_pose.clone()

            mmf = MoveMapFactory()
            mmf.add_bb_action(move_map_action.mm_disable, all_selector)
            mmf.add_chi_action(move_map_action.mm_disable, all_selector)
            mmf.add_bb_action(move_map_action.mm_enable, gen_selector)
            mmf.add_chi_action(move_map_action.mm_enable, gen_selector)
            mmf.add_jump_action(move_map_action.mm_enable, jr)
            mm = mmf.create_movemap_from_pose(pose)
            fr.set_movemap(mm)
            fr.set_task_factory(tf)
            fr.constrain_coords(True)
            fr.apply(pose)
            chain1_residues = [
                i
                for i in range(1, protein_pose.total_residue() + 1)
                if protein_pose.pdb_info().chain(i) != "p"
            ]
            superimpose_mover = pyrosetta.rosetta.protocols.simple_moves.SuperimposeMover(
                protein_pose,
                chain1_residues[0],
                chain1_residues[-1],
                chain1_residues[0],
                chain1_residues[-1],
                False,
            )
            superimpose_mover.apply(pose)
            peptide_pose = list(pose.split_by_chain())[-2]
            protein_pose_list = list(pose.split_by_chain())[:-2]
            protein_pose = protein_pose_list[0]
            if len(protein_pose_list) > 1:
                for i in protein_pose_list[1:]:
                    protein_pose.append_pose_by_jump(i, protein_pose.total_residue())
            return pose, peptide_pose, protein_pose

    RR = RelaxRegion()

    def dump_pdb(pose, file_name):
        pose.dump_pdb(file_name)

    def relax_score(inputs):  # protein, peptide, outputfile, score_or_not
        protein, peptide, output, flag = inputs
        complex_pose, peptide_pose, protein_pose = RR(protein, peptide)
        dump_pdb(peptide_pose, output)
        os.remove(peptide)
        if flag:
            complex_score = scorefxn(complex_pose)
            peptide_score = scorefxn(peptide_pose)
            protein_score = scorefxn(protein_pose)
            # score = complex_score - (protein_score + peptide_score)  # affinity
            score = complex_score  # pose ranking
            return score

else:
    # PyRosetta unavailable — provide a no-op stub so the import succeeds.
    def relax_score(inputs):
        return None
