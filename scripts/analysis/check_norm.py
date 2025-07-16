import logging
import os
from pathlib import Path
import numpy as np

from lucj.params import LUCJParams, CompressedT2Params
from lucj.operator_task.lucj_compressed_t2_task import (
    LUCJCompressedT2Task,
)
from molecules_catalog.util import load_molecular_data
from ffsim.variational.util import interaction_pairs_spin_balanced
import ffsim

filename = f"logs/{os.path.splitext(os.path.relpath(__file__))[0]}.log"
os.makedirs(os.path.dirname(filename), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S %z",
    filename=filename,
)

DATA_ROOT = Path(os.environ.get("LUCJ_DATA_ROOT", "data"))
# DATA_DIR = DATA_ROOT / os.path.basename(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = DATA_ROOT 
MOLECULES_CATALOG_DIR = Path(os.environ.get("MOLECULES_CATALOG_DIR"))
MAX_PROCESSES = 8
OVERWRITE = False


molecule_name = "n2"
basis = "6-31g"
nelectron, norb = 10, 16
molecule_basename = f"{molecule_name}_{basis}_{nelectron}e{norb}o"

bond_distance_range = [1.0, 2.4]

connectivities = [
    "heavy-hex",
    # "square",
    "all-to-all",
]
n_reps_range = list(range(2, 10, 2))

for d in bond_distance_range:
    for connectivity in connectivities:
        for n_reps in n_reps_range:
            task_compressed_t2 = LUCJCompressedT2Task(
                                    molecule_basename=molecule_basename,
                                    bond_distance=d,
                                    lucj_params=LUCJParams(
                                        connectivity=connectivity,
                                        n_reps=n_reps,
                                        with_final_orbital_rotation=True,
                                    ),
                                    compressed_t2_params=CompressedT2Params(
                                        multi_stage_optimization=True,
                                        begin_reps=20,
                                        step=2
                                    )
                                )


            task_compressed_t2_connectivity = LUCJCompressedT2Task(
                                    molecule_basename=molecule_basename,
                                    bond_distance=d,
                                    lucj_params=LUCJParams(
                                        connectivity=connectivity,
                                        n_reps=n_reps,
                                        with_final_orbital_rotation=True,
                                    ),
                                    compressed_t2_params=None,
                                    connectivity_opt=True,
                                )

            operator_filename = DATA_DIR / task_compressed_t2.dirpath / "operator.npz"
            operator = np.load(operator_filename)
            diag_coulomb_mats_compressed_t2 = operator["diag_coulomb_mats"]
            orbital_rotations_compressed_t2 = operator["orbital_rotations"]

            operator_filename = DATA_DIR / task_compressed_t2_connectivity.dirpath / "operator.npz"
            operator = np.load(operator_filename)
            diag_coulomb_mats_compressed_t2_connectivity = operator["diag_coulomb_mats"]
            orbital_rotations_compressed_t2_connectivity = operator["orbital_rotations"]


            # Get molecular data and molecular Hamiltonian
            mol_data = load_molecular_data(
                f"{molecule_basename}_d-{d:.5f}",
                molecules_catalog_dir=MOLECULES_CATALOG_DIR,
            )
            norb = mol_data.norb
            nelec = mol_data.nelec
            mol_hamiltonian = mol_data.hamiltonian

            # Initialize Hamiltonian, initial state, and LUCJ parameters
            hamiltonian = ffsim.linear_operator(mol_hamiltonian, norb=norb, nelec=nelec)
            reference_state = ffsim.hartree_fock_state(norb, nelec)
            pairs_aa, pairs_ab = interaction_pairs_spin_balanced(
                connectivity, norb
            )

            # use CCSD to initialize parameters
            operator = ffsim.UCJOpSpinBalanced.from_t_amplitudes(
                mol_data.ccsd_t2,
                n_reps=n_reps,
                t1=mol_data.ccsd_t1,
                interaction_pairs=(pairs_aa, pairs_ab),
            )
            diag_coulomb_mats_reference = operator.diag_coulomb_mats
            orbital_rotations_reference = operator.orbital_rotations

            print(f"connectivity: {connectivity}, d: {d}, n_reps: {n_reps}")
            for layer in range(n_reps):
                diff_diag_coulomb_mats = diag_coulomb_mats_compressed_t2[layer] - diag_coulomb_mats_reference[layer]
                diff_orbital_rotations = orbital_rotations_compressed_t2[layer] - orbital_rotations_reference[layer]
                diff_diag_coulomb_mats_c = diag_coulomb_mats_compressed_t2_connectivity[layer] - diag_coulomb_mats_reference[layer]
                diff_orbital_rotations_c = orbital_rotations_compressed_t2_connectivity[layer] - orbital_rotations_reference[layer]
                print("coulumb matrices")
                print(f"norm of reference   : {np.sum(np.abs(diag_coulomb_mats_reference[layer]) ** 2):.4f}")
                print(f"norm of compressed  : {np.sum(np.abs(diag_coulomb_mats_compressed_t2[layer]) ** 2):.4f}")
                print(f"norm of compressed-c: {np.sum(np.abs(diag_coulomb_mats_compressed_t2_connectivity[layer]) ** 2):.4f}")
                print(f"norm of difference  : {np.sum(np.abs(diff_diag_coulomb_mats) ** 2):.4f}")
                print(f"norm of difference-c: {np.sum(np.abs(diff_diag_coulomb_mats_c) ** 2):.4f}")
                print("orbital rotation")
                print(f"norm of reference   : {np.sum(np.abs(orbital_rotations_reference[layer]) ** 2):.4f}")
                print(f"norm of compressed  : {np.sum(np.abs(orbital_rotations_compressed_t2[layer]) ** 2):.4f}")
                print(f"norm of compressed-c: {np.sum(np.abs(orbital_rotations_compressed_t2_connectivity[layer]) ** 2):.4f}")
                print(f"norm of difference  : {np.sum(np.abs(diff_orbital_rotations) ** 2):.4f}")
                print(f"norm of difference-c: {np.sum(np.abs(diff_orbital_rotations_c) ** 2):.4f}")
                input()

            