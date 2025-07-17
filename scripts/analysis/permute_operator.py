import logging
import os
from pathlib import Path
import numpy as np

from lucj.params import LUCJParams, CompressedT2Params
from lucj.operator_task.lucj_compressed_t2_task import (
    LUCJCompressedT2Task,
)
import scipy.stats
from molecules_catalog.util import load_molecular_data
from ffsim.variational.util import interaction_pairs_spin_balanced
import ffsim
import pickle

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
    # "heavy-hex",
    # "square",
    "all-to-all",
]
n_reps_range = list(range(4, 14, 2))

def generate_operator_permutation(diag_coulomb_mats, orbital_rotations):
    operators = []
    # original operator
    # operator = ffsim.UCJOpSpinBalanced(
    #     diag_coulomb_mats=diag_coulomb_mats,
    #     orbital_rotations=orbital_rotations,
    #     final_orbital_rotation=final_orbital_rotation,
    # )
    # operators.append(operator)
    list_norm = []
    n_reps, _, _, _ = diag_coulomb_mats.shape

    for layer in range(n_reps):
        list_norm.append(np.sum(np.abs(diag_coulomb_mats_compressed_t2[layer]) ** 2))
    list_norm = np.array(list_norm)

    # sort from small to large
    ascending_norm = np.argsort(list_norm)
    list_diag_coulomb_mat = [None for _ in range(n_reps)]
    list_orbital_rotation = [None for _ in range(n_reps)]
    for idx, diag_coulomb_mat, orbital_rotation in zip(ascending_norm, diag_coulomb_mats, orbital_rotations):
        list_diag_coulomb_mat[idx] = diag_coulomb_mat
        list_orbital_rotation[idx] = orbital_rotation
    diag_coulomb_mats = np.array(list_diag_coulomb_mat)
    orbital_rotations = np.array(list_orbital_rotation)
    operator = ffsim.UCJOpSpinBalanced(
        diag_coulomb_mats=diag_coulomb_mats,
        orbital_rotations=orbital_rotations,
        final_orbital_rotation=final_orbital_rotation,
    )
    operators.append(operator)
    # sort from large to small
    diag_coulomb_mats = np.flip(diag_coulomb_mats, 0)
    orbital_rotations = np.flip(orbital_rotations, 0)
    operator = ffsim.UCJOpSpinBalanced(
        diag_coulomb_mats=diag_coulomb_mats,
        orbital_rotations=orbital_rotations,
        final_orbital_rotation=final_orbital_rotation,
    )
    operators.append(operator)
    # random shuffle
    np.random.shuffle(ascending_norm)
    for idx, diag_coulomb_mat, orbital_rotation in zip(ascending_norm, diag_coulomb_mats, orbital_rotations):
        list_diag_coulomb_mat[idx] = diag_coulomb_mat
        list_orbital_rotation[idx] = orbital_rotation
    diag_coulomb_mats = np.array(list_diag_coulomb_mat)
    orbital_rotations = np.array(list_orbital_rotation)
    operator = ffsim.UCJOpSpinBalanced(
        diag_coulomb_mats=diag_coulomb_mats,
        orbital_rotations=orbital_rotations,
        final_orbital_rotation=final_orbital_rotation,
    )
    operators.append(operator)
    return operators

    
for d in bond_distance_range:
    for connectivity in connectivities:
        for n_reps in n_reps_range:
            print(n_reps)
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

            operator_filename = DATA_DIR / task_compressed_t2.dirpath / "operator.npz"
            ori_data_filename = DATA_DIR / task_compressed_t2.dirpath / "data.pickle"
            asc_data_filename = DATA_DIR / task_compressed_t2.dirpath / "asc_data.pickle"
            des_data_filename = DATA_DIR / task_compressed_t2.dirpath / "des_data.pickle"
            rand_data_filename = DATA_DIR / task_compressed_t2.dirpath / "rand_data.pickle"
            file_list = [asc_data_filename, des_data_filename, rand_data_filename]
            operator = np.load(operator_filename)
            diag_coulomb_mats_compressed_t2 = operator["diag_coulomb_mats"]
            orbital_rotations_compressed_t2 = operator["orbital_rotations"]

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
            final_orbital_rotation = (
                ffsim.variational.util.orbital_rotation_from_t1_amplitudes(mol_data.ccsd_t1)
            )

            operators = generate_operator_permutation(diag_coulomb_mats_compressed_t2, orbital_rotations_compressed_t2)
            for operator, file in zip(operators, file_list):
                final_state = ffsim.apply_unitary(reference_state, operator, norb=norb, nelec=nelec)
                energy = np.vdot(final_state, hamiltonian @ final_state).real
                error = energy - mol_data.fci_energy
                spin_squared = ffsim.spin_square(
                    final_state, norb=mol_data.norb, nelec=mol_data.nelec
                )
                probs = np.abs(final_state) ** 2
                entropy = scipy.stats.entropy(probs)

                data = {
                    "energy": energy,
                    "error": error,
                    "spin_squared": spin_squared,
                    "entropy": entropy,
                }
                with open(file, "wb") as f:
                    pickle.dump(data, f)
            

            