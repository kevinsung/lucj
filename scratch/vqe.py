
import os
from pathlib import Path
import ffsim
from lucj.params import LUCJParams, CompressedT2Params
from molecules_catalog.util import load_molecular_data
from lucj.operator_task.lucj_compressed_t2_task import (
    LUCJCompressedT2Task,
)
import numpy as np

DATA_ROOT = "/media/storage/WanHsuan.Lin/"
# DATA_DIR = DATA_ROOT / os.path.basename(os.path.dirname(os.path.abspath(__file__)))
data_dir = DATA_ROOT 
MOLECULES_CATALOG_DIR = Path(os.environ.get("MOLECULES_CATALOG_DIR"))
MAX_PROCESSES = 16
OVERWRITE = False

molecule_name = "fe2s2"
nelectron, norb = 30, 20
molecule_basename = f"{molecule_name}_{nelectron}e{norb}o"

shots = 100_000
samples_per_batch_range = [1000, 2000, 5000]
n_batches = 3
energy_tol = 1e-5
occupancies_tol = 1e-3
carryover_threshold = 1e-3
max_iterations = 100
symmetrize_spin = True
# TODO set entropy and generate seeds properly
entropy = 0
# max_dim_range = [None, 5e3, 1e4, 1e5, 2e5]
max_dim_range = [None]

task = LUCJCompressedT2Task(
        molecule_basename=molecule_basename,
        bond_distance=None,
        lucj_params=LUCJParams(
            connectivity="all-to-all",
            n_reps=2,
            with_final_orbital_rotation=True,
        ),
        compressed_t2_params=CompressedT2Params(
            multi_stage_optimization=True,
            begin_reps=20,
            step=2
        )
    )

# Get molecular data and molecular Hamiltonian
mol_data = load_molecular_data(
    task.molecule_basename,
    molecules_catalog_dir=MOLECULES_CATALOG_DIR,
)

norb = mol_data.norb
nelec = mol_data.nelec
mol_hamiltonian = mol_data.hamiltonian

# Initialize Hamiltonian, initial state, and LUCJ parameters
hamiltonian = ffsim.linear_operator(mol_hamiltonian, norb=norb, nelec=nelec)
reference_state = ffsim.hartree_fock_state(norb, nelec)

# use CCSD to initialize parameters
operator_filename = data_dir / task.dirpath / "operator.npz"
vqe_filename = data_dir / task.dirpath / "data.pickle"
state_vector_filename = data_dir / task.dirpath / "state_vector.npy"

rng = np.random.default_rng(0)

if os.path.exists(state_vector_filename):
    with open(state_vector_filename, "rb") as f:
        final_state = np.load(f)
else:
    if not os.path.exists(operator_filename):
        logging.info(f"Operator for {task} does not exists.\n")

    operator = np.load(operator_filename)
    diag_coulomb_mats = operator["diag_coulomb_mats"]
    orbital_rotations = operator["orbital_rotations"]
    
    final_orbital_rotation = None
    if mol_data.ccsd_t1 is not None:
        final_orbital_rotation = (
            ffsim.variational.util.orbital_rotation_from_t1_amplitudes(mol_data.ccsd_t1)
        )

    operator = ffsim.UCJOpSpinBalanced(
            diag_coulomb_mats=diag_coulomb_mats,
            orbital_rotations=orbital_rotations,
            final_orbital_rotation=final_orbital_rotation,
        )
    
    # Compute final state
    if not os.path.exists(state_vector_filename):
        final_state = ffsim.apply_unitary(reference_state, operator, norb=norb, nelec=nelec)
        with open(state_vector_filename, "wb") as f:
            np.save(f, final_state)

import time 

start = time.time()
# record vqe energy
print(f"{task} Computing VQE data...\n")
energy = np.vdot(final_state, hamiltonian @ final_state).real
error = energy - mol_data.sci_energy
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

print(f"{task} Saving VQE data...\n")
with open(vqe_filename, "wb") as f:
    pickle.dump(data, f)

print(f"total time: {time.time() - start}")