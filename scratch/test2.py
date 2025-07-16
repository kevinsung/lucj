import os
import pickle
from pathlib import Path

import numpy as np
from molecules_catalog.util import load_molecular_data


from qiskit.primitives import BitArray
from qiskit_addon_sqd.fermion import diagonalize_fermionic_hamiltonian, solve_sci_batch
from functools import partial


sample_filename = "lucj/n2_6-31g_10e16o/bond_distance-1.00000/connectivity-all-to-all/n_reps-2/with_final_orbital_rotation-True/multi_stage_optimization-True/begin_reps-20/step-2/sample.pickle"
with open(sample_filename, "rb") as f:
    bit_array_count = pickle.load(f)
    bit_array = BitArray.from_counts(bit_array_count)

rng = np.random.default_rng(0)

array = bit_array.to_bool_array()

# Generate n unique random integers from the specified range
unique_integers = np.random.choice(np.arange(0, array.shape[0]), size=100_000, replace=False)
array = array[unique_integers]
bit_array = BitArray.from_bool_array(array)


MOLECULES_CATALOG_DIR = Path(os.environ.get("MOLECULES_CATALOG_DIR"))

mol_data = load_molecular_data(
            "n2_6-31g_10e16o_d-1.00000",
            molecules_catalog_dir=MOLECULES_CATALOG_DIR,
        )

# Run SQD

norb = mol_data.norb
nelec = mol_data.nelec
mol_hamiltonian = mol_data.hamiltonian


n_reps_range = list(range(2, 25, 2))
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
max_dim_range = [None, 50_000, 100_000, 200_000]

print("Running SQD...\n")

sci_solver = partial(solve_sci_batch, spin_sq=0.0)
result = diagonalize_fermionic_hamiltonian(
    mol_hamiltonian.one_body_tensor,
    mol_hamiltonian.two_body_tensor,
    bit_array,
    samples_per_batch=1000,
    norb=norb,
    nelec=nelec,
    num_batches=3,
    energy_tol=1e-5,
    occupancies_tol=1e-3,
    max_iterations=100,
    sci_solver=sci_solver,
    symmetrize_spin=True,
    carryover_threshold=1e-3,
    seed=rng,
    max_dim=50_000
)
energy = result.energy + mol_data.core_energy
sci_state = result.sci_state
spin_squared = sci_state.spin_square()
if mol_data.fci_energy is not None:
    error = energy - mol_data.fci_energy
elif mol_data.sci_energy is not None:
    error = energy - mol_data.sci_energy
else:
    error = -1

# Save data
data = {
    "energy": energy,
    "error": error,
    "spin_squared": spin_squared,
    "sci_vec_shape": sci_state.amplitudes.shape,
}

print(data)