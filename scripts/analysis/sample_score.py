import itertools
import os
import pickle
from pathlib import Path
from qiskit.primitives import BitArray
import ffsim
from lucj.params import LUCJParams, CompressedT2Params
from lucj.hardware_sqd_task.lucj_compressed_t2_task import HardwareSQDEnergyTask

from qiskit_addon_sqd.counts import bit_array_to_arrays, bitstring_matrix_to_integers
from qiskit_addon_sqd.subsampling import postselect_by_hamming_right_and_left


import numpy as np

DATA_ROOT = "/media/storage/WanHsuan.Lin/"
MOLECULES_CATALOG_DIR = Path(os.environ.get("MOLECULES_CATALOG_DIR"))

molecule_name = "n2"
basis = "6-31g"
nelectron, norb = 10, 16
molecule_basename = f"{molecule_name}_{basis}_{nelectron}e{norb}o"
bond_distance = 1.2
begin_reps = 20
step = 2


shots = 1_000_000
n_batches = 1
energy_tol = 1e-8
occupancies_tol = 1e-5
carryover_threshold = 1e-4
max_iterations = 10
symmetrize_spin = True
# TODO set entropy and generate seeds properly
entropy = 0

n_reps = 1

max_dim = 1000
samples_per_batch = 4000

task_compressed_t2_hardware = HardwareSQDEnergyTask(
    molecule_basename=molecule_basename,
    bond_distance=bond_distance,
    lucj_params=LUCJParams(
        connectivity="heavy-hex",
        n_reps=n_reps,
        with_final_orbital_rotation=True,
    ),
    compressed_t2_params=CompressedT2Params(
        multi_stage_optimization=True, begin_reps=20, step=2
    ),
    shots=shots,
    samples_per_batch=samples_per_batch,
    n_batches=n_batches,
    energy_tol=energy_tol,
    occupancies_tol=occupancies_tol,
    carryover_threshold=carryover_threshold,
    max_iterations=max_iterations,
    symmetrize_spin=symmetrize_spin,
    entropy=entropy,
    max_dim=max_dim,
    n_hardware_run = 0
)

task_truncated_t2_hardware = HardwareSQDEnergyTask(
    molecule_basename=molecule_basename,
    bond_distance=bond_distance,
    lucj_params=LUCJParams(
        connectivity="heavy-hex",
        n_reps=n_reps,
        with_final_orbital_rotation=True,
    ),
    compressed_t2_params=None,
    shots=shots,
    samples_per_batch=samples_per_batch,
    n_batches=n_batches,
    energy_tol=energy_tol,
    occupancies_tol=occupancies_tol,
    carryover_threshold=carryover_threshold,
    max_iterations=max_iterations,
    symmetrize_spin=symmetrize_spin,
    entropy=entropy,
    max_dim=max_dim,
    n_hardware_run = 0
)


dim = ffsim.dim(norb, (nelectron, nelectron))
strings = ffsim.addresses_to_strings(
    range(dim),
    norb=norb,
    nelec=(nelectron, nelectron),
    bitstring_type=ffsim.BitstringType.STRING,
)


def compute_score(task):
    sample_filename = DATA_ROOT / task.operatorpath / "hardware_sample.pickle"

    state_vector_filename = DATA_ROOT / task.operatorpath / "state_vector.npy"


    with open(sample_filename, "rb") as f:
        samples = pickle.load(f)

    state_vector = np.load(state_vector_filename)

    # Convert BitArray into bitstring and probability arrays
    raw_bitstrings, raw_probs = bit_array_to_arrays(samples)

    # Run configuration recovery loop
    # If we don't have average orbital occupancy information, simply postselect
    # bitstrings with the correct numbers of spin-up and spin-down electrons
    bitstrings, probs = postselect_by_hamming_right_and_left(
        raw_bitstrings, raw_probs, hamming_right=nelectron, hamming_left=nelectron
    )
    score = 0
    count = 0 # count for bitstring that has 0 amplitude in the state vector
    for bitstr in samples:
        index = strings.indexOf(bitstr)
        if np.isclose(state_vector[index], 0):
            count += 1
        else:
            score += state_vector[index]
    return score, count


score_compressed, count_compressed = compute_score(task_compressed_t2_hardware)

score_truncated, count_truncated = compute_score(task_truncated_t2_hardware)

print(f"Compressed Op - score: {score_compressed}, count: {count_compressed}")
print(f"Truncated Op - score: {score_truncated}, count: {count_truncated}")