# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import itertools
import os
import pickle
from pathlib import Path
from qiskit.primitives import BitArray
import ffsim
from lucj.params import LUCJParams, CompressedT2Params

from qiskit_addon_sqd.counts import bit_array_to_arrays, bitstring_matrix_to_integers
from qiskit_addon_sqd.subsampling import postselect_by_hamming_right_and_left

import numpy as np

fractional_gate = True
if fractional_gate:
    from lucj.hardware_sqd_task.lucj_t2_seperate_sqd_task_sci_fg import HardwareSQDEnergyTask
else:
    from lucj.hardware_sqd_task.lucj_compressed_t2_task import HardwareSQDEnergyTask

DATA_ROOT = "/media/storage/WanHsuan.Lin/"
MOLECULES_CATALOG_DIR = Path(os.environ.get("MOLECULES_CATALOG_DIR"))

molecule_name = "n2"
basis = "6-31g"
nelectron, norb = 10, 16
molecule_basename = f"{molecule_name}_{basis}_{nelectron}e{norb}o"
bond_distance = 1.2
# bond_distance = 2.4
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
n_hardware_run = 0
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
    n_hardware_run = n_hardware_run,
    dynamic_decoupling=True
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
    n_hardware_run = n_hardware_run,
    dynamic_decoupling=True
)

nelec = (nelectron // 2, nelectron // 2)
dim = ffsim.dim(norb, nelec)

half_hf_state = "0" * (norb - nelectron // 2) + "1" * (nelectron // 2)
hf_state = half_hf_state + half_hf_state
hf_address = ffsim.strings_to_addresses(
        [hf_state],
        norb,
        nelec,
    )[0]

def compute_score(task):
    if fractional_gate:
        sample_filename = DATA_ROOT / task.operatorpath / f"dynamic_decoupling_xy_opt_0_fractional_gate/hardware_sample_{task.n_hardware_run}.pickle"
    else:
        sample_filename = DATA_ROOT / task.operatorpath / f"dynamic_decoupling_xy_opt_0/hardware_sample_{task.n_hardware_run}.pickle"

    state_vector_filename = DATA_ROOT / task.operatorpath / "state_vector.npy"


    with open(sample_filename, "rb") as f:
        samples = pickle.load(f)

    state_vector = np.load(state_vector_filename)
    print(state_vector.shape)
    # Convert BitArray into bitstring and probability arrays
    raw_bitstrings, raw_probs = bit_array_to_arrays(samples)

    # Run configuration recovery loop
    # If we don't have average orbital occupancy information, simply postselect
    # bitstrings with the correct numbers of spin-up and spin-down electrons
    bitstrings, probs = postselect_by_hamming_right_and_left(
        raw_bitstrings, raw_probs, hamming_right=nelectron // 2, hamming_left=nelectron // 2
    )
    score = 0
    list_score = []
    count = 0 # count for bitstring that has 0 amplitude in the state vector
    print(bitstrings.shape)
    converted_bitstrs = []
    for bitstr in bitstrings:
        converted_bitstr = ''
        for bit in bitstr:
            if bit:
                converted_bitstr = converted_bitstr + '1'
            else:
                converted_bitstr = converted_bitstr + '0'
        converted_bitstrs.append(converted_bitstr)
    addresses = ffsim.strings_to_addresses(
        converted_bitstrs,
        norb,
        nelec,
    )
    for address in addresses:
        if address == hf_address:
            continue
        if np.isclose(state_vector[address], 0, atol = 1e-15):
            count += 1
        else:
            score += (abs(state_vector[address]) ** 2)
            list_score.append(abs(state_vector[address]))
    print(min(list_score))
    return score, count

score_compressed, count_compressed = compute_score(task_compressed_t2_hardware)

score_truncated, count_truncated = compute_score(task_truncated_t2_hardware)

print(f"Compressed Op - score: {score_compressed}, #bitstr with 0 amp: {count_compressed}")
print(f"Truncated Op - score: {score_truncated}, #bitstr with 0 amp: {count_truncated}")

# n2 6-31g
# with default atol
# R=1.2
# (19079424,)
# (35952, 32)
# (19079424,)
# (24266, 32)
# Compressed Op - score: 0.9944673870470951, #bitstr with 0 amp: 1800
# Truncated Op - score: 0.9997713301041902, #bitstr with 0 amp: 22230

# R=2.4
# (19079424,)
# (40638, 32)
# (19079424,)
# (25866, 32)
# Compressed Op - score: 0.9945788605361614, #bitstr with 0 amp: 3276
# Truncated Op - score: 0.9975553910834221, #bitstr with 0 amp: 22340

# with atol=1e-15
# (19079424,)
# (40638, 32)
# (19079424,)
# (25866, 32)
# Compressed Op - score: 0.9945788605361986, #bitstr with 0 amp: 0
# Truncated Op - score: 0.9975553910834319, #bitstr with 0 amp: 17283