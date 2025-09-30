# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import csv
from lucj.hardware_sqd_task.lucj_t2_seperate_sqd_task import HardwareSQDEnergyTask
from lucj.params import LUCJParams, CompressedT2Params
import os
import pickle
from pathlib import Path
import numpy as np

DATA_ROOT = Path(os.environ.get("LUCJ_DATA_ROOT", "data"))

data = [
    {"system": "N2.6-31g.1.2A", "circuit": "random", "average": 0, "std": 0},
    {"system": "N2.6-31g.1.2A", "circuit": "truncated", "average": 0, "std": 0},
    {"system": "N2.6-31g.1.2A", "circuit": "compressed", "average": 0, "std": 0},
    {"system": "N2.6-31g.2.4A", "circuit": "random", "average": 0, "std": 0},
    {"system": "N2.6-31g.2.4A", "circuit": "truncated", "average": 0, "std": 0},
    {"system": "N2.6-31g.2.4A", "circuit": "compressed", "average": 0, "std": 0},
    {"system": "N2.cc-PVDZ.1.2A", "circuit": "random", "average": 0, "std": 0},
    {"system": "N2.cc-PVDZ.1.2A", "circuit": "truncated", "average": 0, "std": 0},
    {"system": "N2.cc-PVDZ.1.2A", "circuit": "compressed", "average": 0, "std": 0},
    {"system": "N2.cc-PVDZ.2.4A", "circuit": "random", "average": 0, "std": 0},
    {"system": "N2.cc-PVDZ.2.4A", "circuit": "truncated", "average": 0, "std": 0},
    {"system": "N2.cc-PVDZ.2.4A", "circuit": "compressed", "average": 0, "std": 0},
    {"system": "Fe2S2", "circuit": "random", "average": 0, "std": 0},
    {"system": "Fe2S2", "circuit": "truncated", "average": 0, "std": 0},
    {"system": "Fe2S2", "circuit": "compressed", "average": 0, "std": 0},
]

bond_distance_range = [1.2, 2.4]
n_hardware_run_range = list(range(0, 10))
n_reps = 1

shots = 1_000_000
n_batches = 10
energy_tol = 1e-8
occupancies_tol = 1e-5
carryover_threshold = 1e-4
max_iterations = 1
symmetrize_spin = True
entropy = 1

max_dim = 4000
samples_per_batch = 4000

molecule_name = "n2"
basis = "6-31g"
nelectron, norb = 10, 16
molecule_basename = f"{molecule_name}_{basis}_{nelectron}e{norb}o"

for i, d in enumerate(bond_distance_range):
    tasks_compressed_t2 = [
        HardwareSQDEnergyTask(
            molecule_basename=molecule_basename,
            bond_distance=d,
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
            dynamic_decoupling=True,
            n_hardware_run=n_hardware_run,
        )
        for n_hardware_run in n_hardware_run_range
    ]

    tasks_random = [
        HardwareSQDEnergyTask(
            molecule_basename=molecule_basename,
            bond_distance=d,
            lucj_params=LUCJParams(
                connectivity="heavy-hex",
                n_reps=n_reps,
                with_final_orbital_rotation=True,
            ),
            compressed_t2_params=None,
            random_op=True,
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
            dynamic_decoupling=True,
            n_hardware_run=n_hardware_run,
        )
        for n_hardware_run in n_hardware_run_range
    ]

    tasks_truncated_t2 = [
        HardwareSQDEnergyTask(
            molecule_basename=molecule_basename,
            bond_distance=d,
            lucj_params=LUCJParams(
                connectivity="heavy-hex",
                n_reps=n_reps,
                with_final_orbital_rotation=True,
            ),
            compressed_t2_params=None,
            random_op=False,
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
            dynamic_decoupling=True,
            n_hardware_run=n_hardware_run,
        )
        for n_hardware_run in n_hardware_run_range
    ]

    list_tasks = [tasks_random, tasks_truncated_t2, tasks_compressed_t2]
    for j, tasks in enumerate(list_tasks):
        num_valid_bitstrings = []
        for task in tasks:
            filepath = DATA_ROOT / task.dirpath / "hardware_sqd_data.pickle"
            if os.path.exists(filepath):
                with open(filepath, "rb") as f:
                    result = pickle.load(f)
                num_valid_bitstrings.append(result["valid_bit_string"])
        data[i * 3 + j]["average"] = np.average(num_valid_bitstrings)
        data[i * 3 + j]["std"] = np.std(num_valid_bitstrings)


molecule_name = "n2"
basis = "cc-pvdz"
nelectron, norb = 10, 26
molecule_basename = f"{molecule_name}_{basis}_{nelectron}e{norb}o"

for i, d in enumerate(bond_distance_range):
    tasks_compressed_t2 = [
        HardwareSQDEnergyTask(
            molecule_basename=molecule_basename,
            bond_distance=d,
            lucj_params=LUCJParams(
                connectivity="heavy-hex",
                n_reps=n_reps,
                with_final_orbital_rotation=True,
            ),
            compressed_t2_params=CompressedT2Params(
                multi_stage_optimization=True, begin_reps=50, step=2
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
            dynamic_decoupling=True,
            n_hardware_run=n_hardware_run,
        )
        for n_hardware_run in n_hardware_run_range
    ]

    tasks_random = [
        HardwareSQDEnergyTask(
            molecule_basename=molecule_basename,
            bond_distance=d,
            lucj_params=LUCJParams(
                connectivity="heavy-hex",
                n_reps=n_reps,
                with_final_orbital_rotation=True,
            ),
            compressed_t2_params=None,
            random_op=True,
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
            dynamic_decoupling=True,
            n_hardware_run=n_hardware_run,
        )
        for n_hardware_run in n_hardware_run_range
    ]

    tasks_truncated_t2 = [
        HardwareSQDEnergyTask(
            molecule_basename=molecule_basename,
            bond_distance=d,
            lucj_params=LUCJParams(
                connectivity="heavy-hex",
                n_reps=n_reps,
                with_final_orbital_rotation=True,
            ),
            compressed_t2_params=None,
            random_op=False,
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
            dynamic_decoupling=True,
            n_hardware_run=n_hardware_run,
        )
        for n_hardware_run in n_hardware_run_range
    ]

    list_tasks = [tasks_random, tasks_truncated_t2, tasks_compressed_t2]
    for j, tasks in enumerate(list_tasks):
        num_valid_bitstrings = []
        for task in tasks:
            filepath = DATA_ROOT / task.dirpath / "hardware_sqd_data.pickle"
            if os.path.exists(filepath):
                with open(filepath, "rb") as f:
                    result = pickle.load(f)
                num_valid_bitstrings.append(result["valid_bit_string"])
        data[6 + i * 3 + j]["average"] = np.average(num_valid_bitstrings)
        data[6 + i * 3 + j]["std"] = np.std(num_valid_bitstrings)


molecule_name = "fe2s2"
nelectron, norb = 30, 20
molecule_basename = f"{molecule_name}_{nelectron}e{norb}o"
d = None

tasks_compressed_t2 = [
    HardwareSQDEnergyTask(
        molecule_basename=molecule_basename,
        bond_distance=d,
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
        dynamic_decoupling=True,
        n_hardware_run=n_hardware_run,
    )
    for n_hardware_run in n_hardware_run_range
]


tasks_random = [
    HardwareSQDEnergyTask(
        molecule_basename=molecule_basename,
        bond_distance=d,
        lucj_params=LUCJParams(
            connectivity="heavy-hex",
            n_reps=n_reps,
            with_final_orbital_rotation=True,
        ),
        compressed_t2_params=None,
        random_op=True,
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
        dynamic_decoupling=True,
        n_hardware_run=n_hardware_run,
    )
    for n_hardware_run in n_hardware_run_range
]

tasks_truncated_t2 = [
    HardwareSQDEnergyTask(
        molecule_basename=molecule_basename,
        bond_distance=d,
        lucj_params=LUCJParams(
            connectivity="heavy-hex",
            n_reps=n_reps,
            with_final_orbital_rotation=True,
        ),
        compressed_t2_params=None,
        random_op=False,
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
        dynamic_decoupling=True,
        n_hardware_run=n_hardware_run,
    )
    for n_hardware_run in n_hardware_run_range
]

list_tasks = [tasks_random, tasks_truncated_t2, tasks_compressed_t2]
for i, tasks in enumerate(list_tasks):
    num_valid_bitstrings = []
    for task in tasks:
        filepath = DATA_ROOT / task.dirpath / "hardware_sqd_data.pickle"
        if os.path.exists(filepath):
            with open(filepath, "rb") as f:
                result = pickle.load(f)
            num_valid_bitstrings.append(result["valid_bit_string"])
    data[12 + i]["average"] = np.average(num_valid_bitstrings)
    data[12 + i]["std"] = np.std(num_valid_bitstrings)


with open("csv/sample_bitstring.csv", "w", newline="") as csvfile:
    fieldnames = ["system", "circuit", "average", "std"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(data)
