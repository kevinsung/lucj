import os
import pickle
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from lucj.params import LUCJParams, CompressedT2Params
from lucj.hardware_sqd_task.lucj_compressed_t2_task import HardwareSQDEnergyTask
from lucj.sqd_energy_task.lucj_random_t2_task import RandomSQDEnergyTask

import json

DATA_ROOT = Path(os.environ.get("LUCJ_DATA_ROOT", "data"))
MOLECULES_CATALOG_DIR = Path(os.environ.get("MOLECULES_CATALOG_DIR"))

molecule_name = "n2"
basis = "6-31g"
nelectron, norb = 10, 16
molecule_basename = f"{molecule_name}_{basis}_{nelectron}e{norb}o"

plots_dir = os.path.join("paper", molecule_basename)
os.makedirs(plots_dir, exist_ok=True)

bond_distance_range = [1.2, 2.4]

n_reps_range = [1]

shots = 100_000
n_batches = 3
energy_tol = 1e-5
occupancies_tol = 1e-3
carryover_threshold = 1e-3
max_iterations = 100
symmetrize_spin = True
# TODO set entropy and generate seeds properly
entropy = 0
max_dim = 4000
samples_per_batch = max_dim

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
            multi_stage_optimization=True,
            begin_reps=20,
            step=2
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
    )
    for n_reps in n_reps_range
    for d in bond_distance_range]


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
            connectivity_opt=False,
            random_op =True,
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
        )
        for n_reps in n_reps_range
        for d in bond_distance_range]

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
            connectivity_opt=False,
            random_op =False,
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
        )
        for n_reps in n_reps_range
        for d in bond_distance_range]


tasks_random_bit_string = [
    RandomSQDEnergyTask(
        molecule_basename=molecule_basename,
        bond_distance=d,
        shots=shots,
        samples_per_batch=samples_per_batch,
        n_batches=n_batches,
        energy_tol=energy_tol,
        valid_string_only=True,
        occupancies_tol=occupancies_tol,
        carryover_threshold=carryover_threshold,
        max_iterations=max_iterations,
        symmetrize_spin=symmetrize_spin,
        entropy=entropy,
        max_dim=max_dim,
    )
    for d in bond_distance_range
]

def load_data(filepath):
    if not os.path.exists(filepath):
        result = {
            "energy": 0,
            "error": 0,
            "spin_squared": 0,
            "sci_vec_shape": (0, 0),
            "n_reps": 0,
        }
    else:
        with open(filepath, "rb") as f:
            result = pickle.load(f)
    return result

print("Loading data")

results_random = {}
for task in tasks_random:
    filepath = DATA_ROOT / task.dirpath / "hardware_sqd_data.pickle"
    results_random[task] = load_data(filepath)

results_truncated_t2 = {}
for task in tasks_truncated_t2:
    filepath = DATA_ROOT / task.dirpath / "hardware_sqd_data.pickle"
    results_truncated_t2[task] = load_data(filepath)

results_compressed_t2 = {}
for task in tasks_compressed_t2:
    filepath = DATA_ROOT / task.dirpath / "hardware_sqd_data.pickle"
    results_compressed_t2[task] = load_data(filepath)

results_random_bit_string = {}
for task in tasks_random_bit_string:
    filepath = DATA_ROOT / task.dirpath / "sqd_data.pickle"
    results_random_bit_string[task] = load_data(filepath)
    # print(filepath)
    # print(results_random_bit_string[task])
    # input()


print("Done loading data.")


width = 0.15
# prop_cycle = plt.rcParams["axes.prop_cycle"]
# colors = prop_cycle.by_key()["color"]
alphas = [0.5, 1.0]

with open('scripts/paper/color.json', 'r') as file:
    colors = json.load(file)


row_error = 0
# row_loss = 2
row_sci_vec_dim = 1

fig, axes = plt.subplots(
    2,
    len(bond_distance_range),
    figsize=(6, 5),  # , layout="constrained"
)

for i, bond_distance in enumerate(bond_distance_range):
    # random bitstring

    tasks_random_bit_string = [
        RandomSQDEnergyTask(
                molecule_basename=molecule_basename,
                bond_distance=bond_distance,
                shots=shots,
                samples_per_batch=samples_per_batch,
                n_batches=n_batches,
                energy_tol=energy_tol,
                valid_string_only=True,
                occupancies_tol=occupancies_tol,
                carryover_threshold=carryover_threshold,
                max_iterations=max_iterations,
                symmetrize_spin=symmetrize_spin,
                entropy=entropy,
                max_dim=max_dim,
            )
    ]

    errors = [results_random_bit_string[task]['error'] for task in tasks_random_bit_string]
    sci_vec_shape = [results_random_bit_string[task]['sci_vec_shape'][0] for task in tasks_random_bit_string]
    axes[row_error, i].bar(
        -1.5 * width,
        errors,
        width=width,
        label="Rand bitstr",
        color=colors["random_bit_string"],
    )
    
    axes[row_sci_vec_dim, i].bar(
        -1.5 * width,
        sci_vec_shape,
        width=width,
        label="Rand bitstr",
        color=colors["random_bit_string"],
    )

    # random lucj
    tasks_random = [
        HardwareSQDEnergyTask(
                molecule_basename=molecule_basename,
                bond_distance=bond_distance,
                lucj_params=LUCJParams(
                    connectivity="heavy-hex",
                    n_reps=n_reps,
                    with_final_orbital_rotation=True,
                ),
                compressed_t2_params=None,
                connectivity_opt=False,
                random_op =True,
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
            )
            for n_reps in n_reps_range]

    errors = [results_random[task]['error'] for task in tasks_random]
    sci_vec_shape = [results_random[task]["sci_vec_shape"][0] for task in tasks_random]

    axes[row_error, i].bar(
        - 0.5 * width,
        errors,
        width=width,
        label="LUCJ random",
        color=colors["lucj_random"],
    )
    
    axes[row_sci_vec_dim, i].bar(
        - 0.5 *  width,
        sci_vec_shape,
        width=width,
        label="LUCJ random",
        color=colors["lucj_random"],
    )

    # LUCJ data
    tasks_truncated_t2 = [
        HardwareSQDEnergyTask(
                molecule_basename=molecule_basename,
                bond_distance=bond_distance,
                lucj_params=LUCJParams(
                    connectivity="heavy-hex",
                    n_reps=n_reps,
                    with_final_orbital_rotation=True,
                ),
                compressed_t2_params=None,
                connectivity_opt=False,
                random_op =False,
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
            )
            for n_reps in n_reps_range]
    
    errors = [results_truncated_t2[task]["error"] for task in tasks_truncated_t2]
    sci_vec_shape = [ results_truncated_t2[task]["sci_vec_shape"][0] for task in tasks_truncated_t2]

    axes[row_error, i].bar(
        0.5 * width,
        errors,
        width=width,
        label="LUCJ truncated",
        color=colors["lucj_truncated"],
    )
    axes[row_sci_vec_dim, i].bar(
        0.5 * width,
        sci_vec_shape,
        width=width,
        label="LUCJ truncated",
        color=colors["lucj_truncated"],
    )


    # compressed_t2
    tasks_compressed_t2 = [
        HardwareSQDEnergyTask(
                molecule_basename=molecule_basename,
                bond_distance=bond_distance,
                lucj_params=LUCJParams(
                connectivity="heavy-hex",
                    n_reps=n_reps,
                    with_final_orbital_rotation=True,
                ),
                compressed_t2_params=CompressedT2Params(
                    multi_stage_optimization=True,
                    begin_reps=20,
                    step=2
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
            )
            for n_reps in n_reps_range]
    
    errors = [results_compressed_t2[task]["error"] for task in tasks_compressed_t2]
    sci_vec_shape = [results_compressed_t2[task]["sci_vec_shape"][0] for task in tasks_compressed_t2]

    axes[row_error, i].bar(
        1.5 * width,
        errors,
        width=width,
        label="LUCJ compressed",
        color=colors["lucj_compressed"],
    )
    axes[row_sci_vec_dim, i].bar(
        1.5 * width,
        sci_vec_shape,
        width=width,
        label="LUCJ compressed",
        color=colors["lucj_compressed"],
    )

    axes[row_error, i].set_title(f"R: {bond_distance} Å ")
    axes[row_error, i].set_yscale("log")
    axes[row_error, i].axhline(1.6e-3, linestyle="--", color="gray")
    axes[row_error, i].set_ylabel("Energy error (Hartree)")
    axes[row_error, i].set_xticks([])
    
    axes[row_sci_vec_dim, i].set_ylabel("SCI subspace")
    axes[row_sci_vec_dim, i].set_xticks([])

    # axes[row_sci_vec_dim, 0].legend(ncol=2, )
    leg = axes[row_sci_vec_dim, 1].legend(
        bbox_to_anchor=(-0.48, -0.05), loc="upper center", ncol=4, columnspacing=0.8, handletextpad=0.2
    )
    # leg = axes[row_sci_vec_dim, 1].legend(
    #     bbox_to_anchor=(0.5, -0.4), loc="upper center", ncol=3
    # )
    leg.set_in_layout(False)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)

    fig.suptitle(
        f"CCSD initial parameters {molecule_name} {basis} ({nelectron}e, {norb}o)"
    )

filepath = os.path.join(
    plots_dir,
    f"{os.path.splitext(os.path.basename(__file__))[0]}_maxdim-{max_dim}.pdf",
)
plt.savefig(filepath)
plt.close()
