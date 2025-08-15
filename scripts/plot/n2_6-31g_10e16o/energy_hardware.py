import os
import pickle
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from lucj.params import LUCJParams, CompressedT2Params
from lucj.hardware_sqd_task.lucj_compressed_t2_task import HardwareSQDEnergyTask

import json

DATA_ROOT = Path(os.environ.get("LUCJ_DATA_ROOT", "data"))
MOLECULES_CATALOG_DIR = Path(os.environ.get("MOLECULES_CATALOG_DIR"))

molecule_name = "n2"
basis = "6-31g"
nelectron, norb = 10, 16
molecule_basename = f"{molecule_name}_{basis}_{nelectron}e{norb}o"

plots_dir = os.path.join("plots", molecule_basename)
os.makedirs(plots_dir, exist_ok=True)

bond_distance_range = [1.2, 2.4]

n_reps = 1

shots = 100_000
n_batches = 3
energy_tol = 1e-8
occupancies_tol = 1e-5
carryover_threshold = 1e-4
max_iterations = 20
symmetrize_spin = True
# TODO set entropy and generate seeds properly
entropies = list(range(1, 11))

max_dim = 1000
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
    for d in bond_distance_range
    for entropy in entropies]


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
        for d in bond_distance_range
        for entropy in entropies]

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
        for d in bond_distance_range
        for entropy in entropies]


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

    # random lucj
    
    errors = []
    errors_min = []
    errors_max = []
    sci_vec_shape = []
    sci_vec_shape_min = []
    sci_vec_shape_max = []

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
            for entropy in entropies]
    errors_n_reps = [results_random[task]['error'] for task in tasks_random]
    sci_vec_shape_n_reps = [results_random[task]["sci_vec_shape"][0] for task in tasks_random]
    errors.append(np.average(errors_n_reps))
    errors_min.append(np.average(errors_n_reps) - np.min(errors_n_reps))
    errors_max.append(np.max(errors_n_reps) - np.average(errors_n_reps))
    sci_vec_shape.append(np.average(sci_vec_shape_n_reps))
    sci_vec_shape_min.append(np.average(sci_vec_shape_n_reps) - np.min(sci_vec_shape_n_reps))
    sci_vec_shape_max.append(np.max(sci_vec_shape_n_reps) - np.average(sci_vec_shape_n_reps))

                 
    axes[row_error, i].errorbar(
        - width,
        errors,
        [errors_min, errors_max],
        color='black',
    )

    axes[row_error, i].bar(
        - width,
        errors,
        width=width,
        label="LUCJ random",
        color=colors["lucj_random"],
    )
    
    axes[row_sci_vec_dim, i].bar(
        - width,
        sci_vec_shape,
        width=width,
        label="LUCJ random",
        color=colors["lucj_random"],
    )


    axes[row_sci_vec_dim, i].errorbar(
        - width,
        sci_vec_shape,
        [sci_vec_shape_min, sci_vec_shape_max],
        color='black',
    )

    # LUCJ data
    errors = []
    errors_min = []
    errors_max = []
    sci_vec_shape = []
    sci_vec_shape_min = []
    sci_vec_shape_max = []

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
            for entropy in entropies]
    
    errors_n_reps = [results_truncated_t2[task]['error'] for task in tasks_truncated_t2]
    sci_vec_shape_n_reps = [results_truncated_t2[task]["sci_vec_shape"][0] for task in tasks_truncated_t2]
    # errors_n_reps = [results_truncated_t2[task]['error'] for task in tasks_truncated_t2]
    # sci_vec_shape_n_reps = [results_truncated_t2[task]["sci_vec_shape"][0] for task in tasks_truncated_t2]
    errors.append(np.average(errors_n_reps))
    errors_min.append(np.average(errors_n_reps) - np.min(errors_n_reps))
    errors_max.append(np.max(errors_n_reps) - np.average(errors_n_reps))
    sci_vec_shape.append(np.average(sci_vec_shape_n_reps))
    sci_vec_shape_min.append(np.average(sci_vec_shape_n_reps) - np.min(sci_vec_shape_n_reps))
    sci_vec_shape_max.append(np.max(sci_vec_shape_n_reps) - np.average(sci_vec_shape_n_reps))

    axes[row_error, i].bar(
        0,
        errors,
        width=width,
        label="LUCJ truncated",
        color=colors["lucj_truncated"],
    )
               
    axes[row_error, i].errorbar(
        0,
        errors,
        [errors_min, errors_max],
        color='black',
    )

    axes[row_sci_vec_dim, i].bar(
        0,
        sci_vec_shape,
        width=width,
        label="LUCJ truncated",
        color=colors["lucj_truncated"],
    )

    axes[row_sci_vec_dim, i].errorbar(
        0,
        sci_vec_shape,
        [sci_vec_shape_min, sci_vec_shape_max],
        color='black',
    )

    # compressed_t2
    errors = []
    errors_min = []
    errors_max = []
    sci_vec_shape = []
    sci_vec_shape_min = []
    sci_vec_shape_max = []

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
            for entropy in entropies]
    
    errors_n_reps = [results_compressed_t2[task]['error'] for task in tasks_compressed_t2]
    sci_vec_shape_n_reps = [results_compressed_t2[task]["sci_vec_shape"][0] for task in tasks_compressed_t2]
    errors.append(np.average(errors_n_reps))
    errors_min.append(np.average(errors_n_reps) - np.min(errors_n_reps))
    errors_max.append(np.max(errors_n_reps) - np.average(errors_n_reps))
    sci_vec_shape.append(np.average(sci_vec_shape_n_reps))
    sci_vec_shape_min.append(np.average(sci_vec_shape_n_reps) - np.min(sci_vec_shape_n_reps))
    sci_vec_shape_max.append(np.max(sci_vec_shape_n_reps) - np.average(sci_vec_shape_n_reps))


    axes[row_error, i].bar(
        width,
        errors,
        width=width,
        label="LUCJ compressed",
        color=colors["lucj_compressed"],
    )
           
    axes[row_error, i].errorbar(
        width,
        errors,
        [errors_min, errors_max],
        color='black',
    )

    axes[row_sci_vec_dim, i].bar(
        width,
        sci_vec_shape,
        width=width,
        label="LUCJ compressed",
        color=colors["lucj_compressed"],
    )
    axes[row_sci_vec_dim, i].errorbar(
        width,
        sci_vec_shape,
        [sci_vec_shape_min, sci_vec_shape_max],
        color='black',
    )



    axes[row_error, i].set_title(f"R: {bond_distance} Å ")
    axes[row_error, i].set_yscale("log")
    axes[row_error, i].axhline(1.6e-3, linestyle="--", color="black")
    axes[row_error, i].set_ylabel("Energy error (Hartree)")
    axes[row_error, i].set_xticks([])
    
    axes[row_sci_vec_dim, i].set_ylabel("SCI subspace")
    axes[row_sci_vec_dim, i].set_xticks([])

    # axes[row_sci_vec_dim, 0].legend(ncol=2, )
    leg = axes[row_sci_vec_dim, 1].legend(
        bbox_to_anchor=(-0.32, -0.05), loc="upper center", ncol=4, columnspacing=0.8, handletextpad=0.2
    )
    # leg = axes[row_sci_vec_dim, 1].legend(
    #     bbox_to_anchor=(0.5, -0.4), loc="upper center", ncol=3
    # )
    leg.set_in_layout(False)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1, top=0.88)

    fig.suptitle(
        f"$N_2$/6-31G ({nelectron}e, {norb}o)"
    )

filepath = os.path.join(
    plots_dir,
    f"{os.path.splitext(os.path.basename(__file__))[0]}_maxdim-{max_dim}.pdf",
)
plt.savefig(filepath)
plt.close()
