import os
import pickle
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from lucj.params import LUCJParams, CompressedT2Params
from lucj.hardware_sqd_task.lucj_t2_seperate_sqd_task_sci import HardwareSQDEnergyTask

import json

matplotlib.rcParams.update({"errorbar.capsize": 5})

DATA_ROOT = Path(os.environ.get("LUCJ_DATA_ROOT", "data"))
MOLECULES_CATALOG_DIR = Path(os.environ.get("MOLECULES_CATALOG_DIR"))

molecule_name = "fe2s2"
nelectron, norb = 30, 20
molecule_basename = f"{molecule_name}_{nelectron}e{norb}o"

plots_dir = os.path.join("paper", molecule_basename)
os.makedirs(plots_dir, exist_ok=True)


n_reps = 1

# shots = 100_000
shots = 1_000_000
n_batches = 10
energy_tol = 1e-8
occupancies_tol = 1e-5
carryover_threshold = 1e-4
max_iterations = 1
symmetrize_spin = True
# TODO set entropy and generate seeds properly
entropy = 1
n_hardware_run_range = list(range(0, 10))


max_dim = 4000
samples_per_batch = 4000
dmrg_energy = -116.6056091  # ref: https://github.com/jrm874/sqd_data_repository/blob/main/classical_reference_energies/2Fe-2S/classical_methods_energies.txt

tasks_compressed_t2 = [
    HardwareSQDEnergyTask(
        molecule_basename=molecule_basename,
        bond_distance=None,
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
        bond_distance=None,
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
        bond_distance=None,
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
    if os.path.exists(filepath):
        results_random[task] = load_data(filepath)

results_truncated_t2 = {}
for task in tasks_truncated_t2:
    filepath = DATA_ROOT / task.dirpath / "hardware_sqd_data.pickle"
    if os.path.exists(filepath):
        results_truncated_t2[task] = load_data(filepath)

results_compressed_t2 = {}
for task in tasks_compressed_t2:
    filepath = DATA_ROOT / task.dirpath / "hardware_sqd_data.pickle"
    if os.path.exists(filepath):
        results_compressed_t2[task] = load_data(filepath)

print("Done loading data.")

width = 0.05
# prop_cycle = axes[row_error].rcParams["axes.prop_cycle"]
# colors = prop_cycle.by_key()["color"]

with open("scripts/paper/color.json", "r") as file:
    colors = json.load(file)


row_error = 0
# row_loss = 2
row_sci_vec_dim = 1

fig, axes = plt.subplots(
    2,
    1,
    figsize=(5, 6),  # , layout="constrained"
)


# random lucj

errors = []
errors_min = []
errors_max = []


errors_n_reps = [
    results_random[task]["energy"] - dmrg_energy
    for task in tasks_random
    if task in results_random
]
errors.append(np.average(errors_n_reps))
errors_min.append(np.average(errors_n_reps) - np.min(errors_n_reps))
errors_max.append(np.max(errors_n_reps) - np.average(errors_n_reps))
sci_vec_shape_random = [
    results_random[task]["sci_vec_shape"][0]
    for task in tasks_random
    if task in results_random
]
sci_vec_shape = np.average(sci_vec_shape_random)
sci_vec_shape_min = [sci_vec_shape - np.min(sci_vec_shape_random)]
sci_vec_shape_max = [np.max(sci_vec_shape_random) - sci_vec_shape]


axes[row_error].errorbar(
    -width,
    errors,
    [errors_min, errors_max],
    color="black",
)

axes[row_error].bar(
    -width,
    errors,
    width=width,
    label="LUCJ-random",
    color=colors["lucj_random"],
)

axes[row_sci_vec_dim].bar(
    -width,
    sci_vec_shape,
    width=width,
    label="LUCJ random",
    color=colors["lucj_random"],
)

axes[row_sci_vec_dim].errorbar(
    -width,
    sci_vec_shape,
    [sci_vec_shape_min, sci_vec_shape_max],
    color="black",
)
# LUCJ data
errors = []
errors_min = []
errors_max = []

errors_n_reps = [
    results_truncated_t2[task]["energy"] - dmrg_energy
    for task in tasks_truncated_t2
    if task in results_truncated_t2
]
errors.append(np.average(errors_n_reps))
errors_min.append(np.average(errors_n_reps) - np.min(errors_n_reps))
errors_max.append(np.max(errors_n_reps) - np.average(errors_n_reps))
sci_vec_shape_truncated = [
    results_truncated_t2[task]["sci_vec_shape"][0]
    for task in tasks_truncated_t2
    if task in results_truncated_t2
]
sci_vec_shape = np.average(sci_vec_shape_truncated)
sci_vec_shape_min = [sci_vec_shape - np.min(sci_vec_shape_truncated)]
sci_vec_shape_max = [np.max(sci_vec_shape_truncated) - sci_vec_shape]


axes[row_error].bar(
    0,
    errors,
    width=width,
    label="LUCJ-truncated",
    color=colors["lucj_truncated"],
)

axes[row_error].errorbar(
    0,
    errors,
    [errors_min, errors_max],
    color="black",
)


axes[row_sci_vec_dim].bar(
    0,
    sci_vec_shape,
    width=width,
    label="LUCJ truncated",
    color=colors["lucj_truncated"],
)

axes[row_sci_vec_dim].errorbar(
    0,
    sci_vec_shape,
    [sci_vec_shape_min, sci_vec_shape_max],
    color="black",
)

# compressed_t2
errors = []
errors_min = []
errors_max = []

errors_n_reps = [
    results_compressed_t2[task]["energy"] - dmrg_energy
    for task in tasks_compressed_t2
    if task in results_compressed_t2
]
errors.append(np.average(errors_n_reps))
errors_min.append(np.average(errors_n_reps) - np.min(errors_n_reps))
errors_max.append(np.max(errors_n_reps) - np.average(errors_n_reps))
sci_vec_shape_compressed = [
    results_compressed_t2[task]["sci_vec_shape"][0]
    for task in tasks_compressed_t2
    if task in results_compressed_t2
]
sci_vec_shape = np.average(sci_vec_shape_compressed)
sci_vec_shape_min = [sci_vec_shape - np.min(sci_vec_shape_compressed)]
sci_vec_shape_max = [np.max(sci_vec_shape_compressed) - sci_vec_shape]


axes[row_error].bar(
    width,
    errors,
    width=width,
    label="LUCJ-compressed",
    color=colors["lucj_compressed"],
)

axes[row_error].errorbar(
    width,
    errors,
    [errors_min, errors_max],
    color="black",
)

axes[row_sci_vec_dim].bar(
    width,
    sci_vec_shape,
    width=width,
    label="LUCJ compressed",
    color=colors["lucj_compressed"],
)

axes[row_sci_vec_dim].errorbar(
    width,
    sci_vec_shape,
    [sci_vec_shape_min, sci_vec_shape_max],
    color="black",
)

fig.suptitle(f"Fe$_2$S$_2$ ({nelectron}e, {norb}o)")
axes[row_error].set_yscale("log")
axes[row_error].axhline(1.6e-3, linestyle="--", color="black")
axes[row_error].set_ylabel("Energy error (Hartree)")
axes[row_error].set_xticks([])
axes[row_error].set_xlim(- 2 * width, 2*width)

axes[row_sci_vec_dim].set_ylabel("SCI subspace")
axes[row_sci_vec_dim].set_xticks([])

# axes[row_sci_vec_dim, 0].legend(ncol=2, )
leg = axes[row_sci_vec_dim].legend(
    bbox_to_anchor=(0.5, -0.05),
    loc="upper center",
    ncol=3,
    columnspacing=0.8,
    handletextpad=0.2,
)
# leg = axes[row_sci_vec_dim, 1].legend(
#     bbox_to_anchor=(0.5, -0.4), loc="upper center", ncol=3
# )
# leg.set_in_layout(False)
plt.tight_layout()
plt.subplots_adjust(left=0.15, right=0.85)

filepath = os.path.join(
    plots_dir,
    f"{os.path.splitext(os.path.basename(__file__))[0]}_maxdim-{max_dim}_shot-{shots}.pdf",
)
plt.savefig(filepath)
plt.close()
