import os
import pickle
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import json
from lucj.params import LUCJParams, CompressedT2Params
from lucj.hardware_sqd_task.lucj_compressed_t2_task import HardwareSQDEnergyTask
from lucj.sqd_energy_task.lucj_random_t2_task import RandomSQDEnergyTask

DATA_ROOT = Path(os.environ.get("LUCJ_DATA_ROOT", "data"))
MOLECULES_CATALOG_DIR = Path(os.environ.get("MOLECULES_CATALOG_DIR"))

molecule_name = "fe2s2"
nelectron, norb = 30, 20
molecule_basename = f"{molecule_name}_{nelectron}e{norb}o"

plots_dir = os.path.join("paper", molecule_basename)
os.makedirs(plots_dir, exist_ok=True)

shots = 100_000
n_batches = 3
energy_tol = 1e-8
occupancies_tol = 1e-5
carryover_threshold = 1e-4
max_iterations = 20
symmetrize_spin = True
# TODO set entropy and generate seeds properly
entropies = list(range(1, 11))
samples_per_batch = 2000
max_dim = samples_per_batch
dmrg_energy = -116.6056091 #ref: https://github.com/jrm874/sqd_data_repository/blob/main/classical_reference_energies/2Fe-2S/classical_methods_energies.txt

tasks_compressed_t2 = [HardwareSQDEnergyTask(
        molecule_basename=molecule_basename,
        bond_distance=None,
        lucj_params=LUCJParams(
            connectivity="heavy-hex",
            n_reps=1,
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
    for entropy in entropies
]


tasks_random = [HardwareSQDEnergyTask(
            molecule_basename=molecule_basename,
            bond_distance=None,
            lucj_params=LUCJParams(
                connectivity="heavy-hex",
                n_reps=1,
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
    for entropy in entropies ]

tasks_truncated_t2 = [HardwareSQDEnergyTask(
            molecule_basename=molecule_basename,
            bond_distance=None,
            lucj_params=LUCJParams(
                connectivity="heavy-hex",
                n_reps=1,
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

energies_random = []
sci_vec_shape_random = []
for task in tasks_random:
    filepath = DATA_ROOT / task.dirpath / "hardware_sqd_data.pickle"
    result = load_data(filepath)
    if result['energy'] < 0:
        energies_random.append(result['energy'])
        sci_vec_shape_random.append(result['sci_vec_shape'][0])

energies_truncated = []
sci_vec_shape_truncated = []
for task in tasks_truncated_t2:
    filepath = DATA_ROOT / task.dirpath / "hardware_sqd_data.pickle"
    result = load_data(filepath)
    if result['energy'] < 0:
        energies_truncated.append(result['energy'])
        sci_vec_shape_truncated.append(result['sci_vec_shape'][0])

energies_compressed = []
sci_vec_shape_compressed = []
for task in tasks_compressed_t2:
    filepath = DATA_ROOT / task.dirpath / "hardware_sqd_data.pickle"
    result = load_data(filepath)
    if result['energy'] < 0:    
        energies_compressed.append(result['energy'])
        sci_vec_shape_compressed.append(result['sci_vec_shape'][0])

print("Done loading data.")

width = 0.15
# prop_cycle = plt.rcParams["axes.prop_cycle"]
# colors = prop_cycle.by_key()["color"]
alphas = [0.5, 1.0]

with open('scripts/paper/color.json', 'r') as file:
    colors = json.load(file)

row_error = 0
row_sci_vec_dim = 1

fig, axes = plt.subplots(
    2,
    1,
    figsize=(6, 6),  # , layout="constrained"
)

# random lucj
errors = np.average(energies_random) - dmrg_energy 
errors_min = [np.average(energies_random) - np.min(energies_random)]
errors_max = [np.max(energies_random) -  np.average(energies_random)]
sci_vec_shape = np.average(sci_vec_shape_random)
sci_vec_shape_min = [sci_vec_shape - np.min(sci_vec_shape_random)]
sci_vec_shape_max = [np.max(sci_vec_shape_random) - sci_vec_shape]

axes[row_error].bar(
    - width,
    errors,
    width=width,
    label="LUCJ random",
    color=colors["lucj_random"],
)                 
axes[row_error].errorbar(
    - width,
    errors,
    [errors_min, errors_max],
    color='black',
)

axes[row_sci_vec_dim].bar(
    - width,
    sci_vec_shape,
    width=width,
    label="LUCJ random",
    color=colors["lucj_random"],
)

axes[row_sci_vec_dim].errorbar(
    - width,
    sci_vec_shape,
    [sci_vec_shape_min, sci_vec_shape_max],
    color='black',
)

# LUCJ data
# print(energies_truncated)
errors = np.average(energies_truncated) - dmrg_energy 
errors_min = [np.average(energies_truncated) - np.min(energies_truncated)]
errors_max = [np.max(energies_truncated) - np.average(energies_truncated)]
sci_vec_shape = np.average(sci_vec_shape_truncated)
sci_vec_shape_min = [sci_vec_shape - np.min(sci_vec_shape_truncated)]
sci_vec_shape_max = [np.max(sci_vec_shape_truncated) - sci_vec_shape]


axes[row_error].bar(
    0,
    errors,
    width=width,
    label="LUCJ truncated",
    color=colors["lucj_truncated"],
)

axes[row_error].errorbar(
    0,
    errors,
    [errors_min, errors_max],
    color='black',
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
    color='black',
)


# compressed_t2
errors = np.average(energies_compressed) - dmrg_energy 
errors_min = [np.average(energies_compressed) - np.min(energies_compressed)]
errors_max = [np.max(energies_compressed) - np.average(energies_compressed)]
sci_vec_shape = np.average(sci_vec_shape_compressed)
sci_vec_shape_min = [sci_vec_shape - np.min(sci_vec_shape_compressed)]
sci_vec_shape_max = [np.max(sci_vec_shape_compressed) - sci_vec_shape]

axes[row_error].bar(
    width,
    errors,
    width=width,
    label="LUCJ compressed",
    color=colors["lucj_compressed"],
)

axes[row_error].errorbar(
    width,
    errors,
    [errors_min, errors_max],
    color='black',
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
    color='black',
)

axes[row_error].set_yscale("log")
axes[row_error].axhline(1.6e-3, linestyle="--", color="gray")
axes[row_error].set_ylabel("Energy error (Hartree)")
axes[row_error].set_xticks([])

axes[row_sci_vec_dim].set_ylabel("SCI subspace")
axes[row_sci_vec_dim].set_xticks([])

leg = axes[row_sci_vec_dim].legend(
    bbox_to_anchor=(0.5, -0.05), loc="upper center", ncol=3
)
leg.set_in_layout(False)
plt.tight_layout()
plt.subplots_adjust(top=0.94,bottom=0.1)

fig.suptitle(
    f"CCSD initial parameters {molecule_name} ({nelectron}e, {norb}o)"
)

filepath = os.path.join(
    plots_dir,
    f"{os.path.splitext(os.path.basename(__file__))[0]}.pdf",
)
plt.savefig(filepath)
plt.close()
