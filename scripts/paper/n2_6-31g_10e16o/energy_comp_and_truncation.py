import json
import os
import pickle
from pathlib import Path

import ffsim
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch

from lucj.params import LUCJParams
from lucj.sqd_energy_task.lucj_compressed_t2_task import SQDEnergyTask
from lucj.uccsd_task.uccsd_sqd_initial_params_task import UCCSDSQDInitialParamsTask

DATA_ROOT = Path(os.environ.get("LUCJ_DATA_ROOT", "data"))
MOLECULES_CATALOG_DIR = Path(os.environ.get("MOLECULES_CATALOG_DIR"))


molecule_name = "n2"
basis = "6-31g"
nelectron, norb = 10, 16
molecule_basename = f"{molecule_name}_{basis}_{nelectron}e{norb}o"

plots_dir = os.path.join("paper", molecule_basename)
os.makedirs(plots_dir, exist_ok=True)

start = 0.9
stop = 2.7
step = 0.1
bond_distance_range = np.linspace(start, stop, num=round((stop - start) / step) + 1)

reference_bond_distance_range = np.linspace(
    start, stop, num=round((stop - start) / 0.05) + 1
)

bond_distance_truncation = 1.2
n_reps_range = [1, 5] + list(range(10, 110, 10))

shots = 100_000
n_batches = 10
energy_tol = 1e-5
occupancies_tol = 1e-3
carryover_threshold = 1e-3
max_iterations = 1
symmetrize_spin = True
entropy = 0
max_dim = 4000
samples_per_batch = max_dim

mol_datas_reference: dict[float, ffsim.MolecularData] = {}
mol_datas_experiment: dict[float, ffsim.MolecularData] = {}

for d in reference_bond_distance_range:
    filepath = os.path.join(
        MOLECULES_CATALOG_DIR,
        "data",
        "molecular_data",
        f"{molecule_basename}_d-{d:.5f}.json.xz",
    )
    mol_datas_reference[d] = ffsim.MolecularData.from_json(filepath, compression="lzma")

for d in bond_distance_range:
    filepath = os.path.join(
        MOLECULES_CATALOG_DIR,
        "data",
        "molecular_data",
        f"{molecule_basename}_d-{d:.5f}.json.xz",
    )
    mol_datas_experiment[d] = ffsim.MolecularData.from_json(
        filepath, compression="lzma"
    )

fci_energies_reference = np.array(
    [mol_data.fci_energy for mol_data in mol_datas_reference.values()]
)
ccsd_energies_reference = np.array(
    [mol_data.ccsd_energy for mol_data in mol_datas_reference.values()]
)
ccsd_errors_reference = np.array(
    [
        abs(mol_data.ccsd_energy - mol_data.fci_energy)
        for mol_data in mol_datas_reference.values()
    ]
)
cisd_energies_reference = np.array(
    [mol_data.cisd_energy for mol_data in mol_datas_reference.values()]
)
cisd_errors_reference = np.array(
    [
        mol_data.cisd_energy - mol_data.fci_energy
        for mol_data in mol_datas_reference.values()
    ]
)
fci_energies_experiment = np.array(
    [mol_data.fci_energy for mol_data in mol_datas_experiment.values()]
)


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


tasks_uccsd = [
    UCCSDSQDInitialParamsTask(
        molecule_basename=molecule_basename,
        bond_distance=d,
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
]

tasks_ucj = [
    SQDEnergyTask(
        molecule_basename=molecule_basename,
        bond_distance=d,
        lucj_params=LUCJParams(
            connectivity="all-to-all",
            n_reps=None,
            with_final_orbital_rotation=True,
        ),
        random_op=False,
        connectivity_opt=False,
        compressed_t2_params=None,
        regularization=False,
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
]

tasks_ucj_truncation = [
    SQDEnergyTask(
        molecule_basename=molecule_basename,
        bond_distance=bond_distance_truncation,
        lucj_params=LUCJParams(
            connectivity="all-to-all",
            n_reps=n_reps,
            with_final_orbital_rotation=True,
        ),
        random_op=False,
        connectivity_opt=False,
        compressed_t2_params=None,
        regularization=False,
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
    for n_reps in n_reps_range + [None]
]

tasks_lucj_square_truncation = [
    SQDEnergyTask(
        molecule_basename=molecule_basename,
        bond_distance=bond_distance_truncation,
        lucj_params=LUCJParams(
            connectivity="square",
            n_reps=n_reps,
            with_final_orbital_rotation=True,
        ),
        random_op=False,
        connectivity_opt=False,
        compressed_t2_params=None,
        regularization=False,
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
    for n_reps in n_reps_range + [None]
]

tasks_lucj_heavy_hex_truncation = [
    SQDEnergyTask(
        molecule_basename=molecule_basename,
        bond_distance=bond_distance_truncation,
        lucj_params=LUCJParams(
            connectivity="heavy-hex",
            n_reps=n_reps,
            with_final_orbital_rotation=True,
        ),
        random_op=False,
        connectivity_opt=False,
        compressed_t2_params=None,
        regularization=False,
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
    for n_reps in n_reps_range + [None]
]

with open("scripts/paper/color.json", "r") as file:
    colors = json.load(file)

alphas = [0.5, 1.0]
linestyles = ["-.", ":", "--"]
markers = ["o", "s", "v", "D", "p", "*", "P", "X"]

list_tasks = [tasks_uccsd, tasks_ucj]
color_keys = ["uccsd", "ucj"]
labels = ["UCCSD", "UCJ"]

list_tasks_truncation = [
    tasks_ucj_truncation,
    tasks_lucj_square_truncation,
    tasks_lucj_heavy_hex_truncation,
]
color_keys_truncation = ["ucj", "lucj_square", "lucj_heavy_hex"]
labels_truncation = ["UCJ", "LUCJ:square", "LUCJ:heavy-hex"]

fig, axes = plt.subplots(3, 2, figsize=(12, 10))

for col_idx, plot_type in enumerate(["vqe", "qsci"]):
    axes[0, col_idx].plot(
        reference_bond_distance_range,
        fci_energies_reference,
        "-",
        label="FCI",
        color=colors["fci"],
    )

    axes[0, col_idx].plot(
        reference_bond_distance_range,
        ccsd_energies_reference,
        linestyle=(0, (5, 1)),
        label="CCSD",
        color=colors["ccsd"],
    )

    axes[0, col_idx].plot(
        reference_bond_distance_range,
        cisd_energies_reference,
        linestyle=(0, (5, 5)),
        label="CISD",
        color=colors["cisd"],
    )

    for tasks, color_key, label, linestyle in zip(
        list_tasks, color_keys, labels, linestyles
    ):
        results = {}
        for task in tasks:
            if plot_type == "vqe":
                if color_key == "uccsd":
                    filepath = DATA_ROOT / task.vqepath / "data.pickle"
                else:
                    filepath = DATA_ROOT / task.operatorpath / "data.pickle"
            else:
                filepath = DATA_ROOT / task.dirpath / "sqd_data.pickle"
            results[task] = load_data(filepath)

        energies = [results[task]["energy"] for task in tasks]
        axes[0, col_idx].plot(
            bond_distance_range,
            energies,
            linestyle,
            label=label,
            color=colors[color_key],
        )

    axes[0, col_idx].set_ylabel("Energy (Hartree)", fontsize=12)
    axes[0, col_idx].set_ylim(-109.2, -108.5)
    axes[0, col_idx].set_xlabel("Bond length (Å)", fontsize=12)
    axes[0, col_idx].set_title(f"{plot_type.upper()}", fontsize=16)
    axes[0, col_idx].margins(x=0.02)

    axes[1, col_idx].plot(
        reference_bond_distance_range,
        ccsd_errors_reference,
        linestyle=(0, (5, 1)),
        label="CCSD",
        color=colors["ccsd"],
    )

    axes[1, col_idx].plot(
        reference_bond_distance_range,
        cisd_errors_reference,
        linestyle=(0, (5, 5)),
        label="CISD",
        color=colors["cisd"],
    )

    for tasks, color_key, label, linestyle in zip(
        list_tasks, color_keys, labels, linestyles
    ):
        results = {}
        for task in tasks:
            if plot_type == "vqe":
                if color_key == "uccsd":
                    filepath = DATA_ROOT / task.vqepath / "data.pickle"
                else:
                    filepath = DATA_ROOT / task.operatorpath / "data.pickle"
            else:
                filepath = DATA_ROOT / task.dirpath / "sqd_data.pickle"
            results[task] = load_data(filepath)

        errors = [results[task]["error"] for task in tasks]
        axes[1, col_idx].plot(
            bond_distance_range,
            errors,
            linestyle,
            label=label,
            color=colors[color_key],
        )

    axes[1, col_idx].axhline(1.6e-3, linestyle="--", color=colors["chemical_precision"])
    axes[1, col_idx].set_ylabel("Energy error (Hartree)", fontsize=12)
    axes[1, col_idx].set_yscale("log")
    axes[1, col_idx].set_ylim(1e-3, 1)
    axes[1, col_idx].set_xlabel("Bond length (Å)", fontsize=12)
    axes[1, col_idx].margins(x=0.02)

    for tasks, color_key, label, marker in zip(
        list_tasks_truncation, color_keys_truncation, labels_truncation, markers
    ):
        results = {}
        for task in tasks:
            if plot_type == "vqe":
                filepath = DATA_ROOT / task.operatorpath / "data.pickle"
            else:
                filepath = DATA_ROOT / task.dirpath / "sqd_data.pickle"
            results[task] = load_data(filepath)

        errors = [results[task]["error"] for task in tasks]
        axes[2, col_idx].plot(
            n_reps_range + [110],
            errors,
            f"{marker}--",
            label=label,
            markersize=5,
            color=colors[color_key],
        )

    axes[2, col_idx].set_ylabel("Energy error (Hartree)", fontsize=12)
    axes[2, col_idx].set_xlabel("Repetitions", fontsize=12)
    axes[2, col_idx].set_yscale("log")
    axes[2, col_idx].set_ylim(1e-3, 1)
    axes[2, col_idx].margins(x=0.02)
    axes[2, col_idx].minorticks_on()

    axes[2, col_idx].text(
        0.97,
        0.05,
        f"bond length {bond_distance_truncation} Å",
        transform=axes[2, col_idx].transAxes,
        ha="right",
        va="bottom",
        fontsize=14,
        # bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )

arrow1 = FancyArrowPatch(
    (105, 2.6e-2),
    (105, 1.3e-1),
    arrowstyle="->",
    mutation_scale=10,
    color=colors["annotation"],
)
axes[2, 0].add_patch(arrow1)
axes[2, 0].text(
    84,
    5.5e-2,
    "truncate interactions",
    ha="center",
    va="center",
    fontsize=11,
    color=colors["annotation"],
)

arrow2 = FancyArrowPatch(
    (95, 1.1e-2),
    (60, 1.1e-2),
    arrowstyle="->",
    mutation_scale=10,
    color=colors["annotation"],
)
axes[2, 0].add_patch(arrow2)
axes[2, 0].text(
    77.5,
    6.5e-3,
    "truncate repetitions",
    ha="center",
    va="center",
    fontsize=11,
    color=colors["annotation"],
)

axes[1, 0].text(
    0.97,
    0.1,
    "1.6 milliHartrees",
    transform=axes[1, 0].transAxes,
    ha="right",
    va="bottom",
    fontsize=11,
    color=colors["chemical_precision"],
)

axes[0, 0].legend()
axes[2, 0].legend()

plt.tight_layout()
plt.subplots_adjust(top=0.91)

fig.suptitle(f"N$_2$ 6-31G ({nelectron}e, {norb}o)", fontsize=18)

filepath = os.path.join(
    plots_dir,
    f"{os.path.splitext(os.path.basename(__file__))[0]}.pdf",
)
plt.savefig(filepath)
print(f"Saved figure to {filepath}.")
plt.close()
