import os
import pickle
from pathlib import Path

import ffsim
import matplotlib.pyplot as plt
import numpy as np
import json 

from lucj.params import LUCJParams
from lucj.uccsd_task.uccsd_sqd_initial_params_task import UCCSDSQDInitialParamsTask
from lucj.sqd_energy_task.lucj_compressed_t2_task import SQDEnergyTask


DATA_ROOT = Path(os.environ.get("LUCJ_DATA_ROOT", "data"))
MOLECULES_CATALOG_DIR = Path(os.environ.get("MOLECULES_CATALOG_DIR"))


molecule_name = "n2"
basis = "6-31g"
nelectron, norb = 10, 16
molecule_basename = f"{molecule_name}_{basis}_{nelectron}e{norb}o"

plots_dir = "paper"
os.makedirs(plots_dir, exist_ok=True)

start = 0.9
stop = 2.7
step = 0.1
bond_distance_range = np.linspace(start, stop, num=round((stop - start) / step) + 1)

reference_bond_distance_range = np.linspace(
    start, stop, num=round((stop - start) / 0.05) + 1
)

shots = 100_000
n_batches = 10
energy_tol = 1e-5
occupancies_tol = 1e-3
carryover_threshold = 1e-3
max_iterations = 1
symmetrize_spin = True
# TODO set entropy and generate seeds properly
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
        mol_data.ccsd_energy - mol_data.fci_energy
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

tasks_lucj_square = [
    SQDEnergyTask(
        molecule_basename=molecule_basename,
        bond_distance=d,
        lucj_params=LUCJParams(
            connectivity="square",
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

tasks_lucj_heavy_hex = [
    SQDEnergyTask(
        molecule_basename=molecule_basename,
        bond_distance=d,
        lucj_params=LUCJParams(
            connectivity="heavy-hex",
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

with open('scripts/paper/color.json', 'r') as file:
    colors = json.load(file)
alphas = [0.5, 1.0]
linestyles = ["--", ":"]
markers = ["o", "s", "v", "D", "p", "*", "P", "X"]

list_tasks = [tasks_uccsd, tasks_ucj, tasks_lucj_square, tasks_lucj_heavy_hex]
color_keys = ["uccsd", "ucj", "lucj_full_square", "lucj_full"]
labels = ["UCCSD", "UCJ", "LUCJ:square", "LUCJ:heavy-hex"]

for plot_type in ["vqe", "sqd"]:
    fig = plt.plot(figsize=(9, 12), layout="constrained")

    plt.plot(
        reference_bond_distance_range,
        fci_energies_reference,
        "-",
        label="FCI",
        color=colors["fci"],
    )

    plt.plot(
        reference_bond_distance_range,
        ccsd_energies_reference,
        "--",
        label="CCSD",
        color=colors["ccsd"],
    )

    plt.plot(
        reference_bond_distance_range,
        cisd_energies_reference,
        "--",
        label="CISD",
        color=colors["cisd"],
    )
    
    for tasks, color_key, label in zip(list_tasks, color_keys, labels):
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
        errors = [results[task]["error"] for task in tasks]
        plt.plot(
            bond_distance_range,
            energies,
            # f"{markers[0]}{linestyles[0]}",
            linestyles[0],
            label=label,
            color=colors[color_key],
        )
        
    plt.legend()
    plt.ylabel("Energy")
    plt.ylim(-109.2, -107.7)
    plt.xlabel("Bond length (Å)")

    # plt.set_yscale("log")
    # plt.axhline(1.6e-3, linestyle="--", color="gray")
    # plt.set_ylabel("Energy error (Hartree)")
    # plt.set_xlabel("Bond length (Å)")

    if plot_type == "vqe":
        plt.title(f"{molecule_basename} ({nelectron}e, {norb}o)")
    else:
        plt.title(f"SQD with CCSD parameters, {molecule_basename} ({nelectron}e, {norb}o)")


    filepath = os.path.join(
        plots_dir, f"{molecule_basename}/{os.path.splitext(os.path.basename(__file__))[0]}_{plot_type}.pdf"
    )
    plt.savefig(filepath)
    print(f"Saved figure to {filepath}.")
    plt.close()

