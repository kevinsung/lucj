import itertools
import os
import pickle
from pathlib import Path

import ffsim
import matplotlib.pyplot as plt
import numpy as np

from lucj.params import LUCJParams
from lucj.tasks.lucj_aqc_mps_task import LUCJAQCMPSTask
from lucj.tasks.lucj_initial_params_task import LUCJInitialParamsTask

DATA_ROOT = Path(os.environ.get("LUCJ_DATA_ROOT", "data"))
MOLECULES_CATALOG_DIR = Path(os.environ.get("MOLECULES_CATALOG_DIR"))


molecule_name = "n2"
basis = "sto-6g"
nelectron, norb = 10, 8
molecule_basename = f"{molecule_name}_{basis}_{nelectron}e{norb}o"

plots_dir = os.path.join("plots", molecule_basename)
os.makedirs(plots_dir, exist_ok=True)

start = 0.9
stop = 2.1
step = 0.1
bond_distance_range = np.linspace(start, stop, num=round((stop - start) / step) + 1)
reference_bond_distance_range = np.linspace(
    start, stop, num=round((stop - start) / 0.05) + 1
)

connectivities = [
    "heavy-hex",
    "hex",
]
n_reps_range = [
    1,
    2,
    3,
]
tasks_lucj_ccsd = [
    LUCJInitialParamsTask(
        molecule_basename=molecule_basename,
        bond_distance=d,
        lucj_params=LUCJParams(
            connectivity="all-to-all",
            n_reps=None,
            with_final_orbital_rotation=True,
        ),
    )
    for d in bond_distance_range
]
tasks_lucj = [
    LUCJInitialParamsTask(
        molecule_basename=molecule_basename,
        bond_distance=d,
        lucj_params=LUCJParams(
            connectivity=connectivity,
            n_reps=n_reps,
            with_final_orbital_rotation=True,
        ),
    )
    for connectivity, n_reps in itertools.product(connectivities, n_reps_range)
    for d in bond_distance_range
]
tasks_aqc_mps = [
    LUCJAQCMPSTask(
        molecule_basename=molecule_basename,
        bond_distance=d,
        lucj_params=LUCJParams(
            connectivity=connectivity,
            n_reps=n_reps,
            with_final_orbital_rotation=True,
        ),
        init_params="ccsd",
        max_bond=None,
        cutoff=1e-10,
    )
    for connectivity, n_reps in itertools.product(connectivities, n_reps_range)
    for d in bond_distance_range
]


mol_datas_reference: dict[float, ffsim.MolecularData] = {}
mol_datas_experiment: dict[float, ffsim.MolecularData] = {}

for d in reference_bond_distance_range:
    filepath = os.path.join(
        MOLECULES_CATALOG_DIR,
        "data",
        "molecular_data",
        f"{molecule_basename}_d-{d:.2f}.json.xz",
    )
    mol_datas_reference[d] = ffsim.MolecularData.from_json(filepath, compression="lzma")

for d in bond_distance_range:
    filepath = os.path.join(
        MOLECULES_CATALOG_DIR,
        "data",
        "molecular_data",
        f"{molecule_basename}_d-{d:.2f}.json.xz",
    )
    mol_datas_experiment[d] = ffsim.MolecularData.from_json(
        filepath, compression="lzma"
    )

hf_energies_reference = np.array(
    [mol_data.hf_energy for mol_data in mol_datas_reference.values()]
)
fci_energies_reference = np.array(
    [mol_data.fci_energy for mol_data in mol_datas_reference.values()]
)
ccsd_energies_reference = np.array(
    [mol_data.ccsd_energy for mol_data in mol_datas_reference.values()]
)
ccsd_errors_reference = ccsd_energies_reference - fci_energies_reference

hf_energies_experiment = np.array(
    [mol_data.hf_energy for mol_data in mol_datas_experiment.values()]
)
fci_energies_experiment = np.array(
    [mol_data.fci_energy for mol_data in mol_datas_experiment.values()]
)

print("Loading data...")
data = {}
for task in tasks_lucj_ccsd:
    filepath = DATA_ROOT / "lucj_initial_params" / task.dirpath / "data.pickle"
    with open(filepath, "rb") as f:
        data[task] = pickle.load(f)
for task in tasks_lucj:
    filepath = DATA_ROOT / "lucj_initial_params" / task.dirpath / "data.pickle"
    with open(filepath, "rb") as f:
        data[task] = pickle.load(f)
for task in tasks_aqc_mps:
    filepath = DATA_ROOT / "lucj_aqc_mps" / task.dirpath / "data.pickle"
    with open(filepath, "rb") as f:
        data[task] = pickle.load(f)
print("Done loading data.")

markers = ["o", "s", "v", "D", "p", "*", "P", "X"]
prop_cycle = plt.rcParams["axes.prop_cycle"]
colors = prop_cycle.by_key()["color"]
alphas = [0.5, 1.0]
linestyles = ["--", ":"]

fig, axes = plt.subplots(3, len(connectivities), figsize=(12, 12), layout="constrained")

for i, connectivity in enumerate(connectivities):
    axes[0, i].plot(
        reference_bond_distance_range,
        ccsd_errors_reference,
        "--",
        label="CCSD",
        color="orange",
    )

    energies = [data[task]["energy"] for task in tasks_lucj_ccsd]
    errors = [data[task]["error"] for task in tasks_lucj_ccsd]
    spin_squares = [data[task]["spin_squared"] for task in tasks_lucj_ccsd]
    axes[0, i].plot(
        bond_distance_range,
        errors,
        f"{markers[0]}{linestyles[0]}",
        label="LUCJ CCSD",
        color=colors[0],
    )
    axes[2, i].plot(
        bond_distance_range,
        spin_squares,
        f"{markers[0]}{linestyles[0]}",
        label="LUCJ CCSD",
        color=colors[0],
    )

    for n_reps, marker, color in zip(n_reps_range, markers[1:], colors[1:]):
        tasks_lucj = [
            LUCJInitialParamsTask(
                molecule_basename=molecule_basename,
                bond_distance=d,
                lucj_params=LUCJParams(
                    connectivity=connectivity,
                    n_reps=n_reps,
                    with_final_orbital_rotation=True,
                ),
            )
            for d in bond_distance_range
        ]
        energies = [data[task]["energy"] for task in tasks_lucj]
        errors = [data[task]["error"] for task in tasks_lucj]
        spin_squares = [data[task]["spin_squared"] for task in tasks_lucj]
        axes[0, i].plot(
            bond_distance_range,
            errors,
            f"{marker}{linestyles[0]}",
            label=f"LUCJ CCSD L={n_reps}",
            color=color,
            alpha=0.5,
        )
        axes[2, i].plot(
            bond_distance_range,
            spin_squares,
            f"{marker}{linestyles[0]}",
            label=f"LUCJ CCSD L={n_reps}",
            color=color,
            alpha=0.5,
        )

        tasks_aqc_mps = [
            LUCJAQCMPSTask(
                molecule_basename=molecule_basename,
                bond_distance=d,
                lucj_params=LUCJParams(
                    connectivity=connectivity,
                    n_reps=n_reps,
                    with_final_orbital_rotation=True,
                ),
                init_params="ccsd",
                max_bond=None,
                cutoff=1e-10,
            )
            for d in bond_distance_range
        ]
        energies = [data[task]["energy"] for task in tasks_aqc_mps]
        errors = [data[task]["error"] for task in tasks_aqc_mps]
        spin_squares = [data[task]["spin_squared"] for task in tasks_aqc_mps]
        norm_devs = [1 - data[task]["norm"] for task in tasks_aqc_mps]
        axes[0, i].plot(
            bond_distance_range,
            errors,
            f"{marker}{linestyles[0]}",
            label=f"LUCJ AQC L={n_reps}",
            color=color,
        )
        axes[1, i].plot(
            bond_distance_range,
            norm_devs,
            f"{marker}{linestyles[0]}",
            label=f"LUCJ AQC L={n_reps}",
            color=color,
        )
        axes[2, i].plot(
            bond_distance_range,
            spin_squares,
            f"{marker}{linestyles[0]}",
            label=f"LUCJ AQC L={n_reps}",
            color=color,
        )

    axes[0, i].legend()
    axes[0, i].set_title(connectivity)
    axes[0, i].set_yscale("log")
    axes[0, i].axhline(1.6e-3, linestyle="--", color="gray")
    axes[0, i].set_ylabel("Energy error (Hartree)")
    axes[0, i].set_xlabel("Bond length (Å)")
    axes[1, i].set_yscale("log")
    axes[1, i].set_ylabel(r"1 - $|\psi|^2$")
    axes[1, i].set_xlabel("Bond length (Å)")
    axes[2, i].set_ylim(0, 0.1)
    axes[2, i].set_ylabel("Spin squared")
    axes[2, i].set_xlabel("Bond length (Å)")
    fig.suptitle(f"{molecule_basename} ({nelectron}e, {norb}o) AQC MPS")

filepath = os.path.join(
    plots_dir,
    f"{os.path.splitext(os.path.basename(__file__))[0]}.pdf",
)
plt.savefig(filepath)
plt.close()
