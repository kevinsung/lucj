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
stop = 1.8
step = 0.1
# bond_distance_range = np.linspace(start, stop, num=round((stop - start) / step) + 1)
# bond_distance_range = [0.9, 1.2, 1.5, 1.8]
# reference_bond_distance_range = np.linspace(
#     start, stop, num=round((stop - start) / 0.05) + 1
# )

bond_distance = 1.2
n_reps_range = [1, 2, 4, 6]
max_bonds = [10, 20, 30, 40, 50]
cutoff = 1e-10


for connectivity in ["square", "heavy-hex"]:
    task_lucj_ccsd = LUCJInitialParamsTask(
        molecule_basename=molecule_basename,
        bond_distance=bond_distance,
        lucj_params=LUCJParams(
            connectivity="all-to-all",
            n_reps=None,
            with_final_orbital_rotation=True,
        ),
    )
    tasks_lucj = [
        LUCJInitialParamsTask(
            molecule_basename=molecule_basename,
            bond_distance=bond_distance,
            lucj_params=LUCJParams(
                connectivity=connectivity,
                n_reps=n_reps,
                with_final_orbital_rotation=True,
            ),
        )
        for n_reps in n_reps_range
    ]
    tasks_aqc_mps = [
        LUCJAQCMPSTask(
            molecule_basename=molecule_basename,
            bond_distance=bond_distance,
            lucj_params=LUCJParams(
                connectivity=connectivity,
                n_reps=n_reps,
                with_final_orbital_rotation=True,
            ),
            init_params="ccsd",
            max_bond=max_bond,
            cutoff=cutoff,
        )
        for n_reps in n_reps_range
        for max_bond in max_bonds
    ]

    mol_datas_reference: dict[float, ffsim.MolecularData] = {}
    mol_datas_experiment: dict[float, ffsim.MolecularData] = {}

    filepath = os.path.join(
        MOLECULES_CATALOG_DIR,
        "data",
        "molecular_data",
        f"{molecule_basename}_d-{bond_distance:.5f}.json.xz",
    )
    mol_data = ffsim.MolecularData.from_json(filepath, compression="lzma")
    ccsd_error = mol_data.ccsd_energy - mol_data.fci_energy

    hf_energies_experiment = np.array(
        [mol_data.hf_energy for mol_data in mol_datas_experiment.values()]
    )
    fci_energies_experiment = np.array(
        [mol_data.fci_energy for mol_data in mol_datas_experiment.values()]
    )

    print("Loading data...")
    data = {}
    task = task_lucj_ccsd
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

    fig, axes = plt.subplots(
        3, len(n_reps_range), figsize=(5 * len(n_reps_range), 12), layout="constrained"
    )
    for i, n_reps in enumerate(n_reps_range):
        axes[0, i].axhline(ccsd_error, linestyle="--", color="orange", label="CCSD")
        axes[0, i].axhline(
            data[task_lucj_ccsd]["error"],
            linestyle="--",
            color="blue",
            label="LUCJ CCSD",
        )
        axes[2, i].axhline(
            data[task_lucj_ccsd]["spin_squared"],
            linestyle="--",
            color="blue",
            label="LUCJ CCSD",
        )

        task_lucj = LUCJInitialParamsTask(
            molecule_basename=molecule_basename,
            bond_distance=bond_distance,
            lucj_params=LUCJParams(
                connectivity=connectivity,
                n_reps=n_reps,
                with_final_orbital_rotation=True,
            ),
        )
        axes[0, i].axhline(
            data[task_lucj]["error"],
            linestyle="--",
            color="red",
            label=f"LUCJ CCSD L={n_reps}",
        )
        axes[2, i].axhline(
            data[task_lucj]["spin_squared"],
            linestyle="--",
            color="red",
            label=f"LUCJ CCSD L={n_reps}",
        )

        tasks_aqc_mps = [
            LUCJAQCMPSTask(
                molecule_basename=molecule_basename,
                bond_distance=bond_distance,
                lucj_params=LUCJParams(
                    connectivity=connectivity,
                    n_reps=n_reps,
                    with_final_orbital_rotation=True,
                ),
                init_params="ccsd",
                max_bond=max_bond,
                cutoff=cutoff,
            )
            for max_bond in max_bonds
        ]
        energies = [data[task]["energy"] for task in tasks_aqc_mps]
        errors = [data[task]["error"] for task in tasks_aqc_mps]
        spin_squares = [data[task]["spin_squared"] for task in tasks_aqc_mps]
        optimize_times = [data[task]["optimize_time"] for task in tasks_aqc_mps]
        axes[0, i].plot(
            max_bonds,
            errors,
            f"{markers[0]}{linestyles[0]}",
            label="LUCJ AQC",
        )
        axes[1, i].plot(
            max_bonds,
            optimize_times,
            f"{markers[0]}{linestyles[0]}",
            label="LUCJ AQC",
        )
        axes[2, i].plot(
            max_bonds,
            spin_squares,
            f"{markers[0]}{linestyles[0]}",
            label="LUCJ AQC",
        )

        axes[0, i].legend()
        axes[0, i].set_title(f"L= {n_reps}")
        axes[0, i].set_yscale("log")
        axes[0, i].axhline(1.6e-3, linestyle="--", color="gray")
        axes[0, i].set_ylabel("Energy error (Hartree)")
        axes[0, i].set_xlabel("Max bond dimension")
        # axes[1, i].set_yscale("log")
        axes[1, i].set_ylabel("Optimization time (s)")
        axes[1, i].set_xlabel("Max bond dimension")
        axes[2, i].set_ylim(0, 0.1)
        axes[2, i].set_ylabel("Spin squared")
        axes[2, i].set_xlabel("Max bond dimension")
        fig.suptitle(
            f"{molecule_basename} ({nelectron}e, {norb}o) AQC MPS {connectivity}"
        )

    filepath = os.path.join(
        plots_dir,
        f"{os.path.splitext(os.path.basename(__file__))[0]}_d-{bond_distance}_{connectivity}.pdf",
    )
    plt.savefig(filepath)
    print(f"Saved figure to {filepath}.")
    plt.close()
