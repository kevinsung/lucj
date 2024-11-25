import os
import pickle
from pathlib import Path

import ffsim
import matplotlib.pyplot as plt

from lucj.params import LUCJParams
from lucj.tasks.lucj_initial_params_task import LUCJInitialParamsTask
from lucj.tasks.lucj_mps_tenpy_task import LUCJMPSTenpyTask

DATA_ROOT = Path(os.environ.get("LUCJ_DATA_ROOT", "data"))
MOLECULES_CATALOG_DIR = Path(os.environ.get("MOLECULES_CATALOG_DIR"))


molecule_name = "n2"
basis = "sto-6g"
nelectron, norb = 10, 8
molecule_basename = f"{molecule_name}_{basis}_{nelectron}e{norb}o"

plots_dir = os.path.join("plots", molecule_basename)
os.makedirs(plots_dir, exist_ok=True)

molecule_name = "n2"
basis = "sto-6g"
nelectron, norb = 10, 8
molecule_basename = f"{molecule_name}_{basis}_{nelectron}e{norb}o"

bond_distance_range = [1.0, 2.0]

connectivity = "square"
n_reps_range = list(range(2, 25, 2)) + [None]
chi_max_range = [100, 250, 500, 1000, 2500, 5000]
svd_min_range = [1e-8]

tasks = [
    LUCJMPSTenpyTask(
        molecule_basename=molecule_basename,
        bond_distance=bond_distance,
        lucj_params=LUCJParams(
            connectivity=connectivity,
            n_reps=n_reps,
            with_final_orbital_rotation=True,
        ),
        chi_max=chi_max,
        svd_min=svd_min,
        params="ccsd",
    )
    for n_reps in n_reps_range
    for chi_max in chi_max_range
    for svd_min in svd_min_range
    for bond_distance in bond_distance_range
]
tasks_exact = [
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
    for bond_distance in bond_distance_range
]

mol_datas = {}
for bond_distance in bond_distance_range:
    filepath = os.path.join(
        MOLECULES_CATALOG_DIR,
        "data",
        "molecular_data",
        f"{molecule_basename}_d-{bond_distance:.2f}.json.xz",
    )
    mol_datas[bond_distance] = ffsim.MolecularData.from_json(
        filepath, compression="lzma"
    )


print("Loading data...")
data = {}
for task in tasks:
    filepath = DATA_ROOT / "lucj_mps_tenpy" / task.dirpath / "data.pickle"
    with open(filepath, "rb") as f:
        data[task] = pickle.load(f)
data_exact = {}
for task in tasks_exact:
    filepath = DATA_ROOT / "lucj_initial_params" / task.dirpath / "data.pickle"
    with open(filepath, "rb") as f:
        data_exact[task] = pickle.load(f)
print("Done loading data.")

markers = ["o", "s", "v", "D", "p", "*", "P", "X"]
prop_cycle = plt.rcParams["axes.prop_cycle"]
colors = prop_cycle.by_key()["color"]
alphas = [0.5, 1.0]
linestyles = ["--", ":"]

for bond_distance in bond_distance_range:
    fig, ax = plt.subplots(1, 1, figsize=(6, 6), layout="constrained")

    task = LUCJInitialParamsTask(
        molecule_basename=molecule_basename,
        bond_distance=bond_distance,
        lucj_params=LUCJParams(
            connectivity=connectivity,
            n_reps=None,
            with_final_orbital_rotation=True,
        ),
    )
    ax.axhline(
        data_exact[task]["error"],
        linestyle="--",
        label="LUCJ exact",
        color=colors[0],
    )

    tasks = [
        LUCJMPSTenpyTask(
            molecule_basename=molecule_basename,
            bond_distance=bond_distance,
            lucj_params=LUCJParams(
                connectivity=connectivity,
                n_reps=None,
                with_final_orbital_rotation=True,
            ),
            chi_max=chi_max,
            svd_min=svd_min_range[-1],
            params="ccsd",
        )
        for chi_max in chi_max_range
    ]
    errors = [data[task]["error"] for task in tasks]
    ax.plot(
        chi_max_range,
        errors,
        f"{markers[0]}{linestyles[0]}",
        label="LUCJ truncated",
        color=colors[2],
    )

    ax.set_title(connectivity)
    ax.set_xscale("log")
    ax.set_yscale("log")
    # ax.axhline(1.6e-3, linestyle="--", color="gray")
    ax.set_ylabel("Energy error (Hartree)")
    ax.set_xlabel("Max bond dimension")
    ax.set_xticks(chi_max_range)
    ax.legend()
    fig.suptitle(
        f"CCSD initial parameters {molecule_name} {basis} ({nelectron}e, {norb}o) R={bond_distance} Å"
    )

    filepath = os.path.join(
        plots_dir,
        f"{os.path.splitext(os.path.basename(__file__))[0]}_bond_distance-{bond_distance}.pdf",
    )
    plt.savefig(filepath)
    plt.close()
