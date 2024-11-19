import itertools
import os
import pickle
from pathlib import Path

import ffsim
import matplotlib.pyplot as plt

from lucj.params import LUCJParams
from lucj.tasks.lucj_initial_params_task import LUCJInitialParamsTask
from lucj.tasks.uccsd_initial_params_task import UCCSDInitialParamsTask

DATA_ROOT = Path(os.environ.get("LUCJ_DATA_ROOT", "data"))
MOLECULES_CATALOG_DIR = Path(os.environ.get("MOLECULES_CATALOG_DIR"))


molecule_name = "n2"
basis = "6-31g"
nelectron, norb = 10, 16
molecule_basename = f"{molecule_name}_{basis}_{nelectron}e{norb}o"

plots_dir = os.path.join("plots", molecule_basename)
os.makedirs(plots_dir, exist_ok=True)

start = 0.9
stop = 2.7
step = 0.1
bond_distance = 1.0

connectivities = [
    "all-to-all",
    "square",
    "heavy-hex",
]
n_reps_range = list(range(2, 25, 2)) + [None]
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
    for connectivity, n_reps in itertools.product(connectivities, n_reps_range)
]
task_uccsd = UCCSDInitialParamsTask(
    molecule_basename=molecule_basename, bond_distance=bond_distance
)


filepath = os.path.join(
    MOLECULES_CATALOG_DIR,
    "data",
    "molecular_data",
    f"{molecule_basename}_d-{bond_distance:.2f}.json.xz",
)
mol_data = ffsim.MolecularData.from_json(filepath, compression="lzma")


print("Loading data...")
data_lucj = {}
for task in tasks_lucj:
    filepath = DATA_ROOT / "lucj_initial_params" / task.dirpath / "data.pickle"
    with open(filepath, "rb") as f:
        data_lucj[task] = pickle.load(f)
data_uccsd = {}

filepath = DATA_ROOT / "uccsd_initial_params" / task_uccsd.dirpath / "data.pickle"
with open(filepath, "rb") as f:
    data_uccsd = pickle.load(f)
print("Done loading data.")

markers = ["o", "s", "v", "D", "p", "*", "P", "X"]
prop_cycle = plt.rcParams["axes.prop_cycle"]
colors = prop_cycle.by_key()["color"]
alphas = [0.5, 1.0]
linestyles = ["--", ":"]

fig, axes = plt.subplots(2, len(connectivities), figsize=(12, 6), layout="constrained")

for i, connectivity in enumerate(connectivities):
    axes[0, i].axhline(
        data_uccsd["error"],
        linestyle="--",
        label="UCCSD init",
        color=colors[0],
    )
    axes[1, i].axhline(
        data_uccsd["spin_squared"],
        linestyle="--",
        label="UCCSD init",
        color=colors[0],
    )

    task_lucj = LUCJInitialParamsTask(
        molecule_basename=molecule_basename,
        bond_distance=bond_distance,
        lucj_params=LUCJParams(
            connectivity=connectivity,
            n_reps=None,
            with_final_orbital_rotation=True,
        ),
    )
    full_n_reps = data_lucj[task_lucj]["n_reps"]
    axes[0, i].axhline(
        data_lucj[task_lucj]["error"],
        linestyle="--",
        label=f"LUCJ full ({full_n_reps} reps)",
        color=colors[1],
    )
    axes[1, i].axhline(
        data_lucj[task_lucj]["spin_squared"],
        linestyle="--",
        label=f"LUCJ full ({full_n_reps} reps)",
        color=colors[1],
    )

    these_n_reps = [n_reps for n_reps in n_reps_range if n_reps is not None]
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
        for n_reps in these_n_reps
    ]
    energies = [data_lucj[task]["energy"] for task in tasks_lucj]
    errors = [data_lucj[task]["error"] for task in tasks_lucj]
    spin_squares = [data_lucj[task]["spin_squared"] for task in tasks_lucj]
    axes[0, i].plot(
        these_n_reps,
        errors,
        f"{markers[0]}{linestyles[0]}",
        label="LUCJ truncated",
        color=colors[2],
    )
    axes[1, i].plot(
        these_n_reps,
        spin_squares,
        f"{markers[0]}{linestyles[0]}",
        label="LUCJ truncated",
        color=colors[2],
    )

    axes[0, i].set_title(connectivity)
    axes[0, i].set_yscale("log")
    axes[0, i].axhline(1.6e-3, linestyle="--", color="gray")
    axes[0, i].set_ylabel("Energy error (Hartree)")
    axes[0, i].set_xlabel("Repetitions")
    axes[0, i].set_xticks(these_n_reps)
    axes[1, i].set_ylim(0, 0.1)
    axes[1, i].set_ylabel("Spin squared")
    axes[1, i].set_xlabel("Repetitions")
    axes[1, i].set_xticks(these_n_reps)
    axes[1, 0].legend()
    fig.suptitle(f"{molecule_basename} ({nelectron}e, {norb}o) CCSD initial parameters")


filepath = os.path.join(
    plots_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}.pdf"
)
plt.savefig(filepath)
plt.close()
