import itertools
import os
import pickle
from pathlib import Path

import ffsim
import matplotlib.pyplot as plt

from lucj.params import LUCJParams
from lucj.tasks.lucj_sqd_initial_params_task import LUCJSQDInitialParamsTask
from lucj.tasks.uccsd_sqd_initial_params_task import UCCSDSQDInitialParamsTask

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
shots = 100_000
samples_per_batch_range = [1000, 2000]
n_batches = 3
max_davidson = 200
entropy = 0
tasks_lucj = [
    LUCJSQDInitialParamsTask(
        molecule_basename=molecule_basename,
        bond_distance=bond_distance,
        lucj_params=LUCJParams(
            connectivity=connectivity,
            n_reps=n_reps,
            with_final_orbital_rotation=True,
        ),
        shots=shots,
        samples_per_batch=samples_per_batch,
        n_batches=n_batches,
        max_davidson=max_davidson,
        entropy=entropy,
    )
    for connectivity, n_reps, samples_per_batch in itertools.product(
        connectivities, n_reps_range, samples_per_batch_range
    )
]
tasks_uccsd = [
    UCCSDSQDInitialParamsTask(
        molecule_basename=molecule_basename,
        bond_distance=bond_distance,
        shots=shots,
        samples_per_batch=samples_per_batch,
        n_batches=n_batches,
        max_davidson=max_davidson,
        entropy=entropy,
    )
    for samples_per_batch in samples_per_batch_range
]


filepath = os.path.join(
    MOLECULES_CATALOG_DIR,
    "data",
    "molecular_data",
    f"{molecule_basename}_d-{bond_distance:.5f}.json.xz",
)
mol_data = ffsim.MolecularData.from_json(filepath, compression="lzma")


print("Loading data...")
data_lucj = {}
for task in tasks_lucj:
    filepath = DATA_ROOT / "lucj_sqd_initial_params" / task.dirpath / "data.pickle"
    with open(filepath, "rb") as f:
        data_lucj[task] = pickle.load(f)
data_uccsd = {}
for task in tasks_uccsd:
    filepath = DATA_ROOT / "uccsd_sqd_initial_params" / task.dirpath / "data.pickle"
    with open(filepath, "rb") as f:
        data_uccsd[task] = pickle.load(f)
print("Done loading data.")

markers = ["o", "s", "v", "D", "p", "*", "P", "X"]
prop_cycle = plt.rcParams["axes.prop_cycle"]
colors = prop_cycle.by_key()["color"]
alphas = [0.5, 1.0]
linestyles = ["--", ":"]

for samples_per_batch in samples_per_batch_range:
    fig, axes = plt.subplots(
        2, len(connectivities), figsize=(12, 6), layout="constrained"
    )
    for i, connectivity in enumerate(connectivities):
        task_uccsd = UCCSDSQDInitialParamsTask(
            molecule_basename=molecule_basename,
            bond_distance=bond_distance,
            shots=shots,
            samples_per_batch=samples_per_batch,
            n_batches=n_batches,
            max_davidson=max_davidson,
            entropy=entropy,
        )
        axes[0, i].axhline(
            data_uccsd[task_uccsd]["error"],
            linestyle="--",
            label="UCCSD init",
            color=colors[0],
        )
        axes[1, i].axhline(
            data_uccsd[task_uccsd]["sci_vec_shape"][0],
            linestyle="--",
            label="UCCSD init",
            color=colors[0],
        )

        task_lucj = LUCJSQDInitialParamsTask(
            molecule_basename=molecule_basename,
            bond_distance=bond_distance,
            lucj_params=LUCJParams(
                connectivity=connectivity,
                n_reps=None,
                with_final_orbital_rotation=True,
            ),
            shots=shots,
            samples_per_batch=samples_per_batch,
            n_batches=n_batches,
            max_davidson=max_davidson,
            entropy=entropy,
        )
        full_n_reps = data_lucj[task_lucj]["n_reps"]
        axes[0, i].axhline(
            data_lucj[task_lucj]["error"],
            linestyle="--",
            label=f"LUCJ full ({full_n_reps} reps)",
            color=colors[1],
        )
        axes[1, i].axhline(
            data_lucj[task_lucj]["sci_vec_shape"][0],
            linestyle="--",
            label=f"LUCJ full ({full_n_reps} reps)",
            color=colors[1],
        )

        these_n_reps = [n_reps for n_reps in n_reps_range if n_reps is not None]
        tasks_lucj = [
            LUCJSQDInitialParamsTask(
                molecule_basename=molecule_basename,
                bond_distance=bond_distance,
                lucj_params=LUCJParams(
                    connectivity=connectivity,
                    n_reps=n_reps,
                    with_final_orbital_rotation=True,
                ),
                shots=shots,
                samples_per_batch=samples_per_batch,
                n_batches=n_batches,
                max_davidson=max_davidson,
                entropy=entropy,
            )
            for n_reps in these_n_reps
        ]
        energies = [data_lucj[task]["energy"] for task in tasks_lucj]
        errors = [data_lucj[task]["error"] for task in tasks_lucj]
        sci_dims = [data_lucj[task]["sci_vec_shape"][0] for task in tasks_lucj]
        axes[0, i].plot(
            these_n_reps,
            errors,
            f"{markers[0]}{linestyles[0]}",
            label="LUCJ truncated",
            color=colors[2],
        )
        axes[1, i].plot(
            these_n_reps,
            sci_dims,
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
        axes[1, i].set_ylabel("SCI dim (single spin)")
        axes[1, i].set_xlabel("Repetitions")
        axes[1, i].set_xticks(these_n_reps)
        axes[1, 0].legend()
        fig.suptitle(
            f"CCSD initial parameters {molecule_name} {basis} ({nelectron}e, {norb}o) R={bond_distance} Å"
        )

    filepath = os.path.join(
        plots_dir,
        f"{os.path.splitext(os.path.basename(__file__))[0]}_samples_per_batch-{samples_per_batch}.pdf",
    )
    plt.savefig(filepath)
    plt.close()
