import itertools
import os
import pickle
from pathlib import Path

import ffsim
import matplotlib.pyplot as plt

from lucj.params import LUCJParams
from lucj.tasks.lucj_sqd_initial_params_task import LUCJSQDInitialParamsTask
from lucj.tasks.uccsd_sqd_initial_params_task import UCCSDSQDInitialParamsTask
from lucj.tasks.lucj_sqd_compressed_t2_task import LUCJSQDCompressedT2Task
from lucj.tasks.lucj_sqd_compressed_t2_connectivity_task import LUCJSQDCompressedT2ConnectivityTask

DATA_ROOT = Path(os.environ.get("LUCJ_DATA_ROOT", "data"))
MOLECULES_CATALOG_DIR = Path(os.environ.get("MOLECULES_CATALOG_DIR"))


molecule_name = "n2"
basis = "sto-6g"
nelectron, norb = 10, 8
molecule_basename = f"{molecule_name}_{basis}_{nelectron}e{norb}o"

plots_dir = os.path.join("plots", molecule_basename)
os.makedirs(plots_dir, exist_ok=True)

start = 0.9
stop = 2.7
step = 0.1
bond_distance = 1.0

connectivities = [
    # "all-to-all",
    "square",
    "heavy-hex",
]
n_reps_range = list(range(2, 25, 2)) 

shots = 100_000
samples_per_batch_range = [100]
n_batches = 3
energy_tol = 1e-5
occupancies_tol = 1e-3
carryover_threshold = 1e-3
max_iterations = 100
symmetrize_spin = True
# TODO set entropy and generate seeds properly
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
        energy_tol=energy_tol,
        occupancies_tol=occupancies_tol,
        carryover_threshold=carryover_threshold,
        max_iterations=max_iterations,
        symmetrize_spin=symmetrize_spin,
        entropy=entropy,
    )
    for connectivity, n_reps, samples_per_batch in itertools.product(
        connectivities, n_reps_range + [None], samples_per_batch_range
    )
]

tasks_compressed_t2 = [
    LUCJSQDCompressedT2Task(
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
        energy_tol=energy_tol,
        occupancies_tol=occupancies_tol,
        carryover_threshold=carryover_threshold,
        max_iterations=max_iterations,
        symmetrize_spin=symmetrize_spin,
        entropy=entropy,
    )
    for connectivity, n_reps, samples_per_batch in itertools.product(
        connectivities, n_reps_range, samples_per_batch_range
    )
]

tasks_compressed_t2_multi_stage = [
    LUCJSQDCompressedT2Task(
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
        energy_tol=energy_tol,
        occupancies_tol=occupancies_tol,
        carryover_threshold=carryover_threshold,
        max_iterations=max_iterations,
        symmetrize_spin=symmetrize_spin,
        entropy=entropy,
    )
    for connectivity, n_reps, samples_per_batch in itertools.product(
        connectivities, n_reps_range, samples_per_batch_range
    )
]


tasks_compressed_t2_connectivity = [
    LUCJSQDCompressedT2ConnectivityTask(
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
        energy_tol=energy_tol,
        occupancies_tol=occupancies_tol,
        carryover_threshold=carryover_threshold,
        max_iterations=max_iterations,
        symmetrize_spin=symmetrize_spin,
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
        energy_tol=energy_tol,
        occupancies_tol=occupancies_tol,
        carryover_threshold=carryover_threshold,
        max_iterations=max_iterations,
        symmetrize_spin=symmetrize_spin,
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
results_compressed_t2 = {}
for task in tasks_compressed_t2:
    filepath = DATA_ROOT / "lucj_sqd_compressed_t2" / task.dirpath / "data.pickle"
    with open(filepath, "rb") as f:
        result = pickle.load(f)
        results_compressed_t2[task] = result

results_compressed_t2_multi_stage = {}
for task in tasks_compressed_t2_multi_stage:
    filepath = DATA_ROOT / "lucj_sqd_compressed_t2_multi_stage" / task.dirpath / "data.pickle"
    with open(filepath, "rb") as f:
        result = pickle.load(f)
        results_compressed_t2_multi_stage[task] = result

results_compressed_t2_connectivity = {}
for task in tasks_compressed_t2_connectivity:
    filepath = DATA_ROOT / "lucj_sqd_compressed_t2_connectivity" / task.dirpath / "data.pickle"
    with open(filepath, "rb") as f:
        result = pickle.load(f)
        results_compressed_t2_connectivity[task] = result

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

fig, axes = plt.subplots(3, len(connectivities), figsize=(12, 9), layout="constrained")

for samples_per_batch in samples_per_batch_range:
    fig, axes = plt.subplots(
        4, len(connectivities), figsize=(12, 6), layout="constrained"
    )
    for i, connectivity in enumerate(connectivities):
        task_uccsd = UCCSDSQDInitialParamsTask(
            molecule_basename=molecule_basename,
            bond_distance=bond_distance,
            shots=shots,
            samples_per_batch=samples_per_batch,
            n_batches=n_batches,
            energy_tol=energy_tol,
            occupancies_tol=occupancies_tol,
            carryover_threshold=carryover_threshold,
            max_iterations=max_iterations,
            symmetrize_spin=symmetrize_spin,
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
            energy_tol=energy_tol,
            occupancies_tol=occupancies_tol,
            carryover_threshold=carryover_threshold,
            max_iterations=max_iterations,
            symmetrize_spin=symmetrize_spin,
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
            data_lucj[task_lucj]["spin_squared"],
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
                energy_tol=energy_tol,
                occupancies_tol=occupancies_tol,
                carryover_threshold=carryover_threshold,
                max_iterations=max_iterations,
                symmetrize_spin=symmetrize_spin,
                entropy=entropy,
            )
            for n_reps in these_n_reps
        ]
        energies = [data_lucj[task]["energy"] for task in tasks_lucj]
        errors = [data_lucj[task]["error"] for task in tasks_lucj]
        spin_squares = [data_lucj[task]["spin_squared"] for task in tasks_lucj]
        sci_vec_shape = [data_lucj[task]["sci_vec_shape"][0] * data_lucj[task]["sci_vec_shape"][0] for task in tasks_lucj]

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
        axes[3, i].plot(
            these_n_reps,
            sci_vec_shape,
            f"{markers[0]}{linestyles[0]}",
            label="LUCJ truncated",
            color=colors[2],
        )

        tasks_compressed_t2 = [
            LUCJSQDCompressedT2Task(
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
                energy_tol=energy_tol,
                occupancies_tol=occupancies_tol,
                carryover_threshold=carryover_threshold,
                max_iterations=max_iterations,
                symmetrize_spin=symmetrize_spin,
                entropy=entropy,
            )
            for n_reps in these_n_reps
        ]
        # for task in tasks_compressed_t2:
        #     print(results_compressed_t2[task])
        #     input()
        energies = [results_compressed_t2[task]['energy'] for task in tasks_compressed_t2]
        errors = [results_compressed_t2[task]["error"] for task in tasks_compressed_t2]
        init_loss = [results_compressed_t2[task]["init_loss"] for task in tasks_compressed_t2]
        final_loss = [results_compressed_t2[task]["final_loss"] for task in tasks_compressed_t2]
        sci_vec_shape = [results_compressed_t2[task]["sci_vec_shape"][0] * results_compressed_t2[task]["sci_vec_shape"][0] for task in tasks_compressed_t2]

        axes[0, i].plot(
            these_n_reps,
            errors,
            f"{markers[0]}{linestyles[0]}",
            label="LUCJ Compressed-t2 truncated",
            color=colors[3],
        )
        axes[1, i].plot(
            these_n_reps,
            spin_squares,
            f"{markers[0]}{linestyles[0]}",
            label="LUCJ Compressed-t2 truncated",
            color=colors[3],
        )

        axes[2, i].plot(
            these_n_reps,
            init_loss,
            f"{markers[0]}{linestyles[0]}",
            label="LUCJ Compressed-t2 truncated",
            color=colors[4],
        )

        axes[2, i].plot(
            these_n_reps,
            final_loss,
            f"{markers[0]}{linestyles[0]}",
            label="LUCJ Compressed-t2 truncated",
            color=colors[3],
        )

        axes[3, i].plot(
            these_n_reps,
            sci_vec_shape,
            f"{markers[0]}{linestyles[0]}",
            label="LUCJ Compressed-t2 truncated",
            color=colors[3],
        )

        tasks_compressed_t2_multi_stage = [
            LUCJSQDCompressedT2Task(
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
                energy_tol=energy_tol,
                occupancies_tol=occupancies_tol,
                carryover_threshold=carryover_threshold,
                max_iterations=max_iterations,
                symmetrize_spin=symmetrize_spin,
                entropy=entropy,
            )
            for n_reps in these_n_reps
        ]

        energies = [results_compressed_t2_multi_stage[task]['energy'] for task in tasks_compressed_t2_multi_stage]
        errors = [results_compressed_t2_multi_stage[task]["error"] for task in tasks_compressed_t2_multi_stage]
        init_loss = [results_compressed_t2_multi_stage[task]["init_loss"] for task in tasks_compressed_t2_multi_stage]
        final_loss = [results_compressed_t2_multi_stage[task]["final_loss"] for task in tasks_compressed_t2_multi_stage]
        sci_vec_shape = [results_compressed_t2_multi_stage[task]["sci_vec_shape"][0] * results_compressed_t2_multi_stage[task]["sci_vec_shape"][0] for task in tasks_compressed_t2_multi_stage]

        axes[0, i].plot(
            these_n_reps,
            errors,
            f"{markers[0]}{linestyles[0]}",
            label="LUCJ Compressed-t2 multi-stage",
            color=colors[5],
        )
        axes[1, i].plot(
            these_n_reps,
            spin_squares,
            f"{markers[0]}{linestyles[0]}",
            label="LUCJ Compressed-t2 multi-stage",
            color=colors[5],
        )

        axes[2, i].plot(
            these_n_reps,
            final_loss,
            f"{markers[0]}{linestyles[0]}",
            label="final loss",
            color=colors[5],
        )

        axes[3, i].plot(
            these_n_reps,
            sci_vec_shape,
            f"{markers[0]}{linestyles[0]}",
            label="LUCJ Compressed-t2 multi-stage",
            color=colors[5],
        )

        tasks_compressed_t2_connectivity = [
            LUCJSQDCompressedT2ConnectivityTask(
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
                energy_tol=energy_tol,
                occupancies_tol=occupancies_tol,
                carryover_threshold=carryover_threshold,
                max_iterations=max_iterations,
                symmetrize_spin=symmetrize_spin,
                entropy=entropy,
            )
            for n_reps in these_n_reps
        ]

        energies = [results_compressed_t2_connectivity[task]['energy'] for task in tasks_compressed_t2_connectivity]
        errors = [results_compressed_t2_connectivity[task]["error"] for task in tasks_compressed_t2_connectivity]
        init_loss = [results_compressed_t2_connectivity[task]["init_loss"] for task in tasks_compressed_t2_connectivity]
        final_loss = [results_compressed_t2_connectivity[task]["final_loss"] for task in tasks_compressed_t2_connectivity]
        sci_vec_shape = [results_compressed_t2_connectivity[task]["sci_vec_shape"][0] * results_compressed_t2_connectivity[task]["sci_vec_shape"][0] for task in tasks_compressed_t2_connectivity]

        axes[0, i].plot(
            these_n_reps,
            errors,
            f"{markers[0]}{linestyles[0]}",
            label="LUCJ Compressed-t2 connectivity",
            color=colors[6],
        )
        axes[1, i].plot(
            these_n_reps,
            spin_squares,
            f"{markers[0]}{linestyles[0]}",
            label="LUCJ Compressed-t2 connectivity",
            color=colors[6],
        )

        axes[2, i].plot(
            these_n_reps,
            final_loss,
            f"{markers[0]}{linestyles[0]}",
            label="final loss",
            color=colors[6],
        )

        axes[3, i].plot(
            these_n_reps,
            sci_vec_shape,
            f"{markers[0]}{linestyles[0]}",
            label="LUCJ Compressed-t2 connectivity",
            color=colors[6],
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

        # axes[1, i].set_ylim(0, 0.1)
        axes[2, i].set_ylabel("loss")
        axes[2, i].set_xlabel("Repetitions")
        axes[2, i].set_xticks(these_n_reps)
        
        axes[1, 0].legend(ncol=2)
        
        fig.suptitle(
            f"CCSD initial parameters {molecule_name} {basis} ({nelectron}e, {norb}o) R={bond_distance} Å"
        )


    filepath = os.path.join(
        plots_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}_{bond_distance}.pdf"
    )
    plt.savefig(filepath)
    plt.close()
