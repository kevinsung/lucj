import itertools
import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt

from lucj.params import LUCJParams, CompressedT2Params
from lucj.sqd_energy_task.lucj_compressed_t2_task import SQDEnergyTask
from lucj.sqd_energy_task.lucj_random_t2_task import RandomSQDEnergyTask
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

bond_distance_range = [1.2, 2.4]

connectivities = [
    "all-to-all",
    # "square",
    "heavy-hex",
]
n_reps_range = list(range(2, 12, 2)) + [None]

shots = 100_000
samples_per_batch_range = [1000]
n_batches = 3
energy_tol = 1e-5
occupancies_tol = 1e-3
carryover_threshold = 1e-3
max_iterations = 100
symmetrize_spin = True
# TODO set entropy and generate seeds properly
entropy = 0

max_dim_range = [250, 500]


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


print("Loading data from random sample")
tasks_random = [
    RandomSQDEnergyTask(
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
        max_dim=max_dim,
    )
    for samples_per_batch, max_dim, bond_distance in itertools.product(
        samples_per_batch_range, max_dim_range, bond_distance_range
    )
]

results_random = {}
for task in tasks_random:
    filepath = DATA_ROOT / task.dirpath / "sqd_data.pickle"
    results_random[task] = load_data(filepath)

print("Done loading data.")

markers = ["o", "s", "v", "D", "p", "*", "P", "X"]
prop_cycle = plt.rcParams["axes.prop_cycle"]
colors = prop_cycle.by_key()["color"]
alphas = [0.7, 0.8, 0.9, 1.0]
linestyles = ["--", ":"]

row_error = 0
row_spin_square = 1
# row_loss = 2
row_sci_vec_dim = 2

for samples_per_batch, connectivity, bond_distance in itertools.product(
    samples_per_batch_range, connectivities, bond_distance_range
):
    fig, axes = plt.subplots(
        3,
        len(max_dim_range),
        figsize=(12, 6),  # , layout="constrained"
    )
    for i, max_dim in enumerate(max_dim_range):
        # UCCSD data
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

        filepath = (
            "lucj/uccsd_sqd_initial_params" / task_uccsd.dirpath / "data.pickle"
        )
        data_uccsd = load_data(filepath)

        axes[row_error, i].axhline(
            data_uccsd["error"],
            linestyle="--",
            label="UCCSD init",
            color=colors[0],
        )
        axes[row_spin_square, i].axhline(
            data_uccsd["spin_squared"],
            linestyle="--",
            label="UCCSD init",
            color=colors[0],
        )
        axes[row_sci_vec_dim, i].axhline(
            data_uccsd["sci_vec_shape"][0],
            linestyle="--",
            label="UCCSD init",
            color=colors[0],
        )

        # random sqd sample
        task_random = RandomSQDEnergyTask(
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
            max_dim=max_dim,
        )

        print(results_random[task_random]["error"])

        axes[row_error, i].axhline(
            results_random[task_random]["error"],
            linestyle="--",
            label="Random bitstr",
            color="red",
        )
        axes[row_spin_square, i].axhline(
            results_random[task_random]["spin_squared"],
            linestyle="--",
            label="Random bitstr",
            color="red",
        )
        axes[row_sci_vec_dim, i].axhline(
            results_random[task_random]["sci_vec_shape"][0],
            linestyle="--",
            label="Random bitstr",
            color="red",
        )

        # LUCJ data
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
                max_dim=max_dim
            )
            for n_reps in n_reps_range
        ]
        data_lucj = {}
        for task in tasks_lucj:
            filepath = (
                "lucj/lucj_sqd_initial_params" / task.dirpath / "data.pickle"
            )
            data_lucj[task] = load_data(filepath)

        task_lucj = tasks_lucj[-1]
        assert task_lucj.lucj_params.n_reps is None
        full_n_reps = data_lucj[task_lucj]["n_reps"]
        axes[row_error, i].axhline(
            data_lucj[task_lucj]["error"],
            linestyle="--",
            label=f"LUCJ full ({full_n_reps} reps)",
            color=colors[1],
        )
        axes[row_sci_vec_dim, i].axhline(
            data_lucj[task_lucj]["sci_vec_shape"][0],
            linestyle="--",
            label=f"LUCJ full ({full_n_reps} reps)",
            color=colors[1],
        )
        axes[row_spin_square, i].axhline(
            data_lucj[task_lucj]["spin_squared"],
            linestyle="--",
            label=f"LUCJ full ({full_n_reps} reps)",
            color=colors[1],
        )

        these_n_reps = [n_reps for n_reps in n_reps_range if n_reps is not None]

        energies = [data_lucj[task]["energy"] for task in tasks_lucj[:-1]]
        errors = [data_lucj[task]["error"] for task in tasks_lucj[:-1]]
        spin_squares = [data_lucj[task]["spin_squared"] for task in tasks_lucj[:-1]]
        sci_vec_shape = [
            data_lucj[task]["sci_vec_shape"][0]
            for task in tasks_lucj[:-1]
        ]
        axes[row_error, i].plot(
            these_n_reps,
            errors,
            f"{markers[0]}{linestyles[0]}",
            label="LUCJ truncated",
            color=colors[2],
        )
        axes[row_spin_square, i].plot(
            these_n_reps,
            spin_squares,
            f"{markers[0]}{linestyles[0]}",
            label="LUCJ truncated",
            color=colors[2],
        )
        axes[row_sci_vec_dim, i].plot(
            these_n_reps,
            sci_vec_shape,
            f"{markers[0]}{linestyles[0]}",
            label="LUCJ truncated",
            color=colors[2],
        )

        # compressed_t2
        tasks_compressed_t2 = [
            SQDEnergyTask(
                molecule_basename=molecule_basename,
                bond_distance=bond_distance,
                lucj_params=LUCJParams(
                    connectivity=connectivity,
                    n_reps=n_reps,
                    with_final_orbital_rotation=True,
                ),
                compressed_t2_params=CompressedT2Params(
                    multi_stage_optimization=True, begin_reps=20, step=2
                ),
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
            for n_reps in these_n_reps
        ]

        results_compressed_t2 = {}
        for task in tasks_compressed_t2:
            filepath = DATA_ROOT / task.dirpath / "sqd_data.pickle"
            results_compressed_t2[task] = load_data(filepath)
            # print(filepath)
            # print(results_compressed_t2[task])
            # input()
            # filepath = DATA_ROOT / task.operatorpath / "opt_data.pickle"
            # with open(filepath, "rb") as f:
            #     result = pickle.load(f)
            # results_compressed_t2[task] = results_compressed_t2[task] | result

        energies = [
            results_compressed_t2[task]["energy"] for task in tasks_compressed_t2
        ]
        errors = [results_compressed_t2[task]["error"] for task in tasks_compressed_t2]
        spin_squares = [
            results_compressed_t2[task]["spin_squared"] for task in tasks_compressed_t2
        ]
        sci_vec_shape = [
            results_compressed_t2[task]["sci_vec_shape"][0]
            for task in tasks_compressed_t2
        ]
        axes[row_error, i].plot(
            these_n_reps,
            errors,
            f"{markers[0]}{linestyles[0]}",
            label="LUCJ Compressed-t2",
            color=colors[5],
        )

        axes[row_spin_square, i].plot(
            these_n_reps,
            spin_squares,
            f"{markers[0]}{linestyles[0]}",
            label="LUCJ Compressed-t2",
            color=colors[5],
        )

        axes[row_sci_vec_dim, i].plot(
            these_n_reps,
            sci_vec_shape,
            f"{markers[0]}{linestyles[0]}",
            label="LUCJ Compressed-t2",
            color=colors[5],
        )

        # compress t2 reg0
        tasks_compressed_t2_reg0 = [
            SQDEnergyTask(
                molecule_basename=molecule_basename,
                bond_distance=bond_distance,
                lucj_params=LUCJParams(
                    connectivity=connectivity,
                    n_reps=n_reps,
                    with_final_orbital_rotation=True,
                ),
                compressed_t2_params=CompressedT2Params(
                    multi_stage_optimization=True, begin_reps=20, step=2
                ),
                regularization=True,
                regularization_option=0,
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
            for n_reps in these_n_reps
        ]
        results_compressed_t2_reg0 = {}
        for task in tasks_compressed_t2_reg0:
            filepath = DATA_ROOT / task.dirpath / "sqd_data.pickle"
            results_compressed_t2_reg0[task] = load_data(filepath)
            

        energies = [
            results_compressed_t2_reg0[task]["energy"]
            for task in results_compressed_t2_reg0
        ]
        errors = [
            results_compressed_t2_reg0[task]["error"]
            for task in results_compressed_t2_reg0
        ]
        spin_squares = [
            results_compressed_t2_reg0[task]["spin_squared"]
            for task in results_compressed_t2_reg0
        ]
        sci_vec_shape = [
            results_compressed_t2_reg0[task]["sci_vec_shape"][0]
            for task in tasks_compressed_t2_reg0
        ]
        axes[row_error, i].plot(
            these_n_reps,
            errors,
            f"{markers[1]}{linestyles[0]}",
            label="LUCJ Compressed-t2-reg0",
            color=colors[5],
            alpha=alphas[0],
        )

        axes[row_spin_square, i].plot(
            these_n_reps,
            spin_squares,
            f"{markers[1]}{linestyles[0]}",
            label="LUCJ Compressed-t2-reg0",
            color=colors[5],
            alpha=alphas[0],
        )

        axes[row_sci_vec_dim, i].plot(
            these_n_reps,
            sci_vec_shape,
            f"{markers[1]}{linestyles[0]}",
            label="LUCJ Compressed-t2-reg0",
            color=colors[5],
            alpha=alphas[0],
        )

        # compress t2 reg1
        tasks_compressed_t2_reg1 = [
            SQDEnergyTask(
                molecule_basename=molecule_basename,
                bond_distance=bond_distance,
                lucj_params=LUCJParams(
                    connectivity=connectivity,
                    n_reps=n_reps,
                    with_final_orbital_rotation=True,
                ),
                compressed_t2_params=CompressedT2Params(
                    multi_stage_optimization=True, begin_reps=20, step=2
                ),
                regularization=True,
                regularization_option=1,
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
            for n_reps in these_n_reps
        ]
        results_compressed_t2_reg1 = {}
        for task in tasks_compressed_t2_reg1:
            filepath = DATA_ROOT / task.dirpath / "sqd_data.pickle"
            results_compressed_t2_reg1[task] = load_data(filepath)

        energies = [
            results_compressed_t2_reg1[task]["energy"]
            for task in results_compressed_t2_reg1
        ]
        errors = [
            results_compressed_t2_reg1[task]["error"]
            for task in results_compressed_t2_reg1
        ]
        spin_squares = [
            results_compressed_t2_reg1[task]["spin_squared"]
            for task in results_compressed_t2_reg1
        ]
        sci_vec_shape = [
            results_compressed_t2_reg1[task]["sci_vec_shape"][0]
            for task in tasks_compressed_t2_reg1
        ]
        axes[row_error, i].plot(
            these_n_reps,
            errors,
            f"{markers[2]}{linestyles[0]}",
            label="LUCJ Compressed-t2-reg1",
            color=colors[5],
            alpha=alphas[1],
        )

        axes[row_spin_square, i].plot(
            these_n_reps,
            spin_squares,
            f"{markers[2]}{linestyles[0]}",
            label="LUCJ Compressed-t2-reg1",
            color=colors[5],
            alpha=alphas[1],
        )

        axes[row_sci_vec_dim, i].plot(
            these_n_reps,
            sci_vec_shape,
            f"{markers[2]}{linestyles[0]}",
            label="LUCJ Compressed-t2-reg1",
            color=colors[5],
            alpha=alphas[1],
        )

        # compress t2 reg2
        tasks_compressed_t2_reg2 = [
            SQDEnergyTask(
                molecule_basename=molecule_basename,
                bond_distance=bond_distance,
                lucj_params=LUCJParams(
                    connectivity=connectivity,
                    n_reps=n_reps,
                    with_final_orbital_rotation=True,
                ),
                compressed_t2_params=CompressedT2Params(
                    multi_stage_optimization=True, begin_reps=20, step=2
                ),
                regularization=True,
                regularization_option=2,
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
            for n_reps in these_n_reps
        ]
        results_compressed_t2_reg2 = {}
        for task in tasks_compressed_t2_reg2:
            filepath = DATA_ROOT / task.dirpath / "sqd_data.pickle"
            results_compressed_t2_reg2[task] = load_data(filepath)

        energies = [
            results_compressed_t2_reg2[task]["energy"]
            for task in results_compressed_t2_reg2
        ]
        errors = [
            results_compressed_t2_reg2[task]["error"]
            for task in results_compressed_t2_reg2
        ]
        spin_squares = [
            results_compressed_t2_reg2[task]["spin_squared"]
            for task in results_compressed_t2_reg2
        ]
        sci_vec_shape = [
            results_compressed_t2_reg2[task]["sci_vec_shape"][0]
            for task in tasks_compressed_t2_reg2
        ]
        axes[row_error, i].plot(
            these_n_reps,
            errors,
            f"{markers[3]}{linestyles[0]}",
            label="LUCJ Compressed-t2-reg2",
            color=colors[5],
            alpha=alphas[2],
        )

        axes[row_spin_square, i].plot(
            these_n_reps,
            spin_squares,
            f"{markers[3]}{linestyles[0]}",
            label="LUCJ Compressed-t2-reg2",
            color=colors[5],
            alpha=alphas[2],
        )

        axes[row_sci_vec_dim, i].plot(
            these_n_reps,
            sci_vec_shape,
            f"{markers[3]}{linestyles[0]}",
            label="LUCJ Compressed-t2-reg2",
            color=colors[5],
            alpha=alphas[2],
        )

        if connectivity != "all-to-all":
            tasks_compressed_t2_connectivity = [
                SQDEnergyTask(
                    molecule_basename=molecule_basename,
                    bond_distance=bond_distance,
                    lucj_params=LUCJParams(
                        connectivity=connectivity,
                        n_reps=n_reps,
                        with_final_orbital_rotation=True,
                    ),
                    compressed_t2_params=None,
                    connectivity_opt=True,
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
                for n_reps in these_n_reps
            ]

            results_compressed_t2_connectivity = {}
            for task in tasks_compressed_t2_connectivity:
                filepath = DATA_ROOT / task.dirpath / "sqd_data.pickle"
                results_compressed_t2_connectivity[task] = load_data(filepath)
                # print(filepath)
                # input()

                # filepath = (
                #     DATA_ROOT / molecule_basename / task.operatorpath / "opt_data.pickle"
                # )
                # with open(filepath, "rb") as f:
                #     result = pickle.load(f)
                # results_compressed_t2_connectivity[task] = results_compressed_t2_connectivity[task] | result

            energies = [
                results_compressed_t2_connectivity[task]["energy"]
                for task in tasks_compressed_t2_connectivity
            ]
            errors = [
                results_compressed_t2_connectivity[task]["error"]
                for task in tasks_compressed_t2_connectivity
            ]
            spin_squares = [
                results_compressed_t2_connectivity[task]["spin_squared"]
                for task in tasks_compressed_t2_connectivity
            ]
            # init_loss = [
            #     results_compressed_t2_connectivity[task]["init_loss"]
            #     for task in tasks_compressed_t2_connectivity
            # ]
            # final_loss = [
            #     results_compressed_t2_connectivity[task]["final_loss"]
            #     for task in tasks_compressed_t2_connectivity
            # ]
            sci_vec_shape = [
                results_compressed_t2_connectivity[task]["sci_vec_shape"][0]
                for task in tasks_compressed_t2_connectivity
            ]

            axes[row_error, i].plot(
                these_n_reps,
                errors,
                f"{markers[0]}{linestyles[0]}",
                label="LUCJ Compressed-t2 connectivity",
                color=colors[6],
            )

            axes[row_spin_square, i].plot(
                these_n_reps,
                spin_squares,
                f"{markers[0]}{linestyles[0]}",
                label="LUCJ Compressed-t2 connectivity",
                color=colors[6],
            )

            # axes[row_loss, i].plot(
            #     these_n_reps,
            #     init_loss,
            #     f"{markers[0]}{linestyles[1]}",
            #     label="LUCJ Compressed-t2 connectivity",
            #     color=colors[6],
            # )

            # axes[row_loss, i].plot(
            #     these_n_reps,
            #     final_loss,
            #     f"{markers[0]}{linestyles[0]}",
            #     # label="final loss",
            #     color=colors[6],
            # )

            axes[row_sci_vec_dim, i].plot(
                these_n_reps,
                sci_vec_shape,
                f"{markers[0]}{linestyles[0]}",
                label="LUCJ Compressed-t2 connectivity",
                color=colors[6],
            )

        axes[row_error, i].set_title(f"max dim: {max_dim}")
        axes[row_error, i].set_yscale("log")
        axes[row_error, i].axhline(1.6e-3, linestyle="--", color="gray")
        axes[row_error, i].set_ylabel("Energy error (Hartree)")
        axes[row_error, i].set_xlabel("Repetitions")
        axes[row_error, i].set_xticks(these_n_reps)
        # axes[row_loss, i].set_ylabel("loss")
        # axes[row_loss, i].set_xlabel("Repetitions")
        # axes[row_loss, i].set_xticks(these_n_reps)

        axes[row_spin_square, i].set_ylim(0, 0.1)
        axes[row_spin_square, i].set_ylabel("Spin square")
        axes[row_spin_square, i].set_xlabel("Repetitions")
        axes[row_spin_square, i].set_xticks(these_n_reps)

        axes[row_sci_vec_dim, i].set_ylabel("SCI subspace")
        axes[row_sci_vec_dim, i].set_xlabel("Repetitions")
        axes[row_sci_vec_dim, i].set_xticks(these_n_reps)

        # axes[row_sci_vec_dim, 0].legend(ncol=2, )
        leg = axes[row_sci_vec_dim, 1].legend(
            bbox_to_anchor=(-0.3, -0.4), loc="upper center", ncol=3
        )
        # leg = axes[row_sci_vec_dim, 1].legend(
        #     bbox_to_anchor=(0.5, -0.4), loc="upper center", ncol=3
        # )
        leg.set_in_layout(False)
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.2)

        fig.suptitle(
            f"CCSD initial parameters {molecule_name} {basis} ({nelectron}e, {norb}o) R={bond_distance} Å / {connectivity}"
        )

    filepath = os.path.join(
        plots_dir,
        f"{os.path.splitext(os.path.basename(__file__))[0]}_{bond_distance}_{samples_per_batch}_{connectivity}.pdf",
    )
    plt.savefig(filepath)
    plt.close()
