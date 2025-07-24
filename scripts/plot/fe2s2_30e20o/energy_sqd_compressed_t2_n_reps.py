import itertools
import os
import pickle
from pathlib import Path

import ffsim
import matplotlib.pyplot as plt

from lucj.params import LUCJParams, CompressedT2Params
from lucj.tasks.lucj_sqd_initial_params_task import LUCJSQDInitialParamsTask
from lucj.sqd_energy_task.lucj_compressed_t2_task import SQDEnergyTask
from lucj.sqd_energy_task.lucj_random_t2_task import RandomSQDEnergyTask

DATA_ROOT = "/media/storage/WanHsuan.Lin/"
MOLECULES_CATALOG_DIR = Path(os.environ.get("MOLECULES_CATALOG_DIR"))

molecule_name = "fe2s2"
nelectron, norb = 30, 20
molecule_basename = f"{molecule_name}_{nelectron}e{norb}o"

plots_dir = os.path.join("plots", molecule_basename)
os.makedirs(plots_dir, exist_ok=True)

connectivities = [
    "all-to-all",
    "heavy-hex",
]
n_reps_range = list(range(2, 12, 2)) 

shots = 100_000
samples_per_batch = 1000
n_batches = 3
energy_tol = 1e-5
occupancies_tol = 1e-3
carryover_threshold = 1e-3
max_iterations = 100
symmetrize_spin = True
# TODO set entropy and generate seeds properly
entropy = 0
max_dim_range = [500, 1000]

dmrg_energy = -116.6056091 #ref: https://github.com/jrm874/sqd_data_repository/blob/main/classical_reference_energies/2Fe-2S/classical_methods_energies.txt

tasks_lucj = [
    SQDEnergyTask(
        molecule_basename=molecule_basename,
        bond_distance=None,
        lucj_params=LUCJParams(
            connectivity=connectivity,
            n_reps=n_reps,
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
    for max_dim, n_reps in itertools.product(max_dim_range, n_reps_range)
    for connectivity in connectivities
]

tasks_lucj_full = [
    SQDEnergyTask(
        molecule_basename=molecule_basename,
        bond_distance=None,
        lucj_params=LUCJParams(
            connectivity=connectivity,
            n_reps=None,
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
    for max_dim in max_dim_range
    for connectivity in connectivities
]

tasks_compressed_t2 = [
    SQDEnergyTask(
        molecule_basename=molecule_basename,
        bond_distance=None,
        lucj_params=LUCJParams(
            connectivity=connectivity,
            n_reps=n_reps,
            with_final_orbital_rotation=True,
        ),
        compressed_t2_params=CompressedT2Params(
            multi_stage_optimization=True,
            begin_reps=20,
            step=2
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
    for max_dim, n_reps in itertools.product(max_dim_range, n_reps_range)
    for connectivity in connectivities
]


tasks_compressed_t2_connectivity = [
    SQDEnergyTask(
        molecule_basename=molecule_basename,
        bond_distance=None,
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
    for connectivity, n_reps in itertools.product(connectivities, n_reps_range)
    for max_dim in max_dim_range
]

tasks_random_bit_string = [
    RandomSQDEnergyTask(
        molecule_basename=molecule_basename,
        bond_distance=None,
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
    for max_dim in max_dim_range
]

tasks_random_valid_bit_string = [
    RandomSQDEnergyTask(
        molecule_basename=molecule_basename,
        bond_distance=None,
        shots=shots,
        valid_string_only=True,
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
    for max_dim in max_dim_range
]


filepath = os.path.join(
    MOLECULES_CATALOG_DIR,
    "data",
    "molecular_data",
    f"{molecule_basename}.json.xz",
)
mol_data = ffsim.MolecularData.from_json(filepath, compression="lzma")


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

print("Loading data...")
results_compressed_t2 = {}
for task in tasks_compressed_t2:
    filepath = DATA_ROOT / task.dirpath / "sqd_data.pickle"
    results_compressed_t2[task] = load_data(filepath)

opt_results_compressed_t2 = {}
for task in tasks_compressed_t2:
    filepath = DATA_ROOT / task.operatorpath / "opt_data.pickle"
    opt_results_compressed_t2[task] = load_data(filepath)

results_compressed_t2_connectivity = {}
for task in tasks_compressed_t2_connectivity:
    filepath = DATA_ROOT / task.dirpath / "sqd_data.pickle"
    results_compressed_t2_connectivity[task] = load_data(filepath)

results_random = {}
for task in tasks_random_bit_string:
    filepath = DATA_ROOT / task.dirpath / "sqd_data.pickle"
    results_random[task] = load_data(filepath)

results_random_valid_bitstrings = {}
for task in tasks_random_valid_bit_string:
    filepath = DATA_ROOT / task.dirpath / "sqd_data.pickle"
    results_random_valid_bitstrings[task] = load_data(filepath)
    
data_lucj = {}
for task in tasks_lucj:
    filepath = DATA_ROOT / task.dirpath / "sqd_data.pickle"
    data_lucj[task] = load_data(filepath)

data_lucj_full = {}
for task in tasks_lucj_full:
    filepath = DATA_ROOT / task.dirpath / "sqd_data.pickle"
    data_lucj_full[task] = load_data(filepath)

print("Done loading data.")

markers = ["o", "s", "v", "D", "p", "*", "P", "X"]
prop_cycle = plt.rcParams["axes.prop_cycle"]
colors = prop_cycle.by_key()["color"]
alphas = [0.5, 1.0]
linestyles = ["--", ":"]

fig, axes = plt.subplots(
    3, len(connectivities) * len(max_dim_range), figsize=(12, 6) #, layout="constrained"
)
for i, (connectivity, max_dim) in enumerate(itertools.product(connectivities, max_dim_range)):
    task_random_bit_string = RandomSQDEnergyTask(
                                molecule_basename=molecule_basename,
                                bond_distance=None,
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
    
    print(f"{connectivity}/max dim-{max_dim}")
    print("results_random")
    print(results_random[task_random_bit_string]["energy"] - dmrg_energy)
    # print(results_random[task_random_bit_string]["error"])
    axes[0, i].axhline(
        results_random[task_random_bit_string]["energy"] - dmrg_energy,
        # results_random[task_random_bit_string]["error"],
        linestyle="--",
        label="Rand bitstr",
        color="red",
    )

    axes[1, i].axhline(
        results_random[task_random_bit_string]["spin_squared"],
        linestyle="--",
        label="Rand bitstr",
        color="red",
    )

    axes[2, i].axhline(
        results_random[task_random_bit_string]["sci_vec_shape"][0],
        linestyle="--",
        label="Rand bitstr",
        color="red",
    )
    
    tasks_random_valid_bit_string = RandomSQDEnergyTask(
                                molecule_basename=molecule_basename,
                                bond_distance=None,
                                shots=shots,
                                samples_per_batch=samples_per_batch,
                                n_batches=n_batches,
                                valid_string_only=True,
                                energy_tol=energy_tol,
                                occupancies_tol=occupancies_tol,
                                carryover_threshold=carryover_threshold,
                                max_iterations=max_iterations,
                                symmetrize_spin=symmetrize_spin,
                                entropy=entropy,
                                max_dim=max_dim,
                            )
    
    print("results_random_valid_bitstring")
    print(results_random_valid_bitstrings[tasks_random_valid_bit_string]["energy"] - dmrg_energy)
    # print(results_random[task_random_bit_string]["error"])
    axes[0, i].axhline(
        results_random_valid_bitstrings[tasks_random_valid_bit_string]["energy"] - dmrg_energy,
        # results_random[task_random_bit_string]["error"],
        linestyle="--",
        label="Rand valid bitstr",
        color="palevioletred"
    )

    axes[1, i].axhline(
        results_random_valid_bitstrings[tasks_random_valid_bit_string]["spin_squared"],
        linestyle="--",
        label="Rand valid bitstr",
        color="palevioletred"
    )

    axes[2, i].axhline(
        results_random_valid_bitstrings[tasks_random_valid_bit_string]["sci_vec_shape"][0],
        linestyle="--",
        label="Rand valid bitstr",
        color="palevioletred",
    )

    task_lucj_full = SQDEnergyTask(
                        molecule_basename=molecule_basename,
                        bond_distance=None,
                        lucj_params=LUCJParams(
                            connectivity=connectivity,
                            n_reps=None,
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
    
    
    axes[0, i].axhline(
        data_lucj_full[task_lucj_full]["energy"] - dmrg_energy,
        # results_random[task_random_bit_string]["error"],
        linestyle="--",
        label="LUCJ full",
        color=colors[1],
    )

    axes[1, i].axhline(
        data_lucj_full[task_lucj_full]["spin_squared"],
        linestyle="--",
        label="LUCJ full",
        color=colors[1],
    )

    axes[2, i].axhline(
        data_lucj_full[task_lucj_full]["sci_vec_shape"][0],
        linestyle="--",
        label="LUCJ full",
        color=colors[1],
    )


    these_n_reps = [n_reps for n_reps in n_reps_range if n_reps is not None]
    tasks_lucj = [
        SQDEnergyTask(
            molecule_basename=molecule_basename,
            bond_distance=None,
            lucj_params=LUCJParams(
                connectivity=connectivity,
                n_reps=n_reps,
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
        for n_reps in these_n_reps
    ]
    energies = [data_lucj[task]["energy"] for task in tasks_lucj]
    if data_lucj[tasks_lucj[0]]["error"] == 0:
        errors = [data_lucj[task]["error"] for task in tasks_lucj]
    else:
        errors = [data_lucj[task]["energy"] - dmrg_energy for task in tasks_lucj]
    spin_squares = [data_lucj[task]["spin_squared"] for task in tasks_lucj]
    sci_vec_shape = [data_lucj[task]["sci_vec_shape"][0] for task in tasks_lucj]

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
    axes[2, i].plot(
        these_n_reps,
        sci_vec_shape,
        f"{markers[0]}{linestyles[0]}",
        label="LUCJ truncated",
        color=colors[2],
    )

    tasks_compressed_t2 = [
        SQDEnergyTask(
        molecule_basename=molecule_basename,
        bond_distance=None,
        lucj_params=LUCJParams(
            connectivity=connectivity,
            n_reps=n_reps,
            with_final_orbital_rotation=True,
        ),
        compressed_t2_params=CompressedT2Params(
            multi_stage_optimization=True,
            begin_reps=20,
            step=2
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
    for n_reps in these_n_reps]

    energies = [results_compressed_t2[task]['energy'] for task in tasks_compressed_t2]
    # errors = [results_compressed_t2[task]["error"] for task in tasks_compressed_t2]
    errors = [results_compressed_t2[task]["energy"] - dmrg_energy  for task in tasks_compressed_t2]
    sci_vec_shape = [results_compressed_t2[task]["sci_vec_shape"][0] for task in tasks_compressed_t2]

    axes[0, i].plot(
        these_n_reps,
        errors,
        f"{markers[0]}{linestyles[0]}",
        label="LUCJ Compressed-t2 multi-stage",
        color=colors[5],
    )
    print(connectivity)
    print(energies)
    print(errors)

    axes[2, i].plot(
        these_n_reps,
        sci_vec_shape,
        f"{markers[0]}{linestyles[0]}",
        label="LUCJ Compressed-t2 multi-stage",
        color=colors[5],
    )
    # if connectivity != "all-to-all":
    if False:
        tasks_compressed_t2_connectivity = [
            SQDEnergyTask(
                molecule_basename=molecule_basename,
                bond_distance=None,
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

        energies = [results_compressed_t2_connectivity[task]['energy'] for task in tasks_compressed_t2_connectivity]
        errors = [results_compressed_t2_connectivity[task]["error"] for task in tasks_compressed_t2_connectivity]
        spin_squares = [results_compressed_t2_connectivity[task]["spin_squared"] for task in tasks_compressed_t2_connectivity]
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
            color=colors[2],
        )

        axes[2, i].plot(
            these_n_reps,
            sci_vec_shape,
            f"{markers[0]}{linestyles[0]}",
            label="LUCJ Compressed-t2 connectivity",
            color=colors[6],
        )

    axes[0, i].set_title(f"{connectivity}/max dim-{max_dim}")
    axes[0, i].set_yscale("log")
    axes[0, i].axhline(1.6e-3, linestyle="--", color="gray")
    axes[0, i].set_ylabel("Energy error (Hartree)")
    axes[0, i].set_xlabel("Repetitions")
    axes[0, i].set_xticks(these_n_reps)

    # axes[1, i].set_ylim(0, 0.1)
    axes[1, i].set_ylabel("Spin squares")
    axes[1, i].set_xlabel("Repetitions")
    axes[1, i].set_xticks(these_n_reps)

    axes[2, i].set_ylabel("SCI subspace")
    axes[2, i].set_xlabel("Repetitions")
    axes[2, i].set_xticks(these_n_reps)

    # axes[2, 0].legend(ncol=2, )
    leg = axes[2, 1].legend(bbox_to_anchor=(0.5, -0.4), loc="upper center", ncol = 4)
    leg.set_in_layout(False)
    plt.tight_layout()
    plt.subplots_adjust(bottom = 0.16)
    

    fig.suptitle(
        f"CCSD initial parameters {molecule_name} ({nelectron}e, {norb}o)"
    )


filepath = os.path.join(
    plots_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}.pdf"
)
plt.savefig(filepath)
plt.close()


fig, axes = plt.subplots(
    1, len(connectivities), figsize=(12, 6) #, layout="constrained"
)
for i, connectivity in enumerate(connectivities):
    tasks_compressed_t2 = [
        SQDEnergyTask(
        molecule_basename=molecule_basename,
        bond_distance=None,
        lucj_params=LUCJParams(
            connectivity=connectivity,
            n_reps=n_reps,
            with_final_orbital_rotation=True,
        ),
        compressed_t2_params=CompressedT2Params(
            multi_stage_optimization=True,
            begin_reps=20,
            step=2
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
    for n_reps in these_n_reps]
    init_loss = [opt_results_compressed_t2[task]["init_loss"] for task in tasks_compressed_t2]
    final_loss = [opt_results_compressed_t2[task]["final_loss"] for task in tasks_compressed_t2]
    axes[i].plot(
        these_n_reps,
        init_loss,
        f"{markers[0]}{linestyles[0]}",
        label="init loss",
        color=colors[0],
    )
    axes[i].plot(
        these_n_reps,
        final_loss,
        f"{markers[0]}{linestyles[0]}",
        label="final loss",
        color=colors[1],
    )
    axes[i].set_title(connectivity)
    axes[i].set_ylabel("loss")
    axes[i].set_xlabel("Repetitions")
    axes[i].set_xticks(these_n_reps)
    axes[i].legend()

    fig.suptitle(
        f"Opterator loss: {molecule_name} ({nelectron}e, {norb}o)"
    )

filepath = os.path.join(
    plots_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}_loss.pdf"
)
plt.savefig(filepath)
plt.close()
    
