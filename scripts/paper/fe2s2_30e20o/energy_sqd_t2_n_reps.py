import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt

from lucj.params import LUCJParams, CompressedT2Params
from lucj.sqd_energy_task.lucj_compressed_t2_task import SQDEnergyTask
from lucj.sqd_energy_task.lucj_random_t2_task import RandomSQDEnergyTask
import json
from molecules_catalog.util import load_molecular_data
from ffsim.variational.util import interaction_pairs_spin_balanced
import ffsim
import numpy as np
from opt_einsum import contract
import pyscf

DATA_ROOT = Path(os.environ.get("LUCJ_DATA_ROOT", "data"))
MOLECULES_CATALOG_DIR = Path(os.environ.get("MOLECULES_CATALOG_DIR"))

molecule_name = "fe2s2"
nelectron, norb = 30, 20
molecule_basename = f"{molecule_name}_{nelectron}e{norb}o"

plots_dir = os.path.join("paper", molecule_basename)
os.makedirs(plots_dir, exist_ok=True)

connectivities = [
    "all-to-all",
    "heavy-hex",
]

n_reps_range = list(range(1, 11))

dmrg_energy = -116.6056091  # ref: https://github.com/jrm874/sqd_data_repository/blob/main/classical_reference_energies/2Fe-2S/classical_methods_energies.txt


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


print("Done loading data.")

markers = ["o", "s", "v", "D", "p", "*", "P", "X"]
linestyles = ["--", ":"]

with open("scripts/paper/color.json", "r") as file:
    colors = json.load(file)


mol_data = load_molecular_data(
    f"{molecule_basename}",
    molecules_catalog_dir=MOLECULES_CATALOG_DIR,
)
norb = mol_data.norb
nelec = mol_data.nelec

def init_loss(n_reps: int, connectivity):
    pairs_aa, pairs_ab = interaction_pairs_spin_balanced(
        connectivity, norb
    )
    c0, c1, c2 = pyscf.ci.cisd.cisdvec_to_amplitudes(
        mol_data.cisd_vec, norb, nelec[0]
    )
    assert abs(c0) > 1e-8
    t1 = c1 / c0
    t2 = c2 / c0 - np.einsum("ia,jb->ijab", t1, t1)
    operator = ffsim.UCJOpSpinBalanced.from_t_amplitudes(
        t2,
        t1=t1,
        n_reps=n_reps,
        interaction_pairs=(pairs_aa, pairs_ab)
    ) 
    diag_coulomb_mats = operator.diag_coulomb_mats
    orbital_rotations = operator.orbital_rotations
    diag_coulomb_mats = np.unstack(diag_coulomb_mats, axis=1)[0]
    nocc, _, _, _ = t2.shape
    reconstructed = (
            1j
            * contract(
                "mpq,map,mip,mbq,mjq->ijab",
                diag_coulomb_mats,
                orbital_rotations,
                orbital_rotations.conj(),
                orbital_rotations,
                orbital_rotations.conj(),
                # optimize="greedy"
            )[:nocc, :nocc, nocc:, nocc:]
        )
    diff = reconstructed - t2
    return 0.5 * np.sum(np.abs(diff) ** 2)


task = RandomSQDEnergyTask(
    molecule_basename=molecule_basename,
    bond_distance=None,
    shots=shots,
    samples_per_batch=samples_per_batch,
    n_batches=n_batches,
    energy_tol=energy_tol,
    valid_string_only=True,
    occupancies_tol=occupancies_tol,
    carryover_threshold=carryover_threshold,
    max_iterations=max_iterations,
    symmetrize_spin=symmetrize_spin,
    entropy=entropy,
    max_dim=max_dim,
)
filepath = DATA_ROOT / task.dirpath / "sqd_data.pickle"
result_random = load_data(filepath)


fig, axes = plt.subplots(
    3,
    len(connectivities),
    figsize=(10, 5),  # , layout="constrained"
)


for i, connectivity in enumerate(connectivities):
    error_avg = np.average(result_random["history_energy"]) - dmrg_energy
    error_min = np.min(result_random["history_energy"]) - dmrg_energy
    error_max = np.max(result_random["history_energy"]) - dmrg_energy

    sci_vec_shape_avg = np.average(result_random["history_sci_vec_shape"][0])
    sci_vec_shape_min = np.min(result_random["history_sci_vec_shape"][0])
    sci_vec_shape_max = np.max(result_random["history_sci_vec_shape"][0])

    axes[0, i].axhline(
        error_avg,
        linestyle="--",
        label="Rand bitstr",
        color=colors["random_bit_string"],
    )

    axes[0, i].axhline(
        error_min,
        linestyle="--",
        # label="Rand bitstr",
        color=colors["random_bit_string"],
        alpha=0.7,
    )

    axes[0, i].axhline(
        error_max,
        linestyle="--",
        # label="Rand bitstr",
        color=colors["random_bit_string"],
        alpha=0.7,
    )

    axes[0, i].axhspan(
        error_min,
        error_max,
        color=colors["random_bit_string"],
        alpha=0.5,
    )

    axes[1, i].axhline(
        sci_vec_shape_avg,
        linestyle="--",
        label="Rand bitstr",
        color=colors["random_bit_string"],
    )

    axes[1, i].axhline(
        sci_vec_shape_min,
        linestyle="--",
        # label="Rand bitstr",
        color=colors["random_bit_string"],
        alpha=0.7,
    )

    axes[1, i].axhline(
        sci_vec_shape_max,
        linestyle="--",
        # label="Rand bitstr",
        color=colors["random_bit_string"],
        alpha=0.7,
    )

    axes[1, i].axhspan(
        sci_vec_shape_min,
        sci_vec_shape_max,
        color=colors["random_bit_string"],
        alpha=0.5,
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
        regularization=False,
        regularization_option=None,
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

    filepath = DATA_ROOT / task_lucj_full.dirpath / "sqd_data.pickle"
    results = load_data(filepath)
    error_avg = np.average(results["history_energy"]) - dmrg_energy
    error_min = np.min(results["history_energy"]) - dmrg_energy
    error_max = np.max(results["history_energy"]) - dmrg_energy

    sci_vec_shape_avg = np.average(results["history_sci_vec_shape"][0])
    sci_vec_shape_min = np.min(results["history_sci_vec_shape"][0])
    sci_vec_shape_max = np.max(results["history_sci_vec_shape"][0])

    # print(error_avg)
    # print(error_min)
    # print(error_max)

    axes[0, i].axhline(
        error_avg,
        linestyle="--",
        label="LUCJ-full",
        color=colors["lucj_full"],
    )

    axes[0, i].axhline(
        error_min,
        linestyle="--",
        # label="LUCJ-full",
        color=colors["lucj_full"],
        alpha=0.7,
    )

    axes[0, i].axhline(
        error_max,
        linestyle="--",
        # label="LUCJ-full",
        color=colors["lucj_full"],
        alpha=0.7,
    )

    axes[0, i].axhspan(
        error_min,
        error_max,
        color=colors["lucj_full"],
        alpha=0.5,
    )


    axes[1, i].axhline(
        sci_vec_shape_avg,
        linestyle="--",
        label="LUCJ-full",
        color=colors["lucj_full"],
    )

    axes[1, i].axhline(
        sci_vec_shape_min,
        linestyle="--",
        # label="LUCJ-full",
        color=colors["lucj_full"],
        alpha=0.7,
    )

    axes[1, i].axhline(
        sci_vec_shape_max,
        linestyle="--",
        # label="LUCJ-full",
        color=colors["lucj_full"],
        alpha=0.7,
    )

    axes[1, i].axhspan(
        sci_vec_shape_min,
        sci_vec_shape_max,
        color=colors["lucj_full"],
        alpha=0.5,
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
                multi_stage_optimization=True, begin_reps=20, step=2
            ),
            regularization=False,
            regularization_option=None,
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
        for n_reps in n_reps_range
    ]

    tasks_truncated = [
        SQDEnergyTask(
            molecule_basename=molecule_basename,
            bond_distance=None,
            lucj_params=LUCJParams(
                connectivity=connectivity,
                n_reps=n_reps,
                with_final_orbital_rotation=True,
            ),
            random_op=False,
            compressed_t2_params=None,
            connectivity_opt=False,
            regularization=False,
            regularization_option=None,
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
        for n_reps in n_reps_range
    ]

    list_tasks = [tasks_truncated, tasks_compressed_t2]
    color_keys = ["lucj_truncated", "lucj_compressed"]
    labels = ["LUCJ-truncated", "LUCJ-compressed"]

    for tasks, color_key, label in zip(list_tasks, color_keys, labels):
        error_avg = []
        error_min = []
        error_max = []

        sci_vec_shape_avg = []
        sci_vec_shape_min = []
        sci_vec_shape_max = []

        for task in tasks:
            filepath = DATA_ROOT / task.dirpath / "sqd_data.pickle"
            results = load_data(filepath)
            energy_avg = np.average(results["history_energy"])
            error_avg.append(energy_avg - dmrg_energy)
            error_min.append(energy_avg - np.min(results["history_energy"]))
            error_max.append(np.max(results["history_energy"]) - energy_avg)

            svs_avg = np.average(results["history_sci_vec_shape"][0])
            sci_vec_shape_avg.append(svs_avg)
            sci_vec_shape_min.append(
                svs_avg - np.min(results["history_sci_vec_shape"][0])
            )
            sci_vec_shape_max.append(
                np.max(results["history_sci_vec_shape"][0]) - svs_avg
            )

            # print(task.dirpath)
            # print(results)
            # print(error_min[-1])
            # print(error_max[-1])
            # input()

            # if color_key == "lucj_compressed":
            #     print(energy_avg)
            #     print(np.min(results["history_energy"]))
            #     input()

        axes[0, i].plot(
            n_reps_range,
            error_avg,
            f"{markers[0]}{linestyles[0]}",
            label=label,
            color=colors[color_key],
        )

        axes[0, i].errorbar(
            n_reps_range,
            error_avg,
            [error_min, error_max],
            color=colors[color_key],
        )

        axes[1, i].plot(
            n_reps_range,
            sci_vec_shape_avg,
            f"{markers[0]}{linestyles[0]}",
            label=label,
            color=colors[color_key],
        )

        axes[1, i].errorbar(
            n_reps_range,
            sci_vec_shape_avg,
            [sci_vec_shape_min, sci_vec_shape_max],
            color=colors[color_key],
        )

        list_loss = [[], []]

        for n_reps in n_reps_range:
            list_loss[0].append(init_loss(n_reps, connectivity))

        for task in tasks_compressed_t2:
            filepath = DATA_ROOT / task.operatorpath / "opt_data.pickle"
            results = load_data(filepath)
            list_loss[1].append(results["final_loss"])
        
        color_keys = ["lucj_truncated", "lucj_compressed"]
        labels = ["LUCJ-truncated", "LUCJ-compressed"]
        for loss, color_key, label in zip(list_loss, color_keys, labels):
            axes[2, i].plot(
                n_reps_range,
                loss,
                f"{markers[0]}{linestyles[0]}",
                label=label,
                color=colors[color_key],
            )

    axes[0, i].set_title(connectivity)
    axes[0, i].set_yscale("log")
    # axes[0, i].axhline(1.6e-3, linestyle="--", color="gray")
    axes[0, i].set_ylabel("Energy error (Hartree)")
    axes[0, i].set_xlabel("Repetitions")
    axes[0, i].set_xticks(n_reps_range)

    axes[1, i].set_ylabel("SCI subspace")
    axes[1, i].set_xlabel("Repetitions")
    axes[1, i].set_xticks(n_reps_range)
    axes[1, i].set_yticks([2000, 4000])

    axes[2, i].set_ylabel("Operator loss")
    axes[2, i].set_xlabel("Repetitions")
    axes[2, i].set_xticks(n_reps_range)
    axes[2, i].set_yscale("log")

    leg = axes[1, 1].legend(bbox_to_anchor=(-0.2, -2.3), loc="upper center", ncol=5)
    leg.set_in_layout(False)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.16)

    fig.suptitle(f"CCSD initial parameters {molecule_name} ({nelectron}e, {norb}o)")

filepath = os.path.join(
    plots_dir,
    f"{os.path.splitext(os.path.basename(__file__))[0]}.pdf",
)
plt.savefig(filepath)
plt.close()
