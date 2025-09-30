# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import logging
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from lucj.params import LUCJParams, CompressedT2Params
from lucj.operator_task.lucj_compressed_t2_task import (
    LUCJCompressedT2Task,
)
from molecules_catalog.util import load_molecular_data
from ffsim.variational.util import interaction_pairs_spin_balanced
import ffsim
from opt_einsum import contract
import json
import itertools
import math

filename = f"logs/{os.path.splitext(os.path.relpath(__file__))[0]}.log"
os.makedirs(os.path.dirname(filename), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S %z",
    filename=filename,
)

DATA_ROOT = Path(os.environ.get("LUCJ_DATA_ROOT", "data"))
# DATA_DIR = DATA_ROOT / os.path.basename(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = DATA_ROOT
MOLECULES_CATALOG_DIR = Path(os.environ.get("MOLECULES_CATALOG_DIR"))

molecule_name = "n2"
basis = "6-31g"
nelectron, norb = 10, 16
molecule_basename = f"{molecule_name}_{basis}_{nelectron}e{norb}o"

plots_dir = os.path.join("paper", molecule_basename)
os.makedirs(plots_dir, exist_ok=True)

bond_distance_range = [1.2, 2.4]

connectivities = [
    "all-to-all",
    "heavy-hex",
]
n_reps_range = list(range(1, 11, 1))

with open("scripts/paper/color.json", "r") as file:
    colors = json.load(file)


def loss(stacked_diag_coulomb_mats, orbital_rotations, t2, pairs_aa, pairs_ab):
    nocc, _, _, _ = t2.shape
    diag_coulomb_mats = np.zeros(orbital_rotations.shape)
    if pairs_aa:
        rows, cols = zip(*pairs_aa)
        diag_coulomb_mats[:, rows, cols] = stacked_diag_coulomb_mats[:, 0, rows, cols]
        diag_coulomb_mats[:, cols, rows] = stacked_diag_coulomb_mats[:, 0, rows, cols]
        rows, cols = zip(*pairs_ab)
        diag_coulomb_mats[:, rows, cols] = stacked_diag_coulomb_mats[:, 1, rows, cols]
        diag_coulomb_mats[:, cols, rows] = stacked_diag_coulomb_mats[:, 1, rows, cols]
    else:
        diag_coulomb_mats = np.unstack(stacked_diag_coulomb_mats, axis=1)[0]

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


fig, axes = plt.subplots(
    4,
    len(bond_distance_range) * len(connectivities),
    figsize=(8, 8),  # , layout="constrained"
)

markers = ["o", "s", "v", "D", "p", "*", "P", "X"]
linestyles = ["--", ":"]

for i, (d, connectivity) in enumerate(
    itertools.product(bond_distance_range, connectivities)
):
    list_average_norm_reference_diagonal_coulumb = []
    list_average_norm_compressed_diagonal_coulumb = []
    list_average_norm_compressed_diagonal_coulumb_reg = []
    list_average_diff_norm_compressed_diagonal_coulumb = []
    list_average_diff_norm_compressed_diagonal_coulumb_reg = []
    list_average_diff_norm_compressed_orbital_rotation = []
    list_average_diff_norm_compressed_orbital_rotation_reg = []
    list_loss_truncation = []
    list_loss_compression = []
    list_loss_compression_reg = []

    # Get molecular data and molecular Hamiltonian
    mol_data = load_molecular_data(
        f"{molecule_basename}_d-{d:.5f}",
        molecules_catalog_dir=MOLECULES_CATALOG_DIR,
    )
    norb = mol_data.norb
    nelec = mol_data.nelec
    mol_hamiltonian = mol_data.hamiltonian

    # Initialize Hamiltonian, initial state, and LUCJ parameters
    hamiltonian = ffsim.linear_operator(mol_hamiltonian, norb=norb, nelec=nelec)
    reference_state = ffsim.hartree_fock_state(norb, nelec)
    pairs_aa, pairs_ab = interaction_pairs_spin_balanced(connectivity, norb)

    operator = ffsim.UCJOpSpinBalanced.from_t_amplitudes(
        mol_data.ccsd_t2,
        n_reps=None,
        t1=mol_data.ccsd_t1,
        interaction_pairs=(pairs_aa, pairs_ab),
    )
    diag_coulomb_mats_reference_full = operator.diag_coulomb_mats
    orbital_rotations_reference_full = operator.orbital_rotations

    list_norm_diag_coulomb_mats_reference_full = [
        np.sum(np.abs(diag_coulomb_mats) ** 2)
        for diag_coulomb_mats in diag_coulomb_mats_reference_full
    ]
    list_norm_orbital_rotations_reference_full = [
        np.sum(np.abs(orbital_rotations) ** 2)
        for orbital_rotations in orbital_rotations_reference_full
    ]

    for n_reps in n_reps_range:
        # use CCSD to initialize parameters
        operator = ffsim.UCJOpSpinBalanced.from_t_amplitudes(
            mol_data.ccsd_t2,
            n_reps=n_reps,
            t1=mol_data.ccsd_t1,
            interaction_pairs=(pairs_aa, pairs_ab),
        )
        diag_coulomb_mats_reference = operator.diag_coulomb_mats
        orbital_rotations_reference = operator.orbital_rotations
        t2_loss = loss(
            diag_coulomb_mats_reference[:n_reps],
            orbital_rotations_reference[:n_reps],
            mol_data.ccsd_t2,
            pairs_aa, 
            pairs_ab
        )
        list_loss_truncation.append(t2_loss)

        task_compressed_t2 = LUCJCompressedT2Task(
            molecule_basename=molecule_basename,
            bond_distance=d,
            lucj_params=LUCJParams(
                connectivity=connectivity,
                n_reps=n_reps,
                with_final_orbital_rotation=True,
            ),
            compressed_t2_params=CompressedT2Params(
                multi_stage_optimization=True, begin_reps=20, step=2
            ),
            regularization=False,
        )

        operator_filename = DATA_DIR / task_compressed_t2.dirpath / "operator.npz"
        operator = np.load(operator_filename)
        diag_coulomb_mats_compressed_t2 = operator["diag_coulomb_mats"] 
        orbital_rotations_compressed_t2 = operator["orbital_rotations"]
        t2_loss = loss(
            diag_coulomb_mats_compressed_t2,
            orbital_rotations_compressed_t2,
            mol_data.ccsd_t2,
            pairs_aa, 
            pairs_ab
        )
        list_loss_compression.append(t2_loss)

        # diag_coulomb_mats_compressed_t2 = diag_coulomb_mats_compressed_t2 % np.pi
        a = np.ones(diag_coulomb_mats_compressed_t2.shape)
        a = a * np.pi

        task_compressed_t2_reg = LUCJCompressedT2Task(
            molecule_basename=molecule_basename,
            bond_distance=d,
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
            regularization_factor=1e-3,
        )
        operator_filename = DATA_DIR / task_compressed_t2_reg.dirpath / "operator.npz"
        operator = np.load(operator_filename)
        orbital_rotations_compressed_t2_reg = operator["orbital_rotations"]
        diag_coulomb_mats_compressed_t2_reg = operator["diag_coulomb_mats"]
        opt_data_filename = (
            DATA_DIR / task_compressed_t2_reg.dirpath / "opt_data.pickle"
        )

        t2_loss = loss(
            diag_coulomb_mats_compressed_t2_reg,
            orbital_rotations_compressed_t2_reg,
            mol_data.ccsd_t2,
            pairs_aa, 
            pairs_ab
        )
        list_loss_compression_reg.append(t2_loss)

        # print(diag_coulomb_mats_compressed_t2_reg[0][0])

        # diag_coulomb_mats_compressed_t2_reg = diag_coulomb_mats_compressed_t2_reg % np.pi

        # print(diag_coulomb_mats_compressed_t2_reg[0][0])
        # input()

        norm_reference_diagonal_coulumb = []
        norm_compressed_diagonal_coulumb = []
        norm_compressed_diagonal_coulumb_reg = []
        diff_norm_compressed_diagonal_coulumb = []
        diff_norm_compressed_diagonal_coulumb_reg = []

        norm_reference_orbital_rotation = []
        norm_compressed_orbital_rotation = []
        norm_compressed_orbital_rotation_reg = []
        diff_norm_compressed_orbital_rotation = []
        diff_norm_compressed_orbital_rotation_reg = []

        for layer in range(n_reps):
            diff_diag_coulomb_mats = (
                diag_coulomb_mats_compressed_t2[layer]
                - diag_coulomb_mats_reference[layer]
            )
            diff_orbital_rotations = (
                orbital_rotations_compressed_t2[layer]
                - orbital_rotations_reference[layer]
            )

            norm_reference_diagonal_coulumb.append(
                np.sqrt(np.sum(np.abs(diag_coulomb_mats_reference[layer]) ** 2))
            )
            norm_compressed_diagonal_coulumb.append(
                np.sqrt(np.sum(np.abs(diag_coulomb_mats_compressed_t2[layer]) ** 2))
            )
            diff_norm_compressed_diagonal_coulumb.append(
                np.sqrt(np.sum(np.abs(diff_diag_coulomb_mats) ** 2))
            )

            norm_reference_orbital_rotation.append(
                np.sqrt(np.sum(np.abs(orbital_rotations_reference[layer]) ** 2))
            )
            norm_compressed_orbital_rotation.append(
                np.sqrt(np.sum(np.abs(orbital_rotations_compressed_t2[layer]) ** 2))
            )
            diff_norm_compressed_orbital_rotation.append(
                np.sqrt(np.sum(np.abs(diff_orbital_rotations) ** 2))
            )

            diff_diag_coulomb_mats_reg = (
                diag_coulomb_mats_compressed_t2_reg[layer]
                - diag_coulomb_mats_reference[layer]
            )
            diff_orbital_rotations_reg = (
                orbital_rotations_compressed_t2_reg[layer]
                - orbital_rotations_reference[layer]
            )

            norm_compressed_diagonal_coulumb_reg.append(
                np.sqrt(np.sum(np.abs(diag_coulomb_mats_compressed_t2_reg[layer]) ** 2))
            )
            diff_norm_compressed_diagonal_coulumb_reg.append(
                np.sqrt(np.sum(np.abs(diff_diag_coulomb_mats_reg) ** 2))
            )

            norm_compressed_orbital_rotation_reg.append(
                np.sqrt(np.sum(np.abs(orbital_rotations_compressed_t2_reg[layer]) ** 2))
            )
            diff_norm_compressed_orbital_rotation_reg.append(
                np.sqrt(np.sum(np.abs(diff_orbital_rotations_reg) ** 2))
            )

        list_average_norm_reference_diagonal_coulumb.append(
            np.average(norm_reference_diagonal_coulumb)
        )
        list_average_norm_compressed_diagonal_coulumb.append(
            np.average(norm_compressed_diagonal_coulumb)
        )
        list_average_diff_norm_compressed_diagonal_coulumb.append(
            np.average(diff_norm_compressed_diagonal_coulumb)
        )
        list_average_diff_norm_compressed_orbital_rotation.append(
            np.average(diff_norm_compressed_orbital_rotation)
        )

        list_average_norm_compressed_diagonal_coulumb_reg.append(
            np.average(norm_compressed_diagonal_coulumb_reg)
        )
        list_average_diff_norm_compressed_diagonal_coulumb_reg.append(
            np.average(diff_norm_compressed_diagonal_coulumb_reg)
        )
        list_average_diff_norm_compressed_orbital_rotation_reg.append(
            np.average(diff_norm_compressed_orbital_rotation_reg)
        )

    # diag coulumn norm
    axes[0, i].axhline(
        np.average(list_norm_diag_coulomb_mats_reference_full),
        # results_random[task_random_bit_string]["error"],
        linestyle="--",
        label="LUCJ-full",
        color=colors["lucj_full"],
    )

    axes[0, i].plot(
        n_reps_range,
        list_average_norm_reference_diagonal_coulumb,
        f"{markers[0]}{linestyles[0]}",
        label="LUCJ-truncated",
        color=colors["lucj_truncated"],
    )

    axes[0, i].plot(
        n_reps_range,
        list_average_norm_compressed_diagonal_coulumb,
        f"{markers[0]}{linestyles[0]}",
        label="LUCJ-compressed",
        color=colors["lucj_compressed"],
    )

    axes[0, i].plot(
        n_reps_range,
        list_average_norm_compressed_diagonal_coulumb_reg,
        f"{markers[0]}{linestyles[0]}",
        label="LUCJ-compressed-reg",
        color=colors["lucj_compressed_quimb2"],
    )

    # diag coulumn norm diff
    axes[1, i].plot(
        n_reps_range,
        list_average_diff_norm_compressed_diagonal_coulumb,
        f"{markers[0]}{linestyles[0]}",
        label="LUCJ-compressed",
        color=colors["lucj_compressed"],
    )

    axes[1, i].plot(
        n_reps_range,
        list_average_diff_norm_compressed_diagonal_coulumb_reg,
        f"{markers[0]}{linestyles[0]}",
        label="LUCJ-compressed-reg",
        color=colors["lucj_compressed_quimb2"],
    )

    # diag coulumn norm diff
    axes[2, i].plot(
        n_reps_range,
        list_average_diff_norm_compressed_orbital_rotation,
        f"{markers[0]}{linestyles[0]}",
        label="LUCJ-compressed",
        color=colors["lucj_compressed"],
    )

    axes[2, i].plot(
        n_reps_range,
        list_average_diff_norm_compressed_orbital_rotation_reg,
        f"{markers[0]}{linestyles[0]}",
        label="LUCJ-compressed-reg",
        color=colors["lucj_compressed_quimb2"],
    )

    # loss
    axes[3, i].plot(
        n_reps_range,
        list_loss_truncation,
        f"{markers[0]}{linestyles[0]}",
        label="LUCJ truncated",
        color=colors["lucj_truncated"],
    )
    
    axes[3, i].plot(
        n_reps_range,
        list_loss_compression,
        f"{markers[0]}{linestyles[0]}",
        label="LUCJ-compressed",
        color=colors["lucj_compressed"],
    )

    axes[3, i].plot(
        n_reps_range,
        list_loss_compression_reg,
        f"{markers[0]}{linestyles[0]}",
        label="LUCJ-compressed-reg",
        color=colors["lucj_compressed_quimb2"],
    )


    # axes[0, i].set_title(f"R={d} Å")
    axes[0, i].set_title(f"R={d} Å, {connectivity} \n diag coulomb, norm", fontsize="small", loc="left")
    axes[0, i].set_xlabel("Repetitions")
    axes[0, i].set_xticks(n_reps_range)
    axes[0, i].set_ylim(0, 15)

    axes[1, i].set_title("diag coulomb, diff norm", fontsize="small", loc="left")
    axes[1, i].set_xlabel("Repetitions")
    axes[1, i].set_xticks(n_reps_range)
    axes[1, i].set_ylim(0, 15)

    axes[2, i].set_title("orb rot, diff norm", fontsize="small", loc="left")
    axes[2, i].set_xlabel("Repetitions")
    axes[2, i].set_xticks(n_reps_range)
    axes[2, i].set_ylim(0, 7)

    axes[3, i].set_title("loss", fontsize="small", loc="left")
    axes[3, i].set_xlabel("Repetitions")
    axes[3, i].set_xticks(n_reps_range)
    # axes[3, i].set_ylim(0, 2)


    # axes[row_sci_vec_dim, 0].legend(ncol=2, )
    leg = axes[3, 0].legend(bbox_to_anchor=(0.44, -0.4), loc="upper left", ncol=3)
    leg.set_in_layout(False)
    plt.tight_layout()
    # plt.subplots_adjust(top=0.9,left=0.06,bottom=0.15)
    plt.subplots_adjust(bottom=0.1, top=0.9)


    fig.suptitle(f"Operator norm: N$_2$/6-31G ({nelectron}e, {norb}o)")
filepath = os.path.join(
    plots_dir,
    f"{os.path.splitext(os.path.basename(__file__))[0]}.pdf",
)
plt.savefig(filepath)
plt.close()
