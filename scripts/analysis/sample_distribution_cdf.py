# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import itertools
import os
import pickle
from pathlib import Path
from qiskit.primitives import BitArray
import ffsim
import matplotlib.pyplot as plt

from lucj.params import LUCJParams, CompressedT2Params
from lucj.sqd_energy_task.lucj_compressed_t2_task_sci import SQDEnergyTask
from qiskit.visualization import plot_distribution
import functools
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt

import json

DATA_ROOT = "/media/storage/WanHsuan.Lin/"
MOLECULES_CATALOG_DIR = Path(os.environ.get("MOLECULES_CATALOG_DIR"))

molecule_name = "n2"
basis = "6-31g"
nelectron, norb = 10, 16
molecule_basename = f"{molecule_name}_{basis}_{nelectron}e{norb}o"
bond_distance = 1.2
begin_reps = 20
step = 2

half_hf_state = "0" * (norb - nelectron // 2) + "1" * (nelectron // 2)
hf_state = half_hf_state + half_hf_state

plots_dir = os.path.join("paper", molecule_basename)
os.makedirs(plots_dir, exist_ok=True)

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

connectivity = "heavy-hex"
connectivity = "all-to-all"

n_reps = 1

task_lucj_full = SQDEnergyTask(
    molecule_basename=molecule_basename,
    bond_distance=bond_distance,
    lucj_params=LUCJParams(
        connectivity=connectivity,
        n_reps=None,
        with_final_orbital_rotation=True,
    ),
    compressed_t2_params=None,
    connectivity_opt=False,
    random_op=False,
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

task_lucj_truncated = SQDEnergyTask(
    molecule_basename=molecule_basename,
    bond_distance=bond_distance,
    lucj_params=LUCJParams(
        connectivity=connectivity,
        n_reps=n_reps,
        with_final_orbital_rotation=True,
    ),
    compressed_t2_params=None,
    connectivity_opt=False,
    random_op=False,
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

task_compressed_t2 = SQDEnergyTask(
    molecule_basename=molecule_basename,
    bond_distance=bond_distance,
    lucj_params=LUCJParams(
        connectivity=connectivity,
        n_reps=n_reps,
        with_final_orbital_rotation=True,
    ),
    compressed_t2_params=CompressedT2Params(
        multi_stage_optimization=True, begin_reps=begin_reps, step=step
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

tasks = [task_lucj_full, task_lucj_truncated, task_compressed_t2]


prop_cycle = plt.rcParams["axes.prop_cycle"]
colors = prop_cycle.by_key()["color"]

samples = []

for task in tasks:
    sample_filename = DATA_ROOT / task.operatorpath / "sample.pickle"

    with open(sample_filename, "rb") as f:
        sample = pickle.load(f)
        sample = BitArray.from_counts(sample)
        sample = BitArray.get_counts(sample)

    samples.append(sample)


color_keys = ["lucj_full", "lucj_truncated", "lucj_compressed"]
legends = ["LUCJ-full", "LUCJ-truncated", "LUCJ-compressed"]

with open("scripts/paper/color.json", "r") as file:
    colors = json.load(file)


def hamming_distance(str1, str2):
    """Calculate the Hamming distance between two bit strings

    Args:
        str1 (str): First string.
        str2 (str): Second string.
    Returns:
        int: Distance between strings.
    Raises:
        VisualizationError: Strings not same length
    """
    return sum(s1 != s2 for s1, s2 in zip(str1, str2))


labels = sorted(functools.reduce(lambda x, y: x.union(y.keys()), samples, set()))
dist = []
prev_d = -1
for item in labels:
    d = hamming_distance(item, hf_state) if item != "rest" else 0
    dist.append(d)

labels = [list(x) for x in zip(*sorted(zip(dist, labels), key=lambda pair: pair[0]))][1]
dist = [list(x) for x in zip(*sorted(zip(dist, labels), key=lambda pair: pair[0]))][0]

labels_dict = OrderedDict()
all_pvalues = []

for execution in samples:
    values = []
    for key in labels:
        if key not in execution:
            labels_dict[key] = 1
            values.append(0)
        else:
            labels_dict[key] = 1
            values.append(execution[key])
    pvalues = np.array(values, dtype=float)
    pvalues /= np.sum(pvalues)
    all_pvalues.append(pvalues)


fig = plt.plot(figsize=(3, 4), layout="constrained")
# Cumulative distributions.
x = np.arange(len(all_pvalues[0]))
for value, legend, c in zip(all_pvalues, legends, color_keys):
    # print(value[:10])
    # input()
    y = value.cumsum()
    plt.plot(x, y, "-", label=legend, color=colors[c])

    # plt.ecdf(value, label=legend, color=c)

x = []
x_ticks = [0, 4, 6, 8, 10, 12]
for d in x_ticks:
    idx = dist.index(d)
    x.append(idx)

# print(x)

plt.xticks(x, x_ticks)
plt.xlabel("Hamming distance to HF state")
plt.ylabel("CDF")
plt.legend()
plt.tight_layout()
plt.subplots_adjust(left=0.15, bottom=0.1, top=0.92)
plt.yscale("log")
filepath = os.path.join(
    plots_dir,
    f"{os.path.splitext(os.path.basename(__file__))[0]}.pdf",
)
plt.title(f"CDF N$_2$/6-31G ({nelectron}e, {norb}o)")
plt.savefig(filepath)
plt.close()
