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
from lucj.hardware_sqd_task.lucj_compressed_t2_task import HardwareSQDEnergyTask

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

plots_dir = os.path.join("plots", molecule_basename)
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

constant_factors = [None, 0.5, 1.5, 2, 2.5]


tasks = [SQDEnergyTask(
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
    t2_constant_factor=c
    )
    for c in constant_factors
]


prop_cycle = plt.rcParams["axes.prop_cycle"]
colors = prop_cycle.by_key()["color"]

samples = []

for task in tasks:
    sample_filename = (
        DATA_ROOT / task.operatorpath / "sample.pickle"
    )

    with open(sample_filename, "rb") as f:
        sample = pickle.load(f)
        sample = BitArray.from_counts(sample)
        sample = BitArray.get_counts(sample)
    
    samples.append(sample)

legend = [ f"factor-{c}" for c in constant_factors ]

color = [colors[i] for i in range(len(constant_factors))]

filepath = os.path.join(
    plots_dir,
    f"{os.path.splitext(os.path.basename(__file__))[0]}.pdf",
)

plot_distribution(
    samples,
    figsize=(30, 20),
    legend=legend,
    color=color,
    number_to_keep=20,
    sort="hamming",
    target_string=hf_state,
    title="Sample Distribution",
    filename=filepath,
    bar_labels=False,
)

# Cumulative distributions.
axs[0].ecdf(data, label="CDF")
n, bins, patches = axs[0].hist(data, n_bins, density=True, histtype="step",
                               cumulative=True, label="Cumulative histogram")