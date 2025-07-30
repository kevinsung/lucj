import itertools
import os
import pickle
from pathlib import Path
from qiskit.primitives import BitArray
import ffsim
import matplotlib.pyplot as plt

from lucj.params import LUCJParams, CompressedT2Params
from lucj.sqd_energy_task.lucj_compressed_t2_task import SQDEnergyTask
from qiskit.visualization import plot_distribution
from lucj.hardware_sqd_task.lucj_compressed_t2_task import HardwareSQDEnergyTask

DATA_ROOT = "/media/storage/WanHsuan.Lin/"
MOLECULES_CATALOG_DIR = Path(os.environ.get("MOLECULES_CATALOG_DIR"))

molecule_name = "fe2s2"
nelectron, norb = 30, 20
molecule_basename = f"{molecule_name}_{nelectron}e{norb}o"
bond_distance = None

molecule_name = "n2"
basis = "6-31g"
nelectron, norb = 10, 16
molecule_basename = f"{molecule_name}_{basis}_{nelectron}e{norb}o"
bond_distance = 1.2

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

n_reps = 1

task_compressed_t2 = SQDEnergyTask(
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


shots = 100_000
n_batches = 3
energy_tol = 1e-8
occupancies_tol = 1e-5
carryover_threshold = 1e-4
max_iterations = 20
symmetrize_spin = True
# TODO set entropy and generate seeds properly
entropy = 0

max_dim = 4000
samples_per_batch = max_dim

task_compressed_t2_hardware = HardwareSQDEnergyTask(
    molecule_basename=molecule_basename,
    bond_distance=bond_distance,
    lucj_params=LUCJParams(
        connectivity="heavy-hex",
        n_reps=n_reps,
        with_final_orbital_rotation=True,
    ),
    compressed_t2_params=CompressedT2Params(
        multi_stage_optimization=True, begin_reps=20, step=2
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
    max_dim=max_dim,
)

task_truncated_t2_hardware = HardwareSQDEnergyTask(
    molecule_basename=molecule_basename,
    bond_distance=bond_distance,
    lucj_params=LUCJParams(
        connectivity="heavy-hex",
        n_reps=n_reps,
        with_final_orbital_rotation=True,
    ),
    compressed_t2_params=None,
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


prop_cycle = plt.rcParams["axes.prop_cycle"]
colors = prop_cycle.by_key()["color"]


sample_filename_lucj = DATA_ROOT / task_lucj_full.operatorpath / "sample.pickle"
sample_filename_compressed_t2 = (
    DATA_ROOT / task_compressed_t2.operatorpath / "sample.pickle"
)
sample_filename_compressed_t2_hardware = (
    DATA_ROOT / task_compressed_t2_hardware.operatorpath / "hardware_sample.pickle"
)
sample_filename_truncated_t2_hardware = (
    DATA_ROOT / task_truncated_t2_hardware.operatorpath / "hardware_sample.pickle"
)

with open(sample_filename_lucj, "rb") as f:
    sample_lucj = pickle.load(f)
    sample_lucj = BitArray.from_counts(sample_lucj)
    sample_lucj = BitArray.get_counts(sample_lucj)

with open(sample_filename_compressed_t2, "rb") as f:
    sample_compressed_t2 = pickle.load(f)
    sample_compressed_t2 = BitArray.from_counts(sample_compressed_t2)
    sample_compressed_t2 = BitArray.get_counts(sample_compressed_t2)

with open(sample_filename_compressed_t2_hardware, "rb") as f:
    sample_compressed_t2_hardware = pickle.load(f)
    sample_compressed_t2_hardware = sample_compressed_t2_hardware.get_counts()

with open(sample_filename_truncated_t2_hardware, "rb") as f:
    sample_truncated_t2_hardware = pickle.load(f)
    sample_truncated_t2_hardware = sample_truncated_t2_hardware.get_counts()

legend = [
    # "LUCJ full",
    # "LUCJ Compressed t2",
    "LUCJ Compressed t2-hardware",
    "LUCJ truncated t2-hardware",
]
samples = [
    # sample_lucj,
    # sample_compressed_t2,
    sample_compressed_t2_hardware,
    sample_truncated_t2_hardware,
]
color = [colors[0], colors[1], colors[2], colors[3]]
color = [colors[1], colors[2], colors[3]]
color = [colors[2], colors[3]]

filepath = os.path.join(
    plots_dir,
    f"{os.path.splitext(os.path.basename(__file__))[0]}_{molecule_basename}.pdf",
)

plot_distribution(
    samples,
    figsize=(30, 20),
    legend=legend,
    color=color,
    number_to_keep=20,
    # sort="hamming",
    # target_string=hf_state,
    title="Sample Distribution",
    filename=filepath,
    bar_labels=False,
)
