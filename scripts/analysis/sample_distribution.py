import itertools
import os
import pickle
from pathlib import Path
from qiskit.primitives import BitArray
import ffsim
import matplotlib.pyplot as plt

from lucj.params import LUCJParams, CompressedT2Params
from lucj.tasks.lucj_sqd_initial_params_task import LUCJSQDInitialParamsTask
from lucj.sqd_energy_task.lucj_compressed_t2_task import SQDEnergyTask
from qiskit.visualization import plot_distribution

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
entropy = 0

max_dim = 500
connectivity = "all-to-all"

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

n_reps = 2

task_compressed_t2 = SQDEnergyTask(
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

prop_cycle = plt.rcParams["axes.prop_cycle"]
colors = prop_cycle.by_key()["color"]


sample_filename_lucj = DATA_ROOT / task_lucj_full.operatorpath / "sample.pickle"
sample_filename_compressed_t2 = DATA_ROOT / task_compressed_t2.operatorpath / "sample.pickle"

with open(sample_filename_lucj, "rb") as f:
    sample_filename_lucj = pickle.load(f)

with open(sample_filename_compressed_t2, "rb") as f:
    sample_filename_compressed_t2 = pickle.load(f)

legend = ["LUCJ full", "LUCJ Compressed t2"]

filepath = os.path.join(
    plots_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}_{molecule_basename}.pdf"
)

plot_distribution([sample_filename_lucj, sample_filename_compressed_t2], 
                  figsize=(30,5), legend=legend, color=[colors[1], colors[5]],
                  number_to_keep = 50,
                title="Sample Distribution", filename=filepath, bar_labels=False)