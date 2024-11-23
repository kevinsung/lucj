import os
import pickle
from pathlib import Path

import ffsim
import matplotlib.pyplot as plt
import numpy as np

from lucj.params import COBYQAParams, LUCJParams
from lucj.tasks.lucj_sqd_cobyqa_task import LUCJSQDCOBYQATask

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
bond_distance_range = np.linspace(start, stop, num=round((stop - start) / step) + 1)
reference_bond_distance_range = np.linspace(
    start, stop, num=round((stop - start) / 0.05) + 1
)

connectivity = "heavy-hex"
n_reps = 1
shots = 100_000
samples_per_batch = 5000
n_batches = 3
max_davidson = 200
# TODO set entropy and generate seeds properly
entropy = None

tasks = [
    LUCJSQDCOBYQATask(
        molecule_basename=molecule_basename,
        bond_distance=d,
        lucj_params=LUCJParams(
            connectivity=connectivity,
            n_reps=n_reps,
            with_final_orbital_rotation=True,
        ),
        cobyqa_params=COBYQAParams(maxiter=10_000),
        shots=shots,
        samples_per_batch=samples_per_batch,
        n_batches=n_batches,
        max_davidson=max_davidson,
        entropy=entropy,
    )
    for d in bond_distance_range
]


mol_datas_reference: dict[float, ffsim.MolecularData] = {}
mol_datas_experiment: dict[float, ffsim.MolecularData] = {}

for d in reference_bond_distance_range:
    filepath = os.path.join(
        MOLECULES_CATALOG_DIR,
        "data",
        "molecular_data",
        f"{molecule_basename}_d-{d:.2f}.json.xz",
    )
    mol_datas_reference[d] = ffsim.MolecularData.from_json(filepath, compression="lzma")

for d in bond_distance_range:
    filepath = os.path.join(
        MOLECULES_CATALOG_DIR,
        "data",
        "molecular_data",
        f"{molecule_basename}_d-{d:.2f}.json.xz",
    )
    mol_datas_experiment[d] = ffsim.MolecularData.from_json(
        filepath, compression="lzma"
    )

hf_energies_reference = np.array(
    [mol_data.hf_energy for mol_data in mol_datas_reference.values()]
)
fci_energies_reference = np.array(
    [mol_data.fci_energy for mol_data in mol_datas_reference.values()]
)
ccsd_energies_reference = np.array(
    [mol_data.ccsd_energy for mol_data in mol_datas_reference.values()]
)
hf_energies_experiment = np.array(
    [mol_data.hf_energy for mol_data in mol_datas_experiment.values()]
)
fci_energies_experiment = np.array(
    [mol_data.fci_energy for mol_data in mol_datas_experiment.values()]
)

print("Loading data...")
results = {}
for task in tasks:
    filepath = DATA_ROOT / "lucj_sqd_cobyqa" / task.dirpath / "result.pickle"
    with open(filepath, "rb") as f:
        result = pickle.load(f)
        results[task] = result
print("Done loading data.")

markers = ["o", "s", "v", "D", "p", "*", "P", "X"]
prop_cycle = plt.rcParams["axes.prop_cycle"]
colors = prop_cycle.by_key()["color"]
alphas = [0.5, 1.0]
linestyles = ["--", ":"]

fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(9, 12), layout="constrained")

# ax0.plot(
#     reference_curves_d_range,
#     hf_energies_reference,
#     "--",
#     label="HF",
#     color="blue",
# )
ax0.plot(
    reference_bond_distance_range,
    ccsd_energies_reference,
    "--",
    label="CCSD",
    color="orange",
)
ax0.plot(
    reference_bond_distance_range,
    fci_energies_reference,
    "-",
    label="FCI",
    color="black",
)


energies = [results[task].fun for task in tasks]
errors = [
    results[task].fun - mol_datas_experiment[task.bond_distance].fci_energy
    for task in tasks
]
nits = [results[task]["nit"] for task in tasks]
ax0.plot(
    bond_distance_range,
    energies,
    f"{markers[0]}{linestyles[0]}",
    label=f"LUCJ {connectivity}, L={n_reps}",
    color=colors[0],
)
ax1.plot(
    bond_distance_range,
    errors,
    f"{markers[0]}{linestyles[0]}",
    label=f"LUCJ {connectivity}, L={n_reps}",
    color=colors[0],
)
ax2.plot(
    bond_distance_range,
    nits,
    f"{markers[0]}{linestyles[0]}",
    label=f"LUCJ {connectivity}, L={n_reps}",
    color=colors[0],
)

ax0.legend()
ax0.set_ylabel("Energy (Hartree)")
ax0.set_xlabel("Bond length (Å)")
ax1.set_yscale("log")
ax1.axhline(1.6e-3, linestyle="--", color="gray")
ax1.set_ylabel("Energy error (Hartree)")
ax1.set_xlabel("Bond length (Å)")
# ax2.set_ylim(0, 1000)
ax2.set_ylabel("Number of iterations")
ax2.set_xlabel("Bond length (Å)")
fig.suptitle(f"{molecule_basename} ({nelectron}e, {norb}o)")


filepath = os.path.join(
    plots_dir, f"{os.path.splitext(os.path.basename(__file__))[0]}.pdf"
)
plt.savefig(filepath)
print(f"Saved figure to {filepath}.")
plt.close()
