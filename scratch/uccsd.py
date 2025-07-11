import logging
import os
import pickle
from dataclasses import dataclass
from pathlib import Path

import ffsim
import numpy as np
import scipy.stats
from ffsim.variational.util import interaction_pairs_spin_balanced
from molecules_catalog.util import load_molecular_data
from lucj.params import LUCJParams
from compressed_t2_multi_stage import double_factorized_t2_compress
from opt_einsum import contract




molecule_name = "n2"
basis = "sto-6g"
nelectron, norb = 10, 8


molecule_basename = f"{molecule_name}_{basis}_{nelectron}e{norb}o"

bond_distance = 1.0

from molecules_catalog.util import load_molecular_data
from pathlib import Path
import os
from ffsim.variational.util import interaction_pairs_spin_balanced

# Get molecular data and molecular Hamiltonian
molecules_catalog_dir = Path(os.environ.get("MOLECULES_CATALOG_DIR"))

mol_data = load_molecular_data(
    f"{molecule_basename}_d-{bond_distance:.5f}",
    molecules_catalog_dir=molecules_catalog_dir,
)
norb = mol_data.norb
nelec = mol_data.nelec
mol_hamiltonian = mol_data.hamiltonian

# Initialize Hamiltonian, initial state, and LUCJ parameters
hamiltonian = ffsim.linear_operator(mol_hamiltonian, norb=norb, nelec=nelec)
reference_state = ffsim.hartree_fock_state(norb, nelec)
pairs_aa, pairs_ab = interaction_pairs_spin_balanced(
    "all-to-all", norb
)


# use CCSD to initialize parameters
nocc, _, nvrt, _ = mol_data.ccsd_t2.shape
operator_filename = "op.npz"

n_reps=2

diag_coulomb_mats, orbital_rotations, _, _ = (
        double_factorized_t2_compress(
            mol_data.ccsd_t2,
            nocc=nocc,
            n_reps=n_reps,
            interaction_pairs=(pairs_aa, pairs_ab),
            multi_stage_optimization=True,
        )
    )
np.savez(operator_filename, diag_coulomb_mats=diag_coulomb_mats, orbital_rotations=orbital_rotations)

t2_reconstructed = (
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
logging.info("UCCSDOpRestricted\n")
operator = ffsim.UCCSDOpRestricted(t1=mol_data.ccsd_t1, t2=t2_reconstructed)

# Compute energy and other properties of final state vector
logging.info("Compute final state\n")
final_state = ffsim.apply_unitary(reference_state, operator, norb=norb, nelec=nelec)

energy = np.vdot(final_state, hamiltonian @ final_state).real
error = energy - mol_data.fci_energy

spin_squared = ffsim.spin_square(
    final_state, norb=mol_data.norb, nelec=mol_data.nelec
)
probs = np.abs(final_state) ** 2
entropy = scipy.stats.entropy(probs)



