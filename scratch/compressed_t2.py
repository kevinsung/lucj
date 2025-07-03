import ffsim.variational
import ffsim.variational.util
import ffsim
import numpy as np
from opt_einsum import contract
import scipy


def _df_tensors_to_params(
    diag_coulomb_mats: np.ndarray,
    orbital_rotations: np.ndarray,
    diag_coulomb_mat_mask: np.ndarray,
):
    _, norb, _ = orbital_rotations.shape
    leaf_logs = [scipy.linalg.logm(mat) for mat in orbital_rotations]
    # include the diagonal element
    leaf_param_real_indices = np.triu_indices(norb, k=1)
    leaf_params_real = np.real(
        np.ravel([leaf_log[leaf_param_real_indices] for leaf_log in leaf_logs])
    )
    # add imag part
    leaf_param_imag_indices = np.triu_indices(norb)
    leaf_params_imag = np.imag(
        np.ravel([leaf_log[leaf_param_imag_indices] for leaf_log in leaf_logs])
    )
    core_param_indices = np.nonzero(diag_coulomb_mat_mask)
    core_params = np.ravel(
        [diag_coulomb_mat[core_param_indices] for diag_coulomb_mat in diag_coulomb_mats]
    )
    return np.concatenate([leaf_params_real, leaf_params_imag, core_params])


def _params_to_leaf_logs(params: np.ndarray, n_tensors: int, norb: int):
    leaf_imag_logs = np.zeros((n_tensors, norb, norb), dtype="complex")
    leaf_logs = np.zeros((n_tensors, norb, norb), dtype="complex")
    # reconstruct the real part
    triu_indices = np.triu_indices(norb, k=1)
    param_length = len(triu_indices[0])
    for i in range(n_tensors):
        leaf_logs[i][triu_indices] = params[i * param_length : (i + 1) * param_length]
        leaf_logs[i] -= leaf_logs[i].T
    # reconstruct the imag part
    triu_indices = np.triu_indices(norb)
    real_begin_index = param_length * n_tensors
    param_length = len(triu_indices[0])
    for i in range(n_tensors):
        leaf_imag_logs[i][triu_indices] = (
            1j
            * params[
                i * param_length + real_begin_index : (i + 1) * param_length
                + real_begin_index
            ]
        )
        leaf_imag_logs_transpose = leaf_imag_logs[i].T
        # keep the diagonal element
        diagonal_element = np.diag(np.diag(leaf_imag_logs_transpose))
        leaf_imag_logs[i] += leaf_imag_logs_transpose
        leaf_imag_logs[i] -= diagonal_element
    leaf_logs += leaf_imag_logs
    return leaf_logs


def _expm_antihermitian(mats: np.ndarray) -> np.ndarray:
    eigs, vecs = np.linalg.eigh(-1j * mats)
    return np.einsum("tij,tj,tkj->tik", vecs, np.exp(1j * eigs), vecs.conj())


def _params_to_df_tensors(
    params: np.ndarray, n_tensors: int, norb: int, diag_coulomb_mat_mask: np.ndarray
):
    leaf_logs = _params_to_leaf_logs(params, n_tensors, norb)
    orbital_rotations = _expm_antihermitian(leaf_logs)
    n_leaf_params = n_tensors * (norb * (norb - 1) // 2 + norb * (norb + 1) // 2)
    core_params = np.real(params[n_leaf_params:])
    param_indices = np.nonzero(diag_coulomb_mat_mask)
    param_length = len(param_indices[0])
    diag_coulomb_mats = np.zeros((n_tensors, norb, norb))
    for i in range(n_tensors):
        diag_coulomb_mats[i][param_indices] = core_params[
            i * param_length : (i + 1) * param_length
        ]
        diag_coulomb_mats[i] += diag_coulomb_mats[i].T
        diag_coulomb_mats[i][range(norb), range(norb)] /= 2
    return diag_coulomb_mats, orbital_rotations


def double_factorized_t2_compress(
    t2: np.ndarray,
    diag_coulomb_mats: np.ndarray,
    orbital_rotations: np.ndarray,
    *,
    nocc: int,
    tol: float = 1e-8,
    n_reps: int | None = None,
    interaction_pairs: tuple[list[tuple[int, int]] | None, list[tuple[int, int]] | None]
    | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    _, _, norb, _ = orbital_rotations.shape
    # diag_coulomb_mats = diag_coulomb_mats.reshape(-1, norb, norb)
    diag_coulomb_mats = diag_coulomb_mats.reshape(-1, norb, norb)[:n_reps]
    # not dealing with stack for now
    # diag_coulomb_mats = np.stack([diag_coulomb_mats, diag_coulomb_mats], axis=1)
    # orbital_rotations = orbital_rotations.reshape(-1, norb, norb)
    orbital_rotations = orbital_rotations.reshape(-1, norb, norb)[:n_reps]
    n_tensors, norb, _ = orbital_rotations.shape


    def fun(x):
        diag_coulomb_mats, orbital_rotations = _params_to_df_tensors(
            x, n_tensors, norb, diag_coulomb_mask
        )

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
        # print(reconstructed)

        return 0.5 * np.sum(np.abs(diff) ** 2)
    
    def fun_dagger(x):
        diag_coulomb_mats, orbital_rotations = _params_to_df_tensors(
            x, n_tensors, norb, diag_coulomb_mask
        )

        two_body_tensor = np.zeros((norb, norb, norb, norb))
        two_body_tensor[nocc:, :nocc, nocc:, :nocc] = t2.transpose(2, 0, 3, 1)
        two_body_tensor[:nocc, nocc:, :nocc, nocc:] = -t2.transpose(0, 2, 1, 3).conj()

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
            )
        )
        # diff = reconstructed - t2
        diff = reconstructed - two_body_tensor# try t-t_dagger
        # print(reconstructed)

        return 0.5 * np.sum(np.abs(diff) ** 2)

    def callback(intermediate_result: scipy.optimize.OptimizeResult):
        print(f"Intermediate result: loss {intermediate_result.fun:.8f}")

    method = "L-BFGS-B"
    # method = "trust-constr"
    # method = "COBYQA"
    # method = "COBYLA"
    options = {"maxiter": 100}

    pairs_aa, pairs_ab = interaction_pairs

    # Zero out diagonal coulomb matrix entries
    pairs = []
    if pairs_aa is not None:
        pairs += pairs_aa
    if pairs_ab is not None:
        pairs += pairs_ab
    if not pairs:
        diag_coulomb_mask = np.ones((norb, norb), dtype=bool)
    else:
        diag_coulomb_mask = np.zeros((norb, norb), dtype=bool)
        rows, cols = zip(*pairs)
        diag_coulomb_mask[rows, cols] = True
        diag_coulomb_mask[cols, rows] = True

    diag_coulomb_mask = np.triu(diag_coulomb_mask)

    # print(orbital_rotations.shape)

    x0 = _df_tensors_to_params(diag_coulomb_mats, orbital_rotations, diag_coulomb_mask)
    # print(f"initial val: {fun(x0)}")
    # print(orbital_rotations[0])
    diag_coulomb_mats_converted, orbital_rotations_converted = _params_to_df_tensors(
        x0, n_tensors, norb, diag_coulomb_mask
    )
    # print(np.allclose(diag_coulomb_mats, diag_coulomb_mats_converted))
    # print(np.allclose(orbital_rotations_converted, orbital_rotations))
    init_loss = fun(x0)
    result = scipy.optimize.minimize(
        fun,
        x0,
        method=method,
        jac=False,
        # callback=callback,
        options=options,
        # fun, x0, method=method, jac=True, callback=callback, options=options
        # fun, x0, method=method
    )
    # print(all(result.x == x0))
    # print(fun(result.x))
    diag_coulomb_mats, orbital_rotations = _params_to_df_tensors(
        result.x, n_tensors, norb, diag_coulomb_mask
    )
    final_loss = fun(result.x)
    # stack here without dealing with interaction constraint for Jaa, Jab
    diag_coulomb_mats = np.stack([diag_coulomb_mats, diag_coulomb_mats], axis=1)
    return diag_coulomb_mats, orbital_rotations, init_loss, final_loss


def from_t_amplitudes_compressed(
    t2: np.ndarray,
    *,
    t1: np.ndarray | None = None,
    n_reps: int | None = None,
    interaction_pairs: tuple[list[tuple[int, int]] | None, list[tuple[int, int]] | None]
    | None = None,
    tol: float = 1e-8,
    optimize=False,
) -> ffsim.UCJOpSpinBalanced:
    if interaction_pairs is None:
        interaction_pairs = (None, None)
    pairs_aa, pairs_ab = interaction_pairs

    nocc, _, nvrt, _ = t2.shape
    norb = nocc + nvrt
    init_loss, final_loss = 0, 0
    diag_coulomb_mats, orbital_rotations = ffsim.linalg.double_factorized_t2(
        t2, tol=tol
    )
    if optimize:
        diag_coulomb_mats, orbital_rotations, init_loss, final_loss = double_factorized_t2_compress(
            t2,
            diag_coulomb_mats,
            orbital_rotations,
            n_reps=n_reps,
            interaction_pairs=interaction_pairs,
            nocc=nocc,
        )
    else:
        diag_coulomb_mats = diag_coulomb_mats.reshape(-1, norb, norb)[:n_reps]
        diag_coulomb_mats = np.stack([diag_coulomb_mats, diag_coulomb_mats], axis=1)
        orbital_rotations = orbital_rotations.reshape(-1, norb, norb)[:n_reps]

        n_vecs, _, _, _ = diag_coulomb_mats.shape
        if n_reps is not None and n_vecs < n_reps:
            # Pad with no-ops to the requested number of repetitions
            diag_coulomb_mats = np.concatenate(
                [diag_coulomb_mats, np.zeros((n_reps - n_vecs, 2, norb, norb))]
            )
            eye = np.eye(norb)
            orbital_rotations = np.concatenate(
                [orbital_rotations, np.stack([eye for _ in range(n_reps - n_vecs)])]
            )

    # Zero out diagonal coulomb matrix entries if requested
    if pairs_aa is not None:
        mask = np.zeros((norb, norb), dtype=bool)
        rows, cols = zip(*pairs_aa)
        mask[rows, cols] = True
        mask[cols, rows] = True
        diag_coulomb_mats[:, 0] *= mask
    if pairs_ab is not None:
        mask = np.zeros((norb, norb), dtype=bool)
        rows, cols = zip(*pairs_ab)
        mask[rows, cols] = True
        mask[cols, rows] = True
        diag_coulomb_mats[:, 1] *= mask

    final_orbital_rotation = None
    if t1 is not None:
        final_orbital_rotation = (
            ffsim.variational.util.orbital_rotation_from_t1_amplitudes(t1)
        )

    return ffsim.UCJOpSpinBalanced(
        diag_coulomb_mats=diag_coulomb_mats,
        orbital_rotations=orbital_rotations,
        final_orbital_rotation=final_orbital_rotation,
    ), init_loss, final_loss


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

n_reps = 4
# use CCSD to initialize parameters
operator, init_loss, final_loss = from_t_amplitudes_compressed(
    mol_data.ccsd_t2,
    n_reps=n_reps,
    t1=mol_data.ccsd_t1,
    interaction_pairs=(pairs_aa, pairs_ab),
    optimize=True,
)

# Compute final state
final_state = ffsim.apply_unitary(reference_state, operator, norb=norb, nelec=nelec)

# Compute energy and other properties of final state vector
energy = np.vdot(final_state, hamiltonian @ final_state).real
error = energy - mol_data.fci_energy

data = {
    "energy": energy,
    "error": error,
    "operator": operator,
    "n_reps": operator.n_reps,
    "init_loss": init_loss, 
    "final_loss": final_loss
}

import pickle 

with open(f"scratch/n2_sto-6g_10e8o_{n_reps}.pickle", "wb") as f:
    pickle.dump(data, f)