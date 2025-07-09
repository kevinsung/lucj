import ffsim.variational
import ffsim.variational.util
import ffsim
import numpy as np
from opt_einsum import contract
import scipy
import jax
import jax.numpy as jnp
from compressed_t2_gradient import double_factorized_t2_compress_heuristic

# jax.config.update("jax_enable_x64", True)


def _reshape_grad(
    core_coulomb_params: jnp.ndarray,
    orbital_rotations_log_jax_tri: jnp.ndarray,
):
    _, norb, _ = orbital_rotations_log_jax_tri.shape
    # include the diagonal element
    # print(orbital_rotations_log_jax_tri[0])
    leaf_param_real_indices = np.triu_indices(norb, k=1)
    # differ by minus for the imaginary part
    # according to webiste https://docs.jax.dev/en/latest/notebooks/autodiff_cookbook.html#complex-numbers-and-differentiation
    # We can use grad to optimize functions, like real-valued loss functions of complex parameters x,
    # by taking steps in the direction of the conjugate of grad(f)(x).

    leaf_params_real = np.real(
        np.ravel(
            [
                orbital_rotation[leaf_param_real_indices]
                for orbital_rotation in orbital_rotations_log_jax_tri
            ]
        )
    )
    # add imag part
    leaf_param_imag_indices = np.triu_indices(norb)
    leaf_params_imag = -np.imag(
        np.ravel(
            [
                orbital_rotation[leaf_param_imag_indices]
                for orbital_rotation in orbital_rotations_log_jax_tri
            ]
        )
    )
    core_coulomb_params = np.real(core_coulomb_params)
    return np.concatenate([leaf_params_real, leaf_params_imag, core_coulomb_params])

def _df_tensors_to_params(
    diag_coulomb_mats_aa: np.ndarray,
    diag_coulomb_mats_ab: np.ndarray,
    orbital_rotations: np.ndarray,
    diag_coulomb_mat_mask_aa: np.ndarray,
    diag_coulomb_mat_mask_ab: np.ndarray,
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
    # print("leaf_params_imag")
    # print(leaf_params_imag)
    # print('-----')
    core_param_indices = np.nonzero(diag_coulomb_mat_mask_aa)
    core_params_aa = np.ravel(
        [
            diag_coulomb_mat[core_param_indices]
            for diag_coulomb_mat in diag_coulomb_mats_aa
        ]
    )
    core_param_indices = np.nonzero(diag_coulomb_mat_mask_ab)
    core_params_ab = np.ravel(
        [
            diag_coulomb_mat[core_param_indices]
            for diag_coulomb_mat in diag_coulomb_mats_ab
        ]
    )
    return np.concatenate([leaf_params_real, leaf_params_imag, core_params_aa, core_params_ab])


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
    params: np.ndarray,
    n_tensors: int,
    norb: int,
    diag_coulomb_mat_mask_aa: np.ndarray,
    diag_coulomb_mat_mask_ab: np.ndarray,
):
    leaf_logs = _params_to_leaf_logs(params, n_tensors, norb)
    # print("leaf_logs")
    # print(leaf_logs)
    # print("----")
    orbital_rotations = _expm_antihermitian(leaf_logs)
    n_leaf_params = n_tensors * (norb * (norb - 1) // 2 + norb * (norb + 1) // 2)
    core_params = np.real(params[n_leaf_params:])
    param_indices = np.nonzero(diag_coulomb_mat_mask_aa)
    param_length = len(param_indices[0])
    diag_coulomb_mats_aa = np.zeros((n_tensors, norb, norb))
    for i in range(n_tensors):
        diag_coulomb_mats_aa[i][param_indices] = core_params[
            i * param_length : (i + 1) * param_length
        ]
        diag_coulomb_mats_aa[i] += diag_coulomb_mats_aa[i].T
        diag_coulomb_mats_aa[i][range(norb), range(norb)] /= 2

    core_params = np.real(params[(n_leaf_params + param_length * n_tensors) :])
    param_indices = np.nonzero(diag_coulomb_mat_mask_ab)
    param_length = len(param_indices[0])
    diag_coulomb_mats_ab = np.zeros((n_tensors, norb, norb))
    for i in range(n_tensors):
        diag_coulomb_mats_ab[i][param_indices] = core_params[
            i * param_length : (i + 1) * param_length
        ]
        diag_coulomb_mats_ab[i] += diag_coulomb_mats_ab[i].T
        diag_coulomb_mats_ab[i][range(norb), range(norb)] /= 2
    return diag_coulomb_mats_aa, diag_coulomb_mats_ab, orbital_rotations


def _make_tau_operator(t2):
    nocc, nvir = t2.shape[0], t2.shape[2]
    norb = nocc + nvir
    t2_til = t2
    t2_bar = -np.einsum("ijab->jiab", t2_til)
    t2_hat = t2_til + t2_bar
    t2_so = np.zeros((2 * norb, 2 * norb, 2 * norb, 2 * norb), dtype=complex)
    occ = [list(range(s * norb, s * norb + nocc)) for s in range(2)]
    vir = [list(range(s * norb + nocc, (s + 1) * norb)) for s in range(2)]
    t2_so[np.ix_(vir[0], occ[0], vir[0], occ[0])] = np.einsum("ijab->aibj", t2_hat)
    t2_so[np.ix_(vir[1], occ[1], vir[1], occ[1])] = np.einsum("ijab->aibj", t2_hat)
    t2_so[np.ix_(vir[0], occ[0], vir[1], occ[1])] = np.einsum("ijab->aibj", t2_til)
    t2_so[np.ix_(vir[1], occ[1], vir[0], occ[0])] = np.einsum("ijab->aibj", t2_til)
    t2_so[np.ix_(vir[0], occ[1], vir[1], occ[0])] = np.einsum("ijab->aibj", t2_bar)
    t2_so[np.ix_(vir[1], occ[0], vir[0], occ[1])] = np.einsum("ijab->aibj", t2_bar)
    t2_so = t2_so - np.einsum("prqs->rpsq", t2_so)
    t2_so = _project(t2_so)
    return t2_so


# projector on the set of 2-body anti-hermitian operators
def _project(T):
    TP = (T - np.einsum("prqs->psqr", T)) / 2.0
    TP = (TP - np.einsum("prqs->qrps", TP)) / 2.0
    TP = (TP - np.einsum("prqs->rpsq", TP).conj()) / 2.0
    return TP

def _project_jnp(T):
    TP = (T - jnp.einsum("prqs->psqr", T)) / 2.0
    TP = (TP - jnp.einsum("prqs->qrps", TP)) / 2.0
    TP = (TP - jnp.einsum("prqs->rpsq", TP).conj()) / 2.0
    return TP


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
    norb = orbital_rotations.shape[-1]
    # diag_coulomb_mats = diag_coulomb_mats.reshape(-1, norb, norb)

    # make t2_so
    t2_so = _make_tau_operator(t2)

    diag_coulomb_mats = diag_coulomb_mats.reshape(-1, norb, norb)[:n_reps]
    orbital_rotations = orbital_rotations.reshape(-1, norb, norb)[:n_reps]
    n_tensors, norb, _ = orbital_rotations.shape

    pairs_aa, pairs_ab = interaction_pairs

    # Zero out diagonal coulomb matrix entries
    if pairs_aa is None:
        diag_coulomb_mask_aa = np.ones((norb, norb), dtype=bool)
    else:
        diag_coulomb_mask_aa = np.zeros((norb, norb), dtype=bool)
        rows, cols = zip(*pairs_aa)
        diag_coulomb_mask_aa[rows, cols] = True
        diag_coulomb_mask_aa[cols, rows] = True
    if pairs_ab is None:
        diag_coulomb_mask_ab = np.ones((norb, norb), dtype=bool)
    else:
        diag_coulomb_mask_ab = np.zeros((norb, norb), dtype=bool)
        rows, cols = zip(*pairs_ab)
        diag_coulomb_mask_ab[rows, cols] = True
        diag_coulomb_mask_ab[cols, rows] = True

    diag_coulomb_mask_aa = np.triu(diag_coulomb_mask_aa)
    diag_coulomb_mask_ab = np.triu(diag_coulomb_mask_ab)

    def fun_jax(core_coulomb_params, orbital_rotations_log_tri):
        orbital_rotations_log_real_tri = jnp.real(orbital_rotations_log_tri)
        orbital_rotations_log_imag_tri = jnp.imag(orbital_rotations_log_tri)
        orbital_rotations_log_real = orbital_rotations_log_real_tri - jnp.transpose(
            orbital_rotations_log_real_tri, (0, 2, 1)
        )
        diagonal_element = jnp.stack(
            [
                jnp.diag(jnp.diag(orbital_rotation))
                for orbital_rotation in orbital_rotations_log_imag_tri
            ],
            axis=0,
        )
        orbital_rotations_log_imag = orbital_rotations_log_imag_tri + jnp.transpose(
            orbital_rotations_log_imag_tri, (0, 2, 1)
        )
        orbital_rotations_log = (
            orbital_rotations_log_real
            + 1j * orbital_rotations_log_imag
            - 1j * diagonal_element
        )

        eigs, vecs = jnp.linalg.eigh(-1j * orbital_rotations_log)
        orbital_rotations_compact = jnp.einsum(
            "tij,tj,tkj->tik", vecs, jnp.exp(1j * eigs), vecs.conj()
        )
        # print(scipy.linalg.expm(orbital_rotations_log[0]))
        # print(orbital_rotations_compact[0])
        # input()

        param_indices_aa = np.nonzero(diag_coulomb_mask_aa)
        param_length_aa = len(param_indices_aa[0])
        param_indices_ab = np.nonzero(diag_coulomb_mask_ab)
        param_length_ab = len(param_indices_ab[0])
        list_partial_diag_coulomb_mat = []
        core_params_ab = np.real(core_coulomb_params[(param_length_aa * n_tensors):])
        for i in range(n_tensors):
            diag_coulomb_mat_aa = jnp.zeros((norb, norb), complex)
            diag_coulomb_mat_aa = diag_coulomb_mat_aa.at[param_indices_aa].set(
                core_coulomb_params[i * param_length_aa : (i + 1) * param_length_aa]
            )
            diag_coulomb_mat_ab = jnp.zeros((norb, norb), complex)
            diag_coulomb_mat_ab = diag_coulomb_mat_ab.at[param_indices_ab].set(
                core_params_ab[i * param_length_ab : (i + 1) * param_length_ab]
            )
            # print("param_indices_aa")
            # print(param_indices_aa)
            # print("diag_coulomb_mat_aa")
            # print(diag_coulomb_mat_aa)
            # print("diag_coulomb_mat_ab")
            # print(diag_coulomb_mat_ab)
            diag_coulomb_mat_ab = diag_coulomb_mat_ab + diag_coulomb_mat_ab.T - jnp.diag(jnp.diag(diag_coulomb_mat_ab))
            # print("new diag_coulomb_mat_ab")
            # print(diag_coulomb_mat_ab)
            # input()
            diag_coulomb_mat = jnp.block([[diag_coulomb_mat_aa, diag_coulomb_mat_ab],
                                           [jnp.zeros((norb, norb), complex), diag_coulomb_mat_aa]])
            list_partial_diag_coulomb_mat.append(diag_coulomb_mat)
            # print(jnp.block([[diag_coulomb_mat_aa, diag_coulomb_mat_ab],
            #         [jnp.zeros((norb, norb), complex), diag_coulomb_mat_aa]]))
            # input()


        # reconstruct orbital_rotations and diagonal coulumn mats
        orbital_rotations = jnp.zeros((n_tensors, 2 * norb, 2 * norb), dtype=complex)
        orbital_rotations = orbital_rotations.at[:, :norb, :norb].set(
            orbital_rotations_compact
        )
        orbital_rotations = orbital_rotations.at[:, norb:, norb:].set(
            orbital_rotations_compact
        )
        
        list_diag_coulomb_mat = []

        for i in range(n_tensors):
            diag_coulomb_mat = jnp.zeros((2 * norb, 2 * norb, 2 * norb, 2 * norb), dtype=complex)
            indices = [[p, p, r, r] for p in range(2 * norb) for r in range(2 * norb)]
            indices = tuple(zip(*indices))
            tmp = list_partial_diag_coulomb_mat[i].ravel()
            diag_coulomb_mat = diag_coulomb_mat.at[indices].set(tmp)
            list_diag_coulomb_mat.append(4 * _project_jnp(diag_coulomb_mat * 1j))

        diag_coulomb_mats = jnp.stack(list_diag_coulomb_mat, axis=0)
        # print(diag_coulomb_mats.shape)

        reconstructed = contract(
            "mPp,mQq,mSs,mRr,mprqs->PRQS",
            orbital_rotations,
            orbital_rotations,
            jnp.conj(orbital_rotations),
            jnp.conj(orbital_rotations),
            diag_coulomb_mats,
            optimize="greedy",
        )
        diff = reconstructed - t2_so
        # print(orbital_rotations[0])
        # input()
        return 0.5 * jnp.sum(jnp.abs(diff) ** 2)

    # value_and_grad_func = jax.value_and_grad(fun_jax, argnums=(0, 1), holomorphic=True)
    value_and_grad_func = jax.value_and_grad(fun_jax, argnums=(0, 1))

    def fun_jac(x):
        orbital_rotations_log = _params_to_leaf_logs(x, n_tensors, norb)
        orbital_rotations_log_jax = jnp.array(orbital_rotations_log)
        mask = jnp.ones((norb, norb), dtype=bool)
        mask = jnp.triu(mask)
        orbital_rotations_log_jax_tri = orbital_rotations_log_jax * mask
        n_leaf_params = n_tensors * (norb * (norb - 1) // 2 + norb * (norb + 1) // 2)
        core_coulomb_params = jnp.array(x[n_leaf_params:] + 0j)

        val, (grad_diag_coulomb_params, grad_orbital_rotations_log_jax_tri) = (
            value_and_grad_func(core_coulomb_params, orbital_rotations_log_jax_tri)
        )
        reshaped_grad = _reshape_grad(
            grad_diag_coulomb_params, grad_orbital_rotations_log_jax_tri
        )
        # print(reshaped_grad)
        return val, reshaped_grad

    def callback(intermediate_result: scipy.optimize.OptimizeResult):
        print(f"Intermediate result: loss {intermediate_result.fun:.8f}")

    method = "L-BFGS-B"
    # method = "trust-constr"
    # method = "COBYQA"
    # method = "COBYLA"
    options = {"maxiter": 100}

    x0 = _df_tensors_to_params(
        diag_coulomb_mats,
        diag_coulomb_mats,
        orbital_rotations,
        diag_coulomb_mask_aa,
        diag_coulomb_mask_ab,
    )
    diag_coulomb_mats_converted_aa, diag_coulomb_mats_converted_ab, orbital_rotations_converted = _params_to_df_tensors(
        x0, n_tensors, norb, diag_coulomb_mask_aa, diag_coulomb_mask_ab
    )

    init_loss, _ = fun_jac(x0)
    print(f"init loss: {init_loss}")
    # assert(init_loss < 1e-20)

    result = scipy.optimize.minimize(
        fun_jac,
        x0,
        # result.x,
        method=method,
        jac=True,
        callback=callback,
        options=options,
    )

    diag_coulomb_mats_aa, diag_coulomb_mats_ab, orbital_rotations = (
        _params_to_df_tensors(
            result.x,
            n_tensors,
            norb,
            diag_coulomb_mask_aa,
            diag_coulomb_mask_ab,
        )
    )
    final_loss, _ = fun_jac(result.x)
    print(f"final loss with gradient: {final_loss}")
    # stack here without dealing with interaction constraint for Jaa, Jab
    diag_coulomb_mats = np.stack([diag_coulomb_mats_aa, diag_coulomb_mats_ab], axis=1)
    return diag_coulomb_mats, orbital_rotations, init_loss, final_loss

def double_factorized_t2_pre_compress(
    t2: np.ndarray,
    diag_coulomb_mats: np.ndarray,
    orbital_rotations: np.ndarray,
    *,
    nocc: int,
    tol: float = 1e-8,
    n_reps: int | None = None,
    interaction_pairs: tuple[list[tuple[int, int]] | None, list[tuple[int, int]] | None]
    | None = None,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    for n in range(22, n_reps, -2):
        diag_coulomb_mats, orbital_rotations, init_loss, final_loss = (
            double_factorized_t2_compress_heuristic(
                t2,
                diag_coulomb_mats,
                orbital_rotations,
                n_reps=n,
                interaction_pairs=interaction_pairs,
                nocc=nocc,
            )
        )
        diag_coulomb_mats, orbital_rotations = double_factorized_t2_compress_heuristic(t2,
            diag_coulomb_mats,
            orbital_rotations,
            n_reps=n_reps,
            interaction_pairs=interaction_pairs,
            nocc=nocc,)
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
) -> tuple[ffsim.UCJOpSpinBalanced, float, float]:
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
        diag_coulomb_mats, orbital_rotations = double_factorized_t2_pre_compress(
                t2,
                diag_coulomb_mats,
                orbital_rotations,
                n_reps=n_reps,
                interaction_pairs=interaction_pairs,
                nocc=nocc,
            )
        diag_coulomb_mats, orbital_rotations, init_loss, final_loss = (
            double_factorized_t2_compress(
                t2,
                diag_coulomb_mats,
                orbital_rotations,
                n_reps=n_reps,
                interaction_pairs=interaction_pairs,
                nocc=nocc,
            )
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

    return (
        ffsim.UCJOpSpinBalanced(
            diag_coulomb_mats=diag_coulomb_mats,
            orbital_rotations=orbital_rotations,
            final_orbital_rotation=final_orbital_rotation,
        ),
        init_loss,
        final_loss,
    )


# Get molecular data and molecular Hamiltonian
# Build N2 molecule
# molecule_name = "n2"
# basis = "6-31g"
# nelectron, norb = 10, 16

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
pairs_aa, pairs_ab = interaction_pairs_spin_balanced("all-to-all", norb)

n_reps = 30
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
    "final_loss": final_loss,
}

import pickle

with open(f"scratch/n2_sto-6g_10e8o_{n_reps}_gradient_multi_stage.pickle", "wb") as f:
    pickle.dump(data, f)

# Compute final state
# final_state = ffsim.apply_unitary(reference_state, operator, norb=norb, nelec=nelec)
