import ffsim.variational
import ffsim.variational.util
import ffsim
import numpy as np
from opt_einsum import contract
import scipy
import jax
import jax.numpy as jnp

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
    # core_param_indices = np.nonzero(diag_coulomb_mat_mask)
    # core_params = np.real(
    #     np.ravel(
    #         [
    #             diag_coulomb_mat[core_param_indices]
    #             for diag_coulomb_mat in grad_diag_coulomb_mats
    #         ]
    #     )
    # )
    # print(leaf_params_real[: 20])
    # print(leaf_params_imag[: 20])
    # input()
    core_coulomb_params = np.real(core_coulomb_params)
    return np.concatenate([leaf_params_real, leaf_params_imag, core_coulomb_params])


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
    # print("leaf_params_imag")
    # print(leaf_params_imag)
    # print('-----')
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
    # print("leaf_logs")
    # print(leaf_logs)
    # print("----")
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
    *,
    nocc: int,
    tol: float = 1e-8,
    n_reps: int | None = None,
    interaction_pairs: tuple[list[tuple[int, int]] | None, list[tuple[int, int]] | None]
    | None = None,
    multi_stage_optimization: bool = True,
    regularization: bool = False,
    regularization_option: int = 0,
    regularization_factor: float | None = None,
    begin_reps: int | None = None,
    step: int = 2
) -> tuple[np.ndarray, np.ndarray, float, float]:
    diag_coulomb_mats, orbital_rotations = ffsim.linalg.double_factorized_t2(
        t2, tol=tol
    )
    ori_diag_coulomb_mats = diag_coulomb_mats
    _, _, norb, _ = orbital_rotations.shape
    orbital_rotations = orbital_rotations.reshape(-1, norb, norb)
    n_reps_full, norb, _ = orbital_rotations.shape
    diag_coulomb_mats = diag_coulomb_mats.reshape(-1, norb, norb)
    # not dealing with stack for now
    if not multi_stage_optimization:
        n_reps_full = n_reps
    if begin_reps is None:
        begin_reps = n_reps_full
    # print(begin_reps)
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

    # diag_coulomb_mask
    diag_coulomb_mask = np.triu(diag_coulomb_mask)
    list_init_loss = []
    list_final_loss = []
    list_reps = [i for i in range(begin_reps, n_reps, -step)] + [n_reps]
    # coefficient for regularization
    if regularization_factor is None:
        regularization_factor = 1e-4
    
    # collect params
    full_diag_coulomb_mats = diag_coulomb_mats
    full_orbital_rotations = orbital_rotations
    diag_coulomb_mats = diag_coulomb_mats[:n_reps]
    orbital_rotations = orbital_rotations[:n_reps]

    for n_tensors in list_reps:
        if n_reps < n_tensors:
            res_diag_coulomb_mats = full_diag_coulomb_mats[n_reps:n_tensors]
            res_orbital_rotations = full_orbital_rotations[n_reps:n_tensors]

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


            param_indices = np.nonzero(diag_coulomb_mask)
            param_length = len(param_indices[0])
            list_diag_coulomb_mats = []
            for i in range(n_reps):
                diag_coulomb_mat = jnp.zeros((norb, norb), complex)
                diag_coulomb_mat = diag_coulomb_mat.at[param_indices].set(
                    core_coulomb_params[
                        i * param_length : (i + 1) * param_length
                    ])
                list_diag_coulomb_mats.append(diag_coulomb_mat)
            diagonal_element = jnp.stack(
                [
                    jnp.diag(jnp.diag(diag_coulomb_mat))
                    for diag_coulomb_mat in list_diag_coulomb_mats
                ],
                axis=0,
            )

            diag_coulomb_mats_tri = jnp.stack(list_diag_coulomb_mats, axis=0)
            diag_coulomb_mats = (
                diag_coulomb_mats_tri
                + jnp.transpose(diag_coulomb_mats_tri, (0, 2, 1))
                - diagonal_element
            )
            orbital_rotations = jnp.einsum(
                "tij,tj,tkj->tik", vecs, jnp.exp(1j * eigs), vecs.conj()
            )
            if n_reps < n_tensors:
                # print(res_diag_coulomb_mats.shape)
                # print(diag_coulomb_mats.shape)
                # input()
                diag_coulomb_mats = jnp.concatenate((diag_coulomb_mats, res_diag_coulomb_mats))
                orbital_rotations = jnp.concatenate((orbital_rotations, res_orbital_rotations))
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

            # regularization term
            regularization_cost = 0
            if regularization:
                for diag_coulomb_mat in diag_coulomb_mats:
                    regularization_cost += jnp.sum(jnp.abs(diag_coulomb_mat) ** 2) 
                if regularization_option == 1:
                    for ori_diag_coulomb_mat in ori_diag_coulomb_mats:
                        regularization_cost -= jnp.sum(jnp.abs(ori_diag_coulomb_mat) ** 2) 
                if regularization_option == 2:
                    regularization_cost = 0
                    for reps in range(n_reps):
                        regularization_cost += (jnp.sum(jnp.abs(ori_diag_coulomb_mats[reps] - diag_coulomb_mats[reps]) ** 2) )

            return 0.5 * jnp.sum(jnp.abs(diff) ** 2) + regularization_factor * jnp.abs(regularization_cost)

        # value_and_grad_func = jax.value_and_grad(fun_jax, argnums=(0, 1), holomorphic=True)
        value_and_grad_func = jax.value_and_grad(fun_jax, argnums=(0, 1))

        def fun_jac(x):
            orbital_rotations_log = _params_to_leaf_logs(x, n_reps, norb)
            orbital_rotations_log_jax = jnp.array(orbital_rotations_log)
            mask = jnp.ones((norb, norb), dtype=bool)
            mask = jnp.triu(mask)
            orbital_rotations_log_jax_tri = orbital_rotations_log_jax * mask
            n_leaf_params = n_reps * (norb * (norb - 1) // 2 + norb * (norb + 1) // 2)
            core_coulomb_params = jnp.array(x[n_leaf_params:] + 0j)

            val, (grad_diag_coulomb_params, grad_orbital_rotations_log_jax_tri) = (
                value_and_grad_func(
                    core_coulomb_params, orbital_rotations_log_jax_tri
                )
            )
            reshaped_grad = _reshape_grad(
                grad_diag_coulomb_params, grad_orbital_rotations_log_jax_tri
            )
            return val, reshaped_grad

        def callback(intermediate_result: scipy.optimize.OptimizeResult):
            print(f"Intermediate result: loss {intermediate_result.fun:.8f}")

        method = "L-BFGS-B"
        # method = "trust-constr"
        # method = "COBYQA"
        # method = "COBYLA"
        options = {"maxiter": 100}


        x0 = _df_tensors_to_params(diag_coulomb_mats, orbital_rotations, diag_coulomb_mask)

        init_loss, _ = fun_jac(x0)
        list_init_loss.append(init_loss)
        result = scipy.optimize.minimize(
            fun_jac,
            x0,
            # result.x,
            method=method,
            jac=True,
            # callback=callback,
            options=options,
        )

        diag_coulomb_mats, orbital_rotations = _params_to_df_tensors(
            result.x, n_reps, norb, diag_coulomb_mask
        )
        final_loss, _ = fun_jac(result.x)
        list_final_loss.append(final_loss)
    
    # stack here without dealing with interaction constraint for Jaa, Jab
    diag_coulomb_mats = np.stack([diag_coulomb_mats, diag_coulomb_mats], axis=1)
    return diag_coulomb_mats, orbital_rotations, list_init_loss[0], list_final_loss[-1]


def from_t_amplitudes_compressed(
    t2: np.ndarray,
    *,
    t1: np.ndarray | None = None,
    n_reps: int | None = None,
    interaction_pairs: tuple[list[tuple[int, int]] | None, list[tuple[int, int]] | None]
    | None = None,
    tol: float = 1e-8,
    optimize: bool = False,
    multi_stage_optimization: bool | None = False,
    begin_reps: int | None = None,
    step: int | None = 2,
    regularization: bool = False,
    regularization_option: int = 0,
    regularization_factor: float | None = None,
) -> ffsim.UCJOpSpinBalanced:
    if interaction_pairs is None:
        interaction_pairs = (None, None)
    pairs_aa, pairs_ab = interaction_pairs

    nocc, _, nvrt, _ = t2.shape
    norb = nocc + nvrt
    init_loss, final_loss = 0, 0
    if optimize:
        diag_coulomb_mats, orbital_rotations, init_loss, final_loss = (
            double_factorized_t2_compress(
                t2,
                n_reps=n_reps,
                interaction_pairs=interaction_pairs,
                nocc=nocc,
                multi_stage_optimization=multi_stage_optimization,
                begin_reps=begin_reps,
                step=step,
                regularization = regularization,
                regularization_option = regularization_option,
                regularization_factor = regularization_factor
            )
        )
    else:
        diag_coulomb_mats, orbital_rotations = ffsim.linalg.double_factorized_t2(
            t2, tol=tol
        )
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