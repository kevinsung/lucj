import dataclasses
import itertools
import os
import shutil

import ffsim
import numpy as np
import scipy.sparse.linalg
from tqdm import tqdm


def interaction_pairs_spin_balanced(
    connectivity: str, norb: int
) -> tuple[list[tuple[int, int]] | None, list[tuple[int, int]] | None]:
    """Returns alpha-alpha and alpha-beta diagonal Coulomb interaction pairs."""
    if connectivity == "all-to-all":
        pairs_aa = None
        pairs_ab = None
    elif connectivity == "square":
        pairs_aa = [(p, p + 1) for p in range(norb - 1)]
        pairs_ab = [(p, p) for p in range(norb)]
    elif connectivity == "hex":
        pairs_aa = [(p, p + 1) for p in range(norb - 1)]
        pairs_ab = [(p, p) for p in range(norb) if p % 2 == 0]
    elif connectivity == "heavy-hex":
        pairs_aa = [(p, p + 1) for p in range(norb - 1)]
        pairs_ab = [(p, p) for p in range(norb) if p % 4 == 0]
    else:
        raise ValueError(f"Invalid connectivity: {connectivity}")
    return pairs_aa, pairs_ab


def interaction_pairs_spinless(
    connectivity: str, norb: int
) -> tuple[list[tuple[int, int]] | None, list[tuple[int, int]] | None]:
    """Returns diagonal Coulomb interation pairs."""
    if connectivity == "all-to-all":
        return None
    elif connectivity == "linear":
        return [(p, p + 1) for p in range(norb - 1)]
    raise ValueError(f"Invalid connectivity: {connectivity}")


def copy_data(task, src_data_dir: str, dst_data_dir: str, dirs_exist_ok: bool = False):
    """Copy task data to another directory."""
    src_dir = os.path.join(src_data_dir, task.dirpath)
    dst_dir = os.path.join(dst_data_dir, task.dirpath)
    shutil.copytree(src_dir, dst_dir, dirs_exist_ok=dirs_exist_ok)


def krylov_matrix(
    observable: scipy.sparse.linalg.LinearOperator, krylov_vecs: np.ndarray
):
    n_vecs = len(krylov_vecs)
    mat = np.zeros((n_vecs, n_vecs), dtype=complex)
    transformed_vecs = [
        observable @ vec for vec in tqdm(krylov_vecs, desc="Transformed vectors")
    ]
    for i in tqdm(range(n_vecs), desc="Diagonal entries"):
        mat[i, i] = np.vdot(krylov_vecs[i], transformed_vecs[i])
    for i, j in tqdm(
        itertools.combinations(range(n_vecs), 2),
        total=n_vecs * (n_vecs - 1) // 2,
        desc="Off-diagonal entries",
    ):
        mat[i, j] = np.vdot(krylov_vecs[i], transformed_vecs[j])
        mat[j, i] = mat[i, j].conjugate()
    return mat


def truncated_ucj_angles_op(
    ucj_angles_op: ffsim.UCJAnglesOpSpinBalanced, n_layers: int
):
    givens_ansatz_ops = [
        truncated_givens_ansatz_op(op, n_layers=n_layers)
        for op in ucj_angles_op.givens_ansatz_ops
    ]
    return dataclasses.replace(
        ucj_angles_op,
        givens_ansatz_ops=givens_ansatz_ops,
    )


def truncated_givens_ansatz_op(
    givens_ansatz_op: ffsim.GivensAnsatzOp, n_layers: int
) -> ffsim.GivensAnsatzOp:
    norb = givens_ansatz_op.norb
    assert n_layers <= norb
    assert len(givens_ansatz_op.interaction_pairs) == norb * (norb - 1) // 2

    if norb % 2:
        n_keep = n_layers * (norb // 2)
    else:
        q, r = divmod(n_layers, 2)
        n_keep = q * (norb - 1) + r * (norb // 2)

    return ffsim.GivensAnsatzOp(
        norb,
        givens_ansatz_op.interaction_pairs[:n_keep],
        thetas=givens_ansatz_op.thetas[:n_keep],
        phis=givens_ansatz_op.phis[:n_keep],
        phase_angles=givens_ansatz_op.phase_angles,
    )


def brickwork(norb: int, n_layers: int):
    for i in range(n_layers):
        for j in range(i % 2, norb - 1, 2):
            yield (j, j + 1)
