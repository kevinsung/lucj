# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import os
from pathlib import Path

import ffsim
import numpy as np
import pyscf.ci
from ffsim.variational.util import interaction_pairs_spin_balanced
from pyscf.fci import cistring
from qiskit.circuit import QuantumCircuit, QuantumRegister


def extract_final_orbital_rotation(
    circuit: QuantumCircuit,
) -> tuple[QuantumCircuit, np.ndarray]:
    """Extract the final orbital rotation from a LUCJ circuit.

    This function extracts the final orbital rotation from a LUCJ circuit in order
    to reduce the number of operations required of the quantum processor.
    Instead of applying the orbital rotation on the quantum processor, we apply it to
    the Hamiltonian as part of classical processing.

    The final orbital rotation is preceded by a diagonal Coulomb operator evolution.
    Since we assume that the circuit will be measured, we remove this operation as well
    because it doesn't affect measurement outcomes.

    Args:
        circuit: The LUCJ circuit. It should not contain measurements.

    Returns:
        The modified circuit and the extracted orbital rotation.
    """
    # Convert the last UCJ operation to a quantum circuit
    ucj_instruction = circuit[-1]
    ucj_circuit = ffsim.qiskit.PRE_INIT.run(ucj_instruction.operation.definition)
    # Extract the last orbital rotation
    orb_rot_instruction = ucj_circuit.data[-1]
    assert np.allclose(
        orb_rot_instruction.operation.orbital_rotation_a,
        orb_rot_instruction.operation.orbital_rotation_b,
    )
    orbital_rotation = orb_rot_instruction.operation.orbital_rotation_a
    # Initialize a new circuit without the final UCJ operation
    truncated_circuit = QuantumCircuit.from_instructions(
        circuit[:-1], qubits=circuit.qubits
    )
    # Add instructions from the UCJ operation, ignoring the final orbital rotation and
    # the diagonal Coulomb evolution before it. The diagonal Coulomb evolution can be
    # ignored because they are diagonal, so they don't affect measurement outcomes
    for instruction in ucj_circuit[:-2]:
        truncated_circuit.append(instruction.operation, circuit.qubits)
    # Simplify the circuit
    truncated_circuit = ffsim.qiskit.PRE_INIT.run(truncated_circuit)
    return truncated_circuit, orbital_rotation


def load_molecular_data(
    molecule_basename: str,
    molecules_catalog_dir: str | bytes | os.PathLike | None = None,
):
    """Load molecular data.

    Args:
        molecule_basename: The base name of files associated with the molecule.
        molecules_catalog_dir: The path to the directory containing the
            molecules-catalog repository. If not specified, it will be read from
            the MOLECULES_CATALOG_DIR environment variable. Failing that, an error
            will be raised.
    """
    if molecules_catalog_dir is None:
        molecules_catalog_dir = os.environ.get("MOLECULES_CATALOG_DIR")
    if molecules_catalog_dir is None:
        raise RuntimeError(
            "Could not determine the path to the molecules-catalog repository. "
            "Either pass the molecules_catalog_dir argument, or set the "
            "MOLECULES_CATALOG_DIR environment variable."
        )
    filepath = (
        Path(molecules_catalog_dir)
        / "data"
        / "molecular_data"
        / f"{molecule_basename}.json.xz"
    )
    return ffsim.MolecularData.from_json(filepath, compression="lzma")


def load_ucj_op_spin_balanced(
    molecule_basename: str,
    connectivity: str,
    n_reps: int,
    params: str,
    seed: int | None = None,
    molecules_catalog_dir: str | bytes | os.PathLike | None = None,
):
    """Load a spin-balanced UCJ operator.

    Args:
        molecules_catalog_dir: Path to the directory containing the
            molecules_catalog repository.
        molecule_basename: The base name of files associated with the molecule.
        connectivity: The connectivity of the LUCJ ansatz.
        n_reps: The number of repetitions of the LUCJ anstaz.
        params: The method for setting the parameters of the ansatz.
            Options:
            - "random"
            - "ccsd"
            - "cisd"
            - "optimized-for-energy"
            - "optimized-for-qsci"
        seed: Random number generator seed. Only used when `params` is ``"random"``.
            Otherwise, ignored.
        molecules_catalog_dir: The path to the directory containing the
            molecules-catalog repository. If not specified, it will be read from
            the MOLECULES_CATALOG_DIR environment variable. Failing that, an error
            will be raised.
    """
    if molecules_catalog_dir is None:
        molecules_catalog_dir = os.environ.get("MOLECULES_CATALOG_DIR")
    if molecules_catalog_dir is None:
        raise RuntimeError(
            "Could not determine the path to the molecules-catalog repository. "
            "Either pass the molecules_catalog_dir argument, or set the "
            "MOLECULES_CATALOG_DIR environment variable."
        )
    data_dir = Path(molecules_catalog_dir) / "data"
    filepath = data_dir / "molecular_data" / f"{molecule_basename}.json.xz"
    mol_data = ffsim.MolecularData.from_json(filepath, compression="lzma")
    norb = mol_data.norb
    nelec = mol_data.nelec

    match params:
        case "random":
            rng = np.random.default_rng(seed)
            n_params = ffsim.UCJOpSpinBalanced.n_params(
                norb=norb,
                n_reps=n_reps,
                interaction_pairs=interaction_pairs_spin_balanced(
                    connectivity=connectivity, norb=norb
                ),
                with_final_orbital_rotation=True,
            )
            parameters = rng.uniform(-10, 10, size=n_params)
            return ffsim.UCJOpSpinBalanced.from_parameters(
                parameters,
                norb=norb,
                n_reps=n_reps,
                interaction_pairs=interaction_pairs_spin_balanced(
                    connectivity=connectivity, norb=norb
                ),
                with_final_orbital_rotation=True,
            )
        case "ccsd":
            return ffsim.UCJOpSpinBalanced.from_t_amplitudes(
                mol_data.ccsd_t2,
                t1=mol_data.ccsd_t1,
                n_reps=n_reps,
                interaction_pairs=interaction_pairs_spin_balanced(
                    connectivity=connectivity, norb=norb
                ),
            )
        case "cisd":
            c0, c1, c2 = pyscf.ci.cisd.cisdvec_to_amplitudes(
                mol_data.cisd_vec, norb, nelec[0]
            )
            assert abs(c0) > 1e-8
            t1 = c1 / c0
            t2 = c2 / c0 - np.einsum("ia,jb->ijab", t1, t1)
            return ffsim.UCJOpSpinBalanced.from_t_amplitudes(
                t2,
                t1=t1,
                n_reps=n_reps,
                interaction_pairs=interaction_pairs_spin_balanced(
                    connectivity=connectivity, norb=norb
                ),
            )
        case "optimized-for-energy":
            # Load parameters
            filepath = (
                data_dir
                / "lucj_params"
                / molecule_basename
                / connectivity
                / f"n_reps-{n_reps}"
                / "optimized-for-energy.npy"
            )
            with open(filepath, "rb") as f:
                parameters = np.load(f)
            # Construct UCJ operator
            return ffsim.UCJOpSpinBalanced.from_parameters(
                parameters,
                norb=norb,
                n_reps=n_reps,
                interaction_pairs=interaction_pairs_spin_balanced(
                    connectivity=connectivity, norb=norb
                ),
                with_final_orbital_rotation=True,
            )
        case "optimized-for-qsci":
            # Load parameters
            filepath = (
                data_dir
                / "lucj_params"
                / molecule_basename
                / connectivity
                / f"n_reps-{n_reps}"
                / "optimized-for-qsci.npy"
            )
            with open(filepath, "rb") as f:
                parameters = np.load(f)
            # Construct UCJ operator
            return ffsim.UCJOpSpinBalanced.from_parameters(
                parameters,
                norb=norb,
                n_reps=n_reps,
                interaction_pairs=interaction_pairs_spin_balanced(
                    connectivity=connectivity, norb=norb
                ),
                with_final_orbital_rotation=True,
            )
        case _:
            raise ValueError(f"Unrecognized parameter setting method: {params}.")


def load_lucj_circuit(
    molecule_basename: str,
    connectivity: str,
    n_reps: int,
    params: str,
    seed: int | None = None,
    measure: bool = False,
    molecules_catalog_dir: str | bytes | os.PathLike | None = None,
):
    """Load a LUCJ circuit.

    Args:
        molecules_catalog_dir: Path to the directory containing the
            molecules_catalog repository.
        molecule_basename: The base name of files associated with the molecule.
        connectivity: The connectivity of the LUCJ ansatz.
        n_reps: The number of repetitions of the LUCJ anstaz.
        params: The method for setting the parameters of the ansatz.
            Options:
            - "random"
            - "ccsd"
            - "cisd"
            - "optimized-for-energy"
            - "optimized-for-qsci"
        seed: Random number generator seed. Only used when `params` is ``"random"``.
            Otherwise, ignored.
        measure: Whether to measure the qubits at the end of the circuit.
        molecules_catalog_dir: The path to the directory containing the
            molecules-catalog repository. If not specified, it will be read from
            the MOLECULES_CATALOG_DIR environment variable. Failing that, an error
            will be raised.
    """
    if molecules_catalog_dir is None:
        molecules_catalog_dir = os.environ.get("MOLECULES_CATALOG_DIR")
    if molecules_catalog_dir is None:
        raise RuntimeError(
            "Could not determine the path to the molecules-catalog repository. "
            "Either pass the molecules_catalog_dir argument, or set the "
            "MOLECULES_CATALOG_DIR environment variable."
        )
    data_dir = Path(molecules_catalog_dir) / "data"
    filepath = data_dir / "molecular_data" / f"{molecule_basename}.json.xz"
    mol_data = ffsim.MolecularData.from_json(filepath, compression="lzma")
    norb = mol_data.norb
    nelec = mol_data.nelec

    ucj_op = load_ucj_op_spin_balanced(
        molecule_basename=molecule_basename,
        connectivity=connectivity,
        n_reps=n_reps,
        params=params,
        seed=seed,
        molecules_catalog_dir=molecules_catalog_dir,
    )

    # Construct Qiskit circuit
    qubits = QuantumRegister(2 * norb)
    circuit = QuantumCircuit(qubits)
    circuit.append(ffsim.qiskit.PrepareHartreeFockJW(norb, nelec), qubits)
    circuit.append(ffsim.qiskit.UCJOpSpinBalancedJW(ucj_op), qubits)
    if measure:
        circuit.measure_all()
    return circuit


def sci_vec_to_fci_vec(
    coeffs: np.ndarray,
    strings_a: np.ndarray,
    strings_b: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
):
    """Convert SCI coefficients and strings to an FCI vector."""
    n_alpha, n_beta = nelec
    addresses_a = cistring.strs2addr(norb, n_alpha, strings_a)
    addresses_b = cistring.strs2addr(norb, n_beta, strings_b)
    fci_vec = np.zeros(ffsim.dims(norb, nelec), dtype=coeffs.dtype)
    fci_vec[np.ix_(addresses_a, addresses_b)] = coeffs
    return fci_vec.reshape(-1)


def enforce_smoothness(scf: pyscf.scf.hf.SCF, scf_new: pyscf.scf.hf.SCF) -> None:
    s12 = pyscf.gto.mole.intor_cross("int1e_ovlp_sph", scf.mol, scf_new.mol)
    w12 = np.einsum("pX,pq,qY->XY", scf.mo_coeff, s12, scf_new.mo_coeff, optimize=True)
    w12 = np.abs(w12)
    idx = [np.argmax(w12[i, :]) for i in range(w12.shape[0])]
    mo_coeff_new = scf_new.mo_coeff[:, idx]
    w12 = np.einsum("pX,pq,qY->XY", scf.mo_coeff, s12, mo_coeff_new, optimize=True)
    for i in range(w12.shape[0]):
        if w12[i, i] < 0:
            mo_coeff_new[:, i] *= -1
    w12 = np.einsum("pX,pq,qY->XY", scf.mo_coeff, s12, mo_coeff_new, optimize=True)
    scf_new.mo_coeff = mo_coeff_new


def orbital_rotation_gate_counts(norb: int) -> dict[str, int]:
    """Gate counts for orbital rotation."""
    return {
        "xx_plus_yy": norb * (norb - 1),
        "p": 2 * norb,
    }


def diag_coulomb_evo_gate_counts(
    norb: int, connectivity: str = "all-to-all"
) -> dict[str, int]:
    """Gate counts for diagonal Coulomb evolution."""
    match connectivity:
        case "all-to-all":
            cphase = (2 * norb) * (2 * norb - 1) // 2
            phase = 2 * norb
        case "square":
            cphase = 2 * (norb - 1) + norb
            phase = 0
        case "hex":
            cphase = 2 * (norb - 1) + (norb - 1) // 2 + 1
            phase = 0
        case "heavy-hex":
            cphase = 2 * (norb - 1) + (norb - 1) // 4 + 1
            phase = 0
        case _:
            raise ValueError("Unrecognized connectivity.")
    return {
        "cphase": cphase,
        "p": phase,
    }


def slater_det_gate_counts(norb: int, nelec: tuple[int, int]) -> dict[str, int]:
    """Gate counts for Slater determinant preparation."""
    n_alpha, n_beta = nelec
    return {
        "xx_plus_yy": n_alpha * (norb - n_alpha) + n_beta * (norb - n_beta),
        "x": n_alpha + n_beta,
    }


def lucj_gate_counts(
    n_reps: int, *, norb: int, nelec: tuple[int, int], connectivity: str = "all-to-all"
) -> dict[str, int]:
    "Gate counts for LUCJ."
    slater_det_counts = slater_det_gate_counts(norb, nelec)
    orbital_rotation_counts = orbital_rotation_gate_counts(norb)
    diag_coulomb_evo_counts = diag_coulomb_evo_gate_counts(norb, connectivity)
    return {
        gate: slater_det_counts.get(gate, 0)
        + n_reps
        * (orbital_rotation_counts.get(gate, 0) + diag_coulomb_evo_counts.get(gate, 0))
        for gate in slater_det_counts.keys()
        | orbital_rotation_counts.keys()
        | diag_coulomb_evo_counts.keys()
    }
