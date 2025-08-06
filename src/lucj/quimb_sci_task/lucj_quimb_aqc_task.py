import dataclasses
import logging
import os
import pickle
import timeit
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import ffsim
import numpy as np
import scipy.optimize
from ffsim.variational.util import (
    interaction_pairs_spin_balanced,
    orbital_rotation_to_parameters,
)
from molecules_catalog.util import load_molecular_data
from qiskit.primitives import BitArray
from qiskit_addon_sqd.fermion import SCIResult, diagonalize_fermionic_hamiltonian
from lucj.tasks.lucj_compressed_t2_task_ffsim.compressed_t2 import from_t_amplitudes_compressed
from lucj.params import LBFGSBParams, LUCJParams, CompressedT2Params
from qiskit_addon_dice_solver import solve_sci_batch, solve_hci
from qiskit.circuit import QuantumCircuit, QuantumRegister
import quimb.tensor
from qiskit_quimb import quimb_circuit

from qiskit_addon_aqc_tensor.objective import MaximizeStateFidelity
from qiskit_aer import AerSimulator
from qiskit_addon_aqc_tensor import generate_ansatz_from_circuit

import pyscf.ci
import pyscf.fci
import autoray
import pyscf
import jax
from ffsim.states.bitstring import convert_bitstring_type
import jax.numpy as jnp

# remove later
filename = f"logs/{os.path.splitext(os.path.relpath(__file__))[0]}.log"
os.makedirs(os.path.dirname(filename), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S %z",
    filename=filename,
)
####

logger = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class LUCJQuimbTask:
    molecule_basename: str
    bond_distance: float | None
    lucj_params: LUCJParams
    lbfgsb_params: LBFGSBParams
    compressed_t2_params: CompressedT2Params | None
    connectivity_opt: bool = False
    random_op: bool = False
    regularization: bool = False,
    regularization_option: int = 0,
    # max_bond: int
    # perm_mps: bool
    # cutoff: int
    seed: int

    @property
    def dirpath(self) -> Path:
        if self.random_op:
            compress_option = "random"
        elif self.connectivity_opt:
            compress_option = "connectivity_opt-True"
        elif self.compressed_t2_params is not None:
            compress_option = self.compressed_t2_params.dirpath
            if self.regularization:
                compress_option = f"{compress_option}/regularization_{self.regularization_option}"
        else:
            compress_option = "truncated"
        return (
            Path(self.molecule_basename)
            / (
                ""
                if self.bond_distance is None
                else f"bond_distance-{self.bond_distance:.5f}"
            )
            / self.lucj_params.dirpath
            / compress_option
            / "quimb_sci_aqc"
            / self.lbfgsb_params.dirpath
            / f"seed-{self.seed}"
        )

    @property
    def operatorpath(self) -> Path:
        if self.random_op:
            compress_option = "random"
        elif self.connectivity_opt:
            compress_option = "connectivity_opt-True"
        elif self.compressed_t2_params is not None:
            compress_option = self.compressed_t2_params.dirpath
            if self.regularization:
                compress_option = f"{compress_option}/regularization_{self.regularization_option}"
        else:
            compress_option = "truncated"
        return (
            Path(self.molecule_basename)
            / (
                ""
                if self.bond_distance is None
                else f"bond_distance-{self.bond_distance:.5f}"
            )
            / self.lucj_params.dirpath
            / compress_option
        )


def load_operator(task: LUCJQuimbTask, data_dir: str, mol_data):
    operator_filename = data_dir / task.operatorpath / "operator.npz"
    if not os.path.exists(operator_filename):
        logging.info(f"Operator for {task} does not exists.\n")
        return None
    logging.info(f"Load operator for {task}.\n")
    operator = np.load(operator_filename)
    diag_coulomb_mats = operator["diag_coulomb_mats"]
    orbital_rotations = operator["orbital_rotations"]

    final_orbital_rotation = None
    if mol_data.ccsd_t1 is not None:
        final_orbital_rotation = (
            ffsim.variational.util.orbital_rotation_from_t1_amplitudes(mol_data.ccsd_t1)
        )
    elif mol_data.ccsd_t2 is None:
        nelec = mol_data.nelec
        norb = mol_data.norb
        c0, c1, c2 = pyscf.ci.cisd.cisdvec_to_amplitudes(
            mol_data.cisd_vec, norb, nelec[0]
        )
        assert abs(c0) > 1e-8
        t1 = c1 / c0
        final_orbital_rotation = (
            ffsim.variational.util.orbital_rotation_from_t1_amplitudes(t1)
        )
    operator = ffsim.UCJOpSpinBalanced(
        diag_coulomb_mats=diag_coulomb_mats,
        orbital_rotations=orbital_rotations,
        final_orbital_rotation=final_orbital_rotation,
    )
    return operator

def get_sci_vec(filepath: str, mol_data, dice_solver):
    if not os.path.exists(filepath):
        if dice_solver:
            logger.debug("\tRunning SCI using Dice solver...")
            t0 = timeit.default_timer()
            sci_energy, sci_state, _ = solve_hci(
                hcore=mol_data.one_body_integrals,
                eri=mol_data.two_body_integrals,
                norb=mol_data.norb,
                nelec=mol_data.nelec,
                # ci_strs=previous_sci_strings,
                select_cutoff=1e-4,
                clean_temp_dir=True,
                mpirun_options=["--quiet", "-n", "8"],
            )
            t1 = timeit.default_timer()
            logger.debug(f"\tFinished running SCI in {t1 - t0} seconds.")
            mol_data.sci_energy = sci_energy + mol_data.core_energy
            mol_data.sci_vec = (
                sci_state.amplitudes,
                sci_state.ci_strs_a,
                sci_state.ci_strs_b,
            )
            # previous_sci_strings = sci_strings
        else:
            logger.debug("\tRunning SCI using PySCF solver...")
            scf = mol_data.scf.run()
            sci = pyscf.fci.SCI(scf)
            ci0 = None
            t0 = timeit.default_timer()
            sci.select_cutoff = 1e-4
            sci_energy, sci_vec = sci.kernel(
                mol_data.one_body_integrals,
                mol_data.two_body_integrals,
                norb=mol_data.norb,
                nelec=mol_data.nelec,
                ci0=ci0,
            )
            t1 = timeit.default_timer()
            logger.debug(f"\tFinished running SCI in {t1 - t0} seconds.")
            if sci.converged:
                mol_data.sci_energy = sci_energy + mol_data.core_energy
                mol_data.sci_vec = (sci_vec, *(sci_vec._strs))
            else:
                logger.info("SCI did not converge.")
        with open(filepath, "wb") as f:
            pickle.dump(mol_data.sci_vec, f)
        return mol_data.sci_vec
    else:
        with open(filepath, "rb") as f:
            sci_vec = pickle.load(f)
        return sci_vec

def get_important_bit_string(sci_vec, tol = 1e-5):
    half_hf_state = "0" * (norb - nelectron // 2) + "1" * (nelectron // 2)
    hf_state = half_hf_state + half_hf_state
    amplitude = sci_vec[0]
    strs_a = sci_vec[1]
    strs_b = sci_vec[2]
    strings_a = convert_bitstring_type(
        [s for s in strs_a],
        input_type=ffsim.BitstringType.INT,
        output_type=ffsim.BitstringType.STRING,
        length=norb,
    )
    strings_b = convert_bitstring_type(
        [s for s in strs_b],
        input_type=ffsim.BitstringType.INT,
        output_type=ffsim.BitstringType.STRING,
        length=norb,
    )
    r, c = amplitude.shape
    important_amp_bitstr = []
    for i in range(r):
        for j in range(c):
            if abs(amplitude[i][j]) - tol > 1e-8:
                bitstr = "".join((strings_b[j], strings_a[i]))
                bitstr = bitstr[::-1]
                if bitstr != hf_state:
                    important_amp_bitstr.append((abs(amplitude[i][j]), bitstr))
    return important_amp_bitstr

def construct_quimb_circuit(circuit: QuantumCircuit):
    # quimb_circ = quimb_circuit(
    #     circuit,
    #     quimb_circuit_class=quimb.tensor.Circuit,
    #     # to_backend=to_backend,
    # )

    quimb_circ = quimb_circuit(
            circuit,
            quimb_circuit_class=quimb.tensor.CircuitMPS,
            max_bond=100,
            cutoff=1e-10,
            progbar=True,
            # to_backend=to_backend,
        )
    return quimb_circ

def constrcut_target_tn(important_amplitude, important_bitstr):
    pass
    
def to_backend(x):
    # return jnp(x, dtype=torch.complex64, device="cuda")
    return jax.device_put(x)


def run_lucj_sqd_quimb_task(
    task: LUCJQuimbTask,
    *,
    data_dir: Path,
    molecules_catalog_dir: Path | None = None,
    bootstrap_task: LUCJQuimbTask | None = None,
    bootstrap_data_dir: Path | None = None,
    overwrite: bool = True,
    use_dice: bool = False,
    run_sqd: bool = False
) -> LUCJQuimbTask:
    logging.info(f"{task} Starting...\n")
    os.makedirs(data_dir / task.dirpath, exist_ok=True)

    result_filename = data_dir / task.dirpath / "data.pickle"
    info_filename = data_dir / task.dirpath / "info.pickle"
    state_vector_filename = data_dir / task.dirpath / "state_vector.npy"
    sample_filename = data_dir / task.dirpath / "sample.pickle"

    if (
        (not overwrite)
        and os.path.exists(result_filename)
        and os.path.exists(info_filename)
    ):
        logger.info(f"Data for {task} already exists. Skipping...\n")
        return task
    intermediate_result_filename = (
        data_dir / task.dirpath / "intermediate_data.pickle"
    )

    # Get molecular data and molecular Hamiltonian
    molecule_basename = task.molecule_basename
    if task.bond_distance is not None:
        molecule_basename += f"_d-{task.bond_distance:.5f}"
    mol_data = load_molecular_data(
        molecule_basename,
        molecules_catalog_dir=molecules_catalog_dir,
    )
    
    norb = mol_data.norb
    nelec = mol_data.nelec
    mol_ham = mol_data.hamiltonian
    pairs_aa, pairs_ab = interaction_pairs_spin_balanced(
        task.lucj_params.connectivity, norb
    )

    # compute SCI vec
    logger.info(f"{task} Load SCI vectors...\n")
    sci_filename = f"{data_dir}/{molecule_basename}/sci_vec.pickle"
    sci_vec = get_sci_vec(sci_filename, mol_data, use_dice)
    important_amp_bitstr = get_important_bit_string(sci_vec, tol=1e-3)
    logger.info(f"{task} Finish loading SCI vectors...\n")

    rng = np.random.default_rng(task.seed)

    if overwrite or (not os.path.exists(result_filename) and not os.path.exists(info_filename)):
        # Generate initial parameters
        if bootstrap_task is None:
            # use CCSD to initialize parameters
            operator = load_operator(task, data_dir, mol_data)
            if operator is None:
                operator, _, _ = from_t_amplitudes_compressed(
                    mol_data.ccsd_t2,
                    n_reps=task.lucj_params.n_reps,
                    t1=mol_data.ccsd_t1 if task.lucj_params.with_final_orbital_rotation else None,
                    interaction_pairs=(pairs_aa, pairs_ab),
                    optimize=True,
                )
                logging.info(f"No operator is found for {task}.\n")
                return
            params = operator.to_parameters(interaction_pairs=(pairs_aa, pairs_ab))
        else:
            bootstrap_result_filename = os.path.join(
                bootstrap_data_dir or data_dir, bootstrap_task.dirpath, "data.pickle"
            )
            with open(bootstrap_result_filename, "rb") as f:
                result = pickle.load(f)
                params = result.x
            if bootstrap_task.lucj_params.n_reps < task.lucj_params.n_reps:
                n_params = ffsim.UCJOpSpinBalanced.n_params(
                    norb=norb,
                    n_reps=task.lucj_params.n_reps,
                    interaction_pairs=(pairs_aa, pairs_ab),
                    with_final_orbital_rotation=bootstrap_task.lucj_params.with_final_orbital_rotation,
                )
                params = np.concatenate([params, np.zeros(n_params - len(params))])
            if (
                task.lucj_params.with_final_orbital_rotation
                and not bootstrap_task.lucj_params.with_final_orbital_rotation
            ):
                params = np.concatenate([params, np.zeros(norb**2)])
                params[-(norb**2) :] = orbital_rotation_to_parameters(
                    np.eye(norb, dtype=complex)
                )

        # Optimize ansatz
        
        # half_hf_state = "0" * (norb - nelectron // 2) + "1" * (nelectron // 2)
        # hf_state = half_hf_state + half_hf_state
        important_amp_bitstr.sort(reverse=True)
        # print(len(important_amp_bitstr))
        # print(important_amp_bitstr[1000])
        # input()
        operator = ffsim.UCJOpSpinBalanced.from_parameters(
            params,
            norb=norb,
            n_reps=task.lucj_params.n_reps,
            interaction_pairs=(pairs_aa, pairs_ab),
            with_final_orbital_rotation=task.lucj_params.with_final_orbital_rotation,
        )
        # Construct Qiskit circuit
        qubits = QuantumRegister(2 * norb)
        circuit = QuantumCircuit(qubits)
        circuit.append(ffsim.qiskit.PrepareHartreeFockJW(norb, nelec), qubits)
        circuit.append(ffsim.qiskit.UCJOpSpinBalancedJW(operator), qubits)
        # change to quimb
        decomposed = circuit.decompose(reps=2)
        quimb_circuit = construct_quimb_circuit(decomposed)

        def loss_mps_bitstr(quimb_circuit):
            # optional: minimize amp of hf state
            loss = 0
            # we use `autoray.do` to allow arbitrary autodiff backends
            for amp, bitstr in important_amp_bitstr:
                psi_b = quimb_circuit.psi.copy()
                # fix the output indices to the correct bitstring
                for i, x in zip(range(quimb_circuit.N), bitstr):
                    psi_b.isel_({psi_b.site_ind(i): x})

                # perform a final simplification and cast
                psi_b.full_simplify_()
                quimb_circuit._maybe_convert(psi_b, None)

                tree = psi_b.contraction_tree(output_inds=(), optimize="auto-hq")

                # perform the full contraction with the tree found
                amp_cir = psi_b.contract(
                    all, output_inds=(), optimize=tree
                )
                # print(amp_cir)
                # print(amp)
                loss += (np.abs(np.abs(amp_cir) - amp))
            print(loss)
            return autoray.do("real", loss)
            # info["x"].append(intermediate_result.x)
            # info["fun"].append(intermediate_result.fun)
            # info["nit"] += 1
            # with open(intermediate_result_filename, "wb") as f:
            #     pickle.dump(intermediate_result, f)
            # if info["nit"] > 3:
            #     if (abs(info["fun"][-1] - info["fun"][-2]) < 1e-5) and (abs(info["fun"][-2] - info["fun"][-3]) < 1e-5):
            #         raise StopIteration("Objective function value does not decrease for two iterations.")
        logger.info(f"{task} Optimizing ansatz...\n")
        t0 = timeit.default_timer()
        info = defaultdict(list)
        info["nit"] = 0

        # result = scipy.optimize.minimize(
        #     fun,
        #     x0=params,
        #     method="COBYQA",
        #     options=dataclasses.asdict(task.cobyqa_params),
        #     callback=callback,
        # )
        # result = scipy.optimize.differential_evolution(
        #     loss_mps_bitstr,
        #     [(-1e3, 1e3) for x in params],
        #     callback=callback,
        #     x0=params,
        #     maxiter=task.lbfgsb_params.maxiter
        #     # workers=2,
        # )
        # print(f"init loss: {loss_mps_bitstr(quimb_circuit)}") # 107.377410381147 
        # tnopt = quimb.tensor.TNOptimizer(
        #     quimb_circuit,
        #     loss_mps_bitstr,
        #     # because we are using dynamic (entry dependent) simplification
        #     autodiff_backend="autograd",
        # )
        # # quimb_circuit = tnopt.optimize(10)
        # t1 = timeit.default_timer()
        # logger.info(f"{task} Done optimizing ansatz in {t1 - t0} seconds.\n")
        # input()
        # print(loss(quimb_circuit))
        # input()
        # tnopt = quimb.tensor.TNOptimizer(
        #         quimb_circuit,                              # the tensor network we want to optimize
        #         loss,                                       # the function we want to minimize
        #         # loss_constants={'target_tn': target_tn},    # supply U to the loss function as a constant TN
        #         # tags=['U3'],                                # only optimize U3 tensors
        #         autodiff_backend='jax',                     # use 'autograd' for non-compiled optimization
        #         optimizer='L-BFGS-B',                       # the optimization algorithm
        #     )
        

        # run aqc tensor
        aqc_ansatz, aqc_initial_parameters = generate_ansatz_from_circuit(
                decomposed, qubits_initially_zero=True
        )

        simulator_settings = AerSimulator(
            method="matrix_product_state",
            matrix_product_state_max_bond_dimension=100,
        )
                
        objective = MaximizeStateFidelity(
            quimb_circuit, aqc_ansatz, simulator_settings
        )

        def callback(intermediate_result: scipy.optimize.OptimizeResult):
            logger.info(f"Task {task} is on iteration {info['nit']}.\n")
            logger.info(f"\tObjective function value: {intermediate_result.fun}.\n")
            info["x"].append(intermediate_result.x)
            info["fun"].append(intermediate_result.fun)
            info["nit"] += 1
            with open(intermediate_result_filename, "wb") as f:
                pickle.dump(intermediate_result, f)
            if info["nit"] > 3:
                if (abs(info["fun"][-1] - info["fun"][-2]) < 1e-5) and (abs(info["fun"][-2] - info["fun"][-3]) < 1e-5):
                    raise StopIteration("Objective function value does not decrease for two iterations.")
      
        result = scipy.optimize.minimize(
            objective,
            aqc_initial_parameters,
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": 100},
            callback=callback,
        )
        
        logger.info(f"{task} Saving data...\n")
        with open(result_filename, "wb") as f:
            pickle.dump(result, f)
        with open(info_filename, "wb") as f:
            pickle.dump(info, f)
    else:
        with open(result_filename, "rb") as f:
            result = pickle.load(f)

    # continue to run sqd
    if os.path.exists(state_vector_filename):
        logging.info(f"{task} Computing state vector\n")
        operator = ffsim.UCJOpSpinBalanced.from_parameters(
            result.x,
            norb=norb,
            n_reps=task.lucj_params.n_reps,
            interaction_pairs=(pairs_aa, pairs_ab),
            with_final_orbital_rotation=task.lucj_params.with_final_orbital_rotation,
        )
        reference_state = ffsim.hartree_fock_state(norb, nelec)
        final_state = ffsim.apply_unitary(reference_state, operator, norb=norb, nelec=nelec)
        with open(state_vector_filename, "wb") as f:
            np.save(f, final_state)
    else:
        with open(state_vector_filename, "rb") as f:
            final_state = np.load(f)

    logging.info(f"{task} Sampling...\n")
    samples = ffsim.sample_state_vector(
        final_state,
        norb=norb,
        nelec=nelec,
        shots=100_000,
        seed=rng,
        bitstring_type=ffsim.BitstringType.INT,
    )
    bit_array = BitArray.from_samples(samples, num_bits=2 * norb)
    bit_array_count = bit_array.get_int_counts()
    with open(sample_filename, "wb") as f:
        pickle.dump(bit_array_count, f)


    # Run SQD
    if run_sqd:
        logging.info(f"{task} Running SQD...\n")
        # sci_solver = partial(solve_sci_batch, spin_sq=0.0)

        result_history_energy = []
        result_history_subspace_dim = []
        result_history = []

        def callback(results: list[SCIResult]):
            result_energy = []
            result_subspace_dim = []
            iteration = len(result_history)
            result_history.append(results)
            logging.info(f"Iteration {iteration}")
            for i, result in enumerate(results):
                result_energy.append(result.energy + mol_data.core_energy)
                result_subspace_dim.append(result.sci_state.amplitudes.shape)
                logging.info(f"\tSubsample {i}")
                logging.info(f"\t\tEnergy: {result.energy + mol_data.core_energy}")
                logging.info(
                    f"\t\tSubspace dimension: {np.prod(result.sci_state.amplitudes.shape)}"
                )
            result_history_energy.append(result_energy)
            result_history_subspace_dim.append(result_subspace_dim)

        result = diagonalize_fermionic_hamiltonian(
            mol_ham.one_body_tensor,
            mol_ham.two_body_tensor,
            bit_array,
            samples_per_batch=2000,
            norb=norb,
            nelec=nelec,
            num_batches=10,
            energy_tol=1e-5,
            occupancies_tol=1e-3,
            max_iterations=1,
            sci_solver=solve_sci_batch,
            symmetrize_spin=True,
            carryover_threshold=1e-3,
            seed=rng,
            callback=callback,
            max_dim=2000,
        )
        logging.info(f"{task} Finish SQD\n")
        energy = result.energy + mol_data.core_energy
        sci_state = result.sci_state
        spin_squared = sci_state.spin_square()
        if mol_data.fci_energy is not None:
            error = energy - mol_data.fci_energy
        elif mol_data.sci_energy is not None:
            error = energy - mol_data.sci_energy
        else:
            error = -1

        # Save data
        data = {
            "energy": energy,
            "error": error,
            "spin_squared": spin_squared,
            "sci_vec_shape": sci_state.amplitudes.shape,
            "history_energy": result_history_energy,
            "history_sci_vec_shape": result_history_subspace_dim,
        }

        logger.info(f"{task} Saving SQD data...\n")
        sqd_result_filename = data_dir / task.dirpath / "sqd_data.pickle"
        with open(sqd_result_filename, "wb") as f:
            pickle.dump(data, f)


molecule_name = "n2"
basis = "6-31g"
nelectron, norb = 10, 16
bond_distance = 1.2
molecule_basename = f"{molecule_name}_{basis}_{nelectron}e{norb}o"

molecule_name = "fe2s2"
nelectron, norb = 30, 20
bond_distance = None
molecule_basename = f"{molecule_name}_{nelectron}e{norb}o"
# molecule_basename = f"{molecule_name}_{basis}_{nelectron}e{norb}o_d-{bond_distance:.5f}"
# mol_data = load_molecular_data(
#     molecule_basename,
#     molecules_catalog_dir="/home/WanHsuan.Lin/molecules-catalog",
# )

# sci_vec = get_sci_vec("scratch/test_sci.pickle", mol_data, True)

# fci = sci_vec_to_fci_vec(
#     sci_vec[0],
#     sci_vec[1],
#     sci_vec[2],
#     norb=norb,
#     nelec=(nelectron//3, nelectron//3))

# amp, bitstr = get_important_bit_string(sci_vec)

# print(amp)
# print(bitstr)

# print("fci")
# print(fci)

# strings = ffsim.addresses_to_strings(
#     fci, norb=norb, nelec=(nelectron//3, nelectron//3), bitstring_type=ffsim.BitstringType.STRING
# )

# print(strings)

lucj_quimb_task = LUCJQuimbTask(
    molecule_basename=molecule_basename,
    bond_distance=bond_distance,
    lucj_params=LUCJParams(
            connectivity="heavy-hex",
            n_reps=1,
            with_final_orbital_rotation=True,
        ),
    compressed_t2_params=CompressedT2Params(
        multi_stage_optimization=True,
        begin_reps=20,
        step=2
    ),
    regularization=False,
    connectivity_opt = False,
    lbfgsb_params=LBFGSBParams(maxiter=100),
    regularization_option = 0,
    seed=0)

run_lucj_sqd_quimb_task(
    lucj_quimb_task,
    data_dir="/media/storage/WanHsuan.Lin/",
    molecules_catalog_dir="/home/WanHsuan.Lin/molecules-catalog",
    overwrite=True,
    run_sqd=True,
    use_dice=True,
)