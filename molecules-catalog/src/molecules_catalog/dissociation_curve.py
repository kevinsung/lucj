# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import dataclasses
import logging
import os
import pathlib
import timeit
from collections.abc import Sequence
from typing import Callable

import ffsim
import matplotlib.pyplot as plt
import numpy as np
import pyscf
import pyscf.cc
import pyscf.ci
import pyscf.fci
import pyscf.mcscf
import pyscf.symm
from pyscf.fci.selected_ci import _as_SCIvector
from qiskit_addon_dice_solver import solve_hci

logger = logging.getLogger(__name__)


# TODO try https://github.com/pyscf/pyscf/blob/master/examples/scf/30-scan_pes.py
def generate_dissociation_curve(
    start: float,
    stop: float,
    step: float,
    atom_factory: Callable[[float], list[tuple[str, tuple[float, float, float]]]],
    molecule_name: str,
    *,
    basis: str,
    charge: int | None = None,
    spin: int = 0,
    symmetry: str | None = None,
    n_frozen: int | None = None,
    active_space: Sequence[int] | None = None,
    irrep_nelec: dict[str, int] | None = None,
    norb: int | None = None,
    nelec: tuple[int, int] | None = None,
    run_ccsd: bool = False,
    run_sci: bool = False,
    store_sci_vec: bool = False,
    dice_solver: bool = False,
    run_fci: bool = False,
    store_fci_vec: bool = False,
    cas_irrep_nocc: dict[str, int] | None = None,
    cas_irrep_ncore: dict[str, int] | None = None,
    wfnsym: str | None = None,
    verbose: int = 0,
    data_dir: str | bytes | os.PathLike = "data",
    save_data: bool = True,
) -> tuple[int, tuple[int, int]]:
    if n_frozen is not None and active_space is not None:
        raise ValueError(
            "You can specify only one of n_frozen and active_space, not both."
        )

    data_dir = pathlib.Path(data_dir)
    os.makedirs(data_dir, exist_ok=True)

    bond_distances = np.linspace(start, stop, num=round((stop - start) / step) + 1)

    previous_mol_data = None

    if norb is None or nelec is None:
        norb, nelec = _get_norb_nelec(
            atom_factory(bond_distances[0]),
            basis=basis,
            charge=charge,
            spin=spin,
            symmetry=symmetry,
            n_frozen=n_frozen,
            active_space=active_space,
            irrep_nelec=irrep_nelec,
        )
    molecule_base_name = f"{molecule_name}_{basis}_{sum(nelec)}e{norb}o"

    for bond_distance in bond_distances:
        logger.debug(f"Bond length = {bond_distance}")

        # Load or create MolecularData
        filepath = data_dir / f"{molecule_base_name}_d-{bond_distance:.5f}.json.xz"
        if os.path.exists(filepath):
            logger.debug("\tLoading molecular data...")
            mol_data = ffsim.MolecularData.from_json(filepath, compression="lzma")
            mol = mol_data.mole
            scf = mol_data.scf.run()
            if active_space is None:
                active_space = list(range(n_frozen or 0, mol.nao_nr()))
            active_space = list(active_space)
        else:
            # Build molecule
            logger.debug("\tBuilding molecule...")
            mol = pyscf.gto.Mole()
            mol.build(
                atom=atom_factory(bond_distance),
                basis=basis,
                charge=charge,
                spin=spin,
                symmetry=symmetry,
                verbose=verbose,
            )
            scf = pyscf.scf.RHF(mol)
            if irrep_nelec is not None:
                scf.irrep_nelec = irrep_nelec

            # Hartree-Fock
            logger.debug("\tRunning SCF...")
            rdm = None
            if previous_mol_data is not None:
                scf_tmp = previous_mol_data.scf
                scf_tmp.mo_coeff = previous_mol_data.mo_coeff
                scf_tmp.mo_occ = previous_mol_data.mo_occ
                rdm = previous_mol_data.scf.make_rdm1()
            scf.kernel(rdm)
            if not scf.converged:
                raise RuntimeError("SCF did not converge.")
            scf.analyze()

            # CASCI initial orbitals
            logger.debug("\tComputing CASCI initial orbitals...")
            if active_space is None:
                active_space = list(range(n_frozen or 0, mol.nao_nr()))
            active_space = list(active_space)
            assert len(active_space) == norb
            nelectron_cas = int(sum(scf.mo_occ[active_space]))
            n_alpha = (nelectron_cas + mol.spin) // 2
            n_beta = (nelectron_cas - mol.spin) // 2
            assert (n_alpha, n_beta) == nelec
            cas = pyscf.mcscf.RCASCI(scf, ncas=norb, nelecas=nelec)
            if cas_irrep_nocc is None:
                mo_coeff = cas.sort_mo(active_space, base=0)
            else:
                mo_coeff = cas.sort_mo_by_irrep(cas_irrep_nocc, cas_irrep_ncore)
            h1e_cas, ecore = cas.get_h1eff(mo_coeff=mo_coeff)
            h2e_cas = cas.get_h2eff(mo_coeff=mo_coeff)

            mol_data = ffsim.MolecularData(
                core_energy=ecore,
                one_body_integrals=h1e_cas,
                two_body_integrals=h2e_cas,
                norb=norb,
                nelec=nelec,
                atom=mol.atom,
                basis=mol.basis,
                spin=mol.spin,
                symmetry=mol.symmetry,
                hf_energy=scf.e_tot,
                mo_coeff=scf.mo_coeff,
                mo_occ=scf.mo_occ,
            )

        # CISD
        if mol_data.cisd_energy is None:
            logger.debug("\tRunning CISD...")
            cisd = pyscf.ci.RCISD(
                scf,
                frozen=[i for i in range(mol.nao_nr()) if i not in active_space]
                or None,
            )
            # TODO try using nroots > 1 to converge CISD calculations
            ci0 = None
            if previous_mol_data is not None:
                ci0 = previous_mol_data.cisd_vec
            cisd.kernel(ci0=ci0)
            if cisd.converged:
                mol_data.cisd_energy = cisd.e_tot
                mol_data.cisd_vec = cisd.ci
            else:
                logger.info("\tCISD did not converge.")

        # CCSD
        if run_ccsd and mol_data.ccsd_energy is None:
            logger.debug("\tRunning CCSD...")
            ccsd = pyscf.cc.RCCSD(
                scf,
                frozen=[i for i in range(mol.nao_nr()) if i not in active_space]
                or None,
            )
            t1, t2 = None, None
            if previous_mol_data is not None:
                t1, t2 = previous_mol_data.ccsd_t1, previous_mol_data.ccsd_t2
            ccsd.kernel(t1, t2)
            if ccsd.converged:
                mol_data.ccsd_energy = ccsd.e_tot
                mol_data.ccsd_t1 = ccsd.t1
                mol_data.ccsd_t2 = ccsd.t2
            else:
                logger.info("\tCCSD did not converge.")

        # SCI
        if run_sci and mol_data.sci_energy is None:
            if dice_solver:
                logger.debug("\tRunning SCI using Dice solver...")
                t0 = timeit.default_timer()
                sci_energy, sci_state, _ = solve_hci(
                    hcore=mol_data.one_body_integrals,
                    eri=mol_data.two_body_integrals,
                    norb=norb,
                    nelec=nelec,
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
                sci = pyscf.fci.SCI(scf)
                ci0 = None
                if (
                    previous_mol_data is not None
                    and previous_mol_data.sci_vec is not None
                ):
                    coeffs, strs_a, strs_b = previous_mol_data.sci_vec
                    ci0 = _as_SCIvector(coeffs, (strs_a, strs_b))
                t0 = timeit.default_timer()
                sci.select_cutoff = 1e-4
                sci_energy, sci_vec = sci.kernel(
                    mol_data.one_body_integrals,
                    mol_data.two_body_integrals,
                    norb=norb,
                    nelec=nelec,
                    ci0=ci0,
                )
                t1 = timeit.default_timer()
                logger.debug(f"\tFinished running SCI in {t1 - t0} seconds.")
                if sci.converged:
                    mol_data.sci_energy = sci_energy + mol_data.core_energy
                    mol_data.sci_vec = (sci_vec, *(sci_vec._strs))
                else:
                    logger.info("SCI did not converge.")

        # FCI
        if run_fci and mol_data.fci_energy is None:
            logger.debug("\tRunning FCI...")
            ci0 = None
            if wfnsym is not None:
                cas.fcisolver.wfnsym = wfnsym
                if previous_mol_data is not None:
                    ci0 = previous_mol_data.fci_vec
            cas.fix_spin_(ss=0)
            cas.kernel(mo_coeff=mo_coeff, ci0=ci0)
            if cas.converged:
                mol_data.fci_energy = cas.e_tot
                mol_data.fci_vec = cas.ci
            else:
                logger.info("\tFCI did not converge.")

        # Other data
        if mol_data.symmetry and mol_data.orbital_symmetries is None:
            mol_data.orbital_symmetries = list(
                pyscf.symm.label_orb_symm(mol, mol.irrep_name, mol.symm_orb, mo_coeff)
            )

        if save_data:
            # Save data to disk
            mol_data_save = mol_data
            if not store_sci_vec:
                mol_data_save = dataclasses.replace(mol_data_save, sci_vec=None)
            if not store_fci_vec:
                mol_data_save = dataclasses.replace(mol_data_save, fci_vec=None)
            logger.debug("\tSaving data...")
            mol_data_save.to_json(filepath, compression="lzma")
            logger.info(f"\tSaved molecular data to {filepath}.")

        logger.debug(f"\tHF energy: {mol_data.hf_energy}.")
        logger.debug(f"\tCISD energy: {mol_data.cisd_energy}.")
        logger.debug(f"\tCCSD energy: {mol_data.ccsd_energy}.")
        logger.debug(f"\tSCI energy: {mol_data.sci_energy}.")
        logger.debug(f"\tFCI energy: {mol_data.fci_energy}.")

        previous_mol_data = mol_data

    return norb, nelec


def plot_dissociation_curve(
    start: float,
    stop: float,
    step: float,
    molecule_name: str,
    basis: str,
    norb: int,
    nelec: tuple[int, int],
    *,
    extension: str = "svg",
    data_dir: str | bytes | os.PathLike = "data",
    plots_dir: str | bytes | os.PathLike = "plots",
):
    data_dir = pathlib.Path(data_dir)
    molecule_base_name = f"{molecule_name}_{basis}_{sum(nelec)}e{norb}o"
    bond_distances = np.linspace(start, stop, num=round((stop - start) / step) + 1)
    mol_datas = [
        ffsim.MolecularData.from_json(
            data_dir / f"{molecule_base_name}_d-{bond_distance:.5f}.json.xz",
            compression="lzma",
        )
        for bond_distance in bond_distances
    ]
    hf_energies = np.array([mol_data.hf_energy for mol_data in mol_datas])
    cisd_energies = np.array([mol_data.cisd_energy for mol_data in mol_datas])
    ccsd_energies = np.array([mol_data.ccsd_energy for mol_data in mol_datas])
    sci_energies = np.array([mol_data.sci_energy for mol_data in mol_datas])
    fci_energies = np.array([mol_data.fci_energy for mol_data in mol_datas])

    _, ax = plt.subplots()

    if not all(x is None for x in fci_energies):
        ax.plot(
            bond_distances,
            fci_energies,
            "-",
            label="FCI",
            color="black",
        )
    ax.plot(
        bond_distances,
        hf_energies,
        "--",
        label="HF",
        color="blue",
    )
    ax.plot(
        bond_distances,
        cisd_energies,
        "--",
        label="CISD",
        color="green",
    )
    ax.plot(
        bond_distances,
        ccsd_energies,
        "--",
        label="CCSD",
        color="orange",
    )
    if not all(x is None for x in sci_energies):
        ax.plot(
            bond_distances,
            sci_energies,
            "--",
            label="SCI",
            color="red",
        )

    ax.legend()
    ax.set_title(f"{molecule_name} dissociation {basis} ({sum(nelec)}e, {norb}o)")

    plots_dir = pathlib.Path(plots_dir)
    os.makedirs(plots_dir, exist_ok=True)
    filepath = plots_dir / f"{molecule_base_name}.{extension}"
    plt.savefig(filepath)
    logger.info(f"Saved plot to {filepath}.")
    plt.close()


def _get_norb_nelec(
    atom: list[tuple[str, tuple[float, float, float]]],
    basis: str,
    charge: int | None = None,
    spin: int = 0,
    symmetry: str | None = None,
    n_frozen: int | None = None,
    active_space: Sequence[int] | None = None,
    irrep_nelec: dict[str, int] | None = None,
) -> tuple[int, tuple[int, int]]:
    # Build molecule
    logger.debug("\tBuilding molecule...")
    mol = pyscf.gto.Mole()
    mol.build(
        atom=atom,
        basis=basis,
        charge=charge,
        spin=spin,
        symmetry=symmetry,
        verbose=0,
    )
    scf = pyscf.scf.RHF(mol)
    if irrep_nelec is not None:
        scf.irrep_nelec = irrep_nelec

    # Hartree-Fock
    scf.kernel()

    # CASCI initial orbitals
    if active_space is None:
        active_space = list(range(n_frozen or 0, mol.nao_nr()))
    active_space = list(active_space)
    norb = len(active_space)
    nelectron_cas = int(sum(scf.mo_occ[active_space]))
    n_alpha = (nelectron_cas + mol.spin) // 2
    n_beta = (nelectron_cas - mol.spin) // 2
    nelec = (n_alpha, n_beta)

    return norb, nelec
