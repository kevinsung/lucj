import numpy as np
import pyscf
import ffsim
import scipy.linalg as LA
import scipy.optimize as OPT
import opt_einsum

def make_tau_operator(t2):
    nocc,nvir = t2.shape[0], t2.shape[2]
    norb = nocc+nvir
    t2_til = t2
    t2_bar = -np.einsum('ijab->jiab',t2_til)
    t2_hat = t2_til+t2_bar
    t2_so  = np.zeros((2*norb, 2*norb, 2*norb, 2*norb), dtype=complex)
    occ = [list(range(s*norb,s*norb+nocc))     for s in range(2)]
    vir = [list(range(s*norb+nocc,(s+1)*norb)) for s in range(2)]
    t2_so[np.ix_(vir[0],occ[0],vir[0],occ[0])] = np.einsum('ijab->aibj',t2_hat)
    t2_so[np.ix_(vir[1],occ[1],vir[1],occ[1])] = np.einsum('ijab->aibj',t2_hat)
    t2_so[np.ix_(vir[0],occ[0],vir[1],occ[1])] = np.einsum('ijab->aibj',t2_til)
    t2_so[np.ix_(vir[1],occ[1],vir[0],occ[0])] = np.einsum('ijab->aibj',t2_til)
    t2_so[np.ix_(vir[0],occ[1],vir[1],occ[0])] = np.einsum('ijab->aibj',t2_bar)
    t2_so[np.ix_(vir[1],occ[0],vir[0],occ[1])] = np.einsum('ijab->aibj',t2_bar)
    t2_so = t2_so-np.einsum('prqs->rpsq',t2_so)
    return t2_so

# projector on the set of 2-body anti-hermitian operators
def project(T):
    TP = ( T-np.einsum('prqs->psqr', T))/2.0
    TP = (TP-np.einsum('prqs->qrps',TP))/2.0
    TP = (TP-np.einsum('prqs->rpsq',TP).conj())/2.0
    return TP

def oo_norm(T):
    return np.abs(T).max()

def oo_distance(A,B):
    return oo_norm(A-B)

def make_masks(norb, n_tensors, idx_aa, idx_ab):
  triu_mask = [(p,r) for p in range(norb) for r in range(norb) if r<p]
  diag_mask = [(p,p) for p in range(norb)]
  n_re_K = norb * (norb - 1) // 2
  n_im_K = norb * (norb + 1) // 2
  n_J_aa = len(idx_aa)
  n_J_ab = len(idx_ab)
  x = np.zeros(n_tensors*(n_re_K+n_im_K+n_J_aa+n_J_ab))
  return triu_mask, diag_mask, x

def fill_vector(vec, mat, mask, start):
  for m,(p,r) in enumerate(mask):
    vec[start+m] = mat[p,r]
  return vec, start+len(mask)

def initialize_parameters(t2, n_tensors=1, idx_aa=[], idx_ab=[]):
  nocc, nvir = t2.shape[0], t2.shape[2]
  norb = nocc+nvir
  Jmats, Umats = ffsim.linalg.double_factorized_decomposition.double_factorized_t2(t2)
  triu_mask, diag_mask, x = make_masks(norb, n_tensors, idx_aa, idx_ab)
  mu = 0    # index of (f,s) pairs
  start = 0 # index of x blocks
  for f in range(Umats.shape[0]):
    for s in range(Umats.shape[1]):
      if(mu<n_tensors):
        Kmat = LA.logm(Umats[f,s,:,:])
        Jmat = Jmats[f,s,:,:]
        x, start = fill_vector(x, Kmat.real, triu_mask, start)
        x, start = fill_vector(x, Kmat.imag, triu_mask, start)
        x, start = fill_vector(x, Kmat.imag, diag_mask, start)        
        x, start = fill_vector(x, Jmat, idx_aa, start)
        x, start = fill_vector(x, Jmat, idx_ab, start)
      mu += 1
  return x

def _params_to_df_tensors(x, n_tensors, idx_aa, idx_ab, norb):
    orbital_rotations = np.zeros((n_tensors, 2*norb, 2*norb), dtype=complex)
    diag_coulomb_mats = np.zeros((n_tensors, 2*norb, 2*norb, 2*norb, 2*norb), dtype=complex)
    n_re_K = norb * (norb - 1) // 2
    n_im_K = norb * (norb + 1) // 2
    n_J_aa = len(idx_aa)
    n_J_ab = len(idx_ab)
    triu_mask = [(p,r) for p in range(norb) for r in range(norb) if r<p]
    diag_mask = [(p,p) for p in range(norb)]
    start = 0
    for i in range(n_tensors):
      K = np.zeros((norb, norb), dtype=complex)
      for m,(p,r) in enumerate(triu_mask):
        K[p,r] += x[start+m]
        K[r,p] -= x[start+m]
      start += n_re_K
      for m,(p,r) in enumerate(triu_mask):
        K[p,r] += 1j*x[start+m] 
        K[r,p] += 1j*x[start+m] 
      start += n_re_K
      for m,(p,p) in enumerate(diag_mask):
        K[p,p] += 1j*x[start+m]
      start += norb
      assert(np.abs(K+np.conj(K.T)).max()<1e-6)
      U = LA.expm(K)
      orbital_rotations[i, :norb, :norb] = U
      orbital_rotations[i, norb:, norb:] = U
      D = np.zeros((2*norb, 2*norb, 2*norb, 2*norb))
      for m,(p,r) in enumerate(idx_aa):
        D[p,p,r,r] = x[start+m] * 0.5
        D[r,r,p,p] = x[start+m] * 0.5
        D[p+norb,p+norb,r+norb,r+norb] = x[start+m] * 0.5
        D[r+norb,r+norb,p+norb,p+norb] = x[start+m] * 0.5
      start += n_J_aa
      for m,(p,r) in enumerate(idx_ab):
        D[p,p,r+norb,r+norb] = x[start+m] * 0.5
        D[r+norb,r+norb,p,p] = x[start+m] * 0.5
        D[r,r,p+norb,p+norb] = x[start+m] * 0.5 # Quesiton: use this line to have consistent definition for aa and ab?
        D[p+norb,p+norb,r,r] = x[start+m] * 0.5 # Quesiton: use this line to have consistent definition for aa and ab?
      start += n_J_ab
      diag_coulomb_mats[i, :, :, :, :] = 4*project(1j*D) # T-dag(T) = exp(-K) iJ exp(K) so the i factor in that projector is to reproduce this relation.
      # print(f"i: {i}")
      # imag = np.imag(diag_coulomb_mats[i, :, :, :, :])
      # print(f"sum of imag: {np.sum(np.abs(imag))}")
      # real = np.real(diag_coulomb_mats[i, :, :, :, :])
      # print(f"sum of real: {np.sum(np.abs(real))}")
      # # print(f"complex diag_coulomb_mats: {np.any(np.iscomplex(diag_coulomb_mats[i, :, :, :, :]))}")
      # print(f"close after projection: {np.allclose(diag_coulomb_mats[i, :, :, :, :], 4*1j*D)}")
      # for try_projecting in range(1):
      #   print(f"complex D: {np.any(np.iscomplex(D))}")
      #   D_projected = project(1j * D)
      #   print(f"complex D_projected: {np.any(np.iscomplex(D_projected))}")
      #   print("|D|, |P(D)|, |D-P(D)| = ",oo_norm(D),oo_norm(D_projected),oo_distance(D,D_projected))
      #   D = D_projected
      # Jaa = np.zeros((n_tensors, 2*norb, 2*norb), dtype=complex)
      # for m,(p,r) in enumerate(idx_aa):
      #   Jaa[p,r] = D[p,p,r,r]
      #   Jaa[r,p] = D[r,r,p,p]
      # Jab = np.zeros((n_tensors, 2*norb, 2*norb), dtype=complex)
      # for m,(p,r) in enumerate(idx_aa):
      #   Jab[p,r] = D[p,p,r+norb,r+norb]
      #   Jab[r,p] = D[r,r,p+norb,p+norb]
      # imag = np.imag(Jaa)
      # print(f"sum of imag Jaa: {np.sum(np.abs(imag))}")
      # imag = np.imag(Jab)
      # print(f"sum of imag Jab: {np.sum(np.abs(imag))}")
      # real = np.real(Jaa)
      # print(f"sum of real Jaa: {np.sum(np.abs(real))}")
      # real = np.real(Jab)
      # print(f"sum of real Jab: {np.sum(np.abs(real))}")
      # print(f"complex Jaa: {np.any(np.iscomplex(Jaa))}")
      # print(f"complex Jab: {np.any(np.iscomplex(Jab))}")
    return diag_coulomb_mats, orbital_rotations

def _df_tensors_to_params(diag_coulomb_mats, orbital_rotations, n_tensors, idx_aa, idx_ab, norb):
  triu_mask, diag_mask, x = make_masks(norb, n_tensors, idx_aa, idx_ab)
  start = 0
  for mu in range(n_tensors):
    Kmat = LA.logm(orbital_rotations[mu,:norb,:norb])
    Jmat = diag_coulomb_mats[mu,:,:,:,:].imag
    Jmat_aa = np.zeros((norb,norb))
    Jmat_ab = np.zeros((norb,norb))
    for p in range(norb):
      for r in range(norb):
        Jmat_aa[p,r] = Jmat[p,p,r,r]
        Jmat_ab[p,r] = Jmat[p,p,r+norb,r+norb]
    #for (p,r) in idx_aa: print("Jaa ",Jmat_aa[p,r])
    #for (p,r) in idx_ab: print("Jab ",Jmat_ab[p,r])
    x, start = fill_vector(x, Kmat.real, triu_mask, start)
    x, start = fill_vector(x, Kmat.imag, triu_mask, start)
    x, start = fill_vector(x, Kmat.imag, diag_mask, start)
    x, start = fill_vector(x, Jmat_aa, idx_aa, start)
    x, start = fill_vector(x, Jmat_ab, idx_ab, start)
  return x

def lucj_operator(diag_coulomb_mats, orbital_rotations):
  return opt_einsum.contract("mPp,mQq,mSs,mRr,mprqs->PRQS",
                              orbital_rotations,
                              orbital_rotations,
                              np.conj(orbital_rotations),
                              np.conj(orbital_rotations),
                              diag_coulomb_mats,
                              optimize="greedy")


'''


mol = pyscf.M(
    atom = 'H 0 0 0; F 0 0 2.8',
    basis = 'sto-6g')

mf = mol.RHF()
mf = pyscf.scf.newton(mf)
mf.run()

mycc = mf.CCSD().run(frozen=1)
print('CCSD correlation energy', mycc.e_corr)



t2 = mycc.t2

nocc,nvir = t2.shape[0], t2.shape[2]
norb = nocc+nvir
t2_til = t2
t2_bar = -np.einsum('ijab->jiab',t2_til)
t2_hat = t2_til+t2_bar

t2_so = np.zeros((2*norb, 2*norb, 2*norb, 2*norb))
occ = [list(range(s*norb,s*norb+nocc)) for s in range(2)]
vir = [list(range(s*norb+nocc,(s+1)*norb)) for s in range(2)]
t2_so[np.ix_(vir[0],occ[0],vir[0],occ[0])] = np.einsum('ijab->aibj',t2_hat)
t2_so[np.ix_(vir[1],occ[1],vir[1],occ[1])] = np.einsum('ijab->aibj',t2_hat)
t2_so[np.ix_(vir[0],occ[0],vir[1],occ[1])] = np.einsum('ijab->aibj',t2_til)
t2_so[np.ix_(vir[1],occ[1],vir[0],occ[0])] = np.einsum('ijab->aibj',t2_til)
t2_so[np.ix_(vir[0],occ[1],vir[1],occ[0])] = np.einsum('ijab->aibj',t2_bar)
t2_so[np.ix_(vir[1],occ[0],vir[0],occ[1])] = np.einsum('ijab->aibj',t2_bar)

t2_so = t2_so-t2_so.transpose((3,2,1,0))

J0, U0 = ffsim.linalg.double_factorized_decomposition.double_factorized_t2(t2)
print(J0.shape, U0.shape)
print(np.abs(LA.det(U0[0,0,:,:])))
n_tensors = 1

x0 = _initialize_params(U0, J0, n_tensors, norb, [(p,p+1) for p in range(norb-1)], [(p,p) for p in range(norb)])

def fun(x):
  diag_coulomb_mats, orbital_rotations = _params_to_df_tensors(
      x, n_tensors, norb, [(p,p+1) for p in range(norb-1)], [(p,p) for p in range(norb)]
  )
  diff = t2_so - 1j*opt_einsum.contract(
      "mPp,mQq,mSs,mRr,mprqs->PRQS",
      orbital_rotations,
      orbital_rotations,
      np.conj(orbital_rotations),
      np.conj(orbital_rotations),
      diag_coulomb_mats,
      optimize="greedy",
  )
  F = 0.5*np.sum(np.abs(diff**2))
  print(F)
  return F

print("Initial discrepancy ",fun(x0))
res = OPT.minimize(fun,x0+np.random.random(len(x0))*0.01,method='CG')
print(res)


'''

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

t2_so = make_tau_operator(mol_data.ccsd_t2)

diag_coulomb_mats, orbital_rotations = ffsim.linalg.double_factorized_decomposition.double_factorized_t2(mol_data.ccsd_t2)

n_tensors = 2
norb = t2_so.shape[0]//2

from ffsim.variational.util import interaction_pairs_spin_balanced

idx_aa, idx_ab = interaction_pairs_spin_balanced("square", norb)
# print(idx_aa)

x0 = initialize_parameters(mol_data.ccsd_t2, n_tensors, idx_aa, idx_ab)

diag_coulomb_mats, orbital_rotations = _params_to_df_tensors(x0, n_tensors, idx_aa, idx_ab, norb)

diff = t2_so - lucj_operator(diag_coulomb_mats, orbital_rotations)


def fun(x):
    diag_coulomb_mats, orbital_rotations = _params_to_df_tensors(
        x, n_tensors, idx_aa, idx_ab, norb
    )
    diff = t2_so - lucj_operator(diag_coulomb_mats, orbital_rotations)
    F = 0.5*np.sum(np.abs(diff**2))
#   print(F)
    return F

res = OPT.minimize(fun,x0+1e-2*(np.random.random(len(x0))-0.5),method='L-BFGS-B', options = {"maxiter": 100})
print(res)

# F = 0.5*np.sum(np.abs(diff**2))
# print("Cost fun ",F)