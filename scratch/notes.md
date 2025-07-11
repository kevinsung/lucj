July 7
1. what is max_davidson?
    - src/lucj/tasks/uccsd_sqd_initial_params_task.py line 111. Can't run this file

July 3
1. Questions for param fitting v
2. t2-t2_dagger does not gives us better result v
3. UCCSD for imaginary part
    def _apply_unitary_(
        self, vec: np.ndarray, norb: int, nelec: int | tuple[int, int], copy: bool
    ) -> np.ndarray:
        if isinstance(nelec, int):
            return NotImplemented
        if copy:
            vec = vec.copy()

        nocc, _ = self.t1.shape
        assert nelec == (nocc, nocc)

        one_body_tensor = np.zeros((norb, norb))
        two_body_tensor = np.zeros((norb, norb, norb, norb))
        one_body_tensor[:nocc, nocc:] = self.t1
        one_body_tensor[nocc:, :nocc] = -self.t1.T
        two_body_tensor[nocc:, :nocc, nocc:, :nocc] = self.t2.transpose(2, 0, 3, 1) change here?
        two_body_tensor[:nocc, nocc:, :nocc, nocc:] = -self.t2.transpose(0, 2, 1, 3)

        linop = protocols.linear_operator(
            hamiltonians.MolecularHamiltonian(
                one_body_tensor=one_body_tensor, two_body_tensor=two_body_tensor
            ),
            norb=norb,
            nelec=nelec,
        )
        vec = scipy.sparse.linalg.expm_multiply(linop, vec, traceA=0.0)

        if self.final_orbital_rotation is not None:
            vec = gates.apply_orbital_rotation(
                vec, self.final_orbital_rotation, norb=norb, nelec=nelec, copy=False
            )

        return vec


July 10
1. modify code to save data
2. run larger data
3. set the limit for max dim to show we get high quality bit string
4. contribute to ffsim
4. Mario's code change to t rather than t dagger