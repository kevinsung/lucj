# lucj

Numerical studies of the LUCJ ansatz.

The instructions below use uv (https://docs.astral.sh/uv/getting-started/installation/).

## Dependencies to install from source

- molecules-catalog https://github.ibm.com/ibm-q-research/molecules-catalog

## Set proper environment variables
set `LUCJ_DATA_ROOT`, where all data except the plots will be stored to, and `MOLECULES_CATALOG_DIR`, where you install `molecules-catalog`.

## Code structure

### Overview
- `paper`: plots for paper
- `plots`: other plots not for paper
- `scripts`: code to generate data 
- `src`: source code
- `scratch`: some temporary files to explore new things
- `csv`: csv files for bitstring number and quimb result

### Scripts
- `analysis`: files to 1) check operator norm, 2) plot sample distribution, and 3) plot CDF wave function
- `hardware`: run circuits on hardware and run SQD on the samples. Files `batch*.py` submit LUCJ-random, LUCJ-truncated, and LUCJ-compressed in one job. Files whose names contain `fractional_gate` use fractional gates. For scripts in `hardware/n2_cc-pvdz_10e26o/accumulate/`, we run `n_hardware_run` times on the hardware and run one SQD based on the accumelated samples across all the runs.
- `hardware_quimb`: run circuits optimized by tensor network optimization on hardware and run SQD on the samples.
- `operator`: generate compressed operator. Required packages: `JAX`.
- `paper`: scripts to generate plots in `paper`
- `plot`: scripts to generate plots in `plot`
- `quimb`: scripts to run tensor network optimization. Scripts with `nomard` use NOMAD as optimizer, and the others use COBYQA from SciPy. Required packages: `quimb` and `NOMAD`.
- `sqd`: scripts to run exact simulation and run SQD on the samples. For N2 6-31g, we will compute VQE data as well.
- `uccsd`: scripts to run UCCSD with original t2 (`uccsd_sqd_init.py`) and compressed t2 (`lucj_compressed_t2.py`)
- `state_vector_task`: scripts to compute state vector via exact simulation

To run a script, do
```bash
uv run $scripts
```

### Source files
- Hardware exp: update files in `src/lucj/hardware_sqd_task/hardware_job` to use the proper Qiskit service
- For sqd-related tasks, the files contain `sci` uses PySCF as solver while the other uses DICE.

### What in scratch
The files are outdated and required updates to the library filepaths. They are kept for the purpose of demonstrating the implementation of various methods.
- `scratch/archive/quimb_sci_task` is the implementation to optimize LUCJ to generate important bitstrings found by SCI
- `scratch/archive/quimb_explore` contains files to explore different way to do TN optimization, including using RBFOpt as optimizer, PEPS, and arbitrary circuit as TN, and trying belif propagation
- To modify parameters for NOMAD, one needs to directly modify `src/lucj/quimb_task/lucj_sqd_quimb_task_nomad.py` and `src/lucj/quimb_task/lucj_sqd_quimb_task_sci_nomad.py`. One might explore the use of [cahce files](https://github.com/bbopt/nomad/tree/master/examples/advanced/batch/UseCacheFileForRerun) to seperate optimization in multiple stages. Detailed parameter settings can be found in NOMAD documentation.




