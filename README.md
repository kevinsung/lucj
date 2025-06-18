# lucj

Numerical studies of the LUCJ ansatz.

The instructions below use uv (https://docs.astral.sh/uv/getting-started/installation/).

## Dependencies to install from source

- molecules-catalog https://github.ibm.com/ibm-q-research/molecules-catalog

## Generate data

```bash
uv run scripts/run/lucj_initial_params/n2_sto-6g_10e8o.py
uv run scripts/run/lucj_aqc_mps/n2_sto-6g_10e8o.py
```

## Generate plots

```bash
uv run scripts/plot/n2_sto-6g_10e8o/energy_aqc_mps_max_bond.py
```
