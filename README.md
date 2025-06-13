# lucj

Numerical studies of the LUCJ ansatz.

## Dependencies

- uv https://docs.astral.sh/uv/getting-started/installation/
- molecules-catalog https://github.ibm.com/ibm-q-research/molecules-catalog

## Generate data

```bash
uv run scripts/run/lucj_initial_params/n2_sto-6g_6e6o.py
uv run scripts/run/lucj_aqc_mps/n2_sto-6g_6e6o.py
```

## Generate plots

```bash
uv run scripts/plot/n2_sto-6g_6e6o/energy_aqc_mps_max_bond.py
```
