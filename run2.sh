#!/bin/bash

# molecule_basename="n2_6-31g_10e16o"

molecule_basename="fe2s2_30e20o"

uv run scripts/operator/$molecule_basename/lucj_compressed_t2_connectivity.py
uv run scripts/sqd/$molecule_basename/lucj_compressed_t2_connectivity.py
