
# echo "uv run scripts/uccsd/n2_6-31g_10e16o/uccsd_sqd_init.py"
# uv run scripts/uccsd/n2_6-31g_10e16o/uccsd_sqd_init.py

# echo "uv run scripts/sqd/n2_6-31g_10e16o/lucj_truncated.py"
# uv run scripts/sqd/n2_6-31g_10e16o/lucj_truncated.py

# echo "uv run scripts/sqd/n2_6-31g_10e16o/lucj_untruncated.py"
# uv run scripts/sqd/n2_6-31g_10e16o/lucj_untruncated.py

echo "uv run scripts/sqd/n2_6-31g_10e16o/lucj_compressed_t2.py"
uv run scripts/sqd/n2_6-31g_10e16o/lucj_compressed_t2.py

echo "uv run scripts/sqd/n2_6-31g_10e16o/lucj_truncated.py"
uv run scripts/sqd/n2_6-31g_10e16o/lucj_truncated.py

echo "uv run src/lucj/sqd_energy_task/lucj_random_t2_task_sci.py"
uv run src/lucj/sqd_energy_task/lucj_random_t2_task_sci.py

echo "uv run scripts/hardware/n2_6-31g_10e16o/lucj_compressed_t2.py"
uv run scripts/hardware/n2_6-31g_10e16o/lucj_compressed_t2.py

echo "uv run scripts/hardware/n2_6-31g_10e16o/lucj_random_t2.py"
uv run scripts/hardware/n2_6-31g_10e16o/lucj_random_t2.py

echo "uv run scripts/hardware/n2_6-31g_10e16o/lucj_truncated_t2.py"
uv run scripts/hardware/n2_6-31g_10e16o/lucj_truncated_t2.py