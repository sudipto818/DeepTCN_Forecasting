import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

files = [
    "test_metrics.py",
    "test_windows.py",
    "test_cqr.py",
    "test_aci.py",
    "test_data_traffic.py",
    "test_model.py",
    "test_distributions.py",
    "test_adaptive.py",
    "test_xseries.py",
]

fails = 0
for f in files:
    print("=" * 50)
    print(f)
    print("=" * 50)
    r = subprocess.run([sys.executable, os.path.join(HERE, f)])
    if r.returncode != 0:
        fails += 1
    print("")

if fails == 0:
    print("all test files passed")
else:
    print(fails, "test file(s) failed")
    sys.exit(1)
