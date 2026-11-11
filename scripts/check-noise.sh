#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m compileall -q lab attention binding vendor/sphere-thermal-cli/sphere_thermal
python3 scripts/check_solver_noise.py
(cd vendor/sphere-thermal-cli && python3 ../../scripts/check_solver_noise.py --directory tests)
