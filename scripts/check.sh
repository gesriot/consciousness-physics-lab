#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m compileall -q lab vendor/sphere-thermal-cli/sphere_thermal
python3 -m unittest discover -s tests -v
(cd vendor/sphere-thermal-cli && python3 -m unittest discover -v)
