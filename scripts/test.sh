#!/usr/bin/env bash
#
# SPDX-FileCopyrightText: 2025 German Aerospace Center (DLR)
# SPDX-License-Identifier: GPL-3.0-or-later
#

# exit script if any command fails
set -e

# pre commit on all files and changes (not only staged ones)
pre-commit run --all-files

# shell scripts
find scripts -type f -name "*.sh" -exec shellcheck --external-sources --shell bash --source-path scripts {} +

# markdown
npm exec markdownlint-cli2 -- "./docs/**/*.md" "./README.md"

# static code checks (ruff: linter)
python -m ruff check slki
if [[ -n $(find slki -name "*.py[i]" -print -quit) ]]; then
  python -m ruff check scripts
fi
# static code checks (ruff: formatter)
python -m ruff format --diff slki
if [[ -n $(find slki -name "*.py[i]" -print -quit) ]]; then
  python -m ruff format --diff scripts
fi

# static type checks (mypy)
python -m mypy slki
if [[ -n $(find slki -name "*.py[i]" -print -quit) ]]; then
  python -m mypy scripts
fi

# license checks
python -m licensecheck
python -m reuse lint

# vulnerability checks
python -m tox --recreate -e vulnerability

# success message
echo
echo "$(tput bold)Congratulations 🎉$(tput sgr0)"
echo "All tests successful."
