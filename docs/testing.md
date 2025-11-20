---
title: Testing
hide: navigation
---
<!--
SPDX-FileCopyrightText: 2025 German Aerospace Center (DLR)
SPDX-License-Identifier: CC-BY-NC-4.0
-->

<!-- markdownlint-disable-next-line MD025 -->
# Testing

This projects provides a few different tests and checks.

In general, we have two files which defines all the tests:

- [.pre-commit-config.yaml](https://github.com/DLR-KI/md-multiline-table/blob/main/.pre-commit-config.yaml)
- [.github/workflows/main.yml](https://github.com/DLR-KI/md-multiline-table/blob/main/.github/workflows/main.yml)
<!--
- [.gitlab-ci.yml](https://github.com/DLR-KI/md-multiline-table/blob/main/.gitlab-ci.yml)
-->

To easily run these tests locally, use:

```bash
./scripts/test.sh
```

If you only want to run the pre-commit hooks manually, use:

```bash
pre-commit run --all-files
```

Running the GitLab CI locally is a bit more complicated.
Is also requires Node.js as well as Docker installed and configured.

```bash
npm exec gitlab-ci-local
# or run a single job, e.g. pre-commit
npm exec gitlab-ci-local -- pre-commit
```

## Python Code Tests

Static code checks using [ruff](https://docs.astral.sh/ruff):

```bash
# linter
python -m ruff check slki
python -m ruff check scripts
# formatter
python -m ruff format --diff slki
python -m ruff format --diff scripts
```

To run automatic static code fixes, use:

```bash
# linter
python -m ruff check --fix slki
python -m ruff check --fix scripts
# formatter
python -m ruff format slki
python -m ruff format scripts
```

Static type checks using [mypy](https://www.mypy-lang.org/):

```bash
python -m mypy slki
python -m mypy scripts
```

## Bash Script Checks

```bash
find scripts -type f -name "*.sh" -exec shellcheck --external-sources --shell bash --source-path scripts {} +
```

## Markdown Checks

Verify documentation (markdown) compliance w.r.t. [markdown linting rules](https://github.com/DavidAnson/markdownlint#rules--aliases) further specified inside the [.markdownlint-cli2.jsonc](http://github.com/DLR-KI/md-multiline-table/blob/main/.markdownlint-cli2.jsonc) configuration file.

```bash
npm exec markdownlint-cli2 -- "./docs/**/*.md" "./README.md"
```

## License Checks

```bash
python -m licensecheck
python -m reuse lint
```

## Vulnerability Checks

```bash
python -m tox --recreate -e vulnerability
```
