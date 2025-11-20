<!--
SPDX-FileCopyrightText: 2025 German Aerospace Center (DLR)
SPDX-License-Identifier: CC-BY-NC-4.0
-->

# SLKI

SLKI (Second Level KI in Weichen) aims to improve the security and reliability of Germany's rail network by harnessing the potential of fixed acceleration sensor data at train switches.

One of the main challenges is to clean and preprocess these noisy time-series data to obtain high-quality and reliable train signal data.

Based on these cleaned signal data it is then possible to provide AI-driven approaches to classify train types, track and predict train speeds, uncover potential anomalies.

A good starting point, to learn more about the project and this repository, is the doumentation published on the [GitHub Pages](https://dlr-ki.github.io/slki).

## Repository Structure

The repository is organized as follows:

- [docs](docs): Markdown based user documentation published on the [GitHub Pages](https://dlr-ki.github.io/slki).
- [LICENSES](LICENSES): All license files used somewhere in this project. Also see [LICENSES.md](LICENSES.md).
- [logs](logs): Optional directory for log files.
- [slki](slki): Signal processing and cleaning pipeline source code. Inclusing its [configuration](slki/config.py) file.
- [notebooks](notebooks): Jupyter notebooks to further analyze the processed singal data.
- [scripts](scripts): Scripts extending this project.

## Requirements

- Python 3.10 or later as well as pip
- virtualenv or venv (highly recommended)

    ```bash
    pip install -U virtualenv
    ```

## Install

There are multiple optional dependencies available:

- `dev`: installs additional development tools
- `notebooks`: installs additional requirements to run the jupyter notebooks
- `torch`: installs PyTorch
- `test`: installs test requirements
- `stubs`: installs further type information
- `docs`: installs documentation requirements
- `all`: installs all optional dependencies

Of course, it is possible to install the software without any additional optional dependencies and just use the signal processing and cleaning pipeline.
Choose your poision based on your own requirements.

```bash
# create virtual environment
virtualenv -p $(which python3.10) .venv
# or
# python -m venv .venv

# activate our virtual environment
source .venv/bin/activate

# update pip (optional)
python -m pip install -U pip

# install
pip install -U -e ".[all]"

# enable git pre-commit hooks (optional)
pre-commit install
```

## Usage

### Signal processing and cleaning pipeline

1. adjust config file: `slki/config.py`
2. run the pipeline

    ```bash
    python -m slki
    # or just
    slki
    ```

### Notebooks

1. ensure that Jupyter Lab is installed

    ```bash
    pip install jupyterlab
    ```

2. open Jupyter Lab

    ```bash
    jupyter lab --notebook-dir=notebooks
    ```

### Python Code

```python
from slki import ...
```

## Testing

This projects provides a few different tests and checks.

> [!Note]
> For detailed information about the tests, please check out the
> [Testing](https://dlr-ki.github.io/slki/testing) section in the documentation.

In gerneral, we have two files which defines all the tests:

- [.pre-commit-config.yaml](.pre-commit-config.yaml)
- [.github/workflows/main.yml](.github/workflows/main.yml)
<!--
- [.gitlab-ci.yml](.gitlab-ci.yml)
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

## Contribution

Please follow the contribution rules:

- use typed Python (type annotations)
- verify Python static code checks with ruff
- document fixes, enhancements, new features, ...
- write scripts and examples OS independent or at least with linux, wsl support
- verify shell script static code check compliancy with [ShellCheck](https://www.shellcheck.net/wiki/)
- verify project license compliancy witout any license conflicts (e.g. for 3rd party libraries, data, models, ...)
- verify documentation (markdown) compliancy w.r.t. [markdown linting rules](https://github.com/DavidAnson/markdownlint#rules--aliases) further specified inside the [.markdownlint-cli2.jsonc](.markdownlint-cli2.jsonc) configuration file
- run all tests successfully

### Documentation

This projects is using the Docstring style from [Google](https://github.com/google/styleguide/blob/gh-pages/pyguide.md#38-comments-and-docstrings).
At least public classes, methods, fields, ... should be documented.

For further documentation we are using [Markdown](https://www.markdownguide.org/) documentation with [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/).
See the [docs](docs) folder for more details.

To locally serve the documentation, feel free to use:

```bash
python -m mkdocs serve
```

### Contributors

<!--
- Lange, Markus
- Heinrich, Florian
- Nakano, Tamon
- Petrausch, Tobias
-->

<a href="https://github.com/mlange01">
  <img
    src="https://images.weserv.nl/?url=avatars.githubusercontent.com/u/69654727?v=4&h=80&w=80&fit=cover&mask=circle&maxage=7d"
    alt="Markus Lange"
    title="Markus Lange"
  />
</a>
<a href="https://github.com/HeinrichAD">
  <img
    src="https://images.weserv.nl/?url=avatars.githubusercontent.com/u/5962361?v=4&h=80&w=80&fit=cover&mask=circle&maxage=7d"
    alt="Florian Heinrich"
    title="Florian Heinrich"
  />
</a>
<a href="https://github.com/tnakano29">
  <img
    src="https://images.weserv.nl/?url=avatars.githubusercontent.com/u/96628134?v=4&h=80&w=80&fit=cover&mask=circle&maxage=7d"
    alt="Tamon Nakano"
    title="Tamon Nakano"
  />
</a>
<a href="https://github.com/petratnw">
  <img
    src="https://images.weserv.nl/?url=avatars.githubusercontent.com/u/178803829?v=4&h=80&w=80&fit=cover&mask=circle&maxage=7d"
    alt="Tobias Petrausch"
    title="Tobias Petrausch"
  />
</a>

## Citation

For accurate citation, refer to the corresponding metadata in the [CITATION.cff](CITATION.cff) file associated with this work.

## License

Please see the file [LICENSE.md](LICENSE.md) for further information about how the content is licensed.
