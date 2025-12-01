# Installation

ArthroScape uses [Poetry](https://python-poetry.org/) for dependency management. This ensures a reproducible environment without polluting your system python.

## Prerequisites

* **Python**: Version >= 3.10 and < 3.13
* **Poetry**: [Installation Instructions](https://python-poetry.org/docs/#installation)
* **Git**: For version control.

## Steps

1. **Clone the repository:**

    ```bash
    git clone https://github.com/neurorishika/ArthroScape.git
    cd ArthroScape
    ```

2. **Install dependencies:**

    Run the following command in the project root to install all required packages into a virtual environment managed by Poetry.

    ```bash
    poetry install
    ```

    This will read `pyproject.toml` and `poetry.lock` to install the exact versions of dependencies.

3. **Verify Installation:**

    You can verify the installation by running the tests (if available) or checking the package version.

    ```bash
    poetry run python -c "import arthroscape; print(arthroscape.__file__)"
    ```

## VS Code Setup

If you use [VS Code](https://code.visualstudio.com/):

1. Open the project folder in VS Code.
2. When opening a Python file, select the Python interpreter.
3. Choose the interpreter associated with the Poetry virtual environment (usually located in `.venv` or a path managed by Poetry).

## Jupyter Notebooks

To use the project's environment in Jupyter Notebooks:

```bash
poetry run jupyter notebook
```

This ensures that the kernel has access to the `arthroscape` package and all its dependencies.
