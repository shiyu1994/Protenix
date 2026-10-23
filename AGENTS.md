# Repository Guidelines

## Project Structure & Module Organization

Protenix is a Python 3.11 package. Core library code lives in `protenix/`, with major areas split into `data/`, `model/`, `metrics/`, `config/`, `utils/`, `tfg/`, and `web_service/`. CLI and executable workflows are in `runner/`, including inference, training, and batch inference entrypoints. Configuration presets live in `configs/`. Utility and data-preparation scripts are in `scripts/`, with database and MSA helpers under `scripts/database/` and `scripts/msa/`. Tests live in `tests/`; documentation in `docs/`; sample inputs in `examples/`; figures and license templates in `assets/`.

## Build, Test, and Development Commands

- `pip install -e .`: install the package locally.
- `pip install -r requirements.txt`: install pinned dependencies.
- `pip install pre-commit && pre-commit install`: enable formatting, linting, license, and docstring hooks.
- `pytest tests/`: run the CI suite.
- `python -m unittest tests/test_utils.py`: run one unittest-style file.
- `protenix pred -i examples/input.json -o ./output -n protenix_base_default_v1.0.0`: smoke-test CLI inference.
- `bash inference_demo.sh` or `bash train_demo.sh`: run examples after configuring data paths.

## Coding Style & Naming Conventions

Use 4-space indentation and standard Python naming: `snake_case` for functions, methods, modules, and variables; `PascalCase` for classes; uppercase for constants. Keep lines within the `flake8` limit of 120 characters. Python and shell files must keep the Apache license header; pre-commit can insert it from `assets/license_header.txt`. Formatting uses `ufmt` with Black/usort; linting uses `flake8`; docstrings use `pydoclint`.

## Testing Guidelines

Add focused tests under `tests/` using the `test_*.py` pattern. Most tests use `unittest.TestCase`; `pytest` is the CI runner. Mark CUDA, network, or large-data tests so they skip cleanly when dependencies are unavailable. Prefer deterministic fixtures from `examples/` and avoid bulky generated outputs.

## Commit & Pull Request Guidelines

Recent history uses short imperative summaries, often `feat: ...`, `[fix] ...`, or `[refactor] ...`. Follow `CONTRIBUTING.md`: keep PRs small and functional, use a `subject: message` title, describe intent and validation, link issues, and include docs or tests for behavior changes. The project squashes PRs, so make the title suitable as the final commit summary.

## Security & Configuration Tips

Do not commit model weights, private datasets, generated `output/` or `test_outputs/`, local logs, or credentials. Set `PROTENIX_ROOT_DIR` and `CUTLASS_PATH` in your shell or job script. Template and RNA MSA workflows may require `kalign` and `hmmer`; document those assumptions in dependent PRs.
