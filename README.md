# QTS-ARCHITECT
Institutional-grade Quantitative Trading System.

This repo is structured to run everything from the repository root (tooling, configs, tests, and Docker).

## Quick start (local)
Prereqs: Python 3.10+, Poetry.

```bash
# Install deps
poetry install

# Run tests
poetry run pytest

# Run (Hydra) default entrypoint
poetry run python -m qts_core.main

# Run live trader (paper mode)
poetry run python -m qts_core.main_live
```

## Configuration (Hydra)
All runtime configuration is under `conf/` (repo root). Use Hydra overrides at the CLI.

## Repo layout (high level)
* `conf/`: Hydra YAML configs (canonical)
* `qts_core/src/qts_core/`: Python package
* `tests/`: canonical test suite
* `docs/`: documentation (and `docs/prototypes/` for experimental artifacts)

## Notes
* Hydra generates `outputs/` and `multirun/` by default; these are ignored via `.gitignore`.
