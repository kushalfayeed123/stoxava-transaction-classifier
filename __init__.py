"""Transaction classification module.

Classifies Plaid-sourced transactions (from potentially several accounts,
with varying schemas) into a client-defined taxonomy using an LLM-backed
classifier, with a deterministic mock backend for offline demos.

This package is run as a flat script/service (see main.py, `uvicorn
main:app`), not imported as `transaction_classifier.*` -- all internal
imports are absolute (`from classifier import ...`) to match. This file is
kept only to mark the directory as a package for tooling (e.g. hatchling
build), not for `from . import ...` style relative imports.
"""

__version__ = "0.2.0"