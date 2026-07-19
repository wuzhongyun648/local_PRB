# Contract Tests

These tests define the minimum interface and state invariants for the seven
active loaders and graph managers. They do not train a model or run an online
experiment.

Run them from the repository root:

```bash
python -m unittest discover -s tests -v
```

Known correctness blockers are marked with `unittest.expectedFailure`. An
unexpected success is treated as a failing suite, so the marker must be removed
when the corresponding implementation is fixed.
