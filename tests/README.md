# Test Suite for Evidence Framework

## Setup

```bash
cd D:/home/ignat/project-third-matter/dft-neb/ph-diagnostic
.venv/Scripts/python.exe -m pytest tests/ -v
```

Or: `bash tmp/pytest_run.sh`

## Test Files

| File | Coverage target | Test count |
|------|----------------|------------|
| test_ph_neb_diagnostic.py | PH library (gudhi wrappers, bottleneck, Wasserstein, summary) | 13 |
| test_master_equation.py | Kinetics: rate matrix, equilibrium, Arrhenius, Chu-Liu-Edmonds | 16 |
| test_string_method.py | Vanden-Eijnden 2007: init, reparametrize, convergence, tangent | 8 |
| test_symmetry_preflight.py | apply_R, Hungarian matching, find_qualifying_op | 7 |
| test_enumeration.py | mic_vec, remove_atom_and_add_H | 7 |
| **TOTAL** | **5 modules** | **58 tests** |

## Coverage (as of 2026-05-28)

| Module | Lines covered | Why some uncovered |
|--------|---------------|---------------------|
| ph_neb_diagnostic.py | 96% | Core library, well-tested |
| master_equation_kinetics.py | 55% | main() benchmark + plot не covered (those are CLI/visual) |
| string_method_prototype.py | 42% | benchmark + plot functions not unit-tested |
| symmetry_preflight_general.py | 35% | run_test() + main() are I/O wrappers |
| multi_endpoint_enumeration.py | 17% | enumerate_sites() writes xyz files (integration test) |

**Total core coverage: 42%.** Algorithmic logic well-tested; CLI/plot/I/O wrappers intentionally not unit-tested (those should be tested via integration tests с real data when available).

## Test Categories

- **Mathematical correctness:** invariants (probability conservation, symmetry, normalization)
- **Algorithmic verification:** known small cases (2-state symmetric → recovers input barrier)
- **Edge cases:** empty diagrams, identical inputs, single atoms
- **Integration:** end-to-end mini-tests (analyze_network through whole pipeline)

## Markers

- `@pytest.mark.slow` — convergence tests (~5-10 sec each)
- `@pytest.mark.integration` — multi-module tests
- `@pytest.mark.requires_data` — needs harvested DFT data (currently 0)

Run без slow: `pytest -m 'not slow'`

## Fixtures (conftest.py)

- `simple_2d_grid` — single-min potential V = x² + y²
- `double_well_grid` — V = (x²-1)² + y²
- `random_cloud_2d` — Gaussian points с known V
- `symmetric_barrier_matrix` — 2-state 43 meV (mack analog)
- `asymmetric_barrier_matrix` — 2-state с ΔE=174 meV (marc analog)
- `three_state_chain` — linear 3-state network

## What's NOT Covered

1. **DFT/MLIP-dependent code paths** — нельзя unit-test без compute. Validated через integration runs.
2. **CLI main() wrappers** — argparse code, not algorithmically interesting.
3. **Plotting** — visual output, not testable algorithmically.
4. **Real chemistry checks** — those require known DFT/experimental reference data.

## Adding New Tests

1. Put new tests в `tests/test_<module>.py`
2. Use fixtures from `conftest.py` (or add new fixtures there)
3. Run `bash tmp/pytest_run.sh` to verify
4. Update этот README с new count

## Why Test Coverage Matters Now

Code stabilized после ADMM-NEB retraction + multiple methodology iterations. **Tests prevent regression** when:
- Refactoring (e.g., L0 over-aggressive cubane criterion)
- Adding new minerals to validation set
- Optimizing performance
- Production wrapping (ASE Optimizer subclass для string method)

Tests are **light protective layer** for paper-grade reproducibility.
