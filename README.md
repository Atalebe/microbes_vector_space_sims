# microbes_vector_space_sims

Reproducible simulation scaffold for the microbial branch of the homeostatic vector space program.

## Scope

This repository is designed to:

1. audit a microbial perturbation or recovery dataset,
2. define a first-pass `(H, S, M)` state table,
3. map baseline geometry,
4. extract a latent recoverability field,
5. compare latent recoverability to explicit recovery observables,
6. measure trajectory velocity and directional drift,
7. run robustness checks,
8. close with variance scaling.

## Repository layout

- `configs/` runtime YAML files
- `data/raw/` raw input tables placed here by the user
- `data/interim/` cleaned intermediate files
- `data/processed/` normalized state tables and derived outputs
- `results/figures/` plots
- `results/tables/` CSV summaries
- `results/logs/` run logs
- `src/` analysis modules
- `docs/` design notes and dataset map
- `scripts/` helper shell entrypoints

## First-pass workflow

```bash
python run_script.py src/01_audit_dataset.py configs/starter_ecoli_heatshock.yaml
python run_script.py src/02_build_state_table.py configs/starter_ecoli_heatshock.yaml
python run_script.py src/03_map_geometry.py configs/starter_ecoli_heatshock.yaml
python run_script.py src/04_fit_residual_field.py configs/starter_ecoli_heatshock.yaml
python run_script.py src/05_build_recoverability.py configs/starter_ecoli_heatshock.yaml
python run_script.py src/06_trajectory_velocity.py configs/starter_ecoli_heatshock.yaml
python run_script.py src/07_validation.py configs/starter_ecoli_heatshock.yaml
python run_script.py src/08_variance_scaling.py configs/starter_ecoli_heatshock.yaml
```

## Input expectation

The pipeline assumes a tidy table with one row per observation and columns such as:

- sample identifier
- organism or strain identifier
- condition or perturbation label
- timepoint
- one baseline proxy for `H`
- one stability proxy for `S`
- one memory proxy for `M`
- optional explicit recovery proxy for `R`

The column names are mapped in the YAML config.

## Minimal reproducibility rule

Every simulation run should be tied to:

- one config file,
- one immutable raw input table,
- one recorded package environment,
- one results folder with logs, tables, and figures.

If a run cannot be reconstructed from those parts, it is not finished.
