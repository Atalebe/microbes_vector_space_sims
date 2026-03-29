
# microbes_vector_space_sims

Microbial homeostatic vector-space simulations and manuscript support code.

This repository builds and analyzes three microbial branches:

- **GSE36599**  
  Staged yeast stress, adaptation, and recovery benchmark branch.

- **GSE4370**  
  Continuous *E. coli* recovery backbone branch.

- **GSE206609**  
  Explicit drug-dependent recoverability branch in *E. coli*.

The workflow is organized around reproducible stages:

1. dataset download and ingestion  
2. audit and PCA geometry  
3. state-table construction in \(H, S, M, R\)  
4. residual field or explicit recoverability analysis  
5. validation, vector velocity, and variance scaling  
6. cross-dataset synthesis and manuscript summary products

## Repository structure

```text
configs/                    YAML configs for each dataset
figures/                    curated manuscript-ready PNG figures
scripts/                    helper shell scripts, including GEO download script
src/                        analysis scripts
src/ingest/                 ingestion scripts
data/raw/                   raw downloaded GEO files, not tracked
data/interim/               intermediate tables, not tracked
data/processed/             processed tables, not tracked
results/                    logs and derived outputs, not tracked

Tracked contents

This GitHub repository tracks:

analysis scripts
ingestion scripts
configuration files
curated manuscript-ready figures
top-level documentation

Large raw downloads and generated outputs are intentionally excluded from version control.

Environment

Create and activate a virtual environment, then install requirements:

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
Data download

The GEO targets can be downloaded with:

bash scripts/download_geo_targets.sh

Downloaded raw files should be placed under data/raw/.

Main script families
Ingestion
src/ingest/ingest_gse36599.py
src/ingest/ingest_gse4370.py
src/ingest/ingest_gse206609.py
Audit
src/01_audit_gse36599.py
src/01_audit_gse4370.py
src/01_audit_gse206609.py
State tables
src/02_build_state_table_gse36599.py
src/02_build_state_table_gse4370.py
src/02_build_state_table_gse206609.py
Core branch analyses
src/03_residual_field_gse36599.py
src/03_residual_field_gse4370.py
src/03_build_explicit_R_gse206609.py
Validation and hardening
src/05_validate_gse36599.py
src/05_validate_gse4370.py
src/05_validate_gse206609.py
src/06_vector_velocity_gse36599.py
src/06_vector_velocity_gse4370.py
src/06_vector_velocity_gse206609.py
src/07_variance_scaling_gse36599.py
src/07_variance_scaling_gse4370.py
src/07_variance_scaling_gse206609.py
Cross-dataset synthesis
src/21_cross_dataset_microbes_synthesis_v2.py
src/22_project_gse206609_onto_gse4370_backbone.py
src/23_plot_gse206609_on_gse4370_backbone.py
src/24_build_microbes_paper_summary_table.py
Minimal execution outline
GSE36599
python src/ingest/ingest_gse36599.py configs/gse36599.yaml
python src/01_audit_gse36599.py configs/gse36599.yaml
python src/02_build_state_table_gse36599.py configs/gse36599.yaml
python src/03_residual_field_gse36599.py configs/gse36599.yaml
python src/04_plot_gse36599.py configs/gse36599.yaml
python src/05_validate_gse36599.py configs/gse36599.yaml
python src/06_vector_velocity_gse36599.py configs/gse36599.yaml
python src/07_variance_scaling_gse36599.py configs/gse36599.yaml
GSE4370
python src/ingest/ingest_gse4370.py configs/gse4370.yaml
python src/01_audit_gse4370.py configs/gse4370.yaml
python src/02_build_state_table_gse4370.py configs/gse4370.yaml
python src/03_residual_field_gse4370.py configs/gse4370.yaml
python src/04_plot_gse4370.py configs/gse4370.yaml
python src/05_validate_gse4370.py configs/gse4370.yaml
python src/06_vector_velocity_gse4370.py configs/gse4370.yaml
python src/07_variance_scaling_gse4370.py configs/gse4370.yaml
GSE206609
python src/ingest/ingest_gse206609.py configs/gse206609.yaml
python src/01_audit_gse206609.py configs/gse206609.yaml
python src/02_build_state_table_gse206609.py configs/gse206609.yaml
python src/03_build_explicit_R_gse206609.py configs/gse206609.yaml
python src/04_plot_gse206609.py configs/gse206609.yaml
python src/05_validate_gse206609.py configs/gse206609.yaml
python src/06_vector_velocity_gse206609.py configs/gse206609.yaml
python src/07_variance_scaling_gse206609.py configs/gse206609.yaml
python src/09_finalize_explicit_R_gse206609.py configs/gse206609.yaml
Notes
Raw GEO downloads are excluded from version control.
Intermediate and processed tables are excluded from version control.
The figures/ directory is intentionally tracked because it contains curated manuscript-ready PNGs used in the paper.
The repository is designed for milestone-level reproducibility, so each branch can be rerun independently before cross-dataset synthesis.
