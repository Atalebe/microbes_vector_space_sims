# Dataset map for the first microbial branch

## Selection rule

Rank candidates using these five filters:

1. baseline, perturbation, and recovery are all present,
2. repeated measurements or staged timepoints are present,
3. variables are interpretable enough for minimal proxies,
4. there is enough breadth to reveal geometry,
5. an explicit recovery variable exists or can be built.

## Best first-pass target families

### Tier 1, strongest direct-start candidates

1. **Escherichia coli heat shock or temperature upshift time series**
   - Why it is strong:
     - simple organism,
     - explicit stress event,
     - time-ordered measurements,
     - natural path toward displacement and recovery analysis.
   - Likely use:
     - first pass geometry,
     - latent recoverability,
     - trajectory velocity.

2. **Yeast stress adaptation and stress recovery series**
   - Why it is strong:
     - adaptation and recovery are both explicit,
     - repeated timepoints are available,
     - memory or conditioning is easier to operationalize.
   - Likely use:
     - first serious latent versus explicit recoverability comparison,
     - conditioning-dependent memory tests.

3. **E. coli stationary phase exit or starvation/recovery data**
   - Why it is strong:
     - recovery and regrowth are biologically direct,
     - good setting for burdened versus restorative regimes,
     - useful for explicit recovery time definitions.

### Tier 2, useful but more delicate

4. **Trimethoprim or other antibiotic response time series in E. coli**
   - Strong for collapse and burdened frontier logic.
   - May be weaker if recovery windows are shallow or absent.

5. **Yeast ethanol stress survival and transcriptome timepoints**
   - Strong for stress tolerance classes.
   - Better for explicit survival than full physiological recovery unless paired with regrowth readouts.

6. **Osmotic or oxidative stress recovery data**
   - Very relevant biologically.
   - Often excellent if viability and regrowth are both present.
   - Needs careful timescale handling.

## Candidate public sources

- NCBI GEO
- NCBI SRA
- ArrayExpress
- PRIDE, if proteomics are central
- supplementary tables from time-series microbial stress papers

## Minimum viable starting table

The starter table should ideally include:

- `sample_id`
- `strain`
- `condition`
- `time_hours`
- `trajectory_id`
- `growth_capacity`
- `viability_fraction`
- `conditioning_score`
- `recovery_fraction`

## Recommended first practical order

1. Pull 2 to 3 candidate datasets.
2. Build a one-page audit sheet for each.
3. Score them against the five selection filters.
4. Choose one as the primary branch.
5. Keep one backup dataset ready in case the winner turns out to be too sparse.
