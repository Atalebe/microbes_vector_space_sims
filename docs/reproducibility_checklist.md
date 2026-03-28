# Reproducibility checklist

## Every run must capture

- raw input filename and checksum,
- exact YAML config used,
- package versions,
- script sequence used,
- output tables and figures,
- notes on dropped rows and quality flags.

## Repo setup rule

The first Git commit should already contain:

- folder structure,
- requirements file,
- starter config,
- scripts,
- dataset map,
- this checklist.

That way the branch starts reproducible instead of trying to become reproducible later.
