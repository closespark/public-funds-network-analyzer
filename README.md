# public-funds-network-analyzer

A transparent, data-driven system for analyzing federal, state, and municipal spending alongside business licenses and property records to surface structural risk patterns for journalists, auditors, and civic investigators—without making allegations.

## Project Structure

```
├── data/           # Raw inputs (immutable - never modify source data)
├── src/            # Source code
├── outputs/        # Generated outputs (reproducible from data/)
├── docs/           # Documentation
└── README.md
```

## Data Contract

### Immutability Rule

**Raw data in `data/` is immutable and must never be modified.**

- All source files in `data/` are treated as read-only
- Raw column names and values are always preserved
- Processing reads from `data/` and writes only to `outputs/`
- Any transformations store both raw and normalized values side-by-side

### Reproducibility

- All outputs in `outputs/` can be regenerated from `data/`
- Every output includes metadata: timestamp, inputs used, row counts
- No manual edits to generated files

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full pipeline
python -m src.main

# Run individual modules
python -m src.ingestion.loader      # Load data
python -m src.normalization.engine  # Normalize entities/addresses
python -m src.joins.engine          # Join datasets
python -m src.graph.builder         # Build graph
python -m src.patterns.detector     # Detect patterns
python -m src.scoring.engine        # Score risks
python -m src.leads.generator       # Generate leads
python -m src.output.reporter       # Export reports
```

## Supported Data Sources

- **Business Licenses**: Local business registration data
- **City Contracts**: Municipal procurement records
- **Delinquent Properties**: Tax delinquent real estate records
- **Federal Prime Awards**: USAspending contracts and assistance
- **Federal Transactions**: Transaction-level spending data
- **Federal Subawards**: Subcontract and subgrant data

## Risk Scoring Disclaimer

**Scores represent structural patterns, not accusations.**

- Risk scores (0-100) are additive and transparent
- Every score can be manually recomputed from documented weights
- No hidden factors or opaque algorithms
- A high score indicates patterns worth investigating, not wrongdoing
