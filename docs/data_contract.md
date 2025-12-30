# Data Contract

This document defines the data handling rules for the Public Funds Network Analyzer.

## Core Principles

### 1. Data Immutability

**Raw data in `data/` is immutable and MUST never be modified.**

- Source files are treated as read-only
- No in-place modifications to any file in `data/`
- All transformations create new columns/files rather than modifying existing ones

### 2. Raw Value Preservation

**Original values are always preserved alongside any transformations.**

For every normalization or transformation:
- The raw column/value is kept intact
- Normalized values are stored in new columns (e.g., `{column}_normalized`)
- Computed IDs are stored in new columns (e.g., `{column}_id`)

Example:
```
Original: "ACME Corporation, Inc."
Normalized: "ACME"
Entity ID: "abc123def456"
```

All three values are retained in the output.

### 3. Reproducibility

**All outputs must be reproducible from source data.**

- Outputs in `outputs/` can be regenerated from `data/`
- Run metadata includes timestamp and input files
- Deterministic hashing ensures consistent IDs across runs
- No randomness in any processing step

### 4. Transparency

**All processing steps are documented and auditable.**

- Every join records: method, confidence level, fields used
- Every pattern records: trigger rules, evidence, source rows
- Every score records: triggered signals, weights, datasets
- Weights and thresholds are documented, not hidden

## Directory Structure

```
data/           # Immutable source data (read-only)
├── *.csv       # Source CSV files
└── README.md   # Data source documentation

outputs/        # Generated outputs (reproducible)
├── *_leads.csv/.json    # Investigation leads
├── *_scores.csv/.json   # Entity risk scores
├── *_graph_nodes.csv    # Graph node list
├── *_graph_edges.csv    # Graph edge list
├── *_patterns.csv       # Detected patterns
├── *_summary.txt/.json  # Summary statistics
└── *_metadata.json      # Run metadata
```

## Data Flow

```
data/ (immutable)
    │
    ▼
┌─────────────────────────────────────┐
│ 1. INGESTION                        │
│    - Load CSV files                 │
│    - Record metadata (source, rows) │
│    - Preserve raw column names      │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ 2. NORMALIZATION                    │
│    - Normalize names/addresses      │
│    - Generate deterministic IDs     │
│    - Store raw + normalized         │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ 3. GRAPH CONSTRUCTION               │
│    - Create nodes (Entity, etc.)    │
│    - Create edges (relationships)   │
│    - Record source for each edge    │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ 4. PATTERN DETECTION                │
│    - Detect structural signals      │
│    - Emit evidence with references  │
│    - No accusations, only patterns  │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ 5. RISK SCORING                     │
│    - Additive scoring (0-100)       │
│    - Transparent weights            │
│    - Score ≠ verdict                │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ 6. LEAD GENERATION                  │
│    - Generate investigation leads   │
│    - No accusatory language         │
│    - Include verification steps     │
└─────────────────────────────────────┘
    │
    ▼
outputs/ (reproducible)
```

## Schema Mappings

Each data source has a defined mapping from raw columns to standardized fields.

### Business Licenses
| Standard Field | Source Column |
|---------------|---------------|
| entity_name | Business Name |
| dba_name | Doing Business As |
| address | Business Address |
| geo_location | Business Geo Location |

### City Contracts
| Standard Field | Source Column |
|---------------|---------------|
| agency | Agency/Department |
| contract_number | Contract Number |
| contract_value | Contract Value |
| supplier_name | Supplier |
| ... | ... |

### Delinquent Properties
| Standard Field | Source Column |
|---------------|---------------|
| property_code | Property Code |
| owner_name_1 | Current Owner Name 1 |
| address | Physical Address |
| total_due | Total Due |
| ... | ... |

### Federal Awards
Federal data uses USAspending column conventions:
- `prime_awardee_uei` - Unique Entity Identifier
- `prime_awardee_name` - Recipient name
- `prime_award_unique_key` - Award identifier
- etc.

## Validation

To verify data contract compliance:

1. **No modifications in data/**: Check git status for changes
2. **Raw preserved**: Verify original columns exist in output
3. **Reproducible**: Re-run pipeline and compare outputs
4. **Documented**: Check metadata.json for complete run information
