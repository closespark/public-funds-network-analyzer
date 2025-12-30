# Risk Scoring Methodology

This document explains the risk scoring system used by the Public Funds Network Analyzer.

## Important Disclaimer

**⚠️ RISK SCORES ARE NOT ACCUSATIONS**

A risk score represents structural patterns that may warrant further investigation. It does NOT indicate:
- Fraud
- Wrongdoing
- Illegal activity
- Any accusation of any kind

High scores simply mean more patterns were detected that a human investigator may want to review.

## Scoring Principles

### Transparency

All weights are documented and publicly accessible. There are no hidden factors.

### Reproducibility

Any score can be manually verified by:
1. Reviewing the patterns that triggered
2. Looking up the weight for each pattern
3. Adding the weights together

### Additivity

Scores are computed by adding pattern weights. Each pattern contributes independently.

## Default Pattern Weights

| Pattern Type | LOW | MEDIUM | HIGH |
|-------------|-----|--------|------|
| Address Density | 5 | 10 | 15 |
| Name Similarity Cluster | 3 | 8 | 12 |
| Vendor Concentration | 5 | 15 | 25 |
| Prime-Subaward Fan-out | 3 | 8 | 15 |
| Transaction Timing Spike | 3 | 7 | 12 |
| Delinquent Property Overlap | 10 | 20 | 30 |
| Cross-Jurisdiction Presence | 2 | 5 | 10 |

### Severity Levels

- **LOW**: Pattern observed but weak signal
- **MEDIUM**: Pattern is notable and worth review
- **HIGH**: Strong pattern that warrants investigation

## Score Calculation

```
Final Score = Base Score (0) + Σ(Pattern Weights)
Maximum Score = 100
```

### Example

Entity "ABC Corp" has the following detected patterns:
- Address Density (MEDIUM): +10
- Delinquent Property Overlap (HIGH): +30
- Cross-Jurisdiction (LOW): +2

**Total Score: 42**

## Score Interpretation

| Score Range | Interpretation |
|-------------|---------------|
| 0-20 | Few patterns detected |
| 20-40 | Some patterns worth noting |
| 40-60 | Multiple patterns warrant review |
| 60-80 | Significant patterns detected |
| 80-100 | Many strong patterns present |

**Remember**: Even a score of 100 is NOT proof of wrongdoing. It simply means many patterns were detected that a human should review.

## Customization

Weights can be customized by passing a custom weight dictionary to the `RiskScoringEngine`:

```python
from src.scoring.engine import RiskScoringEngine, PatternType

custom_weights = {
    PatternType.VENDOR_CONCENTRATION: {
        "LOW": 10,
        "MEDIUM": 25,
        "HIGH": 40,
    },
    # ... other patterns
}

engine = RiskScoringEngine(pattern_weights=custom_weights)
```

## Verification Steps

For any scored entity, verify the score by:

1. Get the entity's triggered signals from the score output
2. For each signal, note the pattern type and severity
3. Look up the weight in the table above
4. Sum all weights
5. Compare to the reported score (should match)

## Limitations

- Scores are based only on detected patterns
- Patterns may have innocent explanations
- False positives are possible
- Low scores don't guarantee clean operations
- Context matters - always investigate before concluding
