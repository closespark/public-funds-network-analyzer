"""Risk scoring engine.

Implements additive scoring (0-100) with transparent weights.
Stores triggered signals, weights, and datasets involved.
Generates explanations per score.

IMPORTANT: Score ≠ verdict. Scores indicate patterns worth investigating,
not wrongdoing.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import pandas as pd

from ..patterns.detector import PatternEvidence, PatternType


# Default weights for each pattern type (0-100 scale contributions)
DEFAULT_PATTERN_WEIGHTS = {
    PatternType.ADDRESS_DENSITY: {
        "LOW": 5,
        "MEDIUM": 10,
        "HIGH": 15,
    },
    PatternType.NAME_SIMILARITY_CLUSTER: {
        "LOW": 3,
        "MEDIUM": 8,
        "HIGH": 12,
    },
    PatternType.VENDOR_CONCENTRATION: {
        "LOW": 5,
        "MEDIUM": 15,
        "HIGH": 25,
    },
    PatternType.PRIME_SUBAWARD_FANOUT: {
        "LOW": 3,
        "MEDIUM": 8,
        "HIGH": 15,
    },
    PatternType.TRANSACTION_TIMING_SPIKE: {
        "LOW": 3,
        "MEDIUM": 7,
        "HIGH": 12,
    },
    PatternType.DELINQUENT_PROPERTY_OVERLAP: {
        "LOW": 10,
        "MEDIUM": 20,
        "HIGH": 30,
    },
    PatternType.CROSS_JURISDICTION: {
        "LOW": 2,
        "MEDIUM": 5,
        "HIGH": 10,
    },
}


@dataclass
class ScoredEntity:
    """An entity with a risk score."""
    
    entity_id: str
    entity_name: str
    score: float
    triggered_signals: list[dict[str, Any]]
    datasets_involved: list[str]
    explanation: str
    scored_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "entity_id": self.entity_id,
            "entity_name": self.entity_name,
            "score": round(self.score, 2),
            "triggered_signals": self.triggered_signals,
            "datasets_involved": self.datasets_involved,
            "explanation": self.explanation,
            "scored_at": self.scored_at.isoformat(),
        }


@dataclass
class ScoreBreakdown:
    """Detailed breakdown of how a score was computed."""
    
    entity_id: str
    base_score: float
    pattern_contributions: list[dict[str, Any]]
    final_score: float
    weight_source: str = "default"
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "entity_id": self.entity_id,
            "base_score": self.base_score,
            "pattern_contributions": self.pattern_contributions,
            "final_score": round(self.final_score, 2),
            "weight_source": self.weight_source,
        }


class RiskScoringEngine:
    """Engine for computing risk scores.
    
    DISCLAIMER: Risk scores (0-100) represent structural patterns that may
    warrant further investigation. A score does NOT indicate wrongdoing,
    fraud, or any illegal activity. All scores can be manually verified
    using the documented weights and evidence.
    """
    
    DISCLAIMER = (
        "RISK SCORE DISCLAIMER: Scores represent structural patterns worth "
        "investigating, NOT accusations of wrongdoing. Every score can be "
        "manually recomputed using the documented weights and evidence. "
        "A high score indicates patterns that warrant further investigation, "
        "not a verdict of any kind."
    )
    
    def __init__(
        self,
        pattern_weights: dict = None,
        max_score: float = 100.0,
        base_score: float = 0.0,
    ):
        """Initialize scoring engine.
        
        Args:
            pattern_weights: Custom weights for patterns (see DEFAULT_PATTERN_WEIGHTS)
            max_score: Maximum possible score
            base_score: Base score before patterns are added
        """
        self.pattern_weights = pattern_weights or DEFAULT_PATTERN_WEIGHTS
        self.max_score = max_score
        self.base_score = base_score
        self._scored_entities: dict[str, ScoredEntity] = {}
        self._breakdowns: dict[str, ScoreBreakdown] = {}
    
    def get_pattern_weight(
        self,
        pattern_type: PatternType,
        severity: str
    ) -> float:
        """Get weight for a pattern type and severity.
        
        Args:
            pattern_type: Type of pattern
            severity: Severity level (LOW, MEDIUM, HIGH)
        
        Returns:
            Weight contribution for this pattern
        """
        type_weights = self.pattern_weights.get(pattern_type, {})
        return type_weights.get(severity, 0)
    
    def compute_entity_score(
        self,
        entity_id: str,
        entity_name: str,
        evidence_list: list[PatternEvidence],
    ) -> ScoredEntity:
        """Compute risk score for an entity.
        
        Args:
            entity_id: Unique entity identifier
            entity_name: Display name of entity
            evidence_list: List of pattern evidence involving this entity
        
        Returns:
            ScoredEntity with computed score and explanation
        """
        score = self.base_score
        triggered_signals = []
        datasets = set()
        contributions = []
        
        for evidence in evidence_list:
            # Check if this entity is involved
            if entity_id not in evidence.entities_involved:
                continue
            
            # Get weight for this pattern
            weight = self.get_pattern_weight(
                evidence.pattern_type,
                evidence.severity
            )
            
            score += weight
            
            # Record signal
            signal = {
                "pattern_type": evidence.pattern_type.value,
                "severity": evidence.severity,
                "weight": weight,
                "description": evidence.description,
                "metrics": evidence.metrics,
            }
            triggered_signals.append(signal)
            
            contributions.append({
                "pattern_type": evidence.pattern_type.value,
                "severity": evidence.severity,
                "weight": weight,
            })
            
            # Track datasets
            for row in evidence.source_rows:
                if "dataset" in row:
                    datasets.add(row["dataset"])
        
        # Cap at max score
        score = min(score, self.max_score)
        
        # Generate explanation
        explanation = self._generate_explanation(
            entity_name, score, triggered_signals
        )
        
        # Create scored entity
        scored = ScoredEntity(
            entity_id=entity_id,
            entity_name=entity_name,
            score=score,
            triggered_signals=triggered_signals,
            datasets_involved=list(datasets),
            explanation=explanation,
        )
        
        # Store breakdown
        breakdown = ScoreBreakdown(
            entity_id=entity_id,
            base_score=self.base_score,
            pattern_contributions=contributions,
            final_score=score,
        )
        
        self._scored_entities[entity_id] = scored
        self._breakdowns[entity_id] = breakdown
        
        return scored
    
    def _generate_explanation(
        self,
        entity_name: str,
        score: float,
        triggered_signals: list[dict]
    ) -> str:
        """Generate human-readable explanation for a score."""
        if not triggered_signals:
            return f"{entity_name}: No patterns detected. Score: {score:.0f}"
        
        parts = [f"{entity_name} (Score: {score:.0f}/100)"]
        parts.append("")
        parts.append("Detected patterns:")
        
        for signal in triggered_signals:
            pattern = signal["pattern_type"]
            severity = signal["severity"]
            weight = signal["weight"]
            desc = signal["description"]
            parts.append(f"  - [{severity}] {pattern}: +{weight} points")
            parts.append(f"    {desc}")
        
        parts.append("")
        parts.append(self.DISCLAIMER)
        
        return "\n".join(parts)
    
    def score_all_entities(
        self,
        graph_builder,
        evidence_list: list[PatternEvidence],
    ) -> list[ScoredEntity]:
        """Score all entities in the graph.
        
        Args:
            graph_builder: GraphBuilder with entity nodes
            evidence_list: All detected pattern evidence
        
        Returns:
            List of ScoredEntity objects sorted by score (descending)
        """
        scored = []
        
        for node_id, node in graph_builder._nodes.items():
            if node.node_type.value != "Entity":
                continue
            
            entity_scored = self.compute_entity_score(
                entity_id=node_id,
                entity_name=node.label,
                evidence_list=evidence_list,
            )
            scored.append(entity_scored)
        
        # Sort by score descending
        scored.sort(key=lambda x: x.score, reverse=True)
        
        return scored
    
    def get_scored_entity(self, entity_id: str) -> Optional[ScoredEntity]:
        """Get a scored entity by ID."""
        return self._scored_entities.get(entity_id)
    
    def get_breakdown(self, entity_id: str) -> Optional[ScoreBreakdown]:
        """Get score breakdown for an entity."""
        return self._breakdowns.get(entity_id)
    
    def get_weights_documentation(self) -> str:
        """Get documentation of all weights used.
        
        This enables manual verification of any score.
        """
        lines = [
            "RISK SCORING WEIGHTS DOCUMENTATION",
            "=" * 40,
            "",
            "These weights determine how pattern detections contribute",
            "to the final risk score. All weights are additive.",
            "",
            "Pattern Type -> Severity -> Weight Contribution",
            "-" * 40,
        ]
        
        for pattern_type, severities in self.pattern_weights.items():
            lines.append(f"\n{pattern_type.value}:")
            for severity, weight in severities.items():
                lines.append(f"  {severity}: +{weight} points")
        
        lines.extend([
            "",
            f"Base score: {self.base_score}",
            f"Maximum score: {self.max_score}",
            "",
            self.DISCLAIMER,
        ])
        
        return "\n".join(lines)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Export all scores as a DataFrame."""
        if not self._scored_entities:
            return pd.DataFrame()
        
        records = [e.to_dict() for e in self._scored_entities.values()]
        return pd.DataFrame(records)
    
    def get_high_risk_entities(self, threshold: float = 50.0) -> list[ScoredEntity]:
        """Get entities above a score threshold.
        
        Args:
            threshold: Minimum score to include
        
        Returns:
            List of ScoredEntity with score >= threshold
        """
        return [
            e for e in self._scored_entities.values()
            if e.score >= threshold
        ]


if __name__ == "__main__":
    print("Risk Scoring Engine")
    print("=" * 40)
    
    engine = RiskScoringEngine()
    
    print("\nWeight documentation:")
    print(engine.get_weights_documentation())
