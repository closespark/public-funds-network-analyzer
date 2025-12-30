"""Pattern detection modules.

Detects structural signals (not accusations) including:
- Address density
- Name similarity clusters
- Vendor concentration
- Prime → subaward fan-out
- Transaction timing spikes
- Delinquent property overlap
- Cross-jurisdiction presence

Each module defines trigger rules, emits structured evidence,
and references exact rows/columns.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

import pandas as pd


class PatternType(Enum):
    """Types of patterns that can be detected."""
    ADDRESS_DENSITY = "address_density"
    NAME_SIMILARITY_CLUSTER = "name_similarity_cluster"
    VENDOR_CONCENTRATION = "vendor_concentration"
    PRIME_SUBAWARD_FANOUT = "prime_subaward_fanout"
    TRANSACTION_TIMING_SPIKE = "transaction_timing_spike"
    DELINQUENT_PROPERTY_OVERLAP = "delinquent_property_overlap"
    CROSS_JURISDICTION = "cross_jurisdiction"


@dataclass
class PatternEvidence:
    """Evidence for a detected pattern."""
    
    pattern_type: PatternType
    description: str
    severity: str  # LOW, MEDIUM, HIGH
    entities_involved: list[str]
    source_rows: list[dict[str, Any]]  # [{dataset, row_index, column}]
    metrics: dict[str, Any] = field(default_factory=dict)
    detected_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "pattern_type": self.pattern_type.value,
            "description": self.description,
            "severity": self.severity,
            "entities_involved": self.entities_involved,
            "source_rows": self.source_rows,
            "metrics": self.metrics,
            "detected_at": self.detected_at.isoformat(),
        }


class BasePatternDetector:
    """Base class for pattern detectors."""
    
    pattern_type: PatternType = None
    
    def __init__(self, **config):
        self.config = config
        self._evidence: list[PatternEvidence] = []
    
    def detect(self, *args, **kwargs) -> list[PatternEvidence]:
        """Detect patterns. Override in subclass."""
        raise NotImplementedError
    
    def get_evidence(self) -> list[PatternEvidence]:
        """Get all detected evidence."""
        return self._evidence
    
    def clear_evidence(self) -> None:
        """Clear detected evidence."""
        self._evidence = []


class AddressDensityDetector(BasePatternDetector):
    """Detect multiple entities at the same address.
    
    Trigger: More than N entities registered at the same address.
    """
    
    pattern_type = PatternType.ADDRESS_DENSITY
    
    def __init__(self, min_entities: int = 3, **config):
        super().__init__(**config)
        self.min_entities = min_entities
    
    def detect(
        self,
        graph_builder,  # GraphBuilder from graph.builder
    ) -> list[PatternEvidence]:
        """Detect high-density addresses."""
        from collections import defaultdict
        
        # Group entities by address
        address_to_entities = defaultdict(list)
        
        for edge in graph_builder._edges:
            if edge.edge_type.value == "LOCATED_AT":
                entity_id = edge.source_id
                address_id = edge.target_id
                address_to_entities[address_id].append({
                    "entity_id": entity_id,
                    "source_dataset": edge.source_dataset,
                })
        
        # Find dense addresses
        for address_id, entities in address_to_entities.items():
            if len(entities) >= self.min_entities:
                # Get address label
                address_node = graph_builder.get_node(address_id)
                address_label = address_node.label if address_node else address_id
                
                evidence = PatternEvidence(
                    pattern_type=self.pattern_type,
                    description=f"Multiple entities ({len(entities)}) at address: {address_label}",
                    severity="HIGH" if len(entities) >= 10 else "MEDIUM" if len(entities) >= 5 else "LOW",
                    entities_involved=[e["entity_id"] for e in entities],
                    source_rows=[
                        {
                            "dataset": e["source_dataset"],
                            "address_id": address_id,
                        }
                        for e in entities
                    ],
                    metrics={
                        "entity_count": len(entities),
                        "address": address_label,
                    },
                )
                self._evidence.append(evidence)
        
        return self._evidence


class NameSimilarityClusterDetector(BasePatternDetector):
    """Detect clusters of similar entity names.
    
    Trigger: Groups of entities with highly similar names.
    """
    
    pattern_type = PatternType.NAME_SIMILARITY_CLUSTER
    _fuzz = None  # Cache rapidfuzz import
    
    def __init__(self, similarity_threshold: float = 0.85, min_cluster_size: int = 2, **config):
        super().__init__(**config)
        self.similarity_threshold = similarity_threshold
        self.min_cluster_size = min_cluster_size
        # Try to import rapidfuzz once during initialization
        if NameSimilarityClusterDetector._fuzz is None:
            try:
                from rapidfuzz import fuzz
                NameSimilarityClusterDetector._fuzz = fuzz
            except ImportError:
                NameSimilarityClusterDetector._fuzz = False  # Mark as unavailable
    
    def detect(
        self,
        graph_builder,
    ) -> list[PatternEvidence]:
        """Detect name similarity clusters."""
        if self._fuzz is False:
            # rapidfuzz not available
            return self._evidence
        
        fuzz = self._fuzz
        
        # Get all entity nodes
        entities = []
        for node_id, node in graph_builder._nodes.items():
            if node.node_type.value == "Entity":
                entities.append({
                    "node_id": node_id,
                    "label": node.label,
                    "source_dataset": node.source_dataset,
                })
        
        # Find similar name pairs
        clusters = []
        used = set()
        
        for i, e1 in enumerate(entities):
            if e1["node_id"] in used:
                continue
            
            cluster = [e1]
            
            for e2 in entities[i+1:]:
                if e2["node_id"] in used:
                    continue
                
                score = fuzz.ratio(e1["label"], e2["label"]) / 100.0
                if score >= self.similarity_threshold:
                    cluster.append(e2)
                    used.add(e2["node_id"])
            
            if len(cluster) >= self.min_cluster_size:
                used.add(e1["node_id"])
                clusters.append(cluster)
        
        # Create evidence for clusters
        for cluster in clusters:
            evidence = PatternEvidence(
                pattern_type=self.pattern_type,
                description=f"Similar entity names found: {', '.join(e['label'][:30] for e in cluster[:5])}{'...' if len(cluster) > 5 else ''}",
                severity="MEDIUM" if len(cluster) >= 3 else "LOW",
                entities_involved=[e["node_id"] for e in cluster],
                source_rows=[
                    {
                        "dataset": e["source_dataset"],
                        "entity_id": e["node_id"],
                    }
                    for e in cluster
                ],
                metrics={
                    "cluster_size": len(cluster),
                    "names": [e["label"] for e in cluster],
                },
            )
            self._evidence.append(evidence)
        
        return self._evidence


class VendorConcentrationDetector(BasePatternDetector):
    """Detect concentration of contracts to few vendors.
    
    Trigger: Agency awards large portion of contracts to limited vendors.
    """
    
    pattern_type = PatternType.VENDOR_CONCENTRATION
    
    def __init__(self, concentration_threshold: float = 0.5, min_contracts: int = 5, **config):
        super().__init__(**config)
        self.concentration_threshold = concentration_threshold
        self.min_contracts = min_contracts
    
    def detect(
        self,
        city_contracts_df: pd.DataFrame,
    ) -> list[PatternEvidence]:
        """Detect vendor concentration by agency."""
        if city_contracts_df is None or len(city_contracts_df) == 0:
            return self._evidence
        
        agency_col = "Agency/Department"
        supplier_col = "Supplier"
        value_col = "Contract Value"
        
        if agency_col not in city_contracts_df.columns:
            return self._evidence
        
        # Group by agency
        for agency, group in city_contracts_df.groupby(agency_col):
            if len(group) < self.min_contracts:
                continue
            
            # Count contracts per supplier
            supplier_counts = group[supplier_col].value_counts()
            total_contracts = len(group)
            
            # Find concentrated suppliers
            for supplier, count in supplier_counts.items():
                concentration = count / total_contracts
                if concentration >= self.concentration_threshold:
                    concentration_pct = f"{concentration:.0%}"
                    evidence = PatternEvidence(
                        pattern_type=self.pattern_type,
                        description=f"Vendor '{supplier}' received {concentration_pct} of {agency} contracts ({count}/{total_contracts})",
                        severity="HIGH" if concentration >= 0.7 else "MEDIUM",
                        entities_involved=[str(supplier)],
                        source_rows=[
                            {
                                "dataset": "city_contracts",
                                "row_index": int(idx),
                                "column": supplier_col,
                            }
                            for idx in group[group[supplier_col] == supplier].index
                        ],
                        metrics={
                            "agency": str(agency),
                            "concentration": concentration,
                            "contract_count": count,
                            "total_contracts": total_contracts,
                        },
                    )
                    self._evidence.append(evidence)
        
        return self._evidence


class PrimeSubawardFanoutDetector(BasePatternDetector):
    """Detect prime contractors with many subawards.
    
    Trigger: Prime contractor subawards to unusually many subcontractors.
    """
    
    pattern_type = PatternType.PRIME_SUBAWARD_FANOUT
    
    def __init__(self, min_subawards: int = 10, **config):
        super().__init__(**config)
        self.min_subawards = min_subawards
    
    def detect(
        self,
        subawards_df: pd.DataFrame,
    ) -> list[PatternEvidence]:
        """Detect prime→subaward fan-out patterns."""
        if subawards_df is None or len(subawards_df) == 0:
            return self._evidence
        
        prime_col = "prime_award_unique_key"
        subawardee_col = "subawardee_name"
        prime_name_col = "prime_awardee_name"
        
        if prime_col not in subawards_df.columns:
            return self._evidence
        
        # Group by prime award
        for prime_award, group in subawards_df.groupby(prime_col):
            unique_subawardees = group[subawardee_col].nunique()
            
            if unique_subawardees >= self.min_subawards:
                prime_name = group[prime_name_col].iloc[0] if prime_name_col in group.columns else prime_award
                
                evidence = PatternEvidence(
                    pattern_type=self.pattern_type,
                    description=f"Prime '{prime_name}' has {unique_subawardees} unique subawardees",
                    severity="HIGH" if unique_subawardees >= 25 else "MEDIUM",
                    entities_involved=[str(prime_name)] + list(group[subawardee_col].unique()[:10]),
                    source_rows=[
                        {
                            "dataset": "federal_subawards",
                            "row_index": int(idx),
                            "prime_award": str(prime_award),
                        }
                        for idx in group.index[:20]  # Limit to first 20 rows
                    ],
                    metrics={
                        "prime_award": str(prime_award),
                        "prime_name": str(prime_name),
                        "unique_subawardees": unique_subawardees,
                        "total_subawards": len(group),
                    },
                )
                self._evidence.append(evidence)
        
        return self._evidence


class TransactionTimingDetector(BasePatternDetector):
    """Detect transaction timing spikes.
    
    Trigger: Unusual concentration of transactions near deadline periods.
    """
    
    pattern_type = PatternType.TRANSACTION_TIMING_SPIKE
    
    def __init__(self, spike_threshold: float = 2.0, **config):
        super().__init__(**config)
        self.spike_threshold = spike_threshold  # Multiplier of average
    
    def detect(
        self,
        transactions_df: pd.DataFrame,
        date_column: str = "action_date",
    ) -> list[PatternEvidence]:
        """Detect transaction timing spikes."""
        if transactions_df is None or len(transactions_df) == 0:
            return self._evidence
        
        if date_column not in transactions_df.columns:
            return self._evidence
        
        # Parse dates
        df = transactions_df.copy()
        df["_date"] = pd.to_datetime(df[date_column], errors="coerce")
        df = df.dropna(subset=["_date"])
        
        if len(df) == 0:
            return self._evidence
        
        # Group by month
        df["_month"] = df["_date"].dt.to_period("M")
        monthly_counts = df.groupby("_month").size()
        
        if len(monthly_counts) < 3:
            return self._evidence
        
        avg_count = monthly_counts.mean()
        
        # Find spikes
        for month, count in monthly_counts.items():
            if count >= avg_count * self.spike_threshold:
                month_str = str(month)
                month_rows = df[df["_month"] == month]
                
                evidence = PatternEvidence(
                    pattern_type=self.pattern_type,
                    description=f"Transaction spike in {month_str}: {count} transactions ({count/avg_count:.1f}x average)",
                    severity="HIGH" if count >= avg_count * 3 else "MEDIUM",
                    entities_involved=[],  # No specific entities
                    source_rows=[
                        {
                            "dataset": "transactions",
                            "row_index": int(idx),
                            "column": date_column,
                        }
                        for idx in month_rows.index[:20]
                    ],
                    metrics={
                        "month": month_str,
                        "transaction_count": int(count),
                        "average_count": float(avg_count),
                        "spike_ratio": float(count / avg_count),
                    },
                )
                self._evidence.append(evidence)
        
        return self._evidence


class DelinquentPropertyOverlapDetector(BasePatternDetector):
    """Detect entities with delinquent property who also receive contracts.
    
    Trigger: Contract recipient owns tax-delinquent property.
    """
    
    pattern_type = PatternType.DELINQUENT_PROPERTY_OVERLAP
    
    def __init__(self, **config):
        super().__init__(**config)
    
    def detect(
        self,
        graph_builder,
    ) -> list[PatternEvidence]:
        """Detect delinquent property overlap."""
        # Find entities with OWNS_PROPERTY edges
        entities_with_property = set()
        property_owners = {}
        
        for edge in graph_builder._edges:
            if edge.edge_type.value == "OWNS_PROPERTY":
                entity_id = edge.source_id
                entities_with_property.add(entity_id)
                if entity_id not in property_owners:
                    property_owners[entity_id] = []
                property_owners[entity_id].append(edge.target_id)
        
        # Find entities with contract edges
        entities_with_contracts = set()
        contract_recipients = {}
        
        for edge in graph_builder._edges:
            if edge.edge_type.value in ("CITY_CONTRACTED_WITH", "FEDERAL_PRIME_AWARDED", "FEDERAL_SUBAWARDED"):
                entity_id = edge.source_id
                entities_with_contracts.add(entity_id)
                if entity_id not in contract_recipients:
                    contract_recipients[entity_id] = []
                contract_recipients[entity_id].append(edge.target_id)
        
        # Find overlap
        overlap = entities_with_property & entities_with_contracts
        
        for entity_id in overlap:
            entity_node = graph_builder.get_node(entity_id)
            entity_name = entity_node.label if entity_node else entity_id
            
            properties = property_owners.get(entity_id, [])
            contracts = contract_recipients.get(entity_id, [])
            
            evidence = PatternEvidence(
                pattern_type=self.pattern_type,
                description=f"Entity '{entity_name}' owns delinquent property and has received {len(contracts)} contract(s)",
                severity="HIGH",
                entities_involved=[entity_id],
                source_rows=[
                    {"dataset": "delinquent_properties", "entity_id": entity_id, "property": prop}
                    for prop in properties
                ] + [
                    {"dataset": "contracts", "entity_id": entity_id, "award": award}
                    for award in contracts
                ],
                metrics={
                    "entity_name": entity_name,
                    "property_count": len(properties),
                    "contract_count": len(contracts),
                },
            )
            self._evidence.append(evidence)
        
        return self._evidence


class CrossJurisdictionDetector(BasePatternDetector):
    """Detect entities operating across multiple jurisdictions.
    
    Trigger: Entity has awards in multiple government levels/agencies.
    """
    
    pattern_type = PatternType.CROSS_JURISDICTION
    
    def __init__(self, min_jurisdictions: int = 2, **config):
        super().__init__(**config)
        self.min_jurisdictions = min_jurisdictions
    
    def detect(
        self,
        graph_builder,
    ) -> list[PatternEvidence]:
        """Detect cross-jurisdiction presence."""
        from collections import defaultdict
        
        # Map entities to their award sources (jurisdictions)
        entity_jurisdictions = defaultdict(set)
        entity_awards = defaultdict(list)
        
        for edge in graph_builder._edges:
            if edge.edge_type.value in ("CITY_CONTRACTED_WITH", "FEDERAL_PRIME_AWARDED", "FEDERAL_SUBAWARDED"):
                entity_id = edge.source_id
                
                # Determine jurisdiction from edge type
                if edge.edge_type.value == "CITY_CONTRACTED_WITH":
                    jurisdiction = "city"
                elif edge.edge_type.value == "FEDERAL_PRIME_AWARDED":
                    jurisdiction = "federal_prime"
                else:
                    jurisdiction = "federal_subaward"
                
                entity_jurisdictions[entity_id].add(jurisdiction)
                entity_awards[entity_id].append({
                    "award_id": edge.target_id,
                    "jurisdiction": jurisdiction,
                })
        
        # Find multi-jurisdiction entities
        for entity_id, jurisdictions in entity_jurisdictions.items():
            if len(jurisdictions) >= self.min_jurisdictions:
                entity_node = graph_builder.get_node(entity_id)
                entity_name = entity_node.label if entity_node else entity_id
                
                awards = entity_awards.get(entity_id, [])
                
                evidence = PatternEvidence(
                    pattern_type=self.pattern_type,
                    description=f"Entity '{entity_name}' has awards in {len(jurisdictions)} jurisdiction(s): {', '.join(sorted(jurisdictions))}",
                    severity="MEDIUM" if len(jurisdictions) >= 2 else "LOW",
                    entities_involved=[entity_id],
                    source_rows=[
                        {
                            "dataset": "awards",
                            "entity_id": entity_id,
                            "award_id": a["award_id"],
                            "jurisdiction": a["jurisdiction"],
                        }
                        for a in awards[:10]
                    ],
                    metrics={
                        "entity_name": entity_name,
                        "jurisdictions": list(jurisdictions),
                        "jurisdiction_count": len(jurisdictions),
                        "award_count": len(awards),
                    },
                )
                self._evidence.append(evidence)
        
        return self._evidence


class PatternDetectionEngine:
    """Engine that runs all pattern detectors."""
    
    def __init__(self, **config):
        self.config = config
        self._detectors = [
            AddressDensityDetector(**config),
            NameSimilarityClusterDetector(**config),
            VendorConcentrationDetector(**config),
            PrimeSubawardFanoutDetector(**config),
            TransactionTimingDetector(**config),
            DelinquentPropertyOverlapDetector(**config),
            CrossJurisdictionDetector(**config),
        ]
        self._all_evidence: list[PatternEvidence] = []
    
    def detect_all(
        self,
        graph_builder=None,
        city_contracts_df: pd.DataFrame = None,
        subawards_df: pd.DataFrame = None,
        transactions_df: pd.DataFrame = None,
    ) -> list[PatternEvidence]:
        """Run all detectors and collect evidence."""
        self._all_evidence = []
        
        for detector in self._detectors:
            if detector.pattern_type == PatternType.ADDRESS_DENSITY and graph_builder:
                detector.detect(graph_builder)
            elif detector.pattern_type == PatternType.NAME_SIMILARITY_CLUSTER and graph_builder:
                detector.detect(graph_builder)
            elif detector.pattern_type == PatternType.VENDOR_CONCENTRATION and city_contracts_df is not None:
                detector.detect(city_contracts_df)
            elif detector.pattern_type == PatternType.PRIME_SUBAWARD_FANOUT and subawards_df is not None:
                detector.detect(subawards_df)
            elif detector.pattern_type == PatternType.TRANSACTION_TIMING_SPIKE and transactions_df is not None:
                detector.detect(transactions_df)
            elif detector.pattern_type == PatternType.DELINQUENT_PROPERTY_OVERLAP and graph_builder:
                detector.detect(graph_builder)
            elif detector.pattern_type == PatternType.CROSS_JURISDICTION and graph_builder:
                detector.detect(graph_builder)
            
            self._all_evidence.extend(detector.get_evidence())
        
        return self._all_evidence
    
    def get_all_evidence(self) -> list[PatternEvidence]:
        """Get all collected evidence."""
        return self._all_evidence
    
    def get_evidence_by_type(self, pattern_type: PatternType) -> list[PatternEvidence]:
        """Get evidence filtered by pattern type."""
        return [e for e in self._all_evidence if e.pattern_type == pattern_type]
    
    def get_evidence_by_severity(self, severity: str) -> list[PatternEvidence]:
        """Get evidence filtered by severity."""
        return [e for e in self._all_evidence if e.severity == severity]
    
    def to_dataframe(self) -> pd.DataFrame:
        """Export all evidence as a DataFrame."""
        if not self._all_evidence:
            return pd.DataFrame()
        
        records = [e.to_dict() for e in self._all_evidence]
        return pd.DataFrame(records)


if __name__ == "__main__":
    print("Pattern Detection Engine initialized")
    
    engine = PatternDetectionEngine()
    print(f"Registered {len(engine._detectors)} detectors:")
    for d in engine._detectors:
        print(f"  - {d.pattern_type.value}")
