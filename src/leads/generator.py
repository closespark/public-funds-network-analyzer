"""Lead generation engine.

Produces journalist-ready investigation leads from detected patterns.
Each lead:
- Has a type and description
- References supporting evidence
- Suggests next verification steps
- Uses no accusatory language

Leads are exported to CSV/JSON and are self-contained.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

import pandas as pd

from ..patterns.detector import PatternEvidence, PatternType
from ..scoring.engine import ScoredEntity


class LeadType(Enum):
    """Types of investigation leads."""
    ADDRESS_CLUSTER = "address_cluster"
    SIMILAR_NAMES = "similar_names"
    CONTRACT_CONCENTRATION = "contract_concentration"
    SUBAWARD_NETWORK = "subaward_network"
    TIMING_ANOMALY = "timing_anomaly"
    PROPERTY_FLAG = "property_flag"
    MULTI_JURISDICTION = "multi_jurisdiction"
    HIGH_RISK_ENTITY = "high_risk_entity"


# Map pattern types to lead types
PATTERN_TO_LEAD = {
    PatternType.ADDRESS_DENSITY: LeadType.ADDRESS_CLUSTER,
    PatternType.NAME_SIMILARITY_CLUSTER: LeadType.SIMILAR_NAMES,
    PatternType.VENDOR_CONCENTRATION: LeadType.CONTRACT_CONCENTRATION,
    PatternType.PRIME_SUBAWARD_FANOUT: LeadType.SUBAWARD_NETWORK,
    PatternType.TRANSACTION_TIMING_SPIKE: LeadType.TIMING_ANOMALY,
    PatternType.DELINQUENT_PROPERTY_OVERLAP: LeadType.PROPERTY_FLAG,
    PatternType.CROSS_JURISDICTION: LeadType.MULTI_JURISDICTION,
}


# Suggested verification steps by lead type
VERIFICATION_STEPS = {
    LeadType.ADDRESS_CLUSTER: [
        "Verify the physical address exists and can accommodate multiple businesses",
        "Check if businesses at this address share ownership, officers, or registered agents",
        "Review business registration documents for common signatures or contacts",
        "Determine if address is a registered agent service, virtual office, or residential property",
    ],
    LeadType.SIMILAR_NAMES: [
        "Compare business registration dates and documents",
        "Check for common officers, agents, or contact information",
        "Review award histories for patterns of sequential or complementary bidding",
        "Verify if entities have distinct operations or share resources",
    ],
    LeadType.CONTRACT_CONCENTRATION: [
        "Review procurement records for competitive bidding compliance",
        "Check for contract modifications, change orders, or extensions",
        "Examine whether other qualified vendors were available",
        "Review relationship history between agency personnel and vendor",
    ],
    LeadType.SUBAWARD_NETWORK: [
        "Map the complete network of prime-to-sub relationships",
        "Check if subawardees share addresses, officers, or ownership with prime",
        "Review subaward amounts relative to prime award scope",
        "Verify subawardees have capacity to perform contracted work",
    ],
    LeadType.TIMING_ANOMALY: [
        "Correlate transaction spikes with fiscal year-end or budget cycles",
        "Review individual transactions during spike period",
        "Check for rush procurements or expedited approvals",
        "Compare with prior year patterns for the same period",
    ],
    LeadType.PROPERTY_FLAG: [
        "Verify current tax delinquency status with local assessor",
        "Review entity's financial capacity indicators",
        "Check contract performance history",
        "Determine if delinquency predates or postdates contract awards",
    ],
    LeadType.MULTI_JURISDICTION: [
        "Map complete award history across jurisdictions",
        "Verify entity maintains actual operations in each jurisdiction",
        "Check for performance issues in any jurisdiction",
        "Review whether awards were competitively bid in each location",
    ],
    LeadType.HIGH_RISK_ENTITY: [
        "Review the complete pattern summary that contributed to the score",
        "Prioritize verification of highest-weighted patterns",
        "Gather additional public records (corporate filings, property records)",
        "Consider requesting public records from contracting agencies",
    ],
}


@dataclass
class InvestigationLead:
    """A journalist-ready investigation lead."""
    
    lead_id: str
    lead_type: LeadType
    title: str
    summary: str
    entities_involved: list[str]
    evidence_references: list[dict[str, Any]]
    verification_steps: list[str]
    severity: str
    source_patterns: list[str]
    generated_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for export."""
        return {
            "lead_id": self.lead_id,
            "lead_type": self.lead_type.value,
            "title": self.title,
            "summary": self.summary,
            "entities_involved": self.entities_involved,
            "evidence_references": self.evidence_references,
            "verification_steps": self.verification_steps,
            "severity": self.severity,
            "source_patterns": self.source_patterns,
            "generated_at": self.generated_at.isoformat(),
        }


class LeadGenerator:
    """Generator for investigation leads."""
    
    def __init__(self):
        self._leads: list[InvestigationLead] = []
        self._lead_counter = 0
    
    def _next_lead_id(self) -> str:
        """Generate next lead ID."""
        self._lead_counter += 1
        return f"LEAD-{self._lead_counter:04d}"
    
    def generate_from_evidence(
        self,
        evidence: PatternEvidence,
    ) -> InvestigationLead:
        """Generate a lead from a single pattern evidence.
        
        Args:
            evidence: Detected pattern evidence
        
        Returns:
            Investigation lead
        """
        lead_type = PATTERN_TO_LEAD.get(
            evidence.pattern_type,
            LeadType.HIGH_RISK_ENTITY
        )
        
        # Generate title
        title = self._generate_title(lead_type, evidence)
        
        # Generate summary (non-accusatory)
        summary = self._generate_summary(lead_type, evidence)
        
        # Get verification steps
        steps = VERIFICATION_STEPS.get(lead_type, [])
        
        # Create evidence references
        references = []
        for row in evidence.source_rows:
            ref = {
                "type": "source_row",
                **row,
            }
            references.append(ref)
        
        # Add metrics as reference
        if evidence.metrics:
            references.append({
                "type": "metrics",
                **evidence.metrics,
            })
        
        lead = InvestigationLead(
            lead_id=self._next_lead_id(),
            lead_type=lead_type,
            title=title,
            summary=summary,
            entities_involved=evidence.entities_involved,
            evidence_references=references,
            verification_steps=steps,
            severity=evidence.severity,
            source_patterns=[evidence.pattern_type.value],
        )
        
        self._leads.append(lead)
        return lead
    
    def _generate_title(
        self,
        lead_type: LeadType,
        evidence: PatternEvidence
    ) -> str:
        """Generate a lead title."""
        metrics = evidence.metrics
        
        if lead_type == LeadType.ADDRESS_CLUSTER:
            count = metrics.get("entity_count", "Multiple")
            address = metrics.get("address", "address")[:50]
            return f"Address Cluster: {count} entities at {address}"
        
        elif lead_type == LeadType.SIMILAR_NAMES:
            size = metrics.get("cluster_size", "Multiple")
            return f"Name Similarity Cluster: {size} related entity names"
        
        elif lead_type == LeadType.CONTRACT_CONCENTRATION:
            vendor = str(metrics.get("agency", "Agency"))[:30]
            conc = metrics.get("concentration", 0)
            return f"Contract Concentration: {conc:.0%} to single vendor in {vendor}"
        
        elif lead_type == LeadType.SUBAWARD_NETWORK:
            count = metrics.get("unique_subawardees", "Multiple")
            return f"Subaward Fan-out: Prime with {count} subawardees"
        
        elif lead_type == LeadType.TIMING_ANOMALY:
            month = metrics.get("month", "Period")
            ratio = metrics.get("spike_ratio", 0)
            return f"Transaction Spike: {ratio:.1f}x average in {month}"
        
        elif lead_type == LeadType.PROPERTY_FLAG:
            name = str(metrics.get("entity_name", "Entity"))[:30]
            return f"Property Flag: {name} with delinquent property and contracts"
        
        elif lead_type == LeadType.MULTI_JURISDICTION:
            count = metrics.get("jurisdiction_count", "Multiple")
            return f"Multi-Jurisdiction: Entity with awards in {count} jurisdictions"
        
        else:
            return f"Investigation Lead: {evidence.description[:50]}"
    
    def _generate_summary(
        self,
        lead_type: LeadType,
        evidence: PatternEvidence
    ) -> str:
        """Generate a non-accusatory summary."""
        metrics = evidence.metrics
        
        base = evidence.description
        
        # Add context without accusation
        context = []
        
        if lead_type == LeadType.ADDRESS_CLUSTER:
            context.append(
                "Multiple entities registered at the same address may indicate "
                "shared resources, a registered agent service, or require verification "
                "of distinct business operations."
            )
        
        elif lead_type == LeadType.SIMILAR_NAMES:
            context.append(
                "Entities with similar names may be related through common ownership "
                "or could be coincidentally named. Review is suggested to clarify relationships."
            )
        
        elif lead_type == LeadType.CONTRACT_CONCENTRATION:
            context.append(
                "High concentration of contracts to a single vendor may reflect "
                "specialized capabilities or limited competition. Review procurement "
                "documentation to understand the context."
            )
        
        elif lead_type == LeadType.SUBAWARD_NETWORK:
            context.append(
                "A prime contractor with many subawardees may indicate appropriate "
                "use of specialized subcontractors or warrant review of the prime's "
                "capacity and subaward practices."
            )
        
        elif lead_type == LeadType.TIMING_ANOMALY:
            context.append(
                "Transaction spikes may correlate with legitimate budget cycles, "
                "year-end spending, or project milestones. Context is needed to "
                "understand the timing pattern."
            )
        
        elif lead_type == LeadType.PROPERTY_FLAG:
            context.append(
                "A contractor with tax-delinquent property may face financial "
                "challenges. This pattern warrants review of contractor performance "
                "and financial capacity without presuming impropriety."
            )
        
        elif lead_type == LeadType.MULTI_JURISDICTION:
            context.append(
                "Operating across multiple jurisdictions may reflect legitimate "
                "business expansion. Review can confirm the entity maintains "
                "appropriate presence and capacity in each location."
            )
        
        return base + "\n\n" + " ".join(context)
    
    def generate_from_scored_entity(
        self,
        scored: ScoredEntity,
        min_score: float = 30.0,
    ) -> Optional[InvestigationLead]:
        """Generate a lead from a high-scoring entity.
        
        Args:
            scored: Scored entity
            min_score: Minimum score to generate lead
        
        Returns:
            Investigation lead or None if below threshold
        """
        if scored.score < min_score:
            return None
        
        # Build evidence references from signals
        references = []
        source_patterns = []
        
        for signal in scored.triggered_signals:
            references.append({
                "type": "pattern_signal",
                "pattern": signal["pattern_type"],
                "severity": signal["severity"],
                "weight": signal["weight"],
                "description": signal["description"],
            })
            if signal["pattern_type"] not in source_patterns:
                source_patterns.append(signal["pattern_type"])
        
        # Determine severity from score
        if scored.score >= 70:
            severity = "HIGH"
        elif scored.score >= 40:
            severity = "MEDIUM"
        else:
            severity = "LOW"
        
        # Generate summary
        summary = (
            f"Entity '{scored.entity_name}' has a risk score of {scored.score:.0f}/100 "
            f"based on {len(scored.triggered_signals)} detected pattern(s). "
            f"Datasets involved: {', '.join(scored.datasets_involved)}.\n\n"
            f"This score reflects structural patterns that may warrant investigation. "
            f"A risk score is NOT an accusation of wrongdoing. Review the underlying "
            f"evidence and follow suggested verification steps."
        )
        
        lead = InvestigationLead(
            lead_id=self._next_lead_id(),
            lead_type=LeadType.HIGH_RISK_ENTITY,
            title=f"High Risk Entity: {scored.entity_name[:40]} (Score: {scored.score:.0f})",
            summary=summary,
            entities_involved=[scored.entity_id],
            evidence_references=references,
            verification_steps=VERIFICATION_STEPS[LeadType.HIGH_RISK_ENTITY],
            severity=severity,
            source_patterns=source_patterns,
        )
        
        self._leads.append(lead)
        return lead
    
    def generate_all(
        self,
        evidence_list: list[PatternEvidence],
        scored_entities: list[ScoredEntity] = None,
        min_score: float = 30.0,
    ) -> list[InvestigationLead]:
        """Generate leads from all evidence and scored entities.
        
        Args:
            evidence_list: All pattern evidence
            scored_entities: Optional list of scored entities
            min_score: Minimum score for entity leads
        
        Returns:
            List of all generated leads
        """
        # Generate from patterns
        for evidence in evidence_list:
            if evidence.severity in ("MEDIUM", "HIGH"):
                self.generate_from_evidence(evidence)
        
        # Generate from scored entities
        if scored_entities:
            for scored in scored_entities:
                self.generate_from_scored_entity(scored, min_score)
        
        return self._leads
    
    def get_leads(self) -> list[InvestigationLead]:
        """Get all generated leads."""
        return self._leads
    
    def get_leads_by_type(self, lead_type: LeadType) -> list[InvestigationLead]:
        """Get leads filtered by type."""
        return [l for l in self._leads if l.lead_type == lead_type]
    
    def get_leads_by_severity(self, severity: str) -> list[InvestigationLead]:
        """Get leads filtered by severity."""
        return [l for l in self._leads if l.severity == severity]
    
    def to_dataframe(self) -> pd.DataFrame:
        """Export leads as a DataFrame."""
        if not self._leads:
            return pd.DataFrame()
        
        records = [l.to_dict() for l in self._leads]
        return pd.DataFrame(records)
    
    def to_json(self) -> list[dict]:
        """Export leads as JSON-serializable list."""
        return [l.to_dict() for l in self._leads]
    
    def clear_leads(self) -> None:
        """Clear all generated leads."""
        self._leads = []
        self._lead_counter = 0


if __name__ == "__main__":
    print("Lead Generation Engine")
    print("=" * 40)
    
    generator = LeadGenerator()
    
    print("\nLead types and verification steps:")
    for lead_type, steps in VERIFICATION_STEPS.items():
        print(f"\n{lead_type.value}:")
        for i, step in enumerate(steps, 1):
            print(f"  {i}. {step}")
