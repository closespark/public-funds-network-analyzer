"""Output and reporting module.

Exports:
- Leads (CSV/JSON)
- Scores (CSV/JSON)
- Graph edges/nodes (CSV)
- Summary statistics
- Run metadata (timestamp, inputs used)

All outputs are:
- Reproducible
- Human-readable
- Machine-readable
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd


@dataclass
class RunMetadata:
    """Metadata for a pipeline run."""
    
    run_id: str
    timestamp: datetime
    input_files: list[str]
    dataset_counts: dict[str, int]
    graph_stats: dict[str, Any]
    pattern_counts: dict[str, int]
    lead_count: int
    scored_entity_count: int
    high_risk_count: int
    duration_seconds: float = 0.0
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp.isoformat(),
            "input_files": self.input_files,
            "dataset_counts": self.dataset_counts,
            "graph_stats": self.graph_stats,
            "pattern_counts": self.pattern_counts,
            "lead_count": self.lead_count,
            "scored_entity_count": self.scored_entity_count,
            "high_risk_count": self.high_risk_count,
            "duration_seconds": self.duration_seconds,
        }


@dataclass
class SummaryStatistics:
    """Summary statistics for a pipeline run."""
    
    total_entities: int
    total_addresses: int
    total_properties: int
    total_awards: int
    total_edges: int
    patterns_detected: int
    high_severity_patterns: int
    medium_severity_patterns: int
    low_severity_patterns: int
    leads_generated: int
    entities_scored: int
    high_risk_entities: int
    average_score: float
    max_score: float
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_entities": self.total_entities,
            "total_addresses": self.total_addresses,
            "total_properties": self.total_properties,
            "total_awards": self.total_awards,
            "total_edges": self.total_edges,
            "patterns_detected": self.patterns_detected,
            "high_severity_patterns": self.high_severity_patterns,
            "medium_severity_patterns": self.medium_severity_patterns,
            "low_severity_patterns": self.low_severity_patterns,
            "leads_generated": self.leads_generated,
            "entities_scored": self.entities_scored,
            "high_risk_entities": self.high_risk_entities,
            "average_score": round(self.average_score, 2),
            "max_score": round(self.max_score, 2),
        }
    
    def to_text(self) -> str:
        """Convert to human-readable text."""
        lines = [
            "=" * 50,
            "PUBLIC FUNDS NETWORK ANALYZER - SUMMARY REPORT",
            "=" * 50,
            "",
            "GRAPH STATISTICS",
            "-" * 30,
            f"  Entities:    {self.total_entities:,}",
            f"  Addresses:   {self.total_addresses:,}",
            f"  Properties:  {self.total_properties:,}",
            f"  Awards:      {self.total_awards:,}",
            f"  Edges:       {self.total_edges:,}",
            "",
            "PATTERN DETECTION",
            "-" * 30,
            f"  Total patterns detected: {self.patterns_detected}",
            f"    HIGH severity:   {self.high_severity_patterns}",
            f"    MEDIUM severity: {self.medium_severity_patterns}",
            f"    LOW severity:    {self.low_severity_patterns}",
            "",
            "LEADS & SCORING",
            "-" * 30,
            f"  Leads generated:    {self.leads_generated}",
            f"  Entities scored:    {self.entities_scored}",
            f"  High-risk entities: {self.high_risk_entities}",
            f"  Average score:      {self.average_score:.1f}",
            f"  Maximum score:      {self.max_score:.1f}",
            "",
            "=" * 50,
        ]
        return "\n".join(lines)


class ReportGenerator:
    """Generator for output reports."""
    
    def __init__(self, output_dir: Path | str):
        """Initialize report generator.
        
        Args:
            output_dir: Directory for output files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._run_timestamp = datetime.utcnow()
        self._run_id = self._run_timestamp.strftime("%Y%m%d_%H%M%S")
    
    def _get_output_path(self, name: str, extension: str) -> Path:
        """Get output file path with timestamp prefix."""
        return self.output_dir / f"{self._run_id}_{name}.{extension}"
    
    def export_leads_csv(
        self,
        leads: list,  # list[InvestigationLead]
    ) -> Path:
        """Export leads to CSV.
        
        Args:
            leads: List of investigation leads
        
        Returns:
            Path to exported file
        """
        if not leads:
            df = pd.DataFrame()
        else:
            records = [l.to_dict() for l in leads]
            df = pd.DataFrame(records)
            
            # Flatten list columns for CSV
            for col in ["entities_involved", "verification_steps", "source_patterns"]:
                if col in df.columns:
                    df[col] = df[col].apply(lambda x: "; ".join(str(i) for i in x) if isinstance(x, list) else x)
            
            if "evidence_references" in df.columns:
                df["evidence_references"] = df["evidence_references"].apply(
                    lambda x: json.dumps(x) if isinstance(x, list) else x
                )
        
        path = self._get_output_path("leads", "csv")
        df.to_csv(path, index=False)
        return path
    
    def export_leads_json(
        self,
        leads: list,
    ) -> Path:
        """Export leads to JSON.
        
        Args:
            leads: List of investigation leads
        
        Returns:
            Path to exported file
        """
        data = [l.to_dict() for l in leads] if leads else []
        
        path = self._get_output_path("leads", "json")
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        return path
    
    def export_scores_csv(
        self,
        scores: list,  # list[ScoredEntity]
    ) -> Path:
        """Export scores to CSV.
        
        Args:
            scores: List of scored entities
        
        Returns:
            Path to exported file
        """
        if not scores:
            df = pd.DataFrame()
        else:
            records = [s.to_dict() for s in scores]
            df = pd.DataFrame(records)
            
            # Flatten list columns
            for col in ["datasets_involved"]:
                if col in df.columns:
                    df[col] = df[col].apply(lambda x: "; ".join(str(i) for i in x) if isinstance(x, list) else x)
            
            if "triggered_signals" in df.columns:
                df["triggered_signals"] = df["triggered_signals"].apply(
                    lambda x: json.dumps(x) if isinstance(x, list) else x
                )
        
        path = self._get_output_path("scores", "csv")
        df.to_csv(path, index=False)
        return path
    
    def export_scores_json(
        self,
        scores: list,
    ) -> Path:
        """Export scores to JSON.
        
        Args:
            scores: List of scored entities
        
        Returns:
            Path to exported file
        """
        data = [s.to_dict() for s in scores] if scores else []
        
        path = self._get_output_path("scores", "json")
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        return path
    
    def export_graph(
        self,
        graph_builder,  # GraphBuilder
    ) -> tuple[Path, Path]:
        """Export graph nodes and edges to CSV.
        
        Args:
            graph_builder: Graph builder with nodes and edges
        
        Returns:
            Tuple of (nodes_path, edges_path)
        """
        nodes_df = graph_builder.to_node_dataframe()
        edges_df = graph_builder.to_edge_dataframe()
        
        nodes_path = self._get_output_path("graph_nodes", "csv")
        edges_path = self._get_output_path("graph_edges", "csv")
        
        nodes_df.to_csv(nodes_path, index=False)
        edges_df.to_csv(edges_path, index=False)
        
        return nodes_path, edges_path
    
    def export_patterns_csv(
        self,
        evidence: list,  # list[PatternEvidence]
    ) -> Path:
        """Export pattern evidence to CSV.
        
        Args:
            evidence: List of pattern evidence
        
        Returns:
            Path to exported file
        """
        if not evidence:
            df = pd.DataFrame()
        else:
            records = [e.to_dict() for e in evidence]
            df = pd.DataFrame(records)
            
            # Flatten list columns
            for col in ["entities_involved"]:
                if col in df.columns:
                    df[col] = df[col].apply(lambda x: "; ".join(str(i) for i in x) if isinstance(x, list) else x)
            
            if "source_rows" in df.columns:
                df["source_rows"] = df["source_rows"].apply(
                    lambda x: json.dumps(x) if isinstance(x, list) else x
                )
            if "metrics" in df.columns:
                df["metrics"] = df["metrics"].apply(
                    lambda x: json.dumps(x) if isinstance(x, dict) else x
                )
        
        path = self._get_output_path("patterns", "csv")
        df.to_csv(path, index=False)
        return path
    
    def compute_summary_statistics(
        self,
        graph_builder=None,
        evidence: list = None,
        leads: list = None,
        scores: list = None,
        high_risk_threshold: float = 50.0,
    ) -> SummaryStatistics:
        """Compute summary statistics.
        
        Args:
            graph_builder: Graph builder with nodes and edges
            evidence: List of pattern evidence
            leads: List of investigation leads
            scores: List of scored entities
            high_risk_threshold: Score threshold for high-risk
        
        Returns:
            Summary statistics
        """
        # Graph stats
        node_counts = graph_builder.get_node_counts_by_type() if graph_builder else {}
        total_edges = graph_builder.get_edge_count() if graph_builder else 0
        
        # Pattern stats
        patterns = evidence or []
        high_severity = sum(1 for e in patterns if e.severity == "HIGH")
        medium_severity = sum(1 for e in patterns if e.severity == "MEDIUM")
        low_severity = sum(1 for e in patterns if e.severity == "LOW")
        
        # Score stats
        scores_list = scores or []
        entities_scored = len(scores_list)
        high_risk = sum(1 for s in scores_list if s.score >= high_risk_threshold)
        avg_score = sum(s.score for s in scores_list) / entities_scored if entities_scored else 0.0
        max_score = max((s.score for s in scores_list), default=0.0)
        
        return SummaryStatistics(
            total_entities=node_counts.get("Entity", 0),
            total_addresses=node_counts.get("Address", 0),
            total_properties=node_counts.get("Property", 0),
            total_awards=node_counts.get("Award", 0),
            total_edges=total_edges,
            patterns_detected=len(patterns),
            high_severity_patterns=high_severity,
            medium_severity_patterns=medium_severity,
            low_severity_patterns=low_severity,
            leads_generated=len(leads or []),
            entities_scored=entities_scored,
            high_risk_entities=high_risk,
            average_score=avg_score,
            max_score=max_score,
        )
    
    def export_summary(
        self,
        stats: SummaryStatistics,
    ) -> tuple[Path, Path]:
        """Export summary statistics.
        
        Args:
            stats: Summary statistics
        
        Returns:
            Tuple of (text_path, json_path)
        """
        # Export text version
        text_path = self._get_output_path("summary", "txt")
        with open(text_path, "w") as f:
            f.write(stats.to_text())
        
        # Export JSON version
        json_path = self._get_output_path("summary", "json")
        with open(json_path, "w") as f:
            json.dump(stats.to_dict(), f, indent=2)
        
        return text_path, json_path
    
    def export_metadata(
        self,
        registry=None,  # DatasetRegistry
        graph_builder=None,
        evidence: list = None,
        leads: list = None,
        scores: list = None,
        duration_seconds: float = 0.0,
    ) -> Path:
        """Export run metadata.
        
        Args:
            registry: Dataset registry with loaded datasets
            graph_builder: Graph builder
            evidence: Pattern evidence
            leads: Investigation leads
            scores: Scored entities
            duration_seconds: Pipeline duration
        
        Returns:
            Path to metadata file
        """
        # Get input files and counts from registry
        input_files = []
        dataset_counts = {}
        if registry:
            for name in registry.list_datasets():
                meta = registry.get_metadata(name)
                if meta:
                    input_files.append(meta.source_file)
                    dataset_counts[name] = meta.row_count
        
        # Get graph stats
        graph_stats = {}
        if graph_builder:
            graph_stats = graph_builder.get_metadata()
        
        # Get pattern counts by type
        pattern_counts = {}
        if evidence:
            for e in evidence:
                ptype = e.pattern_type.value
                pattern_counts[ptype] = pattern_counts.get(ptype, 0) + 1
        
        # High risk count
        high_risk_count = sum(1 for s in (scores or []) if s.score >= 50)
        
        metadata = RunMetadata(
            run_id=self._run_id,
            timestamp=self._run_timestamp,
            input_files=input_files,
            dataset_counts=dataset_counts,
            graph_stats=graph_stats,
            pattern_counts=pattern_counts,
            lead_count=len(leads or []),
            scored_entity_count=len(scores or []),
            high_risk_count=high_risk_count,
            duration_seconds=duration_seconds,
        )
        
        path = self._get_output_path("metadata", "json")
        with open(path, "w") as f:
            json.dump(metadata.to_dict(), f, indent=2, default=str)
        
        return path
    
    def export_all(
        self,
        registry=None,
        graph_builder=None,
        evidence: list = None,
        leads: list = None,
        scores: list = None,
        duration_seconds: float = 0.0,
    ) -> dict[str, Path]:
        """Export all outputs.
        
        Args:
            registry: Dataset registry
            graph_builder: Graph builder
            evidence: Pattern evidence
            leads: Investigation leads
            scores: Scored entities
            duration_seconds: Pipeline duration
        
        Returns:
            Dictionary mapping output type to file path
        """
        outputs = {}
        
        # Export leads
        outputs["leads_csv"] = self.export_leads_csv(leads or [])
        outputs["leads_json"] = self.export_leads_json(leads or [])
        
        # Export scores
        outputs["scores_csv"] = self.export_scores_csv(scores or [])
        outputs["scores_json"] = self.export_scores_json(scores or [])
        
        # Export graph
        if graph_builder:
            nodes_path, edges_path = self.export_graph(graph_builder)
            outputs["graph_nodes"] = nodes_path
            outputs["graph_edges"] = edges_path
        
        # Export patterns
        outputs["patterns"] = self.export_patterns_csv(evidence or [])
        
        # Export summary
        stats = self.compute_summary_statistics(
            graph_builder=graph_builder,
            evidence=evidence,
            leads=leads,
            scores=scores,
        )
        text_path, json_path = self.export_summary(stats)
        outputs["summary_txt"] = text_path
        outputs["summary_json"] = json_path
        
        # Export metadata
        outputs["metadata"] = self.export_metadata(
            registry=registry,
            graph_builder=graph_builder,
            evidence=evidence,
            leads=leads,
            scores=scores,
            duration_seconds=duration_seconds,
        )
        
        return outputs


if __name__ == "__main__":
    from pathlib import Path
    
    print("Report Generator")
    print("=" * 40)
    
    # Example
    reporter = ReportGenerator(Path("outputs"))
    print(f"Output directory: {reporter.output_dir}")
    print(f"Run ID: {reporter._run_id}")
