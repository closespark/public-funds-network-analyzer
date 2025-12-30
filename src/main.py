"""Main entry point for the Public Funds Network Analyzer.

This module orchestrates the full analysis pipeline:
1. Load data from data/ directory
2. Normalize entities and addresses
3. Build relationship graph
4. Detect patterns
5. Score risks
6. Generate leads
7. Export reports to outputs/

Usage:
    python -m src.main [--data-dir DATA] [--output-dir OUTPUT]
"""

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .ingestion.loader import discover_and_load_all, DatasetRegistry
from .normalization.engine import (
    normalize_entity_column,
    normalize_address_column,
)
from .graph.builder import GraphBuilder, build_graph_from_datasets
from .patterns.detector import PatternDetectionEngine
from .scoring.engine import RiskScoringEngine
from .leads.generator import LeadGenerator
from .output.reporter import ReportGenerator


def normalize_datasets(registry: DatasetRegistry) -> DatasetRegistry:
    """Normalize entity and address columns in all datasets.
    
    Args:
        registry: Dataset registry with loaded datasets
    
    Returns:
        Registry with normalized columns added
    """
    print("\nNormalizing datasets...")
    
    # Normalize business licenses
    df = registry.get_dataset("business_licenses")
    if df is not None:
        df = normalize_entity_column(df, "Business Name")
        if "Business Address" in df.columns:
            df = normalize_address_column(df, "Business Address")
        registry._datasets["business_licenses"] = df
        print(f"  Normalized business_licenses: {len(df)} rows")
    
    # Normalize city contracts
    df = registry.get_dataset("city_contracts")
    if df is not None:
        df = normalize_entity_column(df, "Supplier")
        registry._datasets["city_contracts"] = df
        print(f"  Normalized city_contracts: {len(df)} rows")
    
    # Normalize delinquent properties
    df = registry.get_dataset("delinquent_properties")
    if df is not None:
        df = normalize_entity_column(df, "Current Owner Name 1")
        if "Physical Address" in df.columns:
            df = normalize_address_column(df, "Physical Address")
        registry._datasets["delinquent_properties"] = df
        print(f"  Normalized delinquent_properties: {len(df)} rows")
    
    # Normalize federal contracts prime
    df = registry.get_dataset("federal_contracts_prime")
    if df is not None:
        if "prime_awardee_name" in df.columns:
            df = normalize_entity_column(df, "prime_awardee_name")
        if "prime_awardee_address_line_1" in df.columns:
            df = normalize_address_column(df, "prime_awardee_address_line_1")
        registry._datasets["federal_contracts_prime"] = df
        print(f"  Normalized federal_contracts_prime: {len(df)} rows")
    
    # Normalize federal assistance prime
    df = registry.get_dataset("federal_assistance_prime")
    if df is not None:
        if "recipient_name" in df.columns:
            df = normalize_entity_column(df, "recipient_name")
        if "recipient_address_line_1" in df.columns:
            df = normalize_address_column(df, "recipient_address_line_1")
        registry._datasets["federal_assistance_prime"] = df
        print(f"  Normalized federal_assistance_prime: {len(df)} rows")
    
    # Normalize subawards
    for ds_name in ["federal_contracts_subawards", "federal_assistance_subawards"]:
        df = registry.get_dataset(ds_name)
        if df is not None:
            if "subawardee_name" in df.columns:
                df = normalize_entity_column(df, "subawardee_name")
            if "subawardee_address_line_1" in df.columns:
                df = normalize_address_column(df, "subawardee_address_line_1")
            registry._datasets[ds_name] = df
            print(f"  Normalized {ds_name}: {len(df)} rows")
    
    return registry


def run_pipeline(
    data_dir: Path,
    output_dir: Path,
    min_confidence: str = "LOW",
    fuzzy_threshold: float = 80.0,
    min_risk_score: float = 30.0,
) -> dict:
    """Run the full analysis pipeline.
    
    Args:
        data_dir: Directory containing input data files
        output_dir: Directory for output files
        min_confidence: Minimum confidence level for joins
        fuzzy_threshold: Threshold for fuzzy matching
        min_risk_score: Minimum score for lead generation
    
    Returns:
        Dictionary with pipeline results and output paths
    """
    start_time = time.time()
    
    print("=" * 60)
    print("PUBLIC FUNDS NETWORK ANALYZER")
    print("=" * 60)
    print(f"\nData directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Start time: {datetime.now().isoformat()}")
    
    # Step 1: Load data
    print("\n" + "-" * 40)
    print("STEP 1: Loading datasets")
    print("-" * 40)
    
    registry = discover_and_load_all(data_dir)
    
    print(f"\nLoaded {len(registry.list_datasets())} datasets:")
    for name in registry.list_datasets():
        meta = registry.get_metadata(name)
        print(f"  {name}: {meta.row_count} rows, {meta.column_count} columns")
    
    # Step 2: Normalize
    print("\n" + "-" * 40)
    print("STEP 2: Normalizing entities and addresses")
    print("-" * 40)
    
    registry = normalize_datasets(registry)
    
    # Step 3: Build graph
    print("\n" + "-" * 40)
    print("STEP 3: Building relationship graph")
    print("-" * 40)
    
    graph_builder = build_graph_from_datasets(registry)
    
    print(f"\nGraph statistics:")
    print(f"  Total nodes: {graph_builder.get_node_count()}")
    print(f"  Total edges: {graph_builder.get_edge_count()}")
    print(f"\n  Nodes by type:")
    for node_type, count in graph_builder.get_node_counts_by_type().items():
        print(f"    {node_type}: {count}")
    print(f"\n  Edges by type:")
    for edge_type, count in graph_builder.get_edge_counts_by_type().items():
        print(f"    {edge_type}: {count}")
    
    # Step 4: Detect patterns
    print("\n" + "-" * 40)
    print("STEP 4: Detecting patterns")
    print("-" * 40)
    
    pattern_engine = PatternDetectionEngine()
    
    # Get datasets for pattern detection
    city_contracts = registry.get_dataset("city_contracts")
    
    # Combine subawards for analysis
    subawards_dfs = []
    for ds_name in ["federal_contracts_subawards", "federal_assistance_subawards"]:
        df = registry.get_dataset(ds_name)
        if df is not None:
            subawards_dfs.append(df)
    subawards = pd.concat(subawards_dfs, ignore_index=True) if subawards_dfs else None
    
    # Combine transactions for timing analysis
    transactions_dfs = []
    for ds_name in ["federal_contracts_transactions", "federal_assistance_transactions"]:
        df = registry.get_dataset(ds_name)
        if df is not None:
            transactions_dfs.append(df)
    transactions = pd.concat(transactions_dfs, ignore_index=True) if transactions_dfs else None
    
    evidence_list = pattern_engine.detect_all(
        graph_builder=graph_builder,
        city_contracts_df=city_contracts,
        subawards_df=subawards,
        transactions_df=transactions,
    )
    
    print(f"\nPatterns detected: {len(evidence_list)}")
    severity_counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
    for e in evidence_list:
        severity_counts[e.severity] = severity_counts.get(e.severity, 0) + 1
    for severity, count in severity_counts.items():
        print(f"  {severity}: {count}")
    
    # Step 5: Score risks
    print("\n" + "-" * 40)
    print("STEP 5: Scoring risks")
    print("-" * 40)
    
    scoring_engine = RiskScoringEngine()
    scored_entities = scoring_engine.score_all_entities(
        graph_builder=graph_builder,
        evidence_list=evidence_list,
    )
    
    high_risk = [s for s in scored_entities if s.score >= 50]
    
    print(f"\nEntities scored: {len(scored_entities)}")
    print(f"High-risk entities (score >= 50): {len(high_risk)}")
    
    if scored_entities:
        top_5 = scored_entities[:5]
        print("\nTop 5 entities by score:")
        for i, s in enumerate(top_5, 1):
            print(f"  {i}. {s.entity_name[:40]}: {s.score:.0f}")
    
    # Step 6: Generate leads
    print("\n" + "-" * 40)
    print("STEP 6: Generating investigation leads")
    print("-" * 40)
    
    lead_generator = LeadGenerator()
    leads = lead_generator.generate_all(
        evidence_list=evidence_list,
        scored_entities=scored_entities,
        min_score=min_risk_score,
    )
    
    print(f"\nLeads generated: {len(leads)}")
    severity_counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
    for lead in leads:
        severity_counts[lead.severity] = severity_counts.get(lead.severity, 0) + 1
    for severity, count in severity_counts.items():
        print(f"  {severity}: {count}")
    
    # Step 7: Export reports
    print("\n" + "-" * 40)
    print("STEP 7: Exporting reports")
    print("-" * 40)
    
    duration = time.time() - start_time
    
    reporter = ReportGenerator(output_dir)
    outputs = reporter.export_all(
        registry=registry,
        graph_builder=graph_builder,
        evidence=evidence_list,
        leads=leads,
        scores=scored_entities,
        duration_seconds=duration,
    )
    
    print("\nExported files:")
    for output_type, path in outputs.items():
        print(f"  {output_type}: {path}")
    
    # Print summary
    stats = reporter.compute_summary_statistics(
        graph_builder=graph_builder,
        evidence=evidence_list,
        leads=leads,
        scores=scored_entities,
    )
    
    print("\n" + stats.to_text())
    
    print(f"\nPipeline completed in {duration:.2f} seconds")
    
    return {
        "registry": registry,
        "graph_builder": graph_builder,
        "evidence": evidence_list,
        "scored_entities": scored_entities,
        "leads": leads,
        "outputs": outputs,
        "duration": duration,
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Public Funds Network Analyzer"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing input data files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory for output files",
    )
    parser.add_argument(
        "--min-confidence",
        choices=["HIGH", "MEDIUM", "LOW"],
        default="LOW",
        help="Minimum confidence level for joins",
    )
    parser.add_argument(
        "--fuzzy-threshold",
        type=float,
        default=80.0,
        help="Threshold for fuzzy name matching (0-100)",
    )
    parser.add_argument(
        "--min-risk-score",
        type=float,
        default=30.0,
        help="Minimum risk score for lead generation",
    )
    
    args = parser.parse_args()
    
    run_pipeline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        min_confidence=args.min_confidence,
        fuzzy_threshold=args.fuzzy_threshold,
        min_risk_score=args.min_risk_score,
    )


if __name__ == "__main__":
    main()
