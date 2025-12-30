"""Join engine with confidence scoring.

Performs explainable joins across datasets with confidence levels:
- HIGH: UEI/DUNS exact match
- MEDIUM: Exact normalized match
- LOW: Fuzzy match

Every join records: join method, confidence, fields used.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Callable

import pandas as pd


class ConfidenceLevel(Enum):
    """Confidence levels for joins."""
    HIGH = "HIGH"       # UEI/DUNS exact match
    MEDIUM = "MEDIUM"   # Exact normalized match
    LOW = "LOW"         # Fuzzy match


@dataclass
class JoinResult:
    """Result of a join operation."""
    
    left_index: Any
    right_index: Any
    confidence: ConfidenceLevel
    join_method: str
    fields_used: list[str]
    match_score: float = 1.0  # 0-1, for fuzzy matches
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "left_index": self.left_index,
            "right_index": self.right_index,
            "confidence": self.confidence.value,
            "join_method": self.join_method,
            "fields_used": self.fields_used,
            "match_score": self.match_score,
        }


@dataclass
class JoinMetadata:
    """Metadata for a complete join operation."""
    
    join_type: str
    left_dataset: str
    right_dataset: str
    total_matches: int
    high_confidence: int
    medium_confidence: int
    low_confidence: int
    fields_used: list[str]
    results: list[JoinResult] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "join_type": self.join_type,
            "left_dataset": self.left_dataset,
            "right_dataset": self.right_dataset,
            "total_matches": self.total_matches,
            "high_confidence": self.high_confidence,
            "medium_confidence": self.medium_confidence,
            "low_confidence": self.low_confidence,
            "fields_used": self.fields_used,
        }


def _try_import_rapidfuzz():
    """Try to import rapidfuzz for fuzzy matching."""
    try:
        from rapidfuzz import fuzz
        return fuzz
    except ImportError:
        return None


def join_by_uei(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    left_uei_col: str,
    right_uei_col: str,
) -> list[JoinResult]:
    """Join datasets by UEI (Unique Entity Identifier) - HIGH confidence.
    
    Args:
        left_df: Left DataFrame
        right_df: Right DataFrame
        left_uei_col: UEI column in left DataFrame
        right_uei_col: UEI column in right DataFrame
    
    Returns:
        List of JoinResults with HIGH confidence
    """
    results = []
    
    # Create lookup from right DataFrame
    right_lookup = {}
    for idx, row in right_df.iterrows():
        uei = row.get(right_uei_col)
        if uei and pd.notna(uei) and str(uei).strip():
            uei_clean = str(uei).strip().upper()
            if uei_clean not in right_lookup:
                right_lookup[uei_clean] = []
            right_lookup[uei_clean].append(idx)
    
    # Find matches from left DataFrame
    for left_idx, row in left_df.iterrows():
        uei = row.get(left_uei_col)
        if uei and pd.notna(uei) and str(uei).strip():
            uei_clean = str(uei).strip().upper()
            if uei_clean in right_lookup:
                for right_idx in right_lookup[uei_clean]:
                    results.append(JoinResult(
                        left_index=left_idx,
                        right_index=right_idx,
                        confidence=ConfidenceLevel.HIGH,
                        join_method="uei_exact",
                        fields_used=[left_uei_col, right_uei_col],
                        match_score=1.0,
                    ))
    
    return results


def join_by_normalized_name(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    left_name_col: str,
    right_name_col: str,
) -> list[JoinResult]:
    """Join datasets by normalized entity name - MEDIUM confidence.
    
    Args:
        left_df: Left DataFrame
        right_df: Right DataFrame
        left_name_col: Normalized name column in left DataFrame
        right_name_col: Normalized name column in right DataFrame
    
    Returns:
        List of JoinResults with MEDIUM confidence
    """
    results = []
    
    # Create lookup from right DataFrame
    right_lookup = {}
    for idx, row in right_df.iterrows():
        name = row.get(right_name_col)
        if name and pd.notna(name) and str(name).strip():
            name_clean = str(name).strip()
            if name_clean not in right_lookup:
                right_lookup[name_clean] = []
            right_lookup[name_clean].append(idx)
    
    # Find matches from left DataFrame
    for left_idx, row in left_df.iterrows():
        name = row.get(left_name_col)
        if name and pd.notna(name) and str(name).strip():
            name_clean = str(name).strip()
            if name_clean in right_lookup:
                for right_idx in right_lookup[name_clean]:
                    results.append(JoinResult(
                        left_index=left_idx,
                        right_index=right_idx,
                        confidence=ConfidenceLevel.MEDIUM,
                        join_method="normalized_name_exact",
                        fields_used=[left_name_col, right_name_col],
                        match_score=1.0,
                    ))
    
    return results


def join_by_fuzzy_name(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    left_name_col: str,
    right_name_col: str,
    threshold: float = 80.0,
    max_candidates: int = 10,
) -> list[JoinResult]:
    """Join datasets by fuzzy name matching - LOW confidence.
    
    Uses rapidfuzz for efficient fuzzy matching.
    
    Args:
        left_df: Left DataFrame
        right_df: Right DataFrame
        left_name_col: Name column in left DataFrame
        right_name_col: Name column in right DataFrame
        threshold: Minimum match score (0-100)
        max_candidates: Max candidates to consider per left record
    
    Returns:
        List of JoinResults with LOW confidence
    """
    fuzz = _try_import_rapidfuzz()
    if fuzz is None:
        # Fallback to no fuzzy matching if rapidfuzz not available
        return []
    
    results = []
    
    # Build list of right names with indices
    right_names = []
    right_indices = []
    for idx, row in right_df.iterrows():
        name = row.get(right_name_col)
        if name and pd.notna(name) and str(name).strip():
            right_names.append(str(name).strip())
            right_indices.append(idx)
    
    if not right_names:
        return results
    
    # Find fuzzy matches for each left record
    for left_idx, row in left_df.iterrows():
        left_name = row.get(left_name_col)
        if not left_name or pd.isna(left_name) or not str(left_name).strip():
            continue
        
        left_name = str(left_name).strip()
        
        # Score against all right names
        scores = []
        for i, right_name in enumerate(right_names):
            score = fuzz.ratio(left_name, right_name)
            if score >= threshold:
                scores.append((score, right_indices[i]))
        
        # Sort by score and take top matches
        scores.sort(reverse=True)
        for score, right_idx in scores[:max_candidates]:
            results.append(JoinResult(
                left_index=left_idx,
                right_index=right_idx,
                confidence=ConfidenceLevel.LOW,
                join_method="fuzzy_name",
                fields_used=[left_name_col, right_name_col],
                match_score=score / 100.0,
            ))
    
    return results


def join_by_normalized_address(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    left_addr_col: str,
    right_addr_col: str,
) -> list[JoinResult]:
    """Join datasets by normalized address - MEDIUM confidence.
    
    Args:
        left_df: Left DataFrame
        right_df: Right DataFrame
        left_addr_col: Normalized address column in left DataFrame
        right_addr_col: Normalized address column in right DataFrame
    
    Returns:
        List of JoinResults with MEDIUM confidence
    """
    results = []
    
    # Create lookup from right DataFrame
    right_lookup = {}
    for idx, row in right_df.iterrows():
        addr = row.get(right_addr_col)
        if addr and pd.notna(addr) and str(addr).strip():
            addr_clean = str(addr).strip()
            if addr_clean not in right_lookup:
                right_lookup[addr_clean] = []
            right_lookup[addr_clean].append(idx)
    
    # Find matches from left DataFrame
    for left_idx, row in left_df.iterrows():
        addr = row.get(left_addr_col)
        if addr and pd.notna(addr) and str(addr).strip():
            addr_clean = str(addr).strip()
            if addr_clean in right_lookup:
                for right_idx in right_lookup[addr_clean]:
                    results.append(JoinResult(
                        left_index=left_idx,
                        right_index=right_idx,
                        confidence=ConfidenceLevel.MEDIUM,
                        join_method="normalized_address_exact",
                        fields_used=[left_addr_col, right_addr_col],
                        match_score=1.0,
                    ))
    
    return results


def join_by_address_id(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    left_id_col: str,
    right_id_col: str,
) -> list[JoinResult]:
    """Join datasets by address ID (hash) - HIGH confidence.
    
    Args:
        left_df: Left DataFrame
        right_df: Right DataFrame
        left_id_col: Address ID column in left DataFrame
        right_id_col: Address ID column in right DataFrame
    
    Returns:
        List of JoinResults with HIGH confidence (deterministic hash match)
    """
    results = []
    
    # Create lookup from right DataFrame
    right_lookup = {}
    for idx, row in right_df.iterrows():
        addr_id = row.get(right_id_col)
        if addr_id and pd.notna(addr_id) and str(addr_id).strip():
            addr_id_clean = str(addr_id).strip()
            if addr_id_clean not in right_lookup:
                right_lookup[addr_id_clean] = []
            right_lookup[addr_id_clean].append(idx)
    
    # Find matches from left DataFrame
    for left_idx, row in left_df.iterrows():
        addr_id = row.get(left_id_col)
        if addr_id and pd.notna(addr_id) and str(addr_id).strip():
            addr_id_clean = str(addr_id).strip()
            if addr_id_clean in right_lookup:
                for right_idx in right_lookup[addr_id_clean]:
                    results.append(JoinResult(
                        left_index=left_idx,
                        right_index=right_idx,
                        confidence=ConfidenceLevel.MEDIUM,  # Hash match = exact normalized
                        join_method="address_id_exact",
                        fields_used=[left_id_col, right_id_col],
                        match_score=1.0,
                    ))
    
    return results


class JoinEngine:
    """Engine for performing joins with confidence scoring."""
    
    def __init__(
        self,
        min_confidence: ConfidenceLevel = ConfidenceLevel.LOW,
        enable_fuzzy: bool = True,
        fuzzy_threshold: float = 80.0,
    ):
        """Initialize join engine.
        
        Args:
            min_confidence: Minimum confidence level to include
            enable_fuzzy: Whether to enable fuzzy matching
            fuzzy_threshold: Threshold for fuzzy matching (0-100)
        """
        self.min_confidence = min_confidence
        self.enable_fuzzy = enable_fuzzy
        self.fuzzy_threshold = fuzzy_threshold
        self._join_history: list[JoinMetadata] = []
    
    def _filter_by_confidence(
        self,
        results: list[JoinResult]
    ) -> list[JoinResult]:
        """Filter results by minimum confidence level."""
        confidence_order = [
            ConfidenceLevel.HIGH,
            ConfidenceLevel.MEDIUM,
            ConfidenceLevel.LOW,
        ]
        
        min_idx = confidence_order.index(self.min_confidence)
        allowed = set(confidence_order[:min_idx + 1])
        
        return [r for r in results if r.confidence in allowed]
    
    def join_entities(
        self,
        left_df: pd.DataFrame,
        right_df: pd.DataFrame,
        left_name: str,
        right_name: str,
        left_uei_col: str = None,
        right_uei_col: str = None,
        left_name_col: str = None,
        right_name_col: str = None,
        left_normalized_col: str = None,
        right_normalized_col: str = None,
    ) -> JoinMetadata:
        """Join two datasets by entity.
        
        Attempts joins in order of confidence:
        1. UEI exact match (HIGH)
        2. Normalized name exact match (MEDIUM)
        3. Fuzzy name match (LOW)
        
        Args:
            left_df: Left DataFrame
            right_df: Right DataFrame
            left_name: Name of left dataset
            right_name: Name of right dataset
            left_uei_col: UEI column in left DataFrame
            right_uei_col: UEI column in right DataFrame
            left_name_col: Raw name column in left DataFrame
            right_name_col: Raw name column in right DataFrame
            left_normalized_col: Normalized name column in left DataFrame
            right_normalized_col: Normalized name column in right DataFrame
        
        Returns:
            JoinMetadata with results
        """
        all_results = []
        fields_used = []
        
        # Track which pairs are already matched at higher confidence
        matched_pairs = set()
        
        # 1. UEI exact match (HIGH confidence)
        if left_uei_col and right_uei_col:
            results = join_by_uei(left_df, right_df, left_uei_col, right_uei_col)
            for r in results:
                matched_pairs.add((r.left_index, r.right_index))
            all_results.extend(results)
            fields_used.extend([left_uei_col, right_uei_col])
        
        # 2. Normalized name exact match (MEDIUM confidence)
        if left_normalized_col and right_normalized_col:
            results = join_by_normalized_name(
                left_df, right_df,
                left_normalized_col, right_normalized_col
            )
            # Exclude pairs already matched at higher confidence
            results = [
                r for r in results
                if (r.left_index, r.right_index) not in matched_pairs
            ]
            for r in results:
                matched_pairs.add((r.left_index, r.right_index))
            all_results.extend(results)
            fields_used.extend([left_normalized_col, right_normalized_col])
        
        # 3. Fuzzy name match (LOW confidence)
        if self.enable_fuzzy and left_name_col and right_name_col:
            results = join_by_fuzzy_name(
                left_df, right_df,
                left_name_col, right_name_col,
                threshold=self.fuzzy_threshold
            )
            # Exclude pairs already matched at higher confidence
            results = [
                r for r in results
                if (r.left_index, r.right_index) not in matched_pairs
            ]
            all_results.extend(results)
            fields_used.extend([left_name_col, right_name_col])
        
        # Filter by minimum confidence
        all_results = self._filter_by_confidence(all_results)
        
        # Create metadata
        metadata = JoinMetadata(
            join_type="entity_to_entity",
            left_dataset=left_name,
            right_dataset=right_name,
            total_matches=len(all_results),
            high_confidence=sum(
                1 for r in all_results
                if r.confidence == ConfidenceLevel.HIGH
            ),
            medium_confidence=sum(
                1 for r in all_results
                if r.confidence == ConfidenceLevel.MEDIUM
            ),
            low_confidence=sum(
                1 for r in all_results
                if r.confidence == ConfidenceLevel.LOW
            ),
            fields_used=list(set(fields_used)),
            results=all_results,
        )
        
        self._join_history.append(metadata)
        return metadata
    
    def join_by_address(
        self,
        left_df: pd.DataFrame,
        right_df: pd.DataFrame,
        left_name: str,
        right_name: str,
        left_addr_col: str = None,
        right_addr_col: str = None,
        left_addr_id_col: str = None,
        right_addr_id_col: str = None,
    ) -> JoinMetadata:
        """Join two datasets by address.
        
        Args:
            left_df: Left DataFrame
            right_df: Right DataFrame
            left_name: Name of left dataset
            right_name: Name of right dataset
            left_addr_col: Normalized address column in left DataFrame
            right_addr_col: Normalized address column in right DataFrame
            left_addr_id_col: Address ID column in left DataFrame
            right_addr_id_col: Address ID column in right DataFrame
        
        Returns:
            JoinMetadata with results
        """
        all_results = []
        fields_used = []
        
        # Join by address ID (uses normalized address hash)
        if left_addr_id_col and right_addr_id_col:
            results = join_by_address_id(
                left_df, right_df,
                left_addr_id_col, right_addr_id_col
            )
            all_results.extend(results)
            fields_used.extend([left_addr_id_col, right_addr_id_col])
        elif left_addr_col and right_addr_col:
            results = join_by_normalized_address(
                left_df, right_df,
                left_addr_col, right_addr_col
            )
            all_results.extend(results)
            fields_used.extend([left_addr_col, right_addr_col])
        
        # Filter by minimum confidence
        all_results = self._filter_by_confidence(all_results)
        
        # Create metadata
        metadata = JoinMetadata(
            join_type="entity_to_address",
            left_dataset=left_name,
            right_dataset=right_name,
            total_matches=len(all_results),
            high_confidence=sum(
                1 for r in all_results
                if r.confidence == ConfidenceLevel.HIGH
            ),
            medium_confidence=sum(
                1 for r in all_results
                if r.confidence == ConfidenceLevel.MEDIUM
            ),
            low_confidence=sum(
                1 for r in all_results
                if r.confidence == ConfidenceLevel.LOW
            ),
            fields_used=list(set(fields_used)),
            results=all_results,
        )
        
        self._join_history.append(metadata)
        return metadata
    
    def get_join_history(self) -> list[dict]:
        """Get history of all joins performed."""
        return [m.to_dict() for m in self._join_history]
    
    def clear_history(self) -> None:
        """Clear join history."""
        self._join_history = []


if __name__ == "__main__":
    # Example usage
    print("Join Engine example:")
    
    # Create sample data
    left_data = pd.DataFrame({
        "name": ["ACME Corp", "Smith & Jones LLC", "ABC Company"],
        "name_normalized": ["ACME", "SMITH JONES", "ABC"],
        "uei": ["ABC123", None, "XYZ789"],
    })
    
    right_data = pd.DataFrame({
        "entity_name": ["Acme Corporation", "Smith Jones", "XYZ Inc"],
        "entity_name_normalized": ["ACME", "SMITH JONES", "XYZ"],
        "entity_uei": ["ABC123", "DEF456", None],
    })
    
    engine = JoinEngine(enable_fuzzy=True)
    
    result = engine.join_entities(
        left_data, right_data,
        "left_dataset", "right_dataset",
        left_uei_col="uei",
        right_uei_col="entity_uei",
        left_normalized_col="name_normalized",
        right_normalized_col="entity_name_normalized",
    )
    
    print(f"\nTotal matches: {result.total_matches}")
    print(f"  HIGH confidence: {result.high_confidence}")
    print(f"  MEDIUM confidence: {result.medium_confidence}")
    print(f"  LOW confidence: {result.low_confidence}")
    
    for r in result.results:
        print(f"\n  Match: {r.left_index} <-> {r.right_index}")
        print(f"    Confidence: {r.confidence.value}")
        print(f"    Method: {r.join_method}")
        print(f"    Fields: {r.fields_used}")
