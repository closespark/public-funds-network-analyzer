"""CSV loaders for heterogeneous data sources.

Each loader:
- Preserves raw column names alongside standardized fields
- Records metadata (source, date, row count)
- Performs only type casting, no transformations
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from .schema import (
    BUSINESS_LICENSE_FIELDS,
    CITY_CONTRACT_FIELDS,
    DATASET_TYPES,
    DELINQUENT_PROPERTY_FIELDS,
    DatasetMetadata,
    FEDERAL_ASSISTANCE_AWARD_FIELDS,
    FEDERAL_PRIME_AWARD_FIELDS,
    FEDERAL_SUBAWARD_FIELDS,
    FEDERAL_TRANSACTION_FIELDS,
)


class DatasetRegistry:
    """Registry for tracking loaded datasets and their metadata."""
    
    def __init__(self):
        self._datasets: dict[str, pd.DataFrame] = {}
        self._metadata: dict[str, DatasetMetadata] = {}
    
    def register(
        self,
        name: str,
        df: pd.DataFrame,
        metadata: DatasetMetadata
    ) -> None:
        """Register a loaded dataset with its metadata."""
        self._datasets[name] = df
        self._metadata[name] = metadata
    
    def get_dataset(self, name: str) -> Optional[pd.DataFrame]:
        """Get a dataset by name."""
        return self._datasets.get(name)
    
    def get_metadata(self, name: str) -> Optional[DatasetMetadata]:
        """Get metadata for a dataset."""
        return self._metadata.get(name)
    
    def list_datasets(self) -> list[str]:
        """List all registered dataset names."""
        return list(self._datasets.keys())
    
    def get_all_metadata(self) -> dict[str, dict]:
        """Get metadata for all datasets as dictionaries."""
        return {
            name: meta.to_dict()
            for name, meta in self._metadata.items()
        }


def _create_metadata(
    filepath: Path,
    source_type: str,
    df: pd.DataFrame
) -> DatasetMetadata:
    """Create metadata for a loaded dataset."""
    return DatasetMetadata(
        source_file=str(filepath),
        source_type=source_type,
        load_timestamp=datetime.utcnow(),
        row_count=len(df),
        column_count=len(df.columns),
        original_columns=list(df.columns),
        file_size_bytes=filepath.stat().st_size,
    )


def _add_raw_columns(
    df: pd.DataFrame,
    field_mapping: dict[str, str]
) -> pd.DataFrame:
    """Add standardized columns while preserving raw values.
    
    For each standardized field, we:
    1. Keep the original column
    2. Add a new column with the standardized name prefixed with '_std_'
    """
    df = df.copy()
    
    for std_name, raw_name in field_mapping.items():
        if raw_name in df.columns:
            # Add standardized column (just a reference to raw for now)
            df[f"_std_{std_name}"] = df[raw_name]
    
    return df


def load_csv(
    filepath: Path | str,
    source_type: str,
) -> tuple[pd.DataFrame, DatasetMetadata]:
    """Load a CSV file and add standardized field references.
    
    Args:
        filepath: Path to the CSV file
        source_type: Type of dataset (must be in DATASET_TYPES)
    
    Returns:
        Tuple of (DataFrame with raw + standardized columns, metadata)
    
    Raises:
        ValueError: If source_type is not recognized
        FileNotFoundError: If file doesn't exist
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    if source_type not in DATASET_TYPES:
        raise ValueError(
            f"Unknown source type: {source_type}. "
            f"Valid types: {list(DATASET_TYPES.keys())}"
        )
    
    # Load with string dtype to avoid type coercion issues
    df = pd.read_csv(filepath, dtype=str, low_memory=False)
    
    # Add standardized columns
    field_mapping = DATASET_TYPES[source_type]
    df = _add_raw_columns(df, field_mapping)
    
    # Create metadata
    metadata = _create_metadata(filepath, source_type, df)
    
    return df, metadata


def load_business_licenses(filepath: Path | str) -> tuple[pd.DataFrame, DatasetMetadata]:
    """Load business licenses CSV."""
    return load_csv(filepath, "business_licenses")


def load_city_contracts(filepath: Path | str) -> tuple[pd.DataFrame, DatasetMetadata]:
    """Load city contracts CSV."""
    return load_csv(filepath, "city_contracts")


def load_delinquent_properties(filepath: Path | str) -> tuple[pd.DataFrame, DatasetMetadata]:
    """Load delinquent properties CSV."""
    return load_csv(filepath, "delinquent_properties")


def load_federal_contracts_prime(filepath: Path | str) -> tuple[pd.DataFrame, DatasetMetadata]:
    """Load federal contracts prime awards CSV."""
    return load_csv(filepath, "federal_contracts_prime")


def load_federal_assistance_prime(filepath: Path | str) -> tuple[pd.DataFrame, DatasetMetadata]:
    """Load federal assistance prime awards CSV."""
    return load_csv(filepath, "federal_assistance_prime")


def load_federal_transactions(filepath: Path | str) -> tuple[pd.DataFrame, DatasetMetadata]:
    """Load federal transactions CSV."""
    return load_csv(filepath, "federal_transactions")


def load_federal_subawards(filepath: Path | str) -> tuple[pd.DataFrame, DatasetMetadata]:
    """Load federal subawards CSV."""
    return load_csv(filepath, "federal_subawards")


def discover_and_load_all(
    data_dir: Path | str,
    registry: Optional[DatasetRegistry] = None
) -> DatasetRegistry:
    """Discover and load all recognized CSV files from a data directory.
    
    File name patterns:
    - Business_Licenses*.csv -> business_licenses
    - City_Contracts*.csv -> city_contracts
    - Delinquent*.csv -> delinquent_properties
    - Contracts_PrimeAwardSummaries*.csv -> federal_contracts_prime
    - Contracts_PrimeTransactions*.csv -> federal_transactions
    - Contracts_Subawards*.csv -> federal_subawards
    - Assistance_PrimeAwardSummaries*.csv -> federal_assistance_prime
    - Assistance_PrimeTransactions*.csv -> (federal_transactions, assistance type)
    - Assistance_Subawards*.csv -> federal_subawards
    
    Args:
        data_dir: Directory containing CSV files
        registry: Optional existing registry to add to
    
    Returns:
        DatasetRegistry with all loaded datasets
    """
    data_dir = Path(data_dir)
    
    if registry is None:
        registry = DatasetRegistry()
    
    # Define patterns and their loaders
    patterns = [
        ("Business_Licenses*.csv", "business_licenses", load_business_licenses),
        ("City_Contracts*.csv", "city_contracts", load_city_contracts),
        ("Delinquent*.csv", "delinquent_properties", load_delinquent_properties),
        ("Contracts_PrimeAwardSummaries*.csv", "federal_contracts_prime", load_federal_contracts_prime),
        ("Contracts_PrimeTransactions*.csv", "federal_contracts_transactions", load_federal_transactions),
        ("Contracts_Subawards*.csv", "federal_contracts_subawards", load_federal_subawards),
        ("Assistance_PrimeAwardSummaries*.csv", "federal_assistance_prime", load_federal_assistance_prime),
        ("Assistance_PrimeTransactions*.csv", "federal_assistance_transactions", load_federal_transactions),
        ("Assistance_Subawards*.csv", "federal_assistance_subawards", load_federal_subawards),
    ]
    
    for pattern, dataset_name, loader in patterns:
        matches = list(data_dir.glob(pattern))
        
        for i, filepath in enumerate(matches):
            # Handle multiple files of same type
            name = dataset_name if len(matches) == 1 else f"{dataset_name}_{i+1}"
            
            try:
                df, metadata = loader(filepath)
                registry.register(name, df, metadata)
            except Exception as e:
                # Log error but continue with other files
                print(f"Warning: Failed to load {filepath}: {e}")
    
    return registry


if __name__ == "__main__":
    # Example usage
    import sys
    
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data")
    
    print(f"Loading datasets from {data_dir}...")
    registry = discover_and_load_all(data_dir)
    
    print("\nLoaded datasets:")
    for name in registry.list_datasets():
        meta = registry.get_metadata(name)
        print(f"  {name}: {meta.row_count} rows, {meta.column_count} columns")
