"""Entity and address normalization engine.

Normalizes entity names and addresses for safe joins while preserving raw values.
Generates deterministic hashes for entity_id and address_id.
"""

import hashlib
import re
from typing import Optional

import pandas as pd


# Entity name normalization patterns
ENTITY_SUFFIXES = [
    # Order matters - longer patterns first
    r"\bLIMITED\s+LIABILITY\s+COMPANY\b",
    r"\bLIMITED\s+LIABILITY\s+PARTNERSHIP\b",
    r"\bLIMITED\s+PARTNERSHIP\b",
    r"\bLIMITED\b",
    r"\bINCORPORATED\b",
    r"\bCORPORATION\b",
    r"\bCOMPANY\b",
    r"\bL\.?L\.?C\.?\b",
    r"\bL\.?L\.?P\.?\b",
    r"\bL\.?P\.?\b",
    r"\bL\.?T\.?D\.?\b",
    r"\bINC\.?\b",
    r"\bCORP\.?\b",
    r"\bCO\.?\b",
    r"\bP\.?C\.?\b",  # Professional Corporation
    r"\bP\.?L\.?L\.?C\.?\b",  # Professional LLC
    r"\bP\.?A\.?\b",  # Professional Association
]

# Compile entity suffix patterns
ENTITY_SUFFIX_PATTERN = re.compile(
    r"\s*(?:" + "|".join(ENTITY_SUFFIXES) + r")\s*",
    re.IGNORECASE
)

# Common punctuation to remove from entity names
ENTITY_PUNCTUATION = re.compile(r"[.,\-'\"&()]")

# Street suffix standardization
STREET_SUFFIXES = {
    # Street
    r"\bSTREET\b": "ST",
    r"\bSTR\b": "ST",
    # Avenue
    r"\bAVENUE\b": "AVE",
    r"\bAVN\b": "AVE",
    r"\bAV\b": "AVE",
    # Road
    r"\bROAD\b": "RD",
    # Drive
    r"\bDRIVE\b": "DR",
    r"\bDRV\b": "DR",
    # Boulevard
    r"\bBOULEVARD\b": "BLVD",
    r"\bBLV\b": "BLVD",
    # Lane
    r"\bLANE\b": "LN",
    # Court
    r"\bCOURT\b": "CT",
    # Place
    r"\bPLACE\b": "PL",
    # Circle
    r"\bCIRCLE\b": "CIR",
    # Highway
    r"\bHIGHWAY\b": "HWY",
    # Parkway
    r"\bPARKWAY\b": "PKWY",
    # Expressway
    r"\bEXPRESSWAY\b": "EXPY",
    # Terrace
    r"\bTERRACE\b": "TER",
    # Way
    r"\bWAY\b": "WAY",
    # Trail
    r"\bTRAIL\b": "TRL",
    # Square
    r"\bSQUARE\b": "SQ",
    # Pike
    r"\bPIKE\b": "PIKE",
    # Point
    r"\bPOINT\b": "PT",
    # Commons
    r"\bCOMMONS\b": "CMNS",
    # Crossing
    r"\bCROSSING\b": "XING",
    # Alley
    r"\bALLEY\b": "ALY",
    # Suite abbreviations
    r"\bSUITE\b": "STE",
    # Floor
    r"\bFLOOR\b": "FL",
}

# Directional standardization
DIRECTIONALS = {
    r"\bNORTH\b": "N",
    r"\bSOUTH\b": "S",
    r"\bEAST\b": "E",
    r"\bWEST\b": "W",
    r"\bNORTHEAST\b": "NE",
    r"\bNORTHWEST\b": "NW",
    r"\bSOUTHEAST\b": "SE",
    r"\bSOUTHWEST\b": "SW",
}

# Compile patterns for efficiency
STREET_SUFFIX_PATTERNS = [
    (re.compile(pattern, re.IGNORECASE), replacement)
    for pattern, replacement in STREET_SUFFIXES.items()
]

DIRECTIONAL_PATTERNS = [
    (re.compile(pattern, re.IGNORECASE), replacement)
    for pattern, replacement in DIRECTIONALS.items()
]


def normalize_entity_name(name: Optional[str]) -> Optional[str]:
    """Normalize an entity name for matching.
    
    Steps:
    1. Convert to uppercase
    2. Remove punctuation
    3. Strip LLC/INC/CORP variants
    4. Collapse whitespace
    
    Args:
        name: Raw entity name
    
    Returns:
        Normalized name, or None if input is empty
    """
    if not name or pd.isna(name):
        return None
    
    name = str(name).strip()
    if not name:
        return None
    
    # Convert to uppercase
    normalized = name.upper()
    
    # Remove common punctuation
    normalized = ENTITY_PUNCTUATION.sub(" ", normalized)
    
    # Remove entity suffixes
    normalized = ENTITY_SUFFIX_PATTERN.sub(" ", normalized)
    
    # Collapse whitespace
    normalized = " ".join(normalized.split())
    
    return normalized if normalized else None


def normalize_address(address: Optional[str]) -> Optional[str]:
    """Normalize an address for matching.
    
    Steps:
    1. Convert to uppercase
    2. Standardize street suffixes
    3. Standardize directionals
    4. Collapse whitespace
    
    Args:
        address: Raw address string
    
    Returns:
        Normalized address, or None if input is empty
    """
    if not address or pd.isna(address):
        return None
    
    address = str(address).strip()
    if not address:
        return None
    
    # Convert to uppercase
    normalized = address.upper()
    
    # Standardize street suffixes
    for pattern, replacement in STREET_SUFFIX_PATTERNS:
        normalized = pattern.sub(replacement, normalized)
    
    # Standardize directionals
    for pattern, replacement in DIRECTIONAL_PATTERNS:
        normalized = pattern.sub(replacement, normalized)
    
    # Remove extra punctuation but keep numbers
    normalized = re.sub(r"[.,#]", " ", normalized)
    
    # Collapse whitespace
    normalized = " ".join(normalized.split())
    
    return normalized if normalized else None


def generate_hash(value: Optional[str]) -> Optional[str]:
    """Generate a deterministic SHA-256 hash for a value.
    
    Args:
        value: String to hash
    
    Returns:
        First 16 characters of SHA-256 hash, or None if input is empty
    """
    if not value:
        return None
    
    # Ensure value is a string
    if not isinstance(value, str):
        value = str(value)
    
    # Use SHA-256 for deterministic hashing
    hash_bytes = hashlib.sha256(value.encode("utf-8")).hexdigest()
    
    # Return first 16 characters for readability
    return hash_bytes[:16]


def generate_entity_id(name: Optional[str]) -> Optional[str]:
    """Generate a deterministic entity ID from a name.
    
    Uses the normalized name to generate the hash.
    
    Args:
        name: Raw entity name
    
    Returns:
        Entity ID hash, or None if name is empty
    """
    normalized = normalize_entity_name(name)
    return generate_hash(normalized)


def generate_address_id(address: Optional[str]) -> Optional[str]:
    """Generate a deterministic address ID from an address.
    
    Uses the normalized address to generate the hash.
    
    Args:
        address: Raw address string
    
    Returns:
        Address ID hash, or None if address is empty
    """
    normalized = normalize_address(address)
    return generate_hash(normalized)


def normalize_entity_column(
    df: pd.DataFrame,
    raw_column: str,
    normalized_column: str = None,
    id_column: str = None
) -> pd.DataFrame:
    """Add normalized entity name and ID columns to a DataFrame.
    
    Preserves the raw column and adds new columns for normalized value and ID.
    
    Args:
        df: DataFrame to process
        raw_column: Name of column containing raw entity names
        normalized_column: Name for normalized column (default: {raw_column}_normalized)
        id_column: Name for ID column (default: {raw_column}_id)
    
    Returns:
        DataFrame with added columns
    """
    df = df.copy()
    
    if normalized_column is None:
        normalized_column = f"{raw_column}_normalized"
    if id_column is None:
        id_column = f"{raw_column}_id"
    
    df[normalized_column] = df[raw_column].apply(normalize_entity_name)
    df[id_column] = df[normalized_column].apply(generate_hash)
    
    return df


def normalize_address_column(
    df: pd.DataFrame,
    raw_column: str,
    normalized_column: str = None,
    id_column: str = None
) -> pd.DataFrame:
    """Add normalized address and ID columns to a DataFrame.
    
    Preserves the raw column and adds new columns for normalized value and ID.
    
    Args:
        df: DataFrame to process
        raw_column: Name of column containing raw addresses
        normalized_column: Name for normalized column (default: {raw_column}_normalized)
        id_column: Name for ID column (default: {raw_column}_id)
    
    Returns:
        DataFrame with added columns
    """
    df = df.copy()
    
    if normalized_column is None:
        normalized_column = f"{raw_column}_normalized"
    if id_column is None:
        id_column = f"{raw_column}_id"
    
    df[normalized_column] = df[raw_column].apply(normalize_address)
    df[id_column] = df[normalized_column].apply(generate_hash)
    
    return df


def normalize_dataframe(
    df: pd.DataFrame,
    entity_columns: list[str] = None,
    address_columns: list[str] = None
) -> pd.DataFrame:
    """Normalize all specified entity and address columns in a DataFrame.
    
    Args:
        df: DataFrame to process
        entity_columns: List of columns containing entity names
        address_columns: List of columns containing addresses
    
    Returns:
        DataFrame with added normalized columns and IDs
    """
    df = df.copy()
    
    if entity_columns:
        for col in entity_columns:
            if col in df.columns:
                df = normalize_entity_column(df, col)
    
    if address_columns:
        for col in address_columns:
            if col in df.columns:
                df = normalize_address_column(df, col)
    
    return df


if __name__ == "__main__":
    # Example usage
    print("Entity normalization examples:")
    examples = [
        "ACME Corporation, Inc.",
        "Smith & Jones LLC",
        "ABC Limited Liability Company",
        "John's Plumbing L.L.C.",
    ]
    for name in examples:
        normalized = normalize_entity_name(name)
        entity_id = generate_entity_id(name)
        print(f"  '{name}' -> '{normalized}' (ID: {entity_id})")
    
    print("\nAddress normalization examples:")
    addresses = [
        "123 Main Street, Suite 100",
        "456 North Oak Avenue",
        "789 Southwest Highway",
    ]
    for addr in addresses:
        normalized = normalize_address(addr)
        addr_id = generate_address_id(addr)
        print(f"  '{addr}' -> '{normalized}' (ID: {addr_id})")
