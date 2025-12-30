"""Schema definitions for unified internal representation.

Each dataset preserves raw column names alongside standardized fields.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional


@dataclass
class DatasetMetadata:
    """Metadata for a loaded dataset."""
    
    source_file: str
    source_type: str
    load_timestamp: datetime
    row_count: int
    column_count: int
    original_columns: list[str]
    file_size_bytes: int = 0
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "source_file": self.source_file,
            "source_type": self.source_type,
            "load_timestamp": self.load_timestamp.isoformat(),
            "row_count": self.row_count,
            "column_count": self.column_count,
            "original_columns": self.original_columns,
            "file_size_bytes": self.file_size_bytes,
        }


# Standardized field mappings for each dataset type
# Maps standard field name -> source column name(s)

BUSINESS_LICENSE_FIELDS = {
    "entity_name": "Business Name",
    "dba_name": "Doing Business As",
    "address": "Business Address",
    "geo_location": "Business Geo Location",
}

CITY_CONTRACT_FIELDS = {
    "agency": "Agency/Department",
    "contract_number": "Contract Number",
    "contract_value": "Contract Value",
    "supplier_name": "Supplier",
    "procurement_type": "Procurement Type",
    "description": "Description",
    "solicitation_type": "Type of Solicitation",
    "effective_from": "Effective From",
    "effective_to": "Effective To",
}

DELINQUENT_PROPERTY_FIELDS = {
    "property_code": "Property Code",
    "owner_name_1": "Current Owner Name 1",
    "owner_name_2": "Current Owner Name 2",
    "address": "Physical Address",
    "total_due": "Total Due",
    "years_delinquent": "Total Years Delinquent",
    "geo_location": "GIS Location",
}

# Federal data has many columns - we map the key ones
FEDERAL_PRIME_AWARD_FIELDS = {
    "award_id": "prime_award_unique_key",
    "piid": "prime_award_piid",
    "award_amount": "prime_award_amount",
    "recipient_uei": "prime_awardee_uei",
    "recipient_name": "prime_awardee_name",
    "recipient_dba": "prime_awardee_dba_name",
    "recipient_address": "prime_awardee_address_line_1",
    "recipient_city": "prime_awardee_city_name",
    "recipient_state": "prime_awardee_state_code",
    "recipient_zip": "prime_awardee_zip_code",
    "awarding_agency": "prime_award_awarding_agency_name",
    "funding_agency": "prime_award_funding_agency_name",
    "action_date": "prime_award_base_action_date",
    "naics_code": "prime_award_naics_code",
    "description": "prime_award_base_transaction_description",
    "usaspending_permalink": "usaspending_permalink",
}

FEDERAL_TRANSACTION_FIELDS = {
    "transaction_id": "prime_award_transaction_key",
    "award_id": "prime_award_unique_key",
    "action_date": "action_date",
    "action_type": "action_type",
    "federal_action_obligation": "federal_action_obligation",
    "recipient_uei": "recipient_uei",
    "recipient_name": "recipient_name",
    "recipient_address": "recipient_address_line_1",
    "recipient_city": "recipient_city_name",
    "recipient_state": "recipient_state_code",
    "recipient_zip": "recipient_zip_code",
    "awarding_agency": "awarding_agency_name",
    "funding_agency": "funding_agency_name",
    "description": "transaction_description",
}

FEDERAL_SUBAWARD_FIELDS = {
    "prime_award_id": "prime_award_unique_key",
    "prime_award_piid": "prime_award_piid",
    "subaward_number": "subaward_number",
    "subaward_amount": "subaward_amount",
    "subaward_action_date": "subaward_action_date",
    "subawardee_uei": "subawardee_uei",
    "subawardee_name": "subawardee_name",
    "subawardee_dba": "subawardee_dba_name",
    "subawardee_address": "subawardee_address_line_1",
    "subawardee_city": "subawardee_city_name",
    "subawardee_state": "subawardee_state_code",
    "subawardee_zip": "subawardee_zip_code",
    "prime_recipient_uei": "prime_awardee_uei",
    "prime_recipient_name": "prime_awardee_name",
    "description": "subaward_description",
    "usaspending_permalink": "usaspending_permalink",
}

# Assistance (grants) specific mappings
FEDERAL_ASSISTANCE_AWARD_FIELDS = {
    "award_id": "assistance_award_unique_key",
    "fain": "award_id_fain",
    "total_obligated": "total_obligated_amount",
    "total_outlayed": "total_outlayed_amount",
    "recipient_uei": "recipient_uei",
    "recipient_name": "recipient_name",
    "recipient_address": "recipient_address_line_1",
    "recipient_city": "recipient_city_name",
    "recipient_state": "recipient_state_code",
    "recipient_zip": "recipient_zip_code",
    "awarding_agency": "awarding_agency_name",
    "funding_agency": "funding_agency_name",
    "action_date": "award_base_action_date",
    "cfda_numbers": "cfda_numbers_and_titles",
    "description": "prime_award_base_transaction_description",
    "usaspending_permalink": "usaspending_permalink",
}

# Dataset types
DATASET_TYPES = {
    "business_licenses": BUSINESS_LICENSE_FIELDS,
    "city_contracts": CITY_CONTRACT_FIELDS,
    "delinquent_properties": DELINQUENT_PROPERTY_FIELDS,
    "federal_contracts_prime": FEDERAL_PRIME_AWARD_FIELDS,
    "federal_assistance_prime": FEDERAL_ASSISTANCE_AWARD_FIELDS,
    "federal_transactions": FEDERAL_TRANSACTION_FIELDS,
    "federal_subawards": FEDERAL_SUBAWARD_FIELDS,
}
