"""Graph construction layer.

Builds a queryable relationship graph with defined node and edge types.
Exports graph as edge list CSV, node list CSV, and optional NetworkX object.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import pandas as pd


class NodeType(Enum):
    """Types of nodes in the graph."""
    ENTITY = "Entity"
    ADDRESS = "Address"
    PROPERTY = "Property"
    AWARD = "Award"


class EdgeType(Enum):
    """Types of edges in the graph."""
    LOCATED_AT = "LOCATED_AT"
    OWNS_PROPERTY = "OWNS_PROPERTY"
    CITY_CONTRACTED_WITH = "CITY_CONTRACTED_WITH"
    FEDERAL_PRIME_AWARDED = "FEDERAL_PRIME_AWARDED"
    FEDERAL_SUBAWARDED = "FEDERAL_SUBAWARDED"


@dataclass
class Node:
    """A node in the relationship graph."""
    
    node_id: str
    node_type: NodeType
    label: str
    source_dataset: str
    source_row: Any  # Index in source DataFrame
    attributes: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "node_id": self.node_id,
            "node_type": self.node_type.value,
            "label": self.label,
            "source_dataset": self.source_dataset,
            "source_row": str(self.source_row),
            **{f"attr_{k}": v for k, v in self.attributes.items()},
        }


@dataclass
class Edge:
    """An edge in the relationship graph."""
    
    source_id: str
    target_id: str
    edge_type: EdgeType
    source_dataset: str
    confidence: str = "MEDIUM"
    join_method: str = ""
    attributes: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "edge_type": self.edge_type.value,
            "source_dataset": self.source_dataset,
            "confidence": self.confidence,
            "join_method": self.join_method,
            **{f"attr_{k}": v for k, v in self.attributes.items()},
        }


class GraphBuilder:
    """Builder for the relationship graph."""
    
    def __init__(self):
        self._nodes: dict[str, Node] = {}
        self._edges: list[Edge] = []
        self._build_timestamp: Optional[datetime] = None
    
    def add_node(self, node: Node) -> None:
        """Add a node to the graph.
        
        If a node with the same ID already exists, it is updated.
        """
        self._nodes[node.node_id] = node
    
    def add_edge(self, edge: Edge) -> None:
        """Add an edge to the graph.
        
        Both source and target nodes should exist (or be added before export).
        """
        self._edges.append(edge)
    
    def get_node(self, node_id: str) -> Optional[Node]:
        """Get a node by ID."""
        return self._nodes.get(node_id)
    
    def has_node(self, node_id: str) -> bool:
        """Check if a node exists."""
        return node_id in self._nodes
    
    def add_entity_node(
        self,
        entity_id: str,
        name: str,
        source_dataset: str,
        source_row: Any,
        **attributes
    ) -> Node:
        """Add an entity node."""
        node = Node(
            node_id=f"entity:{entity_id}",
            node_type=NodeType.ENTITY,
            label=name,
            source_dataset=source_dataset,
            source_row=source_row,
            attributes=attributes,
        )
        self.add_node(node)
        return node
    
    def add_address_node(
        self,
        address_id: str,
        address: str,
        source_dataset: str,
        source_row: Any,
        **attributes
    ) -> Node:
        """Add an address node."""
        node = Node(
            node_id=f"address:{address_id}",
            node_type=NodeType.ADDRESS,
            label=address,
            source_dataset=source_dataset,
            source_row=source_row,
            attributes=attributes,
        )
        self.add_node(node)
        return node
    
    def add_property_node(
        self,
        property_id: str,
        address: str,
        source_dataset: str,
        source_row: Any,
        **attributes
    ) -> Node:
        """Add a property node."""
        node = Node(
            node_id=f"property:{property_id}",
            node_type=NodeType.PROPERTY,
            label=address,
            source_dataset=source_dataset,
            source_row=source_row,
            attributes=attributes,
        )
        self.add_node(node)
        return node
    
    def add_award_node(
        self,
        award_id: str,
        description: str,
        source_dataset: str,
        source_row: Any,
        award_type: str = "unknown",
        **attributes
    ) -> Node:
        """Add an award node."""
        node = Node(
            node_id=f"award:{award_id}",
            node_type=NodeType.AWARD,
            label=description[:100] if description else "",
            source_dataset=source_dataset,
            source_row=source_row,
            attributes={"award_type": award_type, **attributes},
        )
        self.add_node(node)
        return node
    
    def add_located_at_edge(
        self,
        entity_id: str,
        address_id: str,
        source_dataset: str,
        confidence: str = "MEDIUM",
        **attributes
    ) -> Edge:
        """Add LOCATED_AT edge between entity and address."""
        edge = Edge(
            source_id=f"entity:{entity_id}",
            target_id=f"address:{address_id}",
            edge_type=EdgeType.LOCATED_AT,
            source_dataset=source_dataset,
            confidence=confidence,
            attributes=attributes,
        )
        self.add_edge(edge)
        return edge
    
    def add_owns_property_edge(
        self,
        entity_id: str,
        property_id: str,
        source_dataset: str,
        confidence: str = "MEDIUM",
        **attributes
    ) -> Edge:
        """Add OWNS_PROPERTY edge between entity and property."""
        edge = Edge(
            source_id=f"entity:{entity_id}",
            target_id=f"property:{property_id}",
            edge_type=EdgeType.OWNS_PROPERTY,
            source_dataset=source_dataset,
            confidence=confidence,
            attributes=attributes,
        )
        self.add_edge(edge)
        return edge
    
    def add_city_contract_edge(
        self,
        entity_id: str,
        contract_id: str,
        source_dataset: str,
        confidence: str = "MEDIUM",
        **attributes
    ) -> Edge:
        """Add CITY_CONTRACTED_WITH edge between entity and award."""
        edge = Edge(
            source_id=f"entity:{entity_id}",
            target_id=f"award:{contract_id}",
            edge_type=EdgeType.CITY_CONTRACTED_WITH,
            source_dataset=source_dataset,
            confidence=confidence,
            attributes=attributes,
        )
        self.add_edge(edge)
        return edge
    
    def add_federal_prime_edge(
        self,
        entity_id: str,
        award_id: str,
        source_dataset: str,
        confidence: str = "MEDIUM",
        **attributes
    ) -> Edge:
        """Add FEDERAL_PRIME_AWARDED edge between entity and award."""
        edge = Edge(
            source_id=f"entity:{entity_id}",
            target_id=f"award:{award_id}",
            edge_type=EdgeType.FEDERAL_PRIME_AWARDED,
            source_dataset=source_dataset,
            confidence=confidence,
            attributes=attributes,
        )
        self.add_edge(edge)
        return edge
    
    def add_federal_subaward_edge(
        self,
        entity_id: str,
        award_id: str,
        source_dataset: str,
        confidence: str = "MEDIUM",
        prime_entity_id: str = None,
        **attributes
    ) -> Edge:
        """Add FEDERAL_SUBAWARDED edge between entity and award."""
        if prime_entity_id:
            attributes["prime_entity_id"] = prime_entity_id
        
        edge = Edge(
            source_id=f"entity:{entity_id}",
            target_id=f"award:{award_id}",
            edge_type=EdgeType.FEDERAL_SUBAWARDED,
            source_dataset=source_dataset,
            confidence=confidence,
            attributes=attributes,
        )
        self.add_edge(edge)
        return edge
    
    def get_node_count(self) -> int:
        """Get total number of nodes."""
        return len(self._nodes)
    
    def get_edge_count(self) -> int:
        """Get total number of edges."""
        return len(self._edges)
    
    def get_node_counts_by_type(self) -> dict[str, int]:
        """Get node counts by type."""
        counts = {}
        for node in self._nodes.values():
            type_name = node.node_type.value
            counts[type_name] = counts.get(type_name, 0) + 1
        return counts
    
    def get_edge_counts_by_type(self) -> dict[str, int]:
        """Get edge counts by type."""
        counts = {}
        for edge in self._edges:
            type_name = edge.edge_type.value
            counts[type_name] = counts.get(type_name, 0) + 1
        return counts
    
    def to_node_dataframe(self) -> pd.DataFrame:
        """Export nodes as a DataFrame."""
        if not self._nodes:
            return pd.DataFrame()
        
        records = [node.to_dict() for node in self._nodes.values()]
        return pd.DataFrame(records)
    
    def to_edge_dataframe(self) -> pd.DataFrame:
        """Export edges as a DataFrame."""
        if not self._edges:
            return pd.DataFrame()
        
        records = [edge.to_dict() for edge in self._edges]
        return pd.DataFrame(records)
    
    def export_to_csv(
        self,
        output_dir: Path | str,
        prefix: str = ""
    ) -> tuple[Path, Path]:
        """Export graph as CSV files.
        
        Args:
            output_dir: Directory to write CSV files
            prefix: Optional prefix for filenames
        
        Returns:
            Tuple of (nodes_path, edges_path)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self._build_timestamp = datetime.utcnow()
        
        # Export nodes
        nodes_df = self.to_node_dataframe()
        nodes_path = output_dir / f"{prefix}nodes.csv"
        nodes_df.to_csv(nodes_path, index=False)
        
        # Export edges
        edges_df = self.to_edge_dataframe()
        edges_path = output_dir / f"{prefix}edges.csv"
        edges_df.to_csv(edges_path, index=False)
        
        return nodes_path, edges_path
    
    def to_networkx(self):
        """Export as a NetworkX graph object.
        
        Returns:
            NetworkX DiGraph, or None if networkx not available
        """
        try:
            import networkx as nx
        except ImportError:
            return None
        
        G = nx.DiGraph()
        
        # Add nodes
        for node_id, node in self._nodes.items():
            G.add_node(
                node_id,
                node_type=node.node_type.value,
                label=node.label,
                source_dataset=node.source_dataset,
                **node.attributes
            )
        
        # Add edges
        for edge in self._edges:
            G.add_edge(
                edge.source_id,
                edge.target_id,
                edge_type=edge.edge_type.value,
                source_dataset=edge.source_dataset,
                confidence=edge.confidence,
                **edge.attributes
            )
        
        return G
    
    def get_metadata(self) -> dict[str, Any]:
        """Get graph metadata."""
        return {
            "build_timestamp": (
                self._build_timestamp.isoformat()
                if self._build_timestamp else None
            ),
            "total_nodes": self.get_node_count(),
            "total_edges": self.get_edge_count(),
            "nodes_by_type": self.get_node_counts_by_type(),
            "edges_by_type": self.get_edge_counts_by_type(),
        }


def build_graph_from_datasets(
    registry,  # DatasetRegistry from ingestion.loader
    builder: Optional[GraphBuilder] = None
) -> GraphBuilder:
    """Build a graph from loaded datasets.
    
    This function processes all loaded datasets and creates the graph.
    
    Args:
        registry: DatasetRegistry containing loaded datasets
        builder: Optional existing GraphBuilder to add to
    
    Returns:
        GraphBuilder with nodes and edges
    """
    from ..normalization.engine import (
        generate_entity_id,
        generate_address_id,
    )
    
    if builder is None:
        builder = GraphBuilder()
    
    # Process business licenses
    bl_df = registry.get_dataset("business_licenses")
    if bl_df is not None:
        for idx, row in bl_df.iterrows():
            name = row.get("_std_entity_name") or row.get("Business Name")
            address = row.get("_std_address") or row.get("Business Address")
            
            if name and pd.notna(name):
                entity_id = generate_entity_id(name)
                if entity_id:
                    builder.add_entity_node(
                        entity_id=entity_id,
                        name=str(name),
                        source_dataset="business_licenses",
                        source_row=idx,
                    )
                    
                    # Add address relationship
                    if address and pd.notna(address):
                        address_id = generate_address_id(address)
                        if address_id:
                            builder.add_address_node(
                                address_id=address_id,
                                address=str(address),
                                source_dataset="business_licenses",
                                source_row=idx,
                            )
                            builder.add_located_at_edge(
                                entity_id=entity_id,
                                address_id=address_id,
                                source_dataset="business_licenses",
                            )
    
    # Process city contracts
    cc_df = registry.get_dataset("city_contracts")
    if cc_df is not None:
        for idx, row in cc_df.iterrows():
            supplier = row.get("_std_supplier_name") or row.get("Supplier")
            contract_num = row.get("_std_contract_number") or row.get("Contract Number")
            
            if supplier and pd.notna(supplier):
                entity_id = generate_entity_id(supplier)
                if entity_id and contract_num:
                    builder.add_entity_node(
                        entity_id=entity_id,
                        name=str(supplier),
                        source_dataset="city_contracts",
                        source_row=idx,
                    )
                    
                    contract_id = str(contract_num)
                    description = row.get("Description", "")
                    builder.add_award_node(
                        award_id=contract_id,
                        description=str(description) if description else "",
                        source_dataset="city_contracts",
                        source_row=idx,
                        award_type="city_contract",
                    )
                    
                    builder.add_city_contract_edge(
                        entity_id=entity_id,
                        contract_id=contract_id,
                        source_dataset="city_contracts",
                    )
    
    # Process delinquent properties
    dp_df = registry.get_dataset("delinquent_properties")
    if dp_df is not None:
        for idx, row in dp_df.iterrows():
            owner1 = row.get("_std_owner_name_1") or row.get("Current Owner Name 1")
            address = row.get("_std_address") or row.get("Physical Address")
            prop_code = row.get("Property Code")
            
            if owner1 and pd.notna(owner1) and prop_code:
                entity_id = generate_entity_id(owner1)
                if entity_id:
                    builder.add_entity_node(
                        entity_id=entity_id,
                        name=str(owner1),
                        source_dataset="delinquent_properties",
                        source_row=idx,
                    )
                    
                    # Add property node
                    builder.add_property_node(
                        property_id=str(prop_code),
                        address=str(address) if address and pd.notna(address) else "",
                        source_dataset="delinquent_properties",
                        source_row=idx,
                        total_due=row.get("Total Due"),
                        years_delinquent=row.get("Total Years Delinquent"),
                    )
                    
                    builder.add_owns_property_edge(
                        entity_id=entity_id,
                        property_id=str(prop_code),
                        source_dataset="delinquent_properties",
                    )
    
    # Process federal prime awards (contracts)
    for ds_name in ["federal_contracts_prime", "federal_assistance_prime"]:
        df = registry.get_dataset(ds_name)
        if df is None:
            continue
            
        for idx, row in df.iterrows():
            recipient = row.get("_std_recipient_name") or row.get("prime_awardee_name") or row.get("recipient_name")
            award_id = row.get("_std_award_id") or row.get("prime_award_unique_key") or row.get("assistance_award_unique_key")
            uei = row.get("prime_awardee_uei") or row.get("recipient_uei")
            
            if recipient and pd.notna(recipient) and award_id:
                entity_id = generate_entity_id(recipient)
                # If UEI is available, use it as a more reliable ID
                if uei and pd.notna(uei):
                    entity_id = str(uei)
                
                if entity_id:
                    builder.add_entity_node(
                        entity_id=entity_id,
                        name=str(recipient),
                        source_dataset=ds_name,
                        source_row=idx,
                        uei=str(uei) if uei and pd.notna(uei) else None,
                    )
                    
                    description = row.get("_std_description") or row.get("prime_award_base_transaction_description") or ""
                    builder.add_award_node(
                        award_id=str(award_id),
                        description=str(description)[:200] if description else "",
                        source_dataset=ds_name,
                        source_row=idx,
                        award_type="federal_contract" if "contract" in ds_name else "federal_assistance",
                    )
                    
                    builder.add_federal_prime_edge(
                        entity_id=entity_id,
                        award_id=str(award_id),
                        source_dataset=ds_name,
                        confidence="HIGH" if uei else "MEDIUM",
                    )
                    
                    # Add address if available
                    address = row.get("prime_awardee_address_line_1") or row.get("recipient_address_line_1")
                    if address and pd.notna(address):
                        address_id = generate_address_id(address)
                        if address_id:
                            builder.add_address_node(
                                address_id=address_id,
                                address=str(address),
                                source_dataset=ds_name,
                                source_row=idx,
                            )
                            builder.add_located_at_edge(
                                entity_id=entity_id,
                                address_id=address_id,
                                source_dataset=ds_name,
                            )
    
    # Process subawards
    for ds_name in ["federal_contracts_subawards", "federal_assistance_subawards"]:
        df = registry.get_dataset(ds_name)
        if df is None:
            continue
            
        for idx, row in df.iterrows():
            subawardee = row.get("subawardee_name")
            prime_award_id = row.get("prime_award_unique_key")
            subaward_num = row.get("subaward_number")
            subawardee_uei = row.get("subawardee_uei")
            
            if subawardee and pd.notna(subawardee) and prime_award_id and subaward_num:
                entity_id = generate_entity_id(subawardee)
                if subawardee_uei and pd.notna(subawardee_uei):
                    entity_id = str(subawardee_uei)
                
                if entity_id:
                    builder.add_entity_node(
                        entity_id=entity_id,
                        name=str(subawardee),
                        source_dataset=ds_name,
                        source_row=idx,
                        uei=str(subawardee_uei) if subawardee_uei and pd.notna(subawardee_uei) else None,
                    )
                    
                    # Create subaward node (combine prime+sub for unique ID)
                    subaward_id = f"{prime_award_id}_{subaward_num}"
                    description = row.get("subaward_description") or ""
                    builder.add_award_node(
                        award_id=subaward_id,
                        description=str(description)[:200] if description else "",
                        source_dataset=ds_name,
                        source_row=idx,
                        award_type="federal_subaward",
                    )
                    
                    # Get prime recipient for reference
                    prime_recipient_uei = row.get("prime_awardee_uei")
                    
                    builder.add_federal_subaward_edge(
                        entity_id=entity_id,
                        award_id=subaward_id,
                        source_dataset=ds_name,
                        confidence="HIGH" if subawardee_uei else "MEDIUM",
                        prime_entity_id=str(prime_recipient_uei) if prime_recipient_uei and pd.notna(prime_recipient_uei) else None,
                    )
                    
                    # Add address if available
                    address = row.get("subawardee_address_line_1")
                    if address and pd.notna(address):
                        address_id = generate_address_id(address)
                        if address_id:
                            builder.add_address_node(
                                address_id=address_id,
                                address=str(address),
                                source_dataset=ds_name,
                                source_row=idx,
                            )
                            builder.add_located_at_edge(
                                entity_id=entity_id,
                                address_id=address_id,
                                source_dataset=ds_name,
                            )
    
    return builder


if __name__ == "__main__":
    # Example usage
    print("Graph Builder example:")
    
    builder = GraphBuilder()
    
    # Add sample nodes
    builder.add_entity_node(
        entity_id="abc123",
        name="ACME Corp",
        source_dataset="business_licenses",
        source_row=0,
    )
    
    builder.add_address_node(
        address_id="addr456",
        address="123 Main St",
        source_dataset="business_licenses",
        source_row=0,
    )
    
    builder.add_located_at_edge(
        entity_id="abc123",
        address_id="addr456",
        source_dataset="business_licenses",
    )
    
    print(f"\nNodes: {builder.get_node_count()}")
    print(f"Edges: {builder.get_edge_count()}")
    print(f"\nNode counts by type: {builder.get_node_counts_by_type()}")
    print(f"Edge counts by type: {builder.get_edge_counts_by_type()}")
