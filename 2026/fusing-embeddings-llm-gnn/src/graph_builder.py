"""
Graph Builder for Citation Network

Constructs citation graphs from the preprocessed JSON data containing papers and their citations.
Creates PyTorch Geometric Data objects with node features and edge indices.
"""

import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from torch_geometric.data import Data
from tqdm import tqdm


@dataclass
class GraphStats:
    """Statistics about the constructed citation graph."""
    num_nodes: int
    num_edges: int
    num_papers_with_features: int
    num_papers_with_citations: int
    avg_citations_per_paper: float


class CitationGraphBuilder:
    """
    Builds citation graphs from JSON data containing papers and their citations.

    The graph structure is:
    - Nodes: Papers (both source papers and citing papers)
    - Edges: Citation relationships (citing_paper -> cited_paper)
    - Node features: Will be populated with embeddings from semantic features
    """

    def __init__(self, json_path: str):
        """
        Initialize the graph builder.

        Args:
            json_path: Path to JSON file containing papers with citations
        """
        self.json_path = Path(json_path)
        self.papers_data = []
        self.paper_id_to_idx = {}  # Maps paper DOI to node index
        self.idx_to_paper_id = {}  # Maps node index to paper DOI
        self.paper_metadata = {}   # Stores paper details (title, abstract, etc.)

    def load_data(self) -> None:
        """Load papers data from JSON file."""
        print(f"Loading data from {self.json_path}...")
        with open(self.json_path, 'r', encoding='utf-8') as f:
            self.papers_data = json.load(f)
        print(f"Loaded {len(self.papers_data)} papers")

    def build_graph(self, min_citations: int = 0) -> Tuple[Data, GraphStats]:
        """
        Build citation graph from loaded data.

        Args:
            min_citations: Minimum number of citations required to include a paper

        Returns:
            Tuple of (PyTorch Geometric Data object, GraphStats)
        """
        if not self.papers_data:
            self.load_data()

        # First pass: collect all unique papers (nodes)
        all_paper_dois = set()

        for paper in self.papers_data:
            paper_details = paper.get('paper_details', {})
            paper_doi = paper_details.get('doi')

            if paper_doi:
                all_paper_dois.add(paper_doi)

                # Store metadata for source papers
                self.paper_metadata[paper_doi] = {
                    'paperId': paper_details.get('paperId'),
                    'title': paper_details.get('title'),
                    'abstract': paper_details.get('abstract'),
                    'authors': paper_details.get('authors', []),
                    'year': paper_details.get('year'),
                    'venue': paper_details.get('venue'),
                    'fieldsOfStudy': paper_details.get('fieldsOfStudy', []),
                    'citationCount': paper_details.get('citationCount', 0),
                    'is_source_paper': True
                }

            # Add citing papers
            citations = paper.get('citations', [])
            if len(citations) >= min_citations or min_citations == 0:
                for citing_paper in citations:
                    citing_doi = citing_paper.get('doi')
                    if citing_doi:
                        all_paper_dois.add(citing_doi)

                        # Store metadata for citing papers (if not already stored)
                        if citing_doi not in self.paper_metadata:
                            self.paper_metadata[citing_doi] = {
                                'paperId': citing_paper.get('paperId'),
                                'title': citing_paper.get('title'),
                                'abstract': citing_paper.get('abstract'),
                                'authors': citing_paper.get('authors', []),
                                'year': citing_paper.get('year'),
                                'venue': citing_paper.get('venue'),
                                'fieldsOfStudy': citing_paper.get('fieldsOfStudy', []),
                                'citationCount': citing_paper.get('citationCount', 0),
                                'is_source_paper': False
                            }

        # Create mapping from DOI to node index
        for idx, paper_doi in enumerate(sorted(all_paper_dois)):
            self.paper_id_to_idx[paper_doi] = idx
            self.idx_to_paper_id[idx] = paper_doi

        print(f"Total unique papers (nodes): {len(all_paper_dois)}")

        # Second pass: build edges
        edge_list = []
        papers_with_citations = 0
        total_citations = 0

        for paper in self.papers_data:
            paper_details = paper.get('paper_details', {})
            paper_doi = paper_details.get('doi')

            if not paper_doi:
                continue

            citations = paper.get('citations', [])

            if len(citations) > 0:
                papers_with_citations += 1
                total_citations += len(citations)

            if len(citations) >= min_citations or min_citations == 0:
                target_idx = self.paper_id_to_idx[paper_doi]

                for citing_paper in citations:
                    citing_doi = citing_paper.get('doi')
                    if citing_doi:
                        source_idx = self.paper_id_to_idx[citing_doi]
                        # Edge: citing_paper -> cited_paper
                        edge_list.append([source_idx, target_idx])

        print(f"Total edges (citations): {len(edge_list)}")

        # Create edge_index tensor
        if edge_list:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)

        # Initialize placeholder node features (will be populated with embeddings later)
        num_nodes = len(all_paper_dois)
        x = torch.zeros((num_nodes, 1))  # Placeholder features

        # Create PyTorch Geometric Data object
        data = Data(x=x, edge_index=edge_index)

        # Calculate statistics
        avg_citations = total_citations / max(papers_with_citations, 1)
        stats = GraphStats(
            num_nodes=num_nodes,
            num_edges=len(edge_list),
            num_papers_with_features=len(self.paper_metadata),
            num_papers_with_citations=papers_with_citations,
            avg_citations_per_paper=avg_citations
        )

        return data, stats

    def get_paper_texts(self) -> List[Dict[str, str]]:
        """
        Get list of papers with their text content for embedding generation.

        Returns:
            List of dicts with keys: 'id' (DOI), 'title', 'abstract'
        """
        papers = []
        for doi, metadata in self.paper_metadata.items():
            papers.append({
                'id': doi,
                'title': metadata.get('title', ''),
                'abstract': metadata.get('abstract', '')
            })
        return papers

    def get_node_index(self, paper_doi: str) -> Optional[int]:
        """Get node index for a paper DOI."""
        return self.paper_id_to_idx.get(paper_doi)

    def get_paper_doi(self, node_idx: int) -> Optional[str]:
        """Get paper DOI for a node index."""
        return self.idx_to_paper_id.get(node_idx)

    def get_paper_metadata(self, paper_doi: str) -> Optional[Dict]:
        """Get metadata for a paper."""
        return self.paper_metadata.get(paper_doi)

    def save_mappings(self, output_dir: str) -> None:
        """
        Save paper ID to node index mappings and metadata.

        Args:
            output_dir: Directory to save mappings
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save mappings
        with open(output_path / 'paper_id_to_idx.json', 'w') as f:
            json.dump(self.paper_id_to_idx, f, indent=2)

        with open(output_path / 'idx_to_paper_id.json', 'w') as f:
            json.dump(self.idx_to_paper_id, f, indent=2)

        # Save metadata
        with open(output_path / 'paper_metadata.json', 'w') as f:
            json.dump(self.paper_metadata, f, indent=2)

        print(f"Saved mappings and metadata to {output_path}")

    def load_mappings(self, input_dir: str) -> None:
        """
        Load paper ID to node index mappings and metadata.

        Args:
            input_dir: Directory containing saved mappings
        """
        input_path = Path(input_dir)

        with open(input_path / 'paper_id_to_idx.json', 'r') as f:
            self.paper_id_to_idx = json.load(f)

        with open(input_path / 'idx_to_paper_id.json', 'r') as f:
            # Convert string keys back to integers
            idx_to_doi = json.load(f)
            self.idx_to_paper_id = {int(k): v for k, v in idx_to_doi.items()}

        with open(input_path / 'paper_metadata.json', 'r') as f:
            self.paper_metadata = json.load(f)

        print(f"Loaded mappings and metadata from {input_path}")


def create_citation_graph(json_path: str, min_citations: int = 0) -> Tuple[CitationGraphBuilder, Data, GraphStats]:
    """
    Convenience function to create a citation graph.

    Args:
        json_path: Path to JSON file with papers and citations
        min_citations: Minimum citations to include a paper

    Returns:
        Tuple of (CitationGraphBuilder, Data, GraphStats)
    """
    builder = CitationGraphBuilder(json_path)
    data, stats = builder.build_graph(min_citations=min_citations)
    return builder, data, stats


if __name__ == "__main__":
    # Test the graph builder
    sample_json = "../data/preprocessed_data/papers_with_citations_sample.json"

    builder, graph_data, stats = create_citation_graph(sample_json)

    print("\n=== Graph Statistics ===")
    print(f"Nodes: {stats.num_nodes}")
    print(f"Edges: {stats.num_edges}")
    print(f"Papers with features: {stats.num_papers_with_features}")
    print(f"Papers with citations: {stats.num_papers_with_citations}")
    print(f"Average citations per paper: {stats.avg_citations_per_paper:.2f}")

    # Save mappings
    builder.save_mappings("../data/graph_mappings")

    print("\nGraph built successfully!")
