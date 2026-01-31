"""
Paper Recommendation System using GNN and Semantic Embeddings

Combines:
- Citation graph structure (GNN embeddings)
- Semantic content (LLM embeddings)
- Simple ranking based on similarity
"""

import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import json

from models.llm import LLM
from models.semantic_features import SemanticFeatureExtractor
from models.gnn import CitationGNN, GNNTrainer, create_gnn_model
from graph_builder import CitationGraphBuilder


@dataclass
class RecommendationResult:
    """Result of a paper recommendation query."""
    paper_doi: str
    paper_title: str
    score: float
    rank: int
    metadata: Dict


class PaperRecommendationSystem:
    """
    Paper recommendation system combining GNN and semantic embeddings.

    Workflow:
    1. Build citation graph from data
    2. Extract semantic features using LLM
    3. Train GNN on citation graph with semantic features
    4. For queries: compute query embedding and rank papers by similarity
    """

    def __init__(
        self,
        json_data_path: str,
        llm_config_path: Optional[str] = None,
        device: str = 'cuda:0'
    ):
        """
        Initialize the recommendation system.

        Args:
            json_data_path: Path to JSON file with papers and citations
            llm_config_path: Path to LLM config (optional)
            device: Device to use ('cpu' or 'cuda')
        """
        self.json_data_path = Path(json_data_path)
        self.device = torch.device(device)

        # Initialize components
        self.graph_builder = CitationGraphBuilder(str(self.json_data_path))
        self.semantic_extractor = SemanticFeatureExtractor(llm_config_path=llm_config_path)
        self.llm = LLM(config_path=llm_config_path)

        # Will be initialized later
        self.graph_data = None
        self.graph_stats = None
        self.semantic_features = None
        self.gnn_model = None
        self.gnn_embeddings = None
        self.combined_embeddings = None

        print(f"Initialized recommendation system with data from {self.json_data_path}")

    def build_graph(self, min_citations: int = 0) -> None:
        """
        Build citation graph from data.

        Args:
            min_citations: Minimum citations required to include a paper
        """
        print("\n=== Building Citation Graph ===")
        self.graph_data, self.graph_stats = self.graph_builder.build_graph(min_citations=min_citations)

        print(f"Graph built with {self.graph_stats.num_nodes} nodes and {self.graph_stats.num_edges} edges")
        print(f"Average citations per paper: {self.graph_stats.avg_citations_per_paper:.2f}")

    def extract_semantic_features(self, cache_dir: Optional[str] = None) -> None:
        """
        Extract semantic features for all papers using LLM embeddings.

        Args:
            cache_dir: Directory to cache embeddings (optional)
        """
        print("\n=== Extracting Semantic Features ===")

        if cache_dir:
            self.semantic_extractor.cache_dir = Path(cache_dir)
            self.semantic_extractor.cache_dir.mkdir(parents=True, exist_ok=True)

        # Get papers for embedding
        papers = self.graph_builder.get_paper_texts()
        print(f"Extracting embeddings for {len(papers)} papers...")

        # Extract features
        self.semantic_features = self.semantic_extractor.extract_batch(
            papers,
            show_progress=True
        )

        print(f"Extracted semantic features for {len(self.semantic_features)} papers")

        # Create feature matrix aligned with graph nodes
        self._align_features_with_graph()

    def _align_features_with_graph(self) -> None:
        """Align semantic features with graph node ordering."""
        print("Aligning features with graph nodes...")

        num_nodes = self.graph_stats.num_nodes
        embedding_dim = self.semantic_features[0].get_embedding_dim()

        # Create feature dict for fast lookup
        feature_dict = {f.paper_id: f.combined_embedding for f in self.semantic_features}

        # Align with graph node order
        node_features = []
        for i in range(num_nodes):
            paper_doi = self.graph_builder.get_paper_doi(i)
            if paper_doi in feature_dict:
                node_features.append(feature_dict[paper_doi])
            else:
                # Use zero vector for papers without features
                node_features.append([0.0] * embedding_dim)

        # Convert to tensor
        self.graph_data.x = torch.tensor(node_features, dtype=torch.float32)
        print(f"Aligned features: {self.graph_data.x.shape}")

    def train_gnn(
        self,
        hidden_dim: int = 256,
        output_dim: int = 128,
        num_layers: int = 3,
        gnn_type: str = 'gcn',
        num_epochs: int = 100,
        learning_rate: float = 0.001,
        dropout: float = 0.3
    ) -> List[float]:
        """
        Train GNN model on citation graph.

        Args:
            hidden_dim: Hidden dimension
            output_dim: Output embedding dimension
            num_layers: Number of GNN layers
            gnn_type: Type of GNN ('gcn', 'gat', 'sage')
            num_epochs: Number of training epochs
            learning_rate: Learning rate
            dropout: Dropout rate

        Returns:
            List of training losses
        """
        print("\n=== Training GNN Model ===")

        input_dim = self.graph_data.x.shape[1]
        print(f"Input dimension: {input_dim}")

        # Create model
        self.gnn_model = create_gnn_model(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            gnn_type=gnn_type,
            dropout=dropout
        )

        print(f"Created {gnn_type.upper()} model with {num_layers} layers")
        print(f"Architecture: {input_dim} -> {hidden_dim} -> {output_dim}")

        # Train
        trainer = GNNTrainer(
            model=self.gnn_model,
            learning_rate=learning_rate,
            device=str(self.device)
        )

        losses = trainer.train_unsupervised(
            data=self.graph_data,
            num_epochs=num_epochs,
            verbose=True
        )

        print(f"Training completed. Final loss: {losses[-1]:.4f}")

        # Generate embeddings
        self._generate_gnn_embeddings()

        return losses

    def _generate_gnn_embeddings(self) -> None:
        """Generate GNN embeddings for all papers."""
        print("Generating GNN embeddings...")

        self.gnn_model.eval()
        with torch.no_grad():
            # Move data to device temporarily
            x_device = self.graph_data.x.to(self.device)
            edge_index_device = self.graph_data.edge_index.to(self.device)

            self.gnn_embeddings = self.gnn_model(x_device, edge_index_device)
            self.gnn_embeddings = self.gnn_embeddings.cpu().numpy()

        print(f"Generated embeddings: {self.gnn_embeddings.shape}")

        # Combine with semantic embeddings (simple average for now)
        self._combine_embeddings()

    def _combine_embeddings(self, gnn_weight: float = 0.5, fusion_strategy: str = 'two_tower') -> None:
        """
        Combine GNN embeddings with semantic embeddings.

        Args:
            gnn_weight: Weight for GNN embeddings (1 - gnn_weight for semantic)
            fusion_strategy: Strategy for combining embeddings
                - 'two_tower': Keep separate, combine similarities at query time (default)
                - 'concat': Concatenate after projecting to common dimension
                - 'weighted': Weighted average (only if dimensions match)
        """
        print(f"Combining embeddings (strategy: {fusion_strategy}, GNN weight: {gnn_weight})...")

        # Normalize both embeddings
        gnn_norm = self.gnn_embeddings / (np.linalg.norm(self.gnn_embeddings, axis=1, keepdims=True) + 1e-8)

        # Ensure graph_data.x is on CPU before converting to numpy
        if self.graph_data.x.is_cuda:
            semantic_emb = self.graph_data.x.cpu().numpy()
        else:
            semantic_emb = self.graph_data.x.numpy()
        semantic_norm = semantic_emb / (np.linalg.norm(semantic_emb, axis=1, keepdims=True) + 1e-8)

        # Store normalized embeddings separately for two-tower approach
        self.gnn_embeddings_norm = gnn_norm
        self.semantic_embeddings_norm = semantic_norm
        self.fusion_strategy = fusion_strategy
        self.gnn_weight = gnn_weight
        self.query_projection = None  # Will store projection matrix for consistent query processing

        # Check if dimensions match
        gnn_dim = gnn_norm.shape[1]
        semantic_dim = semantic_norm.shape[1]
        print(f"GNN embedding dim: {gnn_dim}, Semantic embedding dim: {semantic_dim}")

        if fusion_strategy == 'two_tower':
            # Store separately, will combine similarities at query time
            print(f"Using two-tower architecture: GNN ({gnn_dim}D) + Semantic ({semantic_dim}D)")
            print(f"Similarities will be combined with weights: GNN={gnn_weight:.2f}, Semantic={1-gnn_weight:.2f}")
            self.combined_embeddings = None  # Not used in two-tower
        elif fusion_strategy == 'concat':
            if gnn_dim != semantic_dim:
                print(f"Projecting to common dimension for concatenation...")
                # Project to match dimensions using fixed random projection
                np.random.seed(42)  # Fixed seed for reproducibility
                target_dim = min(gnn_dim, semantic_dim)
                if gnn_dim > semantic_dim:
                    # Downsample GNN
                    projection = np.random.randn(gnn_dim, semantic_dim) / np.sqrt(gnn_dim)
                    gnn_projected = gnn_norm @ projection
                    gnn_projected = gnn_projected / (np.linalg.norm(gnn_projected, axis=1, keepdims=True) + 1e-8)
                    self.combined_embeddings = np.concatenate([gnn_projected, semantic_norm], axis=1)
                    # Query projection: project from semantic_dim to semantic_dim (identity - no projection needed)
                    self.query_projection = None
                    self.query_target_dim = semantic_dim
                else:
                    # Downsample semantic
                    projection = np.random.randn(semantic_dim, gnn_dim) / np.sqrt(semantic_dim)
                    semantic_projected = semantic_norm @ projection
                    semantic_projected = semantic_projected / (np.linalg.norm(semantic_projected, axis=1, keepdims=True) + 1e-8)
                    self.combined_embeddings = np.concatenate([gnn_norm, semantic_projected], axis=1)
                    # Store projection for query processing
                    self.query_projection = projection
                    self.query_target_dim = gnn_dim

                self.combined_embeddings = self.combined_embeddings / (
                    np.linalg.norm(self.combined_embeddings, axis=1, keepdims=True) + 1e-8
                )
            else:
                # Dimensions match, just concatenate
                self.combined_embeddings = np.concatenate([gnn_norm, semantic_norm], axis=1)
                self.combined_embeddings = self.combined_embeddings / (
                    np.linalg.norm(self.combined_embeddings, axis=1, keepdims=True) + 1e-8
                )
                # No projection needed since dimensions already match
                self.query_projection = None
                self.query_target_dim = gnn_dim  # or semantic_dim, they're equal
        elif fusion_strategy == 'weighted':
            if gnn_dim != semantic_dim:
                print(f"Error: Weighted fusion requires matching dimensions. Using two-tower instead.")
                self.fusion_strategy = 'two_tower'
                self.combined_embeddings = None
            else:
                # Weighted combination
                self.combined_embeddings = gnn_weight * gnn_norm + (1 - gnn_weight) * semantic_norm
                self.combined_embeddings = self.combined_embeddings / (
                    np.linalg.norm(self.combined_embeddings, axis=1, keepdims=True) + 1e-8
                )
        else:
            raise ValueError(f"Unknown fusion strategy: {fusion_strategy}")

        if self.combined_embeddings is not None:
            print(f"Combined embeddings: {self.combined_embeddings.shape}")
        else:
            print(f"Using separate embeddings for two-tower architecture")

    def search(
        self,
        query: str,
        top_k: int = 10,
        use_combined: bool = True
    ) -> List[RecommendationResult]:
        """
        Search for papers based on query.

        Args:
            query: Query string (keywords or sentence)
            top_k: Number of top results to return
            use_combined: Use combined embeddings (GNN + semantic) vs semantic only

        Returns:
            List of RecommendationResult objects
        """
        print(f"\n=== Searching for: '{query}' ===")

        # Get query embedding (semantic only)
        query_embedding = self.llm.get_embedding(query)
        query_embedding = np.array(query_embedding)
        query_embedding = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)

        # Compute similarities based on fusion strategy
        if use_combined and hasattr(self, 'fusion_strategy'):
            if self.fusion_strategy == 'two_tower':
                # Two-tower: compute semantic and graph-based similarities separately
                print(f"Using two-tower architecture (GNN weight: {self.gnn_weight:.2f})")

                # 1. Semantic similarity
                semantic_similarities = self.semantic_embeddings_norm @ query_embedding

                # 2. Graph-based similarity: use semantic query to find nearest neighbors in graph space
                # First, find papers similar to query in semantic space
                semantic_scores = self.semantic_embeddings_norm @ query_embedding
                top_semantic_indices = np.argsort(semantic_scores)[::-1][:50]  # Top 50 semantically similar

                # Use their average GNN embedding as "pseudo-graph query"
                pseudo_graph_query = np.mean(self.gnn_embeddings_norm[top_semantic_indices], axis=0)
                pseudo_graph_query = pseudo_graph_query / (np.linalg.norm(pseudo_graph_query) + 1e-8)

                # Compute graph-based similarities
                graph_similarities = self.gnn_embeddings_norm @ pseudo_graph_query

                # 3. Combine similarities with weights
                similarities = (1 - self.gnn_weight) * semantic_similarities + self.gnn_weight * graph_similarities
                print(f"Combined semantic and graph similarities")

            elif self.fusion_strategy == 'concat' and self.combined_embeddings is not None:
                # Concatenation strategy
                paper_embeddings = self.combined_embeddings
                print(f"Using concatenated embeddings ({paper_embeddings.shape[1]}D)")

                # For concat strategy, process query embedding to match paper embeddings structure
                paper_dim = paper_embeddings.shape[1]
                query_dim = query_embedding.shape[0]
                print(f"Query dimension: {query_dim}, Paper embedding dimension: {paper_dim}")

                if query_dim * 2 == paper_dim:
                    # Simple case: dimensions already match, just duplicate query
                    query_embedding = np.concatenate([query_embedding, query_embedding])
                    query_embedding = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
                elif hasattr(self, 'query_projection') and self.query_projection is not None:
                    # Use stored projection matrix for consistent results
                    query_projected = query_embedding @ self.query_projection
                    query_projected = query_projected / (np.linalg.norm(query_projected) + 1e-8)
                    # Duplicate for both parts (semantic + graph)
                    query_embedding = np.concatenate([query_projected, query_projected])
                    query_embedding = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
                elif hasattr(self, 'query_target_dim'):
                    # No projection needed, just duplicate to target dimension
                    query_embedding = np.concatenate([query_embedding, query_embedding])
                    query_embedding = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
                else:
                    # Fallback: simple dimension matching (should not happen with proper initialization)
                    target_dim = paper_dim // 2
                    if query_dim > target_dim:
                        query_projected = query_embedding[:target_dim]
                        query_projected = query_projected / (np.linalg.norm(query_projected) + 1e-8)
                    else:
                        query_projected = np.zeros(target_dim)
                        query_projected[:query_dim] = query_embedding
                        query_projected = query_projected / (np.linalg.norm(query_projected) + 1e-8)
                    query_embedding = np.concatenate([query_projected, query_projected])
                    query_embedding = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
                similarities = paper_embeddings @ query_embedding

            else:
                # Weighted or other strategy with combined embeddings
                if self.combined_embeddings is not None:
                    paper_embeddings = self.combined_embeddings
                    print(f"Using combined embeddings ({paper_embeddings.shape[1]}D)")

                    # If dimensions don't match, fall back to semantic only
                    if query_embedding.shape[0] != paper_embeddings.shape[1]:
                        print(f"Warning: Dimension mismatch, falling back to semantic only")
                        paper_embeddings = self.semantic_embeddings_norm

                    similarities = paper_embeddings @ query_embedding
                else:
                    # Fall back to semantic only
                    similarities = self.semantic_embeddings_norm @ query_embedding
                    print("Using semantic embeddings only")
        else:
            # Semantic only
            if hasattr(self, 'semantic_embeddings_norm'):
                similarities = self.semantic_embeddings_norm @ query_embedding
            else:
                # Fallback to graph_data.x
                if self.graph_data.x.is_cuda:
                    paper_embeddings = self.graph_data.x.cpu().numpy()
                else:
                    paper_embeddings = self.graph_data.x.numpy()
                paper_embeddings = paper_embeddings / (np.linalg.norm(paper_embeddings, axis=1, keepdims=True) + 1e-8)
                similarities = paper_embeddings @ query_embedding
            print("Using semantic embeddings only")

        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]

        # Build results
        results = []
        for rank, idx in enumerate(top_indices):
            paper_doi = self.graph_builder.get_paper_doi(idx)
            metadata = self.graph_builder.get_paper_metadata(paper_doi)

            result = RecommendationResult(
                paper_doi=paper_doi,
                paper_title=metadata.get('title', 'Unknown'),
                score=float(similarities[idx]),
                rank=rank + 1,
                metadata=metadata
            )
            results.append(result)

        return results

    def print_results(self, results: List[RecommendationResult], max_results: int = 10) -> None:
        """
        Print recommendation results.

        Args:
            results: List of recommendation results
            max_results: Maximum number of results to print
        """
        print(f"\n{'='*80}")
        print(f"Top {min(len(results), max_results)} Recommended Papers")
        print(f"{'='*80}\n")

        for result in results[:max_results]:
            print(f"Rank {result.rank}: Score = {result.score:.4f}")
            print(f"  Title: {result.paper_title}")
            print(f"  DOI: {result.paper_doi}")

            authors = result.metadata.get('authors', [])
            if authors:
                author_names = [a.get('name', 'Unknown') for a in authors[:3]]
                print(f"  Authors: {', '.join(author_names)}")

            year = result.metadata.get('year')
            venue = result.metadata.get('venue')
            if year:
                print(f"  Year: {year}")
            if venue:
                print(f"  Venue: {venue}")

            print()

    def save_model(self, output_dir: str) -> None:
        """
        Save trained model and embeddings.

        Args:
            output_dir: Directory to save model
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save GNN model
        if self.gnn_model:
            torch.save(self.gnn_model.state_dict(), output_path / 'gnn_model.pt')
            print(f"Saved GNN model to {output_path / 'gnn_model.pt'}")

        # Save embeddings
        if self.gnn_embeddings is not None:
            np.save(output_path / 'gnn_embeddings.npy', self.gnn_embeddings)
            print(f"Saved GNN embeddings to {output_path / 'gnn_embeddings.npy'}")

        if self.combined_embeddings is not None:
            np.save(output_path / 'combined_embeddings.npy', self.combined_embeddings)
            print(f"Saved combined embeddings to {output_path / 'combined_embeddings.npy'}")

        # Save graph mappings
        self.graph_builder.save_mappings(str(output_path / 'graph_mappings'))

        # Save config
        config = {
            'num_nodes': self.graph_stats.num_nodes,
            'num_edges': self.graph_stats.num_edges,
            'embedding_dim': self.combined_embeddings.shape[1] if self.combined_embeddings is not None else None,
        }
        with open(output_path / 'model_config.json', 'w') as f:
            json.dump(config, f, indent=2)

        print(f"Model saved to {output_path}")

    def load_model(self, input_dir: str, model_params: Dict) -> None:
        """
        Load trained model and embeddings.

        Args:
            input_dir: Directory containing saved model
            model_params: Parameters for model initialization
        """
        input_path = Path(input_dir)

        # Load graph mappings
        self.graph_builder.load_mappings(str(input_path / 'graph_mappings'))

        # Load embeddings
        self.gnn_embeddings = np.load(input_path / 'gnn_embeddings.npy')
        self.combined_embeddings = np.load(input_path / 'combined_embeddings.npy')

        # Create and load GNN model
        self.gnn_model = create_gnn_model(**model_params)
        self.gnn_model.load_state_dict(torch.load(input_path / 'gnn_model.pt'))
        self.gnn_model.to(self.device)

        print(f"Model loaded from {input_path}")


if __name__ == "__main__":
    # Example usage
    system = PaperRecommendationSystem(
        json_data_path="../data/preprocessed_data/papers_with_citations.json",
        device='cuda:0'
    )

    # Build pipeline
    system.build_graph()
    system.extract_semantic_features()
    system.train_gnn(num_epochs=50)

    # Search
    results = system.search("deep learning graph neural networks", top_k=5)
    system.print_results(results)

    # Save model
    system.save_model("../models/saved_models")