"""
Visualization Module for Citation Graph and Results

Creates visualizations for:
- Citation network graph
- Paper rankings and scores
- GNN vs Semantic comparison
- Embedding space visualization
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
from tqdm import tqdm
import time
from datetime import datetime

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


class CitationGraphVisualizer:
    """
    Visualizer for citation graphs and recommendation results.
    """

    def __init__(self, graph_builder=None, results_dir: Optional[str] = None):
        """
        Initialize visualizer.

        Args:
            graph_builder: CitationGraphBuilder instance (optional)
            results_dir: Directory containing results (optional)
        """
        self.graph_builder = graph_builder
        self.results_dir = Path(results_dir) if results_dir else None
        self.G = None  # NetworkX graph

    def build_networkx_graph(self) -> nx.DiGraph:
        """
        Build NetworkX directed graph from citation data.

        Returns:
            NetworkX DiGraph
        """
        print("Building NetworkX graph from citation data...")

        self.G = nx.DiGraph()

        # Add nodes
        for doi, metadata in self.graph_builder.paper_metadata.items():
            self.G.add_node(
                doi,
                title=metadata.get('title', 'Unknown')[:50],  # Truncate long titles
                year=metadata.get('year'),
                citations=metadata.get('citationCount', 0),
                is_source=metadata.get('is_source_paper', False)
            )

        # Add edges from citation relationships
        for paper_id, paper_data in self.graph_builder.paper_metadata.items():
            # Get node index
            node_idx = self.graph_builder.get_node_index(paper_id)
            if node_idx is None:
                continue

            # Find papers that cite this paper (by looking at edge_index)
            # This would require access to the graph data
            # For now, we'll use a simpler approach based on metadata

        # Alternative: build from original JSON data
        if hasattr(self.graph_builder, 'papers_data') and self.graph_builder.papers_data:
            for paper in self.graph_builder.papers_data:
                paper_details = paper.get('paper_details', {})
                paper_doi = paper_details.get('doi')

                if paper_doi:
                    citations = paper.get('citations', [])
                    for citing_paper in citations:
                        citing_doi = citing_paper.get('doi')
                        if citing_doi and citing_doi in self.G.nodes and paper_doi in self.G.nodes:
                            self.G.add_edge(citing_doi, paper_doi)

        print(f"Graph built: {self.G.number_of_nodes()} nodes, {self.G.number_of_edges()} edges")
        return self.G

    def visualize_citation_network(
        self,
        output_path: str,
        layout: str = 'spring',
        highlight_papers: Optional[List[str]] = None,
        max_nodes: int = 7000
    ):
        """
        Visualize the citation network.

        Args:
            output_path: Path to save visualization
            layout: Layout algorithm ('spring', 'circular', 'kamada_kawai')
            highlight_papers: List of paper DOIs to highlight
            max_nodes: Maximum number of nodes to show
        """
        if self.G is None:
            self.build_networkx_graph()

        print(f"Visualizing citation network (showing up to {max_nodes} nodes)...")

        # Limit to max_nodes if graph is too large
        if self.G.number_of_nodes() > max_nodes:
            # Get nodes with most connections
            degree_dict = dict(self.G.degree())
            top_nodes = sorted(degree_dict.items(), key=lambda x: x[1], reverse=True)[:max_nodes]
            nodes_to_keep = [node for node, _ in top_nodes]
            G_viz = self.G.subgraph(nodes_to_keep).copy()
        else:
            G_viz = self.G

        # Choose layout
        if layout == 'spring':
            pos = nx.spring_layout(G_viz, k=1, iterations=50, seed=42)
        elif layout == 'circular':
            pos = nx.circular_layout(G_viz)
        elif layout == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(G_viz)
        else:
            pos = nx.spring_layout(G_viz, seed=42)

        # Create figure
        fig, ax = plt.subplots(figsize=(16, 12))

        # Node colors based on properties
        node_colors = []
        node_sizes = []
        for node in G_viz.nodes():
            if highlight_papers and node in highlight_papers:
                node_colors.append('red')
                node_sizes.append(10)
            elif G_viz.nodes[node].get('is_source', False):
                node_colors.append('lightblue')
                node_sizes.append(10)
            else:
                node_colors.append('lightgray')
                node_sizes.append(10)

        # Draw graph
        nx.draw_networkx_edges(
            G_viz, pos,
            edge_color='gray',
            alpha=0.3,
            arrows=True,
            arrowsize=10,
            width=0.5,
            ax=ax
        )

        nx.draw_networkx_nodes(
            G_viz, pos,
            node_color=node_colors,
            node_size=node_sizes,
            alpha=0.7,
            ax=ax
        )

        # Add labels for highlighted papers
        if highlight_papers:
            labels = {node: G_viz.nodes[node].get('title', node[:20])
                     for node in G_viz.nodes() if node in highlight_papers}
            nx.draw_networkx_labels(
                G_viz, pos,
                labels,
                font_size=8,
                font_color='black',
                ax=ax
            )

        ax.set_title(f'Citation Network Graph\n({G_viz.number_of_nodes()} papers, {G_viz.number_of_edges()} citations)',
                    fontsize=16, fontweight='bold')
        ax.axis('off')

        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', label='Highlighted Papers'),
            Patch(facecolor='lightblue', label='Source Papers'),
            Patch(facecolor='lightgray', label='Citing Papers')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved citation network to {output_path}")
        plt.close()

    def visualize_network_stats(self, output_path: str):
        """
        Visualize network statistics (degree distribution, centrality, etc.).

        Args:
            output_path: Path to save visualization
        """
        if self.G is None:
            self.build_networkx_graph()

        print("Visualizing network statistics...")

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. Degree distribution
        degrees = [d for n, d in self.G.degree()]
        in_degrees = [d for n, d in self.G.in_degree()]
        out_degrees = [d for n, d in self.G.out_degree()]

        ax = axes[0, 0]
        ax.hist(degrees, bins=30, alpha=0.7, label='Total Degree', color='blue')
        ax.hist(in_degrees, bins=30, alpha=0.7, label='In-Degree (Citations)', color='green')
        ax.hist(out_degrees, bins=30, alpha=0.7, label='Out-Degree (References)', color='red')
        ax.set_xlabel('Degree')
        ax.set_ylabel('Frequency')
        ax.set_title('Degree Distribution')
        ax.legend()
        ax.set_yscale('log')

        # 2. Top papers by citations (in-degree)
        ax = axes[0, 1]
        in_degree_dict = dict(self.G.in_degree())
        top_cited = sorted(in_degree_dict.items(), key=lambda x: x[1], reverse=True)[:10]
        papers = [self.G.nodes[doi].get('title', doi[:20]) for doi, _ in top_cited]
        citations = [count for _, count in top_cited]

        ax.barh(range(len(papers)), citations, color='steelblue')
        ax.set_yticks(range(len(papers)))
        ax.set_yticklabels(papers, fontsize=8)
        ax.set_xlabel('Number of Citations')
        ax.set_title('Top 10 Most Cited Papers')
        ax.invert_yaxis()

        # 3. Centrality measures
        ax = axes[1, 0]
        try:
            pagerank = nx.pagerank(self.G)
            top_pagerank = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:10]
            papers = [self.G.nodes[doi].get('title', doi[:20]) for doi, _ in top_pagerank]
            scores = [score for _, score in top_pagerank]

            ax.barh(range(len(papers)), scores, color='coral')
            ax.set_yticks(range(len(papers)))
            ax.set_yticklabels(papers, fontsize=8)
            ax.set_xlabel('PageRank Score')
            ax.set_title('Top 10 Papers by PageRank')
            ax.invert_yaxis()
        except:
            ax.text(0.5, 0.5, 'PageRank computation failed', ha='center', va='center')
            ax.set_title('PageRank')

        # 4. Network properties
        ax = axes[1, 1]
        ax.axis('off')

        stats_text = f"""
Network Statistics:

Nodes: {self.G.number_of_nodes()}
Edges: {self.G.number_of_edges()}

Avg Degree: {np.mean(degrees):.2f}
Avg In-Degree: {np.mean(in_degrees):.2f}
Avg Out-Degree: {np.mean(out_degrees):.2f}

Max In-Degree: {max(in_degrees) if in_degrees else 0}
Max Out-Degree: {max(out_degrees) if out_degrees else 0}

Density: {nx.density(self.G):.4f}
"""
        ax.text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center',
               fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.suptitle('Citation Network Statistics', fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved network statistics to {output_path}")
        plt.close()

    def visualize_query_results(
        self,
        query: str,
        results: List[Dict],
        output_path: str,
        top_n: int = 10
    ):
        """
        Visualize results for a single query.

        Args:
            query: Query string
            results: List of result dictionaries
            output_path: Path to save visualization
            top_n: Number of top results to show
        """
        print(f"Visualizing results for query: '{query}'...")

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # 1. Score bar chart
        ax = axes[0]
        top_results = results[:top_n]
        titles = [r.get('title', 'Unknown')[:40] + '...' if len(r.get('title', '')) > 40
                 else r.get('title', 'Unknown') for r in top_results]
        scores = [r.get('score', 0) for r in top_results]

        y_pos = np.arange(len(titles))
        bars = ax.barh(y_pos, scores, color='steelblue', alpha=0.8)

        # Color code by score
        for i, (bar, score) in enumerate(zip(bars, scores)):
            if score > 0.7:
                bar.set_color('darkgreen')
            elif score > 0.5:
                bar.set_color('steelblue')
            else:
                bar.set_color('orange')

        ax.set_yticks(y_pos)
        ax.set_yticklabels(titles, fontsize=9)
        ax.set_xlabel('Similarity Score', fontsize=11)
        ax.set_title(f'Top {top_n} Results\nQuery: "{query}"', fontsize=12, fontweight='bold')
        ax.invert_yaxis()
        ax.set_xlim(0, 1.0)
        ax.grid(axis='x', alpha=0.3)

        # 2. Score distribution
        ax = axes[1]
        all_scores = [r.get('score', 0) for r in results]
        ax.hist(all_scores, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
        ax.axvline(np.mean(all_scores), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(all_scores):.3f}')
        ax.axvline(np.median(all_scores), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(all_scores):.3f}')
        ax.set_xlabel('Similarity Score', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title('Score Distribution (All Results)', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved query results to {output_path}")
        plt.close()

    def visualize_gnn_vs_semantic_comparison(
        self,
        comparison_data: List[Dict],
        output_path: str
    ):
        """
        Visualize GNN vs Semantic comparison.

        Args:
            comparison_data: List of comparison dictionaries
            output_path: Path to save visualization
        """
        print("Visualizing GNN vs Semantic comparison...")

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Extract data
        queries = [d['query'][:30] + '...' if len(d['query']) > 30 else d['query'] for d in comparison_data]
        gnn_scores = [d['gnn_semantic_avg_score'] for d in comparison_data]
        semantic_scores = [d['semantic_only_avg_score'] for d in comparison_data]
        improvements = [d['improvement'] for d in comparison_data]

        # 1. Score comparison
        ax = axes[0, 0]
        x = np.arange(len(queries))
        width = 0.35

        bars1 = ax.bar(x - width/2, gnn_scores, width, label='GNN + Semantic', color='steelblue', alpha=0.8)
        bars2 = ax.bar(x + width/2, semantic_scores, width, label='Semantic Only', color='coral', alpha=0.8)

        ax.set_xlabel('Query', fontsize=11)
        ax.set_ylabel('Average Score', fontsize=11)
        ax.set_title('GNN+Semantic vs Semantic-Only Scores', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(queries, rotation=45, ha='right', fontsize=8)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        # 2. Improvement percentage
        ax = axes[0, 1]
        colors = ['green' if imp > 0 else 'red' for imp in improvements]
        bars = ax.barh(range(len(queries)), improvements, color=colors, alpha=0.7)

        ax.set_yticks(range(len(queries)))
        ax.set_yticklabels(queries, fontsize=8)
        ax.set_xlabel('Improvement (%)', fontsize=11)
        ax.set_title('Performance Improvement\n(GNN+Semantic over Semantic-Only)', fontsize=12, fontweight='bold')
        ax.axvline(0, color='black', linewidth=0.8)
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)

        # 3. Scatter plot
        ax = axes[1, 0]
        ax.scatter(semantic_scores, gnn_scores, s=100, alpha=0.6, color='steelblue')

        # Diagonal line (equal performance)
        min_score = min(min(semantic_scores), min(gnn_scores))
        max_score = max(max(semantic_scores), max(gnn_scores))
        ax.plot([min_score, max_score], [min_score, max_score], 'r--', linewidth=2, label='Equal Performance')

        ax.set_xlabel('Semantic-Only Score', fontsize=11)
        ax.set_ylabel('GNN+Semantic Score', fontsize=11)
        ax.set_title('Score Comparison Scatter Plot', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)

        # 4. Summary statistics
        ax = axes[1, 1]
        ax.axis('off')

        avg_gnn = np.mean(gnn_scores)
        avg_semantic = np.mean(semantic_scores)
        avg_improvement = np.mean(improvements)
        wins = sum(1 for imp in improvements if imp > 0)

        summary_text = f"""
Comparison Summary:

Queries Evaluated: {len(queries)}

Average Scores:
  GNN + Semantic:  {avg_gnn:.4f}
  Semantic Only:   {avg_semantic:.4f}

Average Improvement: {avg_improvement:+.2f}%

Queries where GNN helps: {wins}/{len(queries)}
Queries where GNN hurts: {len(queries) - wins}/{len(queries)}

Best Improvement: {max(improvements):+.2f}%
Worst Improvement: {min(improvements):+.2f}%
"""

        ax.text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center',
               fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

        plt.suptitle('GNN vs Semantic-Only Comparison', fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved comparison to {output_path}")
        plt.close()

    def visualize_all_queries_summary(
        self,
        detailed_results: List[Dict],
        output_path: str
    ):
        """
        Create summary visualization for all queries.

        Args:
            detailed_results: List of detailed result dictionaries
            output_path: Path to save visualization
        """
        print("Visualizing summary of all queries...")

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Extract data
        queries = [r['query'][:25] + '...' if len(r['query']) > 25 else r['query'] for r in detailed_results]
        categories = [r['category'] for r in detailed_results]
        top1_scores = [r['metrics']['top_1_score'] for r in detailed_results]
        avg_scores = [r['metrics']['avg_score'] for r in detailed_results]
        top5_scores = [r['metrics']['top_5_avg_score'] for r in detailed_results]

        # 1. Scores by query
        ax = axes[0, 0]
        x = np.arange(len(queries))
        width = 0.25

        ax.bar(x - width, top1_scores, width, label='Top-1', color='darkgreen', alpha=0.8)
        ax.bar(x, avg_scores, width, label='Average', color='steelblue', alpha=0.8)
        ax.bar(x + width, top5_scores, width, label='Top-5 Avg', color='coral', alpha=0.8)

        ax.set_xlabel('Query', fontsize=11)
        ax.set_ylabel('Score', fontsize=11)
        ax.set_title('Performance Metrics by Query', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(queries, rotation=45, ha='right', fontsize=8)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, 1.0)

        # 2. Score distribution
        ax = axes[0, 1]
        ax.boxplot([top1_scores, avg_scores, top5_scores],
                   labels=['Top-1', 'Average', 'Top-5 Avg'],
                   patch_artist=True,
                   boxprops=dict(facecolor='lightblue', alpha=0.7))
        ax.set_ylabel('Score', fontsize=11)
        ax.set_title('Score Distribution Across Queries', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        # 3. Performance by category
        ax = axes[1, 0]
        category_scores = defaultdict(list)
        for cat, score in zip(categories, avg_scores):
            category_scores[cat].append(score)

        cats = list(category_scores.keys())
        cat_means = [np.mean(category_scores[cat]) for cat in cats]

        ax.barh(range(len(cats)), cat_means, color='steelblue', alpha=0.8)
        ax.set_yticks(range(len(cats)))
        ax.set_yticklabels(cats, fontsize=9)
        ax.set_xlabel('Average Score', fontsize=11)
        ax.set_title('Performance by Category', fontsize=12, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)

        # 4. Overall statistics
        ax = axes[1, 1]
        ax.axis('off')

        stats_text = f"""
Overall Performance Summary:

Total Queries: {len(queries)}
Categories: {len(set(categories))}

Top-1 Score:
  Mean:   {np.mean(top1_scores):.4f}
  Median: {np.median(top1_scores):.4f}
  Std:    {np.std(top1_scores):.4f}

Average Score:
  Mean:   {np.mean(avg_scores):.4f}
  Median: {np.median(avg_scores):.4f}
  Std:    {np.std(avg_scores):.4f}

Top-5 Average:
  Mean:   {np.mean(top5_scores):.4f}
  Median: {np.median(top5_scores):.4f}
  Std:    {np.std(top5_scores):.4f}
"""

        ax.text(0.1, 0.5, stats_text, fontsize=11, verticalalignment='center',
               fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.suptitle('Query Performance Summary', fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved query summary to {output_path}")
        plt.close()


def visualize_results_from_directory(results_dir: str, graph_builder=None):
    """
    Create all visualizations from a results directory.

    Args:
        results_dir: Directory containing results JSON files
        graph_builder: Optional CitationGraphBuilder for network visualization
    """
    results_path = Path(results_dir)
    viz = CitationGraphVisualizer(graph_builder=graph_builder, results_dir=results_dir)

    print(f"\n{'='*80}")
    print(f"Creating visualizations for results in: {results_dir}")
    print(f"{'='*80}\n")

    # Find result files
    detailed_files = list(results_path.glob('detailed_results_*.json'))
    comparison_files = list(results_path.glob('comparison_*.json'))

    if not detailed_files:
        print("No detailed results files found!")
        return

    # Load latest results
    detailed_file = sorted(detailed_files)[-1]
    print(f"Loading results from: {detailed_file.name}")
    print(f"File path: {detailed_file}")
    print(f"File modified: {datetime.fromtimestamp(detailed_file.stat().st_mtime)}")

    with open(detailed_file, 'r') as f:
        detailed_results = json.load(f)

    # Display loaded scores for verification
    if detailed_results:
        top1_scores_check = [r['metrics']['top_1_score'] for r in detailed_results]
        avg_scores_check = [r['metrics']['avg_score'] for r in detailed_results]
        top5_scores_check = [r['metrics']['top_5_avg_score'] for r in detailed_results]

        print(f"\nLoaded {len(detailed_results)} queries")
        print(f"Computed Average Top-1 Score: {np.mean(top1_scores_check):.4f}")
        print(f"Computed Average Score: {np.mean(avg_scores_check):.4f}")
        print(f"Computed Average Top-5 Score: {np.mean(top5_scores_check):.4f}")
        print()

    # Create visualizations directory
    viz_dir = results_path / 'visualizations'
    viz_dir.mkdir(exist_ok=True)

    # Calculate total steps
    total_steps = 1  # Query summary
    if graph_builder:
        total_steps += 2  # Network + stats
    total_steps += min(5, len(detailed_results))  # Individual queries (max 5)
    if comparison_files:
        total_steps += 1  # Comparison

    # Create visualizations with progress bar
    with tqdm(total=total_steps, desc="Creating visualizations",
              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:

        # 1. Citation network visualization (if graph_builder available)
        if graph_builder:
            pbar.set_description("Building citation network graph")
            start_time = time.time()
            viz.visualize_citation_network(
                output_path=str(viz_dir / 'citation_network.svg'),
                layout='spring'
            )
            elapsed = time.time() - start_time
            pbar.update(1)
            pbar.set_postfix_str(f"Network in {elapsed:.1f}s")

            pbar.set_description("Creating network statistics")
            start_time = time.time()
            viz.visualize_network_stats(
                output_path=str(viz_dir / 'network_statistics.svg')
            )
            elapsed = time.time() - start_time
            pbar.update(1)
            pbar.set_postfix_str(f"Stats in {elapsed:.1f}s")

        # 3. Query summaries
        pbar.set_description("Creating query performance summary")
        start_time = time.time()
        viz.visualize_all_queries_summary(
            detailed_results,
            output_path=str(viz_dir / 'query_summary.svg')
        )
        elapsed = time.time() - start_time
        pbar.update(1)
        pbar.set_postfix_str(f"Summary in {elapsed:.1f}s")

        # 4. Individual query visualizations
        num_queries = min(5, len(detailed_results))
        for i, result in enumerate(detailed_results[:num_queries]):
            pbar.set_description(f"Creating query {i+1}/{num_queries} visualization")
            start_time = time.time()
            viz.visualize_query_results(
                query=result['query'],
                results=result['top_results'],
                output_path=str(viz_dir / f'query_{i+1}_results.svg')
            )
            elapsed = time.time() - start_time
            pbar.update(1)
            pbar.set_postfix_str(f"Query {i+1} in {elapsed:.1f}s")

        # 5. Comparison visualization (if available)
        if comparison_files:
            comparison_file = sorted(comparison_files)[-1]
            pbar.set_description("Creating GNN vs Semantic comparison")

            try:
                with open(comparison_file, 'r') as f:
                    comparison_data = json.load(f)

                # Check if it's a list or summary dict
                if isinstance(comparison_data, list) and len(comparison_data) > 0:
                    # Check if first element has required keys
                    if 'query' in comparison_data[0]:
                        start_time = time.time()
                        viz.visualize_gnn_vs_semantic_comparison(
                            comparison_data,
                            output_path=str(viz_dir / 'gnn_vs_semantic.svg')
                        )
                        elapsed = time.time() - start_time
                        pbar.update(1)
                        pbar.set_postfix_str(f"Comparison in {elapsed:.1f}s")
                    else:
                        pbar.write("   Note: Comparison data missing required fields. Skipping.")
                        pbar.update(1)
                elif isinstance(comparison_data, dict):
                    pbar.write("   Note: Loaded summary file. Skipping detailed comparison.")
                    pbar.update(1)
                else:
                    pbar.write("   Warning: Unknown comparison data format. Skipping.")
                    pbar.update(1)
            except Exception as e:
                pbar.write(f"   Error loading comparison data: {e}")
                pbar.update(1)

    print(f"\n{'='*80}")
    print(f"All visualizations saved to: {viz_dir}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    import sys

    # Example usage
    if len(sys.argv) > 1:
        results_dir = sys.argv[1]
    else:
        results_dir = "../results/quick_test"

    # Visualize results
    visualize_results_from_directory(results_dir)

    print("\nVisualization complete!")
    print("To visualize with citation network, run:")
    print("  python visualization.py <results_dir> --with-network")
