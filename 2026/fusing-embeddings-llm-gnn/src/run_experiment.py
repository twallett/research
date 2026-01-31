"""
Main Pipeline Script for Paper Recommendation System

This script runs the complete pipeline:
1. Build citation graph from data
2. Extract semantic features
3. Train GNN model
4. Evaluate on test queries
5. Save results and model
"""

import argparse
import sys
from pathlib import Path
import torch

from recommendation_system import PaperRecommendationSystem
from evaluation import RecommendationEvaluator


def run_experiment(
    data_path: str,
    output_dir: str,
    num_epochs: int = 100,
    hidden_dim: int = 256,
    output_dim: int = 128,
    num_layers: int = 3,
    gnn_type: str = 'gcn',
    learning_rate: float = 0.001,
    dropout: float = 0.3,
    device: str = 'cuda:0',
    save_model: bool = True,
    run_comparison: bool = True,
    verbose: bool = True
):
    """
    Run complete experiment pipeline.

    Args:
        data_path: Path to JSON data file
        output_dir: Directory to save results
        num_epochs: Number of training epochs
        hidden_dim: Hidden dimension for GNN
        output_dim: Output embedding dimension
        num_layers: Number of GNN layers
        gnn_type: Type of GNN ('gcn', 'gat', 'sage')
        learning_rate: Learning rate
        dropout: Dropout rate
        device: Device to use ('cpu' or 'cuda')
        save_model: Whether to save trained model
        run_comparison: Whether to run GNN vs semantic comparison
        verbose: Print detailed output
    """
    print("=" * 80)
    print("CITATION GRAPH PAPER RECOMMENDATION SYSTEM")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Data: {data_path}")
    print(f"  Output: {output_dir}")
    print(f"  GNN Type: {gnn_type}")
    print(f"  Layers: {num_layers}")
    print(f"  Hidden Dim: {hidden_dim}")
    print(f"  Output Dim: {output_dim}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Device: {device}")
    print()

    # Initialize system
    if verbose:
        print("Initializing recommendation system...")

    system = PaperRecommendationSystem(
        json_data_path=data_path,
        device=device
    )

    # Build graph
    if verbose:
        print("\nStep 1: Building citation graph...")

    system.build_graph(min_citations=0)

    if verbose:
        print(f"  Nodes: {system.graph_stats.num_nodes}")
        print(f"  Edges: {system.graph_stats.num_edges}")
        print(f"  Avg citations: {system.graph_stats.avg_citations_per_paper:.2f}")

    # Extract semantic features
    if verbose:
        print("\nStep 2: Extracting semantic features...")

    cache_dir = Path(output_dir) / "semantic_cache"
    system.extract_semantic_features(cache_dir=str(cache_dir))

    # Train GNN
    if verbose:
        print("\nStep 3: Training GNN model...")

    losses = system.train_gnn(
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_layers=num_layers,
        gnn_type=gnn_type,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        dropout=dropout
    )

    if verbose:
        print(f"  Initial loss: {losses[0]:.4f}")
        print(f"  Final loss: {losses[-1]:.4f}")

    # Evaluate
    if verbose:
        print("\nStep 4: Evaluating on test queries...")

    evaluator = RecommendationEvaluator(system)

    # Run main evaluation
    all_results = evaluator.evaluate_all_queries(top_k=10, verbose=verbose)

    # Run comparison if requested
    comparison_metrics = None
    if run_comparison:
        if verbose:
            print("\nStep 5: Comparing GNN vs Semantic-only...")
        comparison_metrics = evaluator.compare_gnn_vs_semantic(top_k=10)

    # Save results
    if verbose:
        print("\nStep 6: Saving results...")

    results_dir = Path(output_dir) / "results"
    evaluator.save_results(all_results, str(results_dir), comparison_metrics=comparison_metrics)

    # Generate and save report
    report = evaluator.generate_report(all_results, comparison_metrics)
    report_path = results_dir / "evaluation_report.txt"
    with open(report_path, 'w') as f:
        f.write(report)

    print(f"\nReport saved to {report_path}")
    print("\n" + report)

    # Save model if requested
    if save_model:
        if verbose:
            print("\nStep 7: Saving trained model...")

        model_dir = Path(output_dir) / "models"
        system.save_model(str(model_dir))

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETED SUCCESSFULLY!")
    print("=" * 80)

    return system, evaluator, all_results


def interactive_mode(system: PaperRecommendationSystem):
    """
    Run interactive query mode.

    Args:
        system: Trained recommendation system
    """
    print("\n" + "=" * 80)
    print("INTERACTIVE QUERY MODE")
    print("=" * 80)
    print("Enter queries to search for papers. Type 'quit' to exit.\n")

    while True:
        try:
            query = input("Query: ").strip()

            if query.lower() in ['quit', 'exit', 'q']:
                print("Exiting...")
                break

            if not query:
                continue

            # Search
            results = system.search(query, top_k=10, use_combined=True)

            # Print results
            print(f"\n{'='*80}")
            print(f"Top 10 Results for: '{query}'")
            print(f"{'='*80}\n")

            for result in results:
                print(f"Rank {result.rank}: Score = {result.score:.4f}")
                print(f"  Title: {result.paper_title}")
                print(f"  DOI: {result.paper_doi}")

                authors = result.metadata.get('authors', [])
                if authors:
                    author_names = [a.get('name', 'Unknown') for a in authors[:3]]
                    print(f"  Authors: {', '.join(author_names)}")

                year = result.metadata.get('year')
                if year:
                    print(f"  Year: {year}")

                print()

        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Citation Graph Paper Recommendation System"
    )

    parser.add_argument(
        '--data',
        type=str,
        required=True,
        help='Path to JSON data file (papers_with_citations.json)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='../results/experiment',
        help='Output directory for results and models'
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs (default: 100)'
    )

    parser.add_argument(
        '--hidden-dim',
        type=int,
        default=256,
        help='Hidden dimension (default: 256)'
    )

    parser.add_argument(
        '--output-dim',
        type=int,
        default=128,
        help='Output embedding dimension (default: 128)'
    )

    parser.add_argument(
        '--num-layers',
        type=int,
        default=3,
        help='Number of GNN layers (default: 3)'
    )

    parser.add_argument(
        '--gnn-type',
        type=str,
        default='gcn',
        choices=['gcn', 'gat', 'sage'],
        help='Type of GNN (default: gcn)'
    )

    parser.add_argument(
        '--lr',
        type=float,
        default=0.001,
        help='Learning rate (default: 0.001)'
    )

    parser.add_argument(
        '--dropout',
        type=float,
        default=0.3,
        help='Dropout rate (default: 0.3)'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use (default: auto-detect)'
    )

    parser.add_argument(
        '--no-save-model',
        action='store_true',
        help='Do not save trained model'
    )

    parser.add_argument(
        '--no-comparison',
        action='store_true',
        help='Skip GNN vs semantic comparison'
    )

    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Run interactive query mode after training'
    )

    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Reduce output verbosity'
    )

    args = parser.parse_args()

    # Run experiment
    system, evaluator, results = run_experiment(
        data_path=args.data,
        output_dir=args.output,
        num_epochs=args.epochs,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        num_layers=args.num_layers,
        gnn_type=args.gnn_type,
        learning_rate=args.lr,
        dropout=args.dropout,
        device=args.device,
        save_model=not args.no_save_model,
        run_comparison=not args.no_comparison,
        verbose=not args.quiet
    )

    # Interactive mode if requested
    if args.interactive:
        interactive_mode(system)


if __name__ == "__main__":
    main()
