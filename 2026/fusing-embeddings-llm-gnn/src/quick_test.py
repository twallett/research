"""
Quick test script to verify the citation graph recommendation system.

This script runs a minimal test on sample data to ensure everything is working.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from recommendation_system import PaperRecommendationSystem
from evaluation import RecommendationEvaluator


def quick_test():
    """Run a quick test of the system."""
    print("=" * 80)
    print("QUICK TEST - Citation Graph Recommendation System")
    print("=" * 80)

    # Path to sample data
    data_path = "../data/preprocessed_data/papers_with_citations_sample.json"

    print(f"\nUsing sample data: {data_path}")

    # Check if data exists
    if not Path(data_path).exists():
        print(f"ERROR: Data file not found at {data_path}")
        print("Please ensure the sample data is in the correct location.")
        return

    print("\n1. Initializing system...")
    system = PaperRecommendationSystem(
        json_data_path=data_path,
        device='cuda:0'
    )

    print("\n2. Building citation graph...")
    system.build_graph()
    print(f"   - Nodes: {system.graph_stats.num_nodes}")
    print(f"   - Edges: {system.graph_stats.num_edges}")

    print("\n3. Extracting semantic features...")
    cache_dir = "../data/semantic_cache_test"
    system.extract_semantic_features(cache_dir=cache_dir)
    print(f"   - Features extracted for {len(system.semantic_features)} papers")

    print("\n4. Training GNN (quick test with 20 epochs)...")
    losses = system.train_gnn(
        hidden_dim=128,
        output_dim=64,
        num_layers=2,
        gnn_type='gcn',
        num_epochs=20,
        learning_rate=0.01
    )
    print(f"   - Initial loss: {losses[0]:.4f}")
    print(f"   - Final loss: {losses[-1]:.4f}")

    print("\n5. Testing search functionality...")

    # Test queries
    test_queries = [
        "deep learning neural networks",
        "IoT security cryptography",
        "graph neural networks"
    ]

    for query in test_queries:
        print(f"\n   Query: '{query}'")
        results = system.search(query, top_k=3, use_combined=True)

        print(f"   Top 3 Results:")
        for i, result in enumerate(results[:3], 1):
            print(f"     {i}. [{result.score:.4f}] {result.paper_title[:60]}...")

    print("\n6. Running evaluation...")
    evaluator = RecommendationEvaluator(system)

    # Evaluate first 3 queries only for quick test
    evaluator.test_queries = evaluator.test_queries[:3]
    all_results = evaluator.evaluate_all_queries(top_k=5, verbose=False)

    print(f"   - Evaluated {len(all_results)} queries")
    for query_info, results, metrics in all_results:
        print(f"   - {query_info['query']}: Top-1 score = {metrics.top_1_score:.4f}")

    print("\n7. Saving results...")
    output_dir = "../results/quick_test"
    evaluator.save_results(all_results, output_dir)

    print("\n" + "=" * 80)
    print("QUICK TEST COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"\nResults saved to: {output_dir}")
    print("\nThe system is working correctly. You can now:")
    print("  1. Run full experiments with: python run_experiment.py --data <path>")
    print("  2. Use the full dataset: papers_with_citations.json")
    print("  3. Adjust hyperparameters for better performance")


if __name__ == "__main__":
    try:
        quick_test()
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
