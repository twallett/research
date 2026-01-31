"""
Evaluation Module for Paper Recommendation System

Evaluates the recommendation system using various metrics and test queries.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
from datetime import datetime

from recommendation_system import PaperRecommendationSystem, RecommendationResult


@dataclass
class EvaluationMetrics:
    """Metrics for evaluating recommendation quality."""
    query: str
    num_results: int
    avg_score: float
    top_1_score: float
    top_5_avg_score: float
    score_variance: float
    has_relevant_results: bool
    relevance_scores: Optional[List[float]] = None


@dataclass
class ComparisonMetrics:
    """Comparison between GNN-based and semantic-only recommendations."""
    query: str
    gnn_semantic_avg_score: float
    semantic_only_avg_score: float
    improvement: float
    overlap_top_5: int
    overlap_top_10: int


class RecommendationEvaluator:
    """
    Evaluator for paper recommendation system.

    Provides methods to:
    - Test with predefined queries
    - Compare GNN vs semantic-only approaches
    - Generate evaluation metrics
    - Save results
    """

    def __init__(self, recommendation_system: PaperRecommendationSystem):
        """
        Initialize evaluator.

        Args:
            recommendation_system: Trained recommendation system
        """
        self.system = recommendation_system
        self.test_queries = self._create_test_queries()

    def _create_test_queries(self) -> List[Dict[str, str]]:
        """Create diverse test queries covering different topics."""
        return [
            {
                "query": "deep learning neural networks",
                "category": "Deep Learning"
            },
            {
                "query": "graph neural networks for molecular property prediction",
                "category": "GNN Applications"
            },
            {
                "query": "natural language processing transformers",
                "category": "NLP"
            },
            {
                "query": "RL for robotics",
                "category": "Robotics & RL"
            },
            {
                "query": "computer vision object detection",
                "category": "Computer Vision"
            },
            {
                "query": "IoT security cryptography",
                "category": "Security"
            },
            {
                "query": "machine learning optimization algorithms",
                "category": "ML Theory"
            },
            {
                "query": "recommender systems collaborative filtering",
                "category": "Recommender Systems"
            },
            {
                "query": "graph convolutional networks",
                "category": "Graph Learning"
            },
            {
                "query": "attention mechanisms neural networks",
                "category": "Attention"
            }
        ]

    def evaluate_single_query(
        self,
        query: str,
        top_k: int = 10,
        use_combined: bool = True
    ) -> Tuple[List[RecommendationResult], EvaluationMetrics]:
        """
        Evaluate a single query.

        Args:
            query: Query string
            top_k: Number of results to retrieve
            use_combined: Use combined embeddings

        Returns:
            Tuple of (results, metrics)
        """
        # Get recommendations
        results = self.system.search(query, top_k=top_k, use_combined=use_combined)

        # Compute metrics
        scores = [r.score for r in results]

        metrics = EvaluationMetrics(
            query=query,
            num_results=len(results),
            avg_score=float(np.mean(scores)) if scores else 0.0,
            top_1_score=scores[0] if scores else 0.0,
            top_5_avg_score=float(np.mean(scores[:5])) if len(scores) >= 5 else float(np.mean(scores)),
            score_variance=float(np.var(scores)) if scores else 0.0,
            has_relevant_results=scores[0] > 0.3 if scores else False,  # Threshold for relevance
            relevance_scores=scores
        )

        return results, metrics

    def evaluate_all_queries(
        self,
        top_k: int = 10,
        use_combined: bool = True,
        verbose: bool = True
    ) -> List[Tuple[Dict, List[RecommendationResult], EvaluationMetrics]]:
        """
        Evaluate all test queries.

        Args:
            top_k: Number of results per query
            use_combined: Use combined embeddings
            verbose: Print progress

        Returns:
            List of (query_info, results, metrics) tuples
        """
        all_results = []

        for query_info in self.test_queries:
            query = query_info['query']
            category = query_info['category']

            if verbose:
                print(f"\nEvaluating query: '{query}' (Category: {category})")

            results, metrics = self.evaluate_single_query(
                query=query,
                top_k=top_k,
                use_combined=use_combined
            )

            if verbose:
                print(f"  Top-1 Score: {metrics.top_1_score:.4f}")
                print(f"  Avg Score: {metrics.avg_score:.4f}")
                print(f"  Top-5 Avg: {metrics.top_5_avg_score:.4f}")

            all_results.append((query_info, results, metrics))

        return all_results

    def compare_gnn_vs_semantic(
        self,
        top_k: int = 10
    ) -> List[ComparisonMetrics]:
        """
        Compare GNN-enhanced vs semantic-only recommendations.

        Args:
            top_k: Number of results to compare

        Returns:
            List of comparison metrics
        """
        print("\n=== Comparing GNN vs Semantic-Only ===")

        comparisons = []

        for query_info in self.test_queries:
            query = query_info['query']

            # Get GNN+Semantic results
            gnn_results, gnn_metrics = self.evaluate_single_query(
                query, top_k=top_k, use_combined=True
            )

            # Get Semantic-only results
            sem_results, sem_metrics = self.evaluate_single_query(
                query, top_k=top_k, use_combined=False
            )

            # Compute overlap
            gnn_dois_5 = set([r.paper_doi for r in gnn_results[:5]])
            sem_dois_5 = set([r.paper_doi for r in sem_results[:5]])
            overlap_5 = len(gnn_dois_5 & sem_dois_5)

            gnn_dois_10 = set([r.paper_doi for r in gnn_results[:10]])
            sem_dois_10 = set([r.paper_doi for r in sem_results[:10]])
            overlap_10 = len(gnn_dois_10 & sem_dois_10)

            # Compute improvement
            improvement = (gnn_metrics.avg_score - sem_metrics.avg_score) / (sem_metrics.avg_score + 1e-8) * 100

            comparison = ComparisonMetrics(
                query=query,
                gnn_semantic_avg_score=gnn_metrics.avg_score,
                semantic_only_avg_score=sem_metrics.avg_score,
                improvement=improvement,
                overlap_top_5=overlap_5,
                overlap_top_10=overlap_10
            )

            comparisons.append(comparison)

            print(f"\nQuery: '{query}'")
            print(f"  GNN+Semantic: {gnn_metrics.avg_score:.4f}")
            print(f"  Semantic-only: {sem_metrics.avg_score:.4f}")
            print(f"  Improvement: {improvement:+.2f}%")
            print(f"  Overlap (Top-5): {overlap_5}/5, (Top-10): {overlap_10}/10")

        return comparisons

    def save_results(
        self,
        all_results: List[Tuple[Dict, List[RecommendationResult], EvaluationMetrics]],
        output_dir: str,
        comparison_metrics: Optional[List[ComparisonMetrics]] = None
    ) -> None:
        """
        Save evaluation results to files.

        Args:
            all_results: Results from evaluate_all_queries
            output_dir: Directory to save results
            comparison_metrics: Optional comparison metrics
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save detailed results for each query
        detailed_results = []
        for query_info, results, metrics in all_results:
            query_result = {
                'query': query_info['query'],
                'category': query_info['category'],
                'metrics': asdict(metrics),
                'top_results': [
                    {
                        'rank': r.rank,
                        'title': r.paper_title,
                        'doi': r.paper_doi,
                        'score': r.score,
                        'year': r.metadata.get('year'),
                        'venue': r.metadata.get('venue')
                    }
                    for r in results[:10]
                ]
            }
            detailed_results.append(query_result)

        with open(output_path / f'detailed_results_{timestamp}.json', 'w') as f:
            json.dump(detailed_results, f, indent=2)

        print(f"Saved detailed results to {output_path / f'detailed_results_{timestamp}.json'}")

        # Save summary statistics
        all_metrics = [metrics for _, _, metrics in all_results]
        summary = {
            'timestamp': timestamp,
            'num_queries': len(all_metrics),
            'avg_top1_score': float(np.mean([m.top_1_score for m in all_metrics])),
            'avg_score': float(np.mean([m.avg_score for m in all_metrics])),
            'avg_top5_score': float(np.mean([m.top_5_avg_score for m in all_metrics])),
            'num_relevant_queries': sum([m.has_relevant_results for m in all_metrics]),
            'relevance_rate': sum([m.has_relevant_results for m in all_metrics]) / len(all_metrics)
        }

        with open(output_path / f'summary_{timestamp}.json', 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"Saved summary to {output_path / f'summary_{timestamp}.json'}")

        # Save comparison metrics if provided
        if comparison_metrics:
            comparison_data = [asdict(c) for c in comparison_metrics]
            with open(output_path / f'comparison_{timestamp}.json', 'w') as f:
                json.dump(comparison_data, f, indent=2)

            print(f"Saved comparison to {output_path / f'comparison_{timestamp}.json'}")

            # Compute comparison summary
            comp_summary = {
                'avg_gnn_semantic_score': float(np.mean([c.gnn_semantic_avg_score for c in comparison_metrics])),
                'avg_semantic_only_score': float(np.mean([c.semantic_only_avg_score for c in comparison_metrics])),
                'avg_improvement_pct': float(np.mean([c.improvement for c in comparison_metrics])),
                'avg_overlap_top5': float(np.mean([c.overlap_top_5 for c in comparison_metrics])),
                'avg_overlap_top10': float(np.mean([c.overlap_top_10 for c in comparison_metrics]))
            }

            with open(output_path / f'comparison_summary_{timestamp}.json', 'w') as f:
                json.dump(comp_summary, f, indent=2)

        # Create summary table
        self._create_summary_table(all_results, output_path, timestamp)

        print(f"\nAll results saved to {output_path}")

    def _create_summary_table(
        self,
        all_results: List[Tuple[Dict, List[RecommendationResult], EvaluationMetrics]],
        output_path: Path,
        timestamp: str
    ) -> None:
        """Create a summary table of results."""
        data = []
        for query_info, results, metrics in all_results:
            data.append({
                'Query': query_info['query'][:50],  # Truncate long queries
                'Category': query_info['category'],
                'Top-1 Score': f"{metrics.top_1_score:.4f}",
                'Avg Score': f"{metrics.avg_score:.4f}",
                'Top-5 Avg': f"{metrics.top_5_avg_score:.4f}",
                'Relevant': 'Yes' if metrics.has_relevant_results else 'No'
            })

        df = pd.DataFrame(data)
        df.to_csv(output_path / f'summary_table_{timestamp}.csv', index=False)
        print(f"Saved summary table to {output_path / f'summary_table_{timestamp}.csv'}")

    def generate_report(
        self,
        all_results: List[Tuple[Dict, List[RecommendationResult], EvaluationMetrics]],
        comparison_metrics: Optional[List[ComparisonMetrics]] = None
    ) -> str:
        """
        Generate a text report of evaluation results.

        Args:
            all_results: Results from evaluation
            comparison_metrics: Optional comparison metrics

        Returns:
            Report as string
        """
        report = []
        report.append("=" * 80)
        report.append("PAPER RECOMMENDATION SYSTEM - EVALUATION REPORT")
        report.append("=" * 80)
        report.append("")

        # Overall statistics
        all_metrics = [metrics for _, _, metrics in all_results]
        report.append("OVERALL STATISTICS")
        report.append("-" * 80)
        report.append(f"Number of test queries: {len(all_metrics)}")
        report.append(f"Average Top-1 Score: {np.mean([m.top_1_score for m in all_metrics]):.4f}")
        report.append(f"Average Score: {np.mean([m.avg_score for m in all_metrics]):.4f}")
        report.append(f"Average Top-5 Score: {np.mean([m.top_5_avg_score for m in all_metrics]):.4f}")
        report.append(f"Queries with relevant results: {sum([m.has_relevant_results for m in all_metrics])}/{len(all_metrics)}")
        report.append("")

        # Per-query results
        report.append("PER-QUERY RESULTS")
        report.append("-" * 80)
        for query_info, results, metrics in all_results:
            report.append(f"\nQuery: {query_info['query']}")
            report.append(f"Category: {query_info['category']}")
            report.append(f"  Top-1 Score: {metrics.top_1_score:.4f}")
            report.append(f"  Avg Score: {metrics.avg_score:.4f}")
            report.append(f"  Top-3 Papers:")
            for i, result in enumerate(results[:3]):
                report.append(f"    {i+1}. [{result.score:.4f}] {result.paper_title}")

        # Comparison metrics
        if comparison_metrics:
            report.append("")
            report.append("=" * 80)
            report.append("GNN vs SEMANTIC-ONLY COMPARISON")
            report.append("=" * 80)
            report.append(f"Average GNN+Semantic Score: {np.mean([c.gnn_semantic_avg_score for c in comparison_metrics]):.4f}")
            report.append(f"Average Semantic-only Score: {np.mean([c.semantic_only_avg_score for c in comparison_metrics]):.4f}")
            report.append(f"Average Improvement: {np.mean([c.improvement for c in comparison_metrics]):+.2f}%")
            report.append(f"Average Top-5 Overlap: {np.mean([c.overlap_top_5 for c in comparison_metrics]):.1f}/5")
            report.append(f"Average Top-10 Overlap: {np.mean([c.overlap_top_10 for c in comparison_metrics]):.1f}/10")

        report.append("")
        report.append("=" * 80)

        return "\n".join(report)


if __name__ == "__main__":
    # Example usage
    from recommendation_system import PaperRecommendationSystem

    # Create and train system
    system = PaperRecommendationSystem(
        json_data_path="../data/preprocessed_data/papers_with_citations_sample.json"
    )
    system.build_graph()
    system.extract_semantic_features()
    system.train_gnn(num_epochs=50)

    # Evaluate
    evaluator = RecommendationEvaluator(system)
    all_results = evaluator.evaluate_all_queries(top_k=10)
    comparison = evaluator.compare_gnn_vs_semantic(top_k=10)

    # Save results
    evaluator.save_results(all_results, "../results", comparison_metrics=comparison)

    # Generate report
    report = evaluator.generate_report(all_results, comparison)
    print(report)
