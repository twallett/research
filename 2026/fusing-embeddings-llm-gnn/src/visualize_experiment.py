"""
Script to visualize experiment results with citation network.

Usage:
    python visualize_experiment.py --results <results_dir> --data <data_json>
"""

import argparse
import sys
import platform
from pathlib import Path
from tqdm import tqdm
import time

from graph_builder import CitationGraphBuilder
from visualization import CitationGraphVisualizer, visualize_results_from_directory


def resolve_path(path_str):
    """
    Resolve path for cross-platform compatibility.
    Converts relative paths to absolute and handles OS-specific path separators.

    Args:
        path_str: Path string to resolve

    Returns:
        Resolved Path object
    """
    path = Path(path_str)

    # Convert to absolute path if relative
    if not path.is_absolute():
        path = Path.cwd() / path

    # Resolve to normalize the path (remove ../ and ./)
    path = path.resolve()

    return path


def get_default_results_dir():
    """
    Get platform-specific default results directory.

    Returns:
        Path object for default results directory
    """
    system = platform.system().lower()

    if system == 'linux' or system == 'darwin':  # darwin is macOS
        # Use /tmp for Linux/Unix systems
        base_dir = Path('/tmp/citation_graph_results')
    else:  # Windows
        # Use current directory/results for Windows
        base_dir = Path.cwd() / 'results'

    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir


def main():
    parser = argparse.ArgumentParser(
        description="Visualize citation graph recommendation results"
    )

    parser.add_argument(
        '--results',
        type=str,
        required=True,
        help='Path to results directory'
    )

    parser.add_argument(
        '--data',
        type=str,
        help='Path to JSON data file (for network visualization)'
    )

    parser.add_argument(
        '--network-only',
        action='store_true',
        help='Only create network visualizations'
    )

    parser.add_argument(
        '--results-only',
        action='store_true',
        help='Only create results visualizations (no network)'
    )

    args = parser.parse_args()

    # Detect operating system
    current_os = platform.system()
    print("="*80)
    print("CITATION GRAPH VISUALIZATION")
    print(f"Operating System: {current_os}")
    print("="*80)

    # Resolve paths for cross-platform compatibility
    results_path = resolve_path(args.results)
    print(f"\nResults directory: {results_path}")

    # Build graph if data provided
    graph_builder = None
    if args.data and not args.results_only:
        data_path = resolve_path(args.data)

        with tqdm(total=2, desc="Loading citation data", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:
            pbar.set_description("Loading citation graph from file")
            start_time = time.time()
            graph_builder = CitationGraphBuilder(str(data_path))
            graph_builder.load_data()
            load_time = time.time() - start_time
            pbar.update(1)
            pbar.set_postfix_str(f"Loaded {len(graph_builder.papers_data)} papers in {load_time:.2f}s")

            # Build the graph to populate metadata
            pbar.set_description("Building graph structure")
            start_time = time.time()
            graph_data, graph_stats = graph_builder.build_graph()
            build_time = time.time() - start_time
            pbar.update(1)
            pbar.set_postfix_str(f"{graph_stats.num_nodes} nodes, {graph_stats.num_edges} edges in {build_time:.2f}s")

    # Create visualizations
    if args.network_only and graph_builder:
        # Only network visualizations
        viz_dir = results_path / 'visualizations'
        viz_dir.mkdir(parents=True, exist_ok=True)

        viz = CitationGraphVisualizer(graph_builder=graph_builder)

        with tqdm(total=2, desc="Creating visualizations", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:
            pbar.set_description("Creating citation network")
            start_time = time.time()
            viz.visualize_citation_network(
                output_path=str(viz_dir / 'citation_network.svg'),
                layout='spring'
            )
            network_time = time.time() - start_time
            pbar.update(1)
            pbar.set_postfix_str(f"Network graph in {network_time:.2f}s")

            pbar.set_description("Creating network statistics")
            start_time = time.time()
            viz.visualize_network_stats(
                output_path=str(viz_dir / 'network_statistics.svg')
            )
            stats_time = time.time() - start_time
            pbar.update(1)
            pbar.set_postfix_str(f"Statistics in {stats_time:.2f}s")

        print(f"\nNetwork visualizations saved to: {viz_dir}")

    else:
        # All visualizations with progress tracking
        print("\n" + "="*80)
        print("CREATING ALL VISUALIZATIONS")
        print("="*80)
        start_total = time.time()

        visualize_results_from_directory(
            str(results_path),
            graph_builder=graph_builder
        )

        total_time = time.time() - start_total
        print(f"\nTotal visualization time: {total_time:.2f}s ({total_time/60:.2f} minutes)")

    print("\n" + "="*80)
    print("VISUALIZATION COMPLETED!")
    print("="*80)


if __name__ == "__main__":
    main()
