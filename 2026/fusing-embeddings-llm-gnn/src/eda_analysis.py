"""
Exploratory Data Analysis (EDA) for Citation/Reference JSON files
Analyzes distributions of citations, abstracts, titles, authors, years, fieldsOfStudy, and venues
"""

import json
import argparse
import os
from collections import Counter
from typing import List, Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style for better-looking plots
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except OSError:
    try:
        plt.style.use('seaborn-darkgrid')
    except OSError:
        plt.style.use('ggplot')
sns.set_palette("husl")


def load_json_data(file_path: str) -> List[Dict[str, Any]]:
    """Load JSON data from file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    print(f"Loading data from {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} papers")
    return data


def count_words(text: str) -> int:
    """Count words in a text string."""
    if not text or not isinstance(text, str):
        return 0
    return len(text.split())


def extract_paper_statistics(data: List[Dict[str, Any]], relation_type: str = 'citations') -> Dict[str, Any]:
    """
    Extract statistics from the JSON data.
    
    Args:
        data: List of paper dictionaries
        relation_type: 'citations' or 'references'
    
    Returns:
        Dictionary containing extracted statistics
    """
    stats = {
        'citation_counts': [],
        'abstract_word_counts': [],
        'title_word_counts': [],
        'author_counts': [],
        'years': [],
        'fieldsOfStudy': [],
        'venues': [],
        'relation_citation_counts': [],
        'relation_abstract_word_counts': [],
        'relation_title_word_counts': [],
        'relation_author_counts': [],
        'relation_years': [],
        'relation_fieldsOfStudy': [],
        'relation_venues': []
    }
    
    relation_key = 'citations' if relation_type == 'citations' else 'references'
    
    for paper in data:
        paper_details = paper.get('paper_details', {})
        
        # Main paper statistics
        citation_count = paper_details.get('citationCount', 0)
        if citation_count is not None:
            stats['citation_counts'].append(citation_count)
        
        abstract = paper_details.get('abstract', '')
        stats['abstract_word_counts'].append(count_words(abstract))
        
        title = paper_details.get('title', '')
        stats['title_word_counts'].append(count_words(title))
        
        authors = paper_details.get('authors', [])
        stats['author_counts'].append(len(authors) if authors else 0)
        
        year = paper_details.get('year')
        if year is not None:
            stats['years'].append(year)
        
        fields = paper_details.get('fieldsOfStudy', [])
        if fields:
            stats['fieldsOfStudy'].extend(fields)
        
        venue = paper_details.get('venue', '')
        if venue:
            stats['venues'].append(venue)
        
        # Related papers (citations/references) statistics
        relations = paper.get(relation_key, [])
        for relation in relations:
            rel_citation_count = relation.get('citationCount', 0)
            if rel_citation_count is not None:
                stats['relation_citation_counts'].append(rel_citation_count)
            
            rel_abstract = relation.get('abstract', '')
            stats['relation_abstract_word_counts'].append(count_words(rel_abstract))
            
            rel_title = relation.get('title', '')
            stats['relation_title_word_counts'].append(count_words(rel_title))
            
            rel_authors = relation.get('authors', [])
            stats['relation_author_counts'].append(len(rel_authors) if rel_authors else 0)
            
            rel_year = relation.get('year')
            if rel_year is not None and rel_year >= 1940:
                stats['relation_years'].append(rel_year)
            
            rel_fields = relation.get('fieldsOfStudy', [])
            if rel_fields:
                stats['relation_fieldsOfStudy'].extend(rel_fields)
            
            rel_venue = relation.get('venue', '')
            if rel_venue:
                stats['relation_venues'].append(rel_venue)
    
    return stats


def create_stacked_histogram(main_data, relation_data, bins, ax, xlabel, title, relation_type, main_color='#3498db', relation_color='#e74c3c'):
    """Create a stacked histogram bar plot."""
    if not main_data and not relation_data:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        return

    # Combine data to determine bins
    all_data = main_data + relation_data
    if not all_data:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        return

    # Create bins
    if isinstance(bins, int):
        bin_edges = np.linspace(min(all_data), max(all_data), bins + 1)
    else:
        bin_edges = bins

    # Count main and relation data in each bin
    main_counts, _ = np.histogram(main_data, bins=bin_edges)
    relation_counts, _ = np.histogram(relation_data, bins=bin_edges)

    # Bin centers for x-axis
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    # Calculate width for each bin individually (important for log-scale bins)
    bin_widths = bin_edges[1:] - bin_edges[:-1]

    # Create stacked bar plot with individual widths for each bar
    ax.bar(bin_centers, main_counts, width=bin_widths * 0.8,
           label='Main Papers', color=main_color, alpha=0.7, edgecolor='black')
    ax.bar(bin_centers, relation_counts, width=bin_widths * 0.8,
           bottom=main_counts, label=f'Related Papers ({relation_type})',
           color=relation_color, alpha=0.7, edgecolor='black')

    ax.set_xlabel(xlabel)
    ax.set_ylabel('Frequency')
    ax.set_title(title)
    ax.legend()

def create_visualizations(stats: Dict[str, Any], relation_type: str, output_dir: str = 'results'):
    """Create and save visualization plots individually (same plots, no subplots)."""
    os.makedirs(output_dir, exist_ok=True)

    main_color = '#3498db'
    relation_color = '#e74c3c'

    # 1. Citation counts
    fig, ax = plt.subplots(figsize=(8, 6))
    main_citations = [c for c in stats['citation_counts'] if c > 0]
    relation_citations = [c for c in stats['relation_citation_counts'] if c > 0]

    if main_citations or relation_citations:
        all_citations = main_citations + relation_citations
        bins = np.logspace(0, np.log10(max(all_citations) + 1), 50) if all_citations else 50
        create_stacked_histogram(
            main_citations, relation_citations, bins, ax,
            'Number of Citations',
            'Distribution of Citation Counts (log scale)',
            relation_type, main_color, relation_color
        )
        ax.set_xscale('log')
        ax.set_yscale('log')
    else:
        ax.text(0.5, 0.5, 'No citation data', ha='center', va='center', transform=ax.transAxes)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'citation_counts.svg'))
    plt.close(fig)

    # 2. Abstract word counts
    fig, ax = plt.subplots(figsize=(8, 6))
    if stats['abstract_word_counts'] or stats['relation_abstract_word_counts']:
        all_abstracts = stats['abstract_word_counts'] + stats['relation_abstract_word_counts']
        bins = np.linspace(0, max(all_abstracts) if all_abstracts else 1000, 51)
        create_stacked_histogram(
            stats['abstract_word_counts'],
            stats['relation_abstract_word_counts'],
            bins, ax,
            'Number of Words in Abstract',
            'Distribution of Abstract Word Counts',
            relation_type, main_color, relation_color
        )
    else:
        ax.text(0.5, 0.5, 'No abstract data', ha='center', va='center', transform=ax.transAxes)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'abstract_word_counts.svg'))
    plt.close(fig)

    # 3. Title word counts
    fig, ax = plt.subplots(figsize=(8, 6))
    if stats['title_word_counts'] or stats['relation_title_word_counts']:
        all_titles = stats['title_word_counts'] + stats['relation_title_word_counts']
        bins = np.linspace(0, max(all_titles) if all_titles else 50, 31)
        create_stacked_histogram(
            stats['title_word_counts'],
            stats['relation_title_word_counts'],
            bins, ax,
            'Number of Words in Title',
            'Distribution of Title Word Counts',
            relation_type, main_color, relation_color
        )
    else:
        ax.text(0.5, 0.5, 'No title data', ha='center', va='center', transform=ax.transAxes)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'title_word_counts.svg'))
    plt.close(fig)

    # 4. Publication years
    fig, ax = plt.subplots(figsize=(10, 6))
    if stats['years'] or stats['relation_years']:
        all_years = stats['years'] + stats['relation_years']
        min_year = min(all_years)
        max_year = max(all_years)
        bins = np.arange(min_year, max_year + 2)
        create_stacked_histogram(
            stats['years'],
            stats['relation_years'],
            bins, ax,
            'Publication Year',
            'Distribution of Publication Years',
            relation_type, main_color, relation_color
        )
        
        # x-axis ticks: every 10 years
        decade_ticks = np.arange((min_year // 10) * 10, ((max_year // 10) + 1) * 10, 10)
        ax.set_xticks(decade_ticks)
        ax.tick_params(axis='x', rotation=45)
    else:
        ax.text(0.5, 0.5, 'No year data', ha='center', va='center', transform=ax.transAxes)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'publication_years.svg'))
    plt.close(fig)

    # 5. Top 15 fields of study
    fig, ax = plt.subplots(figsize=(10, 6))
    if stats['fieldsOfStudy'] or stats['relation_fieldsOfStudy']:
        main_counter = Counter(stats['fieldsOfStudy'])
        rel_counter = Counter(stats['relation_fieldsOfStudy'])
        all_fields = stats['fieldsOfStudy'] + stats['relation_fieldsOfStudy']
        top_fields = Counter(all_fields).most_common(15)

        if top_fields:
            fields, _ = zip(*top_fields)
            main_counts = [main_counter.get(f, 0) for f in fields]
            rel_counts = [rel_counter.get(f, 0) for f in fields]

            y = range(len(fields))
            ax.barh(y, main_counts, color=main_color, alpha=0.7, edgecolor='black', label='Main Papers')
            ax.barh(y, rel_counts, left=main_counts, color=relation_color,
                    alpha=0.7, edgecolor='black',
                    label=f'Related Papers ({relation_type})')
            ax.set_yticks(y)
            ax.set_yticklabels(fields)
            ax.invert_yaxis()
            ax.set_xlabel('Frequency')
            ax.set_title('Top 15 Fields of Study')
            ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'top_fields_of_study.svg'))
    plt.close(fig)

    # 6. Top venues
    fig, ax = plt.subplots(figsize=(10, 6))
    if stats['venues'] or stats['relation_venues']:
        main_counter = Counter(stats['venues'])
        rel_counter = Counter(stats['relation_venues'])
        all_venues = stats['venues'] + stats['relation_venues']
        top_venues = Counter(all_venues).most_common(15)

        if top_venues:
            venues, _ = zip(*top_venues)
            main_counts = [main_counter.get(v, 0) for v in venues]
            rel_counts = [rel_counter.get(v, 0) for v in venues]
            labels = [v[:60] + '...' if len(v) > 60 else v for v in venues]

            y = range(len(venues))
            ax.barh(y, main_counts, color=main_color, alpha=0.7, edgecolor='black', label='Main Papers')
            ax.barh(y, rel_counts, left=main_counts, color=relation_color,
                    alpha=0.7, edgecolor='black',
                    label=f'Related Papers ({relation_type})')
            ax.set_yticks(y)
            ax.set_yticklabels(labels)
            ax.invert_yaxis()
            ax.set_xlabel('Frequency')
            ax.set_title('Top 15 Venues')
            ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'top_venues.svg'))
    plt.close(fig)

    print(f"Saved individual plots to {output_dir}")



def create_detailed_plots(stats: Dict[str, Any], relation_type: str, output_dir: str):
    """Create additional detailed visualization plots."""

    # Box plots for numerical distributions
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Citation counts box plot
    ax1 = axes[0, 0]
    all_citations = stats['citation_counts'] + stats['relation_citation_counts']
    if all_citations and stats['citation_counts'] and stats['relation_citation_counts']:
        citation_data = [stats['citation_counts'], stats['relation_citation_counts']]
        citation_labels = ['Main Papers', f'Related Papers ({relation_type})']
        bp = ax1.boxplot(citation_data, labels=citation_labels, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
        ax1.set_ylabel('Citation Count')
        ax1.set_title('Citation Count Distribution Comparison')
        ax1.set_yscale('log')
    else:
        ax1.text(0.5, 0.5, 'Insufficient citation data', ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('Citation Count Distribution Comparison')

    # Abstract word counts box plot
    ax2 = axes[0, 1]
    all_abstracts = stats['abstract_word_counts'] + stats['relation_abstract_word_counts']
    if all_abstracts and stats['abstract_word_counts'] and stats['relation_abstract_word_counts']:
        abstract_data = [stats['abstract_word_counts'], stats['relation_abstract_word_counts']]
        bp2 = ax2.boxplot(abstract_data, labels=['Main Papers', f'Related Papers ({relation_type})'], patch_artist=True)
        for patch in bp2['boxes']:
            patch.set_facecolor('lightgreen')
        ax2.set_ylabel('Word Count')
        ax2.set_title('Abstract Word Count Distribution Comparison')
    else:
        ax2.text(0.5, 0.5, 'Insufficient abstract data', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Abstract Word Count Distribution Comparison')

    # Author counts box plot
    ax3 = axes[1, 0]
    all_authors = stats['author_counts'] + stats['relation_author_counts']
    if all_authors and stats['author_counts'] and stats['relation_author_counts']:
        author_data = [stats['author_counts'], stats['relation_author_counts']]
        bp3 = ax3.boxplot(author_data, labels=['Main Papers', f'Related Papers ({relation_type})'], patch_artist=True)
        for patch in bp3['boxes']:
            patch.set_facecolor('lightcoral')
        ax3.set_ylabel('Number of Authors')
        ax3.set_title('Author Count Distribution Comparison')
    else:
        ax3.text(0.5, 0.5, 'Insufficient author data', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('Author Count Distribution Comparison')

    # Year distribution comparison
    ax4 = axes[1, 1]
    if stats['years'] and stats['relation_years']:
        year_data = [stats['years'], stats['relation_years']]
        ax4.hist(year_data, bins=30, label=['Main Papers', f'Related Papers ({relation_type})'],
                alpha=0.7, edgecolor='black')
        ax4.set_xlabel('Publication Year')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Publication Year Distribution Comparison')
        ax4.legend()

    plt.tight_layout()
    output_path = os.path.join(output_dir, f'eda_{relation_type}_detailed.svg')
    plt.savefig(output_path, format='svg', bbox_inches='tight')
    print(f"Detailed visualization saved to {output_path}")
    plt.close()


def print_summary_statistics(stats: Dict[str, Any], relation_type: str):
    """Print summary statistics to console."""
    print("\n" + "="*80)
    print(f"EXPLORATORY DATA ANALYSIS SUMMARY - {relation_type.upper()}")
    print("="*80)
    
    # Main papers statistics
    print("\n--- MAIN PAPERS STATISTICS ---")
    if stats['citation_counts']:
        print(f"Citation Counts: Mean={np.mean(stats['citation_counts']):.2f}, "
              f"Median={np.median(stats['citation_counts']):.2f}, "
              f"Min={np.min(stats['citation_counts'])}, Max={np.max(stats['citation_counts'])}")
    
    if stats['abstract_word_counts']:
        print(f"Abstract Word Counts: Mean={np.mean(stats['abstract_word_counts']):.2f}, "
              f"Median={np.median(stats['abstract_word_counts']):.2f}")
    
    if stats['title_word_counts']:
        print(f"Title Word Counts: Mean={np.mean(stats['title_word_counts']):.2f}, "
              f"Median={np.median(stats['title_word_counts']):.2f}")
    
    if stats['author_counts']:
        print(f"Author Counts: Mean={np.mean(stats['author_counts']):.2f}, "
              f"Median={np.median(stats['author_counts']):.2f}, "
              f"Max={np.max(stats['author_counts'])}")
    
    if stats['years']:
        print(f"Publication Years: Mean={np.mean(stats['years']):.1f}, "
              f"Median={np.median(stats['years']):.1f}, "
              f"Range={np.min(stats['years'])}-{np.max(stats['years'])}")
    
    # Related papers statistics
    print(f"\n--- RELATED PAPERS ({relation_type.upper()}) STATISTICS ---")
    if stats['relation_citation_counts']:
        print(f"Citation Counts: Mean={np.mean(stats['relation_citation_counts']):.2f}, "
              f"Median={np.median(stats['relation_citation_counts']):.2f}")
    
    if stats['relation_abstract_word_counts']:
        print(f"Abstract Word Counts: Mean={np.mean(stats['relation_abstract_word_counts']):.2f}, "
              f"Median={np.median(stats['relation_abstract_word_counts']):.2f}")
    
    # Top fields and venues
    if stats['fieldsOfStudy']:
        field_counter = Counter(stats['fieldsOfStudy'])
        print(f"\nTop 5 Fields of Study: {field_counter.most_common(5)}")
    
    if stats['venues']:
        venue_counter = Counter(stats['venues'])
        print(f"\nTop 5 Venues: {venue_counter.most_common(5)}")
    
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Exploratory Data Analysis for Citation/Reference JSON files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze citations data
  python eda_analysis.py --type citations --input src/papers_with_citations_sample.json
  
  # Analyze references data
  python eda_analysis.py --type references --input src/papers_with_references_sample.json
  
  # Use default file paths
  python eda_analysis.py --type citations
  python eda_analysis.py --type references
        """
    )
    
    parser.add_argument(
        '--type',
        type=str,
        choices=['citations', 'references'],
        default='references',
        help='Type of relation to analyze: "citations" or "references"'
    )
    
    parser.add_argument(
        '--input',
        type=str,
        help='Path to input JSON file. If not provided, will use default paths based on type.'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='../results/eda_analysis',
        help='Output directory for visualization files (default: results)'
    )
    
    args = parser.parse_args()
    
    # Determine input file path
    if args.input:
        input_file = args.input
    else:
        # Use default paths
        if args.type == 'citations':
            # Try sample file first, then full file
            if os.path.exists('../data/preprocessed_data/papers_with_citations.json'):
                input_file = '../data/preprocessed_data/papers_with_citations.json'
            elif os.path.exists('../data/preprocessed_data/papers_with_citations_sample.json'):
                input_file = '../data/preprocessed_data/papers_with_citations_sample.json'
            else:
                input_file = 'src/papers_with_citations_sample.json'
        else:  # references
            if os.path.exists('../data/preprocessed_data/papers_with_references.json'):
                input_file = '../data/preprocessed_data/papers_with_references.json'
            elif os.path.exists('../data/preprocessed_data/papers_with_references_sample.json'):
                input_file = '../data/preprocessed_data/papers_with_references_sample.json'
            else:
                input_file = 'src/papers_with_references_sample.json'
    
    # Load data
    try:
        data = load_json_data(input_file)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(f"\nPlease provide a valid input file using --input option")
        return
    
    # Extract statistics
    print("Extracting statistics...")
    stats = extract_paper_statistics(data, args.type)
    
    # Print summary
    print_summary_statistics(stats, args.type)
    
    # Create visualizations
    print("Creating visualizations...")
    create_visualizations(stats, args.type, args.output)
    
    print(f"\nEDA analysis complete! Results saved to {args.output}/")


if __name__ == "__main__":
    main()
