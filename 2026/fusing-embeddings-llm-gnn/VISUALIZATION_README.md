# Citation Graph Visualization Guide

Complete guide for visualizing citation networks and recommendation results.

---

## 📊 Overview

The visualization module creates comprehensive graphs showing:
1. **Citation Network** - NetworkX graph of paper citations
2. **Network Statistics** - Degree distribution, centrality measures, top papers
3. **Query Results** - Score distributions and rankings
4. **GNN vs Semantic Comparison** - Performance comparison charts
5. **Overall Summary** - Aggregate metrics across all queries

---

## 🚀 Quick Start

### Step 1: Install NetworkX (if not already installed)

```bash
pip install networkx
```

### Step 2: Run Visualization on Existing Results

```bash
cd src
python visualize_experiment.py --results ../results/quick_test --data ../data/preprocessed_data/papers_with_citations_sample.json
```

This creates visualizations in `../results/quick_test/visualizations/`

---

## 📋 Command Options

### Basic Usage

```bash
python visualize_experiment.py --results <results_dir> --data <data_json>
```

### Parameters

| Parameter | Required | Description |
|-----------|----------|-------------|
| `--results` | Yes | Path to results directory |
| `--data` | No | Path to JSON data (for network viz) |
| `--network-only` | No | Only create network visualizations |
| `--results-only` | No | Only create results visualizations |

### Examples

**1. Full Visualization (Network + Results)**
```bash
python visualize_experiment.py \
    --results ../results/my_experiment \
    --data ../data/preprocessed_data/papers_with_citations_sample.json
```

**2. Results Only (No Network)**
```bash
python visualize_experiment.py \
    --results ../results/my_experiment \
    --results-only
```

**3. Network Only**
```bash
python visualize_experiment.py \
    --results ../results/my_experiment \
    --data ../data/preprocessed_data/papers_with_citations_sample.json \
    --network-only
```

---

## 📁 Output Files

All visualizations are saved to: `<results_dir>/visualizations/`

### Generated Files

1. **`citation_network.png`**
   - Visual graph of citation relationships
   - Nodes: Papers (colored by type)
   - Edges: Citation links
   - Highlighted: Top papers or query results

2. **`network_statistics.png`**
   - Degree distribution histogram
   - Top 10 most cited papers
   - Top 10 papers by PageRank
   - Network properties summary

3. **`query_summary.png`**
   - Performance metrics across all queries
   - Score distributions (Top-1, Average, Top-5)
   - Performance by category
   - Overall statistics

4. **`query_1_results.png`, `query_2_results.png`, ...`
   - Individual query visualizations
   - Top-N results bar chart
   - Score distribution histogram
   - Created for first 5 queries

5. **`gnn_vs_semantic.png`**
   - GNN+Semantic vs Semantic-only comparison
   - Score comparison bars
   - Improvement percentages
   - Scatter plot analysis
   - Summary statistics

---

## 🎨 Visualization Details

### 1. Citation Network Graph

**Features:**
- Spring layout for natural clustering
- Node colors:
  - 🔴 Red: Highlighted papers
  - 🔵 Light Blue: Source papers (from dataset)
  - ⚪ Gray: Citing papers
- Node size: Based on citation count
- Directed edges: Arrows show citation direction

**Reading the Graph:**
- Arrows point FROM citing paper TO cited paper
- Larger nodes = more citations
- Dense clusters = related research areas

### 2. Network Statistics

**Four Panels:**
- **Top-left**: Degree distribution (log scale)
  - Blue: Total degree
  - Green: In-degree (citations received)
  - Red: Out-degree (references made)

- **Top-right**: Top 10 most cited papers
  - Ranked by in-degree

- **Bottom-left**: PageRank scores
  - Measures paper importance in network

- **Bottom-right**: Network properties
  - Node/edge counts
  - Average degrees
  - Network density

### 3. Query Results

**Per-Query Visualization:**
- **Left panel**: Top-N results ranked by score
  - Green bars: High scores (>0.7)
  - Blue bars: Medium scores (0.5-0.7)
  - Orange bars: Low scores (<0.5)

- **Right panel**: Score distribution
  - All results histogram
  - Mean (red line)
  - Median (green line)

### 4. GNN vs Semantic Comparison

**Four Panels:**
- **Top-left**: Side-by-side score comparison
- **Top-right**: Improvement percentages
  - Green: GNN helps
  - Red: GNN hurts

- **Bottom-left**: Scatter plot
  - Points above diagonal: GNN improves
  - Points below diagonal: GNN hurts

- **Bottom-right**: Summary statistics

### 5. Query Summary

**Overview of All Queries:**
- Performance metrics (Top-1, Avg, Top-5)
- Score distribution boxplots
- Performance by category
- Overall statistics

---

## 💡 Example Workflow

### Complete Experiment with Visualization

```bash
# 1. Run experiment
python run_experiment.py \
    --data ../data/preprocessed_data/papers_with_citations_sample.json \
    --output ../results/my_experiment \
    --epochs 100 \
    --device cuda:0

# 2. Create visualizations
python visualize_experiment.py \
    --results ../results/my_experiment \
    --data ../data/preprocessed_data/papers_with_citations_sample.json

# 3. View results
open ../results/my_experiment/visualizations/
```

### Quick Test with Visualization

```bash
# 1. Run quick test
python quick_test.py

# 2. Visualize
python visualize_experiment.py \
    --results ../results/quick_test \
    --data ../data/preprocessed_data/papers_with_citations_sample.json

# 3. View
ls -lh ../results/quick_test/visualizations/
```

---

## 🔧 Customization

### Using the Visualization API Directly

```python
from graph_builder import CitationGraphBuilder
from visualization import CitationGraphVisualizer

# Load data
builder = CitationGraphBuilder("../data/preprocessed_data/papers_with_citations_sample.json")
builder.load_data()

# Create visualizer
viz = CitationGraphVisualizer(graph_builder=builder)

# Build network
viz.build_networkx_graph()

# Custom citation network
viz.visualize_citation_network(
    output_path="my_network.png",
    layout='kamada_kawai',  # or 'spring', 'circular'
    highlight_papers=['doi1', 'doi2'],  # DOIs to highlight
    max_nodes=50  # Limit nodes shown
)

# Network stats
viz.visualize_network_stats("my_stats.png")
```

### Creating Custom Query Visualizations

```python
import json
from visualization import CitationGraphVisualizer

# Load results
with open('../results/my_experiment/results/detailed_results_*.json') as f:
    results = json.load(f)

viz = CitationGraphVisualizer()

# Visualize specific query
query_result = results[0]
viz.visualize_query_results(
    query=query_result['query'],
    results=query_result['top_results'],
    output_path='custom_query.png',
    top_n=15
)
```

---

## 📐 Layout Options

### Citation Network Layouts

**1. Spring Layout (default)**
```python
layout='spring'
```
- Best for: Small to medium graphs (< 200 nodes)
- Pros: Natural clustering, aesthetically pleasing
- Cons: Can be slow for large graphs

**2. Kamada-Kawai Layout**
```python
layout='kamada_kawai'
```
- Best for: Small graphs (< 100 nodes)
- Pros: Very clear structure
- Cons: Computationally expensive

**3. Circular Layout**
```python
layout='circular'
```
- Best for: Any size graph
- Pros: Fast, consistent
- Cons: Less informative structure

---

## 🎯 Interpretation Guide

### What to Look For

**Citation Network:**
- **Dense clusters**: Related research areas
- **Bridge nodes**: Papers connecting different topics
- **Isolated nodes**: Papers with few citations
- **Star patterns**: Highly influential papers

**Network Statistics:**
- **Power-law distribution**: Typical for citation networks
- **High PageRank**: Important papers in the network
- **High in-degree**: Highly cited papers

**Query Results:**
- **High top-1 score (>0.7)**: Strong match found
- **Narrow distribution**: Consistent quality
- **Wide distribution**: Mixed quality results

**GNN vs Semantic:**
- **Positive improvement**: GNN captures useful structure
- **Negative improvement**: Graph may be too sparse
- **High variance**: Some queries benefit more than others

---

## 📊 Sample Visualizations

### What You'll See

After running visualization, you'll get plots showing:

1. **Network Graph**: Actual citation relationships
2. **Degree Histograms**: How citations are distributed
3. **Top Papers**: Most influential papers
4. **Query Rankings**: How well papers match queries
5. **Performance Comparisons**: GNN impact

---

## 🐛 Troubleshooting

### "No module named 'networkx'"
```bash
pip install networkx
```

### "No detailed results files found"
Make sure you've run an experiment first:
```bash
python quick_test.py
# or
python run_experiment.py --data <data_path>
```

### Graph too cluttered
Reduce number of nodes:
```python
viz.visualize_citation_network(
    output_path="network.png",
    max_nodes=50  # Show only 50 most connected nodes
)
```

### Matplotlib memory issues
Process fewer queries:
```python
# Modify visualization.py line ~481
for i, result in enumerate(detailed_results[:3]):  # Only first 3
```

---

## 📖 Advanced Usage

### Batch Visualization

Visualize multiple experiments:

```bash
for exp in results/*; do
    python visualize_experiment.py --results $exp --results-only
done
```

### Programmatic Access

```python
from visualization import visualize_results_from_directory

# Visualize all results
visualize_results_from_directory(
    results_dir="../results/my_experiment",
    graph_builder=None  # or provide graph_builder
)
```

### Custom Highlighting

Highlight specific papers in network:

```python
# Highlight top query results
top_papers = [r['doi'] for r in query_results[:5]]

viz.visualize_citation_network(
    output_path="highlighted.png",
    highlight_papers=top_papers
)
```

---

## 💾 Saving and Sharing

### Export Format

All visualizations are saved as high-resolution PNG files (300 DPI), suitable for:
- Research papers
- Presentations
- Reports
- Web display

### Changing Format

Modify `visualization.py` to save as PDF:

```python
plt.savefig(output_path.replace('.png', '.pdf'), dpi=300, bbox_inches='tight')
```

---

## 📞 Common Commands Reference

```bash
# After running quick_test.py
python visualize_experiment.py \
    --results ../results/quick_test \
    --data ../data/preprocessed_data/papers_with_citations_sample.json

# After running full experiment
python visualize_experiment.py \
    --results ../results/my_experiment \
    --data ../data/preprocessed_data/papers_with_citations.json

# Results only (no network)
python visualize_experiment.py \
    --results ../results/my_experiment \
    --results-only

# Network only
python visualize_experiment.py \
    --results ../results/my_experiment \
    --data ../data/preprocessed_data/papers_with_citations_sample.json \
    --network-only
```

---

## ✅ Expected Output

After successful visualization, you'll see:

```
================================================================================
CITATION GRAPH VISUALIZATION
================================================================================

Loading citation graph from: ../data/preprocessed_data/papers_with_citations_sample.json
Loaded 10 papers

================================================================================
Creating visualizations for results in: ../results/quick_test
================================================================================

Loading results from: detailed_results_20241116_123456.json

1. Creating citation network visualization...
Building NetworkX graph from citation data...
Graph built: 60 nodes, 105 edges
Visualizing citation network (showing up to 100 nodes)...
Saved citation network to ../results/quick_test/visualizations/citation_network.png

2. Creating network statistics...
Saved network statistics to ../results/quick_test/visualizations/network_statistics.png

3. Creating query performance summary...
Saved query summary to ../results/quick_test/visualizations/query_summary.png

4. Creating individual query visualizations...
Visualizing results for query: 'deep learning neural networks'...
Saved query results to ../results/quick_test/visualizations/query_1_results.png
...

5. Creating GNN vs Semantic comparison...
Loading from: comparison_20241116_123456.json
Saved comparison to ../results/quick_test/visualizations/gnn_vs_semantic.png

================================================================================
All visualizations saved to: ../results/quick_test/visualizations
================================================================================

Visualization complete!
```

---

**Ready to visualize!** Run after any experiment to see your results graphically.

**Documentation Version**: 1.0.0
**Last Updated**: November 2024