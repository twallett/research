# Citation Graph Paper Recommendation System

A GNN-based paper recommendation system that combines citation graph structure with semantic content embeddings to provide intelligent paper recommendations.

## 🎯 Overview

This system implements the research idea of combining:
- **Graph Neural Networks (GNN)** for capturing citation network structure
- **Semantic Embeddings (LLM)** for understanding paper content
- **Simple Ranking** for retrieving relevant papers based on queries

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  INPUT: Papers with Citations (JSON)                           │
└─────────────────────────────────────────────────────────────────┘
                            │
                ┌───────────┴───────────┐
                │                       │
                ▼                       ▼
    ┌──────────────────┐    ┌──────────────────┐
    │ Citation Graph   │    │ Semantic         │
    │ Construction     │    │ Embeddings       │
    │                  │    │ (LLM)            │
    └──────────────────┘    └──────────────────┘
                │                       │
                ▼                       ▼
    ┌──────────────────┐    ┌──────────────────┐
    │ GNN Training     │    │ Text Embeddings  │
    │ (GCN/GAT/SAGE)   │    │                  │
    └──────────────────┘    └──────────────────┘
                │                       │
                └───────────┬───────────┘
                            ▼
                ┌──────────────────────┐
                │ Combined Embeddings  │
                └──────────────────────┘
                            │
                            ▼
                ┌──────────────────────┐
                │ Query Processing &   │
                │ Paper Ranking        │
                └──────────────────────┘
                            │
                            ▼
                ┌──────────────────────┐
                │ Top-K Papers         │
                └──────────────────────┘
```

## 📁 Project Structure

```
Citation_Graphs/
├── src/
│   ├── graph_builder.py           # Citation graph construction
│   ├── models/
│   │   ├── gnn.py                 # GNN models (GCN, GAT, SAGE)
│   │   ├── llm.py                 # LLM for embeddings
│   │   └── semantic_features.py   # Semantic feature extraction
│   ├── recommendation_system.py   # Main recommendation system
│   ├── evaluation.py              # Evaluation and metrics
│   ├── run_experiment.py          # Main pipeline script
│   └── quick_test.py              # Quick test script
├── data/
│   └── preprocessed_data/
│       ├── papers_with_citations_sample.json  # Sample data (10 papers)
│       └── papers_with_citations.json         # Full dataset
├── results/                       # Experiment results
└── requirements.txt               # Python dependencies
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Quick Test (Sample Data)

Test the system on sample data (10 papers):

```bash
cd src
python quick_test.py
```

This will:
- Build citation graph
- Extract semantic features
- Train a small GNN
- Run test queries
- Save results to `../results/quick_test/`

### 3. Run Full Experiment

Run on sample data:

```bash
python run_experiment.py --data ../data/preprocessed_data/papers_with_citations_sample.json --output ../results/sample_experiment
```

Run on full data:

```bash
python run_experiment.py --data ../data/preprocessed_data/papers_with_citations.json --output ../results/full_experiment --epochs 200
```

### 4. Interactive Query Mode

After training, you can interactively search for papers:

```bash
python run_experiment.py --data <data_path> --output <output_dir> --interactive
```

## 🔧 Configuration Options

### GNN Model Options

- `--gnn-type`: Type of GNN (`gcn`, `gat`, `sage`)
- `--num-layers`: Number of GNN layers (default: 3)
- `--hidden-dim`: Hidden dimension (default: 256)
- `--output-dim`: Output embedding dimension (default: 128)

### Training Options

- `--epochs`: Number of training epochs (default: 100)
- `--lr`: Learning rate (default: 0.001)
- `--dropout`: Dropout rate (default: 0.3)
- `--device`: Device (`cpu` or `cuda`)

### Example with Custom Settings

```bash
python run_experiment.py \
    --data ../data/preprocessed_data/papers_with_citations.json \
    --output ../results/custom_experiment \
    --gnn-type gat \
    --num-layers 4 \
    --hidden-dim 512 \
    --output-dim 256 \
    --epochs 200 \
    --lr 0.0005 \
    --device cuda
```

## 📊 Results and Metrics

The system generates comprehensive evaluation results:

### Output Files

1. **detailed_results_[timestamp].json**: Detailed results for each query
2. **summary_[timestamp].json**: Overall performance metrics
3. **comparison_[timestamp].json**: GNN vs Semantic-only comparison
4. **summary_table_[timestamp].csv**: Results table
5. **evaluation_report.txt**: Human-readable report

### Evaluation Metrics

- **Top-1 Score**: Similarity score of the best match
- **Average Score**: Average similarity across top-K results
- **Top-5 Average**: Average of top 5 results
- **Relevance Rate**: Percentage of queries with relevant results

### Sample Results Format

```json
{
  "query": "deep learning neural networks",
  "category": "Deep Learning",
  "metrics": {
    "top_1_score": 0.8234,
    "avg_score": 0.7156,
    "top_5_avg_score": 0.7543
  },
  "top_results": [
    {
      "rank": 1,
      "title": "Deep Neural Networks for Object Detection",
      "score": 0.8234,
      "year": 2019
    }
  ]
}
```

## 🧪 Test Queries

The system is evaluated on diverse queries:

1. "deep learning neural networks"
2. "graph neural networks for molecular property prediction"
3. "natural language processing transformers"
4. "reinforcement learning robotics"
5. "computer vision object detection"
6. "IoT security cryptography"
7. "machine learning optimization algorithms"
8. "recommender systems collaborative filtering"
9. "graph convolutional networks"
10. "attention mechanisms neural networks"

## 📈 GNN vs Semantic-Only Comparison

The system automatically compares:

- **GNN + Semantic**: Uses both graph structure and content
- **Semantic-only**: Uses only content embeddings

Results show:
- Average improvement from GNN
- Overlap in top-K recommendations
- Per-query performance differences

## 💡 Key Features

### 1. Citation Graph Construction
- Builds directed graph from citation relationships
- Handles large-scale citation networks
- Preserves paper metadata

### 2. Multi-layer GNN
- Supports GCN, GAT, and GraphSAGE
- Learns from citation patterns
- Aggregates information from neighbors

### 3. Semantic Understanding
- Uses sentence transformers for embeddings
- Combines title and abstract
- Weighted combination strategies

### 4. Hybrid Embeddings
- Combines GNN and semantic embeddings
- Configurable weighting
- L2 normalization for similarity

### 5. Simple Ranking
- Cosine similarity-based ranking
- Fast query processing
- Top-K retrieval

## 🔬 For Research Publication

### Novel Contributions

1. **Hybrid Approach**: Combines citation graph structure with semantic content
2. **Scalable Architecture**: Handles large citation networks
3. **Comprehensive Evaluation**: Multiple metrics and comparison baselines

### Experiments to Run

1. **Ablation Study**: GNN-only, Semantic-only, Combined
2. **GNN Comparison**: GCN vs GAT vs GraphSAGE
3. **Hyperparameter Search**: Layers, dimensions, learning rates
4. **Scalability Analysis**: Performance on different dataset sizes

### Suggested Metrics for Paper

- Precision@K
- NDCG@K
- Mean Reciprocal Rank (MRR)
- Hit Rate@K
- Novel Discovery Rate (recommendations outside query paper's citations)

## 🛠️ Customization

### Adding New Test Queries

Edit `evaluation.py`, modify `_create_test_queries()`:

```python
def _create_test_queries(self):
    return [
        {
            "query": "your custom query",
            "category": "Your Category"
        },
        # Add more queries...
    ]
```

### Using Different Embedding Models

Edit `src/models/config.json`:

```json
{
  "embedding_model_name": "allenai/scibert_scivocab_uncased"
}
```

### Changing Embedding Combination Strategy

In `recommendation_system.py`, modify `_combine_embeddings()`:

```python
def _combine_embeddings(self, gnn_weight: float = 0.7):
    # Adjust weight: higher value gives more importance to GNN
```

## 📝 Using Full Dataset

To use the full dataset instead of sample:

```bash
python run_experiment.py \
    --data ../data/preprocessed_data/papers_with_citations.json \
    --output ../results/full_experiment \
    --epochs 200 \
    --device cuda
```

**Note**: Full dataset processing requires:
- More memory (recommend 16GB+ RAM)
- GPU for faster training (optional but recommended)
- Longer training time (~30-60 minutes)

## 🐛 Troubleshooting

### Out of Memory

Reduce batch size or use smaller models:
```bash
python run_experiment.py --hidden-dim 128 --output-dim 64
```

### Slow Training

- Use GPU: `--device cuda`
- Reduce epochs: `--epochs 50`
- Use GCN instead of GAT: `--gnn-type gcn`

### Import Errors

Ensure all dependencies are installed:
```bash
pip install -r requirements.txt
```

## 📧 Citation

If you use this code in your research, please cite:

```bibtex
@article{your_paper,
  title={Scientific Paper Recommendation using Citation Graph Neural Networks},
  author={Your Name},
  journal={Your Journal},
  year={2025}
}
```

## 📄 License

MIT License - See LICENSE file for details

---

**Last Updated**: November 2024
**Version**: 1.0.0
