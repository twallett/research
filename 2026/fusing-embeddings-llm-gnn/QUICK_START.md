# Quick Start Guide - Citation Graph Recommendation System

## 🚀 Getting Started in 4 Steps

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Data Preparation

```bash
# Step 1: Preprocess and sample
python datapreprocessing.py --step preprocess \
    --input ../data/raw_data/arxiv-metadata-oai-snapshot.json \
    --n_samples 10000 \
    --output ../data/raw_data/arxiv_cs_stat_sampled.csv \
    --cleaned_output ../data/preprocessed_data/arxiv_cs_stat_sampled_cleaned.csv

# Step 2: Build citations JSON
python datapreprocessing.py --step build_relations \
    --input_csv ../data/preprocessed_data/arxiv_cs_stat_sampled_cleaned.csv \
    --endpoint_type citations \
    --output_json ../data/preprocessed_data/papers_with_citations.json

# Full pipeline (both steps)
python datapreprocessing.py --step all \
    --input ../data/raw_data/arxiv-metadata-oai-snapshot.json \
    --n_samples 10000 \
    --output ../data/raw_data/arxiv_cs_stat_sampled.csv \
    --cleaned_output ../data/preprocessed_data/arxiv_cs_stat_sampled_cleaned.csv \
    --endpoint_type citations \
    --output_json ../data/preprocessed_data/papers_with_citations.json
```

### Step 3: Run Quick Test

```bash
cd src
python quick_test.py
```

This validates everything is working (takes ~2-5 minutes).

### Step 4: Run Full Experiment

```bash
python run_experiment.py \
    --data ../data/preprocessed_data/papers_with_citations_sample.json \
    --output ../results/my_experiment \
    --epochs 100 \
    --device cuda:0
```

---

## 📋 Common Commands

### Sample Data Experiments

```bash
# Basic run with default parameters
python run_experiment.py \
    --data ../data/preprocessed_data/papers_with_citations_sample.json \
    --output ../results/basic

# Using GAT instead of GCN
python run_experiment.py \
    --data ../data/preprocessed_data/papers_with_citations_sample.json \
    --output ../results/gat_experiment \
    --gnn-type gat

# Larger model
python run_experiment.py \
    --data ../data/preprocessed_data/papers_with_citations_sample.json \
    --output ../results/large_model \
    --hidden-dim 512 \
    --output-dim 256 \
    --num-layers 4
```

### Full Dataset Experiments

```bash
# Full data with 200 epochs
python run_experiment.py \
    --data ../data/preprocessed_data/papers_with_citations.json \
    --output ../results/full_data \
    --epochs 200 \
    --device cuda:0

# Production-quality run
python run_experiment.py \
    --data ../data/preprocessed_data/papers_with_citations.json \
    --output ../results/production \
    --gnn-type gat \
    --hidden-dim 512 \
    --output-dim 256 \
    --num-layers 4 \
    --epochs 300 \
    --lr 0.0005 \
    --device cuda:0
```

### Interactive Mode

```bash
# After training, search interactively
python run_experiment.py \
    --data ../data/preprocessed_data/papers_with_citations_sample.json \
    --output ../results/interactive \
    --interactive
```

---

## 🎛️ All Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--data` | (required) | Path to JSON data file |
| `--output` | `../results/experiment` | Output directory |
| `--epochs` | 100 | Number of training epochs |
| `--hidden-dim` | 256 | Hidden layer dimension |
| `--output-dim` | 128 | Output embedding dimension |
| `--num-layers` | 3 | Number of GNN layers |
| `--gnn-type` | gcn | GNN type: gcn, gat, sage |
| `--lr` | 0.001 | Learning rate |
| `--dropout` | 0.3 | Dropout rate |
| `--device` | auto | Device: cpu or cuda:0 |
| `--no-save-model` | False | Don't save trained model |
| `--no-comparison` | False | Skip GNN vs semantic comparison |
| `--interactive` | False | Run interactive mode after training |
| `--quiet` | False | Reduce output verbosity |

---

## 📊 Expected Output

After running an experiment, you'll see:

```
================================================================================
CITATION GRAPH PAPER RECOMMENDATION SYSTEM
================================================================================

Configuration:
  Data: ../data/preprocessed_data/papers_with_citations_sample.json
  Output: ../results/my_experiment
  GNN Type: gcn
  Layers: 3
  Hidden Dim: 256
  Output Dim: 128
  Epochs: 100
  Device: cuda:0

Step 1: Building citation graph...
  Nodes: 60
  Edges: 105
  Avg citations: 1.75

Step 2: Extracting semantic features...
Extracting embeddings for 60 papers...
100%|██████████| 60/60 [00:15<00:00]

Step 3: Training GNN model...
Epoch 10/100, Loss: 0.4523
Epoch 20/100, Loss: 0.3891
...
Epoch 100/100, Loss: 0.2134
  Initial loss: 0.6234
  Final loss: 0.2134

Step 4: Evaluating on test queries...
Evaluating query: 'deep learning neural networks' (Category: Deep Learning)
  Top-1 Score: 0.8234
  Avg Score: 0.7156
  Top-5 Avg: 0.7543

...

Step 5: Comparing GNN vs Semantic-only...
Query: 'deep learning neural networks'
  GNN+Semantic: 0.7156
  Semantic-only: 0.6523
  Improvement: +9.71%
  Overlap (Top-5): 3/5, (Top-10): 7/10

...

Step 6: Saving results...
Saved detailed results to ../results/my_experiment/results/detailed_results_20241116_123456.json
Saved summary to ../results/my_experiment/results/summary_20241116_123456.json

Step 7: Saving trained model...
Saved GNN model to ../results/my_experiment/models/gnn_model.pt

================================================================================
EXPERIMENT COMPLETED SUCCESSFULLY!
================================================================================
```

---

## 📁 Results Structure

```
results/my_experiment/
├── results/
│   ├── detailed_results_[timestamp].json
│   ├── summary_[timestamp].json
│   ├── comparison_[timestamp].json
│   ├── summary_table_[timestamp].csv
│   └── evaluation_report.txt
├── models/
│   ├── gnn_model.pt
│   ├── gnn_embeddings.npy
│   ├── combined_embeddings.npy
│   ├── model_config.json
│   └── graph_mappings/
└── semantic_cache/
    └── [cached embeddings]
```

---

## 🔍 Checking Results

### View Summary Report

```bash
cat results/my_experiment/results/evaluation_report.txt
```

### View Results Table

```bash
cat results/my_experiment/results/summary_table_*.csv
```

### Load in Python

```python
import json

# Load detailed results
with open('results/my_experiment/results/detailed_results_[timestamp].json') as f:
    results = json.load(f)

# Check first query results
print(results[0]['query'])
print(results[0]['metrics'])
print(results[0]['top_results'][0])
```

---

## 🐛 Troubleshooting

### "CUDA out of memory"

Use smaller model:
```bash
python run_experiment.py --hidden-dim 128 --output-dim 64 --device cuda:0
```

Or use CPU:
```bash
python run_experiment.py --device cpu
```

### "File not found"

Check data path:
```bash
ls ../data/preprocessed_data/papers_with_citations_sample.json
```

### "Import error"

Install dependencies:
```bash
pip install -r requirements.txt
```

### Slow on CPU

Use fewer epochs:
```bash
python run_experiment.py --epochs 50 --device cpu
```

---

## 💡 Tips

1. **Start small**: Use sample data first to verify everything works
2. **Use GPU**: 10-20x faster than CPU
3. **Cache embeddings**: Semantic features are cached automatically
4. **Monitor GPU**: Use `nvidia-smi` to check GPU usage
5. **Save models**: Use saved models for quick inference later

---

## 🎯 Recommended Experiment Sequence

### 1. Validation (2 minutes)
```bash
python quick_test.py
```

### 2. Baseline (5 minutes)
```bash
python run_experiment.py \
    --data ../data/preprocessed_data/papers_with_citations_sample.json \
    --output ../results/baseline \
    --epochs 100
```

### 3. Architecture Comparison (15 minutes)
```bash
# GCN
python run_experiment.py --output ../results/gcn --gnn-type gcn

# GAT
python run_experiment.py --output ../results/gat --gnn-type gat

# SAGE
python run_experiment.py --output ../results/sage --gnn-type sage
```

### 4. Full Dataset (30-60 minutes)
```bash
python run_experiment.py \
    --data ../data/preprocessed_data/papers_with_citations.json \
    --output ../results/full \
    --epochs 200 \
    --device cuda:0
```

### 5. Hyperparameter Tuning (varies)
```bash
# Different layer configurations
python run_experiment.py --num-layers 2 --output ../results/layers_2
python run_experiment.py --num-layers 4 --output ../results/layers_4

# Different hidden dimensions
python run_experiment.py --hidden-dim 128 --output ../results/dim_128
python run_experiment.py --hidden-dim 512 --output ../results/dim_512
```

---

## 📞 Need Help?

1. Check `EXPERIMENT_README.md` for detailed documentation
2. Review `IMPLEMENTATION_SUMMARY.md` for architecture details
3. Ensure GPU drivers are installed (for CUDA)
4. Verify Python version (3.8+ required)

---

**Ready to start!** Run `python quick_test.py` to begin.