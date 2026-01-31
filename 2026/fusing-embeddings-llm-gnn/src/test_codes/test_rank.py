from pathlib import Path
import sys

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import torch

# import everything from your rank_test.py
from models.rank import Ranker, RankingMetrics, MIPSKNNIndex

def test_ranker_basic():
    print("\n=== Test: Ranker Basic Functionality ===")

    # Create simple 3D embeddings
    embeddings = np.array([
        [1, 0, 0],   # vector 0
        [0, 1, 0],   # vector 1
        [1, 1, 0],   # vector 2 (closest to both)
        [0, 0, 1],   # vector 3
    ], dtype=np.float32)

    ranker = Ranker(embeddings)

    query = np.array([1, 1, 0], dtype=np.float32)  # similar to index 2

    scores, idx = ranker.rank(query, k=3)

    print("scores: ", scores)
    print("indices:", idx)

    assert idx[0] == 2, "Query should rank embedding 2 first"
    print("OK ✓")


def test_mips_index_faiss_or_numpy():
    print("\n=== Test: MIPSKNNIndex (Faiss or Numpy Fallback) ===")

    emb = np.random.randn(5, 4).astype(np.float32)
    idx = MIPSKNNIndex(dim=4)
    idx.add(emb)

    q = emb[0]  # exact match → inner product highest
    scores, indices = idx.search(q, k=2)

    print("scores:", scores)
    print("indices:", indices)

    assert int(indices[0][0]) == 0, "Closest vector should be itself"
    print("OK ✓")


def test_ranking_metrics():
    print("\n=== Test: Ranking Metrics ===")

    # True edges: node 0 -> neighbors {1,2}
    edge_index = torch.tensor([
        [0, 0, 1],    # source
        [1, 2, 3]     # destination
    ])

    # Ranked list for node 0
    rankings = {
        0: [1, 5, 2, 4, 3]  # relevant at positions 0 and 2
    }

    metrics = RankingMetrics()
    results = metrics.evaluate(edge_index, rankings, K=3)

    print(results)

    # Precision@3 = 2 relevant / 3 = 0.67
    assert abs(results["precision@K"] - 0.67) < 0.05
    # Recall@3 = 2 relevant / 2 = 1.0
    assert abs(results["recall@K"] - 1.0) < 0.01
    # nDCG should be > 0.7
    assert results["nDCG@K"] > 0.7

    print("OK ✓")


if __name__ == "__main__":
    test_ranker_basic()
    test_mips_index_faiss_or_numpy()
    test_ranking_metrics()

    print("\nAll tests passed!")
