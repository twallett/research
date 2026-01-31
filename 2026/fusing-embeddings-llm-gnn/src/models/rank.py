import numpy as np
import torch
import warnings

# --- Ranking index with optional Faiss ---
# try:
#     import faiss
#     _HAS_FAISS = True
# except Exception:
#     _HAS_FAISS = False
#     faiss = None
#     warnings.warn("faiss not available — falling back to numpy brute-force search (slower).")


# --- For testing functionality ---
_HAS_FAISS = False
faiss = None

class MIPSKNNIndex:
    """
    Maximum Inner Product Search (MIPS) index wrapper.
    Uses Faiss if available, otherwise falls back to brute-force.
    Expect input embeddings to be L2-normalized for cosine=MIPS equivalence.
    """

    def __init__(self, dim: int):
        self.dim = dim
        self.use_faiss = _HAS_FAISS
        if self.use_faiss:
            # Faiss IndexFlatIP uses inner-product
            self.index = faiss.IndexFlatIP(dim)
            self._xb = None
        else:
            # fallback store
            self._xb = None

    def add(self, embeddings: np.ndarray):
        xb = embeddings.astype(np.float32)
        assert xb.ndim == 2 and xb.shape[1] == self.dim
        if self.use_faiss:
            self.index.add(xb)
            self._xb = xb
        else:
            # store for brute-force
            if self._xb is None:
                self._xb = xb.copy()
            else:
                self._xb = np.vstack([self._xb, xb])

    def search(self, query: np.ndarray, k: int):
        """
        query: (n, dim) or (dim,)
        returns: (scores, indices) where scores shape (n, k) and indices shape (n, k)
        """
        q = np.array(query, dtype=np.float32)
        if q.ndim == 1:
            q = q.reshape(1, -1)
        assert q.shape[1] == self.dim

        if self.use_faiss:
            scores, idx = self.index.search(q, k)
            return scores, idx
        else:
            # brute-force inner product
            xb = self._xb  # (N, dim)
            # compute q @ xb.T -> shape (n, N)
            sims = q.dot(xb.T)
            # partial sort
            idx = np.argpartition(-sims, range(min(k, sims.shape[1])), axis=1)[:, :k]
            # sort those k
            row_idx = np.arange(sims.shape[0])[:, None]
            sorted_order = np.argsort(-sims[row_idx, idx], axis=1)
            sorted_idx = idx[row_idx, sorted_order]
            sorted_scores = sims[row_idx, sorted_idx]
            return sorted_scores, sorted_idx


class Ranker:
    """
    Wrapper that builds a MIPS index and exposes rank() method.
    Works with either faiss or numpy fallback.
    """

    def __init__(self, embeddings: np.ndarray):
        # normalize embeddings to unit length (cosine similarity)
        eps = 1e-8
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + eps
        self.embeddings = embeddings.astype(np.float32) / norms
        dim = self.embeddings.shape[1]
        self.index = MIPSKNNIndex(dim)
        self.index.add(self.embeddings)

    def rank(self, query_embedding: np.ndarray, k: int):
        """
        query_embedding: (dim,) or (1, dim) or (n, dim)
        returns: (scores, indices) for the first query (if n>1, returns for first)
        """
        q = np.array(query_embedding, dtype=np.float32)
        if q.ndim == 1:
            q = q.reshape(1, -1)
        # normalize
        q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-8)
        scores, indices = self.index.search(q, k)
        return scores[0], indices[0]


# --- Ranking metrics (unchanged except small robustness updates) ---
class RankingMetrics:
    """
    Metrics for citation-based ranking quality:
    - Precision@K
    - Recall@K
    - nDCG@K
    """

    @staticmethod
    def ndcg_at_k(relevance_scores, k):
        rel = np.array(relevance_scores)[:k]
        if rel.size == 0 or rel.sum() == 0:
            return 0.0
        discounts = np.log2(np.arange(2, rel.size + 2))
        dcg = np.sum(rel / discounts)
        ideal = np.sum(sorted(rel, reverse=True) / discounts)
        return float(dcg / ideal) if ideal > 0 else 0.0

    @staticmethod
    def precision_at_k(relevance_scores, k):
        rel = np.array(relevance_scores)[:k]
        if rel.size == 0:
            return 0.0
        return float(rel.sum() / k)

    @staticmethod
    def recall_at_k(relevance_scores, k):
        rel = np.array(relevance_scores)
        total_relevant = rel.sum()
        if total_relevant == 0:
            return 0.0
        return float(rel[:k].sum() / total_relevant)

    def evaluate(self, edge_index, rankings, K=10):
        """
        edge_index: torch.LongTensor [2, num_edges]
        rankings: dict {node -> ranked list of neighbors}
        """
        if isinstance(edge_index, torch.Tensor):
            src = edge_index[0].cpu().numpy()
            dst = edge_index[1].cpu().numpy()
        else:
            src, dst = edge_index

        true_adj = {}
        for s, d in zip(src, dst):
            true_adj.setdefault(int(s), set()).add(int(d))

        precision_list = []
        recall_list = []
        ndcg_list = []

        for node, ranked_list in rankings.items():
            if node not in true_adj:
                continue
            true_neighbors = true_adj[node]
            relevance = [1 if int(r) in true_neighbors else 0 for r in ranked_list]
            precision_list.append(self.precision_at_k(relevance, K))
            recall_list.append(self.recall_at_k(relevance, K))
            ndcg_list.append(self.ndcg_at_k(relevance, K))

        # return averaged metrics (or 0 if empty)
        return {
            "precision@K": float(np.mean(precision_list)).__round__(2) if precision_list else 0.0,
            "recall@K": float(np.mean(recall_list)).__round__(2) if recall_list else 0.0,
            "nDCG@K": float(np.mean(ndcg_list)).__round__(2) if ndcg_list else 0.0
        }