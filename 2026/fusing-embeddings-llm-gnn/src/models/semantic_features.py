# %%-------------------------------------------------------------------------------------------------
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from tqdm import tqdm
from .llm import LLM
# %%-------------------------------------------------------------------------------------------------
@dataclass
class SemanticFeature:
    paper_id: str
    title_embedding: List[float]
    abstract_embedding: List[float]
    combined_embedding: List[float]
    metadata: Dict[str, any] = None

    def to_dict(self) -> Dict:
        return asdict(self)

    @staticmethod
    def from_dict(data: Dict) -> "SemanticFeature":
        return SemanticFeature(**data)

    def get_embedding_dim(self) -> int:
        return len(self.combined_embedding)
# %%-------------------------------------------------------------------------------------------------
class SemanticFeatureExtractor:
    def __init__(self,llm_config_path: Optional[str] = None,combine_strategy: str = "weighted_average",title_weight: float = 0.3,abstract_weight: float = 0.7,cache_dir: Optional[str] = None):
        self.llm = LLM(config_path=llm_config_path)
        self.combine_strategy = combine_strategy
        self.title_weight = title_weight
        self.abstract_weight = abstract_weight
        self.cache_dir = Path(cache_dir) if cache_dir else None

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        if combine_strategy == "weighted_average":
            assert abs(title_weight + abstract_weight - 1.0) < 1e-6, "Weights must sum to 1.0 for weighted_average strategy"

    def extract_single(self,paper_id: str,title: str,abstract: str,metadata: Optional[Dict] = None) -> SemanticFeature:
        title_emb = self._get_embedding(title, prefix="Title: ")
        abstract_emb = self._get_embedding(abstract, prefix="Abstract: ")
        combined_emb = self._combine_embeddings(title_emb, abstract_emb)

        return SemanticFeature(paper_id=paper_id,title_embedding=title_emb,abstract_embedding=abstract_emb,combined_embedding=combined_emb,metadata=metadata or {})

    def extract_batch(self,papers: List[Dict[str, str]],show_progress: bool = True) -> List[SemanticFeature]:
        features = []
        iterator = tqdm(papers, desc="Extracting semantic features") if show_progress else papers

        for paper in iterator:
            if self.cache_dir:
                cached = self._load_from_cache(paper['id'])
                if cached is not None:
                    features.append(cached)
                    continue

            metadata = {k: v for k, v in paper.items() if k not in ['id', 'title', 'abstract']}
            feature = self.extract_single(paper_id=paper['id'], title=paper.get('title', ''),abstract=paper.get('abstract', ''),metadata=metadata)
            if self.cache_dir:
                self._save_to_cache(feature)
            features.append(feature)
        return features

    def extract_from_dataframe(self,df: pd.DataFrame,id_col: str = 'id',title_col: str = 'title',abstract_col: str = 'abstract',show_progress: bool = True) -> List[SemanticFeature]:
        papers = []
        for _, row in df.iterrows():
            paper = {
                'id': str(row[id_col]),
                'title': str(row[title_col]) if pd.notna(row[title_col]) else '',
                'abstract': str(row[abstract_col]) if pd.notna(row[abstract_col]) else ''
            }
            for col in df.columns:
                if col not in [id_col, title_col, abstract_col]:
                    paper[col] = row[col]
            papers.append(paper)

        return self.extract_batch(papers, show_progress=show_progress)

    def save_features(self,features: List[SemanticFeature],output_path: Union[str, Path],format: str = "json") -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "json":
            data = [f.to_dict() for f in features]
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)

        elif format == "npy":
            data = {
                'paper_ids': [f.paper_id for f in features],
                'title_embeddings': np.array([f.title_embedding for f in features]),
                'abstract_embeddings': np.array([f.abstract_embedding for f in features]),
                'combined_embeddings': np.array([f.combined_embedding for f in features]),
                'metadata': [f.metadata for f in features]
            }
            np.save(output_path, data)

        else:
            raise ValueError(f"Unsupported format: {format}")

    def load_features(self,input_path: Union[str, Path],format: str = "json") -> List[SemanticFeature]:
        input_path = Path(input_path)

        if format == "json":
            with open(input_path, 'r') as f:
                data = json.load(f)
            return [SemanticFeature.from_dict(d) for d in data]

        elif format == "npy":
            data = np.load(input_path, allow_pickle=True).item()
            features = []
            for i in range(len(data['paper_ids'])):
                features.append(SemanticFeature(
                    paper_id=data['paper_ids'][i],
                    title_embedding=data['title_embeddings'][i].tolist(),
                    abstract_embedding=data['abstract_embeddings'][i].tolist(),
                    combined_embedding=data['combined_embeddings'][i].tolist(),
                    metadata=data['metadata'][i]
                ))
            return features

        else:
            raise ValueError(f"Unsupported format: {format}")

    def get_feature_matrix(self,features: List[SemanticFeature],embedding_type: str = "combined") -> Tuple[np.ndarray, List[str]]:
        if embedding_type == "title":
            embeddings = [f.title_embedding for f in features]
        elif embedding_type == "abstract":
            embeddings = [f.abstract_embedding for f in features]
        elif embedding_type == "combined":
            embeddings = [f.combined_embedding for f in features]
        else:
            raise ValueError(f"Invalid embedding_type: {embedding_type}")

        paper_ids = [f.paper_id for f in features]
        matrix = np.array(embeddings)

        return matrix, paper_ids

    def compute_similarity(self,feature1: SemanticFeature,feature2: SemanticFeature,embedding_type: str = "combined",metric: str = "cosine") -> float:
        if embedding_type == "title":
            emb1, emb2 = feature1.title_embedding, feature2.title_embedding
        elif embedding_type == "abstract":
            emb1, emb2 = feature1.abstract_embedding, feature2.abstract_embedding
        else:
            emb1, emb2 = feature1.combined_embedding, feature2.combined_embedding

        vec1 = np.array(emb1)
        vec2 = np.array(emb2)

        if metric == "cosine":
            return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
        elif metric == "euclidean":
            return float(-np.linalg.norm(vec1 - vec2))  # Negative for similarity
        else:
            raise ValueError(f"Unknown metric: {metric}")


    def _get_embedding(self, text: str, prefix: str = "") -> List[float]:
        if not text or not text.strip():
            dummy_emb = self.llm.get_embedding("placeholder")
            return [0.0] * len(dummy_emb)

        full_text = f"{prefix}{text}" if prefix else text
        return self.llm.get_embedding(full_text)

    def _combine_embeddings(self,title_emb: List[float],abstract_emb: List[float]) -> List[float]:
        title_vec = np.array(title_emb)
        abstract_vec = np.array(abstract_emb)

        if self.combine_strategy == "weighted_average":
            combined = self.title_weight * title_vec + self.abstract_weight * abstract_vec
            combined = combined / np.linalg.norm(combined)
            return combined.tolist()

        elif self.combine_strategy == "concatenate":
            return np.concatenate([title_vec, abstract_vec]).tolist()

        elif self.combine_strategy == "max_pool":
            return np.maximum(title_vec, abstract_vec).tolist()

        else:
            raise ValueError(f"Unknown combine strategy: {self.combine_strategy}")

    def _get_cache_path(self, paper_id: str) -> Path:
        safe_id = paper_id.replace('/', '_').replace('\\', '_')
        return self.cache_dir / f"{safe_id}.json"

    def _save_to_cache(self, feature: SemanticFeature) -> None:
        if not self.cache_dir:
            return
        cache_path = self._get_cache_path(feature.paper_id)
        with open(cache_path, 'w') as f:
            json.dump(feature.to_dict(), f)

    def _load_from_cache(self, paper_id: str) -> Optional[SemanticFeature]:
        if not self.cache_dir:
            return None
        cache_path = self._get_cache_path(paper_id)
        if cache_path.exists():
            with open(cache_path, 'r') as f:
                data = json.load(f)
            return SemanticFeature.from_dict(data)
        return None

# %%-------------------------------------------------------------------------------------------------

def create_feature_extractor(model_name: Optional[str] = None, device: str = "cuda:0",**kwargs) -> SemanticFeatureExtractor:
    if model_name:
        print(f"Note: To use custom model '{model_name}', update the config.json file")

    return SemanticFeatureExtractor(**kwargs)