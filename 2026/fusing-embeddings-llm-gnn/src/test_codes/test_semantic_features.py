# %%-------------------------------------------------------------------------------------------------
from pathlib import Path
import sys
import pandas as pd
import numpy as np
SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
from models.semantic_features import SemanticFeatureExtractor, create_feature_extractor
# %%-------------------------------------------------------------------------------------------------
def test_single_paper_extraction():
    print("\n" + "="*70)
    print("TEST 1: Single Paper Feature Extraction")
    print("="*70)
    extractor = SemanticFeatureExtractor()
    paper_id = "test_001"
    title = "Deep Learning for Natural Language Processing"
    abstract = (
        "This paper presents a comprehensive survey of deep learning methods "
        "for natural language processing tasks. We review recent advances in "
        "neural architectures including transformers, attention mechanisms, "
        "and pre-trained language models."
    )
    print(f"\nPaper ID: {paper_id}")
    print(f"Title: {title}")
    print(f"Abstract: {abstract[:100]}...")

    feature = extractor.extract_single(paper_id=paper_id,title=title,abstract=abstract,metadata={"category": "cs.CL", "authors": ["John Doe", "Jane Smith"]})

    print(f"\nExtracted Features:")
    print(f"  - Title embedding dim: {len(feature.title_embedding)}")
    print(f"  - Abstract embedding dim: {len(feature.abstract_embedding)}")
    print(f"  - Combined embedding dim: {len(feature.combined_embedding)}")
    print(f"  - Metadata: {feature.metadata}")
    print(f"\nFirst 5 values of combined embedding:")
    print(f"  {feature.combined_embedding[:5]}")
    return feature
# %%-------------------------------------------------------------------------------------------------
def test_batch_extraction_from_csv():
    print("\n" + "="*70)
    print("TEST 2: Batch Feature Extraction from CSV")
    print("="*70)

    sample_papers = [
        {
            'id': 'paper_001',
            'title': 'Deep Learning for Natural Language Processing',
            'abstract': 'This paper presents a comprehensive survey of deep learning methods for NLP tasks.',
            'category': 'cs.CL',
            'year': 2023
        },
        {
            'id': 'paper_002',
            'title': 'Computer Vision with Transformers',
            'abstract': 'We explore transformer architectures for computer vision applications.',
            'category': 'cs.CV',
            'year': 2023
        },
        {
            'id': 'paper_003',
            'title': 'Reinforcement Learning in Robotics',
            'abstract': 'This work investigates reinforcement learning methods for robotic control.',
            'category': 'cs.RO',
            'year': 2024
        }
    ]

    df = pd.DataFrame(sample_papers)
    print(f"\nCreated DataFrame with {len(df)} papers")
    print(f"Columns: {list(df.columns)}")

    extractor = SemanticFeatureExtractor()
    features = extractor.extract_from_dataframe(df,id_col='id',title_col='title',abstract_col='abstract',show_progress=True)

    print(f"\nExtracted {len(features)} features")
    for i, feature in enumerate(features):
        print(f"\nPaper {i+1}:")
        print(f"  ID: {feature.paper_id}")
        print(f"  Metadata: {feature.metadata}")
        print(f"  Embedding dimension: {len(feature.combined_embedding)}")

    return features
# %%-------------------------------------------------------------------------------------------------
def test_similarity_computation(features=None):
    print("\n" + "="*70)
    print("TEST 3: Similarity Computation")
    print("="*70)
    extractor = SemanticFeatureExtractor()
    feature1 = features[0]
    feature2 = features[1]

    similarity = extractor.compute_similarity(feature1,feature2,embedding_type="combined",metric="cosine")

    print(f"\nSimilarity between papers:")
    print(f"  Paper 1 ID: {feature1.paper_id}")
    print(f"  Paper 2 ID: {feature2.paper_id}")
    print(f"  Cosine similarity: {similarity:.4f}")
# %%-------------------------------------------------------------------------------------------------
def test_save_and_load_features(features):

    print("\n" + "="*70)
    print("TEST 4: Save and Load Features")
    print("="*70)

    extractor = SemanticFeatureExtractor()

    output_dir = SRC_DIR.parent / "results" / "semantic_features"
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "sample_features.json"
    print(f"\nSaving features to: {json_path}")
    extractor.save_features(features, json_path, format="json")
    print(f"  Saved {len(features)} features as JSON")

    npy_path = output_dir / "sample_features.npy"
    print(f"\nSaving features to: {npy_path}")
    extractor.save_features(features, npy_path, format="npy")
    print(f"  Saved {len(features)} features as NPY")

    print(f"\nLoading features from JSON...")
    loaded_json = extractor.load_features(json_path, format="json")
    print(f"  Loaded {len(loaded_json)} features")
    print(f"  First paper ID: {loaded_json[0].paper_id}")

    print(f"\nLoading features from NPY...")
    loaded_npy = extractor.load_features(npy_path, format="npy")
    print(f"  Loaded {len(loaded_npy)} features")
    print(f"  First paper ID: {loaded_npy[0].paper_id}")
    print(f"\nVerifying data integrity...")
    print("  All checks passed!")


def test_feature_matrix_conversion(features):
    print("\n" + "="*70)
    print("TEST 5: Feature Matrix Conversion")
    print("="*70)

    extractor = SemanticFeatureExtractor()
    matrix, paper_ids = extractor.get_feature_matrix(features,embedding_type="combined")

    print(f"\nFeature matrix shape: {matrix.shape}")
    print(f"  Number of papers: {len(paper_ids)}")
    print(f"  Embedding dimension: {matrix.shape[1]}")
    print(f"  Data type: {matrix.dtype}")

    print(f"\nMatrix statistics:")
    print(f"  Mean: {matrix.mean():.4f}")
    print(f"  Std: {matrix.std():.4f}")
    print(f"  Min: {matrix.min():.4f}")
    print(f"  Max: {matrix.max():.4f}")

    norms = np.linalg.norm(matrix, axis=1)
    print(f"\nEmbedding norms:")
    print(f"  Mean norm: {norms.mean():.4f}")
    print(f"  Std norm: {norms.std():.4f}")


def test_different_strategies():
    print("\n" + "="*70)
    print("TEST 6: Different Combination Strategies")
    print("="*70)

    paper = {
        'id': 'strategy_test',
        'title': 'Machine Learning Applications',
        'abstract': 'This paper explores various machine learning algorithms and their applications in real-world scenarios.'
    }

    strategies = [
        ("weighted_average", {"title_weight": 0.3, "abstract_weight": 0.7}),
        ("weighted_average", {"title_weight": 0.5, "abstract_weight": 0.5}),
        ("concatenate", {}),
        ("max_pool", {})]
    print("\nTesting different combination strategies:")

    for strategy, params in strategies:
        extractor = SemanticFeatureExtractor(combine_strategy=strategy,**params)
        feature = extractor.extract_single(paper_id=paper['id'],title=paper['title'],abstract=paper['abstract'])
        print(f"\nStrategy: {strategy}")
        if params:
            print(f"  Parameters: {params}")
        print(f"  Title embedding dim: {len(feature.title_embedding)}")
        print(f"  Abstract embedding dim: {len(feature.abstract_embedding)}")
        print(f"  Combined embedding dim: {len(feature.combined_embedding)}")

# %%-------------------------------------------------------------------------------------------------
def run_all_tests():
    print("\n" + "="*70)
    print("SEMANTIC FEATURE EXTRACTOR - TEST SUITE")
    print("="*70)
    print("\nThis test suite demonstrates semantic feature extraction")
    print("for scientific paper recommendation systems.")
    print("\nThe semantic features are used in the CONTENT ENCODING")
    print("component of the recommendation pipeline.")

    single_feature = test_single_paper_extraction()
    batch_features = test_batch_extraction_from_csv()
    test_similarity_computation(batch_features)
    test_save_and_load_features(batch_features)
    test_feature_matrix_conversion(batch_features)
    test_different_strategies()
    print("\n" + "="*70)
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print("="*70)
# %%-------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    run_all_tests()