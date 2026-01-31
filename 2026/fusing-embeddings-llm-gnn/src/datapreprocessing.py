"""
Data Preprocessing and Citation Graph Building Pipeline
"""

import json
import argparse
import pandas as pd
import requests
import time
import re
import nltk
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Semantic Scholar API configuration
SEMANTIC_SCHOLAR_API = "https://api.semanticscholar.org/graph/v1/paper"
API_KEY = "emOBijoTaC4arDO7AWCAIa8ZvShT0kOE6mpgq9dx"


class DataPreprocessor:
    """Class for preprocessing arXiv data: filtering, sampling, and text cleaning."""
    
    def __init__(self, input_path: str):
        """
        Initialize the data preprocessor.
        
        Args:
            input_path: Path to the raw arXiv metadata JSON file
        """
        self.input_path = Path(input_path)
        self.df = None
        
    def filter_and_sample(self, categories: List[str] = None, n_samples: int = 10000, 
                         random_state: int = 42, require_doi: bool = True) -> pd.DataFrame:
        """
        Filter arXiv papers by categories and sample a subset.
        
        Args:
            categories: List of category prefixes to filter (e.g., ['cs.', 'stat.'])
            n_samples: Number of papers to sample
            random_state: Random seed for reproducibility
            require_doi: Whether to require papers to have a DOI
            
        Returns:
            DataFrame with filtered and sampled papers
        """
        if categories is None:
            categories = ['cs.', 'stat.']
            
        print(f"Reading data from {self.input_path}...")
        records = []
        
        with open(self.input_path, 'r') as f:
            for line in f:
                paper = json.loads(line)
                paper_categories = paper.get("categories", "")
                
                # Filter by categories
                if any(cat.startswith(tuple(categories)) for cat in paper_categories.split()):
                    # Check DOI requirement
                    if not require_doi or paper.get('doi') is not None:
                        records.append({
                            'id': paper['id'],
                            'title': paper['title'],
                            'abstract': paper['abstract'],
                            'categories': paper['categories'],
                            'authors': paper['authors'],
                            'comments': paper.get('comments', ''),
                            'journal-ref': paper.get('journal-ref', ''),
                            'doi': paper.get('doi', ''),
                        })
        
        # Convert to DataFrame
        self.df = pd.DataFrame(records)
        print(f'Initial length of data: {len(self.df)}')
        
        # Drop rows with missing abstracts
        self.df.dropna(subset=['abstract'], inplace=True)
        print(f'Length after dropping missing abstracts: {len(self.df)}')
        
        # Sample if requested
        if n_samples > 0 and n_samples < len(self.df):
            self.df = self.df.sample(n=n_samples, random_state=random_state).reset_index(drop=True)
            print(f'Sampled {n_samples} papers')
        else:
            print(f'Using all {len(self.df)} papers (no sampling)')
        
        return self.df
    
    def preprocess_text(self) -> pd.DataFrame:
        """
        Preprocess text by cleaning, tokenizing, and lemmatizing.
        
        Returns:
            DataFrame with added 'clean_text' column
        """
        if self.df is None:
            raise ValueError("No data loaded. Run filter_and_sample() first.")
        
        print("Preprocessing text...")
        
        # Download NLTK resources if needed
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords', quiet=True)
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet', quiet=True)
        
        # Combine title and abstract
        self.df["text"] = self.df["title"].fillna('') + ". " + self.df["abstract"].fillna('')
        
        # Initialize preprocessing tools
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()
        
        def preprocess(text: str) -> str:
            """Preprocess a single text string."""
            if not isinstance(text, str):
                return ""
            
            # Tokenize
            tokens = word_tokenize(text.lower())
            
            # Remove stop words and non-alphabetic tokens
            tokens = [token for token in tokens 
                     if token.isalpha() and token not in stop_words and len(token) > 2]
            
            # Lemmatize
            tokens = [lemmatizer.lemmatize(word) for word in tokens]
            
            return " ".join(tokens)
        
        # Apply preprocessing
        self.df["clean_text"] = self.df["text"].apply(preprocess)
        print("Text preprocessing completed")
        
        return self.df
    
    def save(self, output_path: str) -> None:
        """
        Save the preprocessed DataFrame to CSV.
        
        Args:
            output_path: Path to save the CSV file
        """
        if self.df is None:
            raise ValueError("No data to save. Run preprocessing steps first.")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.df.to_csv(output_path, index=False)
        print(f'Preprocessed data saved to {output_path}')


class CitationGraphBuilder:
    """Class for building citation graphs from Semantic Scholar API."""
    
    def __init__(self, api_key: str = None):
        """
        Initialize the citation graph builder.
        
        Args:
            api_key: Semantic Scholar API key (optional)
        """
        self.api_key = api_key or API_KEY
        self.api_base = SEMANTIC_SCHOLAR_API
        
    def _get_headers(self) -> Dict:
        """Get HTTP headers for API requests."""
        if self.api_key:
            return {"x-api-key": self.api_key}
        else:
            return {'User-Agent': 'Citation_Graphs/1.0 (your-email@example.com)'}
    
    def get_paper_details(self, paper_id: str, max_retries: int = 3) -> Dict:
        """
        Fetch paper details from Semantic Scholar API.
        
        Args:
            paper_id: arXiv ID of the paper
            max_retries: Maximum number of retry attempts
            
        Returns:
            Dictionary with paper details
        """
        url = f"{self.api_base}/ARXIV:{paper_id}"
        params = {
            'fields': 'paperId,title,authors,externalIds,venue,fieldsOfStudy,referenceCount,citationCount,publicationDate,journal,abstract,year,citationCount,influentialCitationCount'
        }
        headers = self._get_headers()
        
        for attempt in range(max_retries):
            try:
                time.sleep(0.5)
                response = requests.get(url, params=params, headers=headers, timeout=10)
                
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 404:
                    print(f"Paper arXiv:{paper_id} not found in Semantic Scholar")
                    return {}
                elif response.status_code == 400:
                    print(f"Bad request (400) for paper arXiv:{paper_id}")
                    return {}
                elif response.status_code == 429:
                    if attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 3
                        print(f"Rate limited (429) for arXiv:{paper_id}. Waiting {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                    else:
                        print(f"Error 429 (rate limited) for paper arXiv:{paper_id} after {max_retries} attempts")
                        return {}
                else:
                    print(f"Error {response.status_code} for paper arXiv:{paper_id}")
                    return {}
                    
            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    print(f"Request exception for arXiv:{paper_id}: {e}. Retrying...")
                    time.sleep(5)
                    continue
                else:
                    print(f"Request exception for arXiv:{paper_id}: {e}")
                    return {}
            except Exception as e:
                print(f"Unexpected error for arXiv:{paper_id}: {e}")
                return {}
        
        return {}
    
    def get_relations(self, paper_id: str, doi_id: str, endpoint_type: str = 'citations', 
                     max_relations: int = 1000, max_retries: int = 3) -> List[Dict]:
        """
        Fetch citations or references from Semantic Scholar API.
        
        Args:
            paper_id: arXiv ID of the paper
            doi_id: DOI of the paper
            endpoint_type: 'citations' or 'references'
            max_relations: Maximum number of relations to retrieve
            max_retries: Maximum number of retry attempts
            
        Returns:
            List of dictionaries with related paper details
        """
        if endpoint_type not in ['citations', 'references']:
            raise ValueError(f"endpoint_type must be 'citations' or 'references', got '{endpoint_type}'")
        
        url = f"{self.api_base}/ARXIV:{paper_id}/{endpoint_type}"
        params = {
            'fields': 'paperId,title,authors,abstract,externalIds,venue,fieldsOfStudy,referenceCount,citationCount,publicationDate,journal,year,influentialCitationCount',
            'limit': min(max_relations, 10000)
        }
        headers = self._get_headers()
        paper_doi = doi_id.lower() if doi_id else None
        
        for attempt in range(max_retries):
            try:
                time.sleep(0.5)
                response = requests.get(url, params=params, headers=headers, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    relation_details = []
                    
                    for relation in data.get('data', []):
                        if endpoint_type == 'citations':
                            related_paper = relation.get('citingPaper', {})
                        else:
                            related_paper = relation.get('citedPaper', {})
                        
                        external_ids = related_paper.get('externalIds', {})
                        doi = external_ids.get('DOI', None) if external_ids else None
                        if doi:
                            doi = doi.lower()
                        
                        paper_info = {
                            'paperId': related_paper.get('paperId'),
                            'title': related_paper.get('title'),
                            'authors': [{'authorId': author.get('authorId'), 'name': author.get('name')} 
                                      for author in related_paper.get('authors', [])],
                            'externalIds': external_ids,
                            'doi': doi,
                            'venue': related_paper.get('venue'),
                            'journal': related_paper.get('journal'),
                            'fieldsOfStudy': related_paper.get('fieldsOfStudy'),
                            'referenceCount': related_paper.get('referenceCount'),
                            'citationCount': related_paper.get('citationCount'),
                            'influentialCitationCount': related_paper.get('influentialCitationCount'),
                            'publicationDate': related_paper.get('publicationDate'),
                            'year': related_paper.get('year'),
                            'abstract': related_paper.get('abstract')
                        }
                        
                        if endpoint_type == 'citations':
                            paper_info['referenced_doi'] = paper_doi
                        else:
                            paper_info['paper_doi'] = paper_doi
                        
                        relation_details.append(paper_info)
                    
                    return relation_details
                    
                elif response.status_code == 404:
                    print(f"Relations - Paper arXiv:{paper_id} not found")
                    return []
                elif response.status_code == 400:
                    print(f"Relations - Bad request (400) for paper arXiv:{paper_id}")
                    return []
                elif response.status_code == 429:
                    if attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 3
                        print(f"Relations - Rate limited (429) for arXiv:{paper_id}. Waiting {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                    else:
                        print(f"Relations - Error 429 for paper arXiv:{paper_id} after {max_retries} attempts")
                        return []
                else:
                    print(f"Error {response.status_code} for paper arXiv:{paper_id}")
                    return []
                    
            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    print(f"Request exception for arXiv:{paper_id}: {e}. Retrying...")
                    time.sleep(5)
                    continue
                else:
                    print(f"Request exception for arXiv:{paper_id}: {e}")
                    return []
            except Exception as e:
                print(f"Unexpected error for arXiv:{paper_id}: {e}")
                return []
        
        return []
    
    def get_paper_with_relations(self, paper_id: str, doi_id: str = None, 
                                endpoint_type: str = 'citations', max_retries: int = 3) -> Dict:
        """
        Get paper details and its citations/references.
        
        Args:
            paper_id: arXiv ID of the paper
            doi_id: DOI of the paper (optional)
            endpoint_type: 'citations' or 'references'
            max_retries: Maximum number of retry attempts
            
        Returns:
            Dictionary containing paper details and relations
        """
        if endpoint_type not in ['citations', 'references']:
            raise ValueError(f"endpoint_type must be 'citations' or 'references', got '{endpoint_type}'")
        
        # Get paper details
        paper_details = self.get_paper_details(paper_id, max_retries)
        if not paper_details:
            return {}
        
        # Extract DOI if not provided
        if not doi_id and paper_details.get('externalIds'):
            doi_id = paper_details.get('externalIds', {}).get('DOI')
        
        # Get relations
        relations = self.get_relations(paper_id, doi_id, endpoint_type=endpoint_type, max_retries=max_retries)
        
        # Set relation keys
        if endpoint_type == 'citations':
            relation_key = 'citations'
            relation_count_key = 'citation_count'
        else:
            relation_key = 'references'
            relation_count_key = 'reference_count'
        
        # Format paper details
        doi_from_ext_ids = paper_details.get('externalIds', {}).get('DOI')
        doi_lower = doi_from_ext_ids.lower() if doi_from_ext_ids else None
        
        formatted_paper_details = {
            'paperId': paper_details.get('paperId'),
            'title': paper_details.get('title'),
            'authors': [{'authorId': author.get('authorId'), 'name': author.get('name')} 
                       for author in paper_details.get('authors', [])],
            'externalIds': paper_details.get('externalIds'),
            'doi': doi_lower,
            'venue': paper_details.get('venue'),
            'journal': paper_details.get('journal'),
            'fieldsOfStudy': paper_details.get('fieldsOfStudy'),
            'referenceCount': paper_details.get('referenceCount'),
            'citationCount': paper_details.get('citationCount'),
            'influentialCitationCount': paper_details.get('influentialCitationCount'),
            'publicationDate': paper_details.get('publicationDate'),
            'year': paper_details.get('year'),
            'abstract': paper_details.get('abstract')
        }
        
        # Combine data
        combined_data = {
            'paper_id': paper_id,
            'paper_details': formatted_paper_details,
            relation_key: relations,
            relation_count_key: len(relations),
            'endpoint_type': endpoint_type
        }
        
        return combined_data
    
    def build_papers_with_relations_json(self, df: pd.DataFrame, endpoint_type: str = 'citations',
                                        delay: float = 1.0, output_path: str = None) -> None:
        """
        Build JSON file with papers and their citations/references.
        
        Args:
            df: DataFrame containing papers
            endpoint_type: 'citations' or 'references'
            delay: Delay in seconds between requests
            output_path: Path to save the JSON file
        """
        if endpoint_type not in ['citations', 'references']:
            raise ValueError(f"endpoint_type must be 'citations' or 'references', got '{endpoint_type}'")
        
        if output_path is None:
            if endpoint_type == 'citations':
                output_path = '../data/preprocessed_data/papers_with_citations.json'
            else:
                output_path = '../data/preprocessed_data/papers_with_references.json'
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        all_papers_data = []
        total_papers = len(df)
        
        desc_text = f'Fetching papers with {endpoint_type}'
        for idx, row in tqdm(df.iterrows(), total=total_papers, desc=desc_text):
            paper_id = row['id']
            doi_id = row.get('doi', None)
            
            paper_data = self.get_paper_with_relations(paper_id, doi_id, endpoint_type=endpoint_type)
            if paper_data:
                all_papers_data.append(paper_data)
            
            time.sleep(delay)
        
        # Save to JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_papers_data, f, indent=2, ensure_ascii=False)
        
        # Print statistics
        if endpoint_type == 'citations':
            relation_count_key = 'citation_count'
            relation_key = 'citations'
        else:
            relation_count_key = 'reference_count'
            relation_key = 'references'
        
        total_relations = sum(paper.get(relation_count_key, 0) for paper in all_papers_data)
        papers_with_relations = sum(1 for paper in all_papers_data if paper.get(relation_count_key, 0) > 0)
        
        print(f'\nPapers with {endpoint_type} saved successfully!')
        print(f'Total papers processed: {len(all_papers_data)}')
        print(f'Total {endpoint_type}: {total_relations}')
        print(f'Papers with {endpoint_type}: {papers_with_relations}/{len(all_papers_data)}')
        print(f'Saved to {output_path}')


def main():
    """Main function with argument parser."""
    parser = argparse.ArgumentParser(
        description='Data Preprocessing and Citation Graph Building Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Step 1: Preprocess and sample data
  python datapreprocessing.py --step preprocess \\
      --input ../data/raw_data/arxiv-metadata-oai-snapshot.json \\
      --n_samples 10000 \\
      --output ../data/raw_data/arxiv_cs_stat_sampled.csv \\
      --cleaned_output ../data/preprocessed_data/arxiv_cs_stat_sampled_cleaned.csv

  # Step 2: Build citations JSON
  python datapreprocessing.py --step build_relations \\
      --input_csv ../data/preprocessed_data/arxiv_cs_stat_sampled_cleaned.csv \\
      --endpoint_type citations \\
      --output_json ../data/preprocessed_data/papers_with_citations.json

  # Full pipeline
  python datapreprocessing.py --step all \\
      --input ../data/raw_data/arxiv-metadata-oai-snapshot.json \\
      --n_samples 10000 \\
      --output ../data/raw_data/arxiv_cs_stat_sampled.csv \\
      --cleaned_output ../data/preprocessed_data/arxiv_cs_stat_sampled_cleaned.csv \\
      --endpoint_type citations \\
      --output_json ../data/preprocessed_data/papers_with_citations.json
        """
    )
    
    parser.add_argument(
        '--step',
        type=str,
        choices=['preprocess', 'build_relations', 'all'],
        required=True,
        help='Processing step: preprocess (filter/sample/clean), build_relations (fetch citations), or all'
    )
    
    # Input arguments
    parser.add_argument(
        '--input',
        type=str,
        help='Path to input arXiv metadata JSON file (required for preprocess/all steps)'
    )
    parser.add_argument(
        '--input_csv',
        type=str,
        help='Path to input CSV file (required for build_relations step)'
    )
    
    # Preprocessing arguments
    parser.add_argument(
        '--n_samples',
        type=int,
        default=10000,
        help='Number of documents to sample (default: 10000, use 0 for all)'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Path to save sampled CSV file (required for preprocess/all steps)'
    )
    parser.add_argument(
        '--cleaned_output',
        type=str,
        help='Path to save cleaned/preprocessed CSV file (required for preprocess/all steps)'
    )
    parser.add_argument(
        '--random_state',
        type=int,
        default=42,
        help='Random seed for sampling (default: 42)'
    )
    
    # Citation building arguments
    parser.add_argument(
        '--endpoint_type',
        type=str,
        choices=['citations', 'references'],
        default='citations',
        help='Type of relations to fetch: citations or references (default: citations)'
    )
    parser.add_argument(
        '--output_json',
        type=str,
        help='Path to save output JSON file with papers and relations (required for build_relations/all steps)'
    )
    parser.add_argument(
        '--delay',
        type=float,
        default=1.0,
        help='Delay in seconds between API requests (default: 1.0)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments based on step
    if args.step in ['preprocess', 'all']:
        if not args.input:
            parser.error("--input is required for preprocess/all steps")
        if not args.output:
            parser.error("--output is required for preprocess/all steps")
        if not args.cleaned_output:
            parser.error("--cleaned_output is required for preprocess/all steps")
    
    if args.step in ['build_relations', 'all']:
        if not args.input_csv and args.step == 'build_relations':
            parser.error("--input_csv is required for build_relations step")
        if not args.output_json:
            parser.error("--output_json is required for build_relations/all steps")
    
    # Execute steps
    if args.step in ['preprocess', 'all']:
        print("="*80)
        print("STEP 1: DATA PREPROCESSING")
        print("="*80)
        
        # Step 1: Filter and sample
        preprocessor = DataPreprocessor(args.input)
        df = preprocessor.filter_and_sample(
            n_samples=args.n_samples,
            random_state=args.random_state
        )
        preprocessor.save(args.output)
        
        # Step 2: Text preprocessing
        preprocessor.preprocess_text()
        preprocessor.save(args.cleaned_output)
        
        print("\n✓ Data preprocessing completed!\n")
    
    if args.step in ['build_relations', 'all']:
        print("="*80)
        print("STEP 2: BUILDING CITATION GRAPH")
        print("="*80)
        
        # Load preprocessed data
        if args.step == 'all':
            input_csv = args.cleaned_output
        else:
            input_csv = args.input_csv
        
        print(f"Loading data from {input_csv}...")
        df = pd.read_csv(input_csv)
        print(f'Loaded {len(df)} papers from CSV')
        
        # Build citation graph
        builder = CitationGraphBuilder()
        builder.build_papers_with_relations_json(
            df=df,
            endpoint_type=args.endpoint_type,
            delay=args.delay,
            output_path=args.output_json
        )
        
        print("\n✓ Citation graph building completed!\n")
    
    print("="*80)
    print("ALL STEPS COMPLETED SUCCESSFULLY!")
    print("="*80)


if __name__ == "__main__":
    main()
