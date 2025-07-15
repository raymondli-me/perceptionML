"""
Embedding generation module for BASIC MODE
Reproduces exact settings from the analysis notebook
"""

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import torch
from typing import Optional, Union, List
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generate embeddings with exact reproducible settings"""
    
    def __init__(self, 
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 batch_size: int = 32,
                 max_seq_length: int = 512,
                 normalize: bool = True,
                 device: Optional[str] = None,
                 random_seed: int = 42):
        """
        Initialize embedding generator
        
        Args:
            model_name: HuggingFace model name (default: MiniLM)
            batch_size: Batch size for encoding
            max_seq_length: Maximum sequence length
            normalize: Whether to L2 normalize embeddings
            device: Device to use (cuda/cpu)
            random_seed: Random seed for reproducibility
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.normalize = normalize
        self.random_seed = random_seed
        
        # Set random seeds
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)
        
        # Set device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        # Load model
        logger.info(f"Loading model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.model.max_seq_length = max_seq_length
        self.model.to(self.device)
        
    def generate_embeddings(self, 
                          texts: Union[List[str], pd.Series],
                          show_progress: bool = True) -> np.ndarray:
        """
        Generate embeddings for texts
        
        Args:
            texts: List or Series of text strings
            show_progress: Whether to show progress bar
            
        Returns:
            numpy array of embeddings (n_texts, embedding_dim)
        """
        if isinstance(texts, pd.Series):
            texts = texts.tolist()
            
        logger.info(f"Generating embeddings for {len(texts)} texts...")
        
        # Generate embeddings
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        
        # Ensure consistent dtype
        embeddings = embeddings.astype(np.float32)
        
        logger.info(f"Generated embeddings shape: {embeddings.shape}")
        return embeddings
    
    def load_precomputed_embeddings(self, 
                                  csv_path: str,
                                  embedding_cols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Load precomputed embeddings from CSV
        
        Args:
            csv_path: Path to CSV with embeddings
            embedding_cols: List of embedding column names (if None, auto-detect)
            
        Returns:
            DataFrame with embeddings
        """
        logger.info(f"Loading precomputed embeddings from: {csv_path}")
        df = pd.read_csv(csv_path)
        
        # Auto-detect embedding columns if not specified
        if embedding_cols is None:
            # Look for columns starting with 'embedding_' or 'dim_'
            embedding_cols = [col for col in df.columns 
                            if col.startswith(('embedding_', 'dim_', 'embed_'))]
            
            # If no specific pattern, assume numeric columns are embeddings
            if not embedding_cols:
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                # Exclude common non-embedding columns
                exclude_cols = ['id', 'ID', 'index', 'treatment', 'outcome', 
                              'social_class', 'ai_rating', 'human_rating']
                embedding_cols = [col for col in numeric_cols 
                                if col not in exclude_cols]
        
        logger.info(f"Found {len(embedding_cols)} embedding dimensions")
        return df
    
    @staticmethod
    def get_model_info(model_name: str) -> dict:
        """Get information about a model"""
        model_info = {
            "sentence-transformers/all-MiniLM-L6-v2": {
                "embedding_dim": 384,
                "max_seq_length": 512,
                "description": "Fast, lightweight model good for semantic similarity"
            },
            "nvidia/NV-Embed-v2": {
                "embedding_dim": 4096,
                "max_seq_length": 32768,
                "description": "High-quality embeddings, larger dimension"
            }
        }
        
        return model_info.get(model_name, {
            "embedding_dim": "unknown",
            "max_seq_length": "unknown", 
            "description": "Custom model"
        })