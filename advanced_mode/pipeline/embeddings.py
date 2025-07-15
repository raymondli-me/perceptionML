#!/usr/bin/env python3
"""Embedding generation and management."""

import numpy as np
import torch
from typing import List, Optional, Union
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from .config import PipelineConfig


class EmbeddingGenerator:
    """Handles text embedding generation."""
    
    def __init__(self, config: PipelineConfig, num_gpus: int = 1):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.device = None
        self.num_gpus = num_gpus
        
    def setup_model(self):
        """Load the embedding model."""
        if self.config.embedding_model == 'pre-computed':
            print("Using pre-computed embeddings, no model needed")
            return
        
        print(f"Loading embedding model: {self.config.embedding_model}")
        
        # Check for GPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Check available GPUs
        if self.device.type == 'cuda':
            n_available = torch.cuda.device_count()
            print(f"Available GPUs: {n_available}")
            if self.num_gpus > n_available:
                print(f"Warning: Requested {self.num_gpus} GPUs but only {n_available} available")
                self.num_gpus = n_available
            print(f"Using {self.num_gpus} GPU(s)")
        else:
            print(f"Using device: {self.device}")
        
        if self.config.embedding_model == 'nvidia/NV-Embed-v2':
            self._load_nvembed()
        elif self.config.embedding_model.startswith('sentence-transformers/'):
            self._load_sentence_transformer()
        elif any(self.config.embedding_model.startswith(prefix) for prefix in 
                 ['intfloat/', 'google/', 'Linq-AI-Research/', 'Alibaba-NLP/', 'Salesforce/']):
            # These models use AutoModel from transformers
            self._load_transformer_model()
        else:
            raise ValueError(f"Unsupported embedding model: {self.config.embedding_model}")
    
    def _load_nvembed(self):
        """Load NV-Embed-v2 model."""
        from transformers import AutoModel
        
        # For multi-GPU, we'll load a model on each GPU
        if self.device.type == 'cuda' and self.num_gpus > 1:
            print(f"Setting up {self.num_gpus} models for multi-GPU processing")
            self.models = []
            for gpu_id in range(self.num_gpus):
                model = AutoModel.from_pretrained(
                    'nvidia/NV-Embed-v2',
                    trust_remote_code=True,
                    torch_dtype=torch.float16
                )
                model = model.to(f'cuda:{gpu_id}')
                model.eval()
                self.models.append(model)
            print(f"✓ Loaded {self.num_gpus} NV-Embed-v2 models on separate GPUs")
        else:
            # Single GPU or CPU
            self.model = AutoModel.from_pretrained(
                'nvidia/NV-Embed-v2',
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32
            )
            self.model = self.model.to(self.device)
            self.model.eval()
            print("✓ NV-Embed-v2 model loaded")
    
    def _load_sentence_transformer(self):
        """Load sentence-transformers model."""
        from sentence_transformers import SentenceTransformer
        
        model_name = self.config.embedding_model
        self.model = SentenceTransformer(model_name)
        
        # Handle multi-GPU setup for sentence-transformers
        if self.device.type == 'cuda' and self.num_gpus > 1:
            print(f"Setting up multi-GPU encoding with {self.num_gpus} GPUs")
            # sentence-transformers handles multi-GPU differently
            # We'll use their built-in multi-process encoding
            self._use_multi_process_encoding = True
        else:
            self.model = self.model.to(self.device)
            self._use_multi_process_encoding = False
        
        print(f"✓ {model_name} model loaded")
    
    def _load_transformer_model(self):
        """Load models using transformers AutoModel."""
        from transformers import AutoModel, AutoTokenizer
        
        model_name = self.config.embedding_model
        print(f"Loading transformer model: {model_name}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        # For multi-GPU, load a model on each GPU
        if self.device.type == 'cuda' and self.num_gpus > 1:
            print(f"Setting up {self.num_gpus} models for multi-GPU processing")
            self.models = []
            for gpu_id in range(self.num_gpus):
                model = AutoModel.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    torch_dtype=torch.float16
                )
                model = model.to(f'cuda:{gpu_id}')
                model.eval()
                self.models.append(model)
            print(f"✓ Loaded {self.num_gpus} {model_name} models on separate GPUs")
        else:
            # Single GPU or CPU
            self.model = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32
            )
            self.model = self.model.to(self.device)
            self.model.eval()
            print(f"✓ {model_name} model loaded")
    
    def generate_embeddings(self, texts: List[str], 
                          save_path: Optional[str] = None) -> np.ndarray:
        """Generate embeddings for a list of texts."""
        if self.config.embedding_model == 'pre-computed':
            raise ValueError("Cannot generate embeddings with 'pre-computed' model")
        
        if self.model is None:
            self.setup_model()
        
        print(f"Generating embeddings for {len(texts)} texts...")
        
        if self.config.embedding_model == 'nvidia/NV-Embed-v2':
            embeddings = self._generate_nvembed(texts)
        elif self.config.embedding_model.startswith('sentence-transformers/'):
            embeddings = self._generate_sentence_transformer(texts)
        else:
            # Use transformer model generation
            embeddings = self._generate_transformer(texts)
        
        if save_path:
            np.save(save_path, embeddings)
            print(f"✓ Embeddings saved to {save_path}")
        
        return embeddings
    
    def _generate_nvembed(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings using NV-Embed-v2."""
        instruction = "Represent this text for clustering and classification: "
        batch_size = self.config.analysis.batch_size
        max_length = self.config.analysis.max_text_length
        
        # Multi-GPU processing
        if hasattr(self, 'models') and len(self.models) > 1:
            import concurrent.futures
            import threading
            
            effective_batch_size = batch_size * self.num_gpus
            print(f"Using {self.num_gpus} GPUs with effective batch size: {effective_batch_size} ({batch_size} per GPU)")
            
            all_embeddings = []
            
            def process_on_gpu(gpu_id, texts_subset):
                """Process a subset of texts on a specific GPU."""
                try:
                    # Set device context and ensure proper synchronization
                    torch.cuda.set_device(gpu_id)
                    with torch.cuda.device(gpu_id):
                        # Clear cache before processing
                        torch.cuda.empty_cache()
                        
                        with torch.cuda.amp.autocast(dtype=torch.float16):
                            embeddings = self.models[gpu_id].encode(
                                texts_subset,
                                instruction=instruction,
                                max_length=max_length
                            )
                        
                        # Synchronize to ensure computation is complete
                        torch.cuda.synchronize(device=gpu_id)
                    
                    # Convert to numpy
                    if hasattr(embeddings, 'cpu'):
                        embeddings = embeddings.cpu().numpy()
                    else:
                        embeddings = np.array(embeddings)
                    return embeddings
                except torch.cuda.OutOfMemoryError as e:
                    print(f"GPU {gpu_id} out of memory. Clearing cache and retrying with smaller batch...")
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize(device=gpu_id)
                    # Return zeros on OOM - let the main loop handle retry
                    return np.zeros((len(texts_subset), 4096))
                except Exception as e:
                    print(f"GPU {gpu_id} failed: {str(e)[:100]}")
                    # Return zeros on failure
                    return np.zeros((len(texts_subset), 4096))
            
            # Process batches
            with torch.no_grad():
                for i in tqdm(range(0, len(texts), effective_batch_size), desc="Generating embeddings"):
                    batch = texts[i:i + effective_batch_size]
                    
                    # Split batch across GPUs
                    splits = []
                    split_size = len(batch) // self.num_gpus
                    for gpu_id in range(self.num_gpus):
                        start_idx = gpu_id * split_size
                        if gpu_id == self.num_gpus - 1:
                            # Last GPU gets remaining texts
                            end_idx = len(batch)
                        else:
                            end_idx = start_idx + split_size
                        if start_idx < len(batch):
                            splits.append((gpu_id, batch[start_idx:end_idx]))
                    
                    # Process in parallel
                    with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_gpus) as executor:
                        futures = [executor.submit(process_on_gpu, gpu_id, texts_subset) 
                                 for gpu_id, texts_subset in splits]
                        results = [future.result() for future in futures]
                    
                    # Concatenate results
                    batch_embeddings = np.vstack(results)
                    all_embeddings.append(batch_embeddings)
            
            embeddings = np.vstack(all_embeddings)
            
        else:
            # Single GPU processing
            effective_batch_size = batch_size
            all_embeddings = []
            
            with torch.no_grad():
                for i in tqdm(range(0, len(texts), effective_batch_size), desc="Generating embeddings"):
                    batch = texts[i:i + effective_batch_size]
                    
                    try:
                        if self.device.type == 'cuda':
                            with torch.cuda.amp.autocast(dtype=torch.float16):
                                embeddings = self.model.encode(
                                    batch,
                                    instruction=instruction,
                                    max_length=max_length
                                )
                        else:
                            embeddings = self.model.encode(
                                batch,
                                instruction=instruction,
                                max_length=max_length
                            )
                        
                        # Convert to numpy
                        if hasattr(embeddings, 'cpu'):
                            embeddings = embeddings.cpu().numpy()
                        else:
                            embeddings = np.array(embeddings)
                        
                        all_embeddings.append(embeddings)
                        
                    except torch.cuda.OutOfMemoryError as oom_e:
                        print(f"\n❌ GPU OUT OF MEMORY on batch {i//effective_batch_size}")
                        print(f"   Current batch size: {len(batch)}")
                        
                        # Try to recover by processing smaller sub-batches
                        print("   Attempting recovery with smaller sub-batches...")
                        torch.cuda.empty_cache()
                        
                        sub_batch_size = max(1, len(batch) // 4)  # Try 1/4 of the batch
                        sub_embeddings = []
                        failed_texts = []
                        
                        for j in range(0, len(batch), sub_batch_size):
                            sub_batch = batch[j:j + sub_batch_size]
                            try:
                                if self.device.type == 'cuda':
                                    with torch.cuda.amp.autocast(dtype=torch.float16):
                                        sub_emb = self.model.encode(
                                            sub_batch,
                                            instruction=instruction,
                                            max_length=max_length
                                        )
                                else:
                                    sub_emb = self.model.encode(
                                        sub_batch,
                                        instruction=instruction,
                                        max_length=max_length
                                    )
                                
                                if hasattr(sub_emb, 'cpu'):
                                    sub_emb = sub_emb.cpu().numpy()
                                else:
                                    sub_emb = np.array(sub_emb)
                                
                                sub_embeddings.append(sub_emb)
                                print(f"   ✓ Recovered sub-batch {j//sub_batch_size + 1}")
                            except Exception as sub_e:
                                print(f"   ❌ Sub-batch {j//sub_batch_size + 1} still failed")
                                failed_texts.extend(sub_batch)
                                # Determine embedding dimension
                                if sub_embeddings:
                                    embedding_dim = sub_embeddings[0].shape[1]
                                elif all_embeddings:
                                    embedding_dim = all_embeddings[-1].shape[1]
                                else:
                                    embedding_dim = 4096
                                sub_embeddings.append(np.zeros((len(sub_batch), embedding_dim)))
                        
                        if failed_texts:
                            print(f"\n⚠️  WARNING: {len(failed_texts)} texts failed to embed and were set to zeros")
                            print(f"   This may significantly impact your analysis quality!")
                            print(f"   Consider reducing batch size with --batch-size flag")
                        
                        if sub_embeddings:
                            all_embeddings.append(np.vstack(sub_embeddings))
                        else:
                            # Complete failure - create zeros
                            if all_embeddings:
                                embedding_dim = all_embeddings[-1].shape[1]
                            else:
                                embedding_dim = 4096
                            all_embeddings.append(np.zeros((len(batch), embedding_dim)))
                            
                    except Exception as e:
                        print(f"\n❌ ERROR: Batch {i//effective_batch_size} failed: {str(e)}")
                        print(f"   This is likely due to an unexpected error.")
                        print(f"   Creating zero embeddings for this batch...")
                        
                        # Create zero embeddings for failed batch
                        if all_embeddings:
                            embedding_dim = all_embeddings[-1].shape[1]
                        else:
                            try:
                                test_emb = self.model.encode([batch[0]], instruction=instruction)
                                embedding_dim = test_emb.shape[-1]
                            except:
                                embedding_dim = 4096
                        
                        all_embeddings.append(np.zeros((len(batch), embedding_dim)))
                        print(f"   ⚠️  Added {len(batch)} zero embeddings")
            
            embeddings = np.vstack(all_embeddings)
        
        print(f"✓ Generated embeddings with shape: {embeddings.shape}")
        
        return embeddings
    
    def _generate_sentence_transformer(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings using sentence-transformers."""
        batch_size = self.config.analysis.batch_size
        
        # Handle multi-GPU encoding for sentence-transformers
        if hasattr(self, '_use_multi_process_encoding') and self._use_multi_process_encoding:
            # Use multi-process encoding for multiple GPUs
            print(f"Using multi-process encoding with {self.num_gpus} processes")
            from sentence_transformers import SentenceTransformer
            import multiprocessing as mp
            
            # Create a pool of processes, each with its own GPU
            pool = self.model.start_multi_process_pool(target_devices=[f'cuda:{i}' for i in range(self.num_gpus)])
            
            # Encode using the pool
            embeddings = self.model.encode_multi_process(
                texts,
                pool,
                batch_size=batch_size,
                normalize_embeddings=False
            )
            
            # Stop the pool
            self.model.stop_multi_process_pool(pool)
        else:
            # Single GPU or CPU encoding
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_numpy=True
            )
        
        print(f"✓ Generated embeddings with shape: {embeddings.shape}")
        return embeddings
    
    def _generate_transformer(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings using transformer models."""
        batch_size = self.config.analysis.batch_size
        max_length = self.config.analysis.max_text_length
        
        # Add instruction prefix for E5 models
        if 'e5' in self.config.embedding_model.lower():
            if 'query' in self.config.embedding_model.lower():
                texts = [f"query: {text}" for text in texts]
            else:
                texts = [f"passage: {text}" for text in texts]
        
        # Multi-GPU processing
        if hasattr(self, 'models') and len(self.models) > 1:
            import concurrent.futures
            
            effective_batch_size = batch_size * self.num_gpus
            print(f"Using {self.num_gpus} GPUs with effective batch size: {effective_batch_size} ({batch_size} per GPU)")
            
            all_embeddings = []
            
            def process_on_gpu(gpu_id, texts_subset):
                """Process a subset of texts on a specific GPU."""
                try:
                    # Set device context and ensure proper synchronization
                    torch.cuda.set_device(gpu_id)
                    with torch.cuda.device(gpu_id):
                        # Clear cache before processing
                        torch.cuda.empty_cache()
                        
                        # Tokenize
                        inputs = self.tokenizer(
                            texts_subset,
                            padding=True,
                            truncation=True,
                            max_length=max_length,
                            return_tensors='pt'
                        ).to(f'cuda:{gpu_id}')
                        
                        # Generate embeddings
                        with torch.no_grad():
                            with torch.cuda.amp.autocast(dtype=torch.float16):
                                outputs = self.models[gpu_id](**inputs)
                                # Use pooler_output if available, else mean pooling
                                if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                                    embeddings = outputs.pooler_output
                                else:
                                    embeddings = outputs.last_hidden_state.mean(dim=1)
                        
                        # Synchronize to ensure computation is complete
                        torch.cuda.synchronize(device=gpu_id)
                        
                        return embeddings.cpu().numpy()
                except torch.cuda.OutOfMemoryError as e:
                    print(f"GPU {gpu_id} out of memory. Clearing cache...")
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize(device=gpu_id)
                    # Try to determine embedding dimension dynamically
                    try:
                        # Try to get a single embedding to determine size
                        with torch.no_grad():
                            test_input = self.tokenizer(
                                texts_subset[:1],
                                padding=True,
                                truncation=True,
                                max_length=max_length,
                                return_tensors='pt'
                            ).to(f'cuda:{gpu_id}')
                            
                            test_output = self.models[gpu_id](**test_input)
                            if hasattr(test_output, 'pooler_output') and test_output.pooler_output is not None:
                                embed_dim = test_output.pooler_output.shape[-1]
                            else:
                                embed_dim = test_output.last_hidden_state.mean(dim=1).shape[-1]
                    except:
                        # If that fails too, try to get from model config
                        if hasattr(self.models[0].config, 'hidden_size'):
                            embed_dim = self.models[0].config.hidden_size
                        else:
                            # Last resort: use a reasonable default
                            embed_dim = 768
                            print(f"Warning: Could not determine embedding dimension, using default {embed_dim}")
                    
                    return np.zeros((len(texts_subset), embed_dim))
            
            # Process batches
            with torch.no_grad():
                for i in tqdm(range(0, len(texts), effective_batch_size), desc="Generating embeddings"):
                    batch = texts[i:i + effective_batch_size]
                    
                    # Split batch across GPUs
                    splits = []
                    split_size = len(batch) // self.num_gpus
                    for gpu_id in range(self.num_gpus):
                        start_idx = gpu_id * split_size
                        if gpu_id == self.num_gpus - 1:
                            end_idx = len(batch)
                        else:
                            end_idx = start_idx + split_size
                        if start_idx < len(batch):
                            splits.append((gpu_id, batch[start_idx:end_idx]))
                    
                    # Process in parallel
                    with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_gpus) as executor:
                        futures = [executor.submit(process_on_gpu, gpu_id, texts_subset) 
                                 for gpu_id, texts_subset in splits]
                        results = [future.result() for future in futures]
                    
                    # Concatenate results
                    batch_embeddings = np.vstack(results)
                    all_embeddings.append(batch_embeddings)
            
            embeddings = np.vstack(all_embeddings)
            
        else:
            # Single GPU processing
            all_embeddings = []
            
            with torch.no_grad():
                for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
                    batch = texts[i:i + batch_size]
                    
                    try:
                        # Tokenize
                        inputs = self.tokenizer(
                            batch,
                            padding=True,
                            truncation=True,
                            max_length=max_length,
                            return_tensors='pt'
                        ).to(self.device)
                        
                        # Generate embeddings
                        if self.device.type == 'cuda':
                            with torch.cuda.amp.autocast(dtype=torch.float16):
                                outputs = self.model(**inputs)
                        else:
                            outputs = self.model(**inputs)
                        
                        # Use pooler_output if available, else mean pooling
                        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                            embeddings = outputs.pooler_output
                        else:
                            # Mean pooling over sequence length
                            embeddings = outputs.last_hidden_state.mean(dim=1)
                        
                        all_embeddings.append(embeddings.cpu().numpy())
                        
                    except Exception as e:
                        print(f"Warning: Batch {i//batch_size} failed: {str(e)[:100]}")
                        # Create zero embeddings for failed batch
                        if all_embeddings:
                            embedding_dim = all_embeddings[-1].shape[1]
                        else:
                            embedding_dim = 768  # Common default
                        all_embeddings.append(np.zeros((len(batch), embedding_dim)))
            
            embeddings = np.vstack(all_embeddings)
        
        # Normalize embeddings for E5 models
        if 'e5' in self.config.embedding_model.lower():
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        return embeddings
    
    def validate_embeddings(self, embeddings: np.ndarray, texts: List[str]) -> bool:
        """Validate embeddings quality and dimensions."""
        print("Validating embeddings...")
        
        # Check shape
        if len(embeddings) != len(texts):
            print(f"❌ Embeddings count ({len(embeddings)}) doesn't match texts ({len(texts)})")
            return False
        
        # Check for NaN or Inf
        if np.any(np.isnan(embeddings)) or np.any(np.isinf(embeddings)):
            print("❌ Embeddings contain NaN or Inf values")
            return False
        
        # Check for all-zero embeddings
        zero_rows = np.all(embeddings == 0, axis=1)
        n_zero = np.sum(zero_rows)
        if n_zero > 0:
            print(f"⚠️  Warning: {n_zero} embeddings are all zeros ({n_zero/len(embeddings)*100:.1f}%)")
        
        # Check embedding statistics
        mean_norm = np.mean(np.linalg.norm(embeddings, axis=1))
        std_norm = np.std(np.linalg.norm(embeddings, axis=1))
        
        print(f"✓ Embeddings validated:")
        print(f"  - Shape: {embeddings.shape}")
        print(f"  - Mean L2 norm: {mean_norm:.4f}")
        print(f"  - Std L2 norm: {std_norm:.4f}")
        print(f"  - Non-zero: {len(embeddings) - n_zero}/{len(embeddings)}")
        
        return True
    
    @staticmethod
    def load_embeddings(path: str) -> np.ndarray:
        """Load pre-computed embeddings from file."""
        print(f"Loading embeddings from {path}")
        embeddings = np.load(path)
        print(f"✓ Loaded embeddings with shape: {embeddings.shape}")
        return embeddings