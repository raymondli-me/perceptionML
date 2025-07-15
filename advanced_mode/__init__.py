"""
PerceptionML Advanced Mode
==========================

Full-featured pipeline for text perception analysis with multiple embeddings,
configuration management, and advanced analysis options.

This mode provides:
- Multiple embedding models (MiniLM, NV-Embed, custom models)
- YAML-based configuration
- GPU acceleration support
- Advanced feature selection methods
- Comprehensive logging and monitoring
"""

from .pipeline import *

__all__ = ["PerceptionMLPipeline", "EmbeddingGenerator", "DMLEstimator"]