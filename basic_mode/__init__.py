"""
PerceptionML BASIC MODE
======================

A lightweight, reproducible implementation of the DML-LME framework
for analyzing text perception using embeddings.

This module provides:
- MiniLM embedding generation with exact settings
- Double Machine Learning (DML) causal analysis
- Feature selection methods (XGBoost, Lasso, Ridge, MI, OLS)
- Comprehensive results table output
- Full data export capabilities

Usage:
    # Command line
    python -m perceptionML.basic_mode --data mydata.csv --treatment X --outcome Y --text text_col --id id_col
    
    # Python API
    from perceptionML.basic_mode import BasicAnalysis
    analysis = BasicAnalysis()
    results = analysis.run(data_path="mydata.csv", treatment="X", outcome="Y", text="text_col")
"""

# Set global random seeds for reproducibility
import os
os.environ['PYTHONHASHSEED'] = '42'

import numpy as np
import random
np.random.seed(42)
random.seed(42)

from .analysis import BasicAnalysis
from .embeddings import EmbeddingGenerator
from .dml import DMLAnalyzer

__version__ = "1.0.0"
__all__ = ["BasicAnalysis", "EmbeddingGenerator", "DMLAnalyzer"]