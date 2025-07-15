"""
PerceptionML: Text Perception Analysis using Language Model Embeddings
=====================================================================

A comprehensive toolkit for analyzing text perception using Double Machine Learning (DML) 
with language model embeddings.

Two modes available:
- Basic Mode: Simplified, reproducible analysis matching paper results
- Advanced Mode: Full pipeline with multiple embeddings and advanced features

Quick Start:
-----------
# Basic Mode (Recommended for most users)
from perceptionML import BasicAnalysis

analysis = BasicAnalysis()
results = analysis.run(
    data_path="mydata.csv",
    treatment_col="X",
    outcome_col="Y", 
    text_col="text"
)

# Advanced Mode
from perceptionML.advanced_mode import PerceptionMLPipeline

pipeline = PerceptionMLPipeline(config_path="config.yaml")
pipeline.run()

For command-line usage:
    python -m perceptionML.basic_mode --help
    python -m perceptionML.advanced_mode --help
"""

__version__ = "2.0.0"

# Import basic mode as primary interface
from .basic_mode import BasicAnalysis, EmbeddingGenerator, DMLAnalyzer

# Make basic mode classes available at package level
__all__ = [
    "BasicAnalysis",
    "EmbeddingGenerator", 
    "DMLAnalyzer",
    "__version__"
]

# For backward compatibility
def get_basic_analysis():
    """Get a BasicAnalysis instance"""
    return BasicAnalysis()

def get_advanced_pipeline(config_path=None):
    """Get an advanced mode pipeline instance"""
    from .advanced_mode.pipeline import PerceptionMLPipeline
    return PerceptionMLPipeline(config_path)