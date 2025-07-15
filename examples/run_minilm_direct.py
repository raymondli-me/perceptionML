#!/usr/bin/env python
"""
Direct Python script to run MiniLM analysis
No module installation required
"""

import sys
import os

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from basic_mode import BasicAnalysis

# Run the analysis
analysis = BasicAnalysis(
    n_pca_components=200,
    n_top_features=6,
    n_folds=5,
    random_seed=42,
    output_dir='minilm_analysis_results'
)

# Find embeddings file
embeddings_path = None
for path in ['fully_anonymized_embeddings_minilm.csv', 
             '../fully_anonymized_embeddings_minilm.csv',
             '../../fully_anonymized_embeddings_minilm.csv']:
    if os.path.exists(path):
        embeddings_path = path
        break

if embeddings_path is None:
    print("Error: Could not find fully_anonymized_embeddings_minilm.csv")
    sys.exit(1)

print(f"Found embeddings at: {embeddings_path}")
print("Running MiniLM analysis...")

results = analysis.run(
    embeddings_path=embeddings_path,
    treatment_col='social_class',
    outcome_col='ai_rating',
    id_col='id',
    precomputed_embeddings=True
)

print("\nAnalysis complete!")
print("Results saved to: minilm_analysis_results/")