#!/bin/bash

# Run BASIC MODE analysis on MiniLM embeddings
# This reproduces the exact analysis from the notebook

echo "=========================================="
echo "   Running MiniLM Analysis with BASIC MODE"
echo "=========================================="
echo ""

# Set the embeddings file path (adjust if your file is in a different location)
EMBEDDINGS_FILE="../fully_anonymized_embeddings_minilm.csv"

# Check multiple possible locations for the embeddings file
if [ -f "fully_anonymized_embeddings_minilm.csv" ]; then
    EMBEDDINGS_FILE="fully_anonymized_embeddings_minilm.csv"
elif [ -f "../fully_anonymized_embeddings_minilm.csv" ]; then
    EMBEDDINGS_FILE="../fully_anonymized_embeddings_minilm.csv"
elif [ -f "../../fully_anonymized_embeddings_minilm.csv" ]; then
    EMBEDDINGS_FILE="../../fully_anonymized_embeddings_minilm.csv"
else
    echo "Error: Could not find fully_anonymized_embeddings_minilm.csv"
    echo ""
    echo "Please specify the path to your embeddings file:"
    echo "  bash run_minilm_analysis.sh /path/to/fully_anonymized_embeddings_minilm.csv"
    echo ""
    echo "Or place the file in the current directory."
    exit 1
fi

# Allow command line argument to override
if [ $# -eq 1 ]; then
    EMBEDDINGS_FILE="$1"
fi

# Verify file exists
if [ ! -f "$EMBEDDINGS_FILE" ]; then
    echo "Error: Embeddings file not found at: $EMBEDDINGS_FILE"
    exit 1
fi

echo "Found embeddings file: $EMBEDDINGS_FILE"
echo ""
echo "Starting MiniLM analysis with:"
echo "  - Treatment: social_class"
echo "  - Outcome: ai_rating"
echo "  - Model: MiniLM (384 dimensions)"
echo "  - Random seed: 42"
echo "  - PCA components: 200"
echo "  - Top features: 6"
echo "  - CV folds: 5"
echo ""

# Set environment variables for reproducibility
export PYTHONHASHSEED=42

# Set environment variables for XGBoost on Apple Silicon
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Run the analysis (use python -m with current directory in path)
PYTHONPATH="." python -m basic_mode \
    --embeddings "$EMBEDDINGS_FILE" \
    --treatment social_class \
    --outcome ai_rating \
    --precomputed \
    --random-seed 42 \
    --pca-components 200 \
    --top-features 6 \
    --cv-folds 5 \
    --output-dir minilm_analysis_results

echo ""
echo "=========================================="
echo "   MiniLM Analysis Complete!              "
echo "=========================================="
echo ""
echo "Results saved to: minilm_analysis_results/"
echo ""
echo "Key output files:"
echo "  - minilm_analysis_results/results_table.csv        # Main results matching paper format"
echo "  - minilm_analysis_results/feature_selections.csv   # Top 6 PC selections for each method"
echo "  - minilm_analysis_results/pca_components.csv       # 200 PCA components"
echo "  - minilm_analysis_results/pca_variance_explained.csv"
echo "  - minilm_analysis_results/analysis_params.json     # Parameters for reproducibility"
echo ""
echo "To view results:"
echo "  cat minilm_analysis_results/results_table.csv | column -t -s,"
echo ""