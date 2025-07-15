#!/bin/bash

# PerceptionML BASIC MODE Runner Script
# This script demonstrates all features of BASIC MODE

echo "=========================================="
echo "   PerceptionML BASIC MODE Demo Script    "
echo "=========================================="
echo ""

# Function to print section headers
print_section() {
    echo ""
    echo "----------------------------------------"
    echo "$1"
    echo "----------------------------------------"
}

# Function to pause between examples
pause() {
    echo ""
    read -p "Press Enter to continue..."
    echo ""
}

# Check if we're in the right directory
if [ ! -f "basic_mode/__init__.py" ]; then
    echo "Error: Please run this script from the perceptionML directory"
    echo "Usage: cd /path/to/perceptionML && bash basic_mode/run_basic_mode.sh"
    exit 1
fi

print_section "1. BASIC MODE Overview"
echo "BASIC MODE is a lightweight implementation of the DML-LME framework"
echo "for analyzing text perception using embeddings."
echo ""
echo "Key features:"
echo "  - MiniLM embeddings (384 dims) with exact reproducible settings"
echo "  - Double Machine Learning (DML) with cross-fitting"
echo "  - 5 feature selection methods"
echo "  - Comprehensive results table output"
echo "  - Full CSV export of all intermediate data"
pause

print_section "2. Installation Check"
echo "Checking Python dependencies..."
python -c "import sentence_transformers; print('✓ sentence-transformers installed')" 2>/dev/null || echo "✗ sentence-transformers missing"
python -c "import xgboost; print('✓ xgboost installed')" 2>/dev/null || echo "✗ xgboost missing"
python -c "import sklearn; print('✓ scikit-learn installed')" 2>/dev/null || echo "✗ scikit-learn missing"
python -c "import pandas; print('✓ pandas installed')" 2>/dev/null || echo "✗ pandas missing"
python -c "import numpy; print('✓ numpy installed')" 2>/dev/null || echo "✗ numpy missing"
echo ""
echo "If any dependencies are missing, install with:"
echo "pip install -r basic_mode/requirements.txt"
pause

print_section "3. Command-Line Help"
echo "Let's see the available options:"
echo ""
echo "$ python -m perceptionML.basic_mode --help"
echo ""
python -m perceptionML.basic_mode --help
pause

print_section "4. Example 1: Generate Test Data"
echo "First, let's create some test data to demonstrate BASIC MODE"
echo ""
cat > test_data.csv << 'EOF'
id,text,social_class,ai_rating
1,"This essay discusses the importance of education in modern society. Education provides opportunities and knowledge.",0.8,0.9
2,"I believe that hard work is essential for success. My family taught me these values from a young age.",0.3,0.4
3,"The economic system creates both opportunities and challenges. We must work together to improve conditions.",0.5,0.6
4,"Personal responsibility and community support are both important. Balance is key to a good life.",0.6,0.7
5,"Growing up, we didn't have much, but we had each other. Family bonds are stronger than material wealth.",0.2,0.3
EOF
echo "Created test_data.csv with 5 sample records"
ls -la test_data.csv
pause

print_section "5. Example 2: Run Analysis with Raw Text"
echo "Running BASIC MODE on raw text data:"
echo ""
echo "$ python -m perceptionML.basic_mode \\"
echo "    --data test_data.csv \\"
echo "    --treatment social_class \\"
echo "    --outcome ai_rating \\"
echo "    --text text \\"
echo "    --id id \\"
echo "    --output-dir test_results"
echo ""
echo "Note: This will generate embeddings and run the full analysis"
echo "(Skipping actual execution to save time - would take ~30 seconds)"
pause

print_section "6. Example 3: Using Precomputed Embeddings"
echo "If you have precomputed embeddings, you can run faster:"
echo ""
echo "$ python -m perceptionML.basic_mode \\"
echo "    --embeddings embeddings_minilm.csv \\"
echo "    --treatment social_class \\"
echo "    --outcome ai_rating \\"
echo "    --precomputed \\"
echo "    --output-dir results_precomputed"
pause

print_section "7. Python API Example"
echo "You can also use BASIC MODE from Python:"
echo ""
cat << 'EOF'
from perceptionML.basic_mode import BasicAnalysis

# Initialize analyzer
analysis = BasicAnalysis(
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    n_pca_components=200,
    n_top_features=6,
    random_seed=42
)

# Run analysis
results = analysis.run(
    data_path="test_data.csv",
    treatment_col="social_class",
    outcome_col="ai_rating",
    text_col="text",
    id_col="id"
)

# Results are automatically displayed and saved as CSVs
EOF
pause

print_section "8. R Integration Example"
echo "BASIC MODE also works with R using reticulate:"
echo ""
cat << 'EOF'
# In R:
source("basic_mode/basic_mode_wrapper.R")
init_perceptionml_basic()

results <- run_basic_analysis(
  data_path = "test_data.csv",
  treatment_col = "social_class",
  outcome_col = "ai_rating",
  text_col = "text",
  id_col = "id"
)

# Get best model
best_model <- get_best_model(results)
EOF
pause

print_section "9. Output Structure"
echo "BASIC MODE creates a timestamped output directory with:"
echo ""
echo "results_YYYYMMDD_HHMMSS/"
echo "├── results_table.csv           # Main results (matching paper format)"
echo "├── embeddings.csv              # Generated embeddings (384 dims)"
echo "├── pca_components.csv          # 200 PCA components"
echo "├── pca_variance_explained.csv  # Variance explained by each PC"
echo "├── feature_selections.csv      # Selected PC indices for each method"
echo "├── raw_data.csv               # Treatment and outcome variables"
echo "└── analysis_params.json        # Parameters for reproducibility"
pause

print_section "10. Reproducibility Settings"
echo "To ensure reproducibility, BASIC MODE uses:"
echo ""
echo "  - Random seed: 42 (default)"
echo "  - MiniLM model: sentence-transformers/all-MiniLM-L6-v2"
echo "  - Batch size: 32"
echo "  - Max sequence length: 512 tokens"
echo "  - L2 normalization: Enabled"
echo "  - PCA components: 200"
echo "  - Top features: 6"
echo "  - CV folds: 5"
echo ""
echo "These match the exact settings from your notebook analysis."
pause

print_section "11. Run Tests"
echo "Let's verify the installation works correctly:"
echo ""
echo "$ python basic_mode/test_basic_mode.py"
echo ""
python basic_mode/test_basic_mode.py
pause

print_section "12. Results Table Format"
echo "BASIC MODE produces a comprehensive table with:"
echo ""
echo "  - Model type (OLS/DML)"
echo "  - Learner method (XGBoost/Lasso/Ridge/OLS/MI)"
echo "  - Feature set (Full/200 PCs/Top 6)"
echo "  - Coefficient with robust standard errors"
echo "  - R² values (full and cross-validated)"
echo "  - Correlation metrics (G, C)"
echo "  - Percentage reduction vs baseline"
echo ""
echo "This matches Table X from your paper exactly."
pause

print_section "13. Advanced Usage"
echo "Custom parameters example:"
echo ""
echo "$ python -m perceptionML.basic_mode \\"
echo "    --data mydata.csv \\"
echo "    --treatment X --outcome Y --text text_col \\"
echo "    --pca-components 100 \\"      # Use 100 PCA components
echo "    --top-features 10 \\"         # Select top 10 features
echo "    --cv-folds 10 \\"            # Use 10-fold cross-validation
echo "    --random-seed 123"           # Different random seed
pause

print_section "14. Quick Reference Card"
cat << 'EOF'
╔══════════════════════════════════════════════════════════════╗
║                  BASIC MODE QUICK REFERENCE                  ║
╠══════════════════════════════════════════════════════════════╣
║ Raw text:                                                    ║
║   python -m perceptionML.basic_mode --data file.csv \        ║
║     --treatment X --outcome Y --text text_col --id id_col   ║
║                                                              ║
║ Precomputed embeddings:                                      ║
║   python -m perceptionML.basic_mode --embeddings emb.csv \  ║
║     --treatment X --outcome Y --precomputed                  ║
║                                                              ║
║ Python API:                                                  ║
║   from perceptionML.basic_mode import BasicAnalysis         ║
║   analysis = BasicAnalysis()                                 ║
║   results = analysis.run(...)                                ║
║                                                              ║
║ Help: python -m perceptionML.basic_mode --help              ║
╚══════════════════════════════════════════════════════════════╝
EOF

echo ""
echo "=========================================="
echo "        BASIC MODE Demo Complete!         "
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Try running on your actual data"
echo "2. Check the README.md for more details"
echo "3. Examine the output CSV files"
echo ""
echo "For more information, see: basic_mode/README.md"
echo ""

# Cleanup
rm -f test_data.csv