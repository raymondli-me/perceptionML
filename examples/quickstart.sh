#!/bin/bash

# PerceptionML Quick Start Example
# This script demonstrates basic usage of PerceptionML

echo "=================================================="
echo "PerceptionML Quick Start Example"
echo "=================================================="
echo ""

# Example 1: Basic usage with minimal configuration
echo "Example 1: Basic analysis with auto-detection"
echo "---------------------------------------------"
cat << 'EOF'
# Analyze any CSV with text data
perceptionml --data your_data.csv

# PerceptionML will automatically:
# - Detect text column (longest average text)
# - Create ID column if missing
# - Identify numeric columns as outcomes
# - Use optimal clustering parameters
EOF
echo ""

# Example 2: Model comparison
echo "Example 2: Compare embedding models"
echo "-----------------------------------"
cat << 'EOF'
# Fast model (384-dim embeddings)
perceptionml --data data.csv \
  --embedding-model sentence-transformers/all-MiniLM-L6-v2 \
  --output minilm_analysis.html

# High-quality model (4096-dim embeddings)
perceptionml --data data.csv \
  --embedding-model nvidia/NV-Embed-v2 \
  --output nvidia_analysis.html
EOF
echo ""

# Example 3: Zero-presence analysis
echo "Example 3: Zero-presence outcome analysis"
echo "-----------------------------------------"
cat << 'EOF'
# Analyze which topics correlate with zero vs non-zero outcomes
perceptionml --data anger_family.csv \
  --y-var gpt_sum_score \
  --x-var human_sum_score \
  --outcome-mode zero_presence \
  --sample-size 10000
EOF
echo ""

# Example 4: Social class analysis (continuous outcomes)
echo "Example 4: Continuous outcome analysis"
echo "--------------------------------------"
cat << 'EOF'
# Analyze continuous outcomes with stratified sampling
perceptionml --data essay_data.csv \
  --y-var ai_rating \
  --x-var social_class \
  --outcome-mode continuous \
  --stratify-by social_class \
  --sample-size 10000
EOF
echo ""

# Example 5: Export results for further analysis
echo "Example 5: Export results to CSV"
echo "--------------------------------"
cat << 'EOF'
# Generate HTML visualization and export all data
perceptionml --data data.csv \
  --export-csv \
  --output-dir ./my_results

# This creates:
# - my_results/analysis.html (interactive visualization)
# - my_results/export_*/  (CSV files with all results)
EOF
echo ""

echo "=================================================="
echo "Note: Example datasets used in test scripts:"
echo "- anger_family.csv: Text with anger ratings"
echo "- essay_analysis_data.csv: Essays with social class"
echo "(NCDS dataset not included in GitHub)"
echo "=================================================="