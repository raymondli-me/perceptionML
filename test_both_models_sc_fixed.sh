#!/bin/bash

# PerceptionML Model Comparison Script
# Tests both NVIDIA NV-Embed-v2 and MiniLM-L6-v2 models

echo "=================================================="
echo "PerceptionML Model Comparison Test"
echo "=================================================="
echo ""

# Common parameters
DATA_FILE="data_github/essay_analysis_data.csv"
Y_VAR="ai_rating"
X_VAR="social_class"
SAMPLE_SIZE="10000"
AUTO_CLUSTER="descriptions"
OUTCOME_MODE="continuous"
STRATIFY_BY="social_class"

# Test 1: MiniLM-L6-v2 (smaller, faster model)
echo "Test 1: Running with MiniLM-L6-v2 model..."
echo "------------------------------------------"
echo ""

perceptionml --data $DATA_FILE \
  --y-var $Y_VAR \
  --x-var $X_VAR \
  --embedding-model sentence-transformers/all-MiniLM-L6-v2 \
  --num-gpus 4 \
  --batch-size 32 \
  --sample-size $SAMPLE_SIZE \
  --auto-cluster $AUTO_CLUSTER \
  --outcome-mode $OUTCOME_MODE \
  --stratify-by $STRATIFY_BY \
  --output minilm_analysis.html

echo ""
echo "✅ MiniLM-L6-v2 analysis complete!"
echo ""
echo "=================================================="
echo ""

# Test 2: NVIDIA NV-Embed-v2 (larger, more powerful model)
echo "Test 2: Running with NVIDIA NV-Embed-v2 model..."
echo "-------------------------------------------------"
echo ""

perceptionml --data $DATA_FILE \
  --y-var $Y_VAR \
  --x-var $X_VAR \
  --embedding-model nvidia/NV-Embed-v2 \
  --num-gpus 4 \
  --batch-size 8 \
  --sample-size $SAMPLE_SIZE \
  --auto-cluster $AUTO_CLUSTER \
  --outcome-mode $OUTCOME_MODE \
  --stratify-by $STRATIFY_BY \
  --output nvidia_analysis.html

echo ""
echo "✅ NVIDIA NV-Embed-v2 analysis complete!"
echo ""
echo "=================================================="
echo "🎉 All tests complete!"
echo ""
echo "Output files:"
echo "  - MiniLM-L6-v2: output/minilm_analysis.html"
echo "  - NV-Embed-v2:  output/nvidia_analysis.html"
echo ""
echo "Model comparison:"
echo "  - MiniLM: 384-dim embeddings, faster processing"
echo "  - NVIDIA: 4096-dim embeddings, better quality"
echo "==================================================" 