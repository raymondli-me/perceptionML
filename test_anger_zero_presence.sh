#!/bin/bash

# Test anger dataset with zero-presence mode and category focus feature
# Using NVIDIA NV-Embed-v2 model with quad GPU support

echo "Testing anger family dataset with zero-presence mode..."
echo "Model: nvidia/NV-Embed-v2"
echo "GPU Mode: Quad GPU (0,1,2,3)"
echo "Sample size: 10,000"
echo "Clustering: Many topics mode"
echo ""
echo "Alternative: For manual control, use these parameters instead:"
echo "  --min-cluster-size 20 --min-samples 3 --umap-neighbors 10 --umap-min-dist 0.0"
echo ""

# Set CUDA devices for quad GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Run the pipeline with zero-presence mode config
# Using manual parameters for maximum topic granularity
python3 run_pipeline.py \
    --config configs/anger_zero_presence.yaml \
    --data data_emotion/emotion_families/anger_family.csv \
    --output anger_zero_presence_nvidia_test.html \
    --export-csv \
    --export-dir exports_anger_nvidia_test \
    --embedding-model nvidia/NV-Embed-v2 \
    --num-gpus 4 \
    --batch-size 8 \
    --min-cluster-size 20 \
    --min-samples 3 \
    --umap-neighbors 10 \
    --umap-min-dist 0.0

echo ""
echo "Test complete! Check anger_zero_presence_nvidia_test.html for results."
echo ""
echo "The visualization should show:"
echo "1. Category focus buttons in the left panel (instead of threshold controls)"
echo "2. Zero-presence statistics showing:"
echo "   - Zero (absent): count and percentage"
echo "   - Non-zero (present): count and percentage"
echo "3. Gallery mode with presence-based categories:"
echo "   - Both Absent (0, 0)"
echo "   - Human Anger Present Only"
echo "   - GPT Anger Present Only"  
echo "   - Both Present (>0, >0)"
echo "   - Agreement Only"
echo ""
echo "To test the category focus feature:"
echo "- Click 'Both Absent (0, 0)' to highlight texts where neither detected anger"
echo "- Click 'Human Anger Present Only' to see texts only humans rated as angry"
echo "- Click 'GPT Anger Present Only' to see texts only GPT rated as angry"
echo "- Click 'Both Present (>0, >0)' to see texts both rated as angry"
echo "- Click 'Agreement Only' to see where human and GPT agreed (both 0 or both >0)"
echo "- Click 'Show All Categories' to return to normal view"
echo ""
echo "Points will be colored:"
echo "- Dark gray: Both absent (no anger detected)"
echo "- Magenta: Human only detected anger"
echo "- Cyan: GPT only detected anger"
echo "- Green: Both detected anger"
echo "- Dimmed dark gray: Non-focused categories when using focus mode"