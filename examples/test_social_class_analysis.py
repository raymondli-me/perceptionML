#!/usr/bin/env python3
"""
Test PerceptionML with social class essay data
Based on test_both_models_sc_fixed.sh
"""

import subprocess
import sys
import os

# Check if we need to copy the essay data
if not os.path.exists("essay_analysis_data.csv"):
    print("Note: essay_analysis_data.csv not found in current directory")
    print("Please copy from: ../data_github/essay_analysis_data.csv")
    print("Or download from the NCDS dataset")
    sys.exit(1)

# Configuration
DATA_FILE = "essay_analysis_data.csv"
Y_VAR = "ai_rating"
X_VAR = "social_class"
SAMPLE_SIZE = "10000"
AUTO_CLUSTER = "descriptions"
OUTCOME_MODE = "continuous"  # Different from anger analysis
STRATIFY_BY = "social_class"

print("================================================")
print("Social Class Essay Analysis with MiniLM")
print("================================================")
print()

# Build the command
cmd = [
    "perceptionml",
    "--data", DATA_FILE,
    "--y-var", Y_VAR,
    "--x-var", X_VAR,
    "--embedding-model", "sentence-transformers/all-MiniLM-L6-v2",
    "--num-gpus", "4",
    "--batch-size", "32",
    "--sample-size", SAMPLE_SIZE,
    "--auto-cluster", AUTO_CLUSTER,
    "--outcome-mode", OUTCOME_MODE,
    "--stratify-by", STRATIFY_BY,
    "--output", "social_class_analysis.html"
]

# Run the command
try:
    result = subprocess.run(cmd, check=True)
    print("\n✅ Social class analysis complete!")
    print("Output saved to: output/social_class_analysis.html")
except subprocess.CalledProcessError as e:
    print(f"\n❌ Error running perceptionml: {e}")
    sys.exit(1)