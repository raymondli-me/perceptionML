#!/usr/bin/env python3
"""
Test PerceptionML with NVIDIA NV-Embed-v2 model (high-quality, 4096-dim embeddings)
Based on test_both_models.sh
"""

import subprocess
import sys

# Configuration
DATA_FILE = "anger_family.csv"
Y_VAR = "gpt_sum_score"
X_VAR = "human_sum_score"
CONTROL_VARS = "num_raters"
SAMPLE_SIZE = "10000"
AUTO_CLUSTER = "descriptions"
OUTCOME_MODE = "zero_presence"
STRATIFY_BY = "human_sum_score"

print("================================================")
print("Running PerceptionML with NVIDIA NV-Embed-v2 model...")
print("================================================")
print()

# Build the command
cmd = [
    "perceptionml",
    "--data", DATA_FILE,
    "--y-var", Y_VAR,
    "--x-var", X_VAR,
    "--control-vars", CONTROL_VARS,
    "--embedding-model", "nvidia/NV-Embed-v2",
    "--num-gpus", "4",
    "--batch-size", "8",  # Smaller batch size for larger model
    "--sample-size", SAMPLE_SIZE,
    "--auto-cluster", AUTO_CLUSTER,
    "--outcome-mode", OUTCOME_MODE,
    "--stratify-by", STRATIFY_BY,
    "--output", "nvidia_analysis.html"
]

# Run the command
try:
    result = subprocess.run(cmd, check=True)
    print("\n✅ NVIDIA NV-Embed-v2 analysis complete!")
    print("Output saved to: output/nvidia_analysis.html")
except subprocess.CalledProcessError as e:
    print(f"\n❌ Error running perceptionml: {e}")
    sys.exit(1)