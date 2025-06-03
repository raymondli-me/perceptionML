#!/usr/bin/env python3
"""
Test 1: MiniLM on Reddit anger data
"""

import subprocess
import sys

print("================================================")
print("Test 1: MiniLM-L6-v2 on Reddit Anger Data")
print("================================================")
print()

cmd = [
    "perceptionml",
    "--data", "anger_family.csv",
    "--y-var", "gpt_sum_score",
    "--x-var", "human_sum_score",
    "--control-vars", "num_raters",
    "--embedding-model", "sentence-transformers/all-MiniLM-L6-v2",
    "--num-gpus", "4",
    "--batch-size", "32",
    "--sample-size", "10000",
    "--auto-cluster", "descriptions",
    "--outcome-mode", "zero_presence",
    "--stratify-by", "human_sum_score",
    "--output", "minilm_reddit_analysis.html"
]

try:
    subprocess.run(cmd, check=True)
    print("\n✅ MiniLM Reddit analysis complete!")
except subprocess.CalledProcessError as e:
    print(f"\n❌ Error: {e}")
    sys.exit(1)