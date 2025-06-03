#!/usr/bin/env python3
"""
Run all 4 test combinations
"""

import subprocess
import sys

tests = [
    ("test_minilm_reddit.py", "MiniLM on Reddit anger data"),
    ("test_nvidia_reddit.py", "NVIDIA on Reddit anger data"),
    ("test_minilm_social_class.py", "MiniLM on social class essays"),
    ("test_nvidia_social_class.py", "NVIDIA on social class essays")
]

print("================================================")
print("Running All PerceptionML Tests")
print("================================================")
print()

for script, description in tests:
    print(f"\n{'='*50}")
    print(f"Running: {description}")
    print(f"{'='*50}")
    
    try:
        subprocess.run([sys.executable, script], check=True)
    except subprocess.CalledProcessError:
        print(f"❌ Failed: {description}")
        continue
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)

print("\n================================================")
print("🎉 All tests complete!")
print("================================================")
print("\nOutput files:")
print("  - minilm_reddit_analysis.html")
print("  - nvidia_reddit_analysis.html") 
print("  - minilm_social_class_analysis.html")
print("  - nvidia_social_class_analysis.html")