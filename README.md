# PerceptionML

**Analyze text perception using Double Machine Learning with Language Model Embeddings**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🚀 Two Modes for Different Needs

### Basic Mode (Recommended for Most Users)
- **Simple**: One command to reproduce paper results
- **Fast**: Optimized for MiniLM embeddings
- **Reproducible**: Fixed random seeds ensure consistent results
- **Complete**: All analyses from the paper in one pipeline

### Advanced Mode  
- **Flexible**: Multiple embedding models and configurations
- **Powerful**: GPU acceleration and batch processing
- **Customizable**: YAML-based configuration system
- **Visual**: Interactive 3D visualizations

## 📦 Installation

```bash
# Clone repository
git clone https://github.com/yourusername/perceptionML.git
cd perceptionML

# Install dependencies
pip install -r requirements.txt

# Install package
pip install -e .
```

## 🎯 Quick Start

### Basic Mode - Command Line

```bash
# Using precomputed embeddings
python -m perceptionML basic \
    --embeddings embeddings.csv \
    --treatment social_class \
    --outcome ai_rating

# Using raw text
python -m perceptionML basic \
    --data data.csv \
    --treatment X \
    --outcome Y \
    --text text_column
```

### Basic Mode - Python API

```python
from perceptionML import BasicAnalysis

# Initialize and run analysis
analysis = BasicAnalysis()
results = analysis.run(
    data_path="data.csv",
    treatment_col="X",
    outcome_col="Y",
    text_col="text"
)
```

### Basic Mode - R Interface

```r
# Save this as run_perceptionML.R
library(reticulate)

# Import perceptionML
perceptionML <- import("perceptionML")

# Create analysis object
analysis <- perceptionML$BasicAnalysis(
  n_pca_components = 200L,
  n_top_features = 6L,
  random_seed = 42L
)

# Run analysis
results <- analysis$run(
  data_path = "data.csv",
  treatment_col = "social_class",
  outcome_col = "ai_rating",
  text_col = "essay_text"
)

# Results are now in R as a data frame
print(results)
```

### Basic Mode - R Command Line Wrapper

```r
# Save this as perception_analysis.R
#!/usr/bin/env Rscript

# Simple R wrapper for command line usage
args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 3) {
  cat("Usage: Rscript perception_analysis.R <data.csv> <treatment> <outcome> [text_column]\n")
  quit(status = 1)
}

# Run analysis using system call
cmd <- sprintf(
  "python -m perceptionML basic --data %s --treatment %s --outcome %s %s",
  args[1], args[2], args[3],
  ifelse(length(args) >= 4, paste("--text", args[4]), "")
)

system(cmd)
```

## 📊 Basic Mode Features

### What You Get

1. **Comprehensive Results Table** matching paper format:
   - OLS baseline (no controls)
   - OLS with full embeddings
   - DML with XGBoost, Lasso, Ridge
   - OLS/DML with PCA components
   - OLS/DML with top selected features

2. **Feature Selection Methods**:
   - XGBoost importance
   - Lasso (L1) coefficients
   - Ridge (L2) coefficients  
   - OLS coefficients
   - Mutual Information

3. **Complete Output**:
   ```
   results/
   ├── results_table.csv          # Main results
   ├── feature_selections.csv     # Selected features
   ├── embeddings.csv            # Generated embeddings
   ├── pca_components.csv        # PCA transformed data
   ├── pca_variance_explained.csv # Variance explained
   └── analysis_params.json      # For reproducibility
   ```

### Example: Reproducing Paper Results

```bash
# Download example data
wget https://example.com/miniLM_embeddings.csv

# Run analysis (bash script)
bash run_minilm_analysis.sh

# Or directly with Python
python -m perceptionML basic \
    --embeddings miniLM_embeddings.csv \
    --treatment social_class \
    --outcome ai_rating \
    --pca-components 200 \
    --top-features 6 \
    --cv-folds 5 \
    --random-seed 42
```

## 🔬 Advanced Mode

For complex analyses with multiple embeddings and visualizations:

### Command Line

```bash
# Create configuration
python -m perceptionML advanced create-config

# Run pipeline
python -m perceptionML advanced \
    --data essays.csv \
    --y-var ai_rating \
    --x-var social_class \
    --embedding-model nvidia/NV-Embed-v2 \
    --num-gpus 4 \
    --auto-cluster
```

### Configuration File

```yaml
pipeline:
  name: "Social Perception Analysis"
  embedding_model: "nvidia/NV-Embed-v2"
  
analysis:
  pca_components: 200
  feature_selection:
    - xgboost
    - lasso
    - ridge
  
output:
  create_visualizations: true
  export_embeddings: true
```

## 📚 Examples

### Example 1: Simple Analysis

```python
from perceptionML import BasicAnalysis

# Minimal example
analysis = BasicAnalysis()
results = analysis.run(
    data_path="reviews.csv",
    treatment_col="product_type",
    outcome_col="rating",
    text_col="review_text"
)

# Access results
print(results[results['Model'] == 'DML'])
```

### Example 2: Custom Parameters

```python
# Custom analysis settings
analysis = BasicAnalysis(
    embedding_model="sentence-transformers/all-mpnet-base-v2",
    n_pca_components=100,
    n_top_features=10,
    n_folds=10,
    random_seed=123
)

results = analysis.run(
    data_path="tweets.csv",
    treatment_col="sentiment",
    outcome_col="retweets",
    text_col="tweet_text",
    output_dir="tweet_analysis/"
)
```

### Example 3: Working with Precomputed Embeddings

```python
# If you already have embeddings
analysis = BasicAnalysis()
results = analysis.run(
    embeddings_path="embeddings.csv",
    treatment_col="treatment",
    outcome_col="outcome",
    precomputed_embeddings=True
)
```

## 🛠️ Methods Overview

### Double Machine Learning (DML)
- Estimates causal effects with high-dimensional controls
- Uses cross-fitting to avoid overfitting
- Provides robust standard errors

### Embedding Models
- **MiniLM**: Fast 384-dimensional embeddings (default)
- **MPNet**: Higher quality 768-dimensional embeddings
- **NV-Embed**: State-of-the-art 4096-dimensional embeddings (advanced mode)

### Feature Selection
- **Alternating selection**: Selects features predictive of both treatment and outcome
- **Multiple methods**: Compare different selection approaches
- **Top-k features**: Focus on most important dimensions

## 📖 Documentation

- [Basic Mode Guide](docs/basic_mode.md)
- [Advanced Mode Guide](docs/advanced_mode.md)
- [API Reference](docs/api_reference.md)
- [Statistical Methods](docs/methods.md)

## 📄 Citation

If you use PerceptionML in your research:

```bibtex
@software{perceptionML,
  title = {PerceptionML: Text Perception Analysis using Double Machine Learning},
  author = {Raymond Li},
  year = {2025},
  url = {https://github.com/raymondli-me/perceptionML}
}
```

## 📝 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Double Machine Learning: Chernozhukov et al. (2018)
- Sentence Transformers: Reimers & Gurevych (2019)
- XGBoost, scikit-learn, and statsmodels communities
