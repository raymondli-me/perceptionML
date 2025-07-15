# PerceptionML BASIC MODE

A lightweight, reproducible implementation of the Double Machine Learning with Language Model Embeddings (DML-LME) framework for analyzing how text perception varies across different contexts.

## Overview

BASIC MODE provides a streamlined command-line tool that reproduces the exact analysis from our research, focusing on:
- **MiniLM embeddings** (384 dimensions) with L2 normalization
- **Double Machine Learning (DML)** causal inference
- **Feature selection** using 5 methods (XGBoost, Lasso, Ridge, OLS, Mutual Information)
- **Comprehensive results table** matching the paper's format
- **Full data export** for reproducibility

## Key Features

- ✅ **Exact reproducibility** with fixed random seeds (default: 42)
- ✅ **Command-line interface** accepting 4 core arguments
- ✅ **Python and R wrappers** for programmatic access
- ✅ **Precomputed embeddings support** for faster iteration
- ✅ **Comprehensive CSV exports** of all intermediate results
- ✅ **No visualization dependencies** - pure analysis output

## Installation

```bash
# From the perceptionML directory
cd perceptionML
pip install -e .

# Install additional dependencies for BASIC MODE
pip install sentence-transformers scikit-learn xgboost pandas numpy scipy
```

## Quick Start

### Command Line Usage

#### Generate embeddings from raw text:
```bash
python -m perceptionML.basic_mode \
  --data your_data.csv \
  --treatment social_class \
  --outcome ai_rating \
  --text essay_text \
  --id essay_id
```

#### Use precomputed embeddings:
```bash
python -m perceptionML.basic_mode \
  --embeddings embeddings_minilm.csv \
  --treatment social_class \
  --outcome ai_rating \
  --precomputed
```

### Python API Usage

```python
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
    data_path="your_data.csv",
    treatment_col="social_class",
    outcome_col="ai_rating", 
    text_col="essay_text",
    id_col="essay_id"
)
```

### R Usage

```r
source("basic_mode_wrapper.R")
init_perceptionml_basic()

results <- run_basic_analysis(
  data_path = "your_data.csv",
  treatment_col = "social_class",
  outcome_col = "ai_rating",
  text_col = "essay_text",
  id_col = "essay_id"
)
```

## Core Arguments

The four core arguments match the main perceptionML framework:

1. **treatment** (X variable): The independent variable/treatment (e.g., social class)
2. **outcome** (Y variable): The dependent variable/outcome (e.g., AI rating)
3. **text**: The text column containing documents to analyze
4. **id** (optional): Unique identifier for each document

## Analysis Pipeline

1. **Baseline OLS**: Simple regression without controls
2. **Full Embeddings**: Analysis with all 384 MiniLM dimensions
3. **PCA Reduction**: Reduce to 200 principal components
4. **Feature Selection**: Select top 6 PCs using 5 methods
5. **DML Analysis**: Causal inference with cross-fitting

## Output Structure

```
basic_mode_results_YYYYMMDD_HHMMSS/
├── results_table.csv           # Main results matching paper format
├── embeddings.csv              # Generated embeddings (if applicable)
├── pca_components.csv          # 200 PCA components
├── pca_variance_explained.csv  # Variance explained by each PC
├── feature_selections.csv      # Selected PC indices for each method
├── raw_data.csv               # Treatment and outcome variables
└── analysis_params.json        # Parameters for reproducibility
```

## Results Table Format

The results table includes:
- **Model**: OLS or DML
- **Learner/Selector**: Method used (OLS, XGBoost, Lasso, Ridge, MI)
- **Embedding**: MiniLM or NVembed
- **Features**: Full, 200 PCs, or Top 6
- **Coeff (θ)**: Effect coefficient
- **Robust SE**: Heteroskedasticity-robust standard errors
- **p-value**: Two-tailed significance test
- **95% CI**: Confidence interval
- **R² values**: Full and cross-validated R² for both outcomes
- **G, C correlations**: Cross-fitted prediction/residual correlations
- **Reduction**: Percentage mediation vs baseline

## Reproducibility

To exactly reproduce our paper results:

```bash
# Using our precomputed embeddings
python -m perceptionML.basic_mode \
  --embeddings fully_anonymized_embeddings_minilm.csv \
  --treatment social_class \
  --outcome ai_rating \
  --precomputed \
  --random-seed 42
```

Key settings for reproducibility:
- Random seed: 42 (default)
- MiniLM model: `sentence-transformers/all-MiniLM-L6-v2`
- Batch size: 32
- Max sequence length: 512
- L2 normalization: Enabled
- PCA components: 200
- Top features: 6
- CV folds: 5

## Advanced Options

### Custom Parameters
```bash
python -m perceptionML.basic_mode \
  --data mydata.csv \
  --treatment X --outcome Y --text text_col \
  --embedding-model "nvidia/NV-Embed-v2" \
  --pca-components 100 \
  --top-features 10 \
  --cv-folds 10 \
  --random-seed 123 \
  --output-dir custom_results
```

### Skip CSV Export
```bash
python -m perceptionML.basic_mode \
  --embeddings embeddings.csv \
  --treatment X --outcome Y \
  --precomputed \
  --no-export
```

## Technical Details

### Double Machine Learning (DML)

We implement the DML framework following Chernozhukov et al. (2018):

1. **Cross-fitting**: 5-fold cross-validation to avoid overfitting
2. **First stage**: Predict X and Y from embeddings Z
3. **Second stage**: Regress residuals ê_Y on ê_X
4. **Sandwich estimator**: Robust standard errors

Point estimate: θ̂ = Σ(ê_Xi × ê_Yi) / Σ(ê_Xi²)

Variance: σ̂²_θ = (1/n²) × Σ(ê_Xi × ê_Yi - θ̂ × ê_Xi²)² / (1/n × Σ ê_Xi²)²

### Feature Selection Methods

1. **XGBoost**: Feature importance from gradient boosting
2. **Lasso**: L1 regularization with cross-validated alpha
3. **Ridge**: L2 regularization with cross-validated alpha
4. **OLS**: Absolute coefficient magnitudes
5. **Mutual Information**: Non-parametric relevance scores

Selection uses alternating approach:
- Select 3 PCs predicting treatment (X)
- Select 3 PCs predicting outcome (Y)
- Combine for 6 unique PCs

### Embedding Generation

MiniLM settings:
- Model: `all-MiniLM-L6-v2`
- Dimensions: 384
- Pooling: Mean pooling
- Normalization: L2 (unit vectors)
- Max length: 512 tokens

## Troubleshooting

### Memory Issues
- Reduce batch size: Add `batch_size=16` to EmbeddingGenerator
- Process in chunks for large datasets

### GPU Support
- Automatically uses GPU if available
- Force CPU: Set `device='cpu'` in EmbeddingGenerator

### Missing Dependencies
```bash
pip install sentence-transformers>=2.2.0
pip install xgboost>=1.7.0
pip install scikit-learn>=1.0.0
```

## Citation

If you use BASIC MODE in your research, please cite:

```bibtex
@article{perceptionml2024,
  title={PerceptionML: Double Machine Learning with Language Model Embeddings},
  author={Your Name et al.},
  journal={Journal Name},
  year={2024}
}
```

## License

Same as perceptionML main package.

## Support

For issues or questions:
- GitHub Issues: [perceptionML repository]
- Documentation: See main perceptionML docs