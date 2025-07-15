# BASIC MODE Quick Start Guide

## 🚀 Run the Demo Script

To see BASIC MODE in action, run:

```bash
cd /path/to/perceptionML
bash basic_mode/run_basic_mode.sh
```

## 📋 Most Common Commands

### 1. Analyze Your Precomputed Embeddings (Fastest)

```bash
python -m perceptionML.basic_mode \
  --embeddings your_embeddings.csv \
  --treatment social_class \
  --outcome ai_rating \
  --precomputed
```

### 2. Analyze Raw Text Data

```bash
python -m perceptionML.basic_mode \
  --data your_data.csv \
  --treatment social_class \
  --outcome ai_rating \
  --text essay_text \
  --id essay_id
```

### 3. Reproduce Paper Results

```bash
python -m perceptionML.basic_mode \
  --embeddings fully_anonymized_embeddings_minilm.csv \
  --treatment social_class \
  --outcome ai_rating \
  --precomputed \
  --random-seed 42
```

## 📊 Output Location

Results are saved to a timestamped directory:
- `basic_mode_results_YYYYMMDD_HHMMSS/`
  - `results_table.csv` - Main results table
  - `embeddings.csv` - Generated embeddings
  - `pca_components.csv` - PCA features
  - `feature_selections.csv` - Selected features
  - And more...

## 🐍 Python Usage

```python
from perceptionML.basic_mode import BasicAnalysis

analysis = BasicAnalysis()
results = analysis.run(
    data_path="data.csv",
    treatment_col="social_class",
    outcome_col="ai_rating",
    text_col="essay_text"
)
```

## 📦 Installation

If you haven't installed dependencies:

```bash
pip install -r basic_mode/requirements.txt
```

## ❓ Help

```bash
python -m perceptionML.basic_mode --help
```

## 📖 Full Documentation

See `basic_mode/README.md` for complete documentation.