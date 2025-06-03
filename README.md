# PerceptionML

**Discover how language shapes perception through advanced text embedding analysis**

[![PyPI version](https://badge.fury.io/py/perceptionml.svg)](https://badge.fury.io/py/perceptionml)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## Vision

PerceptionML bridges the gap between what people write and how it's perceived. By examining text at variable resolutions—from individual word choices to broad semantic themes—it reveals the hidden patterns that shape human and AI judgment.

Whether you're studying social bias, analyzing customer feedback, or researching how AI interprets human writing, PerceptionML provides tools to explore perception at every scale:

- 🔍 **Discover semantic topics** in large text corpora automatically
- 📊 **Measure causal effects** of language patterns on outcomes
- 🎯 **Identify which textual features** drive specific perceptions
- 🌐 **Visualize the landscape** of your text data in interactive 3D

## Key Features

- **Zero-configuration analysis** - Just point to your CSV and go
- **Multiple embedding models** - From fast MiniLM to powerful NVIDIA models
- **Causal inference built-in** - DML (Double Machine Learning) analysis
- **Interactive visualizations** - Explore your data in 3D with rich tooltips
- **Flexible outcome analysis** - Binary, continuous, or zero-presence modes
- **Production ready** - Multi-GPU support, automatic batching, progress tracking

## Installation

### From PyPI (Recommended)

```bash
pip install perceptionml
```

### From Source

```bash
git clone https://github.com/raymondli-me/perceptionml.git
cd perceptionml
pip install -e .
```

### System Requirements

- Python 3.8 or higher
- 4GB+ RAM for typical datasets
- CUDA-capable GPU recommended for faster embeddings (but not required)

## Quick Start

### 1. Simplest Usage - Just Add Data!

```bash
perceptionml --data your_data.csv
```

PerceptionML automatically:
- Detects your text column
- Finds or creates ID columns
- Identifies outcome variables
- Optimizes all parameters

### 2. Model Comparison Example

Compare fast vs. high-quality embeddings:

```bash

# 384-dim embeddings
perceptionml --data anger_family.csv \
       --y-var gpt_sum_score \
       --x-var human_sum_score \
       --control-vars num_raters \
       --embedding-model sentence-transformers/all-MiniLM-L6-v2 \
       --num-gpus 4 \
       --batch-size 32 \
       --sample-size 10000 \
       --auto-cluster descriptions \
       --outcome-mode zero_presence \
       --stratify-by human_sum_score \
       --output minilm_reddit_analysis.html


# 4096-dim embeddings
perceptionml --data anger_family.csv \
       --y-var gpt_sum_score \
       --x-var human_sum_score \
       --control-vars num_raters \
       --embedding-model nvidia/NV-Embed-v2 \
       --num-gpus 4 \
       --batch-size 8 \
       --sample-size 10000 \
       --auto-cluster descriptions \
       --outcome-mode zero_presence \
       --stratify-by human_sum_score \
       --output nvidia_reddit_analysis.html

```

### 3. Real-World Examples

#### Social Class Perception (see National Child Development Study)
```bash
perceptionml --data essay_analysis_data.csv \
       --y-var ai_rating \
       --x-var social_class \
       --embedding-model sentence-transformers/all-MiniLM-L6-v2 \
       --num-gpus 4 \
       --batch-size 32 \
       --sample-size 10000 \
       --auto-cluster descriptions \
       --outcome-mode continuous \
       --stratify-by social_class \
       --output minilm_social_class_analysis.html

perceptionml --data essay_analysis_data.csv \
       --y-var ai_rating \
       --x-var social_class \
       --embedding-model nvidia/NV-Embed-v2 \
       --num-gpus 4 \
       --batch-size 8 \
       --sample-size 10000 \
       --auto-cluster descriptions \
       --outcome-mode continuous \
       --stratify-by social_class \
       --output nvidia_social_class_analysis.html


```
See `examples/quickstart.sh` for more examples.

## Understanding Your Results

The interactive HTML output provides insights at multiple resolutions:

1. **3D Landscape**: Navigate your text data in semantic space, with each point representing a document
2. **Topic Analysis**: Zoom into clusters to see defining keywords and statistics
3. **Causal Effects**: Understand which semantic patterns drive outcomes at different scales
4. **Feature Importance**: Identify the principal variations that matter most
5. **Interactive Exploration**: Drill down from patterns to individual texts
6. **Statistical Synthesis**: Comprehensive metrics help you move from observation to understanding

## Advanced Usage

### Configuration Files

For complex analyses, use YAML configuration:

```yaml
pipeline:
  name: "Customer Feedback Analysis"
  embedding_model: "nvidia/NV-Embed-v2"

data:
  text_column: "review_text"
  outcomes:
    - name: "satisfaction"
      type: "continuous"
      range: [1, 5]

analysis:
  pca_components: 200
  hdbscan_min_cluster_size: 30
```

Run with:
```bash
perceptionml --config config.yaml --data reviews.csv
```

### Programmatic Usage

```python
from pipeline.cli import TextEmbeddingPipeline

# Initialize pipeline
pipeline = TextEmbeddingPipeline(config_path="config.yaml")

# Run analysis
output_path = pipeline.run(
    data_path="data.csv",
    output_name="analysis.html"
)
```

## How It Works: The TRACES Framework

PerceptionML employs the TRACES methodology to reveal patterns at multiple resolutions:

1. **Transform**: Convert text to high-dimensional embeddings using state-of-the-art models
2. **Reduce**: Apply dimensionality reduction to find the most meaningful variations
3. **Analyze**: Measure causal effects using Double Machine Learning (DML)
4. **Cluster**: Discover semantic topics with adaptive HDBSCAN clustering
5. **Explore**: Navigate the data landscape through interactive 3D visualization
6. **Synthesize**: Interpret the statistical patterns to understand perception drivers

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## Citation

If you use PerceptionML in your research, please cite:

```bibtex
@software{perceptionml2024,
  author = {Li, Raymond V.},
  title = {PerceptionML: Discovering The Why In AI Perception},
  year = {2024},
  url = {https://github.com/raymondli-me/perceptionml}
}
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Support

- 📧 Email: raymond@raymondli.me
- 🐛 Issues: [GitHub Issues](https://github.com/raymondli-me/perceptionml/issues)
- 📖 Docs: [Full Documentation](https://perceptionml.readthedocs.io) (coming soon)

---

Built with ❤️ for researchers, data scientists, and anyone curious about how language shapes our world.

*PerceptionML: Understanding the why in AI perception.*
