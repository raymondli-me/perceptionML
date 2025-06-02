# perceptionML

A text embedding analysis pipeline for perception modeling and topic discovery.

## Features

- Generate text embeddings using state-of-the-art models (Sentence Transformers, OpenAI, etc.)
- Dimensionality reduction with UMAP or PCA
- Advanced clustering with HDBSCAN
- Interactive HTML visualizations
- Topic analysis and statistics
- Support for zero-presence analysis and category comparisons
- Multi-GPU support for large datasets

## Installation

```bash
pip install perceptionml
```

## Quick Start

### Simplest Usage - No Configuration Needed!

Just point to your CSV file with text:

```bash
perceptionml --data your_data.csv
```

perceptionML will automatically:
- Detect your text column (longest text)
- Find or create an ID column
- Identify numeric columns as outcomes
- Generate synthetic outcomes if no numeric columns exist
- Use optimal settings for finding many detailed topics

### What Your Data Should Look Like

Minimal CSV (just text):
```csv
text
"This is my first document about..."
"Another document with different content..."
```

CSV with outcomes to analyze:
```csv
id,text,sentiment_score,rating
1,"Great product, highly recommend!",0.95,5
2,"Terrible experience, would not buy again",-0.87,1
```

### Basic Options

```bash
# Specify output file name
perceptionml --data your_data.csv --output my_analysis.html

# Sample large datasets
perceptionml --data your_data.csv --sample-size 10000

# Use specific embedding model
perceptionml --data your_data.csv --embedding-model nvidia/NV-Embed-v2

# Export results to CSV
perceptionml --data your_data.csv --export-csv
```

### Advanced Usage

For more control, you can:

1. **Use configuration files** for complex setups
2. **Adjust clustering granularity**:
   ```bash
   # Many small topics (default)
   perceptionml --data your_data.csv --auto-cluster many
   
   # Medium-sized topics  
   perceptionml --data your_data.csv --auto-cluster medium
   
   # Few large topics
   perceptionml --data your_data.csv --auto-cluster few
   ```

3. **Override specific parameters**:
   ```bash
   perceptionml --data your_data.csv \
       --min-cluster-size 30 \
       --umap-neighbors 15
   ```

## Understanding the Output

The HTML visualization shows:
- **3D scatter plot** of your texts, clustered by topic
- **Topic keywords** extracted from each cluster
- **Statistics** about outcomes in different regions
- **Interactive controls** to explore the data

Click on points to read the original texts. Use the controls to filter by outcome values or focus on specific topics.

## Requirements

- Python 3.8+
- CUDA-capable GPU recommended for faster embedding generation
- 4GB+ RAM for typical datasets

## Support

- Documentation: [https://github.com/raymondli/perceptionml](https://github.com/raymondli/perceptionml)
- Issues: [https://github.com/raymondli/perceptionml/issues](https://github.com/raymondli/perceptionml/issues)
- Author: Raymond V. Li (raymond@raymondli.me)

## License

See LICENSE file for details.