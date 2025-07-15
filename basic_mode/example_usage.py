"""
Example usage of BASIC MODE Python API
"""

from perceptionML.basic_mode import BasicAnalysis
import pandas as pd

# Example 1: Using raw text data
def example_raw_text():
    """Example using raw text data"""
    print("Example 1: Analyzing raw text data")
    
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
        treatment_col="social_class",  # X variable
        outcome_col="ai_rating",       # Y variable
        text_col="essay_text",
        id_col="essay_id",
        precomputed_embeddings=False
    )
    
    # Results are automatically displayed and saved
    # Access the results DataFrame
    print(f"\nTotal models analyzed: {len(results)}")
    print(f"Results saved to: {analysis.output_dir}")
    
    return results


# Example 2: Using precomputed embeddings
def example_precomputed_embeddings():
    """Example using precomputed embeddings"""
    print("\nExample 2: Using precomputed embeddings")
    
    # Initialize analyzer
    analysis = BasicAnalysis(
        n_pca_components=200,
        n_top_features=6,
        random_seed=42
    )
    
    # Run analysis with precomputed embeddings
    results = analysis.run(
        embeddings_path="embeddings_minilm.csv",
        treatment_col="social_class",
        outcome_col="ai_rating",
        id_col="id",
        precomputed_embeddings=True
    )
    
    return results


# Example 3: Custom analysis with specific parameters
def example_custom_analysis():
    """Example with custom parameters"""
    print("\nExample 3: Custom analysis parameters")
    
    # Initialize with custom parameters
    analysis = BasicAnalysis(
        embedding_model="nvidia/NV-Embed-v2",  # Use larger model
        n_pca_components=100,                  # Fewer PCA components
        n_top_features=10,                     # More top features
        n_folds=10,                           # More CV folds
        random_seed=123,                      # Different seed
        output_dir="custom_results"           # Custom output directory
    )
    
    # Run analysis
    results = analysis.run(
        data_path="your_data.csv",
        treatment_col="treatment",
        outcome_col="outcome",
        text_col="text",
        precomputed_embeddings=False
    )
    
    # Access specific results
    baseline = results[results['Features'] == 'N/A'].iloc[0]
    print(f"\nBaseline coefficient: {baseline['Coeff (θ)']:.4f}")
    
    # Get best performing model
    best_model = results.loc[results['Reduction (vs baseline)'].str.rstrip('%').astype(float).idxmax()]
    print(f"Best model: {best_model['Model']} {best_model['Learner/Selector']} "
          f"with {best_model['Reduction (vs baseline)']} reduction")
    
    return results


# Example 4: Programmatic access to components
def example_component_access():
    """Example accessing individual components"""
    print("\nExample 4: Accessing individual components")
    
    from perceptionML.basic_mode import EmbeddingGenerator, DMLAnalyzer
    import numpy as np
    
    # Generate embeddings only
    embedder = EmbeddingGenerator(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        batch_size=32,
        normalize=True,
        random_seed=42
    )
    
    texts = ["This is a sample text.", "Another example sentence."]
    embeddings = embedder.generate_embeddings(texts)
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Use DML analyzer directly
    dml = DMLAnalyzer(n_folds=5, random_seed=42)
    
    # Simulate some data
    np.random.seed(42)
    X = np.random.randn(100)  # Treatment
    y = np.random.randn(100)  # Outcome
    Z = np.random.randn(100, 10)  # Features
    
    # Fit DML
    result = dml.fit_dml(X, y, Z, learner='xgboost')
    print(f"\nDML coefficient: {result['coefficient']:.4f}")
    print(f"Standard error: {result['se']:.4f}")
    print(f"P-value: {result['p_value']:.4e}")
    
    return embeddings, result


if __name__ == "__main__":
    # Run examples (comment out those without data files)
    
    # example_raw_text()
    # example_precomputed_embeddings()
    # example_custom_analysis()
    
    # This example works without data files
    embeddings, dml_result = example_component_access()