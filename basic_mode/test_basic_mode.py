"""
Test script for BASIC MODE
Ensures reproducibility and correct implementation
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from basic_mode import BasicAnalysis, EmbeddingGenerator, DMLAnalyzer

def test_reproducibility():
    """Test that results are reproducible with same seed"""
    print("Testing reproducibility...")
    
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 100
    
    # Create synthetic embeddings
    embeddings1 = np.random.randn(n_samples, 384).astype(np.float32)
    embeddings2 = np.random.randn(n_samples, 384).astype(np.float32)
    
    # Create outcomes
    X = np.random.randn(n_samples)
    y = 0.5 * X + np.random.randn(n_samples) * 0.5
    
    # Run DML twice with same seed
    dml = DMLAnalyzer(n_folds=5, random_seed=42)
    
    result1 = dml.fit_dml(X, y, embeddings1, learner='xgboost')
    
    # Reset and run again
    dml2 = DMLAnalyzer(n_folds=5, random_seed=42)
    result2 = dml2.fit_dml(X, y, embeddings1, learner='xgboost')
    
    # Check reproducibility
    assert np.isclose(result1['coefficient'], result2['coefficient']), \
        f"Coefficients differ: {result1['coefficient']} vs {result2['coefficient']}"
    assert np.isclose(result1['se'], result2['se']), \
        f"SEs differ: {result1['se']} vs {result2['se']}"
    
    print("✓ Reproducibility test passed")
    return True


def test_embedding_generation():
    """Test embedding generation"""
    print("\nTesting embedding generation...")
    
    # Sample texts
    texts = [
        "This is a test sentence.",
        "Another example text for embedding.",
        "Machine learning is fascinating."
    ]
    
    # Generate embeddings
    embedder = EmbeddingGenerator(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        random_seed=42
    )
    
    embeddings = embedder.generate_embeddings(texts, show_progress=False)
    
    # Check properties
    assert embeddings.shape == (3, 384), f"Wrong shape: {embeddings.shape}"
    assert embeddings.dtype == np.float32, f"Wrong dtype: {embeddings.dtype}"
    
    # Check normalization (L2 norm should be 1)
    norms = np.linalg.norm(embeddings, axis=1)
    assert np.allclose(norms, 1.0), f"Not normalized: {norms}"
    
    print("✓ Embedding generation test passed")
    print(f"  Shape: {embeddings.shape}")
    print(f"  L2 norms: {norms}")
    return True


def test_feature_selection():
    """Test feature selection methods"""
    print("\nTesting feature selection...")
    
    # Create synthetic data
    np.random.seed(42)
    n_samples = 200
    n_features = 20
    
    X_data = np.random.randn(n_samples, n_features)
    y_data = X_data[:, 0] * 2 + X_data[:, 1] * 1.5 + np.random.randn(n_samples) * 0.5
    
    dml = DMLAnalyzer(random_seed=42)
    
    # Test each selection method
    methods = ['xgboost', 'lasso', 'ridge', 'ols', 'mi']
    
    for method in methods:
        indices = dml.select_top_features(X_data, y_data, method=method, n_features=6)
        assert len(indices) == 6, f"{method}: Wrong number of features selected"
        assert 0 in indices or 1 in indices, f"{method}: Should select important features"
        print(f"  {method}: selected features {indices}")
    
    print("✓ Feature selection test passed")
    return True


def test_full_pipeline():
    """Test the full analysis pipeline with synthetic data"""
    print("\nTesting full pipeline...")
    
    # Create synthetic dataset
    np.random.seed(42)
    n_samples = 500
    
    # Generate synthetic "embeddings" (already computed)
    embeddings = np.random.randn(n_samples, 384).astype(np.float32)
    
    # Create correlated treatment and outcome
    treatment = np.random.randn(n_samples)
    # Outcome partially explained by treatment and embeddings
    outcome = 0.3 * treatment + 0.5 * embeddings[:, 0] + 0.3 * embeddings[:, 1] + \
              np.random.randn(n_samples) * 0.5
    
    # Create DataFrame
    df = pd.DataFrame(embeddings, columns=[f'embedding_{i}' for i in range(384)])
    df['treatment'] = treatment
    df['outcome'] = outcome
    df['id'] = range(n_samples)
    
    # Save to temporary file
    temp_file = 'test_embeddings.csv'
    df.to_csv(temp_file, index=False)
    
    # Run analysis
    analysis = BasicAnalysis(
        n_pca_components=50,  # Smaller for testing
        n_top_features=6,
        random_seed=42,
        output_dir='test_output'
    )
    
    results = analysis.run(
        embeddings_path=temp_file,
        treatment_col='treatment',
        outcome_col='outcome',
        id_col='id',
        precomputed_embeddings=True
    )
    
    # Check results
    assert len(results) > 0, "No results generated"
    assert 'Model' in results.columns, "Missing Model column"
    assert 'Coeff (θ)' in results.columns, "Missing coefficient column"
    
    # Check baseline exists
    baseline_rows = results[results['Features'] == 'N/A']
    assert len(baseline_rows) == 1, "Should have exactly one baseline"
    
    print("✓ Full pipeline test passed")
    print(f"  Generated {len(results)} result rows")
    
    # Cleanup
    import os
    os.remove(temp_file)
    
    return results


def run_all_tests():
    """Run all tests"""
    print("="*60)
    print("Running BASIC MODE Tests")
    print("="*60)
    
    try:
        test_reproducibility()
        test_embedding_generation()
        test_feature_selection()
        results = test_full_pipeline()
        
        print("\n" + "="*60)
        print("All tests passed! ✓")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"\nTest failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)