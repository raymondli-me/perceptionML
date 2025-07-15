"""
Main analysis class for BASIC MODE
Reproduces the exact analysis from the notebook
"""

import os
# Set threading environment variables BEFORE importing XGBoost
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV, RidgeCV
import xgboost as xgb
from typing import Dict, List, Optional, Union, Tuple
import logging
from pathlib import Path
from datetime import datetime
import json

from .embeddings import EmbeddingGenerator
from .dml import DMLAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BasicAnalysis:
    """Main analysis class for BASIC MODE"""
    
    def __init__(self,
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 n_pca_components: int = 200,
                 n_top_features: int = 6,
                 n_folds: int = 5,
                 random_seed: int = 42,
                 output_dir: Optional[str] = None):
        """
        Initialize BASIC MODE analysis
        
        Args:
            embedding_model: Model name for embeddings
            n_pca_components: Number of PCA components
            n_top_features: Number of top features to select
            n_folds: Number of CV folds for DML
            random_seed: Random seed for reproducibility
            output_dir: Directory to save outputs (if None, creates timestamped dir)
        """
        self.embedding_model = embedding_model
        self.n_pca_components = n_pca_components
        self.n_top_features = n_top_features
        self.n_folds = n_folds
        self.random_seed = random_seed
        
        # Set random seeds
        np.random.seed(random_seed)
        # Also set random state for other libraries
        import random
        random.seed(random_seed)
        
        # Setup output directory
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"basic_mode_results_{timestamp}"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.embedding_generator = EmbeddingGenerator(
            model_name=embedding_model,
            random_seed=random_seed
        )
        self.dml_analyzer = DMLAnalyzer(n_folds=n_folds, random_seed=random_seed)
        
        # Store results
        self.results = []
        self.feature_selections = {}
        
    def run(self,
            data_path: Optional[str] = None,
            embeddings_path: Optional[str] = None,
            treatment_col: Optional[str] = None,
            outcome_col: Optional[str] = None,
            text_col: Optional[str] = None,
            id_col: Optional[str] = None,
            precomputed_embeddings: bool = False) -> pd.DataFrame:
        """
        Run the full analysis pipeline
        
        Args:
            data_path: Path to data CSV
            embeddings_path: Path to precomputed embeddings CSV
            treatment_col: Name of treatment column (X variable)
            outcome_col: Name of outcome column (Y variable)
            text_col: Name of text column (for embedding generation)
            id_col: Name of ID column
            precomputed_embeddings: Whether embeddings are precomputed
            
        Returns:
            DataFrame with comprehensive results
        """
        logger.info("Starting BASIC MODE analysis...")
        
        # Load data
        if precomputed_embeddings:
            df = pd.read_csv(embeddings_path)
            # Extract embeddings
            embedding_cols = [col for col in df.columns 
                            if col.startswith(('embedding_', 'dim_', 'embed_'))]
            if not embedding_cols:
                # Assume all numeric columns except treatment/outcome are embeddings
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                exclude = [treatment_col, outcome_col, id_col] if id_col else [treatment_col, outcome_col]
                embedding_cols = [col for col in numeric_cols if col not in exclude]
                
            embeddings = df[embedding_cols].values
            X = df[treatment_col].values
            y = df[outcome_col].values
            
        else:
            # Load raw data and generate embeddings
            df = pd.read_csv(data_path)
            texts = df[text_col]
            X = df[treatment_col].values
            y = df[outcome_col].values
            
            # Generate embeddings
            embeddings = self.embedding_generator.generate_embeddings(texts)
            
            # Save embeddings
            embedding_df = pd.DataFrame(
                embeddings,
                columns=[f'embedding_{i}' for i in range(embeddings.shape[1])]
            )
            embedding_df[treatment_col] = X
            embedding_df[outcome_col] = y
            if id_col:
                embedding_df[id_col] = df[id_col]
            
            embedding_df.to_csv(self.output_dir / 'embeddings.csv', index=False)
            logger.info(f"Saved embeddings to {self.output_dir / 'embeddings.csv'}")
        
        # Run analyses
        logger.info("Running analyses...")
        
        # 1. Baseline OLS (no controls)
        self._run_baseline_ols(X, y)
        
        # 2. Full embeddings analysis
        self._run_full_embeddings_analysis(X, y, embeddings)
        
        # 3. PCA analysis
        pca_features = self._run_pca_analysis(X, y, embeddings)
        
        # 4. Top features analysis
        self._run_top_features_analysis(X, y, pca_features)
        
        # Create results table
        results_df = self._create_results_table()
        
        # Save all outputs
        self._save_outputs(results_df, X, y, embeddings, pca_features)
        
        # Display results
        try:
            self._display_results(results_df)
        except Exception as e:
            logger.warning(f"Display error (results already saved): {e}")
        
        return results_df
    
    def _run_baseline_ols(self, X: np.ndarray, y: np.ndarray):
        """Run baseline OLS without controls"""
        logger.info("Running baseline OLS...")
        
        result = self.dml_analyzer.fit_ols(X.reshape(-1, 1), y)
        
        self.results.append({
            'Model': 'OLS',
            'Learner/Selector': 'OLS',
            'Embedding': 'None',
            'Features': 'N/A',
            'Coeff (θ)': result['coefficient'],
            'Robust SE': result['se'],
            'p-value': result['p_value'],
            '95% CI': f"[{result['ci'][0]:.4f}, {result['ci'][1]:.4f}]",
            'R²(Y) Full': result['r2'],
            'R²(Y) CV Mean': 'N/A',
            'R²(Y) Folds': 'N/A',
            'R²(X) Full': 0.0,
            'R²(X) CV Mean': 'N/A',
            'R²(X) Folds': 'N/A',
            'G': 'N/A',
            'C': 'N/A',
            'Corr(X,X̂)': 'N/A',
            'Corr(Y,Ŷ)': 'N/A',
            'Reduction (vs baseline)': '0.00%'
        })
        
        self.baseline_coef = result['coefficient']
        
    def _run_full_embeddings_analysis(self, X: np.ndarray, y: np.ndarray, embeddings: np.ndarray):
        """Run analysis with full embeddings"""
        logger.info("Running full embeddings analysis...")
        
        # Determine embedding type
        if embeddings.shape[1] == 384:
            embed_name = 'Minilm'
        elif embeddings.shape[1] == 4096:
            embed_name = 'Nvembed'
        else:
            embed_name = f'Custom{embeddings.shape[1]}'
        
        # OLS with full embeddings as controls
        # Import the notebook-style OLS implementation
        from .dml_notebook import NotebookDMLAnalyzer
        notebook_analyzer = NotebookDMLAnalyzer(n_folds=self.n_folds, random_state=self.random_seed)
        result_dict = notebook_analyzer.run_ols(embeddings, X, y, 'Full', embed_name)
        
        # Convert to our result format
        reduction = (1 - abs(result_dict['theta'] / self.baseline_coef)) * 100
        
        self.results.append({
            'Model': 'OLS',
            'Learner/Selector': 'OLS',
            'Embedding': embed_name,
            'Features': 'Full',
            'Coeff (θ)': result_dict['theta'],
            'Robust SE': result_dict['se'],
            'p-value': result_dict['pval'],
            '95% CI': f"[{result_dict['ci'][0]:.4f}, {result_dict['ci'][1]:.4f}]",
            'R²(Y) Full': result_dict['r2_y_full'],
            'R²(Y) CV Mean': result_dict['r2_y_mean'],
            'R²(Y) Folds': str([f'{r:.2f}' for r in result_dict['r2_y_folds']]) if result_dict['r2_y_folds'] else 'N/A',
            'R²(X) Full': result_dict['r2_x_full'],
            'R²(X) CV Mean': result_dict['r2_x_mean'],
            'R²(X) Folds': str([f'{r:.2f}' for r in result_dict['r2_x_folds']]) if result_dict['r2_x_folds'] else 'N/A',
            'G': f"{result_dict['G']:.4f}" if result_dict['G'] is not None else 'N/A',
            'C': f"{result_dict['C']:.4f}" if result_dict['C'] is not None else 'N/A',
            'Corr(X,X̂)': f"{result_dict['corr_SC_SChat']:.4f}" if result_dict['corr_SC_SChat'] is not None else 'N/A',
            'Corr(Y,Ŷ)': f"{result_dict['corr_AI_AIhat']:.4f}" if result_dict['corr_AI_AIhat'] is not None else 'N/A',
            'Reduction (vs baseline)': f'{reduction:.2f}%'
        })
        
        # DML with different learners
        for learner in ['xgboost', 'lasso', 'ridge']:
            if embed_name == 'Nvembed' and learner in ['lasso', 'ridge']:
                continue  # Skip for high-dimensional NV-Embed
                
            logger.info(f"Running DML with {learner}...")
            result = self.dml_analyzer.fit_dml(X, y, embeddings, learner=learner)
            reduction = (1 - abs(result['coefficient'] / self.baseline_coef)) * 100
            
            self.results.append({
                'Model': 'DML',
                'Learner/Selector': learner.capitalize(),
                'Embedding': embed_name,
                'Features': 'Full',
                'Coeff (θ)': result['coefficient'],
                'Robust SE': result['se'],
                'p-value': result['p_value'],
                '95% CI': f"[{result['ci'][0]:.4f}, {result['ci'][1]:.4f}]",
                'R²(Y) Full': result['r2_y_full'],
                'R²(Y) CV Mean': result['r2_y_cv_mean'],
                'R²(Y) Folds': str([f'{r:.2f}' for r in result['r2_y_cv']]),
                'R²(X) Full': result['r2_x_full'],
                'R²(X) CV Mean': result['r2_x_cv_mean'],
                'R²(X) Folds': str([f'{r:.2f}' for r in result['r2_x_cv']]),
                'G': f"{result['G']:.4f}",
                'C': f"{result['C']:.4f}",
                'Corr(X,X̂)': f"{result['corr_x']:.4f}",
                'Corr(Y,Ŷ)': f"{result['corr_y']:.4f}",
                'Reduction (vs baseline)': f'{reduction:.2f}%'
            })
    
    def _run_pca_analysis(self, X: np.ndarray, y: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
        """Run PCA dimensionality reduction"""
        logger.info(f"Running PCA to {self.n_pca_components} components...")
        
        # Standardize embeddings
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(embeddings)
        
        # Apply PCA
        pca = PCA(n_components=self.n_pca_components, random_state=self.random_seed)
        pca_features = pca.fit_transform(embeddings_scaled)
        
        # Save PCA components
        pca_df = pd.DataFrame(
            pca_features,
            columns=[f'PC{i+1}' for i in range(self.n_pca_components)]
        )
        pca_df.to_csv(self.output_dir / 'pca_components.csv', index=False)
        
        # Save explained variance
        var_explained = pd.DataFrame({
            'PC': [f'PC{i+1}' for i in range(len(pca.explained_variance_ratio_))],
            'Variance_Explained': pca.explained_variance_ratio_,
            'Cumulative_Variance': np.cumsum(pca.explained_variance_ratio_)
        })
        var_explained.to_csv(self.output_dir / 'pca_variance_explained.csv', index=False)
        
        logger.info(f"PCA explains {np.sum(pca.explained_variance_ratio_)*100:.2f}% of variance")
        
        # Run analyses on PCA features
        embed_name = self._get_embedding_name(embeddings)
        
        # OLS with PCA
        from .dml_notebook import NotebookDMLAnalyzer
        notebook_analyzer = NotebookDMLAnalyzer(n_folds=self.n_folds, random_state=self.random_seed)
        result_dict = notebook_analyzer.run_ols(pca_features, X, y, f'{self.n_pca_components} PCs', embed_name)
        reduction = (1 - abs(result_dict['theta'] / self.baseline_coef)) * 100
        
        self.results.append({
            'Model': 'OLS',
            'Learner/Selector': 'OLS',
            'Embedding': embed_name,
            'Features': '200 PCs',
            'Coeff (θ)': result_dict['theta'],
            'Robust SE': result_dict['se'],
            'p-value': result_dict['pval'],
            '95% CI': f"[{result_dict['ci'][0]:.4f}, {result_dict['ci'][1]:.4f}]",
            'R²(Y) Full': result_dict['r2_y_full'],
            'R²(Y) CV Mean': result_dict['r2_y_mean'],
            'R²(Y) Folds': str([f'{r:.2f}' for r in result_dict['r2_y_folds']]) if result_dict['r2_y_folds'] else 'N/A',
            'R²(X) Full': result_dict['r2_x_full'],
            'R²(X) CV Mean': result_dict['r2_x_mean'],
            'R²(X) Folds': str([f'{r:.2f}' for r in result_dict['r2_x_folds']]) if result_dict['r2_x_folds'] else 'N/A',
            'G': f"{result_dict['G']:.4f}" if result_dict['G'] is not None else 'N/A',
            'C': f"{result_dict['C']:.4f}" if result_dict['C'] is not None else 'N/A',
            'Corr(X,X̂)': f"{result_dict['corr_SC_SChat']:.4f}" if result_dict['corr_SC_SChat'] is not None else 'N/A',
            'Corr(Y,Ŷ)': f"{result_dict['corr_AI_AIhat']:.4f}" if result_dict['corr_AI_AIhat'] is not None else 'N/A',
            'Reduction (vs baseline)': f'{reduction:.2f}%'
        })
        
        # DML with PCA
        for learner in ['xgboost', 'lasso', 'ridge']:
            logger.info(f"Running DML with {learner} on PCA features...")
            result = self.dml_analyzer.fit_dml(X, y, pca_features, learner=learner)
            reduction = (1 - abs(result['coefficient'] / self.baseline_coef)) * 100
            
            self.results.append({
                'Model': 'DML',
                'Learner/Selector': learner.capitalize(),
                'Embedding': embed_name,
                'Features': '200 PCs',
                'Coeff (θ)': result['coefficient'],
                'Robust SE': result['se'],
                'p-value': result['p_value'],
                '95% CI': f"[{result['ci'][0]:.4f}, {result['ci'][1]:.4f}]",
                'R²(Y) Full': result['r2_y_full'],
                'R²(Y) CV Mean': result['r2_y_cv_mean'],
                'R²(Y) Folds': str([f'{r:.2f}' for r in result['r2_y_cv']]),
                'R²(X) Full': result['r2_x_full'],
                'R²(X) CV Mean': result['r2_x_cv_mean'],
                'R²(X) Folds': str([f'{r:.2f}' for r in result['r2_x_cv']]),
                'G': f"{result['G']:.4f}",
                'C': f"{result['C']:.4f}",
                'Corr(X,X̂)': f"{result['corr_x']:.4f}",
                'Corr(Y,Ŷ)': f"{result['corr_y']:.4f}",
                'Reduction (vs baseline)': f'{reduction:.2f}%'
            })
        
        return pca_features
    
    def _run_top_features_analysis(self, X: np.ndarray, y: np.ndarray, pca_features: np.ndarray):
        """Run analysis with top selected features"""
        logger.info("Running top features selection...")
        
        embed_name = self._get_embedding_name(pca_features)
        
        # Feature selection methods
        methods = ['xgboost', 'lasso', 'ridge', 'ols', 'mi']
        
        for method in methods:
            logger.info(f"Selecting top {self.n_top_features} features with {method}...")
            
            # Alternating selection between X and y
            # Get scores for all features
            scores_X = self._get_feature_scores(pca_features, X, method)
            scores_y = self._get_feature_scores(pca_features, y, method)
            
            # Alternating selection preserving order
            all_indices = self._get_alternating_selection(scores_X, scores_y, self.n_top_features)
            
            # Store selection
            self.feature_selections[f"{embed_name}_{method}"] = all_indices.tolist()
            
            # Get selected features
            selected_features = pca_features[:, all_indices]
            
            # OLS with selected features
            from .dml_notebook import NotebookDMLAnalyzer
            notebook_analyzer = NotebookDMLAnalyzer(n_folds=self.n_folds, random_state=self.random_seed)
            feature_label = f"Top {self.n_top_features} ({method.replace('mutual_info', 'Mi').replace('xgboost', 'Xgboost').replace('lasso', 'Lasso').replace('ridge', 'Ridge').replace('ols', 'Ols')})"
            result_dict = notebook_analyzer.run_ols(selected_features, X, y, feature_label, embed_name)
            reduction = (1 - abs(result_dict['theta'] / self.baseline_coef)) * 100
            
            self.results.append({
                'Model': 'OLS',
                'Learner/Selector': 'OLS',
                'Embedding': embed_name,
                'Features': feature_label,
                'Coeff (θ)': result_dict['theta'],
                'Robust SE': result_dict['se'],
                'p-value': result_dict['pval'],
                '95% CI': f"[{result_dict['ci'][0]:.4f}, {result_dict['ci'][1]:.4f}]",
                'R²(Y) Full': result_dict['r2_y_full'],
                'R²(Y) CV Mean': result_dict['r2_y_mean'],
                'R²(Y) Folds': str([f'{r:.2f}' for r in result_dict['r2_y_folds']]) if result_dict['r2_y_folds'] else 'N/A',
                'R²(X) Full': result_dict['r2_x_full'],
                'R²(X) CV Mean': result_dict['r2_x_mean'],
                'R²(X) Folds': str([f'{r:.2f}' for r in result_dict['r2_x_folds']]) if result_dict['r2_x_folds'] else 'N/A',
                'G': f"{result_dict['G']:.4f}" if result_dict['G'] is not None else 'N/A',
                'C': f"{result_dict['C']:.4f}" if result_dict['C'] is not None else 'N/A',
                'Corr(X,X̂)': f"{result_dict['corr_SC_SChat']:.4f}" if result_dict['corr_SC_SChat'] is not None else 'N/A',
                'Corr(Y,Ŷ)': f"{result_dict['corr_AI_AIhat']:.4f}" if result_dict['corr_AI_AIhat'] is not None else 'N/A',
                'Reduction (vs baseline)': f'{reduction:.2f}%'
            })
            
            # DML with selected features (only for matching learner)
            if method in ['xgboost', 'lasso', 'ridge']:
                logger.info(f"Running DML with {method} on selected features...")
                result = self.dml_analyzer.fit_dml(X, y, selected_features, learner=method)
                reduction = (1 - abs(result['coefficient'] / self.baseline_coef)) * 100
                
                self.results.append({
                    'Model': 'DML',
                    'Learner/Selector': method.capitalize(),
                    'Embedding': embed_name,
                    'Features': f'Top 6 ({method.capitalize()})',
                    'Coeff (θ)': result['coefficient'],
                    'Robust SE': result['se'],
                    'p-value': result['p_value'],
                    '95% CI': f"[{result['ci'][0]:.4f}, {result['ci'][1]:.4f}]",
                    'R²(Y) Full': result['r2_y_full'],
                    'R²(Y) CV Mean': result['r2_y_cv_mean'],
                    'R²(Y) Folds': str([f'{r:.2f}' for r in result['r2_y_cv']]),
                    'R²(X) Full': result['r2_x_full'],
                    'R²(X) CV Mean': result['r2_x_cv_mean'],
                    'R²(X) Folds': str([f'{r:.2f}' for r in result['r2_x_cv']]),
                    'G': f"{result['G']:.4f}",
                    'C': f"{result['C']:.4f}",
                    'Corr(X,X̂)': f"{result['corr_x']:.4f}",
                    'Corr(Y,Ŷ)': f"{result['corr_y']:.4f}",
                    'Reduction (vs baseline)': f'{reduction:.2f}%'
                })
    
    def _create_results_table(self) -> pd.DataFrame:
        """Create comprehensive results DataFrame"""
        results_df = pd.DataFrame(self.results)
        
        # Add significance stars
        def add_stars(p):
            if isinstance(p, str):
                return p
            if p < 0.001:
                return f'{p:.3e}'
            elif p < 0.01:
                return f'{p:.3e}'
            elif p < 0.05:
                return f'{p:.3e}'
            else:
                return f'{p:.3e}'
        
        results_df['p-value'] = results_df['p-value'].apply(add_stars)
        
        return results_df
    
    def _save_outputs(self, results_df: pd.DataFrame, X: np.ndarray, y: np.ndarray, 
                     embeddings: np.ndarray, pca_features: np.ndarray):
        """Save all outputs to files"""
        logger.info(f"Saving outputs to {self.output_dir}...")
        
        # Save main results
        results_df.to_csv(self.output_dir / 'results_table.csv', index=False)
        
        # Save feature selections
        selections_df = pd.DataFrame([
            {
                'Embedding': key.split('_')[0],
                'Selection Method': key.split('_')[1].capitalize(),
                'Selected PC Indices': ', '.join(map(str, indices))
            }
            for key, indices in self.feature_selections.items()
        ])
        selections_df.to_csv(self.output_dir / 'feature_selections.csv', index=False)
        
        # Save raw data
        data_df = pd.DataFrame({
            'treatment': X,
            'outcome': y
        })
        data_df.to_csv(self.output_dir / 'raw_data.csv', index=False)
        
        # Save analysis parameters
        params = {
            'embedding_model': self.embedding_model,
            'n_pca_components': self.n_pca_components,
            'n_top_features': self.n_top_features,
            'n_folds': self.n_folds,
            'random_seed': self.random_seed,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(self.output_dir / 'analysis_params.json', 'w') as f:
            json.dump(params, f, indent=2)
        
        logger.info(f"All outputs saved to {self.output_dir}")
    
    def _display_results(self, results_df: pd.DataFrame):
        """Display results in terminal"""
        print("\n" + "="*140)
        print(" " * 40 + "FINAL COMPREHENSIVE EMBEDDING MODEL COMPARISON WITH ROBUST INFERENCE")
        print("="*140)
        
        # Format for display
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', None)
        
        print(results_df.to_string(index=False))
        
        print("\n" + "-"*140)
        print("Notes: All p-values are two-tailed. Significance levels: *** p<0.001, ** p<0.01, * p<0.05")
        print("Standard errors are heteroskedasticity-robust (HC0 for OLS, sandwich estimator for DML)")
        print("G = correlation between cross-fitted predictions; C = correlation between cross-fitted residuals")
        print("Corr(X,X̂) and Corr(Y,Ŷ) = correlations between actual values and their cross-fitted predictions")
        print("-"*140)
        
        # Display feature selections
        print("\n\n" + "="*80)
        print(" " * 20 + "TOP 6 PC SELECTION RESULTS")
        print("="*80)
        
        selections_df = pd.DataFrame([
            {
                'Embedding': key.split('_')[0],
                'Selection Method': key.split('_')[1].capitalize(),
                'Selected PC Indices': ', '.join(map(str, indices))
            }
            for key, indices in self.feature_selections.items()
        ])
        
        print(selections_df.to_string(index=False))
        
    def _get_feature_scores(self, X: np.ndarray, y: np.ndarray, method: str) -> np.ndarray:
        """Get feature importance scores for all features"""
        if method == 'xgboost':
            try:
                # Use single thread to avoid segfault on Apple Silicon
                model = xgb.XGBRegressor(
                    n_estimators=100, 
                    max_depth=3,
                    random_state=self.random_seed, 
                    n_jobs=1,
                    tree_method='hist', 
                    verbosity=0,
                    subsample=1.0,  # No subsampling for reproducibility
                    colsample_bytree=1.0,  # No column subsampling
                    colsample_bylevel=1.0,
                    reg_alpha=0,  # No L1 regularization
                    reg_lambda=1,  # Default L2 regularization
                )
                model.fit(X, y)
                return model.feature_importances_
            except Exception as e:
                logger.warning(f"XGBoost failed: {e}. Using RandomForest as fallback.")
                from sklearn.ensemble import RandomForestRegressor
                model = RandomForestRegressor(n_estimators=100, max_depth=3,
                                            random_state=self.random_seed, n_jobs=1)
                model.fit(X, y)
                return model.feature_importances_
            
        elif method == 'lasso':
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            model = LassoCV(cv=5, random_state=self.random_seed, n_jobs=-1)
            model.fit(X_scaled, y)
            return np.abs(model.coef_)
            
        elif method == 'ridge':
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            model = RidgeCV(cv=5)
            model.fit(X_scaled, y)
            return np.abs(model.coef_)
            
        elif method == 'ols':
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
            model.fit(X, y)
            return np.abs(model.coef_)
            
        elif method == 'mi':
            from sklearn.feature_selection import mutual_info_regression
            return mutual_info_regression(X, y, random_state=self.random_seed)
    
    def _get_alternating_selection(self, scores_X: np.ndarray, scores_y: np.ndarray, 
                                  n_features: int = 6) -> np.ndarray:
        """
        Alternating selection from X and y scores, preserving order
        This matches the notebook's get_alt_selection implementation
        """
        # Get ranked indices (descending order of importance)
        ranked_X = np.argsort(scores_X)[::-1]
        ranked_y = np.argsort(scores_y)[::-1]
        
        selected = []
        i, j = 0, 0
        
        # Alternate between X and y rankings
        while len(selected) < n_features:
            # Try to add from X ranking
            while i < len(ranked_X) and len(selected) < n_features:
                if ranked_X[i] not in selected:
                    selected.append(ranked_X[i])
                    break
                i += 1
            
            # Try to add from y ranking  
            while j < len(ranked_y) and len(selected) < n_features:
                if ranked_y[j] not in selected:
                    selected.append(ranked_y[j])
                    break
                j += 1
            
            # If we can't find any more unique features, break
            if i >= len(ranked_X) and j >= len(ranked_y):
                break
        
        # If we still don't have enough features, add the remaining top-ranked ones
        if len(selected) < n_features:
            all_scores = (scores_X + scores_y) / 2  # Average the scores
            all_ranked = np.argsort(all_scores)[::-1]
            for idx in all_ranked:
                if idx not in selected:
                    selected.append(idx)
                    if len(selected) >= n_features:
                        break
        
        return np.array(selected[:n_features])
    
    # Helper methods
    def _get_embedding_name(self, embeddings: np.ndarray) -> str:
        """Get embedding name based on dimensions"""
        if embeddings.shape[1] == 384:
            return 'Minilm'
        elif embeddings.shape[1] == 4096:
            return 'Nvembed'
        elif embeddings.shape[1] == 200:
            return 'Minilm'  # PCA features
        else:
            return f'Custom{embeddings.shape[1]}'
    
    def _calculate_r2(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate R² for OLS"""
        from sklearn.metrics import r2_score
        from sklearn.linear_model import LinearRegression
        
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        return r2_score(y, y_pred)
    
    def _calculate_cv_r2(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate cross-validated R²"""
        from sklearn.model_selection import cross_val_score
        from sklearn.linear_model import LinearRegression
        
        model = LinearRegression()
        scores = cross_val_score(model, X, y, cv=5, scoring='r2')
        return np.mean(scores)
    
    def _get_cv_folds_r2(self, X: np.ndarray, y: np.ndarray) -> str:
        """Get R² for each CV fold"""
        from sklearn.model_selection import cross_val_score
        from sklearn.linear_model import LinearRegression
        
        model = LinearRegression()
        scores = cross_val_score(model, X, y, cv=5, scoring='r2')
        return str([f'{s:.2f}' for s in scores])
    
    def _calculate_correlation(self, X: np.ndarray, treatment: np.ndarray, 
                             outcome: np.ndarray, corr_type: str) -> float:
        """Calculate G or C correlation"""
        result = self.dml_analyzer.fit_dml(treatment, outcome, X, learner='xgboost')
        
        if corr_type == 'predictions':
            return result['G']
        else:
            return result['C']
    
    def _calculate_prediction_correlation(self, X: np.ndarray, y: np.ndarray) -> float:
        """Calculate correlation between actual and predicted values"""
        from sklearn.linear_model import LinearRegression
        
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        
        return np.corrcoef(y, y_pred)[0, 1]