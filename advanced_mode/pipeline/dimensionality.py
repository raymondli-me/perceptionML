#!/usr/bin/env python3
"""Dimensionality reduction using PCA and UMAP."""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import umap
from typing import Dict, Tuple, Optional, List
import pickle

from .config import PipelineConfig


class DimensionalityReducer:
    """Handles PCA and UMAP dimensionality reduction."""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.pca = None
        self.scaler = None
        self.umap_model = None
        
    def fit_pca(self, embeddings: np.ndarray) -> Dict:
        """Fit PCA on embeddings and transform."""
        print(f"Fitting PCA with {self.config.analysis.pca_components} components...")
        
        # Standardize features
        self.scaler = StandardScaler()
        embeddings_scaled = self.scaler.fit_transform(embeddings)
        
        # Fit PCA
        n_components = min(self.config.analysis.pca_components, 
                          embeddings.shape[0] - 1, 
                          embeddings.shape[1])
        
        self.pca = PCA(n_components=n_components, random_state=42)
        pca_features = self.pca.fit_transform(embeddings_scaled)
        
        # Calculate statistics
        explained_var = self.pca.explained_variance_ratio_
        cumulative_var = np.cumsum(explained_var)
        
        print(f"✓ PCA complete:")
        print(f"  - Components: {n_components}")
        print(f"  - Explained variance (first 5): {explained_var[:5] * 100}")
        print(f"  - Cumulative variance at 50 components: {cumulative_var[49]:.2%}")
        
        # Calculate percentiles for each PC
        percentiles = self._calculate_pc_percentiles(pca_features)
        
        results = {
            'features': pca_features,
            'explained_variance': explained_var,
            'cumulative_variance': cumulative_var,
            'percentiles': percentiles,
            'n_components': n_components,
            'pca_model': self.pca,
            'scaler': self.scaler
        }
        
        # Save checkpoint
        self.config.save_checkpoint('pca_results', results)
        
        return results
    
    def _calculate_pc_percentiles(self, pca_features: np.ndarray) -> np.ndarray:
        """Calculate percentile ranks for each PC value."""
        n_samples, n_components = pca_features.shape
        percentiles = np.zeros_like(pca_features)
        
        for i in range(n_components):
            pc_values = pca_features[:, i]
            # Calculate percentile rank for each value
            percentiles[:, i] = (np.searchsorted(np.sort(pc_values), pc_values) / n_samples) * 100
            
        return percentiles
    
    def fit_umap(self, features: np.ndarray, 
                 n_dimensions: Optional[int] = None) -> Dict:
        """Fit UMAP on features (usually PCA-reduced)."""
        if n_dimensions is None:
            n_dimensions = self.config.analysis.umap_dimensions
            
        print(f"Fitting UMAP with {n_dimensions} dimensions...")
        
        self.umap_model = umap.UMAP(
            n_components=n_dimensions,
            n_neighbors=self.config.analysis.umap_n_neighbors,
            min_dist=self.config.analysis.umap_min_dist,
            metric='cosine',
            random_state=42,
            verbose=True
        )
        
        umap_embeddings = self.umap_model.fit_transform(features)
        
        # Normalize to unit cube for visualization
        umap_normalized = self._normalize_umap(umap_embeddings)
        
        print(f"✓ UMAP complete:")
        print(f"  - Output shape: {umap_embeddings.shape}")
        print(f"  - Range: [{umap_embeddings.min():.2f}, {umap_embeddings.max():.2f}]")
        
        results = {
            'embeddings': umap_embeddings,
            'embeddings_normalized': umap_normalized,
            'n_dimensions': n_dimensions,
            'model': self.umap_model
        }
        
        # Save checkpoint
        self.config.save_checkpoint(f'umap_{n_dimensions}d_results', results)
        
        return results
    
    def _normalize_umap(self, embeddings: np.ndarray) -> np.ndarray:
        """Normalize UMAP embeddings to unit cube."""
        # Center at origin
        centered = embeddings - embeddings.mean(axis=0)
        
        # Scale to [-1, 1]
        max_range = np.abs(centered).max()
        if max_range > 0:
            normalized = centered / max_range
        else:
            normalized = centered
            
        return normalized
    
    def transform_new_data(self, embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Transform new data using fitted PCA and UMAP models."""
        if self.pca is None or self.umap_model is None:
            raise ValueError("Models not fitted. Run fit_pca and fit_umap first.")
        
        # PCA transform
        embeddings_scaled = self.scaler.transform(embeddings)
        pca_features = self.pca.transform(embeddings_scaled)
        
        # UMAP transform
        umap_embeddings = self.umap_model.transform(pca_features)
        umap_normalized = self._normalize_umap(umap_embeddings)
        
        return pca_features, umap_normalized
    
    def load_from_checkpoints(self) -> bool:
        """Try to load fitted models from checkpoints."""
        pca_checkpoint = self.config.load_checkpoint('pca_results')
        umap_checkpoint = self.config.load_checkpoint(f'umap_{self.config.analysis.umap_dimensions}d_results')
        
        if pca_checkpoint and umap_checkpoint:
            self.pca = pca_checkpoint['pca_model']
            self.scaler = pca_checkpoint['scaler']
            self.umap_model = umap_checkpoint['model']
            print("✓ Loaded dimensionality reduction models from checkpoints")
            return True
        
        return False
    
    def select_top_pcs_for_dml(self, pca_features: np.ndarray, 
                              outcome_data: Dict[str, np.ndarray],
                              n_pcs: int = 6,
                              methods: Optional[List[str]] = None) -> Tuple[List[int], Dict]:
        """Select top PCs for DML analysis using average importance across outcomes.
        
        Computes importance using multiple methods (XGBoost, Lasso, Ridge, MI) and
        selects top n_pcs based on average importance for treatment and outcome prediction.
        
        Args:
            pca_features: PCA transformed features
            outcome_data: Dictionary of outcome name -> values
            n_pcs: Number of PCs to select
            methods: List of methods to use. If None, uses config setting.
        
        Returns:
            primary_indices: Top PC indices for the primary method
            selection_info: Dictionary with selection info for all methods
        """
        from sklearn.feature_selection import mutual_info_regression
        from sklearn.linear_model import LassoCV, RidgeCV
        import xgboost as xgb
        
        # Use methods from config if not specified
        if methods is None:
            methods = self.config.analysis.dml_pc_selection_methods
        
        print(f"Selecting top {n_pcs} PCs for DML analysis using methods: {', '.join(methods)}")
        
        # We'll need treatment and outcome separately for the new algorithm
        # Assuming 'social_class' is treatment (X) and 'ai_rating' is outcome (Y)
        # Get the outcome names from the config
        outcome_names = list(outcome_data.keys())
        if len(outcome_names) < 2:
            raise ValueError(f"Need at least 2 outcomes for DML analysis, got {len(outcome_names)}")
        
        # Use first two outcomes as treatment and outcome
        treatment_name = outcome_names[0]
        outcome_name = outcome_names[1]
        
        treatment_values = outcome_data[treatment_name]
        outcome_values = outcome_data[outcome_name]
        
        print(f"  Using '{treatment_name}' as treatment and '{outcome_name}' as outcome")
        
        # Store all selection methods results
        selection_methods = {}
        selected_pcs = {}
        importances = {}
        
        # Compute importances for requested methods
        if 'xgboost' in methods:
            print("  Computing XGBoost feature importance...")
            xgb_importance_x = self._compute_xgb_importance(pca_features, treatment_values)
            xgb_importance_y = self._compute_xgb_importance(pca_features, outcome_values)
            avg_xgb_importance = (xgb_importance_x + xgb_importance_y) / 2
            selected_pcs['xgboost'] = np.argsort(avg_xgb_importance)[-n_pcs:][::-1].tolist()
            importances['xgb_importance_x'] = xgb_importance_x.tolist()
            importances['xgb_importance_y'] = xgb_importance_y.tolist()
        
        if 'lasso' in methods:
            print("  Computing Lasso feature selection...")
            lasso_importance_x, lasso_alpha_x = self._compute_lasso_importance(pca_features, treatment_values)
            lasso_importance_y, lasso_alpha_y = self._compute_lasso_importance(pca_features, outcome_values)
            
            # Print selected alphas
            alphas_list = self.config.analysis.lasso_alphas or 'default range'
            print(f"    → Selected alpha for {treatment_name}: {lasso_alpha_x:.6f} (from {alphas_list})")
            print(f"    → Selected alpha for {outcome_name}: {lasso_alpha_y:.6f} (from {alphas_list})")
            
            avg_lasso_importance = (lasso_importance_x + lasso_importance_y) / 2
            selected_pcs['lasso'] = np.argsort(avg_lasso_importance)[-n_pcs:][::-1].tolist()
            importances['lasso_importance_x'] = lasso_importance_x.tolist()
            importances['lasso_importance_y'] = lasso_importance_y.tolist()
            
            # Store selected alphas
            selection_methods['lasso_alpha_x'] = lasso_alpha_x
            selection_methods['lasso_alpha_y'] = lasso_alpha_y
        
        if 'ridge' in methods:
            print("  Computing Ridge feature selection...")
            ridge_importance_x, ridge_alpha_x = self._compute_ridge_importance(pca_features, treatment_values)
            ridge_importance_y, ridge_alpha_y = self._compute_ridge_importance(pca_features, outcome_values)
            
            # Print selected alphas
            alphas_list = self.config.analysis.ridge_alphas or [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
            print(f"    → Selected alpha for {treatment_name}: {ridge_alpha_x:.6f} (from {alphas_list})")
            print(f"    → Selected alpha for {outcome_name}: {ridge_alpha_y:.6f} (from {alphas_list})")
            
            avg_ridge_importance = (ridge_importance_x + ridge_importance_y) / 2
            selected_pcs['ridge'] = np.argsort(avg_ridge_importance)[-n_pcs:][::-1].tolist()
            importances['ridge_importance_x'] = ridge_importance_x.tolist()
            importances['ridge_importance_y'] = ridge_importance_y.tolist()
            
            # Store selected alphas
            selection_methods['ridge_alpha_x'] = ridge_alpha_x
            selection_methods['ridge_alpha_y'] = ridge_alpha_y
        
        for method_name, indices in selected_pcs.items():
            print(f"  ✓ {method_name.capitalize()} selected PCs (by avg importance): {indices}")
        
        # Compute mutual information if requested
        if 'mi' in methods:
            print("  Computing Mutual Information...")
            mi_scores = {}
            for outcome_name, outcome_values in outcome_data.items():
                mi = mutual_info_regression(pca_features, outcome_values, random_state=42)
                mi_scores[outcome_name] = mi
            avg_mi = np.mean(list(mi_scores.values()), axis=0)
            selected_pcs['mi'] = np.argsort(avg_mi)[-n_pcs:][::-1].tolist()
            importances['mi_scores'] = avg_mi.tolist()
            print(f"  ✓ MI selected PCs: {selected_pcs['mi']}")
        
        # Use primary method from config
        primary_method = self.config.analysis.dml_primary_pc_method
        if primary_method not in selected_pcs:
            # If primary method wasn't computed, use the first available method
            primary_method = list(selected_pcs.keys())[0]
            print(f"\n  Warning: Primary method '{self.config.analysis.dml_primary_pc_method}' not in selected methods, using '{primary_method}'")
        
        primary_indices = selected_pcs[primary_method]
        print(f"\n  Primary selection ({primary_method}): {primary_indices}")
        
        # Compile all selection info - include indices with method names
        selection_info = {}
        for method, indices in selected_pcs.items():
            selection_info[f'{method}_indices'] = indices
        
        # Add all importances
        selection_info.update(importances)
        
        # Add selected alpha values
        selection_info.update(selection_methods)
        
        return primary_indices, selection_info
    
    def _compute_xgb_importance(self, features: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Compute XGBoost feature importance for a single target."""
        import xgboost as xgb
        
        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            random_state=42,
            tree_method='hist',
            device='cuda'
        )
        model.fit(features, target)
        return model.feature_importances_
    
    def _compute_lasso_importance(self, features: np.ndarray, target: np.ndarray) -> tuple:
        """Compute Lasso feature importance (absolute coefficients) and selected alpha."""
        from sklearn.linear_model import LassoCV
        from sklearn.preprocessing import StandardScaler
        
        # Standardize features for Lasso
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Get alphas and CV folds from config
        alphas = self.config.analysis.lasso_alphas
        cv_folds = self.config.analysis.regularization_cv_folds or 5
        
        # Use LassoCV to find optimal alpha
        if alphas:
            lasso = LassoCV(cv=cv_folds, random_state=42, max_iter=2000, alphas=alphas, precompute=False)
        else:
            # Use default alpha selection
            lasso = LassoCV(cv=cv_folds, random_state=42, max_iter=2000, precompute=False)
        lasso.fit(features_scaled, target)
        
        # Return both importance scores and selected alpha
        return np.abs(lasso.coef_), lasso.alpha_
    
    def _compute_ridge_importance(self, features: np.ndarray, target: np.ndarray) -> tuple:
        """Compute Ridge feature importance (absolute coefficients) and selected alpha."""
        from sklearn.linear_model import RidgeCV
        from sklearn.preprocessing import StandardScaler
        
        # Standardize features for Ridge
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Get alphas and CV folds from config
        alphas = self.config.analysis.ridge_alphas or [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
        cv_folds = self.config.analysis.regularization_cv_folds or 5
        
        # Use RidgeCV to find optimal alpha
        ridge = RidgeCV(cv=cv_folds, alphas=alphas)
        ridge.fit(features_scaled, target)
        
        # Return both importance scores and selected alpha
        return np.abs(ridge.coef_), ridge.alpha_
    
