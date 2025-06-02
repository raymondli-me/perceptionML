#!/usr/bin/env python3
"""Double Machine Learning analysis for causal inference."""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import xgboost as xgb
import statsmodels.api as sm
from scipy import stats
from typing import Dict, List, Tuple, Optional

from .config import PipelineConfig


class DMLAnalyzer:
    """Performs Double Machine Learning analysis."""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.models = {}
        self.results = {}
        self.residuals = {}  # Store residuals for export
        self.predictions = {}  # Store predictions for export
        
    def run_dml_analysis(self, X: np.ndarray, 
                        outcomes: Dict[str, np.ndarray],
                        feature_names: Optional[List[str]] = None,
                        model_suffix: Optional[str] = None,
                        control_data: Optional[Dict[str, np.ndarray]] = None) -> Dict:
        """
        Run DML analysis for all outcome pairs.
        X: Control features (e.g., PCA components)
        outcomes: Dict of outcome name -> values
        control_data: Optional dict of control variable name -> values
        """
        print("\n=== Running Double Machine Learning Analysis ===")
        
        # Combine features with control variables if provided
        if control_data is not None:
            print(f"  Including {len(control_data)} control variables")
            # Stack control variables with features
            control_matrix = np.column_stack([
                control_data[name] for name in sorted(control_data.keys())
            ])
            X_combined = np.hstack([X, control_matrix])
            
            # Update feature names
            control_names = [f'Control_{name}' for name in sorted(control_data.keys())]
            if feature_names is None:
                feature_names = [f'PC{i}' for i in range(X.shape[1])] + control_names
            else:
                feature_names = feature_names + control_names
        else:
            X_combined = X
            if feature_names is None:
                feature_names = [f'PC{i}' for i in range(X.shape[1])]
        
        results = {
            'feature_names': feature_names,
            'n_features': X_combined.shape[1],
            'n_samples': X_combined.shape[0],
            'has_controls': control_data is not None,
            'n_controls': len(control_data) if control_data else 0
        }
        
        # Calculate contributions for each outcome
        for outcome_name, Y in outcomes.items():
            contributions = self._calculate_contributions(
                X_combined, Y, feature_names
            )
            results[f'contributions_{outcome_name}'] = contributions
        
        # Run analysis for each pair of outcomes
        outcome_names = list(outcomes.keys())
        
        for i, treatment_name in enumerate(outcome_names):
            for j, outcome_name in enumerate(outcome_names):
                if i >= j:  # Skip same outcome and reverse pairs
                    continue
                
                print(f"\nAnalyzing: {treatment_name} → {outcome_name}")
                print(f"  Data shape: {X_combined.shape[0]} samples × {X_combined.shape[1]} features")
                if control_data:
                    print(f"  Including {len(control_data)} control variables")
                print(f"  Using {self.config.analysis.dml_n_folds}-fold cross-validation")
                
                # Run both naive and DML estimation
                naive_result = self._estimate_naive(
                    outcomes[treatment_name], 
                    outcomes[outcome_name]
                )
                
                dml_result = self._estimate_dml(
                    X_combined,
                    outcomes[treatment_name],
                    outcomes[outcome_name],
                    cross_fitted=True
                )
                
                # Calculate reduction in effect
                reduction = (1 - dml_result['theta'] / naive_result['theta']) * 100
                
                # Store results
                pair_key = f"{treatment_name}_to_{outcome_name}"
                results[pair_key] = {
                    'naive': naive_result,
                    'dml': dml_result,
                    'reduction': reduction,
                    'treatment': treatment_name,
                    'outcome': outcome_name
                }
                
                # Store residuals and predictions if available with model suffix
                storage_key = f"{pair_key}_{model_suffix}" if model_suffix else pair_key
                if 'residuals' in dml_result:
                    self.residuals[storage_key] = dml_result['residuals']
                if 'predictions' in dml_result:
                    self.predictions[storage_key] = dml_result['predictions']
                # Store non-CV residuals and predictions
                if 'residuals_full' in dml_result:
                    self.residuals[f"{storage_key}_noncv"] = dml_result['residuals_full']
                if 'predictions_full' in dml_result:
                    self.predictions[f"{storage_key}_noncv"] = dml_result['predictions_full']
        
        # Calculate R² for each outcome
        for outcome_name, Y in outcomes.items():
            r2_result = self._calculate_r2(X_combined, Y)
            results[f'r2_{outcome_name}'] = r2_result
        
        self.results = results
        return results
    
    def _estimate_naive(self, treatment: np.ndarray, outcome: np.ndarray) -> Dict:
        """Estimate naive OLS coefficient without controls."""
        # Calculate correlation
        correlation = np.corrcoef(treatment, outcome)[0, 1]
        
        # Add constant
        X = sm.add_constant(treatment)
        
        # Fit OLS
        model = sm.OLS(outcome, X).fit()
        
        # Extract results
        theta = model.params[1]
        se = model.bse[1]
        conf_int = model.conf_int()
        ci = (conf_int[0][1], conf_int[1][1])  # Get the confidence interval for the treatment coefficient
        pval = model.pvalues[1]
        r2 = model.rsquared
        
        return {
            'theta': float(theta),
            'se': float(se),
            'ci': (float(ci[0]), float(ci[1])),
            'pval': float(pval),
            'r2': float(r2),
            'correlation': float(correlation)
        }
    
    def _estimate_dml(self, X: np.ndarray, treatment: np.ndarray, 
                     outcome: np.ndarray, cross_fitted: bool = True) -> Dict:
        """Estimate DML with optional cross-fitting."""
        if cross_fitted:
            return self._estimate_dml_crossfit(X, treatment, outcome)
        else:
            return self._estimate_dml_simple(X, treatment, outcome)
    
    def _estimate_dml_simple(self, X: np.ndarray, treatment: np.ndarray, 
                           outcome: np.ndarray) -> Dict:
        """Non-cross-fitted DML estimation."""
        # Fit models
        model_outcome = xgb.XGBRegressor(
            n_estimators=self.config.analysis.xgb_n_estimators,
            max_depth=self.config.analysis.xgb_max_depth,
            random_state=42,
            tree_method='hist',
            device='cuda'
        )
        model_outcome.fit(X, outcome)
        
        model_treatment = xgb.XGBRegressor(
            n_estimators=self.config.analysis.xgb_n_estimators,
            max_depth=self.config.analysis.xgb_max_depth,
            random_state=42,
            tree_method='hist',
            device='cuda'
        )
        model_treatment.fit(X, treatment)
        
        # Get residuals
        residuals_outcome = outcome - model_outcome.predict(X)
        residuals_treatment = treatment - model_treatment.predict(X)
        
        # DML theta estimate
        theta = np.sum(residuals_treatment * residuals_outcome) / np.sum(residuals_treatment ** 2)
        
        # Standard error
        n = len(outcome)
        se = np.sqrt(np.sum((residuals_outcome - theta * residuals_treatment) ** 2) / 
                     (n * np.sum(residuals_treatment ** 2)))
        
        # Confidence interval
        ci = (theta - 1.96 * se, theta + 1.96 * se)
        
        # P-value
        t_stat = theta / se
        pval = 2 * (1 - stats.t.cdf(abs(t_stat), n - X.shape[1] - 1))
        
        return {
            'theta': float(theta),
            'se': float(se),
            'ci': (float(ci[0]), float(ci[1])),
            'pval': float(pval),
            'cross_fitted': False
        }
    
    def _estimate_dml_crossfit(self, X: np.ndarray, treatment: np.ndarray, 
                              outcome: np.ndarray) -> Dict:
        """Cross-fitted DML estimation."""
        kf = KFold(n_splits=self.config.analysis.dml_n_folds, shuffle=True, random_state=42)
        
        residuals_outcome_all = np.zeros_like(outcome)
        residuals_treatment_all = np.zeros_like(treatment)
        predictions_outcome_all = np.zeros_like(outcome)
        predictions_treatment_all = np.zeros_like(treatment)
        
        # Track R² scores
        r2_outcome_scores = []
        r2_treatment_scores = []
        
        # Cross-fitting
        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
            print(f"  Fold {fold_idx + 1}/{self.config.analysis.dml_n_folds}...", end=' ', flush=True)
            
            # Split data
            X_train, X_test = X[train_idx], X[test_idx]
            treatment_train, treatment_test = treatment[train_idx], treatment[test_idx]
            outcome_train, outcome_test = outcome[train_idx], outcome[test_idx]
            
            # Fit models on training fold
            model_outcome = xgb.XGBRegressor(
                n_estimators=self.config.analysis.xgb_n_estimators,
                max_depth=self.config.analysis.xgb_max_depth,
                random_state=42,
                tree_method='hist',
                device='cuda'
            )
            model_outcome.fit(X_train, outcome_train)
            
            model_treatment = xgb.XGBRegressor(
                n_estimators=self.config.analysis.xgb_n_estimators,
                max_depth=self.config.analysis.xgb_max_depth,
                random_state=42,
                tree_method='hist',
                device='cuda'
            )
            model_treatment.fit(X_train, treatment_train)
            
            # Predict on test fold
            pred_outcome = model_outcome.predict(X_test)
            pred_treatment = model_treatment.predict(X_test)
            
            predictions_outcome_all[test_idx] = pred_outcome
            predictions_treatment_all[test_idx] = pred_treatment
            residuals_outcome_all[test_idx] = outcome_test - pred_outcome
            residuals_treatment_all[test_idx] = treatment_test - pred_treatment
            
            # Calculate R² for this fold
            r2_outcome_scores.append(r2_score(outcome_test, pred_outcome))
            r2_treatment_scores.append(r2_score(treatment_test, pred_treatment))
            
            print("done", flush=True)
        
        # Calculate DML estimator
        theta = np.sum(residuals_treatment_all * residuals_outcome_all) / np.sum(residuals_treatment_all ** 2)
        
        # Standard error
        n = len(outcome)
        se = np.sqrt(np.sum((residuals_outcome_all - theta * residuals_treatment_all) ** 2) / 
                     (n * np.sum(residuals_treatment_all ** 2)))
        
        # Confidence interval
        ci = (theta - 1.96 * se, theta + 1.96 * se)
        
        # P-value
        t_stat = theta / se
        pval = 2 * (1 - stats.t.cdf(abs(t_stat), n - X.shape[1] - 1))
        
        # Calculate C (correlation between residuals) and G (correlation between predictions)
        corr_resid = np.corrcoef(residuals_outcome_all, residuals_treatment_all)[0, 1]
        corr_pred = np.corrcoef(predictions_outcome_all, predictions_treatment_all)[0, 1]
        
        # Calculate non-cross-validated R² (train on full dataset, predict on full dataset)
        print("  Calculating non-cross-validated R²...", end=' ', flush=True)
        
        # Fit models on entire dataset
        model_outcome_full = xgb.XGBRegressor(
            n_estimators=self.config.analysis.xgb_n_estimators,
            max_depth=self.config.analysis.xgb_max_depth,
            random_state=42,
            tree_method='hist',
            device='cuda'
        )
        model_outcome_full.fit(X, outcome)
        
        model_treatment_full = xgb.XGBRegressor(
            n_estimators=self.config.analysis.xgb_n_estimators,
            max_depth=self.config.analysis.xgb_max_depth,
            random_state=42,
            tree_method='hist',
            device='cuda'
        )
        model_treatment_full.fit(X, treatment)
        
        # Predict on entire dataset
        pred_outcome_full = model_outcome_full.predict(X)
        pred_treatment_full = model_treatment_full.predict(X)
        
        # Calculate non-cross-validated residuals
        residuals_outcome_full = outcome - pred_outcome_full
        residuals_treatment_full = treatment - pred_treatment_full
        
        # Calculate non-cross-validated R²
        r2_y_full = r2_score(outcome, pred_outcome_full)
        r2_x_full = r2_score(treatment, pred_treatment_full)
        
        print("done", flush=True)
        
        return {
            'theta': float(theta),
            'se': float(se),
            'ci': (float(ci[0]), float(ci[1])),
            'pval': float(pval),
            'cross_fitted': True,
            'n_folds': self.config.analysis.dml_n_folds,
            'r2_y': float(np.mean(r2_outcome_scores)),  # Cross-validated R²
            'r2_x': float(np.mean(r2_treatment_scores)),  # Cross-validated R²
            'r2_y_full': float(r2_y_full),  # Non-cross-validated R²
            'r2_x_full': float(r2_x_full),  # Non-cross-validated R²
            'corr_resid': float(corr_resid),
            'corr_pred': float(corr_pred),
            # Include residuals and predictions for export
            'residuals': {
                'outcome': residuals_outcome_all,
                'treatment': residuals_treatment_all
            },
            'predictions': {
                'outcome': predictions_outcome_all,
                'treatment': predictions_treatment_all
            },
            # Include non-CV residuals and predictions
            'residuals_full': {
                'outcome': residuals_outcome_full,
                'treatment': residuals_treatment_full
            },
            'predictions_full': {
                'outcome': pred_outcome_full,
                'treatment': pred_treatment_full
            }
        }
    
    def _calculate_r2(self, X: np.ndarray, Y: np.ndarray) -> Dict:
        """Calculate R² with cross-validation."""
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        r2_scores = []
        
        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            Y_train, Y_test = Y[train_idx], Y[test_idx]
            
            model = xgb.XGBRegressor(
                n_estimators=self.config.analysis.xgb_n_estimators,
                max_depth=self.config.analysis.xgb_max_depth,
                random_state=42,
                tree_method='hist',
                device='cuda'
            )
            model.fit(X_train, Y_train)
            
            Y_pred = model.predict(X_test)
            r2_scores.append(r2_score(Y_test, Y_pred))
        
        return {
            'mean': float(np.mean(r2_scores)),
            'std': float(np.std(r2_scores)),
            'all_folds': [float(r) for r in r2_scores]
        }
    
    def _calculate_contributions(self, X: np.ndarray, Y: np.ndarray,
                               feature_names: List[str]) -> np.ndarray:
        """Calculate SHAP values using XGBoost's native method."""
        model = xgb.XGBRegressor(
            n_estimators=self.config.analysis.xgb_n_estimators,
            max_depth=self.config.analysis.xgb_max_depth,
            random_state=42,
            tree_method='hist',
            device='cuda'
        )
        model.fit(X, Y)
        
        # Use XGBoost's native SHAP calculation which works with GPU models
        # This avoids serialization issues with the SHAP library
        dmatrix = xgb.DMatrix(X)
        
        # pred_contribs=True returns SHAP values with base value as last column
        shap_values_with_base = model.get_booster().predict(dmatrix, pred_contribs=True)
        
        # Extract SHAP values (all columns except last which is base value)
        shap_values = shap_values_with_base[:, :-1]
        
        return shap_values
    
    def prepare_dml_table_data(self) -> Dict:
        """Prepare DML results for visualization table."""
        if not self.results:
            raise ValueError("No DML results available. Run analysis first.")
        
        table_data = {}
        
        # Get all outcome pairs
        for key, value in self.results.items():
            if '_to_' in key:
                # This is a DML result
                treatment = value['treatment']
                outcome = value['outcome']
                
                table_key = f"{treatment}_to_{outcome}"
                table_data[table_key] = {
                    'treatment': treatment,
                    'outcome': outcome,
                    'naive': value['naive'],
                    'dml': value['dml'],
                    'reduction': value['reduction']
                }
        
        return table_data
    
    def get_top_pcs(self, n_pcs: int = 5) -> List[int]:
        """Get indices of top PCs based on overall importance."""
        if self.config.analysis.dml_top_pcs:
            return self.config.analysis.dml_top_pcs[:n_pcs]
        
        # Otherwise, select based on contribution magnitude
        all_contributions = []
        for key, value in self.results.items():
            if key.startswith('contributions_'):
                all_contributions.append(np.abs(value))
        
        if not all_contributions:
            return list(range(n_pcs))
        
        # Average absolute contributions
        avg_contributions = np.mean(all_contributions, axis=0).mean(axis=0)
        top_indices = np.argsort(avg_contributions)[-n_pcs:][::-1]
        
        return top_indices.tolist()