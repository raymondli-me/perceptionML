"""
Double Machine Learning implementation matching the notebook exactly
This module implements the exact methods from the notebook to ensure reproducibility
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from scipy import stats
import xgboost as xgb
from typing import Dict, Tuple


class NotebookDMLAnalyzer:
    """DML analyzer that exactly matches the notebook implementation"""
    
    def __init__(self, n_folds: int = 5, random_state: int = 42):
        self.n_folds = n_folds
        self.random_state = random_state
        np.random.seed(random_state)
        
    def run_ols(self, features, treatment, outcome, feature_name='N/A', emb_type='None'):
        """
        Runs OLS regression exactly as in the notebook with heteroskedasticity-robust standard errors.
        
        This matches the notebook's _run_ols method behavior.
        """
        desc = "Vanilla OLS" if features is None else f"Standard OLS with {features.shape[1]} features"
        print(f"-> Running {desc}...")

        # Full model for coefficient estimation
        X_full = sm.add_constant(treatment) if features is None else sm.add_constant(np.c_[treatment, features])
        model = sm.OLS(outcome, X_full).fit()
        robust_results = model.get_robustcov_results(cov_type='HC0')  # White sandwich estimator

        # Extract coefficient, SE, and 95% CI for treatment variable
        theta = robust_results.params[1]
        se = robust_results.bse[1]
        ci_lower, ci_upper = robust_results.conf_int()[1]

        # Calculate cross-validated R² if features are present
        if features is not None:
            print(f"   Calculating CV R² and correlations for OLS...")
            kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
            r2_y_folds, r2_x_folds = [], []
            
            # Initialize arrays to store cross-fitted predictions and residuals
            cv_pred_y = np.zeros_like(outcome, dtype=float)
            cv_pred_t = np.zeros_like(treatment, dtype=float)
            cv_res_y = np.zeros_like(outcome, dtype=float)
            cv_res_t = np.zeros_like(treatment, dtype=float)
            
            # Also store predictions from models using only features (for comparable C correlation)
            cv_pred_y_features_only = np.zeros_like(outcome, dtype=float)

            for fold_idx, (train_idx, test_idx) in enumerate(kf.split(features)):
                # Get test data
                y_test = outcome[test_idx].astype(float)
                t_test = treatment[test_idx].astype(float)
                X_test = features[test_idx]

                # First, predict Y using ONLY features (not treatment) for comparable residuals
                X_train_features_only = sm.add_constant(features[train_idx])
                ols_y_features_only = sm.OLS(outcome[train_idx], X_train_features_only).fit()
                X_test_features_only = sm.add_constant(X_test)
                pred_y_features_only = ols_y_features_only.predict(X_test_features_only)
                cv_pred_y_features_only[test_idx] = pred_y_features_only

                # Standard OLS with treatment and features (for theta estimation and R²)
                X_train_with_t = sm.add_constant(np.c_[treatment[train_idx], features[train_idx]])
                ols_y = sm.OLS(outcome[train_idx], X_train_with_t).fit()
                X_test_with_t = sm.add_constant(np.c_[t_test, X_test])
                pred_y = ols_y.predict(X_test_with_t)
                
                # Store cross-fitted predictions for the full model
                cv_pred_y[test_idx] = pred_y

                # Calculate R² for outcome (AI) using features-only model (partial R², DML-like)
                sse_y = np.sum((y_test - pred_y_features_only)**2)
                sst_y = np.sum((y_test - y_test.mean())**2)
                r2_y = 1 - (sse_y / sst_y) if sst_y > 0 else 0
                r2_y_folds.append(r2_y)

                # For SC prediction (using only features)
                X_train_sc = sm.add_constant(features[train_idx])
                ols_sc = sm.OLS(treatment[train_idx], X_train_sc).fit()
                X_test_sc = sm.add_constant(X_test)
                pred_t = ols_sc.predict(X_test_sc)
                
                # Store cross-fitted predictions and residuals for T
                cv_pred_t[test_idx] = pred_t
                cv_res_t[test_idx] = t_test - pred_t

                # Calculate R² for treatment (SC)
                sse_t = np.sum((t_test - pred_t)**2)
                sst_t = np.sum((t_test - t_test.mean())**2)
                r2_t = 1 - (sse_t / sst_t) if sst_t > 0 else 0
                r2_x_folds.append(r2_t)

            # Calculate residuals for Y using features-only predictions (comparable to DML)
            cv_res_y = outcome - cv_pred_y_features_only

            # Calculate G correlation using features-only predictions for both
            G_corr = np.corrcoef(cv_pred_y_features_only, cv_pred_t)[0, 1]
            
            # Calculate C correlation using comparable residuals
            C_corr = np.corrcoef(cv_res_y, cv_res_t)[0, 1]
            
            # Calculate correlations between actual and predicted values
            corr_SC_SChat = np.corrcoef(treatment, cv_pred_t)[0, 1]
            corr_AI_AIhat = np.corrcoef(outcome, cv_pred_y_features_only)[0, 1]

            # Calculate full R² for SC prediction
            X_full_sc = sm.add_constant(features)
            model_sc = sm.OLS(treatment, X_full_sc).fit()
            r2_x_full = model_sc.rsquared
            
            # Calculate full R² for Y using features-only (partial R², DML-like)
            X_full_y_features_only = sm.add_constant(features)
            model_y_features_only = sm.OLS(outcome, X_full_y_features_only).fit()
            r2_y_full = model_y_features_only.rsquared

            print("   ...Done.")
            return {
                'Model Type': 'OLS', 'Nuisance/Selection': 'OLS', 'Embedding': emb_type.title(),
                'Features': feature_name, 'theta': theta, 'se': se,
                't_stat': robust_results.tvalues[1], 'pval': robust_results.pvalues[1],
                'ci': (ci_lower, ci_upper),
                'r2_y_full': r2_y_full, 'r2_y_mean': np.mean(r2_y_folds), 'r2_y_folds': r2_y_folds,
                'r2_x_full': r2_x_full, 'r2_x_mean': np.mean(r2_x_folds), 'r2_x_folds': r2_x_folds,
                'G': G_corr, 'C': C_corr,
                'corr_SC_SChat': corr_SC_SChat, 'corr_AI_AIhat': corr_AI_AIhat
            }
        else:
            # Vanilla OLS without features - no CV possible, no G/C correlations
            print("   ...Done.")
            return {
                'Model Type': 'OLS', 'Nuisance/Selection': 'OLS', 'Embedding': emb_type.title(),
                'Features': feature_name, 'theta': theta, 'se': se,
                't_stat': robust_results.tvalues[1], 'pval': robust_results.pvalues[1],
                'ci': (ci_lower, ci_upper), 'r2_y_full': model.rsquared,
                'r2_y_mean': None, 'r2_y_folds': None, 'r2_x_full': 0.0, 'r2_x_mean': None, 'r2_x_folds': None,
                'G': None, 'C': None,
                'corr_SC_SChat': None, 'corr_AI_AIhat': None
            }