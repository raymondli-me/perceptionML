"""
Double Machine Learning (DML) implementation
Follows Chernozhukov et al. (2018) with sandwich estimator
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
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import r2_score
import xgboost as xgb
from scipy import stats
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')


class DMLAnalyzer:
    """Double Machine Learning analyzer with cross-fitting"""
    
    def __init__(self, 
                 n_folds: int = 5,
                 random_seed: int = 42):
        """
        Initialize DML analyzer
        
        Args:
            n_folds: Number of cross-fitting folds
            random_seed: Random seed for reproducibility
        """
        self.n_folds = n_folds
        self.random_seed = random_seed
        np.random.seed(random_seed)
        # Also set random state for other libraries
        import random
        random.seed(random_seed)
        
    def fit_ols(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """
        Fit OLS regression with robust standard errors
        
        Args:
            X: Features
            y: Target
            
        Returns:
            Dictionary with coefficient, SE, p-value, CI, and R²
        """
        # Handle single feature case
        if X.ndim == 1:
            X = X.reshape(-1, 1)
            
        # Fit OLS
        model = LinearRegression()
        model.fit(X, y)
        
        # Get predictions and residuals
        y_pred = model.predict(X)
        residuals = y - y_pred
        
        # Calculate R²
        r2 = r2_score(y, y_pred)
        
        # Get coefficient (first coefficient for single feature)
        coef = model.coef_[0] if X.shape[1] == 1 else model.coef_
        
        # Calculate robust standard errors (HC0)
        n, k = X.shape
        
        # Add intercept for SE calculation
        X_with_intercept = np.column_stack([np.ones(n), X])
        
        # Calculate HC0 robust covariance matrix
        XtX_inv = np.linalg.inv(X_with_intercept.T @ X_with_intercept)
        
        # Heteroskedasticity-consistent covariance
        S = np.zeros((k+1, k+1))
        for i in range(n):
            xi = X_with_intercept[i].reshape(-1, 1)
            S += (residuals[i]**2) * (xi @ xi.T)
        
        # Robust covariance matrix
        V_HC0 = XtX_inv @ S @ XtX_inv
        
        # Standard error for first coefficient (after intercept)
        se = np.sqrt(V_HC0[1, 1])
        
        # T-statistic and p-value
        t_stat = coef / se
        p_value = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=n-k-1))
        
        # 95% confidence interval
        t_critical = stats.t.ppf(0.975, df=n-k-1)
        ci_lower = coef - t_critical * se
        ci_upper = coef + t_critical * se
        
        return {
            'coefficient': float(coef),
            'se': float(se),
            'p_value': float(p_value),
            'ci': (float(ci_lower), float(ci_upper)),
            'r2': float(r2),
            'predictions': y_pred,
            'residuals': residuals
        }
    
    def fit_dml(self, 
                X: np.ndarray, 
                y: np.ndarray,
                Z: np.ndarray,
                learner: str = 'xgboost',
                use_ols_se: bool = False) -> Dict:
        """
        Fit Double Machine Learning with cross-fitting
        
        Args:
            X: Treatment/confounding features
            y: Outcome
            Z: Control features (embeddings)
            learner: ML method ('xgboost', 'lasso', 'ridge')
            
        Returns:
            Dictionary with DML estimates
        """
        n = len(y)
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_seed)
        
        # Initialize arrays for cross-fitted residuals
        X_residuals = np.zeros(n)
        y_residuals = np.zeros(n)
        
        # Arrays for predictions (for correlation calculations)
        X_predictions = np.zeros(n)
        y_predictions = np.zeros(n)
        
        # Cross-validation R² tracking
        cv_r2_X = []
        cv_r2_y = []
        
        # Cross-fitting
        for train_idx, test_idx in kf.split(Z):
            Z_train, Z_test = Z[train_idx], Z[test_idx]
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Fit learners
            if learner == 'xgboost':
                # Predict X from Z
                model_X = xgb.XGBRegressor(
                    n_estimators=100,
                    max_depth=3,
                    learning_rate=0.1,
                    random_state=self.random_seed,
                    n_jobs=1,
                    tree_method='hist'
                )
                model_X.fit(Z_train, X_train)
                X_pred = model_X.predict(Z_test)
                
                # Predict y from Z
                model_y = xgb.XGBRegressor(
                    n_estimators=100,
                    max_depth=3,
                    learning_rate=0.1,
                    random_state=self.random_seed,
                    n_jobs=1,
                    tree_method='hist'
                )
                model_y.fit(Z_train, y_train)
                y_pred = model_y.predict(Z_test)
                
            elif learner == 'lasso':
                # Standardize features
                scaler = StandardScaler()
                Z_train_scaled = scaler.fit_transform(Z_train)
                Z_test_scaled = scaler.transform(Z_test)
                
                # Predict X from Z
                model_X = LassoCV(cv=5, random_state=self.random_seed, n_jobs=-1)
                model_X.fit(Z_train_scaled, X_train)
                X_pred = model_X.predict(Z_test_scaled)
                
                # Predict y from Z
                model_y = LassoCV(cv=5, random_state=self.random_seed, n_jobs=-1)
                model_y.fit(Z_train_scaled, y_train)
                y_pred = model_y.predict(Z_test_scaled)
                
            elif learner == 'ridge':
                # Standardize features
                scaler = StandardScaler()
                Z_train_scaled = scaler.fit_transform(Z_train)
                Z_test_scaled = scaler.transform(Z_test)
                
                # Predict X from Z
                model_X = RidgeCV(cv=5)
                model_X.fit(Z_train_scaled, X_train)
                X_pred = model_X.predict(Z_test_scaled)
                
                # Predict y from Z
                model_y = RidgeCV(cv=5)
                model_y.fit(Z_train_scaled, y_train)
                y_pred = model_y.predict(Z_test_scaled)
                
            else:
                raise ValueError(f"Unknown learner: {learner}")
            
            # Calculate residuals
            X_residuals[test_idx] = X_test - X_pred
            y_residuals[test_idx] = y_test - y_pred
            
            # Store predictions
            X_predictions[test_idx] = X_pred
            y_predictions[test_idx] = y_pred
            
            # Calculate fold R²
            cv_r2_X.append(r2_score(X_test, X_pred))
            cv_r2_y.append(r2_score(y_test, y_pred))
        
        # DML point estimate: θ̂ = Σ(e_Xi * e_Yi) / Σ(e_Xi²)
        theta_hat = np.sum(X_residuals * y_residuals) / np.sum(X_residuals**2)
        
        # Variance estimation (sandwich estimator)
        # σ̂²_θ = (1/n²) * Σ(e_Xi * e_Yi - θ̂ * e_Xi²)² / (1/n Σ e_Xi²)²
        variance_components = (X_residuals * y_residuals - theta_hat * X_residuals**2)**2
        variance = np.sum(variance_components) / (n * np.sum(X_residuals**2))**2
        se = np.sqrt(variance)
        
        # T-statistic and p-value
        t_stat = theta_hat / se
        p_value = 2 * (1 - stats.norm.cdf(np.abs(t_stat)))
        
        # 95% confidence interval
        ci_lower = theta_hat - 1.96 * se
        ci_upper = theta_hat + 1.96 * se
        
        # Full sample R² (for reporting)
        # Refit on full data for final R²
        if learner == 'xgboost':
            model_X_full = xgb.XGBRegressor(n_estimators=100, max_depth=3, 
                                           learning_rate=0.1, random_state=self.random_seed)
            model_X_full.fit(Z, X)
            X_pred_full = model_X_full.predict(Z)
            
            model_y_full = xgb.XGBRegressor(n_estimators=100, max_depth=3,
                                           learning_rate=0.1, random_state=self.random_seed)
            model_y_full.fit(Z, y)
            y_pred_full = model_y_full.predict(Z)
            
        elif learner in ['lasso', 'ridge']:
            scaler = StandardScaler()
            Z_scaled = scaler.fit_transform(Z)
            
            if learner == 'lasso':
                model_X_full = LassoCV(cv=5, random_state=self.random_seed)
                model_y_full = LassoCV(cv=5, random_state=self.random_seed)
            else:
                model_X_full = RidgeCV(cv=5)
                model_y_full = RidgeCV(cv=5)
                
            model_X_full.fit(Z_scaled, X)
            X_pred_full = model_X_full.predict(Z_scaled)
            
            model_y_full.fit(Z_scaled, y)
            y_pred_full = model_y_full.predict(Z_scaled)
        
        r2_X_full = r2_score(X, X_pred_full)
        r2_y_full = r2_score(y, y_pred_full)
        
        # Calculate G and C correlations
        G = np.corrcoef(X_predictions, y_predictions)[0, 1]
        C = np.corrcoef(X_residuals, y_residuals)[0, 1]
        
        # Correlations between actual and predicted
        corr_X = np.corrcoef(X, X_predictions)[0, 1]
        corr_y = np.corrcoef(y, y_predictions)[0, 1]
        
        return {
            'coefficient': float(theta_hat),
            'se': float(se),
            'p_value': float(p_value),
            'ci': (float(ci_lower), float(ci_upper)),
            'r2_x_full': float(r2_X_full),
            'r2_y_full': float(r2_y_full),
            'r2_x_cv': cv_r2_X,
            'r2_y_cv': cv_r2_y,
            'r2_x_cv_mean': float(np.mean(cv_r2_X)),
            'r2_y_cv_mean': float(np.mean(cv_r2_y)),
            'G': float(G),
            'C': float(C),
            'corr_x': float(corr_X),
            'corr_y': float(corr_y),
            'X_residuals': X_residuals,
            'y_residuals': y_residuals,
            'X_predictions': X_predictions,
            'y_predictions': y_predictions
        }
    
    def select_top_features(self,
                          X: np.ndarray,
                          y: np.ndarray,
                          method: str = 'xgboost',
                          n_features: int = 6) -> np.ndarray:
        """
        Select top features using various methods
        
        Args:
            X: Feature matrix
            y: Target variable
            method: Selection method ('xgboost', 'lasso', 'ridge', 'ols', 'mi')
            n_features: Number of features to select
            
        Returns:
            Array of selected feature indices
        """
        if method == 'xgboost':
            model = xgb.XGBRegressor(n_estimators=100, max_depth=3,
                                   random_state=self.random_seed)
            model.fit(X, y)
            importances = model.feature_importances_
            
        elif method == 'lasso':
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            model = LassoCV(cv=5, random_state=self.random_seed)
            model.fit(X_scaled, y)
            importances = np.abs(model.coef_)
            
        elif method == 'ridge':
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            model = RidgeCV(cv=5)
            model.fit(X_scaled, y)
            importances = np.abs(model.coef_)
            
        elif method == 'ols':
            model = LinearRegression()
            model.fit(X, y)
            importances = np.abs(model.coef_)
            
        elif method == 'mi':
            importances = mutual_info_regression(X, y, random_state=self.random_seed)
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Get top indices
        top_indices = np.argsort(importances)[-n_features:][::-1]
        
        return top_indices