#!/usr/bin/env python3
"""Automatic parameter selection system with full transparency."""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import warnings

@dataclass
class ParameterDecision:
    """Records a single parameter decision."""
    parameter: str
    value: Any
    source: str  # 'cli', 'config', 'auto', 'default'
    reason: Optional[str] = None
    alternatives: Optional[Dict[str, Any]] = None
    warnings: Optional[List[str]] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for export."""
        return {
            'parameter': self.parameter,
            'value': self.value,
            'source': self.source,
            'reason': self.reason,
            'alternatives': self.alternatives,
            'warnings': self.warnings
        }


@dataclass
class DatasetProfile:
    """Profile of dataset characteristics."""
    n_samples: int
    n_features: int
    size_category: str  # 'tiny', 'small', 'medium', 'large', 'xlarge'
    text_length_stats: Optional[Dict[str, float]] = None
    sparsity: Optional[float] = None
    
    @classmethod
    def from_data(cls, data_shape: Tuple[int, int], 
                  text_stats: Optional[Dict] = None) -> 'DatasetProfile':
        """Create profile from data characteristics."""
        n_samples, n_features = data_shape
        
        # Categorize dataset size
        if n_samples < 500:
            size_category = 'tiny'
        elif n_samples < 2000:
            size_category = 'small'
        elif n_samples < 10000:
            size_category = 'medium'
        elif n_samples < 50000:
            size_category = 'large'
        else:
            size_category = 'xlarge'
            
        return cls(
            n_samples=n_samples,
            n_features=n_features,
            size_category=size_category,
            text_length_stats=text_stats
        )


class AutoParameterSystem:
    """Intelligent parameter selection with full transparency."""
    
    # Static parameters that should not be auto-adjusted
    STATIC_PARAMS = {'pca_components', 'umap_dimensions', 'dml_top_pcs'}
    
    # ML parameters that are static in auto mode but dynamic in super-auto
    ML_PARAMS = {'dml_n_folds', 'xgb_n_estimators', 'xgb_max_depth', 
                 'lasso_alphas', 'ridge_alphas', 'regularization_cv_folds'}
    
    def __init__(self, dataset_profile: DatasetProfile, 
                 config: Optional[Any] = None,
                 auto_cluster_mode: Optional[str] = None,
                 super_auto_mode: bool = False):
        """Initialize with dataset profile and optional config."""
        self.profile = dataset_profile
        self.config = config
        self.auto_cluster_mode = auto_cluster_mode
        self.super_auto_mode = super_auto_mode
        self.decisions: List[ParameterDecision] = []
        self.education_messages: List[str] = []
        self.super_auto_suggestions: Dict[str, Any] = {}
        
    def select_parameter(self, param_name: str, cli_value: Optional[Any] = None,
                        auto_mode: bool = True) -> Any:
        """
        Select parameter value with hierarchy: CLI > Config > Auto > Default.
        
        Args:
            param_name: Name of parameter
            cli_value: Value from CLI if provided
            auto_mode: Whether auto-selection is enabled
            
        Returns:
            Selected parameter value
        """
        # Check if this is a static parameter
        if param_name in self.STATIC_PARAMS:
            return self._select_static_parameter(param_name, cli_value)
            
        # Check if this is an ML parameter in regular auto mode (not super-auto)
        if param_name in self.ML_PARAMS and auto_mode and not self.super_auto_mode:
            return self._select_ml_parameter_static(param_name, cli_value)
            
        # 1. CLI value (highest priority)
        if cli_value is not None:
            decision = ParameterDecision(
                parameter=param_name,
                value=cli_value,
                source='cli',
                reason='User override from command line'
            )
            self.decisions.append(decision)
            return cli_value
            
        # 2. Config file value (skip if None)
        if self.config and hasattr(self.config.analysis, param_name):
            config_value = getattr(self.config.analysis, param_name)
            if config_value is not None:  # Only use config value if not None
                decision = ParameterDecision(
                    parameter=param_name,
                    value=config_value,
                    source='config',
                    reason='Specified in configuration file'
                )
                # Still compute auto value for comparison
                if auto_mode:
                    auto_value = self._compute_auto_value(param_name)
                    decision.alternatives = {'auto_would_be': auto_value}
                self.decisions.append(decision)
                return config_value
            
        # 3. Auto mode (if enabled)
        if auto_mode:
            auto_value = self._compute_auto_value(param_name)
            source = 'super-auto' if self.super_auto_mode else 'auto'
            decision = ParameterDecision(
                parameter=param_name,
                value=auto_value,
                source=source,
                reason=self._get_auto_reason(param_name, auto_value)
            )
            self.decisions.append(decision)
            return auto_value
            
        # 4. Default fallback
        default_value = self._get_default_value(param_name)
        decision = ParameterDecision(
            parameter=param_name,
            value=default_value,
            source='default',
            reason='Fallback default value',
            warnings=['Consider using auto mode for better results']
        )
        self.decisions.append(decision)
        return default_value
        
    def _select_ml_parameter_static(self, param_name: str, cli_value: Optional[Any]) -> Any:
        """Handle ML parameters in regular auto mode (keep static but show suggestions)."""
        # Always compute what super-auto would suggest
        super_auto_value = self._compute_auto_value(param_name)
        super_auto_reason = self._get_auto_reason(param_name, super_auto_value)
        self.super_auto_suggestions[param_name] = {
            'value': super_auto_value,
            'reason': super_auto_reason
        }
        
        # 1. CLI value (highest priority)
        if cli_value is not None:
            decision = ParameterDecision(
                parameter=param_name,
                value=cli_value,
                source='cli',
                reason='User override from command line',
                alternatives={'super_auto_would_be': super_auto_value}
            )
            self.decisions.append(decision)
            return cli_value
            
        # 2. Config file value (skip if None)
        if self.config and hasattr(self.config.analysis, param_name):
            config_value = getattr(self.config.analysis, param_name)
            if config_value is not None:  # Only use config value if not None
                decision = ParameterDecision(
                    parameter=param_name,
                    value=config_value,
                    source='config',
                    reason='ML parameter (static in auto mode)',
                    alternatives={'super_auto_would_be': super_auto_value}
                )
                self.decisions.append(decision)
                return config_value
            
        # 3. Default value (for ML params in auto mode)
        default_value = self._get_default_value(param_name)
        decision = ParameterDecision(
            parameter=param_name,
            value=default_value,
            source='default',
            reason='ML parameter (static in auto mode)',
            alternatives={'super_auto_would_be': super_auto_value}
        )
        self.decisions.append(decision)
        
        # Add educational message about super-auto
        if param_name == 'dml_n_folds' and super_auto_value != default_value:
            self.education_messages.append(
                f"💡 Super-auto would use {super_auto_value} DML folds for this {self.profile.size_category} dataset"
            )
        
        return default_value
        
    def _select_static_parameter(self, param_name: str, cli_value: Optional[Any]) -> Any:
        """Handle static parameters that shouldn't be auto-adjusted."""
        # Static defaults
        static_defaults = {
            'pca_components': 200,
            'umap_dimensions': 3,
            'dml_top_pcs': None  # This means auto-select 6
        }
        
        if cli_value is not None:
            decision = ParameterDecision(
                parameter=param_name,
                value=cli_value,
                source='cli',
                reason=f'Static parameter overridden by CLI'
            )
            self.decisions.append(decision)
            return cli_value
            
        if self.config and hasattr(self.config.analysis, param_name):
            config_value = getattr(self.config.analysis, param_name)
            decision = ParameterDecision(
                parameter=param_name,
                value=config_value,
                source='config',
                reason=f'Static parameter from config'
            )
            self.decisions.append(decision)
            return config_value
            
        default_value = static_defaults.get(param_name)
        decision = ParameterDecision(
            parameter=param_name,
            value=default_value,
            source='static',
            reason=f'Standard value for consistency across analyses'
        )
        self.decisions.append(decision)
        return default_value
        
    def _compute_auto_value(self, param_name: str) -> Any:
        """Compute intelligent auto value based on dataset characteristics."""
        compute_methods = {
            # Clustering parameters
            'hdbscan_min_cluster_size': self._compute_min_cluster_size,
            'hdbscan_min_samples': self._compute_min_samples,
            
            # UMAP parameters
            'umap_n_neighbors': self._compute_umap_neighbors,
            'umap_min_dist': self._compute_umap_min_dist,
            
            # DML parameters
            'dml_n_folds': self._compute_dml_folds,
            
            # XGBoost parameters
            'xgb_n_estimators': self._compute_xgb_estimators,
            'xgb_max_depth': self._compute_xgb_depth,
            
            # Regularization parameters
            'lasso_alphas': self._compute_lasso_alphas,
            'ridge_alphas': self._compute_ridge_alphas,
            'regularization_cv_folds': self._compute_reg_cv_folds,
            
            # Processing parameters
            'batch_size': self._compute_batch_size,
            'max_text_length': self._compute_max_text_length
        }
        
        if param_name in compute_methods:
            return compute_methods[param_name]()
        else:
            return self._get_default_value(param_name)
            
    def _compute_min_cluster_size(self) -> int:
        """Compute HDBSCAN min_cluster_size based on dataset and target."""
        n_samples = self.profile.n_samples
        
        # Use auto_cluster_mode if specified
        if self.auto_cluster_mode:
            if self.auto_cluster_mode == 'many':
                cluster_percent = 0.002  # 0.2%
                min_size, max_size = 15, 50
            elif self.auto_cluster_mode == 'medium':
                cluster_percent = 0.005  # 0.5%
                min_size, max_size = 30, 100
            else:  # 'few'
                cluster_percent = 0.01   # 1%
                min_size, max_size = 50, 200
        else:
            # Default to 'medium' behavior
            cluster_percent = 0.005
            min_size, max_size = 30, 100
            
        suggested_size = int(n_samples * cluster_percent)
        return max(min_size, min(suggested_size, max_size))
        
    def _compute_min_samples(self) -> int:
        """Compute HDBSCAN min_samples based on min_cluster_size."""
        # First get min_cluster_size
        min_cluster_size = None
        for decision in self.decisions:
            if decision.parameter == 'hdbscan_min_cluster_size':
                min_cluster_size = decision.value
                break
                
        if min_cluster_size is None:
            # Compute it if not yet decided
            min_cluster_size = self._compute_min_cluster_size()
            
        # min_samples is typically 10-25% of min_cluster_size
        return max(5, int(min_cluster_size * 0.15))
        
    def _compute_umap_neighbors(self) -> int:
        """Compute UMAP n_neighbors based on dataset size."""
        n_samples = self.profile.n_samples
        
        if n_samples < 1000:
            return 15
        elif n_samples < 5000:
            return 30
        elif n_samples < 20000:
            return 50
        else:
            return 100
            
    def _compute_umap_min_dist(self) -> float:
        """Compute UMAP min_dist based on clustering mode."""
        if self.auto_cluster_mode == 'many':
            return 0.0  # Tight clusters
        elif self.auto_cluster_mode == 'few':
            return 0.3  # Loose clusters
        else:
            return 0.1  # Medium separation
            
    def _compute_dml_folds(self) -> int:
        """Compute DML n_folds based on dataset size."""
        n_samples = self.profile.n_samples
        
        if n_samples < 500:
            self.education_messages.append(
                "⚠️ Small dataset detected. Using 3-fold CV to maximize training data."
            )
            return 3
        elif n_samples < 2000:
            return 5
        elif n_samples < 10000:
            return 5
        else:
            self.education_messages.append(
                "📊 Large dataset detected. Using 10-fold CV for stable estimates."
            )
            return 10
            
    def _compute_xgb_estimators(self) -> int:
        """Compute XGBoost n_estimators based on dataset."""
        n_samples = self.profile.n_samples
        
        if n_samples < 1000:
            return 50  # Prevent overfitting
        elif n_samples < 10000:
            return 100  # Standard
        else:
            return 200  # More trees for complex data
            
    def _compute_xgb_depth(self) -> int:
        """Compute XGBoost max_depth to prevent overfitting."""
        n_samples = self.profile.n_samples
        n_features = self.profile.n_features
        
        # Calculate feature/sample ratio
        ratio = n_features / n_samples
        
        if n_samples < 1000 or ratio > 0.5:
            self.education_messages.append(
                "⚠️ High feature/sample ratio. Using shallow trees to prevent overfitting."
            )
            return 3
        elif n_samples < 5000:
            return 5
        elif n_samples < 20000:
            return 6
        else:
            return 8
            
    def _compute_batch_size(self) -> int:
        """Compute batch size based on available memory."""
        # This would ideally check GPU memory
        # For now, use conservative estimates
        if self.profile.n_features > 1000:
            return 16
        elif self.profile.n_features > 500:
            return 32
        else:
            return 64
            
    def _compute_max_text_length(self) -> int:
        """Compute max text length based on text statistics."""
        if self.profile.text_length_stats and 'p95' in self.profile.text_length_stats:
            # Use 95th percentile as cutoff
            return min(1024, int(self.profile.text_length_stats['p95']))
        else:
            return 512  # Default
            
    def _compute_lasso_alphas(self) -> List[float]:
        """Compute Lasso alpha range based on dataset characteristics."""
        n_samples = self.profile.n_samples
        n_features = self.profile.n_features
        
        # Base alpha range
        if n_samples < 1000:
            # Small dataset: stronger regularization
            return [0.001, 0.01, 0.1, 1.0, 10.0]
        elif n_samples < 10000:
            # Medium dataset: moderate regularization
            return [0.0001, 0.001, 0.01, 0.1, 1.0]
        else:
            # Large dataset: weaker regularization
            return [0.00001, 0.0001, 0.001, 0.01, 0.1]
            
    def _compute_ridge_alphas(self) -> List[float]:
        """Compute Ridge alpha range based on dataset characteristics."""
        n_samples = self.profile.n_samples
        
        if n_samples < 1000:
            # Small dataset: wider range
            return [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
        elif n_samples < 10000:
            # Medium dataset: standard range
            return [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
        else:
            # Large dataset: focus on smaller alphas
            return [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]
            
    def _compute_reg_cv_folds(self) -> int:
        """Compute CV folds for regularization parameter selection."""
        # Use same logic as DML folds
        return self._compute_dml_folds()
            
    def _get_auto_reason(self, param_name: str, value: Any) -> str:
        """Get human-readable reason for auto selection."""
        # Handle list parameters separately
        if param_name == 'lasso_alphas' and isinstance(value, list):
            return f"Alpha range {value[0]}-{value[-1]} for {self.profile.size_category} dataset"
        elif param_name == 'ridge_alphas' and isinstance(value, list):
            return f"Alpha range {value[0]}-{value[-1]} for {self.profile.size_category} dataset"
        
        reasons = {
            'hdbscan_min_cluster_size': f"Set to {value} ({value/self.profile.n_samples*100:.1f}% of data) "
                                       f"for {self.auto_cluster_mode or 'medium'} clustering",
            'hdbscan_min_samples': f"Set to {value} (15% of min_cluster_size) for flexibility",
            'umap_n_neighbors': f"Set to {value} based on {self.profile.size_category} dataset size",
            'umap_min_dist': f"Set to {value} for {self.auto_cluster_mode or 'medium'} cluster separation",
            'dml_n_folds': f"Set to {value} folds for {self.profile.size_category} dataset",
            'xgb_n_estimators': f"Set to {value} trees for {self.profile.size_category} dataset",
            'xgb_max_depth': f"Set to {value} to balance complexity and overfitting",
            'regularization_cv_folds': f"Set to {value} folds for regularization tuning",
            'batch_size': f"Set to {value} for efficient GPU utilization",
            'max_text_length': f"Set to {value} tokens based on text length distribution"
        }
        return reasons.get(param_name, f"Auto-selected based on dataset characteristics")
        
    def _get_default_value(self, param_name: str) -> Any:
        """Get fallback default value."""
        defaults = {
            'hdbscan_min_cluster_size': 50,
            'hdbscan_min_samples': 10,
            'umap_n_neighbors': 15,
            'umap_min_dist': 0.1,
            'dml_n_folds': 5,
            'xgb_n_estimators': 100,
            'xgb_max_depth': 5,
            'lasso_alphas': [0.001, 0.01, 0.1, 1.0, 10.0],  # Default Lasso alphas
            'ridge_alphas': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],  # Default Ridge alphas
            'regularization_cv_folds': 5,
            'batch_size': 32,
            'max_text_length': 512
        }
        return defaults.get(param_name, None)
        
    def generate_report(self) -> str:
        """Generate human-readable parameter report."""
        lines = [
            "\n🔬 PERCEPTIONML PARAMETER SELECTION REPORT",
            "=" * 60,
            f"Dataset: {self.profile.n_samples:,} samples, {self.profile.n_features} features",
            f"Size category: {self.profile.size_category.upper()}",
            ""
        ]
        
        # Show mode
        if self.super_auto_mode:
            lines.append("🚀 Mode: SUPER-AUTO (full dynamic optimization)")
        else:
            lines.append("🤖 Mode: AUTO (ML parameters static, clustering dynamic)")
        
        if self.auto_cluster_mode:
            lines.append(f"Clustering target: {self.auto_cluster_mode.upper()}")
        lines.append("")
            
        # Group parameters by category
        categories = {
            'Static Parameters': ['pca_components', 'umap_dimensions', 'dml_top_pcs'],
            'Clustering Parameters': ['hdbscan_min_cluster_size', 'hdbscan_min_samples'],
            'UMAP Parameters': ['umap_n_neighbors', 'umap_min_dist'],
            'ML Parameters': ['dml_n_folds', 'xgb_n_estimators', 'xgb_max_depth'],
            'Regularization Parameters': ['lasso_alphas', 'ridge_alphas', 'regularization_cv_folds'],
            'Processing Parameters': ['batch_size', 'max_text_length']
        }
        
        # Create parameter lookup
        param_decisions = {d.parameter: d for d in self.decisions}
        
        for category, params in categories.items():
            category_decisions = [param_decisions.get(p) for p in params if p in param_decisions]
            if category_decisions:
                lines.append(f"📊 {category}:")
                for decision in category_decisions:
                    if decision:
                        source_emoji = {
                            'cli': '🎯',
                            'config': '📄',
                            'auto': '🤖',
                            'default': '📌',
                            'static': '🔒'
                        }.get(decision.source, '•')
                        
                        line = f"  {source_emoji} {decision.parameter}: {decision.value}"
                        if decision.source == 'auto':
                            line += f" ({decision.reason})"
                        elif decision.source == 'super-auto':
                            line += f" (dynamically optimized: {decision.reason})"
                        elif decision.source == 'config' and decision.alternatives:
                            auto_val = decision.alternatives.get('auto_would_be')
                            super_auto_val = decision.alternatives.get('super_auto_would_be')
                            if super_auto_val and super_auto_val != decision.value:
                                line += f" [super-auto would use: {super_auto_val}]"
                            elif auto_val and auto_val != decision.value:
                                line += f" [auto would select: {auto_val}]"
                        elif decision.source == 'default' and decision.alternatives:
                            super_auto_val = decision.alternatives.get('super_auto_would_be')
                            if super_auto_val and super_auto_val != decision.value:
                                # Format list values nicely
                                if isinstance(super_auto_val, list):
                                    line += f" [super-auto: {super_auto_val[0]}-{super_auto_val[-1]}]"
                                else:
                                    line += f" [super-auto: {super_auto_val}]"
                        lines.append(line)
                lines.append("")
                
        # Add education messages
        if self.education_messages:
            lines.append("💡 Insights:")
            for msg in self.education_messages:
                lines.append(f"  {msg}")
            lines.append("")
            
        # Add suggestions
        lines.extend([
            "📝 Suggestions:",
            f"  • Your dataset size ({self.profile.n_samples:,}) is well-suited for this analysis",
            f"  • Consider --auto-cluster [few|medium|many] to adjust clustering granularity",
            f"  • Use --export-csv to save all results for further analysis"
        ])
        
        # Add super-auto suggestion if not in super-auto mode
        if not self.super_auto_mode and self.super_auto_suggestions:
            lines.extend([
                "",
                "🚀 Super-Auto Mode:",
                "  For full ML hyperparameter optimization, use: --super-auto",
                "  This would dynamically adjust:"
            ])
            
            # Show what would be different
            different_params = []
            for param, info in self.super_auto_suggestions.items():
                current_val = None
                for decision in self.decisions:
                    if decision.parameter == param:
                        current_val = decision.value
                        break
                        
                if current_val and current_val != info['value']:
                    if isinstance(info['value'], list):
                        different_params.append(f"    • {param}: {info['value'][0]}-{info['value'][-1]} "
                                              f"(currently: {current_val[0] if isinstance(current_val, list) else current_val})")
                    else:
                        different_params.append(f"    • {param}: {info['value']} (currently: {current_val})")
                        
            if different_params:
                lines.extend(different_params)
            else:
                lines.append("    • All ML parameters are already at optimal values")
                
        lines.append("")
        
        return "\n".join(lines)
        
    def export_metadata(self) -> dict:
        """Export all decisions and metadata for reproducibility."""
        return {
            'dataset_profile': {
                'n_samples': self.profile.n_samples,
                'n_features': self.profile.n_features,
                'size_category': self.profile.size_category,
                'text_length_stats': self.profile.text_length_stats
            },
            'mode': 'super-auto' if self.super_auto_mode else 'auto',
            'auto_cluster_mode': self.auto_cluster_mode,
            'parameter_decisions': [d.to_dict() for d in self.decisions],
            'super_auto_suggestions': self.super_auto_suggestions if not self.super_auto_mode else None,
            'education_messages': self.education_messages,
            'reproducibility': {
                'manual_command': self._generate_manual_command(),
                'super_auto_command': self._generate_super_auto_command() if not self.super_auto_mode else None,
                'notes': 'All auto-selected parameters can be manually specified using CLI flags'
            }
        }
        
    def _generate_manual_command(self) -> str:
        """Generate CLI command to reproduce with manual parameters."""
        flags = []
        for decision in self.decisions:
            if decision.source in ['auto', 'super-auto']:
                param_map = {
                    'hdbscan_min_cluster_size': '--min-cluster-size',
                    'hdbscan_min_samples': '--min-samples',
                    'umap_n_neighbors': '--umap-neighbors',
                    'umap_min_dist': '--umap-min-dist',
                    'dml_n_folds': '--dml-folds',
                    'xgb_n_estimators': '--xgb-estimators',
                    'xgb_max_depth': '--xgb-depth',
                    'lasso_alphas': '--lasso-alphas',
                    'ridge_alphas': '--ridge-alphas',
                    'regularization_cv_folds': '--reg-cv-folds',
                    'batch_size': '--batch-size',
                    'max_text_length': '--max-text-length'
                }
                if decision.parameter in param_map:
                    if isinstance(decision.value, list):
                        # Format list parameters
                        value_str = ','.join(str(v) for v in decision.value)
                        flags.append(f"{param_map[decision.parameter]} '{value_str}'")
                    else:
                        flags.append(f"{param_map[decision.parameter]} {decision.value}")
                    
        return f"python run_pipeline.py -c config.yaml -d data.csv {' '.join(flags)}"
        
    def _generate_super_auto_command(self) -> str:
        """Generate CLI command for super-auto mode."""
        base_cmd = "python run_pipeline.py -c config.yaml -d data.csv --super-auto"
        if self.auto_cluster_mode:
            base_cmd += f" --auto-cluster {self.auto_cluster_mode}"
        return base_cmd