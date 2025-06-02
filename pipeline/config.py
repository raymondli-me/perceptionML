#!/usr/bin/env python3
"""Configuration management for the text embedding pipeline."""

import yaml
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field


@dataclass
class OutcomeConfig:
    """Configuration for an outcome variable."""
    name: str
    display_name: str
    type: str  # 'continuous', 'ordinal', 'categorical'
    range: Optional[List[float]] = None
    categories: Optional[List[str]] = None
    default_thresholds: Dict[str, float] = field(default_factory=dict)
    mode: Optional[str] = None  # 'continuous' or 'zero_presence'
    mode_auto_detected: bool = False  # Track if mode was auto-detected
    
    def __post_init__(self):
        """Validate outcome configuration."""
        if self.type == 'continuous' and self.range is None:
            raise ValueError(f"Continuous outcome '{self.name}' must have a range")
        if self.type == 'categorical' and self.categories is None:
            raise ValueError(f"Categorical outcome '{self.name}' must have categories")
        if self.mode and self.mode not in ['continuous', 'zero_presence']:
            raise ValueError(f"Invalid outcome mode '{self.mode}' for {self.name}. Must be 'continuous' or 'zero_presence'")


@dataclass
class ControlVariable:
    """Configuration for a control variable in DML analysis."""
    name: str
    display_name: Optional[str] = None
    
    def __post_init__(self):
        """Set display name if not provided."""
        if self.display_name is None:
            self.display_name = self.name.replace('_', ' ').title()


@dataclass
class DataConfig:
    """Configuration for data handling."""
    text_column: str
    id_column: str
    outcomes: List[OutcomeConfig]
    sample_size: Optional[int] = None
    sample_seed: Optional[int] = None
    control_variables: Optional[List[ControlVariable]] = None
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'DataConfig':
        """Create DataConfig from dictionary."""
        outcomes = [OutcomeConfig(**o) for o in config_dict['outcomes']]
        
        # Parse control variables if present
        control_vars = None
        if 'control_variables' in config_dict and config_dict['control_variables']:
            control_vars = []
            for cv in config_dict['control_variables']:
                if isinstance(cv, dict):
                    control_vars.append(ControlVariable(**cv))
                else:
                    # Support simple string format
                    control_vars.append(ControlVariable(name=cv))
        
        return cls(
            text_column=config_dict['text_column'],
            id_column=config_dict['id_column'],
            outcomes=outcomes,
            sample_size=config_dict.get('sample_size'),
            sample_seed=config_dict.get('sample_seed'),
            control_variables=control_vars
        )


@dataclass
class AnalysisConfig:
    """Configuration for analysis parameters."""
    pca_components: int = 200
    umap_dimensions: int = 3
    umap_n_neighbors: Optional[int] = 15
    umap_min_dist: Optional[float] = 0.1
    hdbscan_min_cluster_size: Optional[int] = 50
    hdbscan_min_samples: Optional[int] = 10
    dml_top_pcs: Optional[List[int]] = None
    dml_n_folds: Optional[int] = 5
    xgb_n_estimators: Optional[int] = 100
    xgb_max_depth: Optional[int] = 5
    batch_size: Optional[int] = 32
    max_text_length: Optional[int] = 512
    # PC selection methods for DML
    dml_pc_selection_methods: List[str] = field(default_factory=lambda: ['xgboost', 'lasso', 'ridge', 'mi'])
    dml_primary_pc_method: str = 'xgboost'
    # Regularization parameters
    lasso_alphas: Optional[List[float]] = None
    ridge_alphas: Optional[List[float]] = None  
    regularization_cv_folds: Optional[int] = None
    # Auto-parameter mode
    auto_mode: bool = True
    auto_cluster_mode: Optional[str] = None
    super_auto_mode: bool = False
    # Outcome mode detection
    outcome_mode_detection: bool = True  # Enable auto-detection of outcome mode
    outcome_mode_threshold: float = 0.5  # Fraction of zeros to trigger zero_presence mode


@dataclass
class VisualizationConfig:
    """Configuration for visualization parameters."""
    title: str = "Text Embedding Analysis"
    point_size: float = 4.0
    default_opacity: float = 0.8
    essay_font_size: int = 24
    auto_rotate_speed: float = 0.5
    transition_speed: float = 1.5
    topic_text_size: int = 15
    topic_opacity: float = 0.7


@dataclass
class PipelineConfig:
    """Complete pipeline configuration."""
    name: str
    embedding_model: str
    data: DataConfig
    analysis: AnalysisConfig
    visualization: VisualizationConfig
    output_dir: Path
    checkpoint_dir: Path
    
    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            'pipeline': {
                'name': self.name,
                'embedding_model': self.embedding_model
            },
            'data': {
                'text_column': self.data.text_column,
                'id_column': self.data.id_column,
                'outcomes': [
                    {
                        'name': o.name,
                        'display_name': o.display_name,
                        'type': o.type,
                        'range': o.range,
                        'categories': o.categories,
                        'default_thresholds': o.default_thresholds,
                        'mode': o.mode,
                        'mode_auto_detected': o.mode_auto_detected
                    }
                    for o in self.data.outcomes
                ],
                'sample_size': self.data.sample_size,
                'sample_seed': self.data.sample_seed,
                'control_variables': [
                    {'name': cv.name, 'display_name': cv.display_name}
                    for cv in (self.data.control_variables or [])
                ]
            },
            'analysis': {
                'pca_components': self.analysis.pca_components,
                'umap_dimensions': self.analysis.umap_dimensions,
                'umap_n_neighbors': self.analysis.umap_n_neighbors,
                'umap_min_dist': self.analysis.umap_min_dist,
                'hdbscan_min_cluster_size': self.analysis.hdbscan_min_cluster_size,
                'hdbscan_min_samples': self.analysis.hdbscan_min_samples,
                'dml_top_pcs': self.analysis.dml_top_pcs,
                'dml_n_folds': self.analysis.dml_n_folds,
                'xgb_n_estimators': self.analysis.xgb_n_estimators,
                'xgb_max_depth': self.analysis.xgb_max_depth,
                'batch_size': self.analysis.batch_size,
                'max_text_length': self.analysis.max_text_length,
                'dml_pc_selection_methods': self.analysis.dml_pc_selection_methods,
                'dml_primary_pc_method': self.analysis.dml_primary_pc_method,
                'outcome_mode_detection': self.analysis.outcome_mode_detection,
                'outcome_mode_threshold': self.analysis.outcome_mode_threshold
            },
            'visualization': {
                'title': self.visualization.title,
                'point_size': self.visualization.point_size,
                'default_opacity': self.visualization.default_opacity,
                'essay_font_size': self.visualization.essay_font_size,
                'auto_rotate_speed': self.visualization.auto_rotate_speed,
                'transition_speed': self.visualization.transition_speed,
                'topic_text_size': self.visualization.topic_text_size,
                'topic_opacity': self.visualization.topic_opacity
            },
            'output_dir': str(self.output_dir),
            'checkpoint_dir': str(self.checkpoint_dir)
        }
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'PipelineConfig':
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Create sub-configurations
        data_config = DataConfig.from_dict(config_dict['data'])
        analysis_config = AnalysisConfig(**config_dict.get('analysis', {}))
        viz_config = VisualizationConfig(**config_dict.get('visualization', {}))
        
        # Handle paths
        base_dir = Path(config_path).parent.parent
        output_dir = base_dir / config_dict.get('output_dir', 'output')
        checkpoint_dir = base_dir / config_dict.get('checkpoint_dir', 'checkpoints')
        
        return cls(
            name=config_dict['pipeline']['name'],
            embedding_model=config_dict['pipeline']['embedding_model'],
            data=data_config,
            analysis=analysis_config,
            visualization=viz_config,
            output_dir=output_dir,
            checkpoint_dir=checkpoint_dir
        )
    
    def save_checkpoint(self, name: str, data: Any) -> Path:
        """Save checkpoint data."""
        import pickle
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = self.checkpoint_dir / f"{name}.pkl"
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(data, f)
        return checkpoint_path
    
    def load_checkpoint(self, name: str) -> Optional[Any]:
        """Load checkpoint data if exists."""
        import pickle
        checkpoint_path = self.checkpoint_dir / f"{name}.pkl"
        if checkpoint_path.exists():
            with open(checkpoint_path, 'rb') as f:
                return pickle.load(f)
        return None
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'PipelineConfig':
        """Create config from dictionary."""
        # Parse data configuration with outcomes
        outcomes = [OutcomeConfig(**o) for o in config_dict['data']['outcomes']]
        
        # Parse control variables
        control_vars = None
        if 'control_variables' in config_dict['data'] and config_dict['data']['control_variables']:
            control_vars = []
            for cv in config_dict['data']['control_variables']:
                if isinstance(cv, dict):
                    control_vars.append(ControlVariable(**cv))
                else:
                    control_vars.append(ControlVariable(name=cv))
        
        data_config = DataConfig(
            text_column=config_dict['data']['text_column'],
            id_column=config_dict['data']['id_column'],
            outcomes=outcomes,
            sample_size=config_dict['data'].get('sample_size'),
            sample_seed=config_dict['data'].get('sample_seed'),
            control_variables=control_vars
        )
        
        # Parse analysis configuration
        analysis_config = AnalysisConfig(**config_dict.get('analysis', {}))
        
        # Parse visualization configuration
        viz_config = VisualizationConfig(**config_dict.get('visualization', {}))
        
        # Parse directories
        output_dir = Path(config_dict.get('output_dir', 'output'))
        checkpoint_dir = Path(config_dict.get('checkpoint_dir', 'checkpoints'))
        
        return cls(
            name=config_dict['pipeline']['name'],
            embedding_model=config_dict['pipeline']['embedding_model'],
            data=data_config,
            analysis=analysis_config,
            visualization=viz_config,
            output_dir=output_dir,
            checkpoint_dir=checkpoint_dir
        )


def validate_config(config: PipelineConfig) -> None:
    """Validate the pipeline configuration."""
    # Check embedding model
    valid_models = ['nvidia/NV-Embed-v2', 'pre-computed', 'sentence-transformers/all-MiniLM-L6-v2']
    if config.embedding_model not in valid_models and not config.embedding_model.startswith('sentence-transformers/'):
        raise ValueError(f"Invalid embedding model: {config.embedding_model}")
    
    # Check outcome types
    valid_types = ['continuous', 'ordinal', 'categorical']
    for outcome in config.data.outcomes:
        if outcome.type not in valid_types:
            raise ValueError(f"Invalid outcome type '{outcome.type}' for {outcome.name}")
    
    # Check analysis parameters
    if config.analysis.pca_components < 1:
        raise ValueError("PCA components must be >= 1")
    if config.analysis.umap_dimensions not in [2, 3]:
        raise ValueError("UMAP dimensions must be 2 or 3")
    
    print(f"✓ Configuration '{config.name}' validated successfully")