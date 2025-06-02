#!/usr/bin/env python3
"""Data loading and preprocessing utilities."""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from .config import PipelineConfig, OutcomeConfig


class DataLoader:
    """Handles data loading and preprocessing."""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.data = None
        self.embeddings = None
        self.original_data = None  # Store original data before sampling
        self.sample_indices = None  # Store sampling indices for reproducibility
        self.sampling_info = None  # Store sampling metadata
        self.control_data = None  # Store control variables data
        
    def load_data(self, data_path: str, embeddings_path: Optional[str] = None) -> pd.DataFrame:
        """Load data from CSV file."""
        print(f"Loading data from {data_path}...")
        self.data = pd.read_csv(data_path)
        
        # Validate required columns
        required_cols = [self.config.data.text_column, self.config.data.id_column]
        required_cols.extend([o.name for o in self.config.data.outcomes])
        
        missing_cols = set(required_cols) - set(self.data.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Load embeddings if provided
        if embeddings_path:
            print(f"Loading pre-computed embeddings from {embeddings_path}...")
            self.embeddings = np.load(embeddings_path)
            if len(self.embeddings) != len(self.data):
                raise ValueError(f"Embeddings length ({len(self.embeddings)}) doesn't match data length ({len(self.data)})")
        
        # Clean data
        self._clean_data()
        
        # Store original data size
        original_size = len(self.data)
        print(f"✓ Loaded {original_size} records")
        
        # Apply sampling if needed
        # Only apply sampling if sample_size is explicitly set
        if self.config.data.sample_size is not None and len(self.data) > self.config.data.sample_size:
            self._apply_sampling()
        
        # Detect outcome modes
        self._detect_outcome_modes()
        
        # Load control variables if specified
        if self.config.data.control_variables:
            self._load_control_variables()
        
        return self.data
    
    def _clean_data(self) -> None:
        """Clean and preprocess the data."""
        # Remove rows with missing text
        text_col = self.config.data.text_column
        self.data = self.data[self.data[text_col].notna()]
        
        # Handle missing outcomes
        for outcome in self.config.data.outcomes:
            if outcome.type in ['continuous', 'ordinal']:
                # Fill numeric missing values with median
                median_val = self.data[outcome.name].median()
                self.data[outcome.name].fillna(median_val, inplace=True)
            elif outcome.type == 'categorical':
                # Fill categorical missing values with mode or 'Unknown'
                mode_val = self.data[outcome.name].mode()
                if len(mode_val) > 0:
                    self.data[outcome.name].fillna(mode_val[0], inplace=True)
                else:
                    self.data[outcome.name].fillna('Unknown', inplace=True)
    
    def _detect_outcome_modes(self) -> None:
        """Auto-detect appropriate mode for outcome variables."""
        if not self.config.analysis.outcome_mode_detection:
            return
            
        print("\nDetecting outcome modes...")
        for outcome in self.config.data.outcomes:
            if outcome.mode:  # Skip if manually set
                print(f"  {outcome.name}: '{outcome.mode}' (manually set)")
                continue
                
            values = self.data[outcome.name].values
            detected_mode = self.detect_outcome_mode(values, outcome)
            
            if detected_mode != 'continuous':
                outcome.mode = detected_mode
                outcome.mode_auto_detected = True
                print(f"  {outcome.name}: '{detected_mode}' (auto-detected)")
            else:
                outcome.mode = 'continuous'
                print(f"  {outcome.name}: 'continuous' (default)")
    
    def detect_outcome_mode(self, values: np.ndarray, outcome_config: OutcomeConfig) -> str:
        """Auto-detect appropriate mode for outcome variable."""
        # Calculate statistics
        zero_fraction = np.mean(values == 0)
        unique_values = np.unique(values)
        n_unique = len(unique_values)
        
        # Log detection info
        print(f"    Analyzing {outcome_config.name}:")
        print(f"      Zero fraction: {zero_fraction:.2%}")
        print(f"      Unique values: {n_unique}")
        
        # Detection heuristics
        # 1. High zero fraction
        if zero_fraction > self.config.analysis.outcome_mode_threshold:
            print(f"      → Zero-presence mode (>{self.config.analysis.outcome_mode_threshold:.0%} zeros)")
            return "zero_presence"
        
        # 2. Binary with zero
        if n_unique == 2 and 0 in unique_values:
            print(f"      → Zero-presence mode (binary with zero)")
            return "zero_presence"
        
        # 3. Low cardinality with zero
        if n_unique <= 10 and 0 in unique_values and zero_fraction > 0.2:
            print(f"      → Zero-presence mode (low cardinality with significant zeros)")
            return "zero_presence"
        
        # 4. Categorical type with zero
        if outcome_config.type == "categorical" and 0 in unique_values:
            print(f"      → Zero-presence mode (categorical with zero)")
            return "zero_presence"
        
        return "continuous"
    
    def _load_control_variables(self) -> None:
        """Load control variables from the data."""
        print("\nLoading control variables...")
        control_data = {}
        
        for control in self.config.data.control_variables:
            if control.name not in self.data.columns:
                raise ValueError(f"Control variable '{control.name}' not found in data")
            
            control_data[control.name] = self.data[control.name].values
            print(f"  ✓ {control.display_name}: {control.name}")
            
            # Basic statistics
            values = control_data[control.name]
            if np.issubdtype(values.dtype, np.number):
                print(f"    Range: [{np.min(values):.2f}, {np.max(values):.2f}]")
                print(f"    Mean: {np.mean(values):.2f} (±{np.std(values):.2f})")
        
        self.control_data = control_data
    
    def get_outcome_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Calculate statistics for all outcome variables."""
        stats = {}
        
        for outcome in self.config.data.outcomes:
            outcome_data = self.data[outcome.name]
            
            if outcome.type in ['continuous', 'ordinal']:
                stats[outcome.name] = {
                    'type': outcome.type,
                    'display_name': outcome.display_name,
                    'min': float(outcome_data.min()),
                    'max': float(outcome_data.max()),
                    'mean': float(outcome_data.mean()),
                    'median': float(outcome_data.median()),
                    'std': float(outcome_data.std()),
                    'percentiles': {
                        p: float(outcome_data.quantile(p/100))
                        for p in [10, 25, 50, 75, 90]
                    }
                }
            else:  # categorical
                value_counts = outcome_data.value_counts()
                stats[outcome.name] = {
                    'type': outcome.type,
                    'display_name': outcome.display_name,
                    'categories': value_counts.index.tolist(),
                    'counts': value_counts.values.tolist(),
                    'proportions': (value_counts / len(outcome_data)).to_dict()
                }
        
        return stats
    
    def calculate_thresholds(self, outcome: OutcomeConfig) -> Dict[str, float]:
        """Calculate actual threshold values for an outcome."""
        outcome_data = self.data[outcome.name]
        
        # Check if we're in zero-presence mode
        if getattr(outcome, 'mode', 'continuous') == 'zero_presence':
            # In zero-presence mode, thresholds are simple:
            # low = 0 (absent), high = smallest non-zero value
            non_zero_values = outcome_data[outcome_data != 0]
            if len(non_zero_values) > 0:
                min_non_zero = float(non_zero_values.min())
            else:
                min_non_zero = 1.0  # Default if all zeros
            
            return {
                'low': 0.0,  # Zero is "low" (absent)
                'high': min_non_zero,  # Any non-zero is "high" (present)
                'low_percentile': 0,  # Not used in zero-presence mode
                'high_percentile': 100,  # Not used in zero-presence mode
                'mode': 'zero_presence',
                'zero_count': int((outcome_data == 0).sum()),
                'non_zero_count': int((outcome_data != 0).sum()),
                'zero_fraction': float((outcome_data == 0).mean())
            }
        
        # Original continuous mode logic
        if 'low_percentile' in outcome.default_thresholds:
            # Percentile-based thresholds
            low_val = np.percentile(outcome_data, outcome.default_thresholds['low_percentile'])
            high_val = np.percentile(outcome_data, outcome.default_thresholds['high_percentile'])
        else:
            # Fixed value thresholds
            low_val = outcome.default_thresholds.get('low_value', outcome_data.min())
            high_val = outcome.default_thresholds.get('high_value', outcome_data.max())
        
        return {
            'low': float(low_val),
            'high': float(high_val),
            'low_percentile': outcome.default_thresholds.get('low_percentile', 10),
            'high_percentile': outcome.default_thresholds.get('high_percentile', 90),
            'mode': 'continuous'
        }
    
    def get_category_assignments(self) -> Tuple[np.ndarray, Dict[str, List[int]]]:
        """
        Assign data points to categories based on outcome thresholds.
        Returns category labels and indices for each category.
        """
        n_samples = len(self.data)
        
        # For 2 outcomes, use the original color scheme
        if len(self.config.data.outcomes) == 2:
            outcome1, outcome2 = self.config.data.outcomes[:2]
            thresholds1 = self.calculate_thresholds(outcome1)
            thresholds2 = self.calculate_thresholds(outcome2)
            
            data1 = self.data[outcome1.name].values
            data2 = self.data[outcome2.name].values
            
            # Check if either outcome is in zero-presence mode
            mode1 = getattr(outcome1, 'mode', 'continuous')
            mode2 = getattr(outcome2, 'mode', 'continuous')
            
            if mode1 == 'zero_presence' or mode2 == 'zero_presence':
                # Zero-presence mode categories
                categories = []
                category_indices = {
                    'both_absent': [],     # Both zero
                    'first_present': [],   # First non-zero, second zero
                    'second_present': [],  # First zero, second non-zero
                    'both_present': [],    # Both non-zero
                    'agreement': []        # Both have same presence status
                }
                
                for i in range(n_samples):
                    # In zero-presence mode, present = non-zero, absent = zero
                    if mode1 == 'zero_presence':
                        present1 = data1[i] != 0
                    else:
                        # Fallback to threshold-based for continuous
                        present1 = data1[i] > thresholds1['high']
                    
                    if mode2 == 'zero_presence':
                        present2 = data2[i] != 0
                    else:
                        # Fallback to threshold-based for continuous
                        present2 = data2[i] > thresholds2['high']
                    
                    if not present1 and not present2:
                        cat = 'both_absent'
                    elif present1 and not present2:
                        cat = 'first_present'
                    elif not present1 and present2:
                        cat = 'second_present'
                    else:  # both present
                        cat = 'both_present'
                    
                    categories.append(cat)
                    category_indices[cat].append(i)
                    
                    # Also track agreement
                    if present1 == present2:
                        category_indices['agreement'].append(i)
                
            else:
                # Original continuous mode logic
                categories = []
                category_indices = {
                    'both_high': [],
                    'first_high': [],
                    'second_high': [],
                    'both_low': [],
                    'middle': []
                }
                
                for i in range(n_samples):
                    high1 = data1[i] > thresholds1['high']
                    low1 = data1[i] < thresholds1['low']
                    high2 = data2[i] > thresholds2['high']
                    low2 = data2[i] < thresholds2['low']
                    
                    if high1 and high2:
                        cat = 'both_high'
                    elif high1 and low2:
                        cat = 'first_high'
                    elif low1 and high2:
                        cat = 'second_high'
                    elif low1 and low2:
                        cat = 'both_low'
                    else:
                        cat = 'middle'
                    
                    categories.append(cat)
                    category_indices[cat].append(i)
            
            return np.array(categories), category_indices
        
        else:
            # For 1 or 3+ outcomes, use a different scheme
            categories = ['middle'] * n_samples
            category_indices = {'middle': list(range(n_samples))}
            
            # You can implement more complex multi-outcome categorization here
            return np.array(categories), category_indices
    
    def _apply_sampling(self) -> None:
        """Apply stratified or random sampling to the data."""
        from sklearn.model_selection import train_test_split
        
        sample_size = self.config.data.sample_size
        sample_seed = self.config.data.sample_seed or 42
        
        print(f"\nApplying sampling: {sample_size} samples from {len(self.data)} total")
        print(f"  Sample seed: {sample_seed}")
        
        # Store original data
        self.original_data = self.data.copy()
        original_embeddings = self.embeddings.copy() if self.embeddings is not None else None
        
        # Try stratified sampling based on outcomes
        stratify_col = None
        for outcome in self.config.data.outcomes:
            if outcome.type == 'categorical':
                stratify_col = outcome.name
                break
            elif outcome.type in ['continuous', 'ordinal']:
                # Create bins for stratification
                stratify_col = f'{outcome.name}_bin'
                self.data[stratify_col] = pd.qcut(self.data[outcome.name], q=10, labels=False, duplicates='drop')
                break
        
        if stratify_col:
            print(f"  Using stratified sampling on: {stratify_col}")
            # Use train_test_split for stratified sampling
            _, sampled_data, _, sample_indices = train_test_split(
                self.data, 
                np.arange(len(self.data)),
                test_size=sample_size/len(self.data),
                stratify=self.data[stratify_col],
                random_state=sample_seed
            )
            self.sample_indices = sample_indices
            self.data = sampled_data.reset_index(drop=True)
            
            # Remove temporary bin column if created
            if stratify_col.endswith('_bin'):
                self.data.drop(columns=[stratify_col], inplace=True)
                
            sampling_method = 'stratified'
        else:
            print("  Using random sampling")
            # Random sampling
            np.random.seed(sample_seed)
            self.sample_indices = np.random.choice(len(self.data), size=sample_size, replace=False)
            self.data = self.data.iloc[self.sample_indices].reset_index(drop=True)
            sampling_method = 'random'
        
        # Sample embeddings if they exist
        if self.embeddings is not None:
            self.embeddings = self.embeddings[self.sample_indices]
        
        # Store sampling information
        self.sampling_info = {
            'method': sampling_method,
            'original_size': len(self.original_data),
            'sample_size': sample_size,
            'sample_seed': sample_seed,
            'sample_ratio': sample_size / len(self.original_data),
            'stratify_column': stratify_col if sampling_method == 'stratified' else None
        }
        
        print(f"  ✓ Sampled {len(self.data)} records ({self.sampling_info['sample_ratio']:.1%} of original)")
    
    def prepare_visualization_data(self, umap_coords: np.ndarray, 
                                 pca_features: np.ndarray,
                                 pc_percentiles: np.ndarray,
                                 contributions: Dict[str, np.ndarray],
                                 cluster_labels: np.ndarray,
                                 variance_explained: np.ndarray) -> List[Dict]:
        """Prepare data for visualization."""
        viz_data = []
        
        for i in range(len(self.data)):
            # Get top contributing PCs for this sample
            total_contributions = sum(np.abs(contributions[o.name][i]) 
                                    for o in self.config.data.outcomes)
            # Ensure indices are within bounds
            n_pcs = min(pca_features.shape[1], contributions[list(contributions.keys())[0]].shape[1])
            top_6_indices = np.argsort(total_contributions)[-6:][::-1]
            top_6_indices = top_6_indices[top_6_indices < n_pcs]  # Filter out invalid indices
            
            # Create PC info
            pc_info = []
            for pc_idx in top_6_indices:
                pc_dict = {
                    'pc': f'PC{pc_idx}',
                    'percentile': float(pc_percentiles[i, pc_idx]),
                    'variance_total': float(variance_explained[pc_idx] * 100)
                }
                
                # Add contributions for each outcome
                for outcome in self.config.data.outcomes:
                    pc_dict[f'contribution_{outcome.name}'] = float(contributions[outcome.name][i, pc_idx])
                    pc_dict[f'variance_{outcome.name}'] = float(
                        np.abs(contributions[outcome.name][i, pc_idx]) * variance_explained[pc_idx] * 100
                    )
                
                pc_info.append(pc_dict)
            
            # Create data point (scale by 100 to match Three.js scale)
            point_data = {
                'x': float(umap_coords[i, 0] * 100),
                'y': float(umap_coords[i, 1] * 100),
                'z': float(umap_coords[i, 2] * 100) if umap_coords.shape[1] > 2 else 0,
                'id': str(self.data.iloc[i][self.config.data.id_column]),
                'text': str(self.data.iloc[i][self.config.data.text_column]),
                'cluster_id': int(cluster_labels[i]),
                'pc_info': pc_info,
                'all_pc_values': pca_features[i].tolist(),
                'all_pc_percentiles': pc_percentiles[i].tolist()
            }
            
            # Add outcome values and contributions
            for outcome in self.config.data.outcomes:
                point_data[outcome.name] = float(self.data.iloc[i][outcome.name])
                point_data[f'all_pc_contributions_{outcome.name}'] = contributions[outcome.name][i].tolist()
            
            viz_data.append(point_data)
        
        return viz_data