#!/usr/bin/env python3
"""Main pipeline runner for text embedding analysis."""

import click
import pandas as pd
import numpy as np
import time
from pathlib import Path
import sys
import warnings
from typing import Dict, Any
warnings.filterwarnings('ignore')

from .config import PipelineConfig, validate_config
from .data_loader import DataLoader
from .embeddings import EmbeddingGenerator
from .dimensionality import DimensionalityReducer
from .clustering import TopicModeler
from .dml_analysis import DMLAnalyzer
from .visualization import VisualizationGenerator
from .data_exporter import DataExporter
from .auto_parameters import AutoParameterSystem, DatasetProfile


class TextEmbeddingPipeline:
    """Main pipeline orchestrator."""
    
    def __init__(self, config_path: str = None, config_dict: dict = None, num_gpus: int = 1):
        if config_path:
            self.config = PipelineConfig.from_yaml(config_path)
        elif config_dict:
            self.config = PipelineConfig.from_dict(config_dict)
        else:
            raise ValueError("Either config_path or config_dict must be provided")
        
        validate_config(self.config)
        self.num_gpus = num_gpus
        
        # Initialize auto-parameter system (will be configured after data load)
        self.auto_param_system = None
        
        # Initialize components
        self.data_loader = DataLoader(self.config)
        self.embedding_gen = EmbeddingGenerator(self.config, num_gpus=self.num_gpus)
        self.dim_reducer = DimensionalityReducer(self.config)
        self.topic_modeler = TopicModeler(self.config)
        self.dml_analyzer = DMLAnalyzer(self.config)
        self.viz_generator = VisualizationGenerator(self.config)
        self.data_exporter = DataExporter(self.config)
    
    @classmethod
    def from_state(cls, state: Dict[str, Any]) -> 'TextEmbeddingPipeline':
        """Create pipeline from saved state."""
        # Create pipeline with saved config
        pipeline = cls(config_dict=state['config'])
        
        # Restore models
        if 'models' in state:
            if state['models'].get('pca'):
                pipeline.dim_reducer.pca = state['models']['pca']
            if state['models'].get('scaler'):
                pipeline.dim_reducer.scaler = state['models']['scaler']
            if state['models'].get('umap'):
                pipeline.dim_reducer.umap_model = state['models']['umap']
            if state['models'].get('clustering'):
                pipeline.topic_modeler.clustering_model = state['models']['clustering']
        
        # Restore data
        if 'data' in state:
            if state['data'].get('original') is not None:
                pipeline.data_loader.data = state['data']['original']
            if state['data'].get('embeddings') is not None:
                pipeline.embedding_gen.embeddings = state['data']['embeddings']
            if state['data'].get('original_full') is not None:
                pipeline.data_loader.original_data = state['data']['original_full']
            if state['data'].get('sample_indices') is not None:
                pipeline.data_loader.sample_indices = state['data']['sample_indices']
        
        # Restore sampling info
        if 'sampling_info' in state:
            pipeline.data_loader.sampling_info = state['sampling_info']
        
        # Restore DML residuals and predictions
        if 'dml_residuals' in state:
            pipeline.dml_analyzer.residuals = state['dml_residuals']
        if 'dml_predictions' in state:
            pipeline.dml_analyzer.predictions = state['dml_predictions']
        if 'cli_command' in state:
            pipeline._cli_command = state['cli_command']
        
        return pipeline
        
    def run(self, data_path: str, embeddings_path: str = None, 
            output_name: str = None) -> str:
        """Run the complete pipeline."""
        start_time = time.time()
        print(f"\n{'='*60}")
        print(f"Running Text Embedding Pipeline: {self.config.name}")
        print(f"{'='*60}\n")
        
        # Step 1: Load data
        data = self.data_loader.load_data(data_path, embeddings_path)
        
        # Step 2: Generate or load embeddings
        if embeddings_path is None:
            embeddings = self.embedding_gen.generate_embeddings(
                data[self.config.data.text_column].tolist()
            )
            self.embedding_gen.validate_embeddings(
                embeddings, 
                data[self.config.data.text_column].tolist()
            )
        else:
            embeddings = self.data_loader.embeddings
        
        # Step 3: PCA
        pca_results = self.dim_reducer.fit_pca(embeddings)
        
        # Step 4: UMAP
        umap_results = self.dim_reducer.fit_umap(pca_results['features'])
        
        # Step 5: Clustering (use normalized embeddings for consistent coordinates)
        cluster_labels = self.topic_modeler.fit_clusters(umap_results['embeddings_normalized'])
        
        # Extract topics
        topic_keywords = self.topic_modeler.extract_topics(
            data[self.config.data.text_column].tolist(),
            cluster_labels
        )
        
        # Calculate cluster statistics (use normalized embeddings for centroids)
        cluster_stats = self.topic_modeler.calculate_cluster_statistics(
            data, cluster_labels, umap_results['embeddings_normalized']
        )
        
        # Prepare topic visualization data
        topic_viz_data = self.topic_modeler.prepare_topic_visualization(
            topic_keywords, cluster_stats
        )
        
        # Step 6: DML Analysis
        # Prepare outcome data
        outcome_data = {
            outcome.name: data[outcome.name].values 
            for outcome in self.config.data.outcomes
        }
        
        # Select top PCs if not specified
        if self.config.analysis.dml_top_pcs is None:
            top_pcs, pc_selection_info = self.dim_reducer.select_top_pcs_for_dml(
                pca_results['features'],
                outcome_data,
                n_pcs=6
            )
        else:
            top_pcs = self.config.analysis.dml_top_pcs
            pc_selection_info = None
        
        # Get control data if available
        control_data = getattr(self.data_loader, 'control_data', None)
        
        # Run DML on embeddings, all PCs
        print("\nRunning DML on embeddings...")
        dml_results_embeddings = self.dml_analyzer.run_dml_analysis(
            embeddings,
            outcome_data,
            feature_names=[f'Emb{i}' for i in range(embeddings.shape[1])],
            model_suffix='embeddings',
            control_data=control_data
        )
        
        print("\nRunning DML on all 200 PCs...")
        dml_results_all = self.dml_analyzer.run_dml_analysis(
            pca_results['features'],
            outcome_data,
            model_suffix='200pcs',
            control_data=control_data
        )
        
        # Run DML for each PC selection method
        dml_results_by_method = {}
        
        # For backward compatibility, store primary method results as dml_results_top6
        print(f"\nRunning DML on top 6 PCs (primary: {self.config.analysis.dml_primary_pc_method})...")
        dml_results_top6 = self.dml_analyzer.run_dml_analysis(
            pca_results['features'][:, top_pcs],
            outcome_data,
            feature_names=[f'PC{i}' for i in top_pcs],
            model_suffix=f'top6pcs_{self.config.analysis.dml_primary_pc_method}',
            control_data=control_data
        )
        dml_results_by_method[self.config.analysis.dml_primary_pc_method] = dml_results_top6
        
        # Run DML for other PC selection methods if they're in the selection info
        if pc_selection_info:
            for method in self.config.analysis.dml_pc_selection_methods:
                if method == self.config.analysis.dml_primary_pc_method:
                    continue  # Already processed
                    
                method_indices_key = f'{method}_indices'
                if method_indices_key in pc_selection_info:
                    method_indices = pc_selection_info[method_indices_key]
                    print(f"\nRunning DML on top 6 PCs ({method})...")
                    dml_results_method = self.dml_analyzer.run_dml_analysis(
                        pca_results['features'][:, method_indices],
                        outcome_data,
                        feature_names=[f'PC{i}' for i in method_indices],
                        model_suffix=f'top6pcs_{method}',
                        control_data=control_data
                    )
                    dml_results_by_method[method] = dml_results_method
        
        # Step 7: Prepare visualization data
        print("\nPreparing visualization data...")
        
        # Calculate thresholds
        print("  Calculating thresholds for outcomes...")
        thresholds = {
            outcome.name: self.data_loader.calculate_thresholds(outcome)
            for outcome in self.config.data.outcomes
        }
        
        # Get outcome statistics
        print("  Getting outcome statistics...")
        outcome_stats = self.data_loader.get_outcome_statistics()
        
        # Calculate categories
        print("  Calculating category assignments...")
        categories, category_indices = self.data_loader.get_category_assignments()
        
        # Calculate PC global effects first (needed for detailed stats)
        print(f"  Calculating PC global effects for {pca_results['features'].shape[1]} components...")
        pc_global_effects = self._calculate_pc_global_effects(
            pca_results['features'],
            outcome_data,
            thresholds
        )
        
        # Calculate detailed PC statistics (uses global effects cache)
        print("  Calculating detailed PC statistics...")
        pc_stats_data = self._calculate_pc_detailed_stats(
            pca_results['features'],
            outcome_data,
            dml_results_all,
            thresholds,
            pc_global_effects,  # Pass the effects directly
            cluster_labels,  # Pass cluster labels for topic analysis
            topic_keywords  # Pass topic keywords for display
        )
        
        # Prepare visualization data points
        print(f"  Preparing {len(data)} visualization data points...")
        viz_data = self.data_loader.prepare_visualization_data(
            umap_results['embeddings_normalized'],
            pca_results['features'],
            pca_results['percentiles'],
            {o: dml_results_all[f'contributions_{o}'] for o in outcome_data.keys()},
            cluster_labels,
            pca_results['explained_variance']
        )
        
        # Calculate topic statistics for extreme groups
        print("  Calculating topic statistics for extreme groups...")
        topic_stats_data = self.topic_modeler.calculate_extreme_group_statistics(
            data, cluster_labels, thresholds
        )
        
        # Add topic keywords to stats
        print("  Adding topic keywords to statistics...")
        for stats in topic_stats_data:
            stats['keywords'] = topic_keywords.get(stats['topic_id'], f"Topic {stats['topic_id']}")
        
        # Combine all results
        all_results = {
            'viz_data': viz_data,
            'topic_viz_data': topic_viz_data,
            'topic_stats_data': topic_stats_data,
            'thresholds': thresholds,
            'outcome_stats': outcome_stats,
            'pc_global_effects': pc_global_effects,
            'pc_stats_data': pc_stats_data,
            'variance_explained': pca_results['explained_variance'].tolist(),
            'dml_results': dml_results_all,
            'dml_results_embeddings': dml_results_embeddings,
            'dml_results_top6': dml_results_top6,
            'dml_results_by_method': dml_results_by_method,
            'top_pcs': top_pcs,
            'pc_selection_info': pc_selection_info,
            'outcome_modes': {
                outcome.name: getattr(outcome, 'mode', 'continuous')
                for outcome in self.config.data.outcomes
            },
            'has_control_variables': control_data is not None,
            'control_variables': self.config.data.control_variables if self.config.data.control_variables else []
        }
        
        # Store results for potential export
        self._last_results = all_results
        self._pca_results = pca_results
        self._umap_results = umap_results
        self._cluster_labels = cluster_labels
        self._topic_keywords = topic_keywords
        
        # Step 8: Generate HTML
        if output_name is None:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_name = f"analysis_{timestamp}.html"
        
        output_path = self.config.output_dir / output_name
        print(f"\nGenerating HTML output: {output_path}")
        html_path = self.viz_generator.generate_html(all_results, output_path)
        
        # Summary
        elapsed_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"✅ Pipeline completed successfully!")
        print(f"Total time: {elapsed_time/60:.1f} minutes")
        print(f"Output: {html_path}")
        print(f"{'='*60}\n")
        
        return html_path
    
    def _calculate_pc_global_effects(self, pca_features: np.ndarray,
                                   outcome_data: Dict[str, np.ndarray],
                                   thresholds: Dict) -> Dict:
        """Calculate global PC effects for visualization.
        
        For each PC, calculates probabilities:
        - prob_high_if_high: P(outcome > high_threshold | PC is high)
        - prob_high_if_low: P(outcome > high_threshold | PC is low)
        - prob_low_if_high: P(outcome < low_threshold | PC is high)
        - prob_low_if_low: P(outcome < low_threshold | PC is low)
        
        When sufficient variation exists, uses logistic regression to estimate
        conditional probabilities. When no variation exists (e.g., all outcomes
        are 0), reports the marginal probability (base rate) instead.
        
        All values are true probabilities between 0 and 1.
        """
        from sklearn.preprocessing import StandardScaler
        
        pc_effects = {}
        self._pc_global_effects_cache = {}  # Store for use in detailed stats
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(pca_features)
        
        # For each PC
        for pc_idx in range(pca_features.shape[1]):
            if pc_idx % 20 == 0:
                print(f"    Processing PC {pc_idx}/{pca_features.shape[1]}...", flush=True)
            effects = {}
            
            # Create test data: top 10% vs bottom 10% of PC values
            test_data = np.zeros((2, pca_features.shape[1]))
            test_data[0, pc_idx] = np.percentile(pca_features[:, pc_idx], 90)
            test_data[1, pc_idx] = np.percentile(pca_features[:, pc_idx], 10)
            test_scaled = scaler.transform(test_data)
            
            # For each outcome
            for outcome_name, outcome_values in outcome_data.items():
                thresh = thresholds[outcome_name]
                
                # Check if this outcome is in zero-presence mode
                outcome_config = next((o for o in self.config.data.outcomes if o.name == outcome_name), None)
                is_zero_presence = outcome_config and getattr(outcome_config, 'mode', 'continuous') == 'zero_presence'
                
                if is_zero_presence:
                    # Zero-presence mode: calculate empirical presence rates
                    pc_values = pca_features[:, pc_idx]
                    p90 = np.percentile(pc_values, 90)
                    p10 = np.percentile(pc_values, 10)
                    
                    high_pc_mask = pc_values >= p90  # Top 10%
                    low_pc_mask = pc_values <= p10   # Bottom 10%
                    
                    # Binary presence indicator
                    is_present = (outcome_values != 0).astype(int)
                    
                    # Calculate presence rates
                    high_pc_present = np.mean(is_present[high_pc_mask]) if np.sum(high_pc_mask) > 0 else 0
                    low_pc_present = np.mean(is_present[low_pc_mask]) if np.sum(low_pc_mask) > 0 else 0
                    
                    # Calculate average magnitude when present
                    present_mask = outcome_values != 0
                    if np.any(present_mask & high_pc_mask):
                        high_pc_magnitude = np.mean(outcome_values[present_mask & high_pc_mask])
                    else:
                        high_pc_magnitude = 0
                        
                    if np.any(present_mask & low_pc_mask):
                        low_pc_magnitude = np.mean(outcome_values[present_mask & low_pc_mask])
                    else:
                        low_pc_magnitude = 0
                    
                    # Store results in format compatible with visualization (convert to percentages)
                    effects[f'{outcome_name}_high_if_high'] = float(high_pc_present * 100)
                    effects[f'{outcome_name}_high_if_low'] = float(low_pc_present * 100)
                    effects[f'{outcome_name}_high_diff'] = float((high_pc_present - low_pc_present) * 100)
                    # Additional zero-presence specific stats
                    effects[f'{outcome_name}_magnitude_high'] = float(high_pc_magnitude)
                    effects[f'{outcome_name}_magnitude_low'] = float(low_pc_magnitude)
                    # For compatibility, set low probabilities to 1 - presence (as percentages)
                    effects[f'{outcome_name}_low_if_high'] = float((1 - high_pc_present) * 100)
                    effects[f'{outcome_name}_low_if_low'] = float((1 - low_pc_present) * 100)
                    effects[f'{outcome_name}_low_diff'] = float(((1 - high_pc_present) - (1 - low_pc_present)) * 100)
                    
                else:
                    # Continuous mode: Use fixed percentiles and simple empirical rates
                    pc_values = pca_features[:, pc_idx]
                    p90 = np.percentile(pc_values, 90)
                    p10 = np.percentile(pc_values, 10)
                    
                    high_pc_mask = pc_values >= p90  # Top 10% of PC values
                    low_pc_mask = pc_values <= p10   # Bottom 10% of PC values
                    
                    # Use fixed percentiles for outcome thresholds (independent of visual thresholds)
                    # Using 90th percentile for "high" and 10th percentile for "low"
                    outcome_p90 = np.percentile(outcome_values, 90)
                    outcome_p10 = np.percentile(outcome_values, 10)
                    
                    # Define high/low outcomes based on fixed percentiles
                    high_outcome = outcome_values >= outcome_p90  # Top 10% of outcome
                    low_outcome = outcome_values <= outcome_p10   # Bottom 10% of outcome
                    
                    # Calculate simple empirical rates
                    # High outcome rates
                    high_outcome_if_high_pc = np.mean(high_outcome[high_pc_mask]) if np.sum(high_pc_mask) > 0 else 0
                    high_outcome_if_low_pc = np.mean(high_outcome[low_pc_mask]) if np.sum(low_pc_mask) > 0 else 0
                    
                    # Low outcome rates  
                    low_outcome_if_high_pc = np.mean(low_outcome[high_pc_mask]) if np.sum(high_pc_mask) > 0 else 0
                    low_outcome_if_low_pc = np.mean(low_outcome[low_pc_mask]) if np.sum(low_pc_mask) > 0 else 0
                    
                    # Store as percentages
                    effects[f'{outcome_name}_high_if_high'] = float(high_outcome_if_high_pc * 100)
                    effects[f'{outcome_name}_high_if_low'] = float(high_outcome_if_low_pc * 100)
                    effects[f'{outcome_name}_high_diff'] = float((high_outcome_if_high_pc - high_outcome_if_low_pc) * 100)
                    
                    effects[f'{outcome_name}_low_if_high'] = float(low_outcome_if_high_pc * 100)
                    effects[f'{outcome_name}_low_if_low'] = float(low_outcome_if_low_pc * 100)
                    effects[f'{outcome_name}_low_diff'] = float((low_outcome_if_high_pc - low_outcome_if_low_pc) * 100)
            
            pc_effects[pc_idx] = effects
            self._pc_global_effects_cache[pc_idx] = effects
        
        return pc_effects
    
    def _calculate_pc_detailed_stats(self, pca_features: np.ndarray,
                                    outcome_data: Dict[str, np.ndarray],
                                    dml_results: Dict,
                                    thresholds: Dict,
                                    pc_global_effects: Dict,
                                    cluster_labels: np.ndarray,
                                    topic_keywords: Dict[int, str]) -> Dict:
        """Calculate detailed PC statistics for the advanced stats box."""
        from sklearn.model_selection import KFold
        from scipy import stats
        import xgboost as xgb
        import shap
        
        pc_stats = {}
        n_pcs = pca_features.shape[1]
        
        # For efficiency, we'll calculate detailed stats for all PCs but with limited SHAP analysis
        print("    Calculating importance rankings and correlations...")
        
        # Get outcome names
        outcome_names = list(outcome_data.keys())
        if len(outcome_names) >= 2:
            y_name = outcome_names[0]
            x_name = outcome_names[1]
        else:
            print("    Warning: Need at least 2 outcomes for PC statistics")
            return {}
        
        # Train XGBoost models for importance rankings
        model_y = xgb.XGBRegressor(
            n_estimators=self.config.analysis.xgb_n_estimators,
            max_depth=self.config.analysis.xgb_max_depth,
            random_state=42,
            tree_method='hist',
            device='cuda'
        )
        model_x = xgb.XGBRegressor(
            n_estimators=self.config.analysis.xgb_n_estimators,
            max_depth=self.config.analysis.xgb_max_depth,
            random_state=42,
            tree_method='hist',
            device='cuda'
        )
        
        # Fit models
        model_y.fit(pca_features, outcome_data[y_name])
        model_x.fit(pca_features, outcome_data[x_name])
        
        # Get feature importances
        importance_y = model_y.feature_importances_
        importance_x = model_x.feature_importances_
        
        # Calculate rankings with cross-validation for better avg/median estimates
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        ranks_y_all = []
        ranks_x_all = []
        
        for train_idx, _ in kf.split(pca_features):
            # Train models on fold
            model_y_fold = xgb.XGBRegressor(
                n_estimators=50,
                max_depth=4,
                random_state=42,
                tree_method='hist',
                device='cuda'
            )
            model_x_fold = xgb.XGBRegressor(
                n_estimators=50,
                max_depth=4,
                random_state=42,
                tree_method='hist',
                device='cuda'
            )
            
            model_y_fold.fit(pca_features[train_idx], outcome_data[y_name][train_idx])
            model_x_fold.fit(pca_features[train_idx], outcome_data[x_name][train_idx])
            
            # Get rankings for this fold
            imp_y_fold = model_y_fold.feature_importances_
            imp_x_fold = model_x_fold.feature_importances_
            rank_y_fold = np.argsort(np.argsort(-imp_y_fold)) + 1
            rank_x_fold = np.argsort(np.argsort(-imp_x_fold)) + 1
            
            ranks_y_all.append(rank_y_fold)
            ranks_x_all.append(rank_x_fold)
        
        # Convert to numpy arrays
        ranks_y_all = np.array(ranks_y_all)
        ranks_x_all = np.array(ranks_x_all)
        
        # Use original single model rankings as well
        rank_y = np.argsort(np.argsort(-importance_y)) + 1
        rank_x = np.argsort(np.argsort(-importance_x)) + 1
        
        # Calculate SHAP values for a subset of data (for efficiency)
        print("    Calculating SHAP values using XGBoost native method...")
        sample_size = min(1000, len(pca_features))
        sample_idx = np.random.choice(len(pca_features), sample_size, replace=False)
        
        # Use XGBoost's native SHAP calculation which works with both GPU and CPU models
        # This avoids the serialization issues with SHAP library and GPU models
        pca_sample = pca_features[sample_idx]
        
        # Convert to DMatrix for XGBoost native prediction
        dmatrix_sample = xgb.DMatrix(pca_sample)
        
        # Get SHAP values using native XGBoost method
        # pred_contribs=True returns SHAP values with base value as last column
        shap_native_y = model_y.get_booster().predict(dmatrix_sample, pred_contribs=True)
        shap_native_x = model_x.get_booster().predict(dmatrix_sample, pred_contribs=True)
        
        # Extract SHAP values (all columns except last which is base value)
        shap_values_y = shap_native_y[:, :-1]
        shap_values_x = shap_native_x[:, :-1]
        
        # Extract base values (expected values)
        base_value_y = shap_native_y[0, -1]  # Same for all samples
        base_value_x = shap_native_x[0, -1]  # Same for all samples
        
        print(f"    SHAP values calculated: shape {shap_values_y.shape}")
        print(f"    Base values: Y={base_value_y:.4f}, X={base_value_x:.4f}")
        
        # Calculate statistics for each PC
        for pc_idx in range(n_pcs):
            if pc_idx % 50 == 0:
                print(f"    Processing PC {pc_idx}/{n_pcs}...")
            
            # Calculate average and median ranks from cross-validation
            y_ranks_for_pc = ranks_y_all[:, pc_idx]
            x_ranks_for_pc = ranks_x_all[:, pc_idx]
            
            pc_data = {
                'rankings': {
                    'y_avg_rank': int(np.round(np.mean(y_ranks_for_pc))),
                    'y_median_rank': int(np.median(y_ranks_for_pc)),
                    'x_avg_rank': int(np.round(np.mean(x_ranks_for_pc))),
                    'x_median_rank': int(np.median(x_ranks_for_pc))
                },
                'shap_stats': {
                    'y': {
                        'min': float(np.min(shap_values_y[:, pc_idx])),
                        'max': float(np.max(shap_values_y[:, pc_idx])),
                        'range': float(np.max(shap_values_y[:, pc_idx]) - np.min(shap_values_y[:, pc_idx])),
                        'std': float(np.std(shap_values_y[:, pc_idx]))
                    },
                    'x': {
                        'min': float(np.min(shap_values_x[:, pc_idx])),
                        'max': float(np.max(shap_values_x[:, pc_idx])),
                        'range': float(np.max(shap_values_x[:, pc_idx]) - np.min(shap_values_x[:, pc_idx])),
                        'std': float(np.std(shap_values_x[:, pc_idx]))
                    }
                },
                'correlations': {
                    'y': float(np.corrcoef(pca_features[:, pc_idx], outcome_data[y_name])[0, 1]),
                    'x': float(np.corrcoef(pca_features[:, pc_idx], outcome_data[x_name])[0, 1])
                }
            }
            
            # Add extreme analysis from pc_global_effects if available
            if pc_idx in pc_global_effects:
                effects = pc_global_effects[pc_idx]
                
                # Check outcome modes
                y_outcome = next((o for o in self.config.data.outcomes if o.name == y_name), None)
                x_outcome = next((o for o in self.config.data.outcomes if o.name == x_name), None)
                y_mode = getattr(y_outcome, 'mode', 'continuous') if y_outcome else 'continuous'
                x_mode = getattr(x_outcome, 'mode', 'continuous') if x_outcome else 'continuous'
                
                if y_mode == 'zero_presence' or x_mode == 'zero_presence':
                    # Zero-presence mode: show presence rates and magnitudes
                    pc_data['extreme_analysis'] = {}
                    
                    if y_mode == 'zero_presence':
                        pc_data['extreme_analysis']['present_y'] = {
                            'if_high_pc': effects.get(f'{y_name}_high_if_high', 0),
                            'if_low_pc': effects.get(f'{y_name}_high_if_low', 0),
                            'magnitude_high': effects.get(f'{y_name}_magnitude_high', 0),
                            'magnitude_low': effects.get(f'{y_name}_magnitude_low', 0)
                        }
                        pc_data['extreme_analysis']['absent_y'] = {
                            'if_high_pc': effects.get(f'{y_name}_low_if_high', 0),
                            'if_low_pc': effects.get(f'{y_name}_low_if_low', 0)
                        }
                    else:
                        # Continuous mode for Y
                        pc_data['extreme_analysis']['high_y'] = {
                            'if_high_pc': effects.get(f'{y_name}_high_if_high', 0),
                            'if_low_pc': effects.get(f'{y_name}_high_if_low', 0)
                        }
                        pc_data['extreme_analysis']['low_y'] = {
                            'if_high_pc': effects.get(f'{y_name}_low_if_high', 0),
                            'if_low_pc': effects.get(f'{y_name}_low_if_low', 0)
                        }
                    
                    if x_mode == 'zero_presence':
                        pc_data['extreme_analysis']['present_x'] = {
                            'if_high_pc': effects.get(f'{x_name}_high_if_high', 0),
                            'if_low_pc': effects.get(f'{x_name}_high_if_low', 0),
                            'magnitude_high': effects.get(f'{x_name}_magnitude_high', 0),
                            'magnitude_low': effects.get(f'{x_name}_magnitude_low', 0)
                        }
                        pc_data['extreme_analysis']['absent_x'] = {
                            'if_high_pc': effects.get(f'{x_name}_low_if_high', 0),
                            'if_low_pc': effects.get(f'{x_name}_low_if_low', 0)
                        }
                    else:
                        # Continuous mode for X
                        pc_data['extreme_analysis']['high_x'] = {
                            'if_high_pc': effects.get(f'{x_name}_high_if_high', 0),
                            'if_low_pc': effects.get(f'{x_name}_high_if_low', 0)
                        }
                        pc_data['extreme_analysis']['low_x'] = {
                            'if_high_pc': effects.get(f'{x_name}_low_if_high', 0),
                            'if_low_pc': effects.get(f'{x_name}_low_if_low', 0)
                        }
                    
                    # Add mode info
                    pc_data['extreme_analysis']['y_mode'] = y_mode
                    pc_data['extreme_analysis']['x_mode'] = x_mode
                    
                else:
                    # Original continuous mode for both
                    pc_data['extreme_analysis'] = {
                        'high_y': {
                            'if_high_pc': effects.get(f'{y_name}_high_if_high', 0),
                            'if_low_pc': effects.get(f'{y_name}_high_if_low', 0)
                        },
                        'low_y': {
                            'if_high_pc': effects.get(f'{y_name}_low_if_high', 0),
                            'if_low_pc': effects.get(f'{y_name}_low_if_low', 0)
                        },
                        'high_x': {
                            'if_high_pc': effects.get(f'{x_name}_high_if_high', 0),
                            'if_low_pc': effects.get(f'{x_name}_high_if_low', 0)
                        },
                        'low_x': {
                            'if_high_pc': effects.get(f'{x_name}_low_if_high', 0),
                            'if_low_pc': effects.get(f'{x_name}_low_if_low', 0)
                        },
                        'y_mode': 'continuous',
                        'x_mode': 'continuous'
                    }
            
            pc_stats[pc_idx] = pc_data
        
        # Calculate topic-PC associations for all PCs
        print("    Calculating topic-PC associations...")
        
        # For each PC, calculate average percentile by topic
        for pc_idx in range(n_pcs):
            topic_associations = []
            
            # Get PC values and calculate percentiles
            pc_values = pca_features[:, pc_idx]
            pc_percentiles = stats.rankdata(pc_values, 'average') / len(pc_values) * 100
            
            # Calculate average percentile for each topic
            unique_topics = np.unique(cluster_labels)
            for topic_id in unique_topics:
                if topic_id == -1:  # Skip noise cluster
                    continue
                    
                # Get indices for this topic
                topic_mask = cluster_labels == topic_id
                topic_size = np.sum(topic_mask)
                
                if topic_size > 0:
                    # Calculate average percentile for this topic
                    avg_percentile = np.mean(pc_percentiles[topic_mask])
                    
                    # Calculate t-test: topic members vs non-members
                    # Use raw PC values for t-test (not percentiles)
                    topic_pc_values = pc_values[topic_mask]
                    non_topic_pc_values = pc_values[~topic_mask]
                    
                    # Perform independent samples t-test
                    t_stat, p_value = stats.ttest_ind(topic_pc_values, non_topic_pc_values, equal_var=False)
                    
                    # Get topic keywords
                    keywords = topic_keywords.get(int(topic_id), f"Topic {topic_id}")
                    
                    topic_associations.append({
                        'topic_id': int(topic_id),
                        'keywords': keywords,
                        'size': int(topic_size),
                        'avg_percentile': float(avg_percentile),
                        'std_percentile': float(np.std(pc_percentiles[topic_mask])),
                        't_statistic': float(t_stat),
                        'p_value': float(p_value),
                        'topic_mean': float(np.mean(topic_pc_values)),
                        'non_topic_mean': float(np.mean(non_topic_pc_values))
                    })
            
            # Sort by average percentile (descending)
            topic_associations.sort(key=lambda x: x['avg_percentile'], reverse=True)
            
            # Add to PC stats
            pc_stats[pc_idx]['topic_associations'] = topic_associations
        
        return pc_stats
    
    def _prepare_results_for_export(self) -> Dict[str, Any]:
        """Prepare all results for export."""
        if hasattr(self, '_last_results'):
            return self._last_results
        else:
            # If called before run(), return empty dict
            return {}


@click.command()
@click.option('--config', '-c', help='Path to configuration YAML file (optional - will auto-detect if not provided)')
@click.option('--data', '-d', required=True, help='Path to input data CSV file')
@click.option('--embeddings', '-e', help='Path to pre-computed embeddings (optional)')
@click.option('--output', '-o', help='Output HTML filename (optional)')
# Sampling options
@click.option('--sample-size', type=int, default=None,
              help='Sample dataset to this size (e.g., --sample-size 10000). Without flag, no sampling is performed.')
@click.option('--sample-seed', type=int, help='Random seed for sampling (for reproducibility)')
# Model options
@click.option('--embedding-model', help='Override embedding model from config (e.g., nvidia/NV-Embed-v2)')
@click.option('--num-gpus', type=int, default=1, help='Number of GPUs to use for embedding generation (default: 1)')
@click.option('--batch-size', type=int, help='Override batch size from config')
# Export options
@click.option('--export-csv/--no-export-csv', default=True, help='Export all processed data to CSV files (default: enabled)')
@click.option('--export-dir', help='Directory for CSV export (default: timestamped folder)')
@click.option('--exclude-text', is_flag=True, help='Exclude raw text from exports (privacy)')
@click.option('--anonymize-ids', is_flag=True, help='Replace IDs with anonymous integers')
@click.option('--export-state', help='Export complete pipeline state to pickle file (default: auto-generated name)')
# Import options
@click.option('--import-state', help='Import pipeline state from pickle file')
@click.option('--skip-validation', is_flag=True, help='Skip checksum validation when importing')
# PC selection options
@click.option('--pc-selection-methods', 
              help='Comma-separated list of PC selection methods for DML (e.g., "xgboost,lasso" or "all"). Default: all methods')
@click.option('--primary-pc-method', 
              help='Primary PC selection method for downstream tasks. Default: xgboost')
# Clustering options
@click.option('--min-cluster-size', type=int, help='HDBSCAN minimum cluster size')
@click.option('--min-samples', type=int, help='HDBSCAN minimum samples')
@click.option('--umap-neighbors', type=int, help='UMAP n_neighbors parameter')
@click.option('--umap-min-dist', type=float, help='UMAP min_dist parameter')
@click.option('--auto-cluster', type=click.Choice(['few', 'medium', 'many', 'descriptions']), 
              help='Automatically set clustering parameters based on dataset size')
# Auto-mode options
@click.option('--auto/--no-auto', default=True, help='Enable/disable automatic parameter selection (default: enabled)')
@click.option('--super-auto', is_flag=True, help='Enable super-auto mode with full ML hyperparameter optimization')
@click.option('--force-auto', is_flag=True, help='Force auto-selection, ignoring config file values')
@click.option('--preview-params', is_flag=True, help='Preview auto-selected parameters without running')
# Additional parameter overrides
@click.option('--dml-folds', type=int, help='Number of folds for DML cross-validation')
@click.option('--xgb-estimators', type=int, help='XGBoost n_estimators')
@click.option('--xgb-depth', type=int, help='XGBoost max_depth')
# Regularization parameters
@click.option('--lasso-alphas', help='Comma-separated list of Lasso alpha values (e.g., "0.001,0.01,0.1")')
@click.option('--ridge-alphas', help='Comma-separated list of Ridge alpha values (e.g., "0.01,0.1,1.0")')
@click.option('--reg-cv-folds', type=int, help='CV folds for regularization parameter selection')
# Outcome mode and control variables
@click.option('--outcome-mode', type=click.Choice(['auto', 'continuous', 'zero_presence']),
              default='auto', help='Outcome visualization mode (default: auto-detect)')
@click.option('--control-vars', help='Comma-separated list of control variables for DML (e.g., "num_raters,text_length")')
@click.option('--disable-mode-detection', is_flag=True, 
              help='Disable automatic outcome mode detection')
@click.option('--generate-config', help='Generate a config file from your data and save to specified path')
# Column specification options
@click.option('--text-column', help='Name of the text column in your CSV')
@click.option('--id-column', help='Name of the ID column in your CSV')
@click.option('--outcomes', help='Comma-separated Y and X variables (e.g., "anger_score,happiness_score")')
@click.option('--y-var', help='Explicitly set the Y (dependent) variable')
@click.option('--x-var', help='Explicitly set the X (independent) variable')
@click.option('--sampling-method', type=click.Choice(['random', 'stratified']), 
              default='stratified', help='Sampling method (default: stratified)')
@click.option('--stratify-by', help='Column to use for stratified sampling')
def main(config, data, embeddings, output, sample_size, sample_seed, embedding_model, num_gpus, batch_size, export_csv, export_dir, 
         exclude_text, anonymize_ids, export_state, import_state, skip_validation, pc_selection_methods, primary_pc_method,
         min_cluster_size, min_samples, umap_neighbors, umap_min_dist, auto_cluster, auto, super_auto, force_auto, preview_params, 
         dml_folds, xgb_estimators, xgb_depth, lasso_alphas, ridge_alphas, reg_cv_folds, outcome_mode, control_vars, disable_mode_detection,
         generate_config, text_column, id_column, outcomes, y_var, x_var, sampling_method, stratify_by):
    """Run the text embedding analysis pipeline.
    
    Examples:
        # Normal run with auto-mode (default)
        python run_pipeline.py -c config.yaml -d data.csv
        
        # Preview auto-selected parameters without running
        python run_pipeline.py -c config.yaml -d data.csv --preview-params
        
        # Run with auto-clustering for many topics
        python run_pipeline.py -c config.yaml -d data.csv --auto-cluster many
        
        # Enable super-auto mode for full ML hyperparameter optimization
        python run_pipeline.py -c config.yaml -d data.csv --super-auto
        
        # Super-auto with specific clustering target
        python run_pipeline.py -c config.yaml -d data.csv --super-auto --auto-cluster few
        
        # Force auto-selection, ignoring config file values
        python run_pipeline.py -c config.yaml -d data.csv --force-auto
        
        # Super-auto with force (ignores ALL config values)
        python run_pipeline.py -c config.yaml -d data.csv --super-auto --force-auto --auto-cluster many
        
        # Disable auto-mode and use config/defaults
        python run_pipeline.py -c config.yaml -d data.csv --no-auto
        
        # Override specific parameters while keeping auto-mode
        python run_pipeline.py -c config.yaml -d data.csv --xgb-depth 4 --dml-folds 10
        
        # Specify custom regularization parameters
        python run_pipeline.py -c config.yaml -d data.csv --lasso-alphas "0.001,0.01,0.1" --ridge-alphas "0.1,1.0,10.0"
        
        # Run with sampling for large datasets  
        python run_pipeline.py -c config.yaml -d data.csv --sample-size 5000 --sample-seed 42
        
        # Use NVIDIA embeddings with multiple GPUs
        python run_pipeline.py -c config.yaml -d data.csv --embedding-model "nvidia/NV-Embed-v2" --num-gpus 4
        
        # Export all data to CSV
        python run_pipeline.py -c config.yaml -d data.csv --export-csv
        
        # Export with privacy options
        python run_pipeline.py -c config.yaml -d data.csv --export-csv --exclude-text --anonymize-ids
        
        # Save state for later
        python run_pipeline.py -c config.yaml -d data.csv --export-state my_analysis.pkl
        
        # Load saved state
        python run_pipeline.py --import-state my_analysis.pkl -o new_viz.html
        
        # Manual clustering parameters (overrides auto-mode)
        python run_pipeline.py -c config.yaml -d data.csv --min-cluster-size 30 --min-samples 5
        
        # Zero-presence mode for binary outcomes
        python run_pipeline.py -c config.yaml -d emotion_data.csv --outcome-mode zero_presence
        
        # Add control variables to DML analysis
        python run_pipeline.py -c config.yaml -d data.csv --control-vars "num_raters,text_length"
        
        # Auto-detect mode with control variables
        python run_pipeline.py -c config.yaml -d data.csv --control-vars "num_raters" --super-auto
        
        # Disable automatic mode detection
        python run_pipeline.py -c config.yaml -d data.csv --disable-mode-detection --outcome-mode continuous
    """
    # Handle import state mode
    if import_state:
        print(f"\n📥 Loading pipeline state from: {import_state}")
        import joblib
        
        # Load state
        state = joblib.load(import_state)
        
        # Validate
        if not skip_validation:
            DataExporter.validate_state(state, skip_validation=False)
        
        # Create pipeline from state
        pipeline = TextEmbeddingPipeline.from_state(state)
        
        # Generate visualization from saved results
        if 'results' in state:
            output_path = pipeline.viz_generator.generate_html(
                state['results'], 
                output or 'imported_visualization.html'
            )
            print(f"\n✅ Visualization generated from imported state: {output_path}")
        else:
            print("\n⚠️  No results found in state file")
        
        return
    
    # Handle config generation mode
    if generate_config and data:
        from .auto_config import create_auto_config
        from .config import save_config
        
        click.echo(f"\n📝 Generating config file from: {data}")
        auto_config_dict = create_auto_config(data, sample_size)
        save_config(auto_config_dict, generate_config)
        click.echo(f"\n✅ Config file saved to: {generate_config}")
        click.echo("\nYou can now run:")
        click.echo(f"  perceptionml --config {generate_config} --data {data}")
        return
    
    # Always work with a config dictionary
    if config:
        # Load from file
        from .config import load_config
        config_dict = load_config(config)
        click.echo(f"\n📄 Loaded config from: {config}")
    else:
        # Auto-detect from data
        from .auto_config import create_auto_config
        # Pass CLI parameters to auto-config
        cli_params = {
            'y_var': y_var,
            'x_var': x_var,
            'control_vars': control_vars,
            'text_column': text_column,
            'id_column': id_column,
            'embedding_model': embedding_model
        }
        config_dict = create_auto_config(data, sample_size, cli_params=cli_params)
    
    # Apply ALL CLI overrides to the config
    if embedding_model:
        config_dict['pipeline']['embedding_model'] = embedding_model
    if sample_size is not None:
        config_dict['data']['sample_size'] = sample_size
    if sample_seed is not None:
        config_dict['data']['sample_seed'] = sample_seed
    
    # Sampling method configuration
    config_dict['data']['sampling_method'] = sampling_method
    if stratify_by:
        config_dict['data']['stratify_by'] = stratify_by
    
    # Analysis parameters
    if not config_dict.get('analysis'):
        config_dict['analysis'] = {}
    
    # Auto mode settings
    config_dict['analysis']['auto_mode'] = auto
    config_dict['analysis']['auto_cluster_mode'] = auto_cluster
    config_dict['analysis']['super_auto_mode'] = super_auto
    
    # Clustering parameters
    if min_cluster_size is not None:
        config_dict['analysis']['hdbscan_min_cluster_size'] = min_cluster_size
    if min_samples is not None:
        config_dict['analysis']['hdbscan_min_samples'] = min_samples
    if umap_neighbors is not None:
        config_dict['analysis']['umap_n_neighbors'] = umap_neighbors
    if umap_min_dist is not None:
        config_dict['analysis']['umap_min_dist'] = umap_min_dist
    
    # ML parameters
    if dml_folds is not None:
        config_dict['analysis']['dml_n_folds'] = dml_folds
    if xgb_estimators is not None:
        config_dict['analysis']['xgb_n_estimators'] = xgb_estimators
    if xgb_depth is not None:
        config_dict['analysis']['xgb_max_depth'] = xgb_depth
    if batch_size is not None:
        config_dict['analysis']['batch_size'] = batch_size
    
    # PC selection
    if pc_selection_methods:
        if pc_selection_methods.lower() == 'all':
            methods = ['xgboost', 'lasso', 'ridge', 'mi']
        else:
            methods = [m.strip().lower() for m in pc_selection_methods.split(',')]
        config_dict['analysis']['dml_pc_selection_methods'] = methods
    if primary_pc_method:
        config_dict['analysis']['dml_primary_pc_method'] = primary_pc_method.lower()
    
    # Regularization
    if lasso_alphas:
        config_dict['analysis']['lasso_alphas'] = [float(x.strip()) for x in lasso_alphas.split(',')]
    if ridge_alphas:
        config_dict['analysis']['ridge_alphas'] = [float(x.strip()) for x in ridge_alphas.split(',')]
    if reg_cv_folds is not None:
        config_dict['analysis']['regularization_cv_folds'] = reg_cv_folds
    
    # Outcome mode will be set after Y/X variables are processed
    
    # Control variables
    if control_vars:
        control_var_list = [cv.strip() for cv in control_vars.split(',')]
        config_dict['data']['control_variables'] = [
            {'name': cv, 'display_name': cv.replace('_', ' ').title()} 
            for cv in control_var_list
        ]
    
    # Column specifications
    if text_column:
        config_dict['data']['text_column'] = text_column
    if id_column:
        config_dict['data']['id_column'] = id_column
    # Handle explicit Y and X variables or outcomes list
    if y_var or x_var or outcomes:
        new_outcomes = []
        
        # Handle explicit Y and X variables first
        if y_var:
            new_outcomes.append({
                'name': y_var,
                'display_name': y_var.replace('_', ' ').title(),
                'type': 'continuous',
                'range': [0, 100]  # Will be auto-detected from data
            })
        
        if x_var:
            # Ensure we have a Y variable before adding X
            if not y_var and not outcomes:
                raise click.ClickException("--x-var requires --y-var to be specified")
            new_outcomes.append({
                'name': x_var,
                'display_name': x_var.replace('_', ' ').title(),
                'type': 'continuous',
                'range': [0, 100]  # Will be auto-detected from data
            })
        
        # If no explicit Y/X vars but outcomes is provided
        if outcomes and not (y_var or x_var):
            # Parse comma-separated outcomes (Y, X)
            outcome_list = [o.strip() for o in outcomes.split(',')]
            
            if len(outcome_list) >= 1:
                # First one is Y variable
                new_outcomes.append({
                    'name': outcome_list[0],
                    'display_name': outcome_list[0].replace('_', ' ').title(),
                    'type': 'continuous',
                    'range': [0, 100]  # Will be auto-detected from data
                })
            
            if len(outcome_list) >= 2:
                # Second one is X variable
                new_outcomes.append({
                    'name': outcome_list[1],
                    'display_name': outcome_list[1].replace('_', ' ').title(),
                    'type': 'continuous',
                    'range': [0, 100]  # Will be auto-detected from data
                })
        
        # Replace the outcomes completely
        if new_outcomes:
            config_dict['data']['outcomes'] = new_outcomes
    
    # Now set outcome mode AFTER outcomes are finalized
    # If outcome_mode is explicitly set (not 'auto'), disable detection automatically
    if outcome_mode != 'auto':
        config_dict['analysis']['outcome_mode_detection'] = False
        for outcome in config_dict['data']['outcomes']:
            outcome['mode'] = outcome_mode
        print(f"\n🎯 Setting outcome mode to '{outcome_mode}' for all outcomes")
    else:
        config_dict['analysis']['outcome_mode_detection'] = not disable_mode_detection
    
    # Visualization settings
    if not config_dict.get('visualization'):
        config_dict['visualization'] = {}
    
    # Now create pipeline with the complete config
    click.echo("\n📋 Creating pipeline with configuration...")
    pipeline = TextEmbeddingPipeline(config_dict=config_dict, num_gpus=num_gpus)
    
    # If force-auto is enabled, clear auto-adjustable parameters
    if force_auto:
        print("\n🔧 Force-auto mode: Clearing auto-adjustable parameters")
        config_dict['analysis']['hdbscan_min_cluster_size'] = None
        config_dict['analysis']['hdbscan_min_samples'] = None
        config_dict['analysis']['umap_n_neighbors'] = None
        config_dict['analysis']['umap_min_dist'] = None
        config_dict['analysis']['batch_size'] = None
        config_dict['analysis']['max_text_length'] = None
        
        # Clear ML parameters if in super-auto mode
        if super_auto:
            config_dict['analysis']['dml_n_folds'] = None
            config_dict['analysis']['xgb_n_estimators'] = None
            config_dict['analysis']['xgb_max_depth'] = None
            config_dict['analysis']['lasso_alphas'] = None
            config_dict['analysis']['ridge_alphas'] = None
            config_dict['analysis']['regularization_cv_folds'] = None
        
        # Recreate pipeline with cleared config
        pipeline = TextEmbeddingPipeline(config_dict=config_dict, num_gpus=num_gpus)
    
    # Check data size and warn if large
    data_df = pd.read_csv(data, nrows=0)  # Just read headers to check
    total_rows = sum(1 for _ in open(data)) - 1  # Count rows excluding header
    
    if total_rows > 10000 and sample_size is None:
        print(f"\n⚠️  WARNING: Dataset contains {total_rows:,} rows (>10,000)")
        print("   Large datasets may cause:")
        print("   - Slow processing and high memory usage")
        print("   - Large HTML files that may lag in browsers")
        print("   - Long embedding generation times")
        print("   Consider using --sample-size flag (e.g., --sample-size 10000)\n")
    
    # Calculate effective dataset size (after sampling if applicable)
    effective_size = sample_size if sample_size else total_rows
    
    # Get number of features (embedding dimensions)
    # Read config to get expected embedding dimensions
    if pipeline.config.embedding_model == 'nvidia/NV-Embed-v2':
        n_features = 4096
    elif 'all-MiniLM-L6-v2' in pipeline.config.embedding_model:
        n_features = 384
    elif 'all-roberta-large-v1' in pipeline.config.embedding_model:
        n_features = 1024
    else:
        n_features = 768  # Default for most models
    
    # Create dataset profile
    dataset_profile = DatasetProfile.from_data(
        data_shape=(effective_size, n_features),
        text_stats=None  # Could be computed if needed
    )
    
    # Check for conflicting flags
    if super_auto and not auto:
        raise click.BadParameter("Cannot use --super-auto with --no-auto. Super-auto requires auto mode.")
    
    # Initialize auto-parameter system
    auto_param_system = AutoParameterSystem(
        dataset_profile=dataset_profile,
        config=pipeline.config,
        auto_cluster_mode=auto_cluster,
        super_auto_mode=super_auto
    )
    
    # Store in pipeline for later use
    pipeline.auto_param_system = auto_param_system
    
    # Update config auto mode settings
    pipeline.config.analysis.auto_mode = auto
    pipeline.config.analysis.auto_cluster_mode = auto_cluster
    pipeline.config.analysis.super_auto_mode = super_auto
    
    # The auto parameter system will handle all parameter selection
    # based on what's in the config (including CLI overrides)
    
    # Collect CLI overrides for auto-parameter system
    cli_overrides = {
        'hdbscan_min_cluster_size': min_cluster_size,
        'hdbscan_min_samples': min_samples,
        'umap_n_neighbors': umap_neighbors,
        'umap_min_dist': umap_min_dist,
        'dml_n_folds': dml_folds,
        'xgb_n_estimators': xgb_estimators,
        'xgb_max_depth': xgb_depth
    }
    # Remove None values
    cli_overrides = {k: v for k, v in cli_overrides.items() if v is not None}
    
    # Select all parameters and apply to config
    auto_param_system.select_all_parameters(cli_overrides)
    auto_param_system.apply_to_config()
    
    # Generate and print parameter report
    print(auto_param_system.generate_report())
    
    # If preview mode, exit after showing parameters
    if preview_params:
        print("\n✅ Preview complete. Add --no-preview-params to run the analysis.")
        return
    
    # Export the effective config if requested
    if export_csv and export_dir:
        from .config import save_config
        import os
        os.makedirs(export_dir, exist_ok=True)
        config_path = os.path.join(export_dir, 'effective_config.yaml')
        save_config(pipeline.config, config_path)
        click.echo(f"\n📄 Saved effective config to: {config_path}")
    
    # Store CLI command for documentation
    import sys
    pipeline._cli_command = ' '.join(sys.argv)
    
    # Run pipeline
    html_path = pipeline.run(data, embeddings, output)
    
    # Get the results that were just generated
    # We need to reconstruct the results dict from the pipeline components
    results = pipeline._prepare_results_for_export()
    
    # Handle CSV export (enabled by default)
    if export_csv:
        export_path = pipeline.data_exporter.export_all_to_csv(
            pipeline=pipeline,
            results=results,
            output_dir=export_dir,
            exclude_text=exclude_text,
            anonymize_ids=anonymize_ids,
            cli_command=getattr(pipeline, '_cli_command', None)
        )
        print(f"\n📊 Data exported to: {export_path}")
    
    # Handle state export (always export with auto-generated name if not specified)
    if export_state:
        state_path = export_state
    else:
        # Auto-generate state filename based on timestamp and model
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = embedding_model.replace('/', '_').replace('sentence-transformers_', '') if embedding_model else 'default'
        state_path = f"analysis_state_{model_name}_{timestamp}.pkl"
    
    pipeline.data_exporter.export_state(
        pipeline=pipeline,
        results=results,
        output_path=state_path
    )
    print(f"\n💾 State saved to: {state_path}")


if __name__ == '__main__':
    main()