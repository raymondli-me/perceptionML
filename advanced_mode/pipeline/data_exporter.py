#!/usr/bin/env python3
"""Data export functionality for saving all pipeline outputs to CSV and other formats."""

import os
import json
import hashlib
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Union
import pickle
import joblib
import sys

from .config import PipelineConfig


class DataExporter:
    """Handles exporting all pipeline data to various formats."""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        
    def export_all_to_csv(self, pipeline: Any, results: Dict[str, Any], 
                         output_dir: Optional[str] = None,
                         exclude_text: bool = False,
                         anonymize_ids: bool = False,
                         cli_command: Optional[str] = None) -> str:
        """Export all processed data to CSV files in organized directories.
        
        Args:
            pipeline: The TextEmbeddingPipeline instance with fitted models
            results: Dictionary containing all pipeline results
            output_dir: Optional output directory path
            exclude_text: Whether to exclude raw text from exports
            anonymize_ids: Whether to replace IDs with anonymous integers
            
        Returns:
            Path to the export directory
        """
        # Create timestamped output directory
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = self.config.output_dir / f"export_{timestamp}"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n📁 Exporting data to: {output_dir}")
        
        # Save metadata
        self._save_metadata(output_dir, pipeline, results)
        
        # Create directory structure
        dirs = {
            'raw': output_dir / '01_raw_data',
            'dim_reduction': output_dir / '02_dimensionality_reduction',
            'clustering': output_dir / '03_clustering',
            'importance': output_dir / '04_feature_importance',
            'dml': output_dir / '05_dml_analysis',
            'pc_analysis': output_dir / '06_pc_analysis',
            'viz': output_dir / '07_visualization_ready'
        }
        
        for dir_path in dirs.values():
            dir_path.mkdir(exist_ok=True)
        
        # Export each category and track warnings
        print("\n📊 Exporting data categories:")
        export_warnings = []
        
        # 1. Raw data and embeddings
        print("  → Raw data and embeddings")
        try:
            self._export_raw_data(dirs['raw'], pipeline, exclude_text, anonymize_ids)
            print("    ✓ Complete")
        except Exception as e:
            print(f"    ⚠️  Error: {str(e)}")
            export_warnings.append(f"Raw data export: {str(e)}")
        
        # 2. Dimensionality reduction
        print("  → Dimensionality reduction results")
        self._export_dimensionality_reduction(dirs['dim_reduction'], pipeline, results)
        
        # 3. Clustering
        print("  → Clustering and topic modeling")
        try:
            self._export_clustering(dirs['clustering'], pipeline, results)
            print("    ✓ Complete")
        except Exception as e:
            print(f"    ⚠️  Error: {str(e)}")
            export_warnings.append(f"Clustering export: {str(e)}")
        
        # 4. Feature importance
        print("  → Feature importance metrics")
        try:
            self._export_feature_importance(dirs['importance'], pipeline, results)
            print("    ✓ Complete")
        except Exception as e:
            print(f"    ⚠️  Error: {str(e)}")
            export_warnings.append(f"Feature importance export: {str(e)}")
        
        # 5. DML analysis
        print("  → Double ML analysis results")
        try:
            self._export_dml_analysis(dirs['dml'], pipeline, results)
            print("    ✓ Complete")
        except Exception as e:
            print(f"    ⚠️  Error: {str(e)}")
            export_warnings.append(f"DML export: {str(e)}")
        
        # 6. PC analysis
        print("  → PC global and local effects")
        try:
            self._export_pc_analysis(dirs['pc_analysis'], results)
            print("    ✓ Complete")
        except Exception as e:
            print(f"    ⚠️  Error: {str(e)}")
            export_warnings.append(f"PC analysis export: {str(e)}")
        
        # 7. Visualization-ready data
        print("  → Visualization-ready data")
        try:
            self._export_visualization_data(dirs['viz'], results)
            print("    ✓ Complete")
        except Exception as e:
            print(f"    ⚠️  Error: {str(e)}")
            export_warnings.append(f"Visualization data export: {str(e)}")
        
        # Write warnings to a file if any
        if export_warnings:
            warnings_file = output_dir / 'export_warnings.txt'
            with open(warnings_file, 'w') as f:
                f.write("Export Warnings\n" + "="*50 + "\n\n")
                f.write("The following issues were encountered during export:\n\n")
                for warning in export_warnings:
                    f.write(f"- {warning}\n")
                f.write("\nSome data files may be missing from the export.\n")
            print(f"\n⚠️  {len(export_warnings)} warnings written to: export_warnings.txt")
        
        # Generate comprehensive README
        print("\n📝 Generating documentation")
        self._generate_readme(output_dir, pipeline, results, exclude_text, anonymize_ids, cli_command)
        
        print(f"\n✅ Export complete! All data saved to: {output_dir}")
        return str(output_dir)
    
    def _save_metadata(self, output_dir: Path, pipeline: Any, results: Dict[str, Any]):
        """Save metadata about the export."""
        metadata = {
            'export_timestamp': datetime.now().isoformat(),
            'pipeline_config': self.config.to_dict(),
            'data_info': {
                'n_samples': len(results.get('viz_data', [])),
                'n_features': len(pipeline.data_loader.data.columns) if hasattr(pipeline, 'data_loader') else 0,
                'text_column': self.config.data.text_column,
                'id_column': self.config.data.id_column,
                'outcomes': [{'name': o.name, 'display_name': o.display_name, 'type': o.type} 
                            for o in self.config.data.outcomes]
            },
            'model_info': {
                'embedding_model': self.config.embedding_model,
                'pca_components': self.config.analysis.pca_components,
                'umap_dimensions': self.config.analysis.umap_dimensions,
                'umap_n_neighbors': self.config.analysis.umap_n_neighbors,
                'umap_min_dist': self.config.analysis.umap_min_dist,
                'hdbscan_min_cluster_size': self.config.analysis.hdbscan_min_cluster_size,
                'hdbscan_min_samples': self.config.analysis.hdbscan_min_samples,
                'dml_n_folds': self.config.analysis.dml_n_folds,
                'xgb_n_estimators': self.config.analysis.xgb_n_estimators,
                'xgb_max_depth': self.config.analysis.xgb_max_depth,
                'dml_top_pcs': results.get('top_pcs', []),
                'lasso_alphas': self.config.analysis.lasso_alphas,
                'ridge_alphas': self.config.analysis.ridge_alphas,
                'regularization_cv_folds': self.config.analysis.regularization_cv_folds,
                'dml_pc_selection_methods': self.config.analysis.dml_pc_selection_methods,
                'dml_primary_pc_method': self.config.analysis.dml_primary_pc_method
            },
            'package_versions': self._get_package_versions()
        }
        
        # Add auto-parameter metadata if available
        if hasattr(pipeline, 'auto_param_system') and pipeline.auto_param_system:
            metadata['auto_parameter_report'] = pipeline.auto_param_system.export_metadata()
        
        # Add selected regularization alphas if available
        if 'pc_selection_info' in results and results['pc_selection_info']:
            info = results['pc_selection_info']
            selected_alphas = {}
            if 'lasso_alpha_x' in info:
                selected_alphas['lasso_alpha_social_class'] = info['lasso_alpha_x']
            if 'lasso_alpha_y' in info:
                selected_alphas['lasso_alpha_ai_rating'] = info['lasso_alpha_y']
            if 'ridge_alpha_x' in info:
                selected_alphas['ridge_alpha_social_class'] = info['ridge_alpha_x']
            if 'ridge_alpha_y' in info:
                selected_alphas['ridge_alpha_ai_rating'] = info['ridge_alpha_y']
            
            if selected_alphas:
                metadata['model_info']['selected_regularization_alphas'] = selected_alphas
        
        with open(output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
    
    def _get_package_versions(self) -> Dict[str, str]:
        """Get versions of key packages."""
        packages = ['numpy', 'pandas', 'scikit-learn', 'umap-learn', 'xgboost', 
                   'shap', 'torch', 'transformers', 'plotly', 'jinja2']
        versions = {}
        for package in packages:
            try:
                import importlib.metadata
                versions[package] = importlib.metadata.version(package)
            except:
                versions[package] = 'Not installed'
        return versions
    
    def _export_raw_data(self, dir_path: Path, pipeline: Any, 
                        exclude_text: bool, anonymize_ids: bool):
        """Export raw data and embeddings."""
        data = pipeline.data_loader.data.copy()
        
        # Handle anonymization
        if anonymize_ids:
            original_ids = data[self.config.data.id_column].values
            data[self.config.data.id_column] = range(len(data))
            # Save mapping separately for reference
            id_mapping = pd.DataFrame({
                'anonymous_id': range(len(data)),
                'original_id_hash': [hashlib.sha256(str(id).encode()).hexdigest()[:16] 
                                   for id in original_ids]
            })
            id_mapping.to_csv(dir_path / 'id_mapping.csv', index=False)
        
        # Save sampled data
        if exclude_text:
            # Remove text column but keep everything else
            data_export = data.drop(columns=[self.config.data.text_column])
        else:
            data_export = data
        
        # Choose filename based on whether sampling was used
        if hasattr(pipeline.data_loader, 'original_data') and pipeline.data_loader.original_data is not None:
            # Sampling was used - save as sampled_data.csv
            data_export.to_csv(dir_path / 'sampled_data.csv', index=False)
        else:
            # No sampling - save as original_data.csv
            data_export.to_csv(dir_path / 'original_data.csv', index=False)
        
        # Also save full original data if sampling was applied
        if hasattr(pipeline.data_loader, 'original_data') and pipeline.data_loader.original_data is not None:
            original = pipeline.data_loader.original_data.copy()
            
            if anonymize_ids:
                # Apply same anonymization to original data
                original_ids = original[self.config.data.id_column].values
                original[self.config.data.id_column] = range(len(original))
                
                # Update mapping file to include all IDs
                full_id_mapping = pd.DataFrame({
                    'anonymous_id': range(len(original)),
                    'original_id_hash': [hashlib.sha256(str(id).encode()).hexdigest()[:16] 
                                       for id in original_ids],
                    'in_sample': [i in pipeline.data_loader.sample_indices for i in range(len(original))]
                })
                full_id_mapping.to_csv(dir_path / 'id_mapping.csv', index=False)
            
            if exclude_text:
                original_export = original.drop(columns=[self.config.data.text_column])
            else:
                original_export = original
            
            original_export.to_csv(dir_path / 'original_data_full.csv', index=False)
        
        # Save embeddings
        if hasattr(pipeline, 'embedding_gen') and hasattr(pipeline.embedding_gen, 'embeddings'):
            embeddings = pipeline.embedding_gen.embeddings
            if embeddings is not None:
                emb_df = pd.DataFrame(embeddings, columns=[f'emb_{i}' for i in range(embeddings.shape[1])])
                emb_df.insert(0, self.config.data.id_column, data[self.config.data.id_column])
                emb_df.to_csv(dir_path / 'embeddings.csv', index=False)
    
    def _export_dimensionality_reduction(self, dir_path: Path, pipeline: Any, results: Dict[str, Any]):
        """Export PCA and UMAP results."""
        exported_files = []
        missing_data = []
        
        # PCA features - check multiple locations
        pca_features = None
        pca_checkpoint = None
        
        # Try to get PCA results from pipeline
        if hasattr(pipeline, '_pca_results') and pipeline._pca_results is not None:
            pca_features = pipeline._pca_results.get('features')
            pca_checkpoint = pipeline._pca_results
        # Fallback to checkpoint
        elif 'pca_results' in pipeline.__dict__ or hasattr(pipeline, 'pca_results'):
            pca_checkpoint = self.config.load_checkpoint('pca_results')
            if pca_checkpoint:
                pca_features = pca_checkpoint.get('features')
        
        if pca_features is not None and pca_checkpoint is not None:
                pca_df = pd.DataFrame(pca_features, columns=[f'PC{i}' for i in range(pca_features.shape[1])])
                pca_df.insert(0, self.config.data.id_column, pipeline.data_loader.data[self.config.data.id_column])
                pca_df.to_csv(dir_path / 'pca_features.csv', index=False)
                
                # PCA explained variance
                var_df = pd.DataFrame({
                    'PC': range(len(pca_checkpoint['explained_variance'])),
                    'explained_variance_ratio': pca_checkpoint['explained_variance'],
                    'cumulative_variance_ratio': pca_checkpoint['cumulative_variance']
                })
                var_df.to_csv(dir_path / 'pca_explained_variance.csv', index=False)
                
                # PCA percentiles
                percentiles = pca_checkpoint['percentiles']
                perc_df = pd.DataFrame(percentiles, columns=[f'PC{i}_percentile' for i in range(percentiles.shape[1])])
                perc_df.insert(0, self.config.data.id_column, pipeline.data_loader.data[self.config.data.id_column])
                perc_df.to_csv(dir_path / 'pca_percentiles.csv', index=False)
                
                # PCA loadings (if available)
                if hasattr(pipeline.dim_reducer.pca, 'components_'):
                    loadings = pipeline.dim_reducer.pca.components_
                    loadings_df = pd.DataFrame(loadings.T, columns=[f'PC{i}' for i in range(loadings.shape[0])])
                    loadings_df.to_csv(dir_path / 'pca_loadings.csv', index=False)
        
                exported_files.append('PCA features and statistics')
        else:
            missing_data.append('PCA features (pca_features.csv, pca_explained_variance.csv, pca_percentiles.csv)')
            print("  ⚠️  Warning: PCA results not found - skipping PCA export")
        
        # UMAP results - check multiple locations
        umap_coords = None
        
        # Try to get UMAP results from pipeline
        if hasattr(pipeline, '_umap_results') and pipeline._umap_results is not None:
            umap_coords = pipeline._umap_results.get('embeddings_normalized')
        # Fallback to checkpoint
        elif 'umap_3d_results' in pipeline.__dict__ or hasattr(pipeline, 'umap_3d_results'):
            umap_checkpoint = self.config.load_checkpoint('umap_3d_results')
            if umap_checkpoint:
                umap_coords = umap_checkpoint.get('embeddings_normalized')
        
        if umap_coords is not None:
                umap_df = pd.DataFrame(umap_coords, columns=['x', 'y', 'z'])
                umap_df.insert(0, self.config.data.id_column, pipeline.data_loader.data[self.config.data.id_column])
                umap_df.to_csv(dir_path / 'umap_3d.csv', index=False)
                exported_files.append('UMAP 3D coordinates')
        else:
            missing_data.append('UMAP coordinates (umap_3d.csv)')
            print("  ⚠️  Warning: UMAP results not found - skipping UMAP export")
        
        # Report summary
        if exported_files:
            print(f"    Exported: {', '.join(exported_files)}")
        if missing_data:
            print(f"    Missing: {', '.join(missing_data)}")
    
    def _export_clustering(self, dir_path: Path, pipeline: Any, results: Dict[str, Any]):
        """Export clustering and topic modeling results."""
        # Cluster assignments
        if 'viz_data' in results and len(results['viz_data']) > 0:
            cluster_data = []
            for point in results['viz_data']:
                cluster_data.append({
                    self.config.data.id_column: point['id'],
                    'cluster_id': point.get('cluster_id', -1),
                    'topic_keywords': point.get('topic_keywords', '')
                })
            cluster_df = pd.DataFrame(cluster_data)
            cluster_df.to_csv(dir_path / 'cluster_assignments.csv', index=False)
        
        # Topic summary
        if 'topic_viz_data' in results:
            topic_df = pd.DataFrame(results['topic_viz_data'])
            topic_df.to_csv(dir_path / 'topic_summary.csv', index=False)
        
        # Topic extreme stats
        if 'topic_stats_data' in results:
            stats_data = []
            for stat in results['topic_stats_data']:
                flat_stat = {
                    'topic_id': stat['topic_id'],
                    'keywords': stat.get('keywords', f"Topic {stat['topic_id']}"),
                    'size': stat.get('size', 0)
                }
                
                # Add probability/percentage fields for each outcome
                for key, value in stat.items():
                    if key.startswith(('prob_', 'pct_', 'max_impact')):
                        flat_stat[key] = value
                
                stats_data.append(flat_stat)
            
            if stats_data:
                stats_df = pd.DataFrame(stats_data)
                stats_df.to_csv(dir_path / 'topic_extreme_stats.csv', index=False)
    
    def _export_feature_importance(self, dir_path: Path, pipeline: Any, results: Dict[str, Any]):
        """Export feature importance metrics."""
        # PC selection info (MI and XGBoost)
        if 'pc_selection_info' in results and results['pc_selection_info']:
            info = results['pc_selection_info']
            
            # Create comprehensive importance table
            importance_data = []
            n_pcs = len(info.get('mi_scores', []))
            
            for i in range(n_pcs):
                importance_data.append({
                    'PC': i,
                    'MI_score': info['mi_scores'][i] if 'mi_scores' in info else None,
                    'MI_rank': info['mi_indices'].index(i) + 1 if i in info.get('mi_indices', []) else n_pcs + 1,
                    'XGBoost_importance': info['xgb_scores'][i] if 'xgb_scores' in info else None,
                    'XGBoost_rank': info['xgb_indices'].index(i) + 1 if i in info.get('xgb_indices', []) else n_pcs + 1
                })
            
            importance_df = pd.DataFrame(importance_data)
            importance_df.to_csv(dir_path / 'feature_importance_summary.csv', index=False)
            
            # Top PCs summary - now including all methods
            selection_data = {
                'method': [],
                'top_6_pcs': [],
                'purpose': []
            }
            
            methods_info = [
                ('XGBoost', 'xgb_indices', 'Primary method - nonlinear feature importance'),
                ('Lasso', 'lasso_indices', 'L1 regularization - sparse linear selection'),
                ('Ridge', 'ridge_indices', 'L2 regularization - smooth linear selection'),
                ('Mutual Information', 'mi_indices', 'Nonparametric dependency measure')
            ]
            
            for method_name, key, purpose in methods_info:
                if key in info:
                    selection_data['method'].append(method_name)
                    selection_data['top_6_pcs'].append(', '.join([f'PC{i}' for i in info[key][:6]]))
                    selection_data['purpose'].append(purpose)
            
            selection_summary = pd.DataFrame(selection_data)
            selection_summary.to_csv(dir_path / 'pc_selection_summary.csv', index=False)
            
            # Export regularization parameters if available
            if any(key in info for key in ['lasso_alpha_x', 'lasso_alpha_y', 'ridge_alpha_x', 'ridge_alpha_y']):
                reg_params = {
                    'method': [],
                    'outcome': [],
                    'selected_alpha': [],
                    'alpha_range': []
                }
                
                # Lasso alphas
                if 'lasso_alpha_x' in info:
                    lasso_range = pipeline.config.analysis.lasso_alphas or 'default range'
                    reg_params['method'].append('Lasso')
                    reg_params['outcome'].append('social_class')
                    reg_params['selected_alpha'].append(info['lasso_alpha_x'])
                    reg_params['alpha_range'].append(str(lasso_range))
                    
                if 'lasso_alpha_y' in info:
                    lasso_range = pipeline.config.analysis.lasso_alphas or 'default range'
                    reg_params['method'].append('Lasso')
                    reg_params['outcome'].append('ai_rating')
                    reg_params['selected_alpha'].append(info['lasso_alpha_y'])
                    reg_params['alpha_range'].append(str(lasso_range))
                
                # Ridge alphas
                if 'ridge_alpha_x' in info:
                    ridge_range = pipeline.config.analysis.ridge_alphas or [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
                    reg_params['method'].append('Ridge')
                    reg_params['outcome'].append('social_class')
                    reg_params['selected_alpha'].append(info['ridge_alpha_x'])
                    reg_params['alpha_range'].append(str(ridge_range))
                    
                if 'ridge_alpha_y' in info:
                    ridge_range = pipeline.config.analysis.ridge_alphas or [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
                    reg_params['method'].append('Ridge')
                    reg_params['outcome'].append('ai_rating')
                    reg_params['selected_alpha'].append(info['ridge_alpha_y'])
                    reg_params['alpha_range'].append(str(ridge_range))
                
                reg_df = pd.DataFrame(reg_params)
                reg_df.to_csv(dir_path / 'regularization_parameters.csv', index=False)
        
        # True SHAP values (using XGBoost native method)
        for outcome in self.config.data.outcomes:
            outcome_name = outcome.name
            shap_key = f'contributions_{outcome_name}'
            
            if shap_key in results.get('dml_results', {}):
                # These are now real SHAP values from XGBoost's native calculation
                shap_values = results['dml_results'][shap_key]
                shap_df = pd.DataFrame(shap_values, columns=[f'PC{i}_shap_value' for i in range(shap_values.shape[1])])
                shap_df.insert(0, self.config.data.id_column, pipeline.data_loader.data[self.config.data.id_column])
                shap_df.to_csv(dir_path / f'shap_values_{outcome_name}.csv', index=False)
    
    def _export_dml_analysis(self, dir_path: Path, pipeline: Any, results: Dict[str, Any]):
        """Export DML analysis results."""
        # Create subdirectories for residuals and predictions
        residuals_dir = dir_path / 'residuals'
        predictions_dir = dir_path / 'predictions'
        residuals_noncv_dir = dir_path / 'residuals_noncv'
        predictions_noncv_dir = dir_path / 'predictions_noncv'
        residuals_dir.mkdir(exist_ok=True)
        predictions_dir.mkdir(exist_ok=True)
        residuals_noncv_dir.mkdir(exist_ok=True)
        predictions_noncv_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for each PC selection method
        if 'dml_results_by_method' in results:
            for method in results['dml_results_by_method']:
                method_dir = dir_path / f'by_method_{method}'
                method_dir.mkdir(exist_ok=True)
                (method_dir / 'residuals').mkdir(exist_ok=True)
                (method_dir / 'predictions').mkdir(exist_ok=True)
                (method_dir / 'residuals_noncv').mkdir(exist_ok=True)
                (method_dir / 'predictions_noncv').mkdir(exist_ok=True)
        
        # DML effects summary
        if 'dml_results' in results:
            dml_summary = []
            
            # Extract all DML results
            for key, value in results['dml_results'].items():
                if '_to_' in key and isinstance(value, dict):
                    treatment = value.get('treatment', '')
                    outcome = value.get('outcome', '')
                    
                    # Add naive results
                    if 'naive' in value:
                        naive = value['naive']
                        dml_summary.append({
                            'effect': f'{treatment} → {outcome}',
                            'model': 'Naive OLS',
                            'theta': naive['theta'],
                            'se': naive['se'],
                            'ci_lower': naive['ci'][0],
                            'ci_upper': naive['ci'][1],
                            'pval': naive['pval'],
                            'r2': naive.get('r2', None)
                        })
                    
                    # Add DML results
                    if 'dml' in value:
                        dml = value['dml']
                        dml_summary.append({
                            'effect': f'{treatment} → {outcome}',
                            'model': 'DML - 200 PCs',
                            'theta': dml['theta'],
                            'se': dml['se'],
                            'ci_lower': dml['ci'][0],
                            'ci_upper': dml['ci'][1],
                            'pval': dml['pval'],
                            'r2_y': dml.get('r2_y', None),
                            'r2_x': dml.get('r2_x', None),
                            'r2_y_full': dml.get('r2_y_full', None),
                            'r2_x_full': dml.get('r2_x_full', None),
                            'reduction': value.get('reduction', None)
                        })
            
            # Add results from embeddings
            if 'dml_results_embeddings' in results:
                for key, value in results['dml_results_embeddings'].items():
                    if '_to_' in key and 'dml' in value:
                        dml = value['dml']
                        dml_summary.append({
                            'effect': key.replace('_to_', ' → '),
                            'model': 'DML - Embeddings',
                            'theta': dml['theta'],
                            'se': dml['se'],
                            'ci_lower': dml['ci'][0],
                            'ci_upper': dml['ci'][1],
                            'pval': dml['pval'],
                            'r2_y': dml.get('r2_y', None),
                            'r2_x': dml.get('r2_x', None),
                            'r2_y_full': dml.get('r2_y_full', None),
                            'r2_x_full': dml.get('r2_x_full', None)
                        })
            
            # Add results from top 6 PCs (primary method) for backward compatibility
            if 'dml_results_top6' in results:
                for key, value in results['dml_results_top6'].items():
                    if '_to_' in key and 'dml' in value:
                        dml = value['dml']
                        # Get primary method name
                        primary_method = pipeline.config.analysis.dml_primary_pc_method
                        dml_summary.append({
                            'effect': key.replace('_to_', ' → '),
                            'model': f'DML - 6 PCs ({primary_method})',
                            'theta': dml['theta'],
                            'se': dml['se'],
                            'ci_lower': dml['ci'][0],
                            'ci_upper': dml['ci'][1],
                            'pval': dml['pval'],
                            'r2_y': dml.get('r2_y', None),
                            'r2_x': dml.get('r2_x', None),
                            'r2_y_full': dml.get('r2_y_full', None),
                            'r2_x_full': dml.get('r2_x_full', None)
                        })
            
            # Add results from all PC selection methods
            if 'dml_results_by_method' in results:
                for method, method_results in results['dml_results_by_method'].items():
                    for key, value in method_results.items():
                        if '_to_' in key and 'dml' in value:
                            dml = value['dml']
                            dml_summary.append({
                                'effect': key.replace('_to_', ' → '),
                                'model': f'DML - 6 PCs ({method})',
                                'theta': dml['theta'],
                                'se': dml['se'],
                                'ci_lower': dml['ci'][0],
                                'ci_upper': dml['ci'][1],
                                'pval': dml['pval'],
                                'r2_y': dml.get('r2_y', None),
                                'r2_x': dml.get('r2_x', None),
                                'r2_y_full': dml.get('r2_y_full', None),
                                'r2_x_full': dml.get('r2_x_full', None)
                            })
            
            if dml_summary:
                summary_df = pd.DataFrame(dml_summary)
                summary_df.to_csv(dir_path / 'dml_effects_summary.csv', index=False)
                
                # Create a separate file for R² comparisons
                r2_comparison = []
                for row in dml_summary:
                    if row['model'] != 'Naive OLS' and row.get('r2_y_full') is not None:
                        r2_comparison.append({
                            'effect': row['effect'],
                            'model': row['model'],
                            'r2_y_cv': row.get('r2_y', None),
                            'r2_x_cv': row.get('r2_x', None),
                            'r2_y_full': row.get('r2_y_full', None),
                            'r2_x_full': row.get('r2_x_full', None),
                            'r2_y_diff': (row.get('r2_y_full', 0) - row.get('r2_y', 0)) if row.get('r2_y') else None,
                            'r2_x_diff': (row.get('r2_x_full', 0) - row.get('r2_x', 0)) if row.get('r2_x') else None
                        })
                
                if r2_comparison:
                    r2_df = pd.DataFrame(r2_comparison)
                    r2_df.to_csv(dir_path / 'r2_comparison.csv', index=False)
        
        # Export residuals and predictions from DML analyzer
        if hasattr(pipeline, 'dml_analyzer'):
            # Export residuals
            if hasattr(pipeline.dml_analyzer, 'residuals'):
                for pair_key, residuals_dict in pipeline.dml_analyzer.residuals.items():
                    # Determine if this is CV or non-CV
                    is_noncv = pair_key.endswith('_noncv')
                    clean_key = pair_key.replace('_noncv', '') if is_noncv else pair_key
                    
                    # Check if this belongs to a specific PC selection method
                    method_dir = None
                    for method in ['xgboost', 'lasso', 'ridge', 'mi']:
                        if f'top6pcs_{method}' in clean_key:
                            method_dir = dir_path / f'by_method_{method}'
                            break
                    
                    # Prepare the dataframe
                    residuals_df = pd.DataFrame({
                        self.config.data.id_column: pipeline.data_loader.data[self.config.data.id_column],
                        'residual_outcome': residuals_dict['outcome'],
                        'residual_treatment': residuals_dict['treatment']
                    })
                    
                    # Save to appropriate directory
                    if method_dir:
                        if is_noncv:
                            residuals_df.to_csv(method_dir / 'residuals_noncv' / f'residuals_{clean_key}.csv', index=False)
                        else:
                            residuals_df.to_csv(method_dir / 'residuals' / f'residuals_{clean_key}.csv', index=False)
                    else:
                        # Default behavior for non-method-specific residuals
                        if is_noncv:
                            residuals_df.to_csv(residuals_noncv_dir / f'residuals_{clean_key}.csv', index=False)
                        else:
                            residuals_df.to_csv(residuals_dir / f'residuals_{pair_key}.csv', index=False)
            
            # Export predictions
            if hasattr(pipeline.dml_analyzer, 'predictions'):
                for pair_key, predictions_dict in pipeline.dml_analyzer.predictions.items():
                    # Determine if this is CV or non-CV
                    is_noncv = pair_key.endswith('_noncv')
                    clean_key = pair_key.replace('_noncv', '') if is_noncv else pair_key
                    
                    # Check if this belongs to a specific PC selection method
                    method_dir = None
                    for method in ['xgboost', 'lasso', 'ridge', 'mi']:
                        if f'top6pcs_{method}' in clean_key:
                            method_dir = dir_path / f'by_method_{method}'
                            break
                    
                    # Prepare the dataframe
                    predictions_df = pd.DataFrame({
                        self.config.data.id_column: pipeline.data_loader.data[self.config.data.id_column],
                        'predicted_outcome': predictions_dict['outcome'],
                        'predicted_treatment': predictions_dict['treatment']
                    })
                    
                    # Save to appropriate directory
                    if method_dir:
                        if is_noncv:
                            predictions_df.to_csv(method_dir / 'predictions_noncv' / f'predictions_{clean_key}.csv', index=False)
                        else:
                            predictions_df.to_csv(method_dir / 'predictions' / f'predictions_{clean_key}.csv', index=False)
                    else:
                        # Default behavior for non-method-specific predictions
                        if is_noncv:
                            predictions_df.to_csv(predictions_noncv_dir / f'predictions_{clean_key}.csv', index=False)
                        else:
                            predictions_df.to_csv(predictions_dir / f'predictions_{pair_key}.csv', index=False)
        
        # Model diagnostics
        diagnostics = []
        if 'dml_results' in results:
            for key, value in results['dml_results'].items():
                if 'r2_' in key:
                    model, outcome = key.split('_', 1)
                    diagnostics.append({
                        'metric': key,
                        'value': value
                    })
        
        if diagnostics:
            diag_df = pd.DataFrame(diagnostics)
            diag_df.to_csv(dir_path / 'model_diagnostics.csv', index=False)
    
    def _export_pc_analysis(self, dir_path: Path, results: Dict[str, Any]):
        """Export PC analysis results."""
        # PC global effects
        if 'pc_global_effects' in results:
            effects_data = []
            for pc_idx, effects in results['pc_global_effects'].items():
                row = {'PC': pc_idx}
                row.update(effects)
                effects_data.append(row)
            
            effects_df = pd.DataFrame(effects_data)
            effects_df.to_csv(dir_path / 'pc_global_effects.csv', index=False)
        
        # PC detailed stats
        if 'pc_stats_data' in results:
            stats_data = []
            for pc_idx, stats in results['pc_stats_data'].items():
                row = {'PC': pc_idx}
                
                # Flatten nested dictionaries
                if 'rankings' in stats:
                    for k, v in stats['rankings'].items():
                        row[f'ranking_{k}'] = v
                
                if 'correlations' in stats:
                    for k, v in stats['correlations'].items():
                        row[f'correlation_{k}'] = v
                
                if 'shap_stats' in stats:
                    for outcome, shap_data in stats['shap_stats'].items():
                        for stat, value in shap_data.items():
                            row[f'shap_{outcome}_{stat}'] = value
                
                stats_data.append(row)
            
            if stats_data:
                stats_df = pd.DataFrame(stats_data)
                stats_df.to_csv(dir_path / 'pc_detailed_stats.csv', index=False)
        
        # PC-topic associations
        if 'pc_stats_data' in results:
            associations_data = []
            for pc_idx, stats in results['pc_stats_data'].items():
                if 'topic_associations' in stats:
                    for topic in stats['topic_associations']:
                        row = {'PC': pc_idx}
                        row.update(topic)
                        associations_data.append(row)
            
            if associations_data:
                assoc_df = pd.DataFrame(associations_data)
                assoc_df.to_csv(dir_path / 'pc_topic_associations.csv', index=False)
    
    def _export_visualization_data(self, dir_path: Path, results: Dict[str, Any]):
        """Export visualization-ready data."""
        # Complete point cloud data
        if 'viz_data' in results:
            viz_df = pd.DataFrame(results['viz_data'])
            
            # Expand PC info into separate columns if present
            if 'pc_info' in viz_df.columns and len(viz_df) > 0:
                # Remove the nested pc_info column for CSV export
                viz_df = viz_df.drop(columns=['pc_info'])
            
            viz_df.to_csv(dir_path / 'point_cloud.csv', index=False)
        
        # Category assignments
        if 'viz_data' in results and len(results['viz_data']) > 0:
            categories = []
            for point in results['viz_data']:
                categories.append({
                    self.config.data.id_column: point['id'],
                    'category': point.get('category', 'unknown')
                })
            cat_df = pd.DataFrame(categories)
            cat_df.to_csv(dir_path / 'category_labels.csv', index=False)
        
        # Thresholds
        if 'thresholds' in results:
            threshold_data = []
            for outcome, thresholds in results['thresholds'].items():
                threshold_data.append({
                    'outcome': outcome,
                    'low_threshold': thresholds['low'],
                    'high_threshold': thresholds['high']
                })
            thresh_df = pd.DataFrame(threshold_data)
            thresh_df.to_csv(dir_path / 'thresholds.csv', index=False)
    
    def _generate_readme(self, output_dir: Path, pipeline: Any, results: Dict[str, Any],
                        exclude_text: bool, anonymize_ids: bool, cli_command: Optional[str] = None):
        """Generate comprehensive README documentation."""
        # Import the comprehensive readme generator (v2 without line numbers)
        from .data_exporter_readme_v2 import generate_comprehensive_readme_v2 as generate_comprehensive_readme
        
        # Get package versions for documentation
        versions = self._get_package_versions()
        
        # Generate the comprehensive documentation
        readme_content = generate_comprehensive_readme(
            output_dir=output_dir,
            pipeline=pipeline,
            results=results,
            config=self.config,
            exclude_text=exclude_text,
            anonymize_ids=anonymize_ids,
            cli_command=cli_command,
            versions=versions
        )
        
        # Write the README
        with open(output_dir / 'README.md', 'w') as f:
            f.write(readme_content)
        
        return  # Exit early since we're using the new comprehensive generator
        
        # Check if sampling was used
        sampling_info = getattr(pipeline.data_loader, 'sampling_info', None)
        
        readme_content = f"""# PerceptionML Pipeline Export Documentation

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Pipeline Version:** {getattr(pipeline, 'version', '1.0.0')}

## Overview

This directory contains a complete export of all data generated by the PerceptionML text analysis pipeline.
The data is organized into logical categories for easy access and analysis.

## Command Used

```bash
{cli_command or 'python run_pipeline.py [command not captured]'}
```

## Privacy Settings

- **Text Excluded:** {'Yes' if exclude_text else 'No'}
- **IDs Anonymized:** {'Yes' if anonymize_ids else 'No'}

{f'''## Sampling Information

**Note:** This analysis was performed on a random sample of the original dataset for performance reasons.

- **Sampling Method:** {sampling_info['method']}
- **Original Dataset Size:** {sampling_info['original_size']:,} records
- **Sample Size:** {sampling_info['sample_size']:,} records ({sampling_info['sample_ratio']:.1%} of original)
- **Random Seed:** {sampling_info['sample_seed']}
{f"- **Stratification Column:** {sampling_info['stratify_column']}" if sampling_info.get('stratify_column') else ""}

**Rationale:** Working with datasets larger than 10,000 records can significantly impact:
- HTML file size and browser performance
- Processing time for embeddings and analysis
- Memory requirements for visualization

The sampling preserves the statistical properties of the original data{" through stratification" if sampling_info.get('stratify_column') else ""}.
''' if sampling_info else ''}

## Directory Structure

### 01_raw_data/
Original input data and text embeddings.

{f'''- **sampled_data.csv**: The sampled data used for analysis{' (text excluded)' if exclude_text else ''}
  - Columns: {', '.join(pipeline.data_loader.data.columns.tolist()) if hasattr(pipeline, 'data_loader') else 'N/A'}
  - Rows: {len(pipeline.data_loader.data) if hasattr(pipeline, 'data_loader') else 'N/A'}

- **original_data_full.csv**: The complete original dataset before sampling{' (text excluded)' if exclude_text else ''}
  - Columns: Same as sampled_data.csv
  - Rows: {len(pipeline.data_loader.original_data)}
''' if sampling_info else f'''- **original_data.csv**: The input data with all columns{' (text excluded)' if exclude_text else ''}
  - Columns: {', '.join(pipeline.data_loader.data.columns.tolist()) if hasattr(pipeline, 'data_loader') else 'N/A'}
  - Rows: {len(pipeline.data_loader.data) if hasattr(pipeline, 'data_loader') else 'N/A'}
'''}
- **embeddings.csv**: Text embeddings from {self.config.embedding_model}
  - Shape: [n_samples × embedding_dimensions]
  - Columns: ID, emb_0, emb_1, ..., emb_N

{f'''- **id_mapping.csv**: Mapping between anonymous and hashed original IDs
  - Additional column `in_sample` indicates if the ID was included in the sample''' if anonymize_ids and sampling_info else '- **id_mapping.csv**: Mapping between anonymous and hashed original IDs' if anonymize_ids else ''}

### 02_dimensionality_reduction/
Results from PCA and UMAP dimensionality reduction.

- **pca_features.csv**: Principal component scores for each sample
  - Shape: [n_samples × {self.config.analysis.pca_components}]
  - Columns: ID, PC0, PC1, ..., PC199
  
- **pca_explained_variance.csv**: Variance explained by each PC
  - Columns: PC, explained_variance_ratio, cumulative_variance_ratio
  
- **pca_percentiles.csv**: Percentile rank of each sample on each PC
  - Useful for understanding relative positions in PC space
  
- **pca_loadings.csv**: Component loadings (if available)
  - Shows how original features contribute to each PC
  
- **umap_3d.csv**: 3D UMAP coordinates
  - Columns: ID, x, y, z
  - Coordinates are normalized to [-1, 1] range

### 03_clustering/
Topic modeling and clustering results.

- **cluster_assignments.csv**: Cluster/topic assignment for each sample
  - Columns: ID, cluster_id, topic_keywords
  - cluster_id = -1 indicates noise/outliers
  
- **topic_summary.csv**: Summary statistics for each topic
  - Columns: topic_id, size, keywords, centroid coordinates
  
- **topic_extreme_stats.csv**: Topic distribution in extreme outcome groups
  - Shows over/under-representation of topics in high/low outcome groups

### 04_feature_importance/
Global and local feature importance metrics.

- **feature_importance_summary.csv**: PC importance by different methods
  - Columns: PC, MI_score, MI_rank, XGBoost_importance, XGBoost_rank
  
- **pc_selection_summary.csv**: Top 6 PCs selected by each method
  
- **shap_values_[outcome].csv**: SHAP values for each PC's contribution
  - Local explanations of how each PC affects predictions
  - One file per outcome variable

### 05_dml_analysis/
Double Machine Learning causal effect estimates.

- **dml_effects_summary.csv**: All effect estimates in one table
  - Models: Naive OLS, DML-Embeddings, DML-200PCs, DML-Top5PCs
  - Columns: effect, model, theta, se, ci_lower, ci_upper, pval, r2, reduction
  
- **model_diagnostics.csv**: R² and other model fit statistics

- **residuals/**: First-stage residuals from DML cross-fitting
  - Files: residuals_[treatment]_to_[outcome].csv
  - Columns: ID, residual_outcome, residual_treatment
  - Critical for reproducing DML estimates and diagnostics
  
- **predictions/**: First-stage predictions from DML cross-fitting  
  - Files: predictions_[treatment]_to_[outcome].csv
  - Columns: ID, predicted_outcome, predicted_treatment
  - Used to calculate residuals and assess model fit

### 06_pc_analysis/
Detailed PC-level analysis.

- **pc_global_effects.csv**: Probability changes for extreme PC values
  - Shows P(outcome|PC=high) vs P(outcome|PC=low)
  
- **pc_detailed_stats.csv**: Comprehensive PC statistics
  - Rankings, correlations, SHAP statistics
  
- **pc_topic_associations.csv**: Statistical tests of PC-topic relationships
  - Welch's t-test comparing PC values for topic members vs non-members

### 07_visualization_ready/
Data formatted for visualization.

- **point_cloud.csv**: Complete data for each point
  - All coordinates, outcomes, categories, and metadata
  
- **category_labels.csv**: High/low category assignments
  - Based on outcome thresholds
  
- **thresholds.csv**: Thresholds used for categorization
  - 80th percentile for "high", 20th percentile for "low"

## Key Concepts

### Principal Components (PCs)
Linear combinations of embedding dimensions that capture maximum variance.
- PC0 captures the most variance, PC1 the second most, etc.
- Used for dimensionality reduction and interpretability

### SHAP Values
SHapley Additive exPlanations - how much each PC contributes to predictions.
- Positive SHAP = increases predicted outcome
- Negative SHAP = decreases predicted outcome
- Magnitude = strength of effect

### Double ML (DML)
Causal inference method that controls for confounding using machine learning.
- Estimates causal effects between outcomes
- "Reduction" = how much the naive estimate shrinks after controlling for text

### Topics
Clusters found using HDBSCAN on UMAP embeddings.
- Each topic has representative keywords
- Topics may be associated with specific PCs or outcomes

## Loading the Data

### Python
```python
import pandas as pd
import numpy as np

# Load basic data
data = pd.read_csv('01_raw_data/original_data.csv')
embeddings = pd.read_csv('01_raw_data/embeddings.csv')
pca_features = pd.read_csv('02_dimensionality_reduction/pca_features.csv')

# Merge by ID
merged = data.merge(embeddings, on='{self.config.data.id_column}')
merged = merged.merge(pca_features, on='{self.config.data.id_column}')
```

### R
```r
library(tidyverse)

# Load and join data
data <- read_csv("01_raw_data/original_data.csv")
pca <- read_csv("02_dimensionality_reduction/pca_features.csv")
full_data <- data %>%
  inner_join(pca, by = "{self.config.data.id_column}")
```

## Reproducing DML Results

### Using Exported Residuals
```python
# Load DML residuals and verify estimates
residuals = pd.read_csv('05_dml_analysis/residuals/residuals_X_to_Y.csv')

# Reproduce DML theta estimate
theta = np.sum(residuals['residual_treatment'] * residuals['residual_outcome']) / \
        np.sum(residuals['residual_treatment'] ** 2)
        
# Calculate standard error
n = len(residuals)
se = np.sqrt(np.sum((residuals['residual_outcome'] - theta * residuals['residual_treatment']) ** 2) / 
             (n * np.sum(residuals['residual_treatment'] ** 2)))

print(f"Theta: {{theta:.4f}}, SE: {{se:.4f}}")
```

### Verifying First-Stage Predictions
```python
# Load predictions and original data
predictions = pd.read_csv('05_dml_analysis/predictions/predictions_X_to_Y.csv')
original = pd.read_csv('01_raw_data/original_data.csv')

# Calculate residuals from predictions
merged = predictions.merge(original, on='{self.config.data.id_column}')
merged['calc_residual_Y'] = merged['Y'] - merged['predicted_outcome']
merged['calc_residual_X'] = merged['X'] - merged['predicted_treatment']

# Compare with exported residuals
residuals = pd.read_csv('05_dml_analysis/residuals/residuals_X_to_Y.csv')
print(f"Residuals match: {{np.allclose(merged['calc_residual_Y'], residuals['residual_outcome'])}}")
```

## Complete Modeling Documentation

### 1. Text Embeddings
**Model:** {self.config.embedding_model}
- **Purpose:** Convert text into dense vector representations
- **Dimension:** {pipeline.embedding_gen.embeddings.shape[1] if hasattr(pipeline, 'embedding_gen') and hasattr(pipeline.embedding_gen, 'embeddings') and pipeline.embedding_gen.embeddings is not None else 'N/A'}
- **Preprocessing:**
  - Max text length: {self.config.analysis.max_text_length} tokens
  - Batch size: {self.config.analysis.batch_size}
  - Tokenization: Model-specific (likely WordPiece or SentencePiece)
  - Pooling: Model-specific (likely CLS token or mean pooling)

### 2. Principal Component Analysis (PCA)
**Purpose:** Dimensionality reduction while preserving maximum variance
- **Components retained:** {self.config.analysis.pca_components}
- **Preprocessing:** StandardScaler (mean=0, std=1)
- **Solver:** SVD (automatically selected by scikit-learn)
- **Random state:** 42
- **Explained variance:** See `pca_explained_variance.csv`
- **Implementation:** scikit-learn {versions.get('scikit-learn', 'unknown')}

### 3. UMAP (Uniform Manifold Approximation and Projection)
**Purpose:** Non-linear dimensionality reduction for visualization
- **Output dimensions:** {self.config.analysis.umap_dimensions}
- **n_neighbors:** {self.config.analysis.umap_n_neighbors}
  - Controls local vs global structure preservation
  - Lower = more local detail, Higher = more global structure
- **min_dist:** {self.config.analysis.umap_min_dist}
  - Minimum distance between points in low-dimensional space
  - Lower = tighter clumps, Higher = more evenly distributed
- **Metric:** cosine
  - Appropriate for high-dimensional data like embeddings
- **Random state:** 42
- **Normalization:** Output scaled to [-1, 1] range
- **Implementation:** umap-learn {versions.get('umap-learn', 'unknown')}

### 4. Clustering (HDBSCAN)
**Purpose:** Identify topics/themes in the embedding space
- **Algorithm:** HDBSCAN (Hierarchical Density-Based Spatial Clustering)
- **min_cluster_size:** {self.config.analysis.hdbscan_min_cluster_size}
  - Minimum points to form a cluster
- **min_samples:** {self.config.analysis.hdbscan_min_samples}
  - Conservative constraint on cluster formation
- **Metric:** Euclidean (on UMAP coordinates)
- **cluster_selection_epsilon:** 0.0 (default)
- **cluster_selection_method:** 'eom' (Excess of Mass)
- **Noise handling:** Points not in any cluster assigned label -1
- **Implementation:** hdbscan/scikit-learn

### 5. Topic Keyword Extraction
**Purpose:** Generate interpretable labels for clusters
- **Method:** c-TF-IDF (Class-based TF-IDF)
- **Tokenization:** Simple word tokenization
- **Stop words:** English stop words removed
- **Max features per topic:** Top 10 keywords
- **Keyword format:** "word1 - word2 - word3..."

### 6. Feature Importance Analysis

#### 6.1 Mutual Information (MI)
**Purpose:** Measure statistical dependence between PCs and outcomes
- **Estimator:** k-nearest neighbors based
- **n_neighbors:** 3 (default)
- **Random state:** 42
- **Usage:** Reference comparison for XGBoost selection

#### 6.2 XGBoost Feature Importance
**Purpose:** Select most predictive PCs for DML analysis
- **n_estimators:** {self.config.analysis.xgb_n_estimators}
- **max_depth:** 6 (for feature selection)
- **learning_rate:** 0.3 (default)
- **tree_method:** 'hist' (GPU-accelerated)
- **device:** 'cuda' (if available)
- **Random state:** 42
- **Importance type:** 'gain' (average gain across all splits)
- **Implementation:** xgboost {versions.get('xgboost', 'unknown')}

#### 6.3 SHAP Values
**Purpose:** Local feature importance for individual predictions
- **Explainer:** TreeExplainer (exact for tree models)
- **Background data:** Full training set
- **Interaction effects:** Not computed (first-order only)
- **Implementation:** shap {versions.get('shap', 'unknown')}

### 7. Double Machine Learning (DML)

#### 7.1 General Framework
**Purpose:** Estimate causal effects while controlling for high-dimensional confounders
- **Method:** Partially linear model with cross-fitting
- **Folds:** {self.config.analysis.dml_n_folds}
- **Cross-fitting:** Prevents overfitting in nuisance estimation
- **Sample splitting:** 50/50 for each fold

#### 7.2 Nuisance Models (First Stage)
**Model:** XGBoost Regressor
- **n_estimators:** {self.config.analysis.xgb_n_estimators}
- **max_depth:** {self.config.analysis.xgb_max_depth}
- **learning_rate:** 0.3 (default)
- **subsample:** 1.0 (default)
- **colsample_bytree:** 1.0 (default)
- **tree_method:** 'hist'
- **device:** 'cuda' (if available)
- **Random state:** 42
- **Early stopping:** Not used
- **Regularization:** 
  - reg_alpha (L1): 0 (default)
  - reg_lambda (L2): 1 (default)

#### 7.3 Treatment Effect Estimation (Second Stage)
**Method:** OLS on residualized outcomes
- **Standard errors:** Heteroskedasticity-robust (HC1)
- **Confidence intervals:** 95% (normal approximation)
- **Hypothesis testing:** Two-sided t-tests

#### 7.4 Model Variants
1. **Naive OLS:** Simple regression without controls
2. **DML-Embeddings:** Using all {pipeline.embedding_gen.embeddings.shape[1] if hasattr(pipeline, 'embedding_gen') and hasattr(pipeline.embedding_gen, 'embeddings') and pipeline.embedding_gen.embeddings is not None else 'N/A'} embedding dimensions
3. **DML-200PCs:** Using all 200 principal components
4. **DML-Top6PCs:** Using top 6 PCs selected by average importance

### 8. PC Effect Analysis

#### 8.1 Global Effects
**Purpose:** Estimate outcome probabilities at PC extremes
- **Method:** Logistic regression on all PCs
- **Test points:** 90th percentile (high) vs 10th percentile (low)
- **Solver:** 'lbfgs' with max_iter=1000
- **Regularization:** L2 with C=1.0 (default)
- **Preprocessing:** StandardScaler on all PCs

#### 8.2 PC-Topic Associations
**Purpose:** Test if topics have different PC distributions
- **Test:** Welch's t-test (unequal variances)
- **Comparison:** Topic members vs non-members
- **Multiple testing:** No correction applied
- **Effect size:** Mean difference in PC values

### 9. Threshold Determination
**Purpose:** Define "high" and "low" outcome values
- **Method:** Percentile-based
- **High threshold:** 80th percentile
- **Low threshold:** 20th percentile
- **Application:** Used for categorization and extreme group analysis

### 10. Statistical Ranking (Cross-validation)
**Purpose:** Robust importance rankings for PCs
- **Method:** 5-fold cross-validation
- **Splits:** Stratified random (shuffle=True)
- **Aggregation:** 
  - Average rank across folds
  - Median rank across folds
- **Random state:** 42

### 11. Visualization Processing
**Point cloud normalization:**
- UMAP coordinates scaled to [-100, 100] for Three.js
- Colors determined by outcome categories or topics
- Opacity: {self.config.visualization.default_opacity}
- Point size: {self.config.visualization.point_size}

**Category definitions:**
- both_high: Both outcomes > 80th percentile
- first_high: Y > 80th, X < 20th percentile  
- second_high: X > 80th, Y < 20th percentile
- both_low: Both outcomes < 20th percentile
- middle: All other points

### 12. Random Seeds and Reproducibility
**Fixed seeds used throughout:**
- General random state: 42
- Train/test splits: 42
- Model initialization: 42
- UMAP initialization: 42
- Bootstrap sampling: Not used

**Version control:**
- Python: {sys.version}
- NumPy: {versions.get('numpy', 'unknown')}
- Pandas: {versions.get('pandas', 'unknown')}
- Scikit-learn: {versions.get('scikit-learn', 'unknown')}
- XGBoost: {versions.get('xgboost', 'unknown')}
- UMAP: {versions.get('umap-learn', 'unknown')}
- SHAP: {versions.get('shap', 'unknown')}

### 13. Computational Details
**Hardware used:** Information not captured (recommend adding)
**Parallelization:**
- XGBoost: Default (all cores)
- UMAP: Default (usually single-threaded)
- Embeddings: Batch processing on GPU if available

**Memory optimization:**
- PCA: Incremental processing not used
- Embeddings: Processed in batches of {self.config.analysis.batch_size}

## Complete Configuration

```json
{json.dumps(self.config.to_dict(), indent=2)}
```

## Reproducibility Checklist

To exactly reproduce these results:

1. **Environment Setup:**
   ```bash
   # Create virtual environment
   python -m venv perceptionml_env
   source perceptionml_env/bin/activate  # or `perceptionml_env\\Scripts\\activate` on Windows
   
   # Install exact package versions
   pip install numpy=={versions.get('numpy', 'X.X.X')}
   pip install pandas=={versions.get('pandas', 'X.X.X')}
   pip install scikit-learn=={versions.get('scikit-learn', 'X.X.X')}
   pip install umap-learn=={versions.get('umap-learn', 'X.X.X')}
   pip install xgboost=={versions.get('xgboost', 'X.X.X')}
   pip install shap=={versions.get('shap', 'X.X.X')}
   # ... etc for all packages
   ```

2. **Data Preparation:**
   - Ensure text encoding is UTF-8
   - Verify column names match configuration
   - Check for missing values (pipeline may handle differently)

3. **Execution:**
   ```bash
   {cli_command or 'python run_pipeline.py -c [config_file] -d [data_file]'}
   ```

4. **Verification:**
   - Compare explained variance ratios
   - Check cluster sizes and assignments
   - Verify DML estimates and standard errors

## Notes on Variability

Even with fixed seeds, results may vary slightly due to:
- Floating-point precision differences across platforms
- GPU vs CPU execution (especially for XGBoost)
- Different versions of underlying libraries (BLAS, LAPACK)
- Operating system differences

For exact reproduction, use the saved state file:
```bash
python run_pipeline.py --import-state [state_file]
```

## Questions?

For questions about this export or the PerceptionML pipeline, please refer to the main documentation or contact the maintainers.
"""
        
        with open(output_dir / 'README.md', 'w') as f:
            f.write(readme_content)
    
    def export_state(self, pipeline: Any, results: Dict[str, Any], 
                    output_path: str, compress: int = 3) -> str:
        """Export complete pipeline state to pickle file.
        
        Args:
            pipeline: The TextEmbeddingPipeline instance
            results: All pipeline results
            output_path: Path for the pickle file
            compress: Compression level (0-9, default 3)
            
        Returns:
            Path to saved file
        """
        print(f"\n💾 Exporting pipeline state to: {output_path}")
        
        # Prepare state dictionary
        state = {
            'version': '1.0.0',
            'created': datetime.now().isoformat(),
            'config': self.config.to_dict(),
            'python_version': sys.version,
            'package_versions': self._get_package_versions(),
            
            # Models and transformers
            'models': {
                'pca': getattr(pipeline.dim_reducer, 'pca', None),
                'scaler': getattr(pipeline.dim_reducer, 'scaler', None),
                'umap': getattr(pipeline.dim_reducer, 'umap_model', None),
            },
            
            # Data
            'data': {
                'original': pipeline.data_loader.data if hasattr(pipeline, 'data_loader') else None,
                'embeddings': getattr(pipeline.embedding_gen, 'embeddings', None) if hasattr(pipeline, 'embedding_gen') else None,
                'original_full': getattr(pipeline.data_loader, 'original_data', None),
                'sample_indices': getattr(pipeline.data_loader, 'sample_indices', None),
            },
            
            # Sampling information
            'sampling_info': getattr(pipeline.data_loader, 'sampling_info', None),
            'cli_command': getattr(pipeline, '_cli_command', None),
            
            # Results
            'results': results,
            
            # Checksums for validation
            'checksums': self._compute_checksums(results)
        }
        
        # Add clustering model if available
        if hasattr(pipeline, 'topic_modeler') and hasattr(pipeline.topic_modeler, 'clustering_model'):
            state['models']['clustering'] = pipeline.topic_modeler.clustering_model
        
        # Add DML models and results if available
        if hasattr(pipeline, 'dml_analyzer'):
            state['models']['dml_models'] = getattr(pipeline.dml_analyzer, 'fitted_models', {})
            # Add residuals and predictions
            if hasattr(pipeline.dml_analyzer, 'residuals'):
                state['dml_residuals'] = pipeline.dml_analyzer.residuals
            if hasattr(pipeline.dml_analyzer, 'predictions'):
                state['dml_predictions'] = pipeline.dml_analyzer.predictions
        
        # Add auto-parameter system if available
        if hasattr(pipeline, 'auto_param_system') and pipeline.auto_param_system:
            state['auto_param_system'] = {
                'metadata': pipeline.auto_param_system.export_metadata(),
                'decisions': [d.to_dict() for d in pipeline.auto_param_system.decisions],
                'dataset_profile': {
                    'n_samples': pipeline.auto_param_system.profile.n_samples,
                    'n_features': pipeline.auto_param_system.profile.n_features,
                    'size_category': pipeline.auto_param_system.profile.size_category
                }
            }
        
        # Save with joblib for better compression and compatibility
        joblib.dump(state, output_path, compress=compress)
        
        # Get file size
        file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
        print(f"✅ State exported successfully! File size: {file_size:.1f} MB")
        
        return output_path
    
    def _compute_checksums(self, results: Dict[str, Any]) -> Dict[str, str]:
        """Compute checksums for validation."""
        checksums = {}
        
        # Checksum for viz_data
        if 'viz_data' in results:
            viz_str = json.dumps(results['viz_data'], sort_keys=True, default=str)
            checksums['viz_data'] = hashlib.sha256(viz_str.encode()).hexdigest()
        
        # Checksum for DML results
        if 'dml_results' in results:
            dml_str = json.dumps(results['dml_results'], sort_keys=True, default=str)
            checksums['dml_results'] = hashlib.sha256(dml_str.encode()).hexdigest()
        
        return checksums
    
    @staticmethod
    def validate_state(state: Dict[str, Any], skip_validation: bool = False) -> bool:
        """Validate loaded state for compatibility and integrity.
        
        Args:
            state: Loaded state dictionary
            skip_validation: Whether to skip checksum validation
            
        Returns:
            True if valid, raises exception otherwise
        """
        # Check version
        if state.get('version') != '1.0.0':
            import warnings
            warnings.warn(f"State version {state.get('version')} may not be fully compatible")
        
        # Validate checksums
        if not skip_validation and 'checksums' in state:
            print("🔍 Validating checksums...")
            
            # Recompute checksums
            exporter = DataExporter(PipelineConfig.from_dict(state['config']))
            current_checksums = exporter._compute_checksums(state['results'])
            
            for key, expected in state['checksums'].items():
                if key in current_checksums:
                    if current_checksums[key] != expected:
                        raise ValueError(f"Checksum mismatch for {key}. Data may be corrupted.")
            
            print("✅ Checksums valid!")
        
        return True