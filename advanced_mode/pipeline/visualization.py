#!/usr/bin/env python3
"""HTML visualization generation using Jinja2."""

import json
import numpy as np
from jinja2 import Environment, FileSystemLoader, PackageLoader, ChoiceLoader
from pathlib import Path
from typing import Dict, List, Any
import importlib.resources as pkg_resources

from .config import PipelineConfig


class VisualizationGenerator:
    """Generates interactive HTML visualizations."""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        
        # Try multiple loaders: package first, then file system fallback
        loaders = []
        
        # Try to load from package
        try:
            loaders.append(PackageLoader('pipeline', 'templates'))
        except ImportError:
            pass
        
        # Add file system loader as fallback
        template_dir = Path(__file__).parent / 'templates'
        if template_dir.exists():
            loaders.append(FileSystemLoader(template_dir))
        
        # Use ChoiceLoader to try multiple sources
        self.env = Environment(loader=ChoiceLoader(loaders))
        
    def generate_html(self, data: Dict[str, Any], output_path: str) -> str:
        """Generate the complete HTML visualization."""
        print("\nGenerating HTML visualization...")
        
        # Load template
        template = self.env.get_template('base.html.j2')
        
        # Prepare template context
        context = self._prepare_context(data)
        
        # Render HTML
        html_content = template.render(**context)
        
        # Save to file
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"✓ Visualization saved to: {output_file}")
        return str(output_file)
    
    def _prepare_context(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare all data for template context."""
        
        # Calculate cloud center (coordinates already scaled by 100)
        cloud_center = {
            'x': float(np.mean([d['x'] for d in data['viz_data']])),
            'y': float(np.mean([d['y'] for d in data['viz_data']])),
            'z': float(np.mean([d['z'] for d in data['viz_data']]))
        }
        
        # Get outcome modes from data
        outcome_modes = data.get('outcome_modes', {})
        
        # Prepare outcome configurations for JS
        outcomes_config = []
        for outcome in self.config.data.outcomes:
            outcome_mode = outcome_modes.get(outcome.name, 'continuous')
            outcome_config = {
                'name': outcome.name,
                'display_name': outcome.display_name,
                'type': outcome.type,
                'mode': outcome_mode,
                'thresholds': data['thresholds'][outcome.name],
                'statistics': data['outcome_stats'][outcome.name]
            }
            outcomes_config.append(outcome_config)
        
        # Prepare category colors based on number of outcomes and mode
        if len(self.config.data.outcomes) == 2:
            # Check if any outcome is in zero-presence mode
            any_zero_presence = any(
                outcome_modes.get(o.name, 'continuous') == 'zero_presence' 
                for o in self.config.data.outcomes
            )
            
            if any_zero_presence:
                # Zero-presence mode color scheme - using continuous mode colors
                category_colors = {
                    'both_absent': [1.0, 1.0, 0.0],      # Yellow (same as both_low)
                    'first_present': [1.0, 0.0, 1.0],    # Magenta (same as first_high)
                    'second_present': [0.0, 1.0, 1.0],   # Cyan (same as second_high)
                    'both_present': [0.0, 1.0, 0.0],     # Green (same as both_high)
                    'agreement': [0.0, 0.9, 0.0],        # Slightly darker green
                    'disagreement': [1.0, 0.0, 0.0],     # Pure bright red
                    'hidden': [0.3, 0.3, 0.3],           # Light gray for unchecked categories
                    # For backward compatibility
                    'both_high': [0.0, 1.0, 0.0],
                    'first_high': [1.0, 0.0, 1.0],
                    'second_high': [0.0, 1.0, 1.0],
                    'both_low': [1.0, 1.0, 0.0],
                    'middle': [0.5, 0.5, 0.5]
                }
            else:
                # Original continuous mode color scheme
                category_colors = {
                    'both_high': [0.0, 1.0, 0.0],      # Green
                    'first_high': [1.0, 0.0, 1.0],     # Magenta
                    'second_high': [0.0, 1.0, 1.0],    # Cyan
                    'both_low': [1.0, 1.0, 0.0],       # Yellow
                    'middle': [0.4, 0.4, 0.4]          # Gray
                }
        else:
            # Use different scheme for other cases
            category_colors = {
                'high': [0.0, 1.0, 0.0],           # Green
                'low': [1.0, 0.0, 0.0],            # Red
                'middle': [0.4, 0.4, 0.4]          # Gray
            }
        
        # Format numbers for display
        def format_number(x):
            if isinstance(x, (int, np.integer)):
                return int(x)
            elif isinstance(x, (float, np.floating)):
                return round(float(x), 4)
            return x
        
        # Prepare DML results for display
        dml_table_data = {}
        if 'dml_results' in data:
            dml_table_data = self._format_dml_results_combined(data)
        
        # Create context
        context = {
            # Configuration
            'config': {
                'visualization': {
                    'title': self.config.visualization.title,
                    'point_size': self.config.visualization.point_size,
                    'default_opacity': self.config.visualization.default_opacity,
                    'essay_font_size': self.config.visualization.essay_font_size,
                    'auto_rotate_speed': self.config.visualization.auto_rotate_speed,
                    'transition_speed': self.config.visualization.transition_speed,
                    'topic_text_size': self.config.visualization.topic_text_size,
                    'topic_opacity': self.config.visualization.topic_opacity
                },
                'data': {
                    'text_column': self.config.data.text_column,
                    'id_column': self.config.data.id_column,
                    'outcomes': outcomes_config
                },
                'analysis': {
                    'dml_primary_pc_method': self.config.analysis.dml_primary_pc_method
                }
            },
            
            # Data
            'viz_data': json.dumps(data['viz_data']),
            'topic_viz_data': json.dumps(data.get('topic_viz_data', [])),
            'topic_stats_data': json.dumps(data.get('topic_stats_data', [])),
            'cloud_center': cloud_center,
            'total_count': len(data['viz_data']),
            
            # Category colors
            'category_colors': json.dumps(category_colors),
            
            # Gallery category labels
            'gallery_categories': self._get_gallery_categories(outcome_modes),
            
            # Outcomes
            'outcomes_config': json.dumps(outcomes_config),
            'outcome_stats': data['outcome_stats'],
            'outcome_modes': outcome_modes,  # Pass as dict, not JSON
            
            # PC analysis
            'pc_global_effects': json.dumps(data.get('pc_global_effects', {})),
            'pc_variance_explained': json.dumps(data.get('variance_explained', [])),
            'pc_stats_data': json.dumps(data.get('pc_stats_data', {})),
            
            # DML results
            'dml_results': dml_table_data,
            'has_dml': bool(dml_table_data),
            
            # Top PCs string
            'top_pcs_string': ', '.join([f'PC{i}' for i in data.get('top_pcs', [])]),
            
            # PC selection info
            'pc_selection_info': json.dumps(data.get('pc_selection_info', {})),
            
            # Control variables
            'has_control_variables': data.get('has_control_variables', False),
            'control_variables': json.dumps([
                {'name': cv.name, 'display_name': cv.display_name} 
                for cv in data.get('control_variables', [])
            ])
        }
        
        return context
    
    def _get_gallery_categories(self, outcome_modes: Dict[str, str]) -> Dict[str, str]:
        """Get gallery category labels based on outcome modes."""
        if len(self.config.data.outcomes) != 2:
            return {}
            
        outcome1, outcome2 = self.config.data.outcomes[:2]
        mode1 = outcome_modes.get(outcome1.name, 'continuous')
        mode2 = outcome_modes.get(outcome2.name, 'continuous')
        
        # Check if either is in zero-presence mode
        if mode1 == 'zero_presence' or mode2 == 'zero_presence':
            # Zero-presence mode labels
            return {
                'both_absent': f'Both Absent (0, 0)',
                'first_present': f'{outcome1.display_name} Present Only',
                'second_present': f'{outcome2.display_name} Present Only',
                'both_present': f'Both Present (>0, >0)',
                'agreement': f'Agreement (same presence status)'
            }
        else:
            # Continuous mode labels
            return {
                'both_high': f'Both High',
                'first_high': f'{outcome1.display_name} High, {outcome2.display_name} Low',
                'second_high': f'{outcome1.display_name} Low, {outcome2.display_name} High',
                'both_low': f'Both Low',
                'middle': f'Middle (mixed)'
            }
    
    def _format_dml_results_combined(self, data: Dict) -> Dict:
        """Format all DML results (embeddings, 200 PCs, 6 PCs) for display in table."""
        formatted = {}
        
        # Get all DML results
        dml_200 = data.get('dml_results', {})
        dml_emb = data.get('dml_results_embeddings', {})
        dml_6 = data.get('dml_results_top6', {})
        dml_by_method = data.get('dml_results_by_method', {})
        top_pcs = data.get('top_pcs', [])
        pc_selection_info = data.get('pc_selection_info', {})
        
        # Find all outcome pairs from 200 PC results
        for key, value in dml_200.items():
            if '_to_' in key and isinstance(value, dict) and 'naive' in value:
                treatment = value['treatment']
                outcome = value['outcome']
                
                # Create combined result with all models
                formatted[key] = {
                    'treatment': treatment,
                    'outcome': outcome,
                    'naive': self._format_estimate(value['naive']),
                    'dml': self._format_estimate(value['dml']),  # 200 PCs
                    'dml_200pcs': self._format_estimate(value['dml']),
                    'reduction': round(value['reduction'], 1)
                }
                
                # Add embeddings model if available
                emb_key = f"{treatment}_to_{outcome}"
                if emb_key in dml_emb and 'dml' in dml_emb[emb_key]:
                    formatted[key]['dml_embeddings'] = self._format_estimate(dml_emb[emb_key]['dml'])
                
                # Add 6 PCs model if available (primary method for backward compatibility)
                if emb_key in dml_6 and 'dml' in dml_6[emb_key]:
                    dml_6_data = self._format_estimate(dml_6[emb_key]['dml'])
                    dml_6_data['pcs_used'] = ', '.join([f'PC{i}' for i in top_pcs])
                    formatted[key]['dml_6pcs'] = dml_6_data
                
                # Add results for each PC selection method
                formatted[key]['dml_by_method'] = {}
                for method, method_results in dml_by_method.items():
                    if emb_key in method_results and 'dml' in method_results[emb_key]:
                        method_data = self._format_estimate(method_results[emb_key]['dml'])
                        # Add the PCs used for this method
                        method_indices_key = f'{method}_indices'
                        if method_indices_key in pc_selection_info:
                            method_indices = pc_selection_info[method_indices_key]
                            method_data['pcs_used'] = ', '.join([f'PC{i}' for i in method_indices])
                        formatted[key]['dml_by_method'][method] = method_data
        
        return formatted
    
    def _format_estimate(self, estimate: Dict) -> Dict:
        """Format a statistical estimate for display."""
        formatted = {
            'theta': round(estimate['theta'], 3),
            'se': round(estimate['se'], 3),
            'ci_lower': round(estimate['ci'][0], 3),
            'ci_upper': round(estimate['ci'][1], 3),
            'pval': estimate['pval'],  # Keep full precision for p-values
            'cross_fitted': estimate.get('cross_fitted', False)
        }
        
        # Add R² and correlation statistics if available
        if 'r2' in estimate:
            formatted['r2'] = round(estimate['r2'], 3)
        if 'r2_y' in estimate:
            formatted['r2_y'] = round(estimate['r2_y'], 3)
        if 'r2_x' in estimate:
            formatted['r2_x'] = round(estimate['r2_x'], 3)
        if 'r2_y_full' in estimate:
            formatted['r2_y_full'] = round(estimate['r2_y_full'], 3)
        if 'r2_x_full' in estimate:
            formatted['r2_x_full'] = round(estimate['r2_x_full'], 3)
        if 'correlation' in estimate:
            formatted['correlation'] = round(estimate['correlation'], 3)
        if 'corr_resid' in estimate:
            formatted['corr_resid'] = round(estimate['corr_resid'], 3)
        if 'corr_pred' in estimate:
            formatted['corr_pred'] = round(estimate['corr_pred'], 3)
            
        return formatted