#!/usr/bin/env python3
"""Automatic configuration detection from data files."""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import click


def detect_columns(df: pd.DataFrame) -> Dict[str, str]:
    """Detect column types from dataframe."""
    columns = {}
    
    # Find text column (longest average string length)
    text_cols = []
    for col in df.columns:
        if df[col].dtype == 'object':
            avg_len = df[col].astype(str).str.len().mean()
            text_cols.append((col, avg_len))
    
    if text_cols:
        text_col = max(text_cols, key=lambda x: x[1])[0]
        columns['text_column'] = text_col
    else:
        raise ValueError("No text column found in data")
    
    # Find ID column (unique values)
    id_candidates = []
    for col in df.columns:
        if col != columns['text_column']:
            n_unique = df[col].nunique()
            if n_unique == len(df) or n_unique > len(df) * 0.95:
                id_candidates.append(col)
    
    if id_candidates:
        # Prefer columns with 'id' in name
        id_col = next((c for c in id_candidates if 'id' in c.lower()), id_candidates[0])
        columns['id_column'] = id_col
    else:
        # Generate ID column
        columns['id_column'] = 'generated_id'
        columns['generate_id'] = True
    
    return columns


def detect_outcomes(df: pd.DataFrame, text_col: str, id_col: str) -> Tuple[List[Dict], List[str]]:
    """Detect numeric columns that could be outcomes or control variables."""
    outcomes = []
    potential_controls = []
    
    # Find all numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove ID column if numeric
    if id_col in numeric_cols:
        numeric_cols.remove(id_col)
    
    # Analyze each numeric column
    for col in numeric_cols:
        values = df[col].dropna()
        if len(values) == 0:
            continue
            
        # Detect if it's zero-inflated (good for zero-presence mode)
        zero_percent = (values == 0).sum() / len(values)
        n_unique = values.nunique()
        
        # Heuristics for control variables vs outcomes
        is_likely_control = False
        
        # Likely control variable if:
        # - Has few unique values (e.g., 2-20) suggesting categories
        # - Name contains control-related keywords
        # - Is integer type with limited range
        control_keywords = ['age', 'gender', 'sex', 'group', 'condition', 'batch', 
                          'rater', 'year', 'month', 'day', 'time', 'count', 'num_']
        
        if (n_unique <= 20 or 
            any(keyword in col.lower() for keyword in control_keywords) or
            (values.dtype == 'int64' and n_unique < len(values) * 0.1)):
            is_likely_control = True
        
        outcome = {
            'name': col,
            'display_name': col.replace('_', ' ').title(),
            'type': 'continuous',
            'range': [float(values.min()), float(values.max())]
        }
        # Store metadata separately (not part of config)
        outcome_metadata = {
            'n_unique': n_unique,
            'is_likely_control': is_likely_control
        }
        
        # Auto-detect zero-presence mode
        if zero_percent > 0.3:  # More than 30% zeros
            outcome['mode'] = 'zero_presence'
            outcome_metadata['zero_percent'] = zero_percent
        
        if is_likely_control:
            potential_controls.append(col)
        
        outcomes.append((outcome, outcome_metadata))
    
    # Separate outcomes from controls based on metadata
    actual_outcomes = []
    for outcome, metadata in outcomes:
        if not metadata.get('is_likely_control', False):
            actual_outcomes.append(outcome)  # Just append the outcome dict, not the tuple
    
    # If we marked everything as control, take first 2 as outcomes
    if not actual_outcomes and outcomes:
        # Take first 2 outcome tuples and extract just the outcome dict
        actual_outcomes = [o[0] for o in outcomes[:2]]
        for outcome_dict in actual_outcomes:
            if outcome_dict['name'] in potential_controls:
                potential_controls.remove(outcome_dict['name'])
    else:
        actual_outcomes = actual_outcomes[:2]  # Limit to 2 outcomes
    
    return actual_outcomes, potential_controls


def create_auto_config(data_path: str, sample_size: Optional[int] = None, cli_params: Optional[Dict] = None) -> Dict:
    """Create automatic configuration from data file."""
    cli_params = cli_params or {}
    
    # Read sample of data
    df = pd.read_csv(data_path, nrows=1000)  # Sample for detection
    
    # Use CLI parameters if provided, otherwise detect
    if cli_params.get('text_column'):
        text_col = cli_params['text_column']
        columns = {'text_column': text_col}
    else:
        columns = detect_columns(df)
        text_col = columns['text_column']
    
    if cli_params.get('id_column'):
        id_col = cli_params['id_column']
        columns['id_column'] = id_col
    else:
        if 'id_column' not in columns:
            columns = detect_columns(df)
        id_col = columns['id_column']
    
    # Handle outcomes based on CLI parameters
    if cli_params.get('y_var') and cli_params.get('x_var'):
        # Use explicitly provided Y and X variables
        outcomes = [
            {
                'name': cli_params['y_var'],
                'display_name': cli_params['y_var'].replace('_', ' ').title(),
                'type': 'continuous',
                'range': [0, 100]
            },
            {
                'name': cli_params['x_var'],
                'display_name': cli_params['x_var'].replace('_', ' ').title(),
                'type': 'continuous',
                'range': [0, 100]
            }
        ]
        # Get control variables from CLI or detect remaining numeric columns
        if cli_params.get('control_vars'):
            control_vars = [cv.strip() for cv in cli_params['control_vars'].split(',')]
        else:
            _, control_vars = detect_outcomes(df, text_col, id_col)
            # Remove Y and X from control vars if they were detected
            control_vars = [cv for cv in control_vars if cv not in [cli_params['y_var'], cli_params['x_var']]]
    else:
        # Detect outcomes and control variables
        outcomes, control_vars = detect_outcomes(df, text_col, id_col)
    
    if not outcomes:
        # No numeric columns - create synthetic outcomes for exploration
        click.echo("\n⚠️  No numeric columns found. Creating synthetic outcomes for text exploration.")
        outcomes = [
            {
                'name': 'text_length',
                'display_name': 'Text Length',
                'type': 'continuous',
                'range': [0, 1000],
                'synthetic': True
            },
            {
                'name': 'sentence_count', 
                'display_name': 'Sentence Count',
                'type': 'continuous',
                'range': [0, 100],
                'synthetic': True
            }
        ]
    
    # Create config
    config = {
        'pipeline': {
            'name': 'Auto-detected Configuration',
            'embedding_model': cli_params.get('embedding_model', 'sentence-transformers/all-MiniLM-L6-v2')
        },
        'data': {
            'text_column': text_col,
            'id_column': id_col,
            'outcomes': outcomes[:2],  # Limit to 2 outcomes
            'sample_size': sample_size
        },
        'analysis': {
            'auto_mode': True  # Use our new defaults
        },
        'visualization': {
            'title': f'Text Analysis: {Path(data_path).stem}'
        }
    }
    
    # Add generation flag if needed
    if columns.get('generate_id'):
        config['data']['generate_id'] = True
    
    # Add control variables if detected
    if control_vars:
        config['data']['control_variables'] = [
            {'name': cv, 'display_name': cv.replace('_', ' ').title()} 
            for cv in control_vars[:5]  # Limit to 5 control variables
        ]
    
    # Report what we found
    click.echo(f"\n{'='*70}")
    click.echo("🔍 AUTO-DETECTED CONFIGURATION")
    click.echo(f"{'='*70}")
    
    click.echo(f"\n📊 Dataset: {Path(data_path).name}")
    click.echo(f"   Rows: {len(df):,}")
    
    click.echo(f"\n📝 Column Detection:")
    click.echo(f"   Text variable: '{text_col}'")
    click.echo(f"   ID variable: '{id_col}' {'(will be generated)' if columns.get('generate_id') else ''}")
    
    if len(outcomes) >= 2:
        click.echo(f"   Y variable: '{outcomes[0]['name']}' (outcome 1)")
        click.echo(f"   X variable: '{outcomes[1]['name']}' (outcome 2)")
    elif len(outcomes) == 1:
        click.echo(f"   Y variable: '{outcomes[0]['name']}' (outcome 1)")
        click.echo(f"   X variable: None (will use synthetic)")
    else:
        click.echo(f"   Y variable: 'text_length' (synthetic)")
        click.echo(f"   X variable: 'sentence_count' (synthetic)")
    
    if control_vars:
        click.echo(f"\n🎛️  Control Variables Detected ({len(control_vars)}):")
        for cv in control_vars[:5]:
            click.echo(f"   - {cv}")
        if len(control_vars) > 5:
            click.echo(f"   ... and {len(control_vars) - 5} more")
    else:
        click.echo(f"\n🎛️  Control Variables: None detected")
    
    click.echo(f"\n🤖 Embedding Model: {config['pipeline']['embedding_model']}")
    click.echo(f"🎯 Clustering Mode: Many topics (detailed granularity)")
    
    # Warnings and recommendations
    total_rows = len(pd.read_csv(data_path))
    if total_rows > 10000:
        click.echo(f"\n⚠️  LARGE DATASET WARNING")
        click.echo(f"   Your dataset has {total_rows:,} rows.")
        click.echo(f"   For faster processing, we recommend sampling to 10,000 rows:")
        click.echo(f"\n   perceptionml --data {Path(data_path).name} --sample-size 10000")
    
    click.echo(f"\n💡 TO CUSTOMIZE:")
    click.echo("   - See all options: perceptionml --help")
    click.echo("   - Generate editable config: perceptionml --data your_data.csv --generate-config my_config.yaml")
    click.echo("   - Override specific columns: perceptionml --data your_data.csv --control-vars 'age,gender'")
    click.echo("   - Change clustering: perceptionml --data your_data.csv --auto-cluster medium")
    click.echo("   - Maximum clusters: perceptionml --data your_data.csv --auto-cluster descriptions")
    
    click.echo(f"\n{'='*70}\n")
    
    return config


def generate_synthetic_outcomes(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    """Generate synthetic outcomes from text for exploration."""
    # Text length
    df['text_length'] = df[text_col].astype(str).str.len()
    
    # Sentence count (approximate)
    df['sentence_count'] = df[text_col].astype(str).str.count('[.!?]+')
    
    return df