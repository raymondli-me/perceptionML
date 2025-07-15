#!/usr/bin/env python3
"""Clustering using HDBSCAN and topic extraction with c-TF-IDF."""

import numpy as np
import pandas as pd
import hdbscan
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Dict, List, Tuple, Optional
import re
from collections import Counter

from .config import PipelineConfig


class TopicModeler:
    """Handles clustering and topic extraction."""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.clusterer = None
        self.vectorizer = None
        
    def fit_clusters(self, embeddings: np.ndarray) -> np.ndarray:
        """Fit HDBSCAN clustering on embeddings."""
        print("Fitting HDBSCAN clustering...")
        
        self.clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.config.analysis.hdbscan_min_cluster_size,
            min_samples=self.config.analysis.hdbscan_min_samples,
            cluster_selection_method='eom',
            metric='euclidean',
            core_dist_n_jobs=-1
        )
        
        cluster_labels = self.clusterer.fit_predict(embeddings)
        
        # Get cluster statistics
        unique_clusters = [c for c in np.unique(cluster_labels) if c != -1]
        n_noise = np.sum(cluster_labels == -1)
        
        print(f"✓ Clustering complete:")
        print(f"  - Clusters found: {len(unique_clusters)}")
        print(f"  - Noise points: {n_noise} ({n_noise/len(cluster_labels)*100:.1f}%)")
        
        # Print cluster sizes
        for cluster_id in unique_clusters[:10]:  # First 10 clusters
            size = np.sum(cluster_labels == cluster_id)
            print(f"  - Cluster {cluster_id}: {size} points")
        
        return cluster_labels
    
    def extract_topics(self, texts: List[str], 
                      cluster_labels: np.ndarray,
                      n_words: int = 5) -> Dict[int, str]:
        """Extract topic keywords using c-TF-IDF."""
        print("Extracting topic keywords...")
        
        # Preprocess texts
        processed_texts = [self._preprocess_text(text) for text in texts]
        
        # Group texts by cluster
        cluster_docs = self._group_texts_by_cluster(processed_texts, cluster_labels)
        
        # Fit on all documents
        all_docs = list(cluster_docs.values())
        doc_labels = list(cluster_docs.keys())
        
        # Create TF-IDF vectorizer
        # Adjust min_df based on number of documents
        min_df = 1 if len(all_docs) < 10 else 2
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=min_df,
            max_df=0.95
        )
        
        if not all_docs:
            return {}
        
        tfidf_matrix = self.vectorizer.fit_transform(all_docs)
        feature_names = self.vectorizer.get_feature_names_out()
        
        # Extract keywords for each cluster
        topic_keywords = {}
        
        for idx, cluster_id in enumerate(doc_labels):
            if cluster_id == -1:  # Skip noise cluster
                continue
            
            # Get TF-IDF scores for this cluster
            cluster_tfidf = tfidf_matrix[idx].toarray().flatten()
            
            # Get top keywords
            top_indices = cluster_tfidf.argsort()[-n_words*2:][::-1]
            keywords = []
            
            for i in top_indices:
                if cluster_tfidf[i] > 0 and len(keywords) < n_words:
                    keyword = feature_names[i]
                    # Filter out single letters and numbers
                    if len(keyword) > 1 and not keyword.isdigit():
                        keywords.append(keyword.title())
            
            topic_keywords[cluster_id] = ' - '.join(keywords[:n_words])
        
        print(f"✓ Extracted keywords for {len(topic_keywords)} topics")
        
        return topic_keywords
    
    def _preprocess_text(self, text: str) -> str:
        """Basic text preprocessing."""
        text = str(text).lower()
        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        # Keep only letters and spaces
        text = re.sub(r'[^a-z\s]', ' ', text)
        # Remove extra spaces
        text = ' '.join(text.split())
        return text
    
    def _group_texts_by_cluster(self, texts: List[str], 
                               cluster_labels: np.ndarray) -> Dict[int, str]:
        """Group texts by cluster ID."""
        cluster_docs = {}
        
        for text, cluster_id in zip(texts, cluster_labels):
            if cluster_id not in cluster_docs:
                cluster_docs[cluster_id] = []
            cluster_docs[cluster_id].append(text)
        
        # Concatenate texts for each cluster
        for cluster_id in cluster_docs:
            cluster_docs[cluster_id] = ' '.join(cluster_docs[cluster_id])
        
        return cluster_docs
    
    def calculate_cluster_statistics(self, data: pd.DataFrame,
                                   cluster_labels: np.ndarray,
                                   embeddings: np.ndarray) -> List[Dict]:
        """Calculate statistics for each cluster."""
        stats = []
        unique_clusters = [c for c in np.unique(cluster_labels) if c != -1]
        
        for cluster_id in unique_clusters:
            cluster_mask = cluster_labels == cluster_id
            cluster_size = np.sum(cluster_mask)
            
            if cluster_size == 0:
                continue
            
            # Calculate centroid
            cluster_points = embeddings[cluster_mask]
            centroid = cluster_points.mean(axis=0)
            
            # Calculate outcome statistics
            cluster_stats = {
                'cluster_id': int(cluster_id),
                'size': int(cluster_size),
                'centroid': centroid.tolist()
            }
            
            # Add statistics for each outcome
            for outcome in self.config.data.outcomes:
                outcome_data = data[outcome.name][cluster_mask]
                
                if outcome.type in ['continuous', 'ordinal']:
                    cluster_stats[f'{outcome.name}_mean'] = float(outcome_data.mean())
                    cluster_stats[f'{outcome.name}_std'] = float(outcome_data.std())
                    cluster_stats[f'{outcome.name}_median'] = float(outcome_data.median())
                else:  # categorical
                    # Get mode
                    mode_value = outcome_data.mode()
                    if len(mode_value) > 0:
                        cluster_stats[f'{outcome.name}_mode'] = str(mode_value[0])
                    cluster_stats[f'{outcome.name}_diversity'] = len(outcome_data.unique())
            
            stats.append(cluster_stats)
        
        return stats
    
    def prepare_topic_visualization(self, topic_keywords: Dict[int, str],
                                  cluster_stats: List[Dict]) -> List[Dict]:
        """Prepare topic data for visualization."""
        topic_viz = []
        
        # Create lookup for cluster stats
        stats_lookup = {s['cluster_id']: s for s in cluster_stats}
        
        for cluster_id, keywords in topic_keywords.items():
            if cluster_id in stats_lookup:
                stats = stats_lookup[cluster_id]
                centroid = stats['centroid']
                # Handle both 2D and 3D centroids
                x = float(centroid[0] * 100) if len(centroid) > 0 else 0
                y = float(centroid[1] * 100) if len(centroid) > 1 else 0
                z = float(centroid[2] * 100) if len(centroid) > 2 else 0
                
                topic_viz.append({
                    'topic_id': int(cluster_id),
                    'label': keywords,  # For compatibility with JS
                    'keywords': keywords,
                    'x': x,  # Scale to match point cloud
                    'y': y,
                    'z': z,
                    'centroid': centroid,
                    'size': int(stats['size'])
                })
        
        # Sort by size (largest first)
        topic_viz.sort(key=lambda x: x['size'], reverse=True)
        
        return topic_viz
    
    def calculate_extreme_group_statistics(self, data: pd.DataFrame,
                                         cluster_labels: np.ndarray,
                                         thresholds: Dict) -> List[Dict]:
        """Calculate probability of extreme outcomes for each topic."""
        topic_stats = []
        unique_clusters = [c for c in np.unique(cluster_labels) if c != -1]
        
        for cluster_id in unique_clusters:
            cluster_mask = cluster_labels == cluster_id
            cluster_data = data[cluster_mask]
            cluster_size = len(cluster_data)
            
            if cluster_size == 0:
                continue
            
            stats = {
                'topic_id': int(cluster_id),
                'size': cluster_size
            }
            
            # Calculate probabilities for each outcome
            max_prob = 0
            for outcome in self.config.data.outcomes:
                outcome_values = cluster_data[outcome.name].values
                all_values = data[outcome.name].values
                threshold = thresholds[outcome.name]
                
                # Check if this outcome is in zero-presence mode
                outcome_mode = getattr(outcome, 'mode', 'continuous')
                
                if outcome_mode == 'zero_presence':
                    # Zero-presence mode: calculate presence rates
                    # In topic
                    topic_present = np.sum(outcome_values != 0)
                    topic_present_rate = topic_present / cluster_size if cluster_size > 0 else 0
                    
                    # Outside topic
                    non_topic_mask = ~cluster_mask
                    non_topic_values = all_values[non_topic_mask]
                    non_topic_present = np.sum(non_topic_values != 0)
                    non_topic_size = len(non_topic_values)
                    non_topic_present_rate = non_topic_present / non_topic_size if non_topic_size > 0 else 0
                    
                    # Relative risk
                    relative_risk = topic_present_rate / non_topic_present_rate if non_topic_present_rate > 0 else float('inf')
                    
                    # Average magnitude when present
                    if topic_present > 0:
                        avg_magnitude = np.mean(outcome_values[outcome_values != 0])
                    else:
                        avg_magnitude = 0
                    
                    # Store zero-presence specific stats
                    stats[f'presence_rate_{outcome.name}'] = topic_present_rate
                    stats[f'non_topic_presence_rate_{outcome.name}'] = non_topic_present_rate
                    stats[f'relative_risk_{outcome.name}'] = relative_risk
                    stats[f'avg_magnitude_{outcome.name}'] = avg_magnitude
                    
                    # For compatibility, also store as prob/pct with presence
                    stats[f'prob_{outcome.name}_present'] = topic_present_rate
                    stats[f'prob_{outcome.name}_absent'] = 1 - topic_present_rate
                    stats[f'pct_{outcome.name}_present'] = topic_present_rate * 100
                    stats[f'pct_{outcome.name}_absent'] = (1 - topic_present_rate) * 100
                    
                    # For backward compatibility, map high/low to present/absent
                    stats[f'prob_{outcome.name}_high'] = topic_present_rate
                    stats[f'prob_{outcome.name}_low'] = 1 - topic_present_rate
                    stats[f'pct_{outcome.name}_high'] = topic_present_rate * 100
                    stats[f'pct_{outcome.name}_low'] = (1 - topic_present_rate) * 100
                    
                    # Track maximum probability for sorting
                    max_prob = max(max_prob, topic_present_rate)
                    
                else:
                    # Continuous mode: Use fixed 10th/90th percentiles (like PCA stats)
                    # NOT the visual thresholds
                    outcome_p90 = np.percentile(outcome_values, 90)
                    outcome_p10 = np.percentile(outcome_values, 10)
                    
                    # Count extreme values using fixed percentiles
                    n_high = np.sum(outcome_values >= outcome_p90)
                    n_low = np.sum(outcome_values <= outcome_p10)
                    
                    # Calculate probabilities
                    prob_high = n_high / cluster_size
                    prob_low = n_low / cluster_size
                    
                    # Add both prob_ and pct_ versions for compatibility
                    stats[f'prob_{outcome.name}_high'] = prob_high
                    stats[f'prob_{outcome.name}_low'] = prob_low
                    stats[f'pct_{outcome.name}_high'] = prob_high * 100  # Convert to percentage
                    stats[f'pct_{outcome.name}_low'] = prob_low * 100
                    
                    # Track maximum probability
                    max_prob = max(max_prob, prob_high, prob_low)
            
            stats['max_impact_prob'] = max_prob
            topic_stats.append(stats)
        
        # Sort by maximum impact
        topic_stats.sort(key=lambda x: x['max_impact_prob'], reverse=True)
        
        return topic_stats