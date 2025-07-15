"""
Command-line interface for BASIC MODE
"""

import argparse
import sys
from pathlib import Path
import logging

from .analysis import BasicAnalysis

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="PerceptionML BASIC MODE - Reproducible DML-LME Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate embeddings from raw text
  python -m perceptionML.basic_mode --data mydata.csv --treatment social_class --outcome ai_rating --text essay_text --id essay_id
  
  # Use precomputed embeddings
  python -m perceptionML.basic_mode --embeddings embeddings.csv --treatment social_class --outcome ai_rating --precomputed
  
  # Specify custom parameters
  python -m perceptionML.basic_mode --data mydata.csv --treatment X --outcome Y --text text_col --pca-components 100 --top-features 10
        """
    )
    
    # Data inputs
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument('--data', type=str, help='Path to CSV file with raw data')
    data_group.add_argument('--embeddings', type=str, help='Path to CSV file with precomputed embeddings')
    
    # Column specifications
    parser.add_argument('--treatment', type=str, required=True,
                       help='Name of treatment column (X variable, e.g., social_class)')
    parser.add_argument('--outcome', type=str, required=True,
                       help='Name of outcome column (Y variable, e.g., ai_rating)')
    parser.add_argument('--text', type=str,
                       help='Name of text column (required if using raw data)')
    parser.add_argument('--id', type=str,
                       help='Name of ID column (optional)')
    
    # Embedding options
    parser.add_argument('--precomputed', action='store_true',
                       help='Flag to indicate embeddings are precomputed')
    parser.add_argument('--embedding-model', type=str,
                       default='sentence-transformers/all-MiniLM-L6-v2',
                       help='Embedding model to use (default: MiniLM)')
    
    # Analysis parameters
    parser.add_argument('--pca-components', type=int, default=200,
                       help='Number of PCA components (default: 200)')
    parser.add_argument('--top-features', type=int, default=6,
                       help='Number of top features to select (default: 6)')
    parser.add_argument('--cv-folds', type=int, default=5,
                       help='Number of cross-validation folds (default: 5)')
    parser.add_argument('--random-seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    
    # Output options
    parser.add_argument('--output-dir', type=str,
                       help='Directory to save outputs (default: timestamped directory)')
    parser.add_argument('--no-export', action='store_true',
                       help='Skip exporting CSV files')
    
    args = parser.parse_args()
    
    # Validation
    if args.data and not args.text and not args.precomputed:
        parser.error("--text is required when using raw data (--data)")
    
    # Initialize analysis
    logger.info("Initializing BASIC MODE analysis...")
    analysis = BasicAnalysis(
        embedding_model=args.embedding_model,
        n_pca_components=args.pca_components,
        n_top_features=args.top_features,
        n_folds=args.cv_folds,
        random_seed=args.random_seed,
        output_dir=args.output_dir
    )
    
    # Run analysis
    try:
        if args.precomputed:
            logger.info("Using precomputed embeddings...")
            results = analysis.run(
                embeddings_path=args.embeddings,
                treatment_col=args.treatment,
                outcome_col=args.outcome,
                id_col=args.id,
                precomputed_embeddings=True
            )
        else:
            logger.info("Generating embeddings from text...")
            results = analysis.run(
                data_path=args.data,
                treatment_col=args.treatment,
                outcome_col=args.outcome,
                text_col=args.text,
                id_col=args.id,
                precomputed_embeddings=False
            )
        
        logger.info("Analysis complete!")
        
        if not args.no_export:
            logger.info(f"Results saved to: {analysis.output_dir}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        return 1


if __name__ == '__main__':
    sys.exit(main())