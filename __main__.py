"""
PerceptionML Command Line Interface
===================================

Main entry point for the perceptionML package.
Provides access to both basic and advanced modes.
"""

import argparse
import sys
import os

def main():
    parser = argparse.ArgumentParser(
        prog='perceptionML',
        description='Text Perception Analysis using Language Model Embeddings',
        epilog='For mode-specific help: python -m perceptionML.basic_mode --help'
    )
    
    parser.add_argument(
        'mode',
        choices=['basic', 'advanced'],
        help='Analysis mode to run'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='%(prog)s 2.0.0'
    )
    
    # Parse just the mode argument
    args, remaining = parser.parse_known_args()
    
    if args.mode == 'basic':
        # Run basic mode with remaining arguments
        from .basic_mode.__main__ import main as basic_main
        sys.argv = ['basic_mode'] + remaining
        basic_main()
    
    elif args.mode == 'advanced':
        # Run advanced mode with remaining arguments
        from .advanced_mode.run_pipeline import main as advanced_main
        sys.argv = ['advanced_mode'] + remaining
        advanced_main()

if __name__ == "__main__":
    main()