#!/usr/bin/env python3
"""Dynamic code reference finder for documentation."""

import ast
import inspect
from pathlib import Path
from typing import Tuple, Optional


class CodeReferenceFinder:
    """Find line numbers for specific code patterns dynamically."""
    
    def __init__(self, base_path: Path = None):
        self.base_path = base_path or Path(__file__).parent.parent
    
    def find_function(self, file_path: str, function_name: str) -> Optional[Tuple[int, int]]:
        """Find the line range of a function in a file.
        
        Returns:
            Tuple of (start_line, end_line) or None if not found
        """
        full_path = self.base_path / file_path
        try:
            with open(full_path, 'r') as f:
                tree = ast.parse(f.read())
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == function_name:
                    return (node.lineno, node.end_lineno)
        except Exception:
            return None
        
        return None
    
    def find_class_method(self, file_path: str, class_name: str, method_name: str) -> Optional[Tuple[int, int]]:
        """Find the line range of a method within a class."""
        full_path = self.base_path / file_path
        try:
            with open(full_path, 'r') as f:
                tree = ast.parse(f.read())
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and node.name == class_name:
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef) and item.name == method_name:
                            return (item.lineno, item.end_lineno)
        except Exception:
            return None
        
        return None
    
    def find_assignment(self, file_path: str, variable_name: str) -> Optional[int]:
        """Find the line number where a variable is assigned."""
        full_path = self.base_path / file_path
        try:
            with open(full_path, 'r') as f:
                lines = f.readlines()
            
            for i, line in enumerate(lines, 1):
                # Simple pattern matching for assignments
                if f"{variable_name} =" in line or f"{variable_name}=" in line:
                    return i
        except Exception:
            return None
        
        return None
    
    def find_code_block(self, file_path: str, code_snippet: str) -> Optional[Tuple[int, int]]:
        """Find the line range containing a specific code snippet."""
        full_path = self.base_path / file_path
        try:
            with open(full_path, 'r') as f:
                lines = f.readlines()
            
            # Normalize the snippet
            snippet_lines = code_snippet.strip().split('\n')
            first_line = snippet_lines[0].strip()
            
            for i, line in enumerate(lines):
                if first_line in line:
                    # Check if subsequent lines match
                    match = True
                    for j, snippet_line in enumerate(snippet_lines[1:], 1):
                        if i + j >= len(lines) or snippet_line.strip() not in lines[i + j]:
                            match = False
                            break
                    
                    if match:
                        return (i + 1, i + len(snippet_lines))
        except Exception:
            return None
        
        return None
    
    def get_line_reference(self, file_path: str, identifier: str, ref_type: str = "auto") -> str:
        """Get a formatted line reference for documentation.
        
        Args:
            file_path: Path to the file
            identifier: Function name, class.method, or code snippet
            ref_type: Type of reference ("function", "method", "assignment", "code", "auto")
        
        Returns:
            Formatted reference like "lines 123-456" or "line 123"
        """
        if ref_type == "auto":
            # Try to determine type automatically
            if '.' in identifier:
                ref_type = "method"
            elif '=' in identifier or len(identifier.split('\n')) > 1:
                ref_type = "code"
            else:
                ref_type = "function"
        
        result = None
        
        if ref_type == "function":
            result = self.find_function(file_path, identifier)
        elif ref_type == "method":
            parts = identifier.split('.')
            if len(parts) == 2:
                result = self.find_class_method(file_path, parts[0], parts[1])
        elif ref_type == "assignment":
            line = self.find_assignment(file_path, identifier)
            if line:
                return f"line {line}"
        elif ref_type == "code":
            result = self.find_code_block(file_path, identifier)
        
        if result:
            if isinstance(result, tuple):
                if result[0] == result[1]:
                    return f"line {result[0]}"
                else:
                    return f"lines {result[0]}-{result[1]}"
            else:
                return f"line {result}"
        
        return "line number not found"


# Example usage function
def get_dynamic_references():
    """Example of how to use this in the README generator."""
    finder = CodeReferenceFinder()
    
    # Find where CLI command is stored
    cli_ref = finder.get_line_reference("run_pipeline.py", "pipeline._cli_command")
    
    # Find the export_all_to_csv method
    export_ref = finder.get_line_reference("pipeline/data_exporter.py", "DataExporter.export_all_to_csv")
    
    # Find specific code block
    sampling_ref = finder.get_line_reference(
        "pipeline/data_loader.py",
        "if self.config.data.sample_seed is not None:",
        ref_type="code"
    )
    
    return {
        'cli_command': cli_ref,
        'export_method': export_ref,
        'sampling_code': sampling_ref
    }


if __name__ == "__main__":
    # Test the finder
    finder = CodeReferenceFinder()
    
    print("Testing dynamic code reference finder...")
    print("=" * 70)
    
    # Test finding a function
    ref = finder.get_line_reference("pipeline/data_loader.py", "DataLoader.load_data")
    print(f"DataLoader.load_data: {ref}")
    
    # Test finding assignment
    ref = finder.get_line_reference("run_pipeline.py", "pipeline._cli_command", ref_type="assignment")
    print(f"CLI command assignment: {ref}")
    
    # Test finding method
    ref = finder.get_line_reference("pipeline/dml_analysis.py", "DMLAnalyzer.run_dml_analysis")
    print(f"DML run_dml_analysis: {ref}")