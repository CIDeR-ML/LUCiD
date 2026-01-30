#!/usr/bin/env python3
"""
Script to clear outputs from all Jupyter notebooks in the current directory and subdirectories.

This script finds all .ipynb files and removes their cell outputs and execution counts,
making them clean for version control.
"""

import json
import os
from pathlib import Path


def clear_notebook_outputs(notebook_path):
    """
    Clear outputs and execution counts from a Jupyter notebook.
    
    Parameters
    ----------
    notebook_path : str or Path
        Path to the notebook file
        
    Returns
    -------
    bool
        True if notebook was modified, False if no changes needed
    """
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        print(f"Error reading {notebook_path}: {e}")
        return False
    
    modified = False
    
    # Clear outputs and execution counts from all cells
    for cell in notebook.get('cells', []):
        # Clear outputs
        if 'outputs' in cell and cell['outputs']:
            cell['outputs'] = []
            modified = True
            
        # Clear execution count
        if 'execution_count' in cell and cell['execution_count'] is not None:
            cell['execution_count'] = None
            modified = True
    
    # Clear notebook-level execution count if it exists
    if 'execution_count' in notebook and notebook['execution_count'] is not None:
        notebook['execution_count'] = None
        modified = True
    
    # Write back only if modified
    if modified:
        try:
            with open(notebook_path, 'w', encoding='utf-8') as f:
                json.dump(notebook, f, indent=1, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"Error writing {notebook_path}: {e}")
            return False
    
    return False


def main():
    """Main function to process all notebooks in the current directory."""
    notebooks_dir = Path(".")

    # Find all .ipynb files recursively
    notebook_files = list(notebooks_dir.rglob("*.ipynb"))
    
    if not notebook_files:
        print("No Jupyter notebooks found in the current directory")
        return
    
    print(f"Found {len(notebook_files)} notebook(s)")
    
    modified_count = 0
    
    for notebook_path in notebook_files:
        print(f"Processing: {notebook_path}")
        
        if clear_notebook_outputs(notebook_path):
            print(f"  ✓ Cleared outputs from {notebook_path}")
            modified_count += 1
        else:
            print(f"  - No outputs to clear in {notebook_path}")
    
    print(f"\nCompleted! Modified {modified_count} out of {len(notebook_files)} notebooks")
    
    if modified_count > 0:
        print("\nNext steps:")
        print("1. Review the changes with: git diff")
        print("2. Add to staging: git add *.ipynb")
        print("3. Commit changes: git commit -m 'Clear Jupyter notebook outputs'")


if __name__ == "__main__":
    main()