#!/usr/bin/env python3
"""
Clean Python cache files and directories from the project.

This script removes all __pycache__ directories and .pyc files
from the project directory and subdirectories.
"""

import os
import shutil
from pathlib import Path


def clean_python_cache(root_dir=None):
    """
    Remove all Python cache files and directories.
    
    Args:
        root_dir: Root directory to clean (default: current script directory)
    """
    if root_dir is None:
        root_dir = Path(__file__).parent
    else:
        root_dir = Path(root_dir)
    
    if not root_dir.exists():
        print(f"❌ Directory {root_dir} does not exist")
        return
    
    print(f"🧹 Cleaning Python cache files in: {root_dir}")
    
    # Counters
    pycache_dirs_removed = 0
    pyc_files_removed = 0
    
    # Remove __pycache__ directories
    for pycache_dir in root_dir.rglob("__pycache__"):
        try:
            shutil.rmtree(pycache_dir)
            pycache_dirs_removed += 1
            print(f"  📁 Removed: {pycache_dir.relative_to(root_dir)}")
        except Exception as e:
            print(f"  ❌ Failed to remove {pycache_dir}: {e}")
    
    # Remove .pyc files
    for pyc_file in root_dir.rglob("*.pyc"):
        try:
            pyc_file.unlink()
            pyc_files_removed += 1
            print(f"  🗑️  Removed: {pyc_file.relative_to(root_dir)}")
        except Exception as e:
            print(f"  ❌ Failed to remove {pyc_file}: {e}")
    
    # Remove .pyo files (optimized bytecode)
    pyo_files_removed = 0
    for pyo_file in root_dir.rglob("*.pyo"):
        try:
            pyo_file.unlink()
            pyo_files_removed += 1
            print(f"  🗑️  Removed: {pyo_file.relative_to(root_dir)}")
        except Exception as e:
            print(f"  ❌ Failed to remove {pyo_file}: {e}")
    
    # Summary
    print(f"\n✅ Cleanup complete:")
    print(f"  📁 __pycache__ directories removed: {pycache_dirs_removed}")
    print(f"  🗑️  .pyc files removed: {pyc_files_removed}")
    print(f"  🗑️  .pyo files removed: {pyo_files_removed}")
    
    total_removed = pycache_dirs_removed + pyc_files_removed + pyo_files_removed
    if total_removed == 0:
        print("  🎉 No cache files found - project already clean!")
    else:
        print(f"  🎉 Total items removed: {total_removed}")


def main():
    """Main function with command line argument support."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Clean Python cache files from the project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python clean_cache.py                    # Clean current project
    python clean_cache.py /path/to/project   # Clean specific directory
    python clean_cache.py --dry-run          # Show what would be removed
        """
    )
    
    parser.add_argument('directory', nargs='?', default=None,
                        help='Directory to clean (default: current project)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be removed without actually removing')
    
    args = parser.parse_args()
    
    # Set root directory
    if args.directory:
        root_dir = Path(args.directory).resolve()
    else:
        root_dir = Path(__file__).parent
    
    if args.dry_run:
        print(f"🔍 DRY RUN - Scanning for Python cache files in: {root_dir}")
        
        pycache_dirs = list(root_dir.rglob("__pycache__"))
        pyc_files = list(root_dir.rglob("*.pyc"))
        pyo_files = list(root_dir.rglob("*.pyo"))
        
        print(f"\nWould remove:")
        print(f"  📁 __pycache__ directories: {len(pycache_dirs)}")
        for d in pycache_dirs:
            print(f"    - {d.relative_to(root_dir)}")
        
        print(f"  🗑️  .pyc files: {len(pyc_files)}")
        for f in pyc_files:
            print(f"    - {f.relative_to(root_dir)}")
        
        print(f"  🗑️  .pyo files: {len(pyo_files)}")
        for f in pyo_files:
            print(f"    - {f.relative_to(root_dir)}")
        
        total = len(pycache_dirs) + len(pyc_files) + len(pyo_files)
        print(f"\nTotal items that would be removed: {total}")
        
        if total > 0:
            print("\nRun without --dry-run to actually remove these files.")
    else:
        clean_python_cache(root_dir)


if __name__ == "__main__":
    main()