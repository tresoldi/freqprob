#!/usr/bin/env python3
"""
Script to fix D212 docstring formatting errors.
This script finds multi-line docstrings that start with a summary on a separate line
and moves the summary to the first line.
"""

import re
import sys
from pathlib import Path

def fix_d212_in_file(file_path):
    """Fix D212 errors in a single file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Pattern to match problematic docstrings
        # Match: """<whitespace_or_newline><whitespace><text>
        # This finds docstrings that start with """ followed by whitespace/newline, then indented text
        pattern = r'([ \t]*""")\s*\n([ \t]+)([A-Za-z][^\n]*)'
        
        def replacement(match):
            quote_part = match.group(1)  # The """ part with indentation
            indent = match.group(2)      # The indentation of the first line
            first_line = match.group(3)  # The first line content
            
            # Return the fixed version: summary on the same line as """
            return f'{quote_part}{first_line}\n{indent}'
        
        # Apply the fix
        fixed_content = re.sub(pattern, replacement, content)
        
        # Only write if there were changes
        if fixed_content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(fixed_content)
            print(f"Fixed D212 errors in {file_path}")
            return True
        else:
            print(f"No D212 errors found in {file_path}")
            return False
            
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

def main():
    """Main function to fix D212 errors in all Python files."""
    # Get all Python files in freqprob/ directory
    freqprob_files = list(Path("freqprob").glob("**/*.py"))
    scripts_files = list(Path("scripts").glob("**/*.py")) if Path("scripts").exists() else []
    
    all_files = freqprob_files + scripts_files
    
    fixed_count = 0
    for file_path in all_files:
        if fix_d212_in_file(file_path):
            fixed_count += 1
    
    print(f"\nFixed D212 errors in {fixed_count} files")

if __name__ == "__main__":
    main()