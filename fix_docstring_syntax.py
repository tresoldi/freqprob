#!/usr/bin/env python3
"""
Fix docstring syntax errors introduced by the D212 fix script.

This script fixes cases where code was incorrectly placed immediately after
docstring closing quotes without proper line breaks.
"""

import os
import re
from pathlib import Path

def fix_docstring_syntax_errors(file_path):
    """Fix syntax errors where code follows docstring closing quotes."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # Pattern 1: Fix """..."""def function_name
    content = re.sub(
        r'(""")(\w+\s*\()',
        r'\1\n    \2',
        content
    )
    
    # Pattern 2: Fix """..."""class_attribute: type = value
    content = re.sub(
        r'(""")([\w_]+\s*:\s*[\w\[\],\s]+\s*=)',
        r'\1\n    \2',
        content
    )
    
    # Pattern 3: Fix """..."""if/else/for/while statements
    content = re.sub(
        r'(""")(if\s|else\s|elif\s|for\s|while\s|return\s|pass)',
        r'\1\n        \2',
        content
    )
    
    # Pattern 4: Fix """..."""variable_assignment
    content = re.sub(
        r'(""")([a-zA-Z_]\w*\s*=)',
        r'\1\n        \2',
        content
    )
    
    # Pattern 5: Fix """..."""operation_name: str (specific case from validation.py)
    content = re.sub(
        r'(""")(operation_name:\s*str)',
        r'\1\n    \2',
        content
    )
    
    # Pattern 6: Fix class field definitions immediately after docstrings
    content = re.sub(
        r'(""")([a-zA-Z_]\w*:\s*[A-Z][\w\[\],\s]*\s*=)',
        r'\1\n    \2',
        content
    )
    
    # Write back if changes were made
    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    return False

def main():
    """Fix syntax errors in all Python files."""
    base_dir = Path("freqprob")
    python_files = list(base_dir.glob("*.py"))
    
    fixed_files = []
    for file_path in python_files:
        if fix_docstring_syntax_errors(file_path):
            fixed_files.append(file_path)
            print(f"Fixed syntax errors in {file_path}")
    
    if fixed_files:
        print(f"\nFixed syntax errors in {len(fixed_files)} files")
    else:
        print("No syntax errors found to fix")

if __name__ == "__main__":
    main()