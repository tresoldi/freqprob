#!/usr/bin/env python3
"""
Fix remaining syntax errors from D212 docstring fixes.

This script handles specific patterns that the first fix script missed.
"""

import os
import re
from pathlib import Path

def fix_remaining_syntax_errors(file_path):
    """Fix remaining syntax errors that black couldn't handle."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # Pattern 1: Fix """docstring"""variable: type pattern
    content = re.sub(
        r'(""")([a-zA-Z_]\w*:\s*\w+)',
        r'\1\n    \2',
        content
    )
    
    # Pattern 2: Fix """docstring"""def __init__( pattern  
    content = re.sub(
        r'(""")(\s*def\s+\w+\s*\()',
        r'\1\n    \2',
        content
    )
    
    # Pattern 3: Fix """docstring"""if/while/for statements
    content = re.sub(
        r'(""")(\s*(?:if|while|for|with|try|except|finally|else|elif)\s)',
        r'\1\n        \2',
        content
    )
    
    # Pattern 4: Fix """docstring"""self.attribute = value
    content = re.sub(
        r'(""")(\s*self\.\w+)',
        r'\1\n        \2',
        content
    )
    
    # Pattern 5: Fix """docstring"""pass
    content = re.sub(
        r'(""")(\s*pass\s*$)',
        r'\1\n        \2',
        content,
        flags=re.MULTILINE
    )
    
    # Pattern 6: Fix the validation.py specific case """Container for validation test results."""test_name: str
    content = re.sub(
        r'(""")test_name:\s*str',
        r'\1\n    test_name: str',
        content
    )
    
    # Pattern 7: Fix the validation.py specific case """Container for performance measurement results."""operation_name: str
    content = re.sub(
        r'(""")operation_name:\s*str',
        r'\1\n    operation_name: str',
        content
    )
    
    # Pattern 8: Fix lazy.py specific case """Lazy computation for Maximum Likelihood Estimation."""def __init__(self):
    content = re.sub(
        r'(""")def\s+__init__\s*\(self\):',
        r'\1\n    def __init__(self):',
        content
    )
    
    # Write back if changes were made
    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    return False

def main():
    """Fix remaining syntax errors in all Python files."""
    base_dir = Path("freqprob")
    python_files = list(base_dir.glob("*.py"))
    
    fixed_files = []
    for file_path in python_files:
        if fix_remaining_syntax_errors(file_path):
            fixed_files.append(file_path)
            print(f"Fixed remaining syntax errors in {file_path}")
    
    if fixed_files:
        print(f"\nFixed remaining syntax errors in {len(fixed_files)} files")
    else:
        print("No remaining syntax errors found to fix")

if __name__ == "__main__":
    main()