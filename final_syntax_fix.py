#!/usr/bin/env python3
"""
Final fix for all remaining syntax errors from D212 docstring fixes.
"""

import re
from pathlib import Path

def fix_file_specific_issues():
    """Fix specific issues in individual files."""
    
    # Fix utils.py - the """..."""if isinstance pattern
    utils_path = Path("freqprob/utils.py")
    if utils_path.exists():
        with open(utils_path, 'r') as f:
            content = f.read()
        # Fix the specific case where there's invalid indentation
        content = re.sub(
            r'(""")(\s*if isinstance\(text, str\):)',
            r'\1\n    \2',
            content
        )
        with open(utils_path, 'w') as f:
            f.write(content)
    
    # Fix lazy.py - similar issue with indentation
    lazy_path = Path("freqprob/lazy.py")
    if lazy_path.exists():
        with open(lazy_path, 'r') as f:
            content = f.read()
        # Fix indentation issues
        content = re.sub(
            r'(""")(\s*self\.lazy_computer = lazy_computer)',
            r'\1\n        \2',
            content
        )
        with open(lazy_path, 'w') as f:
            f.write(content)
    
    # Fix cache.py
    cache_path = Path("freqprob/cache.py")
    if cache_path.exists():
        with open(cache_path, 'r') as f:
            content = f.read()
        # Fix indentation issues
        content = re.sub(
            r'(""")(\s*if cache_instance is None:)',
            r'\1\n        \2',
            content
        )
        with open(cache_path, 'w') as f:
            f.write(content)
    
    # Fix vectorized.py
    vectorized_path = Path("freqprob/vectorized.py")
    if vectorized_path.exists():
        with open(vectorized_path, 'r') as f:
            content = f.read()
        # Fix return statement issue
        content = re.sub(
            r'(""")(\s*return BatchScorer\(scorers\))',
            r'\1\n        \2',
            content
        )
        with open(vectorized_path, 'w') as f:
            f.write(content)
    
    # Fix validation.py - specific import issue
    validation_path = Path("freqprob/validation.py")
    if validation_path.exists():
        with open(validation_path, 'r') as f:
            content = f.read()
        # Fix the import statement that got mangled
        content = re.sub(
            r'(""")(\s*from \. import VectorizedScorer)',
            r'\1\n        \2',
            content
        )
        with open(validation_path, 'w') as f:
            f.write(content)
    
    # Fix profiling.py
    profiling_path = Path("freqprob/profiling.py")
    if profiling_path.exists():
        with open(profiling_path, 'r') as f:
            content = f.read()
        # Fix the import statement
        content = re.sub(
            r'(""")(\s*from \.memory_efficient import \()',
            r'\1\n        \2',
            content
        )
        with open(profiling_path, 'w') as f:
            f.write(content)
    
    # Fix memory_efficient.py
    memory_path = Path("freqprob/memory_efficient.py")
    if memory_path.exists():
        with open(memory_path, 'r') as f:
            content = f.read()
        # Fix the function definition issue
        content = re.sub(
            r'(""")(\s*compressed = CompressedFrequencyDistribution\()',
            r'\1\n    \2',
            content
        )
        with open(memory_path, 'w') as f:
            f.write(content)

def main():
    """Apply all final fixes."""
    fix_file_specific_issues()
    print("Applied final syntax fixes to all problematic files")

if __name__ == "__main__":
    main()