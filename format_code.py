#!/usr/bin/env python3
"""
Code formatting script for FreqProb project.
Applies isort and black to all Python files.
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str], description: str) -> bool:
    """Run a command and return success status."""
    print(f"Running {description}...")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error in {description}:")
        print(e.stderr)
        return False


def main() -> int:
    """Main function to format code."""
    # Get all Python files
    python_files = list(Path(".").glob("**/*.py"))

    if not python_files:
        print("No Python files found.")
        return 0

    print(f"Found {len(python_files)} Python files to format.")

    # Run isort
    success = run_command(["python", "-m", "isort", "."], "isort (import sorting)")

    if not success:
        return 1

    # Run black
    success = run_command(["python", "-m", "black", "."], "black (code formatting)")

    if not success:
        return 1

    print("✅ Code formatting completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
