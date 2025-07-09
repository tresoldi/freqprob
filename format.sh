#!/bin/bash
# Format all Python files with isort and black

set -e  # Exit on any error

echo "🔧 Formatting Python code..."

# Check if isort is available
if ! command -v python -m isort &> /dev/null; then
    echo "❌ isort not found. Install with: pip install isort"
    exit 1
fi

# Check if black is available
if ! command -v python -m black &> /dev/null; then
    echo "❌ black not found. Install with: pip install black"
    exit 1
fi

# Run isort
echo "📦 Running isort..."
python -m isort .

# Run black
echo "🖤 Running black..."
python -m black .

echo "✅ Code formatting completed!"

# Show any changes made
if ! git diff --quiet; then
    echo ""
    echo "📝 Files modified:"
    git diff --name-only
    echo ""
    echo "To commit these changes:"
    echo "  git add ."
    echo "  git commit -m 'Apply code formatting with isort and black'"
fi