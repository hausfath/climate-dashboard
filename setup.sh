#!/bin/bash
# Setup script for Climate Dashboard

set -e

echo "Setting up Climate Dashboard..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Create data directory
mkdir -p data

echo ""
echo "Setup complete!"
echo ""
echo "To run the dashboard:"
echo "  source venv/bin/activate"
echo "  python run.py"
echo ""
echo "Or use individual commands:"
echo "  python run.py dashboard    # Run dashboard only"
echo "  python run.py update       # Update data only"
echo "  python run.py update --force  # Force refresh data"
echo ""
echo "To run the automatic scheduler (updates daily at 6 AM):"
echo "  python scheduler.py"
