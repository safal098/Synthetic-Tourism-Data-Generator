#!/bin/bash

# Enhanced Pokhara Data Generator - Example Usage Script

echo "🚀 Enhanced Pokhara Data Generator"
echo "=================================="
echo

# Check if Ollama is running
echo "🔍 Checking if Ollama is running..."
if ! curl -s http://localhost:11434/api/tags > /dev/null; then
    echo "❌ Ollama is not running!"
    echo "Please start Ollama first:"
    echo "  ollama serve"
    exit 1
fi

echo "✅ Ollama is running"
echo

# Check if model is available
echo "🔍 Checking if gemma:7b model is available..."
MODEL_CHECK=$(curl -s http://localhost:11434/api/tags | grep -o '"name":"[^"]*"' | grep gemma:7b)
if [ -z "$MODEL_CHECK" ]; then
    echo "❌ gemma:7b model not found!"
    echo "Please pull the model first:"
    echo "  ollama pull gemma:7b"
    exit 1
fi

echo "✅ gemma:7b model is available"
echo

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

echo
echo "🎯 Ready to generate data!"
echo
echo "Available commands:"
echo "  1. Generate all categories:     python pokhara_data_generator.py"
echo "  2. Generate only positive:      python pokhara_data_generator.py --category positive"
echo "  3. Generate only sarcasm:       python pokhara_data_generator.py -cat sarcasm"
echo "  4. Custom config:              python pokhara_data_generator.py --config my_config.json"
echo "  5. Dry run (test config):      python pokhara_data_generator.py --dry-run"
echo
echo "📊 Progress will be shown in real-time"
echo "📝 Detailed logs: pokhara_generation.log"
echo "📁 Output files: outputs/*.tsv"
echo
echo "Press Enter to start generating all categories, or Ctrl+C to exit..."
read

echo "🚀 Starting generation..."
python pokhara_data_generator.py