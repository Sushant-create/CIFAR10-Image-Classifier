#!/usr/bin/env bash
# launch.sh - one-command setup + launch for Linux/Mac
# Creates a virtual environment (first run only), installs dependencies,
# and starts the Flask app.

set -e

cd "$(dirname "$0")"

if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate

echo "Installing dependencies..."
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

if [ ! -f "model/cifar10_cnn_model.h5" ]; then
    echo ""
    echo "No trained model found at model/cifar10_cnn_model.h5"
    echo "Run 'python train_model.py' (or the CIFAR-10.ipynb notebook) first"
    echo "to train and save the model. The app will still start, but"
    echo "predictions will fail until a model exists."
    echo ""
fi

echo "Starting the CIFAR-10 Classifier app at http://localhost:5000 ..."
python main.py
