@echo off
REM launch.bat - one-command setup + launch for Windows
REM Creates a virtual environment (first run only), installs dependencies,
REM and starts the Flask app.

cd /d "%~dp0"

if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)

call venv\Scripts\activate.bat

echo Installing dependencies...
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

if not exist "model\cifar10_cnn_model.h5" (
    echo.
    echo No trained model found at model\cifar10_cnn_model.h5
    echo Run "python train_model.py" (or the CIFAR-10.ipynb notebook) first
    echo to train and save the model. The app will still start, but
    echo predictions will fail until a model exists.
    echo.
)

echo Starting the CIFAR-10 Classifier app at http://localhost:5000 ...
python main.py

pause
