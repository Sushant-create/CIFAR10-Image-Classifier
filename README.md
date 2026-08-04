# CIFAR-10 Image Classification using CNN

A Convolutional Neural Network that classifies 32×32 color images into 10
everyday categories (planes, cars, animals, ships, trucks...), wrapped in a
small Flask web app so you can drag in an image and get a live prediction.

## Objective
Build a Convolutional Neural Network (CNN) to classify images from the
CIFAR-10 dataset into 10 distinct categories, with a target test accuracy
of **85%**.

## Dataset
* **Name:** CIFAR-10
* **Link:** [https://www.cs.toronto.edu/~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html)
* **Details:** 60,000 32×32×3 (RGB) color images in 10 classes — 50,000
  training images and 10,000 test images, loaded directly via
  `tensorflow.keras.datasets.cifar10.load_data()` (no manual download
  needed).
* **Classes:** Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse,
  Ship, Truck.

## Libraries Used
| Library | Purpose |
|---|---|
| TensorFlow / Keras | Building, training, and saving the CNN |
| NumPy | Array/tensor manipulation |
| SciPy | Supporting scientific computations used by Keras image augmentation |
| Scikit-learn | Classification report & confusion matrix |
| Matplotlib | Plotting sample images and training curves |
| Seaborn | Confusion matrix heatmap |
| Flask | Serving the trained model as a web app |
| Pillow (PIL) | Reading and resizing uploaded images in the app |

## Methodology
1. **Load the data** — CIFAR-10 is loaded pre-split into train/test sets
   via Keras.
2. **Explore the data** — shapes are printed and a 5×5 grid of sample
   images with their labels is plotted to sanity-check the data.
3. **Preprocess** — pixel values are normalized from the `[0, 255]` range
   to `[0, 1]` by dividing by 255.
4. **Augment** — `ImageDataGenerator` applies random rotation, width/height
   shifts, and horizontal flips on the fly during training, so the model
   sees more varied examples and generalizes better instead of memorizing
   the training set.
5. **Build** — a CNN is built with three convolutional blocks (increasing
   filter depth), each followed by batch normalization, max pooling, and
   dropout for regularization, then a dense classification head.
6. **Train** — compiled with the Adam optimizer and sparse categorical
   cross-entropy loss, trained for up to 30 epochs with a
   `ReduceLROnPlateau` callback that halves the learning rate when
   validation loss plateaus.
7. **Evaluate** — accuracy/loss curves are plotted to check for
   overfitting, then the model is evaluated on the held-out test set and
   scored with a classification report and confusion matrix.
8. **Save & serve** — the trained model is saved to
   `model/cifar10_cnn_model.h5` and loaded by the Flask app (`main.py`) to
   serve predictions on new, user-uploaded images.

## Model Architecture
```
Input (32, 32, 3)
├─ Conv2D(32, 3x3) → BatchNorm → Conv2D(32, 3x3) → BatchNorm → MaxPool(2x2) → Dropout(0.25)
├─ Conv2D(64, 3x3) → BatchNorm → Conv2D(64, 3x3) → BatchNorm → MaxPool(2x2) → Dropout(0.25)
├─ Conv2D(128, 3x3) → BatchNorm → Conv2D(128, 3x3) → BatchNorm → MaxPool(2x2) → Dropout(0.25)
├─ Flatten
├─ Dense(128, relu) → BatchNorm → Dropout(0.5)
└─ Dense(10, softmax)
```
* **Optimizer:** Adam
* **Loss:** Sparse categorical cross-entropy
* **Regularization:** Batch normalization + progressive dropout (0.25 → 0.5)
  to reduce overfitting
* **Callback:** `ReduceLROnPlateau` (monitors `val_loss`, factor 0.5,
  patience 3)

## Results
* The notebook trains for up to 30 epochs and plots training vs.
  validation accuracy/loss to visually confirm the model isn't
  overfitting.
* After training, the model is evaluated on the 10,000-image test set,
  reporting overall **test accuracy** and **test loss**.
* A full **classification report** (precision/recall/F1 per class) and a
  **confusion matrix heatmap** are generated to show exactly which
  classes the model confuses most often (commonly cat/dog and
  automobile/truck in CIFAR-10).
* Exact numbers depend on the training run on your machine/GPU — re-run
  `CIFAR-10.ipynb` or `train_model.py` to reproduce metrics and plots
  locally.

## Conclusion
This project shows an end-to-end CNN image classification pipeline: data
loading, augmentation, a regularized convolutional architecture, training
with adaptive learning rate reduction, and thorough evaluation via
accuracy curves, a classification report, and a confusion matrix. The
trained model is then productionized behind a lightweight Flask API and
a simple web UI, turning the notebook experiment into an app that anyone
can use to classify their own images in real time. Future improvements
could include deeper architectures (e.g. ResNet-style skip connections),
transfer learning from ImageNet-pretrained models, or further tuning of
augmentation strength to push past the 85% accuracy target.

---

## Project Structure
```
CIFAR10-Image-Classifier/
├── CIFAR-10.ipynb              # Main notebook: full training pipeline
├── train_model.py              # Script version of the notebook (CLI training)
├── main.py                     # Flask app entry point (the "app")
├── test_app.py                 # Automated tests for the Flask app
├── templates/
│   └── index.html              # Web UI (upload + live predictions)
├── model/
│   └── cifar10_cnn_model.h5    # Saved trained model (generated by training)
├── test_samples/
│   ├── generate_test_samples.py  # Pulls sample CIFAR-10 images to test with
│   └── README.md
├── requirements.txt
├── launch.sh                   # One-command setup + launch (Linux/Mac)
├── launch.bat                  # One-command setup + launch (Windows)
├── .gitignore
└── .gitattributes
```

## Getting Started

### 1. Train the model
Run the notebook `CIFAR-10.ipynb` top-to-bottom, **or** from the command
line:
```bash
pip install -r requirements.txt
python train_model.py
```
Either path saves the trained model to `model/cifar10_cnn_model.h5`.

### 2. Launch the app
```bash
./launch.sh        # Linux/Mac
launch.bat         # Windows
```
This creates a virtual environment, installs dependencies, and starts the
app at **http://localhost:5000**.

Prefer to run it manually instead of the launch script?
```bash
pip install -r requirements.txt
python main.py
```

### 3. Try it out
Don't have an image handy? Generate a few real CIFAR-10 test images to
drag into the app:
```bash
python test_samples/generate_test_samples.py --count 20
```

### 4. Run the tests
```bash
python -m unittest test_app.py -v
```
