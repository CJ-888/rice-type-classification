# GrainPalette - Rice Type Classification Project

GrainPalette is a deep learning and Flask-based rice type classification project.  
The user uploads a rice grain image, and the trained model predicts the rice type.

## Project Structure

```text
GrainPalette/
├── app.py
├── predict.py
├── requirements.txt
├── README.md
├── data/
│   ├── train/
│   └── validation/
├── training/
│   ├── train.py
│   └── train.ipynb
├── templates/
│   ├── index.html
│   ├── details.html
│   └── results.html
└── static/
    ├── css/
    │   └── style.css
    └── uploads/
```

## Recommended Environment

Use **Python 3.7** because `tensorflow==2.3.2` works best with older Python versions.

## Installation

```bash
conda create -n grainpalette python=3.7
conda activate grainpalette
pip install -r requirements.txt
python -m ipykernel install --user --name grainpalette --display-name "Python (grainpalette)"
```

## Dataset Structure

Put your rice images in this format:

```text
data/
├── train/
│   ├── arborio/
│   ├── basmati/
│   ├── ipsala/
│   ├── jasmine/
│   └── karacadag/
└── validation/
    ├── arborio/
    ├── basmati/
    ├── ipsala/
    ├── jasmine/
    └── karacadag/
```

Change the folder names if your classes are different.

## Train the Model

From the project root:

```bash
python training/train.py
```

This will create:
- `training/rice.h5`
- `training/class_indices.json`
- `training/accuracy_plot.png`
- `training/loss_plot.png`

## Run the Flask App

```bash
python app.py
```

Then open:

```text
http://127.0.0.1:5000/
```

## Important Note

Your original topic mentions **MobileNetV4**, but the given prerequisites use **TensorFlow 2.3.2 / Keras 2.3.1**.  
That stack does **not** provide MobileNetV4, so this project uses **MobileNet** for compatibility.

If your lecturer specifically requires MobileNetV4, you would need a newer TensorFlow setup.
