# CIFAR-10 Image Classification using Deep Learning

A ResNet50-based image classifier trained on the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset and served through a FastAPI app with a simple web UI. Upload a photo and get the predicted class with the top-3 confidences.

**Classes:** airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

## How it works

- **Model:** `Input(32x32x3) → RandomFlip → RandomTranslation → Resizing(128) → ResNet50 preprocessing → ResNet50 → softmax`. Preprocessing is built into the model, so it takes raw 0-255 pixels.
- **Model size reduction:** see `model_shrink.ipynb` for how the smaller model (`cifar10_resnet50_small.keras`) was produced.
- **Inference:** uploaded images are converted to RGB, resized to the model input size, and classified. Test-time augmentation (TTA) averages the prediction of the image and its mirrored copy.
- **API:** FastAPI backend with a static front end in `static/`.

## Project structure

```
.
├── main.py                        # FastAPI app (model loading + /predict endpoint)
├── model_shrink.ipynb             # Notebook: model training / shrinking
├── static/                        # Web UI (index.html)
├── requirements.txt               # Python dependencies
├── render.yaml                    # Render deployment config
└── cifar10_resnet50_small.keras   # Trained model
```

## Setup

```bash
git clone https://github.com/gogulvkn/cifar10_classification.git
cd cifar10_classification
pip install -r requirements.txt
```

## Run locally

```bash
uvicorn main:app --reload
```

Open http://127.0.0.1:8000 in your browser and upload an image.

## API

| Method | Endpoint   | Description                                  |
|--------|------------|----------------------------------------------|
| GET    | `/`        | Web UI                                       |
| GET    | `/health`  | Health check, returns `{"status": "ok"}`     |
| POST   | `/predict` | Classify an uploaded image (`file` form field) |

Example:

```bash
curl -X POST -F "file=@cat.jpg" http://127.0.0.1:8000/predict
```

Response:

```json
{
  "label": "cat",
  "confidence": 0.93,
  "top3": [
    {"label": "cat", "confidence": 0.93},
    {"label": "dog", "confidence": 0.05},
    {"label": "deer", "confidence": 0.01}
  ]
}
```

## Configuration

Set these environment variables to change behaviour:

| Variable      | Default                         | Description                                                    |
|---------------|---------------------------------|----------------------------------------------------------------|
| `MODEL_PATH`  | `cifar10_resnet50_small.keras`  | Path to the `.keras` model file                                |
| `MODEL_URL`   | none                            | Direct download link used if the model file is missing         |
| `RESIZE_MODE` | `squash`                        | `squash` keeps the whole photo, `crop` center-crops to a square |
| `TTA`         | `1`                             | Set to `0` to disable mirrored-image averaging                 |

## Deployment

The repo includes a `render.yaml` for deploying to [Render](https://render.com). To keep the repo small, host the model file elsewhere (for example Hugging Face Hub) and set `MODEL_URL`.

## Limitations

CIFAR-10 images are 32x32, and the model only knows the 10 classes above. Anything outside them (a lion, a bus, a person) is forced into the closest of the 10 classes, so predictions on unrelated images can look confident and still be wrong.

## Tech stack

Python, TensorFlow/Keras, FastAPI, Pillow, NumPy
