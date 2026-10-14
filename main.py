import io
import os
import urllib.request
from pathlib import Path

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps

BASE_DIR = Path(__file__).parent
MODEL_PATH = os.getenv("MODEL_PATH", str(BASE_DIR / "cifar10_resnet50_small.keras"))
RESIZE_MODE = os.getenv("RESIZE_MODE", "squash")  # "squash" keeps the whole photo, "crop" center-crops
TTA = os.getenv("TTA", "1") == "1"  # average prediction of image + mirrored image
MODEL_URL = os.getenv("MODEL_URL")  # optional: direct download link to the .keras file

# Standard CIFAR-10 label order (same as tf.keras.datasets.cifar10)
CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

# The notebook model is: Input(32,32,3) -> RandomFlip -> RandomTranslation
# -> Resizing(128) -> Lambda(resnet50.preprocess_input) -> ResNet50 -> softmax.
# Preprocessing lives inside the model, so we feed raw 0-255 pixels.

if not os.path.exists(MODEL_PATH):
    if not MODEL_URL:
        raise RuntimeError(
            f"Model not found at {MODEL_PATH}. Add the file next to main.py or set MODEL_URL."
        )
    print(f"Downloading model from {MODEL_URL} ...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False,
    safe_mode=False,
    custom_objects={"preprocess_input": tf.keras.applications.resnet50.preprocess_input},
)
IMG_H = model.input_shape[1] or 32
IMG_W = model.input_shape[2] or 32

@tf.function(reduce_retracing=True)
def infer(x):
    return model(x, training=False)


# Warm-up so the first real request is fast
infer(tf.zeros((1, IMG_H, IMG_W, 3), dtype=tf.float32))
print(f"Model ready. Input size: {IMG_H}x{IMG_W}")

app = FastAPI(title="CIFAR-10 ResNet50 Classifier")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


def preprocess(img: Image.Image) -> np.ndarray:
    img = ImageOps.exif_transpose(img).convert("RGB")
    if RESIZE_MODE == "crop":
        img = ImageOps.fit(img, (IMG_W, IMG_H), Image.LANCZOS)  # center-crop to square, then resize
    else:
        img = img.resize((IMG_W, IMG_H), Image.BOX)  # whole photo, area-averaged like CIFAR
    x = np.asarray(img, dtype="float32")  # raw 0-255
    return np.expand_dims(x, axis=0)


@app.get("/")
def index():
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload an image file (JPG, PNG, WebP).")
    try:
        img = Image.open(io.BytesIO(await file.read()))
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read that image.")

    x = preprocess(img)
    probs = infer(tf.constant(x)).numpy()[0]
    if TTA:
        flipped = np.ascontiguousarray(x[:, :, ::-1, :])
        probs = (probs + infer(tf.constant(flipped)).numpy()[0]) / 2
    top = np.argsort(probs)[::-1][:3]
    return {
        "label": CLASSES[int(top[0])],
        "confidence": float(probs[top[0]]),
        "top3": [{"label": CLASSES[int(i)], "confidence": float(probs[i])} for i in top],
    }
