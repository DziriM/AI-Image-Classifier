from fastapi import FastAPI, UploadFile, File
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import io
from PIL import Image

app = FastAPI()
model = ResNet50(weights="imagenet")

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents))
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    preds = model.predict(img_array)
    result = decode_predictions(preds, top=3)[0]

    return {"predictions": [{ "label": label, "probability": float(prob) } for (_, label, prob) in result]}
