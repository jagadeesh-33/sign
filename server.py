from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from PIL import Image
from io import BytesIO
import math

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = tf.keras.models.load_model("mini_project (1).h5")
CLASS_NAMES = ['Forgery','Real']
def read_file_as_image(data) -> np.ndarray:
    img = Image.open(BytesIO(data)).convert('RGB')
    img = img.resize((224, 224))
    image = np.array(img)
    return image
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, 0)
    predictions = np.array(model.predict(img_batch))
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    return{
        'class': predicted_class,
        'confidence': float(confidence)
    }
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host='localhost', port=8000)