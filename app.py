from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from tensorflow.keras.models import model_from_json
import numpy as np
import pandas as pd
import base64
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

# Load the model from JSON and weights
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model.h5")

# Helper function to convert base64 to image
def get_image_from_base64_string(b64str):
    try:
        encoded_data = b64str.split(',')[1]
        image_data = BytesIO(base64.b64decode(encoded_data))
        img = Image.open(image_data)
        return img
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error decoding image: {str(e)}")

# Pydantic model for input validation
class ImageInput(BaseModel):
    image: str

@app.get("/")
def home():
    return {"message": "Hello World"}

@app.post("/predict")
def predict(data: ImageInput):
    try:
        # Convert the base64 string to an image and resize
        img = get_image_from_base64_string(data.image)
        img = img.resize((224, 224))
        
        # Convert the image to a NumPy array
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # Prepare for prediction

        # Convert to DataFrame
        df = pd.DataFrame(img_array.reshape(-1, img_array.shape[-1]))

        # Make a prediction
        prediction = loaded_model.predict(img_array)
        result = np.argmax(prediction, axis=1)

        # Convert to JSON and return as string response
        json_response = {
            "prediction_probabilities": prediction.tolist(),
            "predicted_class": int(result[0])
        }
        return json_response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in prediction: {str(e)}")

# To run the server, use: uvicorn filename:app --reload
