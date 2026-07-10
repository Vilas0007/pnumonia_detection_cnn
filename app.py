from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Create uploads folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load trained model
model = load_model("pneumonia_model.keras")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    if "image" not in request.files:
        return "No file uploaded"

    file = request.files["image"]

    if file.filename == "":
        return "No file selected"

    # Secure filename
    filename = secure_filename(file.filename)

    # Save uploaded file
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    # Image preprocessing
    image = Image.open(filepath).convert("RGB")
    image = image.resize((64, 64))
    image = np.array(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)

    # Prediction
    prediction = model.predict(image)

    probability = prediction[0][0]

    if probability > 0.5:
        result = "PNEUMONIA"
        confidence = probability * 100
    else:
        result = "NORMAL"
        confidence = (1 - probability) * 100

    return render_template(
        "index.html",
        prediction=result,
        confidence=round(confidence, 2),
        image=filename
    )


if __name__ == "__main__":
    app.run(debug=True)