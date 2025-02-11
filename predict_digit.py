import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load the model
model = load_model("digit_recognition_model.h5")

# Function to predict a digit
def predict_digit(img_path):
    img = image.load_img(img_path, target_size=(28, 28), color_mode="grayscale")
    img_array = image.img_to_array(img).reshape(1, 28, 28, 1) / 255.0
    prediction = model.predict(img_array)
    return np.argmax(prediction)

# Predict digit from a sample image
img_path = "dataset/image1.png"  # Change this to your image path
digit = predict_digit(img_path)
print(f"Predicted Digit: {digit}")
