import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
import numpy as np


model = MobileNetV2(weights='imagenet')

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

def predict_image(img):
    preds = model.predict(img)
    decoded_preds = decode_predictions(preds, top=3)[0]
    return decoded_preds

def print_predictions(predictions):
    for _, label, prob in predictions:
        print(f'{label}: {prob * 100:.2f}% confidence')

# You need to provide the path to your own image, just remember to use the escape sequence or raw string
img_path = r"D:\Programming\Python\Artificial Intelligence\Animals\Train\Carnivores\fox - Copy.jpg"
preprocessed_img = preprocess_image(img_path)

# Make predictions
predictions = predict_image(preprocessed_img)

# Print the top three predictions
print_predictions(predictions)
