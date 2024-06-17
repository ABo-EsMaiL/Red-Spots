import tensorflow as tf
from tensorflow.keras.models import load_model
import gdown
import json
import os

# Disable TensorFlow warnings about missing CUDA drivers and TensorRT
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

class_names = ['Abnormal', 'Erythrodermic', 'Guttate', 'Inverse', 'Nail', 'Normal', 
               'Not Define', 'Palm Soles', 'Plaque', 'Psoriatic Arthritis', 'Pustular', 'Scalp']

def download_file(url, output):
    gdown.download(url, output, quiet=False)

def load_model():
    with open('models/model_urls.json', 'r') as f:
        urls = json.load(f)
    
    modelkeras = 'models/Trying-96.keras'

    if not os.path.exists(modelkeras):
        download_file(urls['modelkeras'], modelkeras)

    model = load_model(modelkeras,safe_mode=False)
    
    return model

def predict_image(model, img):
    predictions = model.predict(img)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)
    return predicted_class, confidence

