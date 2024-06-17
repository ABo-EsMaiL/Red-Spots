import tensorflow as tf
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
    
    model_json_path = 'models/Acc97.json'
    model_weights_path = 'models/Acc97.weights.h5'

    if not os.path.exists(model_json_path):
        download_file(urls['model_json_url'], model_json_path)

    if not os.path.exists(model_weights_path):
        download_file(urls['model_weights_url'], model_weights_path)

    with open(model_json_path, 'r') as json_file:
        model_json = json_file.read()
    model = tf.keras.models.model_from_json(model_json)

    model.load_weights(model_weights_path)
    
    return model

def predict_image(model, img):
    predictions = model.predict(img)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)
    return predicted_class, confidence
