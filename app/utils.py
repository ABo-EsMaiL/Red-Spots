from PIL import Image
import numpy as np
import tensorflow as tf

IMAGE_SIZE = 299

def preprocess_image(image_file):
    img = Image.open(image_file)
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array
