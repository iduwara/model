import os

from flask import Flask
# import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from flask import request
from keras_preprocessing.image import load_img
# from flask_cors import CORS, cross_origin
import PIL
import tensorflow as tf
# from tensorflow.keras.preprocessing.image import load_img, img_to_array
# from tensorflow.keras.models import load_model
import numpy as np

app = Flask(__name__)
# cors = CORS(app)
# app.config['CORS_HEADERS'] = 'Content-Type'


def get_model():
    global model
    model = load_model('train_model.h5')
    print("Model Loaded")


def preprocess_image(image, target_size):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    return image


print(" Loading keras model")
get_model()


@app.route('/predict', methods=['POST'])
# @cross_origin()
def predict():
    imagefile = request.files['imageFile']
    imagepath = "./images/" + imagefile.filename
    imagefile.save(imagepath)

    processed_image = preprocess_image(load_img(imagepath), target_size=(180, 180))

    prediction = model.predict(processed_image).tolist()
    percentage = abs(prediction[0][0] + prediction[0][1])
    response = {
        'predictionPercentage': percentage
    }

    os.remove(imagepath)

    return response


if __name__ == '__main__':
    app.run(port=3000, debug=True)
