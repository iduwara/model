import base64
import io

from PIL.Image import Image
from flask import Flask
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from flask import request
from keras_preprocessing.image import load_img

app = Flask(__name__)


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


    return response


if __name__ == '__main__':
    app.run(port=3000, debug=True)
