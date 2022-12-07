from __future__ import division, print_function
from ctypes import resize
# coding=utf-8
import sys
import os
import glob
import re
from grpc import protos_and_services
import numpy as np
from tensorflow.keras.utils import load_img, img_to_array 
import keras.utils as image
# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model


# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'models/finalhairimageclassifier.h5'

# Load your trained model
model = load_model(MODEL_PATH)
#model._make_predict_function()        # Necessary
# print('Model loaded. Start serving...')


# You can also use pretrained model from Keras
# Check https://keras.io/applications/
#from keras.applications.resnet50 import ResNet50
#model = ResNet50(weights='imagenet')
#model.save('')
print('Model loaded. Check http://127.0.0.1:5000/')


# def model_predict(img_path, model):
#     img = image.load_img(img_path, target_size=(256, 256))

#     # Preprocessing the image
#     x = image.img_to_array(img)
#     # x = np.true_divide(x, 255)
#     x = np.expand_dims(x, axis=0)

#     # Be careful how your trained model deals with the input
#     # otherwise, it won't make correct prediction!
#     x = preprocess_input(x, mode='caffe')

#     preds = model.predict(x)
#     return preds

def model_predict(fname,model):
    """returns top 5 categories for an image.

    :param fname : path to the file 
    """
    # ResNet50 is trained on color images with 224x224 pixels
    input_shape = (256, 256, 3)

    # load and resize image ----------------------

    img = image.load_img(fname, target_size=input_shape[:2])
    x = image.img_to_array(img)

    # preprocess image ---------------------------

    # make a batch
    import numpy as np
    x = np.expand_dims(x, axis=0)
    print(x.shape)

    # apply the preprocessing function of resnet50
    # img_array = resnet50.preprocess_input(x)

    # model = resnet50.ResNet50(weights='imagenet',
    #                           input_shape=input_shape)
    preds = model.predict(x)
    return preds

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        print(f'Result: {preds:.2f}')
        # Process your result for human
        pred_class = preds.argmax(axis=-1) 
                   # Simple argmax
        #pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
        result = str(pred_class)
        print(result)               # Convert to string
        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)