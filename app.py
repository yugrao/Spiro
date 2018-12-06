#!flask/bin/python
from flask import Flask, jsonify
from flask import abort
from flask import request
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from keras.models import load_model
import random as r
import pickle as p
import numpy as np
from PIL import Image
import logging
import json
import matplotlib.pyplot as plt

app = Flask(__name__)

data = [
    {
        'id': 1,
        'data': [1,2,3,4,5],
        'output':3
    },
    {
        'id': 2,
        'data': [6,7,8,9,10],
        'output':8
    }
]

@app.route('/spiro/data', methods=['GET'])
def get_spiro_data():
    return jsonify({'data': data})

@app.route('/spiro/data/<int:data_id>', methods=['GET'])
def get_spiro_data2(data_id):
    d = [d for d in data if d['id'] == int(data_id)]
    if len(d) == 0:
        abort(404)
    return jsonify({'data': d[0]})

def predict_data(a):
    model = load_model('Spiral_Model-96.88-1021094522.h5')
    x, y = np.array(a).T
    x = list(map(lambda x_i:(x_i-480)*2/3, x))
    y = list(map(lambda y_i:(y_i-480)*2/3, y))
    plt.plot(x, y, "ko-", linewidth=5, markersize=0)
    plt.axis([-300, 300, 300, -300])
    plt.axis("off")
    fig = plt.gcf()
    fig.set_size_inches(5,5)
    fig.savefig("thisisarequestimg.png", dpi=25)
    plt.clf()
    img_a = np.asarray(Image.open("thisisarequestimg.png"))
    img_a = [list(map(lambda rgbx: (rgbx[0] * 299.0/1000 + rgbx[1] * 587.0/1000 + rgbx[2] * 114.0/1000)/255, row)) for row in img_a]
    logging.warning(np.array([np.array(img_a)]).shape)
    logging.warning(np.array([np.array([np.array(img_a)])]).shape)
    return model.predict(np.array([np.array([np.array(img_a)])]))

@app.route('/spiro/data', methods=['POST'])
def add_data():
    if not request.form:
        abort(400)
    pt_list = [[float(coord.strip("( )")) for coord in pt.split(",")] for pt in request.form.getlist("data[]")]
    new_data = {
        'id': data[-1]['id'] + 1,
        'data': pt_list
    }
    logging.warning(new_data['data'])
    prediction = predict_data(new_data['data'])
    new_data['output'] = prediction
    data.append(new_data)
    return str(prediction), 201

@app.route('/')
def index():
    return "Hey rasvik!"

if __name__ == '__main__':
    app.run(debug=True)
