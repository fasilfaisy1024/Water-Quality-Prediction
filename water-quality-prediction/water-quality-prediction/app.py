import json
from flask import Flask, render_template ,jsonify
import os
from flask import request

import numpy as np
import pickle

from datetime import datetime 

project_dir = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)

loaded_model = pickle.load(open('randforest.sav', 'rb'))



def isfloat(num):
    try:
        float(num)
        return True
    except ValueError:
        return False




@app.route('/', methods=['GET'])
def home():
    return render_template("index.html")




@app.route("/predict", methods = ["POST"])
def predict():
    # [[7.808856, 193.553212,17329.80216,8.061362,333.775777,392.449580,19.903225,66.396293,2.798243	]]
    float_features = []
    for x in request.form.values():
        if not isfloat(x):
            response = {"status" : 500,"status_msg": "Some fields are empty !"}
            return jsonify(response)

        float_features.append(float(x))

    input_elems = []
    input_elems += [float_features]

    res = loaded_model.predict(input_elems)[0]


    response = {"status" : 200,"status_msg": "Water is Contaminated 👎🏻"}

    if res == 1:
        response = {"status" : 200,"status_msg": "Water is safe to drink 👍🏻"}

    return jsonify(response)


if __name__ == '__main__':
    app.run(debug=True)