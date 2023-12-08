# Using flask to make an api 
# import necessary libraries and functions 
from flask import Flask, jsonify, request 

import numpy as np
import pickle
from flask_cors import CORS
import json

# creating a Flask app 
app = Flask(__name__) 
CORS(app)

@app.route('/predict/', methods = ['POST']) 
def predict():
	request_json = json.loads(request.get_json()["data"])
	model_pickle = request_json.get("model")
	x_test = request_json.get("x_test")
	coords_test = request_json.get("coords_test")
	pickle_file = open(model_pickle, 'rb')
	loaded_model = pickle.load(pickle_file)
	x_test = np.array([np.array(x_test)])
	coords_test = np.array([np.array(coords_test)])
	pred = loaded_model.predict(x_test, coords_test)
	return jsonify({'data': float(np.round(pred[0][0][0]))}) 


# driver function 
if __name__ == '__main__': 
	app.run(port=8888, debug = True) 
