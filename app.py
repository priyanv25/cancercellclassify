# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 15:46:57 2020

@author: Admin
"""


import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    return render_template('result.html', prediction_text="CELL ANALYSIS --{}".format(prediction))

if __name__=="__main__":
    app.run(debug=True)