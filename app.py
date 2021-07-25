import numpy as np
from flask import Flask, request, jsonify, render_template, flash
import pickle
import os

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
#from gevent.pywsgi import WSGIServer

#import tensorflow as tf
from PIL import Image
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import tensorflow_hub as hub

import sys

from config import *

import pandas as pd

app = Flask(__name__)
app.secret_key = os.urandom(24)

# load models
module = hub.KerasLayer("./models/bit_s-r50x1_1")
svm_model =  pickle.load(open('./models/model_svc_feature_2021.pkl', 'rb'))
rf_model =  pickle.load(open('./models/model_rf_meta_2021.pkl', 'rb'))

# load image feature dataset
base_dir = './models/'
features_PAD = pd.read_csv(os.path.join(base_dir, 'feature_last.csv'), index_col=0)
df_meta = pd.read_csv(os.path.join(base_dir, 'meta_last.csv'), index_col=0)

training_features = features_PAD.drop(['labels','diagnostic', 'type'],axis=1)
encoded_clinical_meta = df_meta.drop(['diagnostic', 'type'],axis=1)


target_multiple = features_PAD['diagnostic'].values
target_binary = features_PAD['type'].values


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            print('Not a valid Code', file=sys.stderr)
            return render_template('index.html', msg='No file selected')
        else:
            f = request.files['file']
            # Save the file to ./uploads
            basepath = os.path.dirname(__file__)
            file_path = os.path.join(
                basepath, 'uploads', secure_filename(f.filename))
            f.save(file_path)
            print('model saved to uploads file')

            ##------------For image feature extraction test------------------
            original_image = load_img(file_path, target_size=(224, 224))
            numpy_image = img_to_array(original_image)

            # print(numpy_image.shape)
            image_batch = np.expand_dims(numpy_image, axis=0)

            #images = ...  # A batch of images with shape [batch_size, height, width, 3].
            image_features = module(image_batch)  # Features with shape [batch_size, 2048]. 

            prediction_image_features = svm_model.predict_proba(image_features)

            print("Feature prediction:", prediction_image_features)


            ##------------Get Clinical Informations------------------
            age = request.form['age']
            gender = request.form['gender']
            h_diameter = request.form['diameter'] # horizontal diameter
            v_diameter = request.form['diamater2'] # vertical diameter
            smoke = request.form['smoke']
            alcohol = request.form['alcohol']
            cancer1 = request.form['cancer1'] # Skin cancer history
            cancer2 = request.form['cancer2'] # Cancer history
            fitspatrick = request.form['fitspatrick']
            itching = request.form['itching']
            hurting = request.form['hurting']
            growing = request.form['growing']
            changing = request.form['changing']
            bleeding = request.form['bleeding']
            elevation = request.form['elevation']

            int_features = [age, gender, h_diameter, v_diameter, smoke, alcohol, cancer1, cancer2, fitspatrick,
                            itching, hurting, growing, changing, bleeding, elevation]
            final_features = [np.array(int_features)]
  
            prediction_meta_features = rf_model.predict_proba(final_features)

            print('Clinical meta data feature prediction', prediction_meta_features)
            
            ##------------Soft Voting --------------------------------
            prediction_image_features = np.asarray(prediction_image_features)
            prediction_meta_features_np_array = np.asarray(prediction_meta_features)
            
            soft_voting_proba = np.mean(np.array([ prediction_image_features, prediction_meta_features_np_array ]), axis=0 )
            soft_voting_class = np.argmax(soft_voting_proba, axis=1)
            print("FINAL_RESULT", soft_voting_class)

            final_result = int(soft_voting_class[0])

            """    disease= 0, cancer= 1
            class 0, 3, and 5 is disease
            class 1, 2, and 4 is cancer
            """
            if final_result in (0, 3, 5):
                prediction = "Safe"
                color = "green"
                statement = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Integer efficitur velit eget tellus pharetra congue. \
                Sed pellentesque, est id porta lacinia, orci metus iaculis nibh, sit amet auctor eros tellus sit amet sapien. Pellentesque lacus \
                augue, vestibulum varius posuere at congue vel felis. Pellentesque non justo vel elit luctus sollicitudin. Curabitur eu porttitor \
                odio. Class aptent taciti sociosqu ad litora torquent per conubia nostra, per inceptos himenaeos. Aenean ullamcorper lacus ac \
                libero ultrices, non blandit ipsum vehicula. Suspendisse ultrices congue urna, nec finibus metus. Nullam ex ligula" 

            elif final_result is 1 or 2 or 4:
                prediction = "Dangerous"
                color = "red"
                statement = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Integer efficitur velit eget tellus pharetra congue. \
                Sed pellentesque, est id porta lacinia, orci metus iaculis nibh, sit amet auctor eros tellus sit amet sapien. Pellentesque lacus \
                augue, vestibulum varius posuere at congue vel felis. Pellentesque non justo vel elit luctus sollicitudin. Curabitur eu porttitor \
                odio. Class aptent taciti sociosqu ad litora torquent per conubia nostra, per inceptos himenaeos. Aenean ullamcorper lacus ac \
                libero ultrices, non blandit ipsum vehicula. Suspendisse ultrices congue urna, nec finibus metus. Nullam ex ligula" 

            # Remove uploaded image after prediction
            print('Deleting File at Path: ' + file_path)
            os.remove(file_path)
            print('Deleting File at Path - Success - ')

            return render_template('results.html', outputs = [final_result, prediction, statement, color])


    return render_template('index.html')


@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/results')
def results():
    return render_template('results.html')

@app.route('/login')
def login():
    return render_template('login.html')



if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host='0.0.0.0', port=port, debug=True, threaded=True)
