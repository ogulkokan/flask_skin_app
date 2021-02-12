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
from sklearn.preprocessing import PowerTransformer
from sklearn.decomposition import PCA



app = Flask(__name__)
app.secret_key = os.urandom(24)

# load models




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

        #print(request.form.get('age'))    ## this one check if age exist and return none if empyt
        print(request.form['diameter'])
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


            ##------------For image feature extraction test------------------#
            original_image = load_img(file_path, target_size=(224, 224))
            numpy_image = img_to_array(original_image)

            print(numpy_image.shape)
            image_batch = np.expand_dims(numpy_image, axis=0)

            #images = ...  # A batch of images with shape [batch_size, height, width, 3].
            module = hub.KerasLayer("./models/bit_s-r50x1_1")
            features = module(image_batch)  # Features with shape [batch_size, 2048]. 
            print(features)
            ##------------For image feature extraction test END------------------#

            ##------------Get Clinical Informations------------------#
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
            #prediction = model.predict(final_features)
            print(final_features)
            # load model
            rf_model =  pickle.load(open('./models/rf_model.pkl', 'rb'))
            prediction = rf_model.predict(final_features)

            print('############---------------------------#######################')
            print('Clinical feature prediction')
            print(prediction)
            print('############---------------------------#######################')


            ##------------Get Clinical Informations END------------------#
            
            image_features = features
            
            # scale image features
            yj = PowerTransformer(method = 'yeo-johnson')
            X_train_feature_yj = yj.fit_transform(training_features)

            # apply PCA
            pca = PCA(433)

            print('############----DENEME--------#######################')
            
            print(X_train_feature_yj.shape)
            X_train_feature_yj_pca = pca.fit_transform(X_train_feature_yj)
            X_train_feature_yj_pca_fusion = np.concatenate((X_train_feature_yj_pca, encoded_clinical_meta), axis=1)

            X_test_feature_yj = yj.transform(image_features)
            X_test_feature_yj_pca = pca.transform(X_test_feature_yj)
            #print(X_test_feature_yj_pca.shape)

            X_test_feature_yj_pca_fusion = np.concatenate((X_test_feature_yj_pca, final_features), axis=1)
            # load model
            svm_model =  pickle.load(open('./models/svm_model.pkl', 'rb'))
            prediction_svm = svm_model.predict(X_test_feature_yj_pca_fusion)
            print(X_test_feature_yj_pca_fusion.shape)
            print('svm_prediction', prediction_svm)

            print('############----DENEME--------#######################')


            print('############---------------------------#######################')
            print('SOFT VOTING TEST')

            def fit_multiple_estimators(classifiers, X_list, y, sample_weights = None):

                # Convert the labels `y` using LabelEncoder, because the predict method is using index-based pointers
                # which will be converted back to original data later.
                #le_ = LabelEncoder()
                #le_.fit(y)
                #transformed_y = le_.transform(y)
                
                
                # Fit all estimators with their respective feature arrays
                estimators_ = [clf.fit(X, y) if sample_weights is None else clf.fit(X, y, sample_weights) for clf, X, y in zip([clf for _, clf in classifiers], X_list, y)]

                return estimators_


            def predict_from_multiple_estimator(estimators, X_list, weights = None):

                # Predict 'soft' voting with probabilities

                pred1 = np.asarray([clf.predict_proba(X) for clf, X in zip(estimators, X_list)])
                pred2 = np.average(pred1, axis=0, weights=weights)
                pred = np.argmax(pred2, axis=1)

                # Convert integer predictions to original labels:
                return pred, pred2     #label_encoder.inverse_transform(pred)

            X_train_list = [X_train_feature_yj_pca_fusion, encoded_clinical_meta]
            y_list = [target_multiple, target_multiple]

            X_test_list = [X_test_feature_yj_pca_fusion, final_features]

            # Make sure the number of estimators here are equal to number of different feature datas
            classifiers = [('svc',  svm_model), ('rf', rf_model)]

            fitted_estimators = fit_multiple_estimators(classifiers, X_train_list, y_list, sample_weights=None)
            y_pred, y_pred_proba = predict_from_multiple_estimator(fitted_estimators, X_test_list)
            print(y_pred, y_pred_proba)
            print('############---------------------------#######################')


            """    disease= 0, cancer= 1
            class 0, 3, and 5 is disease
            class 1, 2, and 4 is cancer
            """
            if y_pred == 0 or 3 or 5:
                prediction = "Safe"
                color = "green"
                statement = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Integer efficitur velit eget tellus pharetra congue. \
                Sed pellentesque, est id porta lacinia, orci metus iaculis nibh, sit amet auctor eros tellus sit amet sapien. Pellentesque lacus \
                augue, vestibulum varius posuere at congue vel felis. Pellentesque non justo vel elit luctus sollicitudin. Curabitur eu porttitor \
                odio. Class aptent taciti sociosqu ad litora torquent per conubia nostra, per inceptos himenaeos. Aenean ullamcorper lacus ac \
                libero ultrices, non blandit ipsum vehicula. Suspendisse ultrices congue urna, nec finibus metus. Nullam ex ligula" 

            else:
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


            return render_template('results.html', result = [y_pred, y_pred_proba, prediction, statement, color])




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
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port, debug=True)
