import numpy as np
from flask import Flask, request, jsonify, render_template
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

app = Flask(__name__)

#model = pickle.load(open('./models/model.pkl', 'rb'))

module = hub.KerasLayer("./models/bit_s-r50x1_1")

print('Hub is ready')

#module = hub.KerasLayer("https://tfhub.dev/google/bit/s-r50x1/1")
#images = ...  # A batch of images with shape [batch_size, height, width, 3].
#features = module(images)  # Features with shape [batch_size, 2048].


@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('index.html')


@app.route('/about')
def about():
    return render_template('about.html')


# route and function to handle the upload page
@app.route('/test', methods=['GET', 'POST'])
def test():
    if request.method == 'POST':
        # check if there is a file in the request
        if 'file' not in request.files:
            return render_template('test.html', msg='No file selected')
        file = request.files['file']
        # if no file is selected
        if file.filename == '':
            return render_template('test.html', msg='No file selected')
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(path)

            # text input:
            text = request.form['text']
            processed_text = text
            print(text)


            # perform analysis on it
    return render_template("test.html")




@app.route('/predict',methods=['GET','POST'])
def predict():
    """    disease= 0, cancer= 1
    class 0, 3, and 5 is disease
    class 1, 2, and 4 is cancer
    """
    if request.method == 'POST':
        print('YEEEEEEEEEEEEEEEEEEEEEEEEES')
        ##------------For image upload test------------------#
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        print('model saved to uploads file')
        ##------- for image upload test END-------------------#


        ##------------For image feature extraction test------------------#
        original_image = load_img(file_path, target_size=(224, 224))
        numpy_image = img_to_array(original_image)

        print(numpy_image.shape)
        image_batch = np.expand_dims(numpy_image, axis=0)

        #images = ...  # A batch of images with shape [batch_size, height, width, 3].
        features = module(image_batch)  # Features with shape [batch_size, 2048]. 
        print(features)
        ##------------For image feature extraction test END------------------#
        
        int_features = [int(x) for x in request.form.values()]
        final_features = [np.array(int_features)]
        prediction = model.predict(final_features)
        print(prediction)

        return 'Feature seti aldik hayirlisiylan'

        #print(prediction)
        #prediction = model.predict(final_features)

        #output = round(prediction[0])

        #if output is 0 or output is 3 or output is 5:
         #   output2 = 'Low Risk Lesion'
        #else:
         #   output2 = 'High Risk Lesion'


        ##--------her sey bittikten sonra fotoyu kaldir ------------###
        #print('Deleting File at Path: ' + file_path)
        #os.remove(file_path)
        #print('Deleting File at Path - Success - ')

        #return render_template('results.html', prediction_text2=output2)
    else:
        print('NOOOOOOOOOOOOOOOOOOOOOOOOO')

    


@app.route('/results',methods=['GET','POST'])
def results():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)
