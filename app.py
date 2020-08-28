import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('index.html')


@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
    """    disease= 0, cancer= 1
    class 0, 3, and 5 is disease
    class 1, 2, and 4 is cancer
    """
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    #print(prediction)
    #prediction = model.predict(final_features)

    output = round(prediction[0])

    if output is 0 or output is 3 or output is 5:
        output2 = 'Low Risk Lesion'
    else:
        output2 = 'High Risk Lesion'

    return render_template('results.html', prediction_text2=output2)

@app.route('/results',methods=['GET','POST'])
def results():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)
