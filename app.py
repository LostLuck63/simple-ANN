import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import tensorflow
from tensorflow.keras.models import load_model

app = Flask(__name__)
clf = load_model('test4.h5')
cv = pickle.load(open('scaler.pkl', 'rb'))


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
    vect=cv.transform(final_features)
    prediction = clf.predict(vect)

    

    return render_template('index.html', prediction_text='Employee Salary should be $ {}'.format(prediction))



if __name__ == "__main__":
    app.run(debug=True)