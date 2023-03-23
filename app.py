import pickle
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd
import sklearn


app = Flask(__name__)
## Load the model
xgbmodel = pickle.load(open('xgbmodel.pkl','rb'))
scaler = pickle.load(open('scaler.pkl','rb'))
 
@app.route('/')
def home():
    return render_template('home.html')

# @app.route('/predict_api_1',methods=['POST'])
# def predict_api():
#     data = request.json['data']
#     print(data)
#     print(np.array([float(x) for x in data.values()]).reshape(1,-1))
#     new_data = scaler.transform(np.array([float(x) for x in data.values()]).reshape(1,-1))
#     output = xgbmodel.predict(new_data)
#     print(output[0])

#     return jsonify(output[0])
@app.route('/predict',methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    final_input = scaler.transform(np.array(data).reshape(1,-1))
    print(final_input)
    output = xgbmodel.predict(final_input)[0]
    if output==0:
        output_text = "No Rain"
    else:
        output_text = "Rain"
    return render_template("home.html",prediction_text=output_text)
# @app.route('/trial')
# def trial():
#     return render_template('trial.html')

if __name__ == "__main__":
    app.run(debug=True)