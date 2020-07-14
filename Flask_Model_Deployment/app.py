from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    '''
    for rendering results on HTML GUI
    '''
    float_features = [float(x) for x in request.form.values()]
    final_features = [np.array(float_features)]
    prediction = model.predict(final_features)

    if prediction == 0:
        prediction_text = 'Player 1 Wins'
    elif prediction == 1:
        prediction_text = 'Player 2 Wins'

    return render_template('index.html', prediction_text=prediction_text)

#@app.route('/predict_api', methods=['POST'])
#def predict_api():
 #   '''
  #  for direct API calls
   # '''
    #data = request.get_json(force=True)
    #prediction = model.predict([np.array(list(data.values()))])

    #output = prediction[0]
    #return jsonify(output)


app.run('127.0.0.1', debug=True)