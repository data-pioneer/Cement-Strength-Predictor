import pickle
from flask import Flask, render_template,request


app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))
scaler = pickle.load(open('scaler.pkl','rb'))

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods = ['GET','POST'])
def predict():
    input_features = [request.form.get('Material_Quality'), request.form.get('Additive_catalyst'), request.form.get('Ash_Component'), request.form.get('Water_Mix'), request.form.get('Plasticizer'), request.form.get('Moderate_Aggregator'), request.form.get('Refined_Aggregator'), request.form.get('Formulation_Duration')]
   
   
    scaled_features = scaler.transform([input_features])

    predictedOutput= model.predict(scaled_features)[0]
    roundedOutput = round(predictedOutput, 2)
    return render_template('index.html', predictedOutput = f'Predicted Compression Strength of the new cement batch is : {roundedOutput}')


if __name__ == '__main__':
    app.run(debug=True)
    