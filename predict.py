from flask import Flask, jsonify, request
import pickle
import pandas as pd

app = Flask(__name__)

CLASS = ['D', 'C', 'B', 'A']
# Load the DictVectorizer and model
with open('body_performance_model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)

def predict(df, dv, model):
    dicts = df.to_dict(orient="records")
    X = dv.transform(dicts)
    y_pred = model.predict(X)
    return y_pred

@app.route("/")
def hello_world():
    return "<p>Go to /predict</p>"

@app.route("/predict", methods=['GET', 'POST'])
def body_inference():
    if request.method == 'POST':
        data = request.json  # Get JSON data from request
        if not data:
            return jsonify({'code': 400, 'status': 'Bad Request', 'message': 'No input data provided'}), 400

        # Make prediction using the predict function
        y_pred = predict(pd.DataFrame(data, index=[0]), dv, model)

        # Build response
        response = {
            'code': 200,
            'status': 'OK',
            'result': {
                'class': y_pred[0]
            }
        }
        return jsonify(response)
        
    return "Please use POST method to access the body performance model."

if __name__ == "__main__":
    app.run(debug=True)
