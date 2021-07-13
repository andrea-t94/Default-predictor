import os
from flask import Flask, jsonify, request, url_for, redirect
import numpy as np
import pandas as pd

from feature_config import features
from default_predictor import BaselineModel
from preprocessor import Processor


def message(content, code):
    return jsonify({'message':content}), code

def parse_args(request_dict):
    """Parse model features from incoming requests formatted in
    JSON."""
    # Initialize missing_data as False.
    missing_data = False
    missing_list=[]
# Parse out the features from the request_dict.
    x_list = []
    for feature in features:
        value = request_dict.get(feature, None)
        if value:
            x_list.append(value)
        else:
            # Handle missing features.
            x_list.append(np.nan)
            missing_data = True
            missing_list.append(feature)
    return x_list, missing_data, missing_list

# model selection
try:
    model = BaselineModel(name="default-predictor")
    model.load_best_model()
except:
    raise FileNotFoundError("No model found, please run the train.py before!")

# text processor load
try:
    processor = Processor()
    processor.load_processor()
except:
    raise FileNotFoundError("No encoder found, something went wrong with train.py")

app = Flask(__name__)
app.debug = False
app.config['JWT_KEY'] = os.getenv('JWT_KEY') or 'secret'


@app.route("/api/detection", methods=['POST'])
def detection():
    if request.method == 'POST':
        if not request.json:
            return message("POST request does not have the JSON part!", 400)

        #create dataframe from JSON
        x_list, missing_data, missing_list = parse_args(request.json)
        input_df = pd.DataFrame([x_list],columns=features)

        # categorical and boolean preprocessing
        df_test_encoded = processor.encode_labels(input_df)
        # replace missing values
        df_test_prep = processor.prep_missing_val(df_test_encoded)
        X_test = df_test_prep.astype(float).to_numpy()  #keras doesn't accept int

        # inference
        prediction = model.class_prediction(X_test)
        print(missing_list)
        response = dict(ESTIMATE=str(prediction[0][0]), MISSING_DATA=missing_data)
        return message(response, 422)

    else:
        return redirect(url_for("/"))


@app.route("/", methods=['GET'])
def default():
    return message(
        "Welcome to Andrea Tamburri default predictor. Send a POST request with a JSON file containing all the features TBD",
        200)


if __name__ == "__main__":
    app.run()
