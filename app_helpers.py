import numpy as np
from flask import jsonify

from feature_config import FEATURES

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
    for feature in FEATURES:
        value = request_dict.get(feature, None)
        if value:
            x_list.append(value)
        else:
            # Handle missing features.
            x_list.append(np.nan)
            missing_data = True
            missing_list.append(feature)
    return x_list, missing_data, missing_list
