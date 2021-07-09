import pandas as pd
from typing import Optional, Dict
from joblib import dump, load
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
import numpy as np
import warnings



class Processor:
    '''
    Preprocess the dataset, dealing  with categorical and missing data, accordingly to the configurations
    that resemble the data structure
    '''

    def __init__(self,
                 categorical_config: Optional[Dict[str,str]] = None,
                 missing_val_config: Optional[Dict[str,str]] = None
                 ):
        if categorical_config:
            self.categorical_config = categorical_config
            self.bool_to_cast = [key for key, val in self.categorical_config.items() if val == "bool"]
            self.cat_to_transform = [key for key, val in self.categorical_config.items() if val == "categorical"]
        else:
            warnings.warn("Missing configuration for pre-processing categorical values, this method won't work")

        if missing_val_config:
            self.missing_val_config = missing_val_config
            self.features_replace_zero = [key for key, val in self.missing_val_config.items() if val == "zero"]
            self.features_replace_mean = [key for key, val in self.missing_val_config.items() if val == "mean"]
        else:
            warnings.warn("Missing configuration for pre-processing missing values, this method won't work")

    def prep_missing_val(self, input: pd.DataFrame):
        if hasattr(self, 'imputer'):
            output = input.copy()
            output[self.features_replace_zero] = output[self.features_replace_zero].fillna(0)
            # replace with mean
            output[self.features_replace_mean] = self.imputer.transform(output[self.features_replace_mean].values)
        else:
            #build from scratch
            self.imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
            #replace with zero
            output = input.copy()
            output[self.features_replace_zero] = output[self.features_replace_zero].fillna(0)
            #replace with mean
            output[self.features_replace_mean] = self.imputer.fit_transform(output[self.features_replace_mean].values)

        return output

    def encode_labels(self, input: pd.DataFrame):
        #encode categorical label
        if hasattr(self, 'encoder'):
            output = input.copy()
            output[self.cat_to_transform] = self.encoder.transform(input[self.cat_to_transform])
            output[self.bool_to_cast] = output[self.bool_to_cast].astype(int)
        else:
            #build from scratch
            #when we find new class, we find the 99 as integer, useful to detect when new classes are put
            # One hot encoding in better in terms of performance predictive, becasue it give no ordinal value, but it's more
            # difficult to implement, since require extra memory to store, reducing computational performance, and it's worst to deal with new values
            self.encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=99)

            # categorical and boolean preprocessing
            output = input.copy()
            output[self.bool_to_cast] = output[self.bool_to_cast].astype(int)
            output[self.cat_to_transform] = self.encoder.fit_transform(
            output[self.cat_to_transform])

        return output

    def decode_labels(self, input):
        #decode int label, e.g. output of model into genre_name
        if hasattr(self, 'encoder'):
            output = input.copy()
            output[self.cat_to_transform] = self.encoder.inverse_transform(input[self.cat_to_transform])
        else:
            raise ValueError("No loaded encoder found, please encode labels before decoding")
        return output

    def save_processor(self, filepath: Optional[str] = 'artifacts/'):
        if hasattr(self, 'imputer'):
            with open(filepath + 'imputer.pkl', 'wb') as fw:
                dump(self.imputer, fw)
        else:
            warnings.warn("No imputer fitted during pre-processing, nothing to save!")
        if hasattr(self, 'encoder'):
            with open(filepath + 'encoder.pkl', 'wb') as fw:
                dump(self.encoder, fw)
        else:
            warnings.warn("No encoder fitted during pre-processing, nothing to save!")

    def load_processor(self, filepath: Optional[str] = 'artifacts/'):
        if self.categorical_config:
            self.encoder = load(filepath + "encoder.pkl")
        if self.missing_val_config:
            self.imputer = load(filepath + "imputer.pkl")

        return self.encoder, self.imputer






