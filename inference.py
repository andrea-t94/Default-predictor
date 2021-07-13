import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

from preprocessor import Processor
from default_predictor import BaselineModel

#test data ingestion
df = pd.read_csv("dataset.csv", delimiter = ";")

#TODO split the dataset in the two parts
df_train = df[df["default"].notnull()]
df_test = df[df["default"].isnull()]
df_test.drop(["default","uuid"], axis=1, inplace=True)

# model selection
try:
    model = BaselineModel(name="default-predictor")
    model.load_best_model()
    print("Model successfully loaded, with a tested accuracy of {}".format(model.best_acc))
except:
    raise FileNotFoundError("No model found, please run the train.py before!")

# text processor load
try:
    processor = Processor()
    processor.load_processor()
    print("Processor successfully loaded")
except:
    raise FileNotFoundError("No encoder found, something went wrong with train.py")

#categorical and boolean preprocessing
df_test_encoded = processor.encode_labels(df_test)

#replace missing values
df_test_prep = processor.prep_missing_val(df_test_encoded)
X_test = df_test_prep.to_numpy()

#inference
prediction = model.class_prediction(X_test)
print(prediction)
#TODO write csv with predicted probabilities