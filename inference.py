import pandas as pd

from preprocessor import Processor
from default_predictor import BaselineModel

#test data ingestion
df = pd.read_csv("dataset.csv", delimiter = ";")
df_test = df[df["default"].isnull()]
df_uuid = df_test[["uuid"]]
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

#write output
df_prediction = df_uuid.copy()
df_prediction["prediction"] = prediction
df_prediction.to_csv("output_inference/predictions.csv", sep=";", index=False)