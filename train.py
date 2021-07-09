import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
from sklearn.model_selection import train_test_split
import uuid

from preprocessor import Processor
from config import categorical_config, missing_val_config, model_config
from default_predictor import BaselineModel

#raw data ingestion
df = pd.read_csv("/Users/andreatamburri/Downloads/dataset.csv", delimiter = ";")
#TODO split the dataset in the two parts
df_train = df[df["default"].notnull()]
print(df_train.describe())
df_test = df[df["default"].isnull()]

#data preprocessing
processor = Processor(categorical_config=categorical_config, missing_val_config=missing_val_config)
#categorical and boolean preprocessing
df_train_encoded = processor.encode_labels(df_train)
print(df_train_encoded.head())
#replace missing values
df_train_prep = processor.prep_missing_val(df_train_encoded)
#standard scaling not in the scope of the case study, it will improve predictive performances (for numeric features), bnut is out of the scope

print(df_train_prep.head())

#save processor
processor.save_processor()

#training testing dataset prep
X = df_train_prep.drop(["default","uuid"],axis=1).astype(float)  #keras doesn't accept int
print(X.dtypes)
y = df_train_prep["default"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=63)

#model creation and training
model = BaselineModel(name = "default-predictor", uuid = uuid.uuid1())
input_shape = (X_train.shape[1])
print(input_shape)
model.build(input_shape=input_shape, config= model_config)
model.train(X_train, y_train, (X_test, y_test))
score = model.test(X_test, y_test)
print(score)