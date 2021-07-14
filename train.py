import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
from sklearn.model_selection import StratifiedKFold
import uuid

from preprocessor import Processor
from model_config import model_config
from default_predictor import BaselineModel

#raw data ingestion
df = pd.read_csv("dataset.csv", delimiter = ";")
df_train = df[df["default"].notnull()]

#data preprocessing
processor = Processor()
#categorical and boolean preprocessing
df_train_encoded = processor.encode_labels(df_train)
#replace missing values
df_train_prep = processor.prep_missing_val(df_train_encoded)
#standard scaling not in the scope of the case study, it will improve predictive performances (for numeric features), but is out of the scope

#save processor
processor.save_processor()

#training testing dataset prep
X = df_train_prep.drop(["default","uuid"],axis=1).astype(float).to_numpy()  #keras doesn't accept int
y = df_train_prep["default"].to_numpy()

#model creation and training with cross-validation
skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
fold_n = 5
accuracies = []
aurocs = []
unique_model_uuid =  uuid.uuid1() #this guarantees that we consider the model unique for all the folds
for train_index, test_index in skf.split(X, y):
    print("KFold cross validation number {}".format(fold_n))
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    model = BaselineModel(name = "default-predictor", uuid = unique_model_uuid)
    model.build(input_shape=X_train.shape[1], config=model_config)
    model.train(X_train, y_train, (X_test, y_test))
    score = model.test(X_test, y_test)
    accuracies.append(score["Accuracy %"])
    aurocs.append(score["AUROC %"])
    fold_n += 1

average_accuracy = sum(accuracies)/len(accuracies)
average_auroc = sum(aurocs)/len(aurocs)
print("average accuracy {}".format(average_accuracy))
print("average auroc {}".format(average_auroc))