import pandas as pd

from preprocessor import Processor

#raw data ingestion
df = pd.read_csv("dataset.csv", delimiter = ";")
df_train = df[df["default"].notnull()]
df_test = df[df["default"].isnull()]
#data preprocessing
processor = Processor()
#categorical and boolean preprocessing
df_train_encoded = processor.encode_labels(df_train)
#replace missing values
df_train_prep = processor.prep_missing_val(df_train_encoded)
#standard scaling not in the scope of the case study, it will improve predictive performances (for numeric features), but is out of the scope
df_train_prep.to_csv("prep_dataset.csv", sep=";", index=False)


#categorical and boolean preprocessing
df_test_encoded = processor.encode_labels(df_test)
#replace missing values
df_test_prep = processor.prep_missing_val(df_test_encoded)
#standard scaling not in the scope of the case study, it will improve predictive performances (for numeric features), but is out of the scope
df_test_prep.to_csv("prep_dataset_test.csv", sep=";", index=False)
