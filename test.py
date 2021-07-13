import pandas as pd

from preprocessor import Processor

df = pd.read_csv("/Users/andreatamburri/Downloads/dataset.csv", delimiter = ";")

#TODO split the dataset in the two parts
df_train = df[df["default"].notnull()]
df_test = df[df["default"].isnull()]
print(df_test.head())
df_test.drop(["default"], axis=1, inplace=True)

#load trained processor
processor = Processor()
processor.load_processor()
#categorical and boolean preprocessing
df_test_encoded = processor.encode_labels(df_test)


#replace missing values
df_test_prep = processor.prep_missing_val(df_test_encoded)
print(df_test_prep.to_numpy().shape)