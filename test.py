import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

from preprocessor import Processor
from config import categorical_config, missing_val_config

df = pd.read_csv("/Users/andreatamburri/Downloads/dataset.csv", delimiter = ";")

#TODO split the dataset in the two parts
df_train = df[df["default"].notnull()]
df_test = df[df["default"].isnull()]
print(df_test.head())
df_test.drop(["default"], axis=1, inplace=True)



processor = Processor(categorical_config=categorical_config, missing_val_config=missing_val_config)
processor.load_processor()
#categorical and boolean preprocessing
df_test_encoded = processor.encode_labels(df_test)


#replace missing values
df_test_prep = processor.prep_missing_val(df_test_encoded)
print(df_test_prep.to_numpy().shape)