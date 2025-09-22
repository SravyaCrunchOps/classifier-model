import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

os.makedirs("../feature_store", exist_ok=True)
FEATURE_STORE_PATH = "../feature_store"


def preprocessing(df):
    X = df.drop(columns=['airborne', 'feathers', 'domestic', 'aquatic', 'fins', 'catsize', 'animal_name', 'class_type', 'class_name'])
    y = df['class_name']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    df.to_csv(f'{FEATURE_STORE_PATH}/preprocessed_data.csv', index=False)

    feature_names = X.columns.to_list()
    joblib.dump(feature_names, f'{FEATURE_STORE_PATH}/feature_names.pkl')

    return X_train, X_test, y_train, y_test