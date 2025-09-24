import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os


feature_df = pd.read_csv("../../datasets/feature_engg/feature_df.csv")

# split train/test datasets
X = feature_df.drop(columns=['airborne', 'feathers', 'domestic', 'aquatic', 'fins', 'catsize', 'class_type', 'class_name'])
y = feature_df['class_name']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# scaling on numerical data only
scaler = StandardScaler()

# save animal_names separately
animal_name_train = X_train['animal_name'].reset_index(drop=True)
animal_name_test = X_test['animal_name'].reset_index(drop=True)

# remove animal_names for scaling
X_train_nameless = X_train.drop(columns=['animal_name'])
X_test_nameless = X_test.drop(columns=['animal_name'])

X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_nameless), columns=X_train_nameless.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test_nameless), columns=X_test_nameless.columns)

# save train/test files separetly
output_dir = "../../datasets/preprocess/"
os.makedirs(output_dir, exist_ok=True)

X_train_scaled.to_csv(f"{output_dir}/X_train_dataset.csv", index_label=False)
X_test_scaled.to_csv(f"{output_dir}/X_test_dataset.csv", index_label=False)
y_train.to_csv(f"{output_dir}/y_train_dataset.csv", index_label=False)
y_test.to_csv(f"{output_dir}/y_test_dataset.csv", index_label=False)

# save preprocessed data in csv for later use.
# Add 'animal_name' back
X_train_scaled['animal_name'] = animal_name_train
X_test_scaled['animal_name'] = animal_name_test

# Save to CSV
preprocessing_df = pd.concat([X_train_scaled, X_test_scaled], axis=0).reset_index(drop=True)
print(preprocessing_df.head(3))
# save preprocessed csv file
preprocessing_df.to_csv(f'{output_dir}/preprocessing_df.csv', index=False)

# save feature names
feature_store_path = "./../../feature_store"
os.makedirs(feature_store_path, exist_ok=True)
joblib.dump(X_train_scaled.columns.to_list(), f'{feature_store_path}/feature_names.pkl')

# save scaler
utility_path = "./../../utility"
os.makedirs(utility_path, exist_ok=True)
joblib.dump(scaler, f'{utility_path}/scaler.pkl')

