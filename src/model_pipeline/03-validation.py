import pandas as pd
import joblib
from sklearn.model_selection import StratifiedKFold, cross_val_score

preprocess_path = "../../datasets/preprocess"

X_test_dataset = pd.read_csv(f"{preprocess_path}/X_test_dataset.csv")
y_test = pd.read_csv(f"{preprocess_path}/y_test_dataset.csv").values.ravel()
X_train_dataset = pd.read_csv(f"{preprocess_path}/X_train_dataset.csv")
y_train_dataset = pd.read_csv(f"{preprocess_path}/y_train_dataset.csv").values.ravel()

base_model = joblib.load("../../models/base_model.pkl")

# # cross validation: k-fold
# k_fold_cv = cross_val_score(base_model, X_train_dataset, y_train_dataset, cv=5)
# print("k fold score: ", k_fold_cv.mean() * 100)

# cross vallidation: stratified k-fold validation
strat_cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
cv_scores = cross_val_score(base_model, X_train_dataset, y_train_dataset, cv=strat_cv, scoring='accuracy')

print(f"Strat cv score: {cv_scores.mean() * 100}")

