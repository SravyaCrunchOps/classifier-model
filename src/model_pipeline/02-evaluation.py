from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import joblib

preprocess_path = "../../datasets/preprocess"

X_test_dataset = pd.read_csv(f"{preprocess_path}/X_test_dataset.csv")
y_test = pd.read_csv(f"{preprocess_path}/y_test_dataset.csv").values.ravel()

X_train_dataset = pd.read_csv(f"{preprocess_path}/X_train_dataset.csv")
y_train_dataset = pd.read_csv(f"{preprocess_path}/y_train_dataset.csv").values.ravel()

base_model = joblib.load("../../models/base_model.pkl")

# predict
y_pred = base_model.predict(X_test_dataset)

# evaluation
accuracy = accuracy_score(y_test, y_pred) # 100%
print('accuracy: ', accuracy * 100)

cr = classification_report(y_test, y_pred)
# print("calssification report: ", cr)

cm = confusion_matrix(y_test, y_pred)
# print("confusion matrix report: ", cm)

# train/test scores
train_score = base_model.score(X_train_dataset, y_train_dataset)
test_score = base_model.score(X_test_dataset, y_test)

print('train score %: ', train_score * 100)
print('test score %: ', test_score * 100)

# accuracy:  87.5
# train score %:  98.94736842105263
# test score %:  87.5
