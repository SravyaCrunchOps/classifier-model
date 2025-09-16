# Step-1: Import necessary libraries

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step-2: Load `.csv` dataset
df = pd.read_csv("./zoo_animals_data.csv")
print(df.head(3))

# Step-3: Data Cleaning
print(df.isnull().sum()) # no null values found so data is in clean shape

# Step-4: Exploratory Data Analysis
print(df.describe())

print(df['class_type'].unique())

class_type_counts = df['class_type'].value_counts()
print(class_type_counts)
# class_type 1 has highest count of 41 and class_type 6 has lowest count of 4

# Visualizing class_type distribution
plt.figure(figsize=(10, 6))
sns.countplot(x='class_type', data=df, palette='viridis')
plt.title('Distribution of Class Types')
plt.xlabel('Class Type')
plt.ylabel('Count')
plt.show()


# step-5: Feature Enggineering 
df['can_fly'] = (df['airborne'] == 1) & (df['feathers'] == 1)
df['can_swim'] = (df['aquatic'] == 1) & (df['fins'] == 1)
df['is_domestic_pet'] = (df['domestic'] == 1) & (df['catsize'] == 1)


# step-6: Data Preprocessing
df['can_fly'] = df['can_fly'].astype(int)
df['can_swim'] = df['can_swim'].astype(int)
df['is_domestic_pet'] = df['is_domestic_pet'].astype(int)

print(df.head())
# Note: save preprocessed data in '.csv; file for future use

X = df.drop(columns=['airborne', 'feathers', 'domestic', 'aquatic', 'fins', 'catsize', 'animal_name', 'class_type', 'class_name'])
y = df['class_name']
feature_names = X.columns.to_list()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()

X_train['legs'] = scaler.fit_transform(X_train[['legs']])
X_test['legs'] = scaler.transform(X_test[['legs']])


# Step-7: Model Training
model = LogisticRegression()
model.fit(X_train, y_train)


# step-8: Model Evaluation
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100}%")

print("classification report: ")
print(classification_report(y_test, y_pred))

print("Confusion Matrix: ")
print(confusion_matrix(y_test, y_pred))


if accuracy < 0.9:
    # Step-9: Cross-Validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    print("CV Scores: %.2f%%" % cv_scores.mean() * 100)


    # Step-10: Hyperparameter Tuning
    param_grid = {
        "C": [0.01, 0.1, 1, 10, 100],
        "solver": ["lbfs", "saga"],
        "penalty": ["l2"]
    }
    grid = GridSearchCV(model, param_grid=param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train, y_train)
    print("Best Params: ", grid.best_params_)
    print("Best CV Score: ", grid.best_score_ * 100)
    best_model = grid.best_estimator_
    print('best_model: ', best_model)


# save model in .pkl format
import joblib
joblib.dump(model, 'models/classifier_model.pkl')
joblib.dump(feature_names, 'models/feature_names.pkl')
joblib.dump(df, 'models/preprocessed_data.csv')
