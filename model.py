# Step-1: Import necessary libraries

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
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
# there are 7 unique class_types

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

# step-5 : Data Preprocessing
# Here all the data is in numerical format so no need of encoding categorical data

# step-6: Split Train & Test Data
X = df.drop(columns=['animal_name', 'class_type', 'class_name'])
y = df['class_name']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step-7: Model Training
model = LogisticRegression()
model.fit(X_train, y_train)

# step-8: Model Evaluation

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy*100:.2f}%")

print("classification report: ")
print(classification_report(y_test, y_pred))

# save model in .pkl format
import joblib
joblib.dump(model, 'zoo_animal_classifier_model.pkl')

