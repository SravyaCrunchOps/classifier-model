import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score


MERGRED_DATASET_PATH = "../datasets/final_dataset.csv"

def validation(model):
    df = pd.read_csv(MERGRED_DATASET_PATH)
    X = df.drop(columns=['airborne', 'feathers', 'domestic', 'aquatic', 'fins', 'catsize', 'animal_name', 'class_type', 'class_name'])
    y = df['class_name']

    # cross validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')

    cv_mean = cv_scores.mean() * 100

    print(f"CV Scroes %: {cv_mean}")
    print(f"cv: {cv}")

    # CV Scores %: 94.14285714285715
    # cv: StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
    
    return cv_mean, cv