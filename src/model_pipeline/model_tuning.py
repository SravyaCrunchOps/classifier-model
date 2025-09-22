from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import pandas as pd

LOGS_PATH = "../logs"

def hyperparameter_tuning(X_train, y_train, cv):

    param_grid = {
        "classifier__C": [0.01, 0.1, 1, 10, 100],
        "classifier__solver": ["lbfgs", "saga"],
        "classifier__penalty": ["l2"],
        'classifier__max_iter': [2000]
    }

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression())
    ])


    grid = GridSearchCV(pipe, param_grid=param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    print(f'best model: {best_model}')
    print(f'best params: {grid.best_params_}')
    print(f'best cv scores: {grid.best_score_ * 100}')

    # best model: Pipeline(steps=[('scaler', StandardScaler()),
    #             ('classifier', LogisticRegression(C=0.1))])
    # best params: {'classifier__C': 0.1, 'classifier__penalty': 'l2', 'classifier__solver': 'lbfgs'}
    # best cv scores: 91.47058823529412

    # print(grid.cv_results_)

    results = pd.DataFrame(grid.cv_results_)
    grid_results = results[['param_classifier__C', 'param_classifier__solver', 'mean_test_score', 'std_test_score']]
    grid_results.to_csv(f'{LOGS_PATH}/gridsearch_logs.csv', index=False)

    return best_model
