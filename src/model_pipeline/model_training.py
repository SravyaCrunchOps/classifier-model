from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def model_training(X_train, y_train):
    
    # Create pipeline with scaler and model
    model_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression())
    ])
    
    # Fit pipeline on training data
    model_pipeline.fit(X_train, y_train)

    return model_pipeline
