from data_pipeline.index import data_pipeline
from model_pipeline.index import model_pipeline


if __name__ == '__main__':
    print(" ---- Start Data Pipeline ----- ")
    X_train, X_test, y_train, y_test = data_pipeline()

    print(" ----- Start Model Pipeline ----- ")
    model_pipeline(X_train, X_test, y_train, y_test)

    print(" -- Model deployed locally -> `models/` -- ")
