from data_pipeline.data_ingestion import data_ingestion
from data_pipeline.data_validation import validate_data
from data_pipeline.data_cleaning import data_cleaning
from data_pipeline.data_eda import eda
from data_pipeline.data_feature_engg import feature_engg
from data_pipeline.data_preprocessing import preprocessing


def data_pipeline():
    # 1. data_ingestion
    merged_df = data_ingestion()

    # 2. data validation
    valid_df = validate_data(merged_df)
    print('valid_df: ', valid_df)

    # 3. data cleaning
    nulls = data_cleaning(valid_df)
    if not nulls.empty:
        print("Columns with null values: ")
        print(nulls)
    else:
        print("No null values found in the dataset.")

    # 4. eda
    eda(merged_df)
    
    # 5. feature engg
    df = feature_engg(merged_df)

    # 6. data preprocessing
    X_train, X_test, y_train, y_test = preprocessing(df)

    return X_train, X_test, y_train, y_test

