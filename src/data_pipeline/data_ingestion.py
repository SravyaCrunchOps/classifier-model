import pandas as pd
import re

ZOO_DATASET_PATH = "../datasets/zoo_data.csv"
CLASS_DATASET_PATH = "../datasets/class.csv"
MERGRED_DATASET_PATH = "../datasets/final_dataset.csv"


def data_ingestion():
    zoo_df = pd.read_csv(ZOO_DATASET_PATH)
    class_df = pd.read_csv(CLASS_DATASET_PATH)

    # split 'class_dataset' 'animal_names' column in separate rows
    class_df['animal_name'] = class_df['Animal_Names'].apply(lambda x: re.split(r",\s*", x))

    class_df = class_df.explode('animal_name')

    # remove unnecessary columns in 'class_dataset'
    class_df = class_df.drop(columns=['Animal_Names', 'Number_Of_Animal_Species_In_Class', 'Class_Number'])
    
    # rename 'Class_Type' to 'class_name'
    class_df = class_df.rename(columns={'Class_Type': 'class_name'})

    # merge two datasets
    merged_df = pd.merge(zoo_df, class_df, on='animal_name', how='left')

    # save this in datasets folder
    merged_df.to_csv(MERGRED_DATASET_PATH)
    return merged_df
