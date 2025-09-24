import pandera.pandas as pa
from pandera import Column, Check
import pandas as pd
import os


# 1.Define the schema
schema = pa.DataFrameSchema({
    "animal_name": Column(str, nullable=False),
    "hair": Column(int, checks=[Check.isin([0, 1])], nullable=False),
    "feathers": Column(int, checks=[Check.isin([0, 1])], nullable=False),
    "eggs": Column(int, checks=[Check.isin([0, 1])], nullable=False),
    "milk": Column(int, checks=[Check.isin([0, 1])], nullable=False),
    "airborne": Column(int, checks=[Check.isin([0, 1])], nullable=False),
    "aquatic": Column(int, checks=[Check.isin([0, 1])], nullable=False), 
    "predator": Column(int, checks=[Check.isin([0, 1])], nullable=False),
    "toothed": Column(int, checks=[Check.isin([0, 1])], nullable=False),
    "backbone": Column(int, checks=[Check.isin([0, 1])], nullable=False),
    "breathes": Column(int, checks=[Check.isin([0, 1])], nullable=False),
    "venomous": Column(int, checks=[Check.isin([0, 1])], nullable=False),
    "fins": Column(int, checks=[Check.isin([0, 1])], nullable=False),  
    "legs": Column(int, checks=[Check.isin([0, 2, 4, 5, 6, 8])], nullable=False),
    "tail": Column(int, checks=[Check.isin([0, 1])], nullable=False),
    "domestic": Column(int, checks=[Check.isin([0, 1])], nullable=False), 
    "catsize": Column(int, checks=[Check.isin([0, 1])], nullable=False),
    "class_type": Column(int, [Check.ge(1), Check.le(7)], nullable=False),
    "class_name": Column(str, Check.isin(["Mammal", "Fish", "Amphibian", "Bird", "Invertebrate", "Bug", "Reptile"]), nullable=False),
})


def check_validation(df):
    try:
        validated_df = schema.validate(df, lazy=True) # lazy=True to catch all errors
        print(validated_df.head(3))
        print("Data validation successful!")

        # save validated dataset, not necessarily needed since it is same as merged df.
        output_dir = "../../datasets/validation"
        os.makedirs(output_dir, exist_ok=True)  # This creates the directory if it doesn't exist
        validated_df.to_csv(f"{output_dir}/validated_df.csv", index=False)
        print(f"✅ Validated data saved to: {output_dir}/validated_df.csv")
        
    except pa.errors.SchemaErrors as err:
        print("Data validation failed:")
        print(err.failure_cases)
        # log error
        output_dir = "../../logs/validation"
        os.makedirs(output_dir, exist_ok=True)
        
        err.failure_cases.to_csv(f"{output_dir}/error_df.csv", index=False)
        print(f"Error saved to: {output_dir}/error_df.csv")



# 2. Start validation  
df = pd.read_csv("../../datasets/ingestion/merged_df.csv")

# cleaned data
file_path = '../../datasets/cleaned/zoo_dataset_cleaned.csv'
if os.path.exists(file_path):
    df = pd.read_csv(file_path)
else:
    print(f"File not found: {file_path}")


print(df.head(3))
print("Shape: ", df.shape)

# check quality
print("\nFind missing values: ")
print(df.isnull().sum())

print("\nData types")
print(df.dtypes)
check_validation(df)
