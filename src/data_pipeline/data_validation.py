import pandera.pandas as pa
from pandera import Column, DataFrameSchema, Check
import pandas as pd


# Define the schema
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

def validate_data(df):
    try:
        validated_df = schema.validate(df)
        print("Data validation successful!")
        return validated_df
    except pa.errors.SchemaErrors as err:
        print("Data validation failed:")
        print(err.failure_cases)
        return None
