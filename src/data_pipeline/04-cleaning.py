import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os


df = pd.read_csv("../../datasets/eda/eda_df.csv")

df_clean = df.copy()

# 1. Remove duplicate rows
print(f"Removing {df_clean.duplicated().sum()} duplicates...")
df_clean = df_clean.drop_duplicates()

# 2. Handle missing values
print("Handling missing values...")

# 2.1 Remove row with missing animal_name (can't impute this)
df_clean = df_clean.dropna(subset=['animal_name'])

# Impute 'class_type' based on class_name whre possible
class_mapping = {
    'Mammal': 1, 'Bird': 2, 'Reptile': 3, 'Fish': 4, 
    'Amphibian': 5, 'Bug': 6, 'Invertebrate': 7
}
    
# 2.2 Fill class_type from class_name
mask = df_clean['class_type'].isnull() & df_clean['class_name'].notnull()
df_clean.loc[mask, 'class_type'] = df_clean.loc[mask, 'class_name'].map(class_mapping)


# 2.3 Fill class_name from class_type (reverse mapping)
reverse_mapping = {v: k for k, v in class_mapping.items()}
mask = df_clean['class_name'].isnull() & df_clean['class_type'].notnull()
df_clean.loc[mask, 'class_name'] = df_clean.loc[mask, 'class_type'].map(reverse_mapping)

# 2.4 Remove rows where both class_type and class_name are missing
df_clean = df_clean.dropna(subset=['class_type', 'class_name'], how='all')


# 2.4 Impute numerical missing values with mode (binary features)
binary_columns = ['hair', 'feathers', 'eggs', 'milk', 'airborne', 'aquatic', 
    'predator', 'toothed', 'backbone', 'breathes', 'venomous', 
    'fins', 'tail', 'domestic', 'catsize']
    
for col in binary_columns:
    if col in df_clean.columns and df_clean[col].isnull().any():
        mode_val = df_clean[col].mode()[0]
        df_clean[col] = df_clean[col].fillna(mode_val)
        print(f"Imputed {col} with mode: {mode_val}")


# 2.5 Impute legs with median (non-binary feature)
if 'legs' in df_clean.columns and df_clean['legs'].isnull().any():
    median_legs = df_clean['legs'].median()
    df_clean['legs'] = df_clean['legs'].fillna(median_legs)
    print(f"Imputed legs with median: {median_legs}")


# 3. Fix data types
print("Fixing data types...")

# Convert float columns to int where appropriate
int_columns = ['hair', 'venomous', 'legs', 'class_type']
for col in int_columns:
    if col in df_clean.columns:
        df_clean[col] = df_clean[col].astype(int)

print("\n=== VALIDATION ===")
print(f"Final shape: {df_clean.shape}")
print(f"Missing values:\n{df_clean.isnull().sum()}")
print(f"Duplicates: {df_clean.duplicated().sum()}")
print(f"Data types:\n{df_clean.dtypes}")

# 6. Save cleaned dataset
output_dir = "../../datasets/cleaned"
os.makedirs(output_dir, exist_ok=True)

df_clean.to_csv(f"{output_dir}/zoo_dataset_cleaned.csv", index=False)
print(f"\nCleaned dataset saved with {df_clean.shape[0]} rows and {df_clean.shape[1]} columns")

