import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os


df = pd.read_csv("../../datasets/ingestion/merged_df.csv")

# 1. Data Quality Check
print(f"Shape: {df.shape}")
print("------")

print(f"Information: {df.info()}")
print("------")

print(f"Describe: {df.describe(include='all')}")
print("------")

print(f"\nNull values ? \n{df.isnull().sum().sort_values(ascending=False)}")
print("------")

print(f"Duplicated values ? \n{df.duplicated().sum()}")

for col in df.columns:
    print(f'{col}: ', df[col].nunique())


# save df 
output_dir = "../../datasets/eda"
os.makedirs(output_dir, exist_ok=True)
df.to_csv(f"{output_dir}/eda_df.csv", index_label=False)
