import os
import pandas as pd

data_dir = "data/goemotions"

# Load train, val and test datasets
df_train = pd.read_csv(os.path.join(data_dir, "goemotions_all_train_balanced.csv"))
df_val = pd.read_csv(os.path.join(data_dir, "goemotions_all_val_balanced.csv"))
df_test = pd.read_csv(os.path.join(data_dir, "goemotions_all_test_balanced.csv"))

# Display basic info
print("Train dataset:")
print(f"  Shape: {df_train.shape}")
print(f"  Columns: {df_train.columns.tolist()}")

print("\nValidation dataset:")
print(f"  Shape: {df_val.shape}")
print(f"  Columns: {df_val.columns.tolist()}")

print("\nTest dataset:")
print(f"  Shape: {df_test.shape}")
print(f"  Columns: {df_test.columns.tolist()}")

# Show first rows
print("\nFirst row of train data:")
print(df_train.head(1))
