import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

# Create the evaluation directory if it doesn't exist
os.makedirs('./evaluation', exist_ok=True)

# Path to your data file - replace with your actual file path
data_file = 'purged_labeled_v2.csv'

# Read the data
df = pd.read_csv(data_file)

# Extract features and target
X = df[['custom_id', 'message']]
y = df['class']

# First split: separate out test set (20%)
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Second split: separate training and validation sets
# Splitting the remaining 80% into 70/10 ratio
# This means validation should be 12.5% of the temp set (10% of total)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.125, random_state=42, stratify=y_temp
)

# Combine features and targets back into DataFrames
train_df = pd.concat([X_train, pd.DataFrame(y_train)], axis=1)
val_df = pd.concat([X_val, pd.DataFrame(y_val)], axis=1)
test_df = pd.concat([X_test, pd.DataFrame(y_test)], axis=1)

# Save to CSV files
train_df.to_csv('./evaluation/train.csv', index=False)
val_df.to_csv('./evaluation/validation.csv', index=False)
test_df.to_csv('./evaluation/test.csv', index=False)

# Print statistics
print(f"Original dataset: {len(df)} samples")
print(f"Training set: {len(train_df)} samples ({len(train_df)/len(df)*100:.1f}%)")
print(f"Validation set: {len(val_df)} samples ({len(val_df)/len(df)*100:.1f}%)")
print(f"Test set: {len(test_df)} samples ({len(test_df)/len(df)*100:.1f}%)")

# Print class distribution in each set
for name, dataset in [("Original", df), ("Train", train_df), 
                     ("Validation", val_df), ("Test", test_df)]:
    class_counts = dataset['class'].value_counts().sort_index()
    total = len(dataset)
    percentages = [count/total*100 for count in class_counts]
    
    print(f"\n{name} set class distribution:")
    print(f"Class counts: {class_counts.values}")
    print(f"Class percentages: [{percentages[0]:.1f}%|{percentages[1]:.1f}%|" +
          f"{percentages[2]:.1f}%|{percentages[3]:.1f}%]")