import json
import pandas as pd
import re

'''
This file is used to convert the jsonl file used to batch process labels to CSV files:
1. unlabeled.csv - with data extracted from batch file
2. labeled.csv - with classes updated from response file
'''

# File paths
#This is the raw input file
batch_file = './datas/inf_batch_v2.jsonl'
#This is the output file generared by labeler
response_file = './datas/silver_label_v2.jsonl'


unlabeled_csv = 'unlabeled_v2.csv'
labeled_csv = 'labeled_v2.csv'

# Initialize lists to store data
custom_ids = []
messages = []
classes = []

# Read the inference batch file
with open(batch_file, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        
        # Extract custom_id
        custom_id = data["custom_id"]
        
        # Extract the full content
        full_content = data["body"]["messages"][1]["content"]
        
        # Define the pattern to find the actual input message
        pattern = r"现在请处理：\nInput:(.*?)$"
        match = re.search(pattern, full_content, re.DOTALL)
        
        if match:
            # Extract just the user message part
            message = match.group(1).strip()
        else:
            # Fallback if pattern doesn't match
            message = "Error: Could not extract message"
            
        # Add to lists
        custom_ids.append(custom_id)
        messages.append(message)
        classes.append(-1)  # Initialize class as -1

# Create DataFrame
df = pd.DataFrame({
    'custom_id': custom_ids,
    'message': messages,
    'class': classes
})

# Save to unlabeled CSV
df.to_csv(unlabeled_csv, index=False, quoting=1)
print(f"Successfully created {unlabeled_csv} with {len(df)} entries")

# Now read the response file and update the class values
class_mapping = {}  # To store custom_id to class mapping
error_count = 0
error_ids = []

with open(response_file, 'r', encoding='utf-8') as f:
    for line in f:
        try:
            data = json.loads(line)
            custom_id = data["custom_id"]
            
            # Extract the response content
            content = data["response"]["body"]["choices"][0]["message"]["content"]
            
            # Check if the content is in the expected format
            valid_pattern = r'^\s*(?:[\[(](\d+)[\])]|(\d+))\s*$'
            match = re.search(valid_pattern, content)
            
            if match:
                # Valid format found, extract the class value and convert to integer
                class_value = int(match.group(1) if match.group(1) is not None else match.group(2))
                class_mapping[custom_id] = class_value
            else:
                # Content doesn't match expected format
                error_count += 1
                error_ids.append(custom_id)
        except Exception as e:
            error_count += 1
            error_ids.append(custom_id)
            continue

# Print error count only
print(f"Total errors found: {error_count}")

# Update the class values in the DataFrame - now ensuring values are integers
for i, row in df.iterrows():
    custom_id = row['custom_id']
    if custom_id in class_mapping:
        # Explicitly convert to integer to ensure proper assignment
        df.at[i, 'class'] = int(class_mapping[custom_id])

# Save to labeled CSV with classes updated
df.to_csv(labeled_csv, index=False, quoting=1)
print(f"Successfully created {labeled_csv} with {len(df)} entries and {len(class_mapping)} labeled records")




## This part of the program will remove the invalid data from the original data. 
def purge_invalid_entries(df, output_file="purged_labeled_v2.csv"):
    # Make a copy to avoid modifying the original
    purged_df = df.copy()
    # Filter out entries with class -1
    purged_df = purged_df[purged_df['class'] != -1]
    # Save to CSV
    purged_df.to_csv(output_file, index=False, quoting=1)
    # Calculate class distribution
    class_counts = [0, 0, 0, 0]  # For classes 0, 1, 2, 3
    
    for class_value in range(4):
        class_counts[class_value] = len(purged_df[purged_df['class'] == class_value])
    # Print statistics
    total_size = len(purged_df)
    print(f"Successfully created {output_file}")
    print(f"Removed {len(df) - len(purged_df)} invalid entries")
    print(f"Kept {len(purged_df)} valid entries")
    print(f"Class distribution [0|1|2|3]: [{class_counts[0]}|{class_counts[1]}|{class_counts[2]}|{class_counts[3]}]")
    
    return purged_df

# After updating the class values, call the function
purged_df = purge_invalid_entries(df, "purged_labeled_v2.csv")