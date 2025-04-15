import pandas as pd
import json

# File paths
csv_file = './datas/simple.csv'  # Your CSV file with username, comment, and class
prompt_file = './datas/prompt_v2.txt'  # Your prompt template file
output_file = 'inf_batch_v2.jsonl'  # Output JSONL file

# Read the prompt template
with open(prompt_file, 'r', encoding='utf-8') as f:
    prompt_template = f.read().strip()

# Read the CSV data (with proper handling of commas in text)
df = pd.read_csv(csv_file, quoting=1)  # quoting=1 means QUOTE_ALL

# Open the output file
with open(output_file, 'w', encoding='utf-8') as f:
    # Process each row in the CSV
    for index, row in df.iterrows():
        username = row['username']
        comment = row['comment']
        
        # Format the user message with the comment data
        user_message = prompt_template + username + "+"+ comment
        # Create the JSONL entry
        entry = {
            "custom_id": f"row-{index}",  # Use row number as custom_id
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "deepseek-ai/DeepSeek-V3",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a highly advanced AI assistant that classifies comments."
                    },
                    {
                        "role": "user",
                        "content": user_message
                    }
                ],
                "stream": True,
                "max_tokens": 20
            }
        }
        
        # Write the entry as a JSON line
        f.write(json.dumps(entry, ensure_ascii=False) + '\n')

print(f"Successfully created batch inference file at {output_file}")