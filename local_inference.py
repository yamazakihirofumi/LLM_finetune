import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from tqdm.auto import tqdm
import os
import time
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np

# --- 1. Define paths and settings ---
local_model_path = "./Qwen2.5-0.5B"  # You can change this to your preferred model
test_data_path = "./pre_process/evaluation/test.csv"
output_dir = "./evaluation_results_0.5B"
os.makedirs(output_dir, exist_ok=True)

# --- 2. Load the test dataset ---
print(f"Loading test data from: {test_data_path}")
test_df = pd.read_csv(test_data_path)
print(f"Loaded {len(test_df)} test samples")

# Check class distribution
class_distribution = test_df['class'].value_counts().sort_index()
print("Class distribution in test set:")
for class_id, count in class_distribution.items():
    print(f"Class {class_id}: {count} samples ({count/len(test_df)*100:.2f}%)")

# --- 3. Load the Tokenizer and Model ---
print(f"Loading tokenizer from: {local_model_path}")
tokenizer = AutoTokenizer.from_pretrained(local_model_path, trust_remote_code=True)
print("Tokenizer loaded.")

# Set padding token if not already set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print(f"Set pad_token to eos_token: {tokenizer.pad_token}")

# IMPORTANT: Set padding_side to 'left' for decoder-only models
tokenizer.padding_side = "left"
print(f"Set padding_side to {tokenizer.padding_side} for proper generation")

print(f"Loading model from: {local_model_path}")
model = AutoModelForCausalLM.from_pretrained(
    local_model_path,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True
)
print("Model loaded.")
model.eval()

# --- 4. Prepare input prompts ---
print("Preparing input prompts...")
prompts_for_batch = []
true_labels = []
custom_ids = []

# System message defining the classification task
system_message = """You are an assistant that classifies comments into one of four categories:
0: Positive - Comments that express gratitude, satisfaction, or positive sentiment
1: Mild - Comments that are neutral or contain mild opinions
2: Negative - Comments that express criticism, anger, or negative sentiment
3: Irrelevant - Comments that are off-topic or not relevant to the context

Return only the category number in square brackets like this: [0], [1], [2], or [3]."""

# Process each test sample
for idx, row in test_df.iterrows():
    custom_id = row['custom_id'] if 'custom_id' in test_df.columns else f"row-{idx}"
    message = row['message']
    true_label = row['class']
    
    # Construct the prompt
    prompt = f"<|im_start|>system\n{system_message}<|im_end|>\n<|im_start|>user\nClassify this comment: \"{message}\"<|im_end|>\n<|im_start|>assistant\n"
    
    prompts_for_batch.append(prompt)
    true_labels.append(true_label)
    custom_ids.append(custom_id)
    
    # Print a sample prompt
    if idx == 0:
        print("\nExample prompt:")
        print(prompt)
        print("-" * 50)

print(f"Prepared {len(prompts_for_batch)} prompts for inference")

# --- 5. Run inference in batches ---
print("Starting inference...")
outputs_list = []
inference_batch_size = 4  # Adjust based on your GPU's VRAM capacity
# Normally is 4 but 400 work for 0.5B model

for i in tqdm(range(0, len(prompts_for_batch), inference_batch_size), desc="Inference Progress"):
    batch_end = min(i + inference_batch_size, len(prompts_for_batch))
    batch_prompts = prompts_for_batch[i:batch_end]
    
    # Tokenize batch
    batch_inputs = tokenizer(
        batch_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=model.config.max_position_embeddings if hasattr(model.config, 'max_position_embeddings') else 2048
    ).to(model.device)
    
    try:
        with torch.no_grad():
            # Generate with appropriate parameters for classification
            generated_ids = model.generate(
                **batch_inputs,
                max_new_tokens=50,  # Shorter for classification responses
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False  # Use greedy decoding for classification
            )
        
        # Decode generated outputs
        for j in range(batch_end - i):
            # For left padding, find where the actual tokens start
            input_ids = batch_inputs['input_ids'][j]
            non_pad_start = torch.nonzero(input_ids != tokenizer.pad_token_id)[0].item() if tokenizer.padding_side == 'left' else 0
            actual_input_length = input_ids.size(0) - non_pad_start
            
            # Extract only the newly generated tokens
            generated = generated_ids[j]
            gen_tokens = generated[actual_input_length:]
            decoded_output = tokenizer.decode(gen_tokens, skip_special_tokens=True)
            outputs_list.append(decoded_output.strip())
    
    except Exception as e:
        print(f"Error during generation for batch starting at index {i}: {e}")
        # Fill with error placeholders
        outputs_list.extend(["[ERROR]"] * (batch_end - i))

print("Inference complete.")

# --- 6. Parse predictions and calculate metrics ---
print("Extracting predictions and calculating metrics...")
predicted_labels = []
raw_outputs = []

# Extract predicted class from model outputs
for output in outputs_list:
    raw_outputs.append(output)
    
    # Try to extract classification number using regex
    import re
    class_match = re.search(r'\[(\d+)\]', output)
    
    if class_match:
        # If we find a class number in brackets, use it
        predicted_label = int(class_match.group(1))
        if predicted_label not in [0, 1, 2, 3]:
            predicted_label = -1  # Invalid class prediction
    else:
        # If no class number found, try to find class keywords
        if "positive" in output.lower():
            predicted_label = 0
        elif "mild" in output.lower() or "neutral" in output.lower():
            predicted_label = 1
        elif "negative" in output.lower():
            predicted_label = 2
        elif "irrelevant" in output.lower():
            predicted_label = 3
        else:
            predicted_label = -1  # Could not determine class
    
    predicted_labels.append(predicted_label)

# Calculate metrics
valid_predictions = [pred for pred in predicted_labels if pred != -1]
valid_indices = [i for i, pred in enumerate(predicted_labels) if pred != -1]
valid_true_labels = [true_labels[i] for i in valid_indices]

if len(valid_predictions) > 0:
    accuracy = accuracy_score(valid_true_labels, valid_predictions)
    print(f"Accuracy on valid predictions: {accuracy:.4f} ({len(valid_predictions)}/{len(predicted_labels)} samples)")
    
    # Detailed metrics
    report = classification_report(valid_true_labels, valid_predictions, target_names=["Positive", "Mild", "Negative", "Irrelevant"])
    print("\nClassification Report:")
    print(report)
    
    # Confusion matrix
    cm = confusion_matrix(valid_true_labels, valid_predictions)
    print("\nConfusion Matrix:")
    print(cm)
else:
    print("No valid predictions found. Check model outputs.")

# --- 7. Save results ---
# Create results dataframe
results_df = pd.DataFrame({
    'custom_id': custom_ids,
    'message': [test_df['message'].iloc[i] for i in range(len(outputs_list))],
    'true_label': true_labels,
    'predicted_label': predicted_labels,
    'model_output': raw_outputs
})

# Save to CSV
results_path = os.path.join(output_dir, f"baseline_evaluation_results.csv")
results_df.to_csv(results_path, index=False)

# Save metrics
with open(os.path.join(output_dir, "baseline_metrics.txt"), 'w') as f:
    f.write(f"Model: {local_model_path}\n")
    f.write(f"Test set: {test_data_path}\n")
    f.write(f"Total samples: {len(test_df)}\n")
    f.write(f"Valid predictions: {len(valid_predictions)} ({len(valid_predictions)/len(predicted_labels)*100:.2f}%)\n")
    if len(valid_predictions) > 0:
        f.write(f"Accuracy: {accuracy:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\nConfusion Matrix:\n")
        f.write(str(cm))

print(f"Results saved to {output_dir}")
print(f"Baseline evaluation complete!")