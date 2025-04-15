import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import pandas as pd
from tqdm.auto import tqdm
import os
import re
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --- 1. Define paths and settings ---
base_model_path = "./Qwen2.5-3B"
adapter_path = "./qlora_checkpoints/final_model"  # Path to your fine-tuned LoRA adapter
test_data_path = "./pre_process/evaluation/test.csv"
output_dir = "./tuned_evaluation_results_3B"
os.makedirs(output_dir, exist_ok=True)

# --- 2. Load test data ---
print(f"Loading test data from: {test_data_path}")
test_df = pd.read_csv(test_data_path)
print(f"Loaded {len(test_df)} test samples")

# --- 3. Load tokenizer and model ---
print(f"Loading tokenizer from: {base_model_path}")
tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = "left"  # Important for generation

# Load the base model first
print(f"Loading base model from: {base_model_path}")
model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

# Load the LoRA adapter
print(f"Loading LoRA adapter from: {adapter_path}")
model = PeftModel.from_pretrained(model, adapter_path)
model.eval()

# --- 4. Prepare prompts ---
print("Preparing input prompts...")
prompts_for_batch = []
true_labels = []
custom_ids = []

# System message
system_message = """You are an assistant that classifies comments into one of four categories:
0: Positive - Comments that express gratitude, satisfaction, or positive sentiment
1: Mild - Comments that are neutral or contain mild opinions
2: Negative - Comments that express criticism, anger, or negative sentiment
3: Irrelevant - Comments that are off-topic or not relevant to the context

Return only the category number in square brackets like this: [0], [1], [2], or [3]."""

# Create prompts
for idx, row in test_df.iterrows():
    custom_id = row['custom_id'] if 'custom_id' in test_df.columns else f"row-{idx}"
    message = row['message']
    true_label = row['class']
    
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
inference_batch_size = 4  # Adjust based on your GPU memory

for i in tqdm(range(0, len(prompts_for_batch), inference_batch_size), desc="Inference Progress"):
    batch_end = min(i + inference_batch_size, len(prompts_for_batch))
    batch_prompts = prompts_for_batch[i:batch_end]
    
    # Tokenize batch
    batch_inputs = tokenizer(
        batch_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    ).to(model.device)
    
    try:
        with torch.no_grad():
            # Generate with appropriate parameters for classification
            generated_ids = model.generate(
                **batch_inputs,
                max_new_tokens=20,  # Short output for classification
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False     # Greedy decoding
            )
        
        # Decode generated outputs
        for j in range(batch_end - i):
            # Extract only the newly generated tokens
            input_ids = batch_inputs['input_ids'][j]
            non_pad_start = torch.nonzero(input_ids != tokenizer.pad_token_id)[0].item() if tokenizer.padding_side == 'left' else 0
            actual_input_length = input_ids.size(0) - non_pad_start
            
            generated = generated_ids[j]
            gen_tokens = generated[actual_input_length:]
            decoded_output = tokenizer.decode(gen_tokens, skip_special_tokens=True)
            outputs_list.append(decoded_output.strip())
    
    except Exception as e:
        print(f"Error during generation for batch starting at index {i}: {e}")
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
results_df = pd.DataFrame({
    'custom_id': custom_ids,
    'message': [test_df['message'].iloc[i] for i in range(len(outputs_list))],
    'true_label': true_labels,
    'predicted_label': predicted_labels,
    'model_output': raw_outputs
})

# Save to CSV
results_path = os.path.join(output_dir, "finetuned_evaluation_results.csv")
results_df.to_csv(results_path, index=False)

# Save metrics
with open(os.path.join(output_dir, "finetuned_metrics.txt"), 'w') as f:
    f.write(f"Base Model: {base_model_path}\n")
    f.write(f"LoRA Adapter: {adapter_path}\n")
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
print(f"Finetuned model evaluation complete!")

# --- 8. Compare with baseline ---
try:
    baseline_path = os.path.join(output_dir, "baseline_metrics.txt")
    if os.path.exists(baseline_path):
        with open(baseline_path, 'r') as f:
            baseline_content = f.read()
            
        # Try to extract baseline accuracy
        baseline_acc_match = re.search(r'Accuracy: (0\.\d+)', baseline_content)
        if baseline_acc_match:
            baseline_acc = float(baseline_acc_match.group(1))
            improvement = (accuracy - baseline_acc) * 100
            
            print("\n=== Improvement Summary ===")
            print(f"Baseline accuracy: {baseline_acc:.4f}")
            print(f"Fine-tuned accuracy: {accuracy:.4f}")
            print(f"Absolute improvement: {improvement:.2f} percentage points")
            print(f"Relative improvement: {(improvement / baseline_acc):.2f}%")
            
            # Save comparison
            with open(os.path.join(output_dir, "improvement_summary.txt"), 'w') as f:
                f.write("=== Improvement Summary ===\n")
                f.write(f"Baseline accuracy: {baseline_acc:.4f}\n")
                f.write(f"Fine-tuned accuracy: {accuracy:.4f}\n")
                f.write(f"Absolute improvement: {improvement:.2f} percentage points\n")
                f.write(f"Relative improvement: {(improvement / baseline_acc):.2f}%\n")
except Exception as e:
    print(f"Could not compare with baseline: {e}")