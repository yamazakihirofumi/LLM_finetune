import os
import torch
import pandas as pd
from datasets import Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    PeftModel
)
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

# 1. Configuration
MODEL_PATH = "./Qwen2.5-3B"
TRAIN_FILE = "./pre_process/evaluation/train.csv"
VALIDATION_FILE = "./pre_process/evaluation/validation.csv"
OUTPUT_DIR = "./qlora_checkpoints"

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 2. Load tokenizer and model with quantization config
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

# Set padding token if not set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print(f"Set pad_token to eos_token: {tokenizer.pad_token}")

# Configure quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,              # Quantize to 4-bit
    bnb_4bit_use_double_quant=True, # Use double quantization
    bnb_4bit_quant_type="nf4",      # Normalized float 4
    bnb_4bit_compute_dtype=torch.float16,
)

print("Loading model with 4-bit quantization...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

# 3. Prepare LoRA config and set up model for training
print("Configuring LoRA...")
lora_config = LoraConfig(
    r=16,                  # Rank dimension
    lora_alpha=32,         # Alpha scaling
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Prepare model for k-bit training
print("Preparing model for training...")
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()  # Show percentage of trainable parameters

# 4. Load and preprocess the training data
print(f"Loading training data from {TRAIN_FILE}...")
train_df = pd.read_csv(TRAIN_FILE)
print(f"Loaded {len(train_df)} training samples")

print(f"Loading validation data from {VALIDATION_FILE}...")
val_df = pd.read_csv(VALIDATION_FILE)
print(f"Loaded {len(val_df)} validation samples")

# Create the system prompt for the classification task
system_prompt = """You are an assistant that classifies comments into one of four categories:
0: Positive - Comments that express gratitude, satisfaction, or positive sentiment
1: Mild - Comments that are neutral or contain mild opinions
2: Negative - Comments that express criticism, anger, or negative sentiment
3: Irrelevant - Comments that are off-topic or not relevant to the context

Return only the category number in square brackets like this: [0], [1], [2], or [3]."""

# Function to format each example
def format_prompt(row):
    message = row["message"]
    true_label = row["class"]
    
    # Format the prompt with system message and the comment to classify
    prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\nClassify this comment: \"{message}\"<|im_end|>\n<|im_start|>assistant\n[{true_label}]<|im_end|>"
    
    return prompt

# Process and tokenize datasets in one step
def process_and_tokenize(row):
    prompt = format_prompt(row)
    tokenized = tokenizer(
        prompt,
        truncation=True,
        max_length=512,
        padding="max_length",
        return_tensors=None  # Important! Don't return tensors yet
    )
    
    # Add labels for causal language modeling (shifted input_ids)
    tokenized["labels"] = tokenized["input_ids"].copy()
    
    return tokenized

print("Processing and tokenizing datasets...")
# Apply to each row individually, not in batch
train_processed = [process_and_tokenize(row) for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Processing train")]
val_processed = [process_and_tokenize(row) for _, row in tqdm(val_df.iterrows(), total=len(val_df), desc="Processing val")]

# Convert to HuggingFace datasets with the tokenized data directly
train_tokenized = Dataset.from_list(train_processed)
val_tokenized = Dataset.from_list(val_processed)

# Print an example
print("\nExample tokenized data:")
print(f"Input length: {len(train_tokenized[0]['input_ids'])}")
print("-" * 50)

# 6. Configure training
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=8,
    logging_steps=25,
    learning_rate=2e-4,
    weight_decay=0.01,
    warmup_steps=20,
    save_steps=100,
    fp16=True,
    report_to="none",
    save_total_limit=3,
    remove_unused_columns=False,
)

# 7. Define a simple data collator - no additional processing needed
# since we've already prepared the data properly
from transformers.data.data_collator import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer)

# 8. No compute_metrics function needed for our setup
# We're focusing on training only since evaluation is handled separately

# 9. Create Trainer - simplified for our use case
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# 10. Train the model
print("Starting training...")
trainer.train()

# 11. Save the fine-tuned model
print("Saving model...")
model.save_pretrained(os.path.join(OUTPUT_DIR, "final_model"))

print("Fine-tuning complete! The model is saved in:", os.path.join(OUTPUT_DIR, "final_model"))