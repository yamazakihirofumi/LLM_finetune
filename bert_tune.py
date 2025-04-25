import os
import torch
import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, f1_score
import random
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# Set paths and parameters
MODEL_NAME = "bert-base-chinese"
TRAIN_FILE = "./pre_process/evaluation/train.csv"
VAL_FILE = "./pre_process/evaluation/validation.csv"
TEST_FILE = "./pre_process/evaluation/test.csv"
OUTPUT_DIR = "./bert_checkpoints_improved"
NUM_LABELS = 4  # 0: Positive, 1: Mild, 2: Negative, 3: Irrelevant
MAX_LENGTH = 128
BATCH_SIZE = 32  # Increased batch size
EPOCHS = 10  # More epochs with early stopping
LEARNING_RATE = 3e-5  # Slightly higher learning rate
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 0.1  # 10% of total steps
PATIENCE = 3  # Early stopping patience
CLASS_WEIGHTS = True  # Use class weights to handle imbalance

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load data
print(f"Loading data...")
print(f"Training data: {TRAIN_FILE}")
print(f"Validation data: {VAL_FILE}")
print(f"Test data: {TEST_FILE}")

train_df = pd.read_csv(TRAIN_FILE)
val_df = pd.read_csv(VAL_FILE)
test_df = pd.read_csv(TEST_FILE)

print(f"Train: {len(train_df)} examples")
print(f"Validation: {len(val_df)} examples")
print(f"Test: {len(test_df)} examples")

# Analyze class distribution
train_class_counts = train_df['class'].value_counts().sort_index()
print("Class distribution in training set:")
for class_id, count in train_class_counts.items():
    print(f"Class {class_id}: {count} samples ({count/len(train_df)*100:.2f}%)")

# Calculate class weights
if CLASS_WEIGHTS:
    total_samples = len(train_df)
    class_weights = {}
    for class_id, count in train_class_counts.items():
        # Inverse frequency weighting
        class_weights[class_id] = total_samples / (count * len(train_class_counts))
    
    # Convert to tensor for loss function
    class_weights_tensor = torch.tensor([class_weights[i] for i in range(NUM_LABELS)], 
                                        dtype=torch.float)
    
    print("Class weights:")
    for class_id, weight in class_weights.items():
        print(f"Class {class_id}: {weight:.4f}")
else:
    class_weights_tensor = None

# Load tokenizer and model
print(f"Loading tokenizer and model...")
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertForSequenceClassification.from_pretrained(
    MODEL_NAME, 
    num_labels=NUM_LABELS,
    problem_type="single_label_classification"
)

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Using device: {device}")

# Function to prepare dataset
def prepare_dataset(df):
    texts = df["message"].tolist()
    labels = df["class"].tolist()
    
    # Tokenize texts
    encodings = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )
    
    # Create tensor dataset
    input_ids = encodings["input_ids"]
    attention_mask = encodings["attention_mask"]
    token_type_ids = encodings.get("token_type_ids", None)
    labels_tensor = torch.tensor(labels)
    
    if token_type_ids is not None:
        return TensorDataset(input_ids, attention_mask, token_type_ids, labels_tensor)
    else:
        return TensorDataset(input_ids, attention_mask, labels_tensor)

# Prepare datasets
print("Preparing datasets...")
train_dataset = prepare_dataset(train_df)
val_dataset = prepare_dataset(val_df)
test_dataset = prepare_dataset(test_df)

# Create sampler for handling class imbalance
if CLASS_WEIGHTS:
    train_labels = train_df['class'].values
    class_sample_counts = np.array([len(train_df[train_df['class'] == t]) for t in range(NUM_LABELS)])
    weights = 1. / class_sample_counts
    samples_weights = weights[train_labels]
    samples_weights = torch.from_numpy(samples_weights)
    sampler = WeightedRandomSampler(samples_weights.type('torch.DoubleTensor'), 
                                   len(samples_weights))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
else:
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Create remaining data loaders
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# Set up optimizer and scheduler
# Freeze embeddings for faster training
for param in model.bert.embeddings.parameters():
    param.requires_grad = False

# Set up optimizer with weight decay
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() 
                if not any(nd in n for nd in no_decay) and p.requires_grad], 
     'weight_decay': WEIGHT_DECAY},
    {'params': [p for n, p in model.named_parameters() 
                if any(nd in n for nd in no_decay) and p.requires_grad], 
     'weight_decay': 0.0}
]

optimizer = AdamW(optimizer_grouped_parameters, lr=LEARNING_RATE)

# Create learning rate scheduler
total_steps = len(train_loader) * EPOCHS
warmup_steps = int(total_steps * WARMUP_STEPS)
scheduler = get_linear_schedule_with_warmup(
    optimizer, 
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)

# Loss function with class weights
if class_weights_tensor is not None:
    class_weights_tensor = class_weights_tensor.to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights_tensor)
else:
    criterion = torch.nn.CrossEntropyLoss()

# Training function
def train_and_validate(model, train_loader, val_loader, optimizer, scheduler, criterion, epochs, patience):
    best_val_f1 = 0
    patience_counter = 0
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_acc': [],
        'val_f1': []
    }
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        # Training
        model.train()
        train_loss = 0
        train_steps = 0
        
        for batch in tqdm(train_loader, desc="Training"):
            # Handle different dataset formats (with or without token_type_ids)
            if len(batch) == 4:
                input_ids, attention_mask, token_type_ids, labels = batch
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                token_type_ids = token_type_ids.to(device)
                labels = labels.to(device)
                
                # Forward pass
                outputs = model(
                    input_ids=input_ids, 
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids
                )
            else:
                input_ids, attention_mask, labels = batch
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)
                
                # Forward pass
                outputs = model(
                    input_ids=input_ids, 
                    attention_mask=attention_mask
                )
            
            logits = outputs.logits
            loss = criterion(logits, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            train_steps += 1
        
        avg_train_loss = train_loss / train_steps
        history['train_loss'].append(avg_train_loss)
        print(f"Average training loss: {avg_train_loss:.4f}")
        
        # Validation
        model.eval()
        val_loss = 0
        val_steps = 0
        val_predictions = []
        val_labels_list = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                # Handle different dataset formats
                if len(batch) == 4:
                    input_ids, attention_mask, token_type_ids, labels = batch
                    input_ids = input_ids.to(device)
                    attention_mask = attention_mask.to(device)
                    token_type_ids = token_type_ids.to(device)
                    labels = labels.to(device)
                    
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids
                    )
                else:
                    input_ids, attention_mask, labels = batch
                    input_ids = input_ids.to(device)
                    attention_mask = attention_mask.to(device)
                    labels = labels.to(device)
                    
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
                
                logits = outputs.logits
                loss = criterion(logits, labels)
                
                val_loss += loss.item()
                val_steps += 1
                
                predictions = torch.argmax(logits, dim=-1).cpu().numpy()
                val_predictions.extend(predictions)
                val_labels_list.extend(labels.cpu().numpy())
        
        avg_val_loss = val_loss / val_steps
        history['val_loss'].append(avg_val_loss)
        
        val_accuracy = accuracy_score(val_labels_list, val_predictions)
        history['val_acc'].append(val_accuracy)
        
        val_f1 = f1_score(val_labels_list, val_predictions, average='weighted')
        history['val_f1'].append(val_f1)
        
        print(f"Validation - Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.4f}, F1: {val_f1:.4f}")
        
        # Save model if it's the best so far (based on F1 score)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            print(f"New best validation F1 score! Saving model...")
            model.save_pretrained(os.path.join(OUTPUT_DIR, "best_model"))
            tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "best_model"))
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")
            
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    # Plot training history
    plt.figure(figsize=(12, 8))
    
    # Plot loss
    plt.subplot(2, 1, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot metrics
    plt.subplot(2, 1, 2)
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.plot(history['val_f1'], label='Validation F1')
    plt.title('Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'training_history.png'))
    
    return history

# Train model
print("Starting training...")
history = train_and_validate(
    model, 
    train_loader, 
    val_loader, 
    optimizer, 
    scheduler, 
    criterion, 
    EPOCHS, 
    PATIENCE
)

# Load best model for evaluation
print("Loading best model for evaluation...")
best_model_path = os.path.join(OUTPUT_DIR, "best_model")
model = BertForSequenceClassification.from_pretrained(best_model_path)
model.to(device)

# Save final model (just in case the best model is different)
print("Saving final model...")
model.save_pretrained(os.path.join(OUTPUT_DIR, "final_model"))
tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "final_model"))

# Evaluate on test set
print("Evaluating on test set...")
model.eval()
test_predictions = []
test_labels_list = []
test_losses = []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Testing"):
        # Handle different dataset formats
        if len(batch) == 4:
            input_ids, attention_mask, token_type_ids, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            token_type_ids = token_type_ids.to(device)
            labels = labels.to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
        else:
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        
        logits = outputs.logits
        loss = criterion(logits, labels)
        test_losses.append(loss.item())
        
        predictions = torch.argmax(logits, dim=-1).cpu().numpy()
        test_predictions.extend(predictions)
        test_labels_list.extend(labels.cpu().numpy())

test_accuracy = accuracy_score(test_labels_list, test_predictions)
test_f1 = f1_score(test_labels_list, test_predictions, average='weighted')
report = classification_report(test_labels_list, test_predictions)

print(f"Test loss: {np.mean(test_losses):.4f}")
print(f"Test accuracy: {test_accuracy:.4f}")
print(f"Test F1 score: {test_f1:.4f}")
print(f"Classification report:\n{report}")

# Save test results
with open(os.path.join(OUTPUT_DIR, "test_results.txt"), "w") as f:
    f.write(f"Test loss: {np.mean(test_losses):.4f}\n")
    f.write(f"Test accuracy: {test_accuracy:.4f}\n")
    f.write(f"Test F1 score: {test_f1:.4f}\n\n")
    f.write(f"Classification report:\n{report}")

# Create results dataframe with per-class probabilities
test_texts = test_df["message"].tolist()
all_probs = []

# Re-run inference to get probabilities
model.eval()
with torch.no_grad():
    for batch in tqdm(test_loader, desc="Getting probabilities"):
        if len(batch) == 4:
            input_ids, attention_mask, token_type_ids, _ = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            token_type_ids = token_type_ids.to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
        else:
            input_ids, attention_mask, _ = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()
        all_probs.extend(probs)

# Create detailed results dataframe
results_df = pd.DataFrame({
    "text": test_texts,
    "true_label": test_labels_list,
    "predicted_label": test_predictions,
    "correct": [1 if pred == true else 0 for pred, true in zip(test_predictions, test_labels_list)],
    "prob_class_0": [prob[0] for prob in all_probs],
    "prob_class_1": [prob[1] for prob in all_probs],
    "prob_class_2": [prob[2] for prob in all_probs],
    "prob_class_3": [prob[3] for prob in all_probs],
    "confidence": [prob[pred] for prob, pred in zip(all_probs, test_predictions)]
})

# Analyze errors
errors_df = results_df[results_df["correct"] == 0]
print(f"Total errors: {len(errors_df)} ({len(errors_df)/len(results_df)*100:.2f}%)")

# Error analysis by class
error_by_class = {}
for true_label in range(NUM_LABELS):
    class_samples = results_df[results_df["true_label"] == true_label]
    class_errors = class_samples[class_samples["correct"] == 0]
    error_rate = len(class_errors) / len(class_samples) if len(class_samples) > 0 else 0
    error_by_class[true_label] = {
        "samples": len(class_samples),
        "errors": len(class_errors),
        "error_rate": error_rate
    }

print("\nError analysis by class:")
for class_id, stats in error_by_class.items():
    print(f"Class {class_id}: {stats['errors']}/{stats['samples']} errors ({stats['error_rate']*100:.2f}%)")

# Save detailed predictions
results_df.to_csv(os.path.join(OUTPUT_DIR, "detailed_test_predictions.csv"), index=False)
errors_df.to_csv(os.path.join(OUTPUT_DIR, "prediction_errors.csv"), index=False)

print(f"Fine-tuning complete! The best model is saved in: {os.path.join(OUTPUT_DIR, 'best_model')}")
print(f"Test predictions saved to {os.path.join(OUTPUT_DIR, 'detailed_test_predictions.csv')}")