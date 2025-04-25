import os
import torch
import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    classification_report, 
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

# Set paths and parameters
MODEL_PATH = "./bert_checkpoints/final_model"  # Or "./bert_checkpoints/best_model"
TEST_FILE = "./pre_process/evaluation/test.csv"
OUTPUT_DIR = "./evaluation_results/bert"
MAX_LENGTH = 128
BATCH_SIZE = 32
CLASS_NAMES = ["Positive", "Mild", "Negative", "Irrelevant"]

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load the test data
print(f"Loading test data from {TEST_FILE}...")
test_df = pd.read_csv(TEST_FILE)
print(f"Loaded {len(test_df)} test examples")

# Check class distribution
class_distribution = test_df['class'].value_counts().sort_index()
print("Class distribution in test set:")
for class_id, count in class_distribution.items():
    print(f"Class {class_id} ({CLASS_NAMES[class_id]}): {count} samples ({count/len(test_df)*100:.2f}%)")

# Load model and tokenizer
print(f"Loading model and tokenizer from {MODEL_PATH}...")
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Using device: {device}")

# Tokenize test data
print("Tokenizing test data...")
test_texts = test_df["message"].tolist()
test_labels = test_df["class"].tolist()

# Tokenize in batches to avoid memory issues
all_predictions = []
all_confidences = []

model.eval()
with torch.no_grad():
    for i in tqdm(range(0, len(test_texts), BATCH_SIZE), desc="Evaluating"):
        batch_texts = test_texts[i:i+BATCH_SIZE]
        
        # Tokenize batch
        encoded_batch = tokenizer(
            batch_texts,
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )
        
        # Move to device
        input_ids = encoded_batch["input_ids"].to(device)
        attention_mask = encoded_batch["attention_mask"].to(device)
        
        # Get predictions
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        
        # Get predicted class and confidence
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        predictions = torch.argmax(logits, dim=1).cpu().numpy()
        confidences = torch.max(probabilities, dim=1)[0].cpu().numpy()
        
        all_predictions.extend(predictions)
        all_confidences.extend(confidences)

# Calculate metrics
accuracy = accuracy_score(test_labels, all_predictions)
precision = precision_score(test_labels, all_predictions, average='weighted')
recall = recall_score(test_labels, all_predictions, average='weighted')
f1 = f1_score(test_labels, all_predictions, average='weighted')

print(f"\n--- Overall Metrics ---")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Detailed classification report
report = classification_report(test_labels, all_predictions, target_names=CLASS_NAMES)
print("\n--- Classification Report ---")
print(report)

# Create confusion matrix
cm = confusion_matrix(test_labels, all_predictions)
print("\n--- Confusion Matrix ---")
print(cm)

# Save detailed results to CSV
results_df = pd.DataFrame({
    "message": test_texts,
    "true_label": test_labels,
    "true_label_name": [CLASS_NAMES[label] for label in test_labels],
    "predicted_label": all_predictions,
    "predicted_label_name": [CLASS_NAMES[pred] for pred in all_predictions],
    "confidence": all_confidences,
    "correct": [1 if pred == label else 0 for pred, label in zip(all_predictions, test_labels)]
})

# Save the results
results_path = os.path.join(OUTPUT_DIR, "evaluation_results.csv")
results_df.to_csv(results_path, index=False)
print(f"\nSaved detailed results to {results_path}")

# Create visualizations
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))

# Create class accuracy bar chart
class_accuracies = []
for i in range(len(CLASS_NAMES)):
    class_indices = [idx for idx, label in enumerate(test_labels) if label == i]
    if class_indices:
        class_preds = [all_predictions[idx] for idx in class_indices]
        class_true = [test_labels[idx] for idx in class_indices]
        class_acc = accuracy_score(class_true, class_preds)
        class_accuracies.append(class_acc)
    else:
        class_accuracies.append(0)

plt.figure(figsize=(10, 6))
plt.bar(CLASS_NAMES, class_accuracies)
plt.xlabel('Class')
plt.ylabel('Accuracy')
plt.title('Accuracy by Class')
plt.ylim(0, 1)
for i, v in enumerate(class_accuracies):
    plt.text(i, v + 0.05, f'{v:.2f}', ha='center')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "class_accuracies.png"))

# Look at incorrect predictions
incorrect_df = results_df[results_df["correct"] == 0].copy()
incorrect_df = incorrect_df.sort_values(by="confidence", ascending=False)

# Save top high-confidence mistakes
high_conf_mistakes = incorrect_df.head(20)
high_conf_mistakes.to_csv(os.path.join(OUTPUT_DIR, "high_confidence_mistakes.csv"), index=False)

# Summary metrics file
with open(os.path.join(OUTPUT_DIR, "metrics_summary.txt"), "w") as f:
    f.write(f"--- BERT Model Evaluation Results ---\n\n")
    f.write(f"Model path: {MODEL_PATH}\n")
    f.write(f"Test data: {TEST_FILE} ({len(test_df)} examples)\n\n")
    f.write(f"Overall Accuracy: {accuracy:.4f}\n")
    f.write(f"Weighted Precision: {precision:.4f}\n")
    f.write(f"Weighted Recall: {recall:.4f}\n")
    f.write(f"Weighted F1 Score: {f1:.4f}\n\n")
    f.write(f"Classification Report:\n{report}\n\n")
    f.write(f"Confusion Matrix:\n{cm}\n\n")
    f.write(f"Class-wise Accuracy:\n")
    for i, name in enumerate(CLASS_NAMES):
        f.write(f"{name}: {class_accuracies[i]:.4f}\n")
    f.write(f"\nErrors Analysis:\n")
    f.write(f"Total errors: {len(incorrect_df)} ({len(incorrect_df)/len(test_df)*100:.2f}% of test set)\n")
    
    # Create error transition matrix (from->to)
    f.write(f"\nTransition Matrix (From True -> To Predicted):\n")
    transition = {}
    for true_name, pred_name in zip(incorrect_df["true_label_name"], incorrect_df["predicted_label_name"]):
        key = f"{true_name} -> {pred_name}"
        transition[key] = transition.get(key, 0) + 1
    
    for key, count in sorted(transition.items(), key=lambda x: x[1], reverse=True):
        f.write(f"{key}: {count} ({count/len(incorrect_df)*100:.2f}% of errors)\n")

print(f"\nEvaluation complete! All results saved to {OUTPUT_DIR}")
print(f"To compare with baseline performance, check the metrics summary at {os.path.join(OUTPUT_DIR, 'metrics_summary.txt')}")