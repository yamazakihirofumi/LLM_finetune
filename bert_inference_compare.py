import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BertForSequenceClassification, BertConfig
from peft import PeftModel, PeftConfig
import pandas as pd
from tqdm.auto import tqdm
import os
import time
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Define paths and settings ---
original_model_path = "./bert-base-chinese"
finetuned_model_path = "./bert_checkpoints_improved/final_model"
test_data_path = "./pre_process/evaluation/test.csv"
output_dir = "./evaluation_results/bert_comparison"
os.makedirs(output_dir, exist_ok=True)
MAX_LENGTH = 128
BATCH_SIZE = 32
NUM_LABELS = 4  # 0: Positive, 1: Mild, 2: Negative, 3: Irrelevant
CLASS_NAMES = ["Positive", "Mild", "Negative", "Irrelevant"]

# --- 2. Load the test dataset ---
print(f"Loading test data from: {test_data_path}")
test_df = pd.read_csv(test_data_path)
print(f"Loaded {len(test_df)} test samples")

# Check class distribution
class_distribution = test_df['class'].value_counts().sort_index()
print("Class distribution in test set:")
for class_id, count in class_distribution.items():
    print(f"Class {class_id} ({CLASS_NAMES[class_id]}): {count} samples ({count/len(test_df)*100:.2f}%)")

# --- 3. Function to load models ---
def load_model(model_path, is_finetuned=False):
    print(f"Loading model from: {model_path}")
    
    try:
        # For fine-tuned LoRA models
        if is_finetuned:
            try:
                # First try loading as a PeftModel (for LoRA/QLoRA fine-tuned models)
                config = PeftConfig.from_pretrained(model_path)
                # Load the base model first
                base_model = AutoModelForSequenceClassification.from_pretrained(
                    config.base_model_name_or_path, 
                    num_labels=NUM_LABELS
                )
                model = PeftModel.from_pretrained(base_model, model_path)
                print(f"Loaded as PeftModel with base: {config.base_model_name_or_path}")
            except:
                # If not a PeftModel, try loading as a regular model
                model = AutoModelForSequenceClassification.from_pretrained(model_path)
                print("Loaded as regular fine-tuned model")
        else:
            # For original pre-trained models, we need to add a classification head
            config = BertConfig.from_pretrained(model_path, num_labels=NUM_LABELS)
            model = BertForSequenceClassification.from_pretrained(model_path, config=config)
            print("Loaded pre-trained model with classification head")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_path if not is_finetuned else original_model_path
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        return model, tokenizer
    
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

# --- 4. Function to run inference ---
def run_inference(model, tokenizer, texts, labels):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    all_predictions = []
    start_time = time.time()
    
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Inference"):
        batch_texts = texts[i:i+BATCH_SIZE]
        
        # Tokenize
        encoded_input = tokenizer(
            batch_texts,
            padding="max_length",
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        ).to(device)
        
        # Run inference
        with torch.no_grad():
            outputs = model(**encoded_input)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1).cpu().numpy()
        
        all_predictions.extend(predictions.tolist())
    
    end_time = time.time()
    inference_time = end_time - start_time
    samples_per_second = len(texts) / inference_time
    
    # Calculate metrics
    accuracy = accuracy_score(labels, all_predictions)
    report = classification_report(labels, all_predictions, target_names=CLASS_NAMES, output_dict=True)
    cm = confusion_matrix(labels, all_predictions)
    
    return {
        'predictions': all_predictions,
        'accuracy': accuracy,
        'report': report,
        'confusion_matrix': cm,
        'inference_time': inference_time,
        'samples_per_second': samples_per_second
    }

# --- 5. Load models and run inference ---
test_texts = test_df["message"].tolist()
true_labels = test_df["class"].tolist()

# Evaluate original model
try:
    print("\n=== Evaluating Original Model ===")
    original_model, original_tokenizer = load_model(original_model_path, is_finetuned=False)
    original_results = run_inference(original_model, original_tokenizer, test_texts, true_labels)
    print(f"Original model accuracy: {original_results['accuracy']:.4f}")
    print(f"Inference time: {original_results['inference_time']:.2f}s ({original_results['samples_per_second']:.2f} samples/s)")
except Exception as e:
    print(f"Error evaluating original model: {e}")
    original_results = None

# Evaluate fine-tuned model
try:
    print("\n=== Evaluating Fine-tuned Model ===")
    finetuned_model, finetuned_tokenizer = load_model(finetuned_model_path, is_finetuned=True)
    finetuned_results = run_inference(finetuned_model, finetuned_tokenizer, test_texts, true_labels)
    print(f"Fine-tuned model accuracy: {finetuned_results['accuracy']:.4f}")
    print(f"Inference time: {finetuned_results['inference_time']:.2f}s ({finetuned_results['samples_per_second']:.2f} samples/s)")
except Exception as e:
    print(f"Error evaluating fine-tuned model: {e}")
    finetuned_results = None

# --- 6. Visualize results ---
if original_results and finetuned_results:
    # Create comparison dataframe
    comparison_df = pd.DataFrame({
        'message': test_texts,
        'true_label': true_labels,
        'true_class': [CLASS_NAMES[label] for label in true_labels],
        'original_prediction': original_results['predictions'],
        'original_class': [CLASS_NAMES[pred] for pred in original_results['predictions']],
        'finetuned_prediction': finetuned_results['predictions'],
        'finetuned_class': [CLASS_NAMES[pred] for pred in finetuned_results['predictions']],
        'original_correct': [1 if pred == true else 0 for pred, true in zip(original_results['predictions'], true_labels)],
        'finetuned_correct': [1 if pred == true else 0 for pred, true in zip(finetuned_results['predictions'], true_labels)],
        'improvement': [1 if fine_pred == true and orig_pred != true else
                       (-1 if fine_pred != true and orig_pred == true else 0) 
                       for fine_pred, orig_pred, true in zip(
                            finetuned_results['predictions'], 
                            original_results['predictions'], 
                            true_labels)]
    })
    
    # Save comparison results
    comparison_df.to_csv(os.path.join(output_dir, "model_comparison.csv"), index=False)
    
    # Summary statistics
    improvement_counts = comparison_df['improvement'].value_counts()
    improved = improvement_counts.get(1, 0)
    worsened = improvement_counts.get(-1, 0)
    unchanged = improvement_counts.get(0, 0)
    
    # Per-class performance
    class_performance = pd.DataFrame({
        'Class': CLASS_NAMES,
        'Original Accuracy': [original_results['report'][name]['precision'] for name in CLASS_NAMES],
        'Fine-tuned Accuracy': [finetuned_results['report'][name]['precision'] for name in CLASS_NAMES],
        'Difference': [finetuned_results['report'][name]['precision'] - original_results['report'][name]['precision'] 
                      for name in CLASS_NAMES]
    })
    
    # Create visualizations
    plt.figure(figsize=(15, 10))
    
    # Overall accuracy comparison
    plt.subplot(2, 2, 1)
    plt.bar(['Original', 'Fine-tuned'], 
            [original_results['accuracy'], finetuned_results['accuracy']],
            color=['blue', 'green'])
    plt.title('Overall Accuracy Comparison')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    
    # Changes in predictions
    plt.subplot(2, 2, 2)
    plt.bar(['Improved', 'Worsened', 'Unchanged'], 
            [improved, worsened, unchanged],
            color=['green', 'red', 'gray'])
    plt.title('Changes in Predictions')
    plt.ylabel('Number of Samples')
    
    # Per-class accuracy
    plt.subplot(2, 2, 3)
    class_performance.plot(x='Class', y=['Original Accuracy', 'Fine-tuned Accuracy'], 
                          kind='bar', ax=plt.gca(), rot=0)
    plt.title('Per-class Accuracy')
    plt.ylim(0, 1)
    
    # Confusion matrix difference (fine-tuned - original)
    plt.subplot(2, 2, 4)
    cm_diff = finetuned_results['confusion_matrix'] - original_results['confusion_matrix']
    sns.heatmap(cm_diff, annot=True, fmt='d', cmap='coolwarm', 
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title('Confusion Matrix Difference (Fine-tuned - Original)')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "model_comparison.png"))
    
    # Save summary report
    with open(os.path.join(output_dir, "comparison_summary.txt"), 'w') as f:
        f.write("=== BERT Model Comparison ===\n\n")
        f.write(f"Original model: {original_model_path}\n")
        f.write(f"Fine-tuned model: {finetuned_model_path}\n")
        f.write(f"Test set: {test_data_path} ({len(test_df)} samples)\n\n")
        
        f.write("=== Performance Metrics ===\n")
        f.write(f"Original model accuracy: {original_results['accuracy']:.4f}\n")
        f.write(f"Fine-tuned model accuracy: {finetuned_results['accuracy']:.4f}\n")
        f.write(f"Absolute improvement: {finetuned_results['accuracy'] - original_results['accuracy']:.4f}\n")
        f.write(f"Relative improvement: {(finetuned_results['accuracy'] - original_results['accuracy'])/original_results['accuracy']*100:.2f}%\n\n")
        
        f.write("=== Inference Speed ===\n")
        f.write(f"Original model: {original_results['inference_time']:.2f}s ({original_results['samples_per_second']:.2f} samples/s)\n")
        f.write(f"Fine-tuned model: {finetuned_results['inference_time']:.2f}s ({finetuned_results['samples_per_second']:.2f} samples/s)\n\n")
        
        f.write("=== Prediction Changes ===\n")
        f.write(f"Improved: {improved} samples ({improved/len(test_df)*100:.2f}%)\n")
        f.write(f"Worsened: {worsened} samples ({worsened/len(test_df)*100:.2f}%)\n")
        f.write(f"Unchanged: {unchanged} samples ({unchanged/len(test_df)*100:.2f}%)\n\n")
        
        f.write("=== Per-class Performance ===\n")
        f.write(class_performance.to_string(index=False))
        
        f.write("\n\n=== Original Model Classification Report ===\n")
        for class_name in CLASS_NAMES:
            metrics = original_results['report'][class_name]
            f.write(f"{class_name}: Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}, F1={metrics['f1-score']:.4f}\n")
        
        f.write("\n=== Fine-tuned Model Classification Report ===\n")
        for class_name in CLASS_NAMES:
            metrics = finetuned_results['report'][class_name]
            f.write(f"{class_name}: Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}, F1={metrics['f1-score']:.4f}\n")

    print(f"\nComparison complete! Results saved to {output_dir}")
    print(f"Accuracy improvement: {finetuned_results['accuracy'] - original_results['accuracy']:.4f} ({(finetuned_results['accuracy'] - original_results['accuracy'])/original_results['accuracy']*100:.2f}%)")
    print(f"Improved predictions: {improved} samples ({improved/len(test_df)*100:.2f}%)")
    print(f"Worsened predictions: {worsened} samples ({worsened/len(test_df)*100:.2f}%)")

elif original_results:
    # Only original model worked
    print("Only original model evaluation was successful")
    # Save original model results
    results_df = pd.DataFrame({
        'message': test_texts,
        'true_label': true_labels,
        'predicted_label': original_results['predictions'],
        'correct': [1 if pred == true else 0 for pred, true in zip(original_results['predictions'], true_labels)]
    })
    results_df.to_csv(os.path.join(output_dir, "original_model_results.csv"), index=False)
    
    # Save metrics
    with open(os.path.join(output_dir, "original_model_metrics.txt"), 'w') as f:
        f.write(f"Model: {original_model_path}\n")
        f.write(f"Test set: {test_data_path}\n")
        f.write(f"Total samples: {len(test_df)}\n")
        f.write(f"Accuracy: {original_results['accuracy']:.4f}\n\n")
        f.write("Classification Report:\n")
        for class_name in CLASS_NAMES:
            metrics = original_results['report'][class_name]
            f.write(f"{class_name}: Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}, F1={metrics['f1-score']:.4f}\n")

elif finetuned_results:
    # Only fine-tuned model worked
    print("Only fine-tuned model evaluation was successful")
    # Save fine-tuned model results
    results_df = pd.DataFrame({
        'message': test_texts,
        'true_label': true_labels,
        'predicted_label': finetuned_results['predictions'],
        'correct': [1 if pred == true else 0 for pred, true in zip(finetuned_results['predictions'], true_labels)]
    })
    results_df.to_csv(os.path.join(output_dir, "finetuned_model_results.csv"), index=False)
    
    # Save metrics
    with open(os.path.join(output_dir, "finetuned_model_metrics.txt"), 'w') as f:
        f.write(f"Model: {finetuned_model_path}\n")
        f.write(f"Test set: {test_data_path}\n")
        f.write(f"Total samples: {len(test_df)}\n")
        f.write(f"Accuracy: {finetuned_results['accuracy']:.4f}\n\n")
        f.write("Classification Report:\n")
        for class_name in CLASS_NAMES:
            metrics = finetuned_results['report'][class_name]
            f.write(f"{class_name}: Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}, F1={metrics['f1-score']:.4f}\n")

else:
    print("Both model evaluations failed. Please check the error messages above.")

print("Evaluation complete!")