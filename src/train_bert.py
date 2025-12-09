"""
train_bert.py
--------------
Fine-tunes BERT (bert-base-uncased) for binary sentiment classification.

Assumes the following are already done in previous src modules:
- Clean text created
- BERT tokens created in df['bert_tokens_id']
- Train/test split already done
- Utility functions for padding & dataloaders already available

Returns:
    Trained BERT model, plus evaluation metrics.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc

from bert_utils import (
    pad_truncate,
    create_bert_tensors,
    create_bert_dataloaders
)


# --------------------------------------------
# 1. Load Preprocessed Inputs (IDs + Masks)
# --------------------------------------------

def prepare_bert_inputs(df, X_train, X_test, y_train, y_test, max_len=256):
    """
    Uses pre-existing bert_tokens_id column and splits into
    padded input_ids, attention_masks, and label arrays.
    """
    # Generate full padded matrix + attention masks (no repetition)
    input_ids_all, attention_masks_all = create_bert_tensors(df, max_len)

    # Use same indices as ML/LSTM split
    train_idx = X_train.index
    test_idx = X_test.index

    X_train_ids = input_ids_all[train_idx]
    X_train_masks = attention_masks_all[train_idx]
    y_train_bert = y_train.values.astype(np.int64)

    X_test_ids = input_ids_all[test_idx]
    X_test_masks = attention_masks_all[test_idx]
    y_test_bert = y_test.values.astype(np.int64)

    print("Train shapes:", X_train_ids.shape, X_train_masks.shape, y_train_bert.shape)
    print("Test shapes: ", X_test_ids.shape, X_test_masks.shape, y_test_bert.shape)

    return X_train_ids, X_train_masks, y_train_bert, X_test_ids, X_test_masks, y_test_bert


# --------------------------------------------
# 2. Training Loop
# --------------------------------------------

def train_bert(
    train_loader,
    val_loader,
    device,
    epochs=3,
    learning_rate=2e-5
):
    """
    Fine-tunes BERT using AdamW + linear warmup scheduler.
    """
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=2
    )
    model.to(device)

    # Optimizer + scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate, eps=1e-8)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    print("Model loaded — starting training...")

    best_val_loss = float("inf")
    best_model_path = "best_bert_model.pt"

    import time, datetime

    def format_time(t):
        return str(datetime.timedelta(seconds=int(round(t))))

    for epoch in range(epochs):
        print(f"\n======== Epoch {epoch+1}/{epochs} ========")

        t0 = time.time()
        model.train()
        total_train_loss = 0.0

        # -------------------------
        # Training batches
        # -------------------------
        for step, batch in enumerate(train_loader):
            b_ids = batch['input_ids'].to(device)
            b_mask = batch['attention_mask'].to(device)
            b_labels = batch['labels'].to(device)

            model.zero_grad()
            outputs = model(b_ids, attention_mask=b_mask, labels=b_labels)
            loss = outputs.loss
            total_train_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

            if (step + 1) % 200 == 0:
                print(f"  Batch {step+1}/{len(train_loader)} - Avg Loss: {total_train_loss/(step+1):.4f}")

        avg_train_loss = total_train_loss / len(train_loader)
        print(f"  Training Loss: {avg_train_loss:.4f}")
        print("  Epoch time:", format_time(time.time() - t0))

        # -------------------------
        # Validation
        # -------------------------
        model.eval()
        total_val_loss = 0.0

        for batch in val_loader:
            b_ids = batch['input_ids'].to(device)
            b_mask = batch['attention_mask'].to(device)
            b_labels = batch['labels'].to(device)

            with torch.no_grad():
                outputs = model(b_ids, attention_mask=b_mask, labels=b_labels)
                total_val_loss += outputs.loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        print(f"  Validation Loss: {avg_val_loss:.4f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            print("  Best model saved →", best_model_path)

    print("\nTraining complete.")
    return best_model_path


# --------------------------------------------
# 3. Evaluation on Test Set
# --------------------------------------------

def evaluate_bert(best_model_path, test_loader, device):
    """
    Loads the best saved model and evaluates on test set.
    """
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=2
    )
    model.load_state_dict(torch.load(best_model_path))
    model.to(device)
    model.eval()

    all_logits = []
    all_labels = []

    for batch in test_loader:
        b_ids = batch['input_ids'].to(device)
        b_mask = batch['attention_mask'].to(device)
        b_labels = batch['labels'].to(device)

        with torch.no_grad():
            outputs = model(b_ids, attention_mask=b_mask)
            logits = outputs.logits

        all_logits.append(logits.cpu().numpy())
        all_labels.append(b_labels.cpu().numpy())

    logits = np.concatenate(all_logits, axis=0)
    y_true = np.concatenate(all_labels, axis=0)

    # Convert logits → probabilities
    probs = torch.softmax(torch.tensor(logits), dim=1)[:, 1].numpy()
    y_pred = (probs >= 0.5).astype(int)

    # Metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    fpr, tpr, _ = roc_curve(y_true, probs)
    auc_score = auc(fpr, tpr)

    print("\n===== BERT Test Metrics =====")
    print("Accuracy :", acc)
    print("Precision:", prec)
    print("Recall   :", rec)
    print("F1 Score :", f1)
    print("AUC      :", auc_score)

    return y_pred, probs, y_true


# --------------------------------------------
# 4. Main Entry Point (Optional)
# --------------------------------------------

if __name__ == "__main__":
    print("⚠️ This script is intended to be imported, not run directly.")
