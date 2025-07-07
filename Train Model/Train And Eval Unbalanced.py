import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import csv
from datetime import datetime
from collections import Counter
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

# === Config ===
train_json = "self train/train_split.json"           # Original imbalanced training dataset
val_json_balanced   = "new train/val_split_balanced.json"
val_json_unbalanced = "self train/val_split.json"
batch_size = 4
epochs     = 10
num_frames = 20
device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Focal Loss ===
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        # Calculate Binary Cross Entropy
        BCE = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        # Calculate p_t
        p_t = torch.where(targets == 1, torch.sigmoid(logits), 1 - torch.sigmoid(logits))
        # Apply Focal Loss formula
        return (self.alpha * (1 - p_t) ** self.gamma * BCE).mean()

# === Dataset ===
class FrameSequenceDataset(Dataset):
    def __init__(self, index_file, transform=None):
        with open(index_file, 'r') as f:
            self.data = json.load(f)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        frames_dir = entry["frame_dir"]
        # Get up to num_frames frame names
        frame_names = entry.get("frames", [])[:num_frames]
        label = entry["label"]

        images = []
        for fn in frame_names:
            path = os.path.join(frames_dir, fn)
            img = Image.open(path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            images.append(img)

        # Pad with black images if the number of frames is insufficient
        if len(images) < num_frames:
            # Assuming all images have the same shape after transform
            C, H, W = images[0].shape
            for _ in range(num_frames - len(images)):
                images.append(torch.zeros(C, H, W))

        video_tensor = torch.stack(images)  # [T, C, H, W]
        return video_tensor, torch.tensor(label, dtype=torch.float32)

# === Model ===
class CNN_LSTM(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        # Load a pre-trained ResNet18 model
        base_cnn = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # Replace the final fully connected layer with an identity layer to get features
        base_cnn.fc = nn.Identity()
        self.cnn = base_cnn
        # LSTM layer to process sequences of features
        self.lstm = nn.LSTM(input_size=512, hidden_size=hidden_dim, batch_first=True) # ResNet18 features are 512-dim
        # Final fully connected layer for classification
        self.fc = nn.Sequential(
            nn.Dropout(0.5), # Dropout for regularization
            nn.Linear(hidden_dim, 1), # Output a single logit for binary classification
        )

    def forward(self, x):  # x: [B, T, C, H, W] (Batch, Time/Frames, Channels, Height, Width)
        B, T, C, H, W = x.size()
        # Reshape input for CNN processing (treat all frames in a batch as individual images)
        x = x.view(B * T, C, H, W)
        with torch.no_grad(): # Do not track gradients for CNN features (assuming pre-trained and frozen)
            feats = self.cnn(x)
        # Reshape features back to sequence for LSTM
        feats = feats.view(B, T, -1) # [B, T, feature_dim]
        # Pass features through LSTM
        lstm_out, _ = self.lstm(feats)
        # Take the output from the last time step of LSTM
        out = self.fc(lstm_out[:, -1, :])
        return out.squeeze(1) # Remove the singleton dimension

# === Evaluation Function ===
import time

def evaluate(model, json_path, transform, threshold=None):
    dataset = FrameSequenceDataset(json_path, transform)
    loader = DataLoader(dataset, batch_size=1, shuffle=False) # Batch size 1 for per-video timing
    probs, labels = [], []

    total_video_time = 0.0
    total_inf_time = 0.0
    model.eval() # Set model to evaluation mode
    with torch.no_grad(): # Disable gradient calculation during inference
        for clips, lbl in loader:
            # Start timing per-video processing
            t0 = time.time()

            clips = clips.to(device)
            # Inference timing
            t_inf_start = time.time()
            logit = model(clips)
            inf_dur = time.time() - t_inf_start
            total_inf_time += inf_dur

            prob = torch.sigmoid(logit).item() # Convert logit to probability
            probs.append(prob)
            labels.append(int(lbl.item()))

            # End timing per-video
            total_video_time += (time.time() - t0)

    probs = np.array(probs)
    labels = np.array(labels)

    # Compute average times
    num_samples = len(labels)
    avg_video_time = total_video_time / num_samples
    avg_face_time = total_inf_time / (num_samples * num_frames) # Assuming num_frames faces per video
    print(f"Average video processing time per sample: {avg_video_time:.3f} s")
    print(f"Average face prediction time per frame: {avg_face_time:.3f} s")

    # ROC method to select threshold (Youden's J statistic)
    fpr, tpr, roc_thr = roc_curve(labels, probs)
    youden = tpr - fpr
    idx_roc = np.argmax(youden)
    thr_roc = roc_thr[idx_roc]

    # PR curve method to select threshold (maximize F1-score)
    prec, rec, pr_thr = precision_recall_curve(labels, probs)
    f1_scores = 2 * prec * rec / (prec + rec + 1e-8) # Add small epsilon to avoid division by zero
    idx_pr = np.argmax(f1_scores[:-1]) # Exclude the last point which corresponds to precision/recall of 0
    thr_pr = pr_thr[idx_pr]

    # Choose threshold
    if threshold is None:
        threshold = thr_roc # Default to Youden's J threshold if not specified

    print(f"== Eval on {os.path.basename(json_path)} ==")
    print(f" Youden's J best-thr: {thr_roc:.3f}, max-F1 best-thr: {thr_pr:.3f}")
    print(f" Using threshold = {threshold:.3f}")

    # Make predictions based on the chosen threshold
    preds = (probs >= threshold).astype(int)
    # Compute and display confusion matrix
    cm = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Real", "Fake"])
    disp.plot(cmap="Blues")
    plt.title(f"CM_{os.path.basename(json_path)}_(thr={threshold:.3f})")
    plt.savefig(f"cm_{os.path.basename(json_path)}.png")
    # Print classification report (precision, recall, f1-score, support)
    print(classification_report(labels, preds, target_names=["Real", "Fake"]))

# === Main ===
if __name__ == "__main__":
    # 1) Build data loaders
    transform = transforms.Compose([
        transforms.Resize((224, 224)), # Resize images to 224x224
        transforms.ToTensor(),         # Convert PIL Image to PyTorch Tensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # Normalize with ImageNet stats
    ])

    train_ds = FrameSequenceDataset(train_json, transform)
    val_ds_bal = FrameSequenceDataset(val_json_balanced, transform)
    val_ds_unbal = FrameSequenceDataset(val_json_unbalanced, transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader_bal = DataLoader(val_ds_bal, batch_size=batch_size, shuffle=False)
    val_loader_unbal = DataLoader(val_ds_unbal, batch_size=batch_size, shuffle=False)

    # 2) Open CSV log file and write header
    log_csv = "training_log_unbalanced.csv"
    with open(log_csv, "w", newline="") as f:
        csv.writer(f).writerow([
            "epoch", "train_loss", "train_acc",
            "val_acc_balanced", "val_acc_unbalanced"
        ])

    # 3) Initialize model, loss function, and optimizer
    model = CNN_LSTM().to(device) # Move model to appropriate device (CPU/GPU)
    criterion = FocalLoss(gamma=2.0, alpha=0.9) # Initialize Focal Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4) # Adam optimizer with learning rate

    # 4) Training loop
    best_val_acc = 0.0
    for ep in range(1, epochs + 1):
        model.train() # Set model to training mode
        total_loss, total_corr = 0, 0
        for clips, labels in tqdm(train_loader, desc=f"Epoch {ep} [Train]"):
            clips, labels = clips.to(device), labels.to(device) # Move data to device
            logits = model(clips) # Forward pass
            loss = criterion(logits, labels) # Calculate loss
            optimizer.zero_grad() # Zero gradients
            loss.backward() # Backpropagation
            optimizer.step() # Update weights

            total_loss += loss.item() * clips.size(0)
            # Calculate training accuracy
            total_corr += ((torch.sigmoid(logits) > 0.5) == labels.bool()).sum().item()

        train_acc = total_corr / len(train_ds)
        train_loss = total_loss / len(train_ds)

        # --- Calculate accuracy on balanced validation set (val_acc_balanced) ---
        model.eval() # Set model to evaluation mode
        total_val_corr = 0
        with torch.no_grad(): # Disable gradient calculation
            for clips_v, labels_v in val_loader_bal:
                clips_v, labels_v = clips_v.to(device), labels_v.to(device)
                logits_v = model(clips_v)
                preds_v = (torch.sigmoid(logits_v) > 0.5).int()
                total_val_corr += (preds_v.cpu() == labels_v.int().cpu()).sum().item()
        val_acc_balanced = total_val_corr / len(val_ds_bal)

        # --- Calculate accuracy on unbalanced validation set (val_acc_unbalanced) ---
        total_unbal_corr = 0
        with torch.no_grad(): # Disable gradient calculation
            for clips_u, labels_u in val_loader_unbal:
                clips_u, labels_u = clips_u.to(device), labels_u.to(device)
                logits_u = model(clips_u)
                preds_u = (torch.sigmoid(logits_u) > 0.5).int()
                total_unbal_corr += (preds_u.cpu() == labels_u.int().cpu()).sum().item()
        val_acc_unbalanced = total_unbal_corr / len(val_ds_unbal)

        print(
            f"Epoch {ep}: loss={train_loss:.4f} | train_acc={train_acc:.4f}"
            f" | val_acc_bal={val_acc_balanced:.4f} | val_acc_unbal={val_acc_unbalanced:.4f}"
        )

        # --- Append to CSV log ---
        with open(log_csv, "a", newline="") as f:
            csv.writer(f).writerow([
                ep, round(train_loss, 4), round(train_acc, 4),
                round(val_acc_balanced, 4), round(val_acc_unbalanced, 4)
            ])

        # Save the best model (based on balanced validation set performance)
        if val_acc_balanced > best_val_acc:
            best_val_acc = val_acc_balanced
            torch.save(model.state_dict(), "best_unbalanced_model.pth")

    # 5) Save the final model
    ckpt = "cnn_lstm_unbalanced_train.pth"
    torch.save(model.state_dict(), ckpt)
    print(f"✅ Model saved: {ckpt}")

    # 6) Evaluate on balanced and unbalanced validation sets
    print("\nEvaluating on balanced validation set:")
    evaluate(model, val_json_balanced, transform)
    print("\nEvaluating on unbalanced validation set:")
    evaluate(model, val_json_unbalanced, transform)