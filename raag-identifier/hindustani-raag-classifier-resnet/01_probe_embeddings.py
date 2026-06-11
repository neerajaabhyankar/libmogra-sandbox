# Stage 0 sanity probe (see plan.md).
#
# Question: are the precomputed crc-jeevster `clip_mean` (300-dim) embeddings for the
# 5-class Hindustani subset separable at all by a small MLP head?
#
# - Loads outputs/2s/crc-jeevster/{train,test}_<idx>.npz from embeddings-exploration/
# - Matches each <idx> to its raag label via the HF dataset
# - Stratified 5-fold CV over the 123 train clips with a small MLP (Linear -> ReLU ->
#   Dropout -> Linear), features standardized per fold
# - Reports per-fold accuracy/macro-F1, an aggregated 5x5 confusion matrix (saved as
#   PNG), and a majority-class baseline for reference
# - Also trains on the full 123 train clips and reports on the 8 official test clips
#   (too small/unbalanced to be a real held-out set, but reported for reference)

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from probe_common import load_split, MLPHead, LABEL_NAMES

OUT_DIR = Path(__file__).parent / "outputs" / "probe"

SEED = 0
N_FOLDS = 5
HIDDEN_DIM = 128
DROPOUT = 0.2
EPOCHS = 200
LR = 1e-3
WEIGHT_DECAY = 1e-2
PATIENCE = 20


def train_one(X_train, y_train, X_val, y_val, num_classes):
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)

    Xt = torch.tensor(X_train, dtype=torch.float32)
    yt = torch.tensor(y_train, dtype=torch.long)
    Xv = torch.tensor(X_val, dtype=torch.float32)
    yv = torch.tensor(y_val, dtype=torch.long)

    torch.manual_seed(SEED)
    model = MLPHead(X_train.shape[1], num_classes, hidden_dims=(HIDDEN_DIM,), dropout=DROPOUT)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_state = None
    epochs_since_improvement = 0

    for epoch in range(EPOCHS):
        model.train()
        opt.zero_grad()
        logits = model(Xt)
        loss = loss_fn(logits, yt)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(Xv), yv).item()

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1
            if epochs_since_improvement >= PATIENCE:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_preds = model(Xv).argmax(dim=1).numpy()

    return val_preds, model, scaler


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    X, y = load_split("train")
    num_classes = len(LABEL_NAMES)
    print(f"Loaded {X.shape[0]} train clips, {X.shape[1]}-dim embeddings, "
          f"{num_classes} classes")

    majority_class = np.bincount(y).argmax()
    majority_acc = (y == majority_class).mean()
    print(f"Majority-class baseline accuracy: {majority_acc:.3f} "
          f"(class {majority_class}={LABEL_NAMES[majority_class]})")

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    all_val_preds = np.zeros_like(y)
    fold_accs, fold_f1s = [], []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        val_preds, _, _ = train_one(X[train_idx], y[train_idx], X[val_idx], y[val_idx], num_classes)
        all_val_preds[val_idx] = val_preds

        acc = accuracy_score(y[val_idx], val_preds)
        f1 = f1_score(y[val_idx], val_preds, average="macro")
        fold_accs.append(acc)
        fold_f1s.append(f1)
        print(f"Fold {fold}: acc={acc:.3f} macro-F1={f1:.3f} (n_val={len(val_idx)})")

    print(f"\nCV mean accuracy: {np.mean(fold_accs):.3f} +/- {np.std(fold_accs):.3f}")
    print(f"CV mean macro-F1: {np.mean(fold_f1s):.3f} +/- {np.std(fold_f1s):.3f}")

    cm = confusion_matrix(y, all_val_preds, labels=range(num_classes))
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Stage 0: aggregated CV confusion matrix (crc-jeevster clip_mean + MLP head)")
    plt.tight_layout()
    out_path = OUT_DIR / "stage0_cv_confusion_matrix.png"
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved confusion matrix to {out_path}")

    # Train on full train set, evaluate on the official (small, unbalanced) test split.
    print("\n--- Reference: train on all 123 train clips, eval on 8 test clips ---")
    X_test, y_test = load_split("test")
    print(f"Test set: {X_test.shape[0]} clips, label counts "
          f"{dict(zip(*np.unique(y_test, return_counts=True)))}")

    scaler = StandardScaler().fit(X)
    X_train_s = scaler.transform(X)
    X_test_s = scaler.transform(X_test)

    Xt = torch.tensor(X_train_s, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.long)
    torch.manual_seed(SEED)
    model = MLPHead(X.shape[1], num_classes, hidden_dims=(HIDDEN_DIM,), dropout=DROPOUT)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(EPOCHS):
        model.train()
        opt.zero_grad()
        loss = loss_fn(model(Xt), yt)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        test_preds = model(torch.tensor(X_test_s, dtype=torch.float32)).argmax(dim=1).numpy()

    test_acc = accuracy_score(y_test, test_preds)
    print(f"Test accuracy: {test_acc:.3f} ({(y_test == test_preds).sum()}/{len(y_test)})")
    print(f"True:      {[LABEL_NAMES[l] for l in y_test]}")
    print(f"Predicted: {[LABEL_NAMES[l] for l in test_preds]}")


if __name__ == "__main__":
    main()
