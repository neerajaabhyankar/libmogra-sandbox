#!/usr/bin/env python3
"""
Training and evaluation script for Path Scoring Model
"""

# ============================================================================
# Imports
# ============================================================================

from pathlib import Path
from imports import *

# dataset
from make_dataset import all_samples_train, all_samples_val
# scoring model
from models import PathScoringModel, contrastive_loss
# dataloading
from dataloaders import PathPairDataset, collate_fn, pad_and_stack


def main(expt_name):
    # Seed
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    np.random.seed(42)

    # ============================================================================
    # Setup Experiment Directory
    # ============================================================================
    expt_dir = Path(expt_name)
    expt_dir.mkdir(parents=True, exist_ok=True)
    print(f"Experiment directory: {expt_dir}")

    # ============================================================================
    # Load Dataset
    # ============================================================================
    print("Loading datasets...")
    global all_samples_train, all_samples_val
    train_dataloader = DataLoader(
        PathPairDataset(all_samples_train),
        batch_size=8,
        shuffle=True,
        collate_fn=collate_fn
    )
    val_dataloader = DataLoader(
        PathPairDataset(all_samples_val),
        batch_size=8,
        shuffle=True,
        collate_fn=collate_fn
    )
    # Sanity check: look at a batch
    # model = PathScoringModel()
    # for batch in train_dataloader:
    #     scores = model.forward(batch['good_feats'], batch['good_len'], batch['bad_feats'], batch['bad_len'])
    #     print("Scores shape:", scores[0].shape, scores[1].shape)  # [B]
    #     break
    # breakpoint()
    print(f"Train batches: {len(train_dataloader)}, Val batches: {len(val_dataloader)}")
    
    """
    DEBUGGING: Can we interpolate random labels?
    --> Yes! ROC AUC > 0.9 even on the validation set o_o
    """
    all_samples_train_shuffled = []
    for phrase in all_samples_train:
        goods = len(phrase["good_paths"])
        bads = len(phrase["bad_paths"])
        alls = goods + bads
        all_paths = phrase["good_paths"] + phrase["bad_paths"]
        # DEBUGGING: change class imbalance
        new_good_indices = np.random.choice(range(alls), size=alls//2, replace=False)
        new_good_paths = [all_paths[ii] for ii in range(alls) if ii in new_good_indices]
        new_bad_paths = [all_paths[ii] for ii in range(alls) if ii not in new_good_indices]
        all_samples_train_shuffled.append({
            "good_paths": new_good_paths,
            "bad_paths": new_bad_paths,
        })
    all_samples_train = all_samples_train_shuffled

    # Randomize validation labels too -- although this shouldn't be necessary
    all_samples_val_shuffled = []
    for phrase in all_samples_val:
        goods = len(phrase["good_paths"])
        bads = len(phrase["bad_paths"])
        alls = goods + bads
        all_paths = phrase["good_paths"] + phrase["bad_paths"]
        # DEBUGGING: change class imbalance
        new_good_indices = np.random.choice(range(alls), size=alls//2, replace=False)
        new_good_paths = [all_paths[ii] for ii in range(alls) if ii in new_good_indices]
        new_bad_paths = [all_paths[ii] for ii in range(alls) if ii not in new_good_indices]
        all_samples_val_shuffled.append({
            "good_paths": new_good_paths,
            "bad_paths": new_bad_paths,
        })
    all_samples_val = all_samples_val_shuffled

    # ============================================================================
    # Load Model + Optimizer
    # ============================================================================
    print("Initializing model...")
    model = PathScoringModel()
    # model.to("mps")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 20
    # Sanity check: process a batch
    # for batch in train_dataloader:
    #     good_path_feats, good_lens = batch['good_feats'], batch['good_len']
    #     bad_path_feats, bad_lens = batch['bad_feats'], batch['bad_len']
    #     score_good, score_bad = model(good_path_feats, good_lens, bad_path_feats, bad_lens)
    #     loss = contrastive_loss(score_good, score_bad)
    #     loss.backward()
    #     optimizer.step()
    #     break
    # breakpoint()

    # ============================================================================
    # Train
    # ============================================================================
    print("Starting training...")

    train_losses = []
    val_losses = []
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for batch in train_dataloader:
            optimizer.zero_grad()

            good_path_feats, good_lens = batch['good_feats'], batch['good_len']
            bad_path_feats, bad_lens = batch['bad_feats'], batch['bad_len']

            score_good, score_bad = model(good_path_feats, good_lens, bad_path_feats, bad_lens)
            loss = contrastive_loss(score_good, score_bad)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_batch in val_dataloader:
                val_good_feats, val_good_lens = val_batch['good_feats'], val_batch['good_len']
                val_bad_feats, val_bad_lens = val_batch['bad_feats'], val_batch['bad_len']

                val_score_good, val_score_bad = model(val_good_feats, val_good_lens, val_bad_feats, val_bad_lens)
                val_loss += contrastive_loss(val_score_good, val_score_bad).item()

        epoch_train_loss = total_loss / len(train_dataloader)
        epoch_val_loss = val_loss / len(val_dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_train_loss}, Val Loss: {epoch_val_loss}")
        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)

    # Plot training and validation losses
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.legend()
    plt.ylabel('Contrastive Loss')
    plt.xlabel('Epoch')
    plt.savefig(expt_dir / 'training_loss.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved training loss plot to {expt_dir / 'training_loss.png'}")

    # ============================================================================
    # Eval
    # ============================================================================
    print("Evaluating model...")
    model.eval()
    # get scores for all paths
    good_path_scores = []
    bad_path_scores = []
    for batch in val_dataloader:
        good_path_feats, good_lens = batch["good_feats"], batch["good_len"]
        bad_path_feats, bad_lens = batch["bad_feats"], batch["bad_len"]

        score_good, score_bad = model(good_path_feats, good_lens, bad_path_feats, bad_lens)
        good_path_scores.extend(score_good.detach().numpy())
        bad_path_scores.extend(score_bad.detach().numpy())
        # DEBUGGING: random results give a bad score
        # score_good, score_bad = np.random.randn(8), np.random.randn(8)
        # good_path_scores.extend(score_good)
        # bad_path_scores.extend(score_bad)

    # Scatter plot of scores
    plt.figure()
    plt.scatter(good_path_scores, np.ones(len(good_path_scores)), label='Good Path Scores', marker='o')
    plt.scatter(bad_path_scores, np.zeros(len(bad_path_scores)), label='Bad Path Scores', marker='x')
    plt.xlabel('Predicted Path Score')
    plt.ylabel('Ground truth Path Score')
    plt.ylim(-1, 2)
    plt.yticks([0, 1])
    plt.legend()
    plt.savefig(expt_dir / 'score_scatter.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved score scatter plot to {expt_dir / 'score_scatter.png'}")
    
    # ===============
    # Non Pairwise
    # ===============
    # measure cross entropy loss using good_path_scores, bad_path_scores
    # good_path_scores have label 1, bad_path_scores have label 0
    good_labels = torch.ones(len(good_path_scores))
    bad_labels = torch.zeros(len(bad_path_scores))
    all_scores = torch.tensor(good_path_scores + bad_path_scores, dtype=torch.float32)
    all_labels = torch.cat([good_labels, bad_labels])
    auc_score = roc_auc_score(all_labels.numpy(), all_scores.numpy())
    print(f"Classification AUC: {auc_score}")

    # plot roc curve
    fpr, tpr, thresholds = roc_curve(all_labels.numpy(), all_scores.numpy())
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC={auc_score:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guessing')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.savefig(expt_dir / 'roc_curve.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved ROC curve to {expt_dir / 'roc_curve.png'}")
    
    # # ===============
    # # Pairwise
    # # ===============
    # good_path_scores = np.array(good_path_scores)
    # bad_path_scores = np.array(bad_path_scores)
    
    # pairwise_clamped = np.clip(good_path_scores - bad_path_scores, 0, 1)
    # pairwise_labels = np.ones_like(pairwise_clamped)
    # # create the negative class by duplicating this (??)
    # all_pairwise_scores = np.concatenate([pairwise_clamped, 1 - pairwise_clamped])
    # all_pairwise_labels = np.concatenate([pairwise_labels, 1 - pairwise_labels])
    
    # auc_score = roc_auc_score(all_pairwise_labels, all_pairwise_scores)
    # print(f"Pairwise Ranking AUC: {auc_score}")

    # # plot roc curve
    # fpr, tpr, thresholds = roc_curve(all_pairwise_labels, all_pairwise_scores)
    # plt.figure(figsize=(8, 6))
    # plt.plot(fpr, tpr, label=f'ROC Curve (AUC={auc_score:.4f})')
    # plt.plot([0, 1], [0, 1], 'k--', label='Random Guessing')
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('ROC Curve - Pairwise Ranking')
    # plt.legend()
    # plt.savefig(expt_dir / 'roc_curve.png', dpi=150, bbox_inches='tight')
    # plt.close()
    # print(f"Saved ROC curve to {expt_dir / 'roc_curve.png'}")

    # ============================================================================
    # Save Model
    # ============================================================================
    model_path = expt_dir / f"{expt_name}.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Saved model to {model_path}")

    # ============================================================================
    # Inspect Model
    # ============================================================================
    print("Inspecting model with UMAP visualization...")
    # probe the model's GRU layer..
    # pass all samples through the model and get the hidden state
    gru_hidden_states_good = []
    gru_hidden_states_bad = []

    for traj in all_samples_train:
        good_paths = traj["good_paths"]
        bad_paths = traj["bad_paths"]
        good_feats, good_lens = pad_and_stack(good_paths)
        bad_feats, bad_lens = pad_and_stack(bad_paths)
        with torch.no_grad():
            _, h_good = model.path_encoder(model.node_proj(good_feats))
            _, h_bad = model.path_encoder(model.node_proj(bad_feats))
        gru_hidden_states_good.extend(h_good.squeeze(0).numpy())
        gru_hidden_states_bad.extend(h_bad.squeeze(0).numpy())

    gru_hidden_states_good = np.array(gru_hidden_states_good)
    gru_hidden_states_bad = np.array(gru_hidden_states_bad)

    # UMAP visualization (conditionally available)
    try:
        import umap
        UMAP_AVAILABLE = True
    except ImportError:
        UMAP_AVAILABLE = False
        print("UMAP not available - skipping visualization")
        print(f"\nExperiment complete! All other results saved to: {expt_dir}/")
        return

    all_hidden_states = np.vstack([gru_hidden_states_good, gru_hidden_states_bad])
    umap_model = umap.UMAP(n_components=2, random_state=42)
    umap_embeddings = umap_model.fit_transform(all_hidden_states)

    umap_good = umap_embeddings[:len(gru_hidden_states_good)]
    umap_bad = umap_embeddings[len(gru_hidden_states_good):]

    plt.figure(figsize=(10, 6))
    plt.scatter(umap_good[:, 0], umap_good[:, 1], label='Good Paths', marker='o', alpha=0.6)
    plt.scatter(umap_bad[:, 0], umap_bad[:, 1], label='Bad Paths', marker='x', alpha=0.6)
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.title('UMAP Visualization of GRU Hidden States (Train)')
    plt.legend()
    plt.savefig(expt_dir / 'umap_train.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved UMAP train visualization to {expt_dir / 'umap_train.png'}")

    # Transform validation data
    gru_hidden_states_good_val = []
    gru_hidden_states_bad_val = []

    for traj in all_samples_val:
        good_paths = traj["good_paths"]
        bad_paths = traj["bad_paths"]
        good_feats, good_lens = pad_and_stack(good_paths)
        bad_feats, bad_lens = pad_and_stack(bad_paths)
        with torch.no_grad():
            _, h_good = model.path_encoder(model.node_proj(good_feats))
            _, h_bad = model.path_encoder(model.node_proj(bad_feats))
        gru_hidden_states_good_val.extend(h_good.squeeze(0).numpy())
        gru_hidden_states_bad_val.extend(h_bad.squeeze(0).numpy())

    gru_hidden_states_good_val = np.array(gru_hidden_states_good_val)
    gru_hidden_states_bad_val = np.array(gru_hidden_states_bad_val)

    umap_good_val = umap_model.transform(gru_hidden_states_good_val)
    umap_bad_val = umap_model.transform(gru_hidden_states_bad_val)

    plt.figure(figsize=(10, 6))
    plt.scatter(umap_good_val[:, 0], umap_good_val[:, 1], label='Good Paths', marker='o', alpha=0.6)
    plt.scatter(umap_bad_val[:, 0], umap_bad_val[:, 1], label='Bad Paths', marker='x', alpha=0.6)
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.title('UMAP Visualization of GRU Hidden States (Val)')
    plt.legend()
    plt.savefig(expt_dir / 'umap_val.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved UMAP val visualization to {expt_dir / 'umap_val.png'}")

    print(f"\nExperiment complete! All results saved to: {expt_dir}/")
    return


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train and evaluate Path Scoring Model')
    parser.add_argument(
        '--expt_name', required=True,
        type=str, help='Name of the experiment (used for saving results)'
    )

    args = parser.parse_args()
    main(args.expt_name)
