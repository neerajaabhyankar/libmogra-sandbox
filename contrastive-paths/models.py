from imports import *


class PathStats:
    """
    Draws statistics from the paths in the dataset using various methods
    Concatenates the stats into a single tensor for each path
    """
    def __init__(self):
        self.none = None

    def forward_path(self, node_feats_seq, path_lens):
        """
        node_feats_seq: [batch_size, max_path_len, node_feat_dim] raw node coordinates
        path_lens: [batch_size] lengths of each path for packing
        """
        batch_size, T, node_feat_dim = node_feats_seq.shape
        
        # 1. Curvature statistics
        """
        treat each path edge as a vector
        and compute the curvature statistics for each path
        include the average edge direction, std of edge directions, moments, etc.
        """
        edges = node_feats_seq[:, 1:T, :] - node_feats_seq[:, :T-1, :]  # (batch_size, T-1, node_feat_dim)
        # non_zero edges
        range_tensor = torch.arange(T-1, device=edges.device).unsqueeze(0)  # (1, T)
        mask = range_tensor < (path_lens-1).unsqueeze(1)
        mask = mask.unsqueeze(2)  # (batch_size, T, 1)
        masked_edges = edges * mask  # (batch_size, T, 3), zeroes out excess edges

        # compute the average edge direction
        counts = (path_lens - 1).clamp(min=1).float()
        mean_edge_length = masked_edges.sum(dim=1) / counts.unsqueeze(1)
        
        # compute the standard deviation of edge directions
        mean_expanded = mean_edge_length.unsqueeze(1)  # (B, 1)
        squared_diffs = (masked_edges - mean_expanded) ** 2  # (B, T)
        var = squared_diffs.sum(dim=1) / counts.unsqueeze(1)  # (B,)
        std = torch.sqrt(var)
        
        # # TODO(neeraja):
        # skewness ≈ ((x - mean)^3).sum(dim=1) / count / std^3
        # kurtosis ≈ ((x - mean)^4).sum(dim=1) / count / std^4
        
        stats_tensor = torch.cat([
            mean_edge_length,  # (B, 3)
            std,  # (B, 3)
        ], dim=1)
        return stats_tensor
    

class PathScoringModel(nn.Module):
    """
    A model that transforms sequences of node coordinates into a scalar score
    representing the "goodness" of the path.
    
    The model consists of:
    1. A node encoder that projects raw node coordinates (x, y, z) into
       a higher-dimensional embedding space.
    2. A path encoder that uses a GRU to process sequences of node embeddings
       and produce a fixed-size path embedding.
    3. A scoring head that projects the path embedding to a scalar score.
    """
    
    def __init__(self, node_feat_dim=3, hidden_dim=6, node_emb_dim=6, path_dim=6):
        super().__init__()
        
        # Node encoder: from (x, y, z) → embedding
        self.node_proj = nn.Sequential(
            nn.Linear(node_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_emb_dim)
        )  # Learnable

        # Path encoder: GRU over node embeddings
        self.path_encoder = nn.GRU(
            input_size=node_emb_dim,
            hidden_size=path_dim,
            batch_first=True
        )  # Learnable

        # Scoring head: project path embedding to scalar
        self.score_head = nn.Sequential(
            nn.Linear(path_dim, 1)
        )  # Learnable

    def forward_path(self, node_feats_seq, path_lens):
        """
        node_feats_seq: [batch_size, max_path_len, node_feat_dim] raw node coordinates
        path_lens: [batch_size] lengths of each path for packing
        """
        batch_size, T, _ = node_feats_seq.shape
        x = self.node_proj(node_feats_seq)  # [batch_size, max_path_len, node_emb_dim]

        packed = nn.utils.rnn.pack_padded_sequence(x, path_lens.cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.path_encoder(packed)  # h: [1, batch_size, path_dim]
        h = h.squeeze(0)  # [batch_size, path_dim]

        score = self.score_head(h).squeeze(-1)  # [batch_size]
        return score

    def forward(self, path1_feats, path1_lens, path2_feats, path2_lens):
        s1 = self.forward_path(path1_feats, path1_lens)
        s2 = self.forward_path(path2_feats, path2_lens)
        return s1, s2


def contrastive_loss(score1, score2, margin=1.0):
    # we want score1 (good) > score2 (bad)
    return F.relu(score2 - score1 + margin).mean()
