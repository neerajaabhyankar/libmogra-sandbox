import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.utils.data import Dataset, DataLoader
import numpy as np


class TemporalGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_time_steps):
        super(TemporalGNN, self).__init__()
        
        # Params
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_time_steps = num_time_steps
        
        # GRU to model temporal dependencies between node features
        self.gru = torch.nn.GRU(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        
        # Graph Convolutional Network layers for node embeddings
        self.gcn1 = GCNConv(2*hidden_dim, 2*hidden_dim)
        self.gcn2 = GCNConv(2*hidden_dim, 2*hidden_dim)
        
        # Output layer for node classification
        self.classifier = torch.nn.Linear(2*hidden_dim, 1)

    def forward(self, x, edges, masks):
        """
        x: (batch_size, num_nodes, num_time_steps, input_dim)
        edges: (num_nodes, num_nodes)
        masks: (num_nodes, num_classes)
        """
        batch_size, NN, tt, ff = x.shape
        assert NN == len(masks)
        assert ff == self.input_dim
        assert tt == self.num_time_steps
        edges_b = edges.unsqueeze(0).repeat(batch_size, 1, 1)
        
        # Step 1: Use Bidirectional GRU to capture temporal dependencies
        # gru expects shape (batch_size * NN, num_time_steps, input_dim)
        gru_out, _ = self.gru(x.view(batch_size * NN, self.num_time_steps, self.input_dim))
        # gru_out: reshape to (batch_size, NN, num_time_steps, 2*hidden_dim)
        gru_out = gru_out.view(batch_size, NN, self.num_time_steps, 2*self.hidden_dim)
        
        # Step 2: Aggregate temporal information with mean pooling over time
        # final_node_embeddings: shape (batch_size, NN, 2*hidden_dim)
        temporal_node_embeddings = gru_out.mean(dim=2)
        
        # Step 3: Apply GCN layers
        node_embeddings = F.relu(self.gcn1(temporal_node_embeddings, edges_b))  # Shape: (batch_size, NN, 2*hidden_dim)
        node_embeddings = F.relu(self.gcn2(node_embeddings, edges_b))  # Shape: (batch_size, NN, 2*hidden_dim)
        # node_embeddings_t = F.relu(self.gcn3(node_embeddings_t, edges_b))  # Shape: (batch_size, NN, hidden_dim)
        # node_embeddings_t = F.relu(self.gcn4(node_embeddings_t, edges_b))  # Shape: (batch_size, NN, hidden_dim)
        
        # Step 4: Apply a linear classifier to predict for each node
        # logits: shape (batch_size, NN)
        logits = self.classifier(node_embeddings)
        logits = logits.squeeze(2)

        # Step 5: Constraint-aware node classification
        # Apply softmax over nodes in each equivalence class independently
        node_predictions = torch.tensor(np.zeros((batch_size, 12, NN)), dtype=torch.float32)
        for ii, mask in enumerate(masks.T):
            # mask: shape NN
            # masked_logits: shape batch_size x NN
            masked_logits = logits.clone()
            masked_logits[:, np.where(mask==0)[0]] = -1e9
            class_predictions = F.log_softmax(masked_logits, dim=1)
            node_predictions[:, ii] += class_predictions

        return node_predictions


def swarwise_loss(out, labels, bs):
    return F.nll_loss(out.view(bs*12, -1), labels.view(bs*12), ignore_index=-1)
