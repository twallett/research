"""
Graph Neural Network Models for Citation Networks

Implements GNN architectures for learning paper embeddings from citation graphs.
Supports multiple GNN layers: GCN, GAT, GraphSAGE.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, global_mean_pool
from torch_geometric.data import Data
from typing import Optional, List
import numpy as np


class CitationGNN(nn.Module):
    """
    Graph Neural Network for learning paper embeddings from citation networks.

    Architecture:
    1. Input layer: Projects initial node features to hidden dimension
    2. GNN layers: Multiple graph convolution layers
    3. Output layer: Final paper embeddings

    The model learns to aggregate information from neighboring papers (citations)
    to create meaningful paper representations.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 128,
        num_layers: int = 3,
        gnn_type: str = 'gcn',
        dropout: float = 0.3,
        use_batch_norm: bool = True
    ):
        """
        Initialize the Citation GNN.

        Args:
            input_dim: Dimension of input node features (embedding dimension)
            hidden_dim: Hidden dimension for GNN layers
            output_dim: Output embedding dimension
            num_layers: Number of GNN layers
            gnn_type: Type of GNN layer ('gcn', 'gat', 'sage')
            dropout: Dropout rate
            use_batch_norm: Whether to use batch normalization
        """
        super(CitationGNN, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.gnn_type = gnn_type.lower()
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # GNN layers
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if use_batch_norm else None

        for i in range(num_layers):
            in_channels = hidden_dim
            out_channels = hidden_dim if i < num_layers - 1 else output_dim

            if self.gnn_type == 'gcn':
                conv = GCNConv(in_channels, out_channels)
            elif self.gnn_type == 'gat':
                # GAT with 4 attention heads
                heads = 4 if i < num_layers - 1 else 1
                conv = GATConv(in_channels, out_channels // heads, heads=heads, dropout=dropout)
            elif self.gnn_type == 'sage':
                conv = SAGEConv(in_channels, out_channels)
            else:
                raise ValueError(f"Unknown GNN type: {gnn_type}")

            self.convs.append(conv)

            if use_batch_norm and i < num_layers - 1:
                self.batch_norms.append(nn.BatchNorm1d(out_channels))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the GNN.

        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge indices [2, num_edges]

        Returns:
            Paper embeddings [num_nodes, output_dim]
        """
        # Input projection
        x = self.input_proj(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # GNN layers
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)

            # Apply batch norm and activation (except for last layer)
            if i < len(self.convs) - 1:
                if self.use_batch_norm:
                    x = self.batch_norms[i](x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        # L2 normalize embeddings
        x = F.normalize(x, p=2, dim=1)

        return x

    def get_embeddings(self, x: torch.Tensor, edge_index: torch.Tensor) -> np.ndarray:
        """
        Get paper embeddings as numpy array.

        Args:
            x: Node features
            edge_index: Edge indices

        Returns:
            Paper embeddings as numpy array
        """
        self.eval()
        with torch.no_grad():
            embeddings = self.forward(x, edge_index)
        return embeddings.cpu().numpy()


class CitationGNNWithAttention(nn.Module):
    """
    Enhanced Citation GNN with self-attention mechanism.

    This model combines GNN layers with attention to better capture
    the importance of different citation relationships.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 128,
        num_gnn_layers: int = 3,
        num_attention_heads: int = 4,
        dropout: float = 0.3
    ):
        """
        Initialize the Citation GNN with Attention.

        Args:
            input_dim: Dimension of input node features
            hidden_dim: Hidden dimension
            output_dim: Output embedding dimension
            num_gnn_layers: Number of GNN layers
            num_attention_heads: Number of attention heads
            dropout: Dropout rate
        """
        super(CitationGNNWithAttention, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_gnn_layers = num_gnn_layers
        self.dropout = dropout

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # GNN layers (GAT for built-in attention)
        self.convs = nn.ModuleList()
        for i in range(num_gnn_layers):
            in_channels = hidden_dim
            out_channels = hidden_dim

            conv = GATConv(
                in_channels,
                out_channels // num_attention_heads,
                heads=num_attention_heads,
                dropout=dropout,
                concat=True
            )
            self.convs.append(conv)

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Node features
            edge_index: Edge indices

        Returns:
            Paper embeddings
        """
        # Input projection
        x = self.input_proj(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # GNN layers with attention
        for i, conv in enumerate(self.convs):
            x_new = conv(x, edge_index)
            x_new = F.relu(x_new)
            x_new = F.dropout(x_new, p=self.dropout, training=self.training)

            # Residual connection
            if i > 0:
                x = x + x_new
            else:
                x = x_new

        # Output projection
        x = self.output_proj(x)

        # L2 normalize
        x = F.normalize(x, p=2, dim=1)

        return x

    def get_embeddings(self, x: torch.Tensor, edge_index: torch.Tensor) -> np.ndarray:
        """Get embeddings as numpy array."""
        self.eval()
        with torch.no_grad():
            embeddings = self.forward(x, edge_index)
        return embeddings.cpu().numpy()


class GNNTrainer:
    """
    Trainer for Citation GNN models.

    Supports unsupervised training using link prediction as pretext task.
    """

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-5,
        device: str = 'cuda:0'
    ):
        """
        Initialize the trainer.

        Args:
            model: GNN model to train
            learning_rate: Learning rate
            weight_decay: Weight decay for regularization
            device: Device to train on ('cpu' or 'cuda')
        """
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)

        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

    def train_unsupervised(
        self,
        data: Data,
        num_epochs: int = 100,
        neg_sampling_ratio: float = 1.0,
        verbose: bool = True
    ) -> List[float]:
        """
        Train the model using link prediction (unsupervised).

        Args:
            data: PyTorch Geometric Data object
            num_epochs: Number of training epochs
            neg_sampling_ratio: Ratio of negative samples to positive
            verbose: Whether to print progress

        Returns:
            List of training losses
        """
        self.model.train()
        data = data.to(self.device)

        losses = []

        for epoch in range(num_epochs):
            self.optimizer.zero_grad()

            # Forward pass
            embeddings = self.model(data.x, data.edge_index)

            # Link prediction loss (positive edges should have high similarity)
            loss = self._link_prediction_loss(
                embeddings,
                data.edge_index,
                neg_sampling_ratio=neg_sampling_ratio
            )

            # Backward pass
            loss.backward()
            self.optimizer.step()

            losses.append(loss.item())

            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}")

        return losses

    def _link_prediction_loss(
        self,
        embeddings: torch.Tensor,
        edge_index: torch.Tensor,
        neg_sampling_ratio: float = 1.0
    ) -> torch.Tensor:
        """
        Compute link prediction loss.

        Positive pairs: connected nodes should have high similarity
        Negative pairs: randomly sampled non-connected nodes should have low similarity

        Args:
            embeddings: Node embeddings
            edge_index: Positive edges
            neg_sampling_ratio: Ratio of negative to positive samples

        Returns:
            Loss value
        """
        # Positive edges
        pos_edge_index = edge_index
        num_pos = pos_edge_index.size(1)

        # Compute positive similarities
        src_emb = embeddings[pos_edge_index[0]]
        dst_emb = embeddings[pos_edge_index[1]]
        pos_sim = (src_emb * dst_emb).sum(dim=1)

        # Negative sampling
        num_neg = int(num_pos * neg_sampling_ratio)
        num_nodes = embeddings.size(0)

        neg_src = torch.randint(0, num_nodes, (num_neg,), device=embeddings.device)
        neg_dst = torch.randint(0, num_nodes, (num_neg,), device=embeddings.device)

        neg_src_emb = embeddings[neg_src]
        neg_dst_emb = embeddings[neg_dst]
        neg_sim = (neg_src_emb * neg_dst_emb).sum(dim=1)

        # Binary cross-entropy loss
        pos_loss = F.binary_cross_entropy_with_logits(
            pos_sim,
            torch.ones_like(pos_sim)
        )
        neg_loss = F.binary_cross_entropy_with_logits(
            neg_sim,
            torch.zeros_like(neg_sim)
        )

        loss = pos_loss + neg_loss

        return loss


def create_gnn_model(
    input_dim: int,
    hidden_dim: int = 256,
    output_dim: int = 128,
    num_layers: int = 3,
    gnn_type: str = 'gcn',
    use_attention: bool = False,
    **kwargs
) -> nn.Module:
    """
    Factory function to create GNN models.

    Args:
        input_dim: Input feature dimension
        hidden_dim: Hidden dimension
        output_dim: Output embedding dimension
        num_layers: Number of layers
        gnn_type: Type of GNN ('gcn', 'gat', 'sage')
        use_attention: Whether to use attention-enhanced model
        **kwargs: Additional arguments

    Returns:
        GNN model
    """
    if use_attention:
        return CitationGNNWithAttention(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_gnn_layers=num_layers,
            **kwargs
        )
    else:
        return CitationGNN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            gnn_type=gnn_type,
            **kwargs
        )
