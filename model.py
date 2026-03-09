from typing import Dict

import torch
import torch.nn as nn


class ESGPolicyValueNet(nn.Module):
    """Policy-value network for PPO-CE.

    Args:
        num_labels: Number of label ids in the vocabulary.
        node_feat_dim: Dimension of node feature vectors.
        label_emb_dim: Embedding size for entity labels.
        hidden_dim: Hidden size for node and graph encoders.
        max_nodes: Maximum number of nodes per observation.
    """

    def __init__(
        self,
        num_labels: int,
        node_feat_dim: int = 7,
        label_emb_dim: int = 32,
        hidden_dim: int = 128,
        max_nodes: int = 32,
    ):
        super().__init__()
        self.max_nodes = max_nodes
        self.label_emb = nn.Embedding(num_labels + 1, label_emb_dim)
        self.node_mlp = nn.Sequential(
            nn.Linear(node_feat_dim + label_emb_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.node_logit_head = nn.Linear(hidden_dim, 1)
        self.stop_head = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: Dict[str, torch.Tensor]):
        """Run a forward pass.

        Args:
            obs: Observation tensors with node features, labels, masks, and action mask.
        """

        node_feats = obs["node_feats"]
        label_ids = obs["label_ids"].long()
        selected_mask = obs["selected_mask"]
        valid_mask = obs["valid_mask"]
        action_mask = obs["action_mask"]

        label_vec = self.label_emb(label_ids)
        selected_flag = selected_mask.unsqueeze(-1)
        x = torch.cat([node_feats, label_vec, selected_flag], dim=-1)
        hidden = self.node_mlp(x)

        valid_denom = valid_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        graph_hidden = (hidden * valid_mask.unsqueeze(-1)).sum(dim=1) / valid_denom
        selected_ratio = selected_mask.sum(dim=1, keepdim=True) / valid_denom
        global_hidden = torch.cat([graph_hidden, selected_ratio], dim=-1)

        node_logits = self.node_logit_head(hidden).squeeze(-1)
        stop_logit = self.stop_head(global_hidden)
        logits = torch.cat([node_logits, stop_logit], dim=-1)
        masked_logits = logits.masked_fill(action_mask <= 0, -1e9)
        value = self.value_head(global_hidden).squeeze(-1)
        return masked_logits, value
