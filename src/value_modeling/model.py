from __future__ import annotations

import torch
from torch import nn


class OTTValueModel(nn.Module):
    def __init__(
        self,
        cat_cardinalities: list[int],
        n_numeric: int,
        vocab_size: int,
        embed_dim: int = 24,
        hidden_dim: int = 96,
    ) -> None:
        super().__init__()
        self.cat_embeddings = nn.ModuleList([nn.Embedding(card + 1, embed_dim, padding_idx=0) for card in cat_cardinalities])
        self.token_embedding = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=0)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=4,
            dim_feedforward=hidden_dim,
            dropout=0.1,
            batch_first=True,
        )
        self.text_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        input_dim = embed_dim * len(cat_cardinalities) + embed_dim + n_numeric
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.value_head = nn.Linear(hidden_dim, 1)
        self.watch_head = nn.Linear(hidden_dim, 1)
        self.retention_head = nn.Linear(hidden_dim, 1)

    def forward(self, cat: torch.Tensor, num: torch.Tensor, tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        cat_vec = torch.cat([emb(cat[:, i]) for i, emb in enumerate(self.cat_embeddings)], dim=1)
        token_mask = tokens.eq(0)
        token_vec = self.token_embedding(tokens)
        encoded = self.text_encoder(token_vec, src_key_padding_mask=token_mask)
        valid = (~token_mask).float().unsqueeze(-1)
        text_vec = (encoded * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)
        shared = self.backbone(torch.cat([cat_vec, num, text_vec], dim=1))
        return {
            "value": self.value_head(shared),
            "watch": torch.relu(self.watch_head(shared)),
            "retention_logit": self.retention_head(shared),
        }
