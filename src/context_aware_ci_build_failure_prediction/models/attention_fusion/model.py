from __future__ import annotations

import torch
from torch import Tensor, nn


class AttentionFusionClassifier(nn.Module):
    """Fuse three embeddings with learned modality attention."""

    def __init__(
        self,
        embedding_dim: int,
        model_dim: int = 128,
        attention_dim: int = 64,
        classifier_hidden_dim: int = 128,
        dropout: float = 0.2,
        separate_projections: bool = True,
    ) -> None:
        super().__init__()

        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive.")
        if model_dim <= 0:
            raise ValueError("model_dim must be positive.")
        if attention_dim <= 0:
            raise ValueError("attention_dim must be positive.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")

        self.embedding_dim = embedding_dim
        self.model_dim = model_dim
        self.separate_projections = separate_projections

        def make_projection() -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(embedding_dim, model_dim),
                nn.LayerNorm(model_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )

        if separate_projections:
            self.message_projection = make_projection()
            self.diff_projection = make_projection()
            self.context_projection = make_projection()
        else:
            self.shared_projection = make_projection()

        # Converts each projected modality vector to one scalar score.
        self.attention_scorer = nn.Sequential(
            nn.Linear(model_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1),
        )

        self.classifier = nn.Sequential(
            nn.Linear(model_dim, classifier_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_dim, 1),
        )

    def _validate_inputs(
        self,
        message_embedding: Tensor,
        diff_embedding: Tensor,
        context_embedding: Tensor,
    ) -> None:
        expected_dim = self.embedding_dim

        for name, tensor in (
            ("message_embedding", message_embedding),
            ("diff_embedding", diff_embedding),
            ("context_embedding", context_embedding),
        ):
            if tensor.ndim != 2:
                raise ValueError(
                    f"{name} must have shape [batch_size, embedding_dim], "
                    f"but received {tuple(tensor.shape)}."
                )

            if tensor.shape[-1] != expected_dim:
                raise ValueError(
                    f"{name} has embedding dimension {tensor.shape[-1]}, "
                    f"but expected {expected_dim}."
                )

        batch_sizes = {
            message_embedding.shape[0],
            diff_embedding.shape[0],
            context_embedding.shape[0],
        }

        if len(batch_sizes) != 1:
            raise ValueError("All three inputs must have the same batch size.")

    def forward(
        self,
        message_embedding: Tensor,
        diff_embedding: Tensor,
        context_embedding: Tensor,
        return_attention: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        self._validate_inputs(
            message_embedding,
            diff_embedding,
            context_embedding,
        )

        if self.separate_projections:
            message = self.message_projection(message_embedding)
            diff = self.diff_projection(diff_embedding)
            context = self.context_projection(context_embedding)
        else:
            message = self.shared_projection(message_embedding)
            diff = self.shared_projection(diff_embedding)
            context = self.shared_projection(context_embedding)

        # Shape: [batch_size, 3, model_dim]
        modalities = torch.stack(
            [message, diff, context],
            dim=1,
        )

        # Shape before squeeze: [batch_size, 3, 1]
        # Shape after squeeze:  [batch_size, 3]
        scores = self.attention_scorer(modalities).squeeze(-1)

        # Normalize over the three modalities.
        attention_weights = torch.softmax(scores, dim=1)

        # Shape: [batch_size, 3, 1]
        expanded_weights = attention_weights.unsqueeze(-1)

        # Weighted sum over the modality dimension.
        # Shape: [batch_size, model_dim]
        fused = torch.sum(expanded_weights * modalities, dim=1)

        # Shape: [batch_size]
        logits = self.classifier(fused).squeeze(-1)

        if return_attention:
            return logits, attention_weights

        return logits
    

