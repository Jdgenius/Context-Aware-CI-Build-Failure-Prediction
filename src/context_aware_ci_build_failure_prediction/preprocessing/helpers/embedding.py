import torch
from typing import Any
from transformers import AutoTokenizer, AutoModel

from ..modules.repo_manager import MODEL_NAME
from ..types import TokenizationMetadata

class CodeBERTEmbedder:
    def __init__(
        self,
        model_name: str = MODEL_NAME,
        device: str | None = None,
        max_length: int = 512,
        use_float16_output: bool = True
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.use_float16_output = use_float16_output

        print(f"Loading {model_name} on {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    @property
    def metadata(self) -> dict[str, Any]:
        hidden_size = getattr(getattr(self.model, "config", None), "hidden_size", None)

        return {
            "model_name": self.model_name,
            "max_length": self.max_length,
            "pooling": "attention_mask_mean_pooling_last_hidden_state",
            "embedding_dimension": hidden_size,
            "output_dtype": str(torch.float16 if self.use_float16_output else torch.float32),
            "device": str(self.device),
        }

    @torch.no_grad()
    def embed_texts_with_metadata(
        self,
        texts: list[str],
        batch_size: int = 32
    ) -> tuple[torch.Tensor, list[TokenizationMetadata]]:
        """
        Returns tensor of shape [num_texts, hidden_dim].
        For CodeBERT-base, hidden_dim is usually 768.
        """
        all_embeddings = []
        all_metadata: list[TokenizationMetadata] = []

        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start:start + batch_size]
            untruncated = self.tokenizer(
                batch_texts,
                padding=False,
                truncation=False,
            )
            token_counts_before_truncation = [
                len(input_ids) for input_ids in untruncated["input_ids"]
            ]

            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            retained_token_counts = encoded["attention_mask"].sum(dim=1).tolist()
            all_metadata.extend(
                TokenizationMetadata(
                    token_count_before_truncation=int(original_count),
                    retained_token_count=int(retained_count),
                    was_tokenizer_truncated=original_count > retained_count,
                )
                for original_count, retained_count in zip(
                    token_counts_before_truncation,
                    retained_token_counts,
                    strict=True,
                )
            )

            encoded = {
                key: value.to(self.device)
                for key, value in encoded.items()
            }

            outputs = self.model(**encoded)
            last_hidden = outputs.last_hidden_state
            attention_mask = encoded["attention_mask"].unsqueeze(-1)

            masked_hidden = last_hidden * attention_mask
            summed = masked_hidden.sum(dim=1)
            counts = attention_mask.sum(dim=1).clamp(min=1)

            embeddings = summed / counts
            embeddings = embeddings.detach().cpu()

            if self.use_float16_output:
                embeddings = embeddings.half()

            all_embeddings.append(embeddings)

        return torch.cat(all_embeddings, dim=0), all_metadata

    @torch.no_grad()
    def embed_texts(
        self,
        texts: list[str],
        batch_size: int = 32
    ) -> torch.Tensor:
        embeddings, _ = self.embed_texts_with_metadata(
            texts=texts,
            batch_size=batch_size,
        )
        return embeddings
