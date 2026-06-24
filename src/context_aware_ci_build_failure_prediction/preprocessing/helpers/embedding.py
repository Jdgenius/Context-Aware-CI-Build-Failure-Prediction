import torch
from transformers import AutoTokenizer, AutoModel

from ..modules.repo_manager import MODEL_NAME

class CodeBERTEmbedder:
    def __init__(
        self,
        model_name: str = MODEL_NAME,
        device: str | None = None,
        max_length: int = 512,
        use_float16_output: bool = True
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.use_float16_output = use_float16_output

        print(f"Loading {model_name} on {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def embed_texts(
        self,
        texts: list[str],
        batch_size: int = 32
    ) -> torch.Tensor:
        """
        Returns tensor of shape [num_texts, hidden_dim].
        For CodeBERT-base, hidden_dim is usually 768.
        """
        all_embeddings = []

        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start:start + batch_size]

            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
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

        return torch.cat(all_embeddings, dim=0)