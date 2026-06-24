import torch
import gc

from pathlib import Path
from typing import Any

class EmbeddingShardWriter:
    def __init__(
        self,
        output_dir: str = "./embedding_shards",
        shard_size: int = 5000
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.shard_size = shard_size
        self.buffer: list[dict[str, Any]] = []
        self.shard_index = 0

    def add(self, record: dict[str, Any]) -> None:
        self.buffer.append(record)

        if len(self.buffer) >= self.shard_size:
            self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return

        shard_path = self.output_dir / f"shard_{self.shard_index:05d}.pt"

        message_embeddings = torch.stack(
            [r["message_embedding"] for r in self.buffer]
        )
        diff_embeddings = torch.stack(
            [r["diff_embedding"] for r in self.buffer]
        )
        context_embeddings = torch.stack(
            [r["context_embedding"] for r in self.buffer]
        )

        labels = [r["label"] for r in self.buffer]

        payload = {
            "repo": [r["repo"] for r in self.buffer],
            "commit_sha": [r["commit_sha"] for r in self.buffer],
            "message_embedding": message_embeddings,
            "diff_embedding": diff_embeddings,
            "context_embedding": context_embeddings,
            "label": labels,
        }

        torch.save(payload, shard_path)

        print(f"Saved shard: {shard_path} with {len(self.buffer)} samples")

        self.buffer.clear()
        self.shard_index += 1

        gc.collect()

    def close(self) -> None:
        self.flush()