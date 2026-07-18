from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from collections import OrderedDict

import torch
from torch.utils.data import TensorDataset

from context_aware_ci_build_failure_prediction.preprocessing.modules.shard_writer import (
    iter_text_sidecar,
    load_embedding_shard,
)
from context_aware_ci_build_failure_prediction.preprocessing.types import (
    normalize_build_label,
)


@dataclass(frozen=True)
class LoadedSampleTable:
    message_embeddings: torch.Tensor
    diff_embeddings: torch.Tensor
    context_embeddings: torch.Tensor
    labels: torch.Tensor
    repos: list[str]
    commit_shas: list[str]
    sample_ids: list[str]
    build_ids: list[str | None]

    @property
    def features(self) -> torch.Tensor:
        return torch.cat(
            [self.message_embeddings, self.diff_embeddings, self.context_embeddings],
            dim=1,
        )

    def to_tensor_dataset(self) -> TensorDataset:
        return TensorDataset(
            self.message_embeddings,
            self.diff_embeddings,
            self.context_embeddings,
            self.labels,
        )

    def subset(self, indices: list[int]) -> LoadedSampleTable:
        index_tensor = torch.tensor(indices, dtype=torch.long)
        return LoadedSampleTable(
            message_embeddings=self.message_embeddings[index_tensor],
            diff_embeddings=self.diff_embeddings[index_tensor],
            context_embeddings=self.context_embeddings[index_tensor],
            labels=self.labels[index_tensor],
            repos=[self.repos[index] for index in indices],
            commit_shas=[self.commit_shas[index] for index in indices],
            sample_ids=[self.sample_ids[index] for index in indices],
            build_ids=[self.build_ids[index] for index in indices],
        )


@dataclass(frozen=True)
class LoadedSampleSplits:
    train: LoadedSampleTable
    validation: LoadedSampleTable
    test: LoadedSampleTable
    train_repos: list[str]
    validation_repos: list[str]
    test_repos: list[str]

    @property
    def embedding_dim(self) -> int:
        return int(self.train.message_embeddings.shape[1])


def load_training_pairs_from_pt_shards(
    source_dir: str | Path,
    num_samples: int | None,
    *,
    shard_glob: str = "shard_*.pt",
    dtype: torch.dtype = torch.float32,
) -> TensorDataset:
    """
    Load training examples from preprocessing .pt shards in filename order.

    Returns a TensorDataset whose examples are:
        (message_embedding, diff_embedding, context_embedding, label)

    This matches AttentionFusionClassifier.forward(message, diff, context), with
    labels represented as float tensors containing 1.0 for successful builds and
    0.0 for unsuccessful builds.
    """
    table = load_sample_table_from_pt_shards(
        source_dir=source_dir,
        num_samples=num_samples,
        shard_glob=shard_glob,
        dtype=dtype,
    )
    return table.to_tensor_dataset()


def load_sample_table_from_pt_shards(
    source_dir: str | Path,
    num_samples: int | None,
    *,
    shard_glob: str = "shard_*.pt",
    dtype: torch.dtype = torch.float32,
) -> LoadedSampleTable:
    if num_samples is not None and num_samples < 0:
        raise ValueError("num_samples must be non-negative or None.")

    source_path = Path(source_dir)
    if not source_path.exists():
        raise FileNotFoundError(f"Source directory does not exist: {source_path}")
    if not source_path.is_dir():
        raise NotADirectoryError(f"Source path is not a directory: {source_path}")

    shard_paths = sorted(
        path
        for path in source_path.glob(shard_glob)
        if path.is_file() and not path.name.endswith(".tmp")
    )
    if not shard_paths:
        raise FileNotFoundError(
            f"No .pt shard files matching {shard_glob!r} found in {source_path}"
        )

    remaining = num_samples
    message_batches: list[torch.Tensor] = []
    diff_batches: list[torch.Tensor] = []
    context_batches: list[torch.Tensor] = []
    label_batches: list[torch.Tensor] = []
    repos: list[str] = []
    commit_shas: list[str] = []
    sample_ids: list[str] = []
    build_ids: list[str | None] = []

    for shard_path in shard_paths:
        if remaining == 0:
            break

        payload = load_embedding_shard(shard_path)
        message_batch = _payload_tensor(payload, "message_embeddings", shard_path, dtype)
        diff_batch = _payload_tensor(payload, "diff_embeddings", shard_path, dtype)
        context_batch = _payload_tensor(payload, "context_embeddings", shard_path, dtype)
        labels = _payload_tensor(payload, "labels", shard_path, torch.float32)

        if labels.ndim != 1:
            raise ValueError(
                f"Shard {shard_path} labels must be one-dimensional; "
                f"got shape {tuple(labels.shape)}."
            )
        labels = _repair_missing_labels_from_sidecar(labels, shard_path)

        _validate_batch_shapes(
            message_batch=message_batch,
            diff_batch=diff_batch,
            context_batch=context_batch,
            labels=labels,
            shard_path=shard_path,
        )

        if remaining is not None:
            keep = min(remaining, labels.shape[0])
            message_batch = message_batch[:keep]
            diff_batch = diff_batch[:keep]
            context_batch = context_batch[:keep]
            labels = labels[:keep]
            remaining -= keep
        else:
            keep = labels.shape[0]

        message_batches.append(message_batch)
        diff_batches.append(diff_batch)
        context_batches.append(context_batch)
        label_batches.append(labels)
        repos.extend(str(value) for value in _payload_list(payload, "repos", shard_path, keep))
        commit_shas.extend(
            str(value) for value in _payload_list(payload, "commit_shas", shard_path, keep)
        )
        sample_ids.extend(
            str(value) for value in _payload_list(payload, "sample_ids", shard_path, keep)
        )
        build_ids.extend(
            value if value is None else str(value)
            for value in _payload_list(payload, "build_ids", shard_path, keep)
        )

    if not message_batches:
        return LoadedSampleTable(
            message_embeddings=torch.empty((0, 0), dtype=dtype),
            diff_embeddings=torch.empty((0, 0), dtype=dtype),
            context_embeddings=torch.empty((0, 0), dtype=dtype),
            labels=torch.empty((0,), dtype=torch.float32),
            repos=[],
            commit_shas=[],
            sample_ids=[],
            build_ids=[],
        )

    messages = torch.cat(message_batches, dim=0)
    diffs = torch.cat(diff_batches, dim=0)
    contexts = torch.cat(context_batches, dim=0)
    labels = torch.cat(label_batches, dim=0)

    _validate_embedding_dimensions(messages, diffs, contexts)

    return LoadedSampleTable(
        message_embeddings=messages,
        diff_embeddings=diffs,
        context_embeddings=contexts,
        labels=labels,
        repos=repos,
        commit_shas=commit_shas,
        sample_ids=sample_ids,
        build_ids=build_ids,
    )


def load_training_pairs(
    source_dir: str | Path,
    num_samples: int | None = 300,
    *,
    shard_glob: str = "shard_*.pt",
    dtype: torch.dtype = torch.float32,
) -> TensorDataset:
    return load_training_pairs_from_pt_shards(
        source_dir=source_dir,
        num_samples=num_samples,
        shard_glob=shard_glob,
        dtype=dtype,
    )


def load_sample_splits(
    source_dir: str | Path,
    num_samples: int | None = 300,
    *,
    validation_fraction: float = 0.2,
    test_fraction: float = 0.2,
    seed: int = 0,
    shuffle: bool = True,
    shard_glob: str = "shard_*.pt",
    dtype: torch.dtype = torch.float32,
) -> LoadedSampleSplits:
    table = load_sample_table(
        source_dir=source_dir,
        num_samples=num_samples,
        shard_glob=shard_glob,
        dtype=dtype,
    )
    return split_sample_table_by_repo(
        table=table,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
        seed=seed,
        shuffle=shuffle,
    )


def load_sample_table(
    source_dir: str | Path,
    num_samples: int | None = 300,
    *,
    shard_glob: str = "shard_*.pt",
    dtype: torch.dtype = torch.float32,
) -> LoadedSampleTable:
    return load_sample_table_from_pt_shards(
        source_dir=source_dir,
        num_samples=num_samples,
        shard_glob=shard_glob,
        dtype=dtype,
    )


def split_sample_table_by_repo(
    *,
    table: LoadedSampleTable,
    validation_fraction: float = 0.2,
    test_fraction: float = 0.2,
    seed: int = 0,
    shuffle: bool = True,
) -> LoadedSampleSplits:
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in [0, 1).")
    if not 0.0 <= test_fraction < 1.0:
        raise ValueError("test_fraction must be in [0, 1).")
    if validation_fraction + test_fraction >= 1.0:
        raise ValueError("validation_fraction + test_fraction must be less than 1.")

    repo_to_indices: OrderedDict[str, list[int]] = OrderedDict()
    for index, repo in enumerate(table.repos):
        repo_to_indices.setdefault(repo, []).append(index)

    repos = list(repo_to_indices)
    if shuffle and repos:
        generator = torch.Generator().manual_seed(seed)
        permutation = torch.randperm(len(repos), generator=generator).tolist()
        repos = [repos[index] for index in permutation]

    train_repos, validation_repos, test_repos = _split_repos(
        repos=repos,
        repo_to_indices=repo_to_indices,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
    )

    train_indices = _indices_for_repos(train_repos, repo_to_indices)
    validation_indices = _indices_for_repos(validation_repos, repo_to_indices)
    test_indices = _indices_for_repos(test_repos, repo_to_indices)

    return LoadedSampleSplits(
        train=table.subset(train_indices),
        validation=table.subset(validation_indices),
        test=table.subset(test_indices),
        train_repos=train_repos,
        validation_repos=validation_repos,
        test_repos=test_repos,
    )


def _split_repos(
    *,
    repos: list[str],
    repo_to_indices: OrderedDict[str, list[int]],
    validation_fraction: float,
    test_fraction: float,
) -> tuple[list[str], list[str], list[str]]:
    if not repos:
        return [], [], []

    total_samples = sum(len(indices) for indices in repo_to_indices.values())
    target_sample_counts = {
        "train": total_samples * (1.0 - validation_fraction - test_fraction),
        "validation": total_samples * validation_fraction,
        "test": total_samples * test_fraction,
    }
    split_repos: dict[str, list[str]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    split_sample_counts = {
        "train": 0,
        "validation": 0,
        "test": 0,
    }
    split_order = tuple(
        split
        for split in ("train", "validation", "test")
        if target_sample_counts[split] > 0
    )
    repo_order = {repo: index for index, repo in enumerate(repos)}

    repos_by_size = sorted(
        repos,
        key=lambda repo: (-len(repo_to_indices[repo]), repo_order[repo]),
    )
    for repo in repos_by_size:
        repo_size = len(repo_to_indices[repo])
        best_split = max(
            split_order,
            key=lambda split: (
                (
                    target_sample_counts[split] - split_sample_counts[split]
                ) / max(target_sample_counts[split], 1.0),
                -abs(
                    target_sample_counts[split]
                    - (split_sample_counts[split] + repo_size)
                ) / max(target_sample_counts[split], 1.0),
                -split_order.index(split),
            ),
        )
        split_repos[best_split].append(repo)
        split_sample_counts[best_split] += repo_size

    return (
        split_repos["train"],
        split_repos["validation"],
        split_repos["test"],
    )


def _indices_for_repos(
    repos: list[str],
    repo_to_indices: OrderedDict[str, list[int]],
) -> list[int]:
    return [
        index
        for repo in repos
        for index in repo_to_indices[repo]
    ]


def _payload_tensor(
    payload: dict,
    key: str,
    shard_path: Path,
    dtype: torch.dtype,
) -> torch.Tensor:
    value = payload.get(key)
    if value is None:
        raise ValueError(f"Shard {shard_path} is missing required tensor {key!r}.")

    tensor = torch.as_tensor(value, dtype=dtype).detach().cpu()
    if key != "labels" and tensor.ndim != 2:
        raise ValueError(
            f"Shard {shard_path} tensor {key!r} must be two-dimensional; "
            f"got shape {tuple(tensor.shape)}."
        )
    return tensor


def _payload_list(
    payload: dict,
    key: str,
    shard_path: Path,
    keep: int,
) -> list:
    value = payload.get(key)
    if value is None:
        raise ValueError(f"Shard {shard_path} is missing required field {key!r}.")
    if len(value) < keep:
        raise ValueError(
            f"Shard {shard_path} field {key!r} has {len(value)} values, "
            f"but {keep} are required."
        )
    return list(value[:keep])


def _repair_missing_labels_from_sidecar(
    labels: torch.Tensor,
    shard_path: Path,
) -> torch.Tensor:
    if not torch.isnan(labels).any():
        return labels

    sidecar_path = _sidecar_path_for_shard(shard_path)
    if not sidecar_path.exists():
        raise ValueError(
            f"Shard {shard_path} contains NaN labels and paired sidecar "
            f"{sidecar_path} does not exist."
        )

    repaired_labels = []
    for index, record in enumerate(iter_text_sidecar(sidecar_path)):
        normalized_label = normalize_build_label(record.get("label"))
        if normalized_label is None:
            raise ValueError(
                f"Sidecar {sidecar_path} has missing label at record {index}; "
                "cannot build supervised training pairs."
            )
        repaired_labels.append(float(normalized_label))

    repaired = torch.tensor(repaired_labels, dtype=torch.float32)
    if repaired.shape != labels.shape:
        raise ValueError(
            f"Sidecar {sidecar_path} has {repaired.shape[0]} labels, but shard "
            f"{shard_path} has label shape {tuple(labels.shape)}."
        )

    return torch.where(torch.isnan(labels), repaired, labels)


def _sidecar_path_for_shard(shard_path: Path) -> Path:
    if shard_path.suffix != ".pt":
        raise ValueError(f"Expected .pt shard path, got {shard_path}")
    return shard_path.with_name(f"{shard_path.stem}.text.jsonl.gz")


def _validate_batch_shapes(
    *,
    message_batch: torch.Tensor,
    diff_batch: torch.Tensor,
    context_batch: torch.Tensor,
    labels: torch.Tensor,
    shard_path: Path,
) -> None:
    batch_sizes = {
        message_batch.shape[0],
        diff_batch.shape[0],
        context_batch.shape[0],
        labels.shape[0],
    }
    if len(batch_sizes) != 1:
        raise ValueError(
            f"Shard {shard_path} has misaligned batch sizes: "
            f"message={message_batch.shape[0]}, diff={diff_batch.shape[0]}, "
            f"context={context_batch.shape[0]}, labels={labels.shape[0]}."
        )


def _validate_embedding_dimensions(
    messages: torch.Tensor,
    diffs: torch.Tensor,
    contexts: torch.Tensor,
) -> None:
    dimensions = {
        messages.shape[1],
        diffs.shape[1],
        contexts.shape[1],
    }
    if len(dimensions) != 1:
        raise ValueError(
            "Message, diff, and context embeddings must have the same dimension; "
            f"got message={messages.shape[1]}, diff={diffs.shape[1]}, "
            f"context={contexts.shape[1]}."
        )


__all__ = [
    "LoadedSampleTable",
    "LoadedSampleSplits",
    "load_sample_table",
    "load_sample_table_from_pt_shards",
    "load_sample_splits",
    "load_training_pairs",
    "load_training_pairs_from_pt_shards",
    "split_sample_table_by_repo",
]
