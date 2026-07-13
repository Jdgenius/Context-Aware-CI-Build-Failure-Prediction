from __future__ import annotations

from types import SimpleNamespace

import torch

from context_aware_ci_build_failure_prediction.preprocessing import process as process_module
from context_aware_ci_build_failure_prediction.preprocessing.helpers import embedding as embedding_module
from context_aware_ci_build_failure_prediction.preprocessing.helpers.embedding import CodeBERTEmbedder
from context_aware_ci_build_failure_prediction.preprocessing.types import (
    RawSample,
    TextArtifact,
    TokenizationMetadata,
)


class FakeTokenizer:
    def __call__(
        self,
        texts,
        padding=False,
        truncation=False,
        max_length=None,
        return_tensors=None,
    ):
        if isinstance(texts, str):
            texts = [texts]

        tokenized = [self._token_ids(text) for text in texts]

        if truncation:
            tokenized = [
                self._truncate(ids, max_length)
                for ids in tokenized
            ]

        if padding:
            target_length = max(len(ids) for ids in tokenized)
            attention_masks = []
            padded_ids = []

            for ids in tokenized:
                padding_length = target_length - len(ids)
                padded_ids.append(ids + ([0] * padding_length))
                attention_masks.append(([1] * len(ids)) + ([0] * padding_length))
        else:
            padded_ids = tokenized
            attention_masks = [[1] * len(ids) for ids in tokenized]

        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor(padded_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            }

        return {
            "input_ids": padded_ids,
            "attention_mask": attention_masks,
        }

    def _token_ids(self, text: str) -> list[int]:
        words = text.split()
        return [101] + [200 + i for i, _ in enumerate(words)] + [102]

    def _truncate(self, ids: list[int], max_length: int | None) -> list[int]:
        if max_length is None or len(ids) <= max_length:
            return ids

        retained = ids[:max_length]
        retained[-1] = 102
        return retained


class FakeAutoTokenizer:
    @staticmethod
    def from_pretrained(model_name: str):
        return FakeTokenizer()


class FakeModel:
    def __init__(self):
        self.config = SimpleNamespace(hidden_size=3)

    def to(self, device: str):
        self.device = device
        return self

    def eval(self):
        return self

    def __call__(self, input_ids, attention_mask):
        base = input_ids.to(torch.float32).unsqueeze(-1)
        factors = torch.tensor([1.0, 0.5, 0.25], device=input_ids.device)
        return SimpleNamespace(last_hidden_state=base * factors)


class FakeAutoModel:
    @staticmethod
    def from_pretrained(model_name: str):
        return FakeModel()


def make_fake_embedder(monkeypatch, max_length: int = 6, use_float16_output: bool = True):
    monkeypatch.setattr(embedding_module, "AutoTokenizer", FakeAutoTokenizer)
    monkeypatch.setattr(embedding_module, "AutoModel", FakeAutoModel)
    return CodeBERTEmbedder(
        model_name="fake-codebert",
        device="cpu",
        max_length=max_length,
        use_float16_output=use_float16_output,
    )


def test_embedder_metadata_matches_configuration_and_is_copy(monkeypatch):
    embedder = make_fake_embedder(monkeypatch, max_length=6, use_float16_output=True)
    embeddings = embedder.embed_texts(["one two"])

    metadata = embedder.metadata

    assert metadata["model_name"] == "fake-codebert"
    assert metadata["max_length"] == 6
    assert metadata["pooling"] == "attention_mask_mean_pooling_last_hidden_state"
    assert metadata["embedding_dimension"] == 3
    assert metadata["output_dtype"] == str(embeddings.dtype)
    assert metadata["device"] == "cpu"

    metadata["max_length"] = 999
    assert embedder.metadata["max_length"] == 6


def test_tokenization_metadata_counts_special_tokens_and_batches(monkeypatch):
    embedder = make_fake_embedder(monkeypatch, max_length=6, use_float16_output=False)
    texts = [
        "",
        "one two",
        "one two three four",
        "one two three four five",
        "cafe unicode café 漢字",
    ]

    _, metadata = embedder.embed_texts_with_metadata(texts, batch_size=2)

    assert metadata == [
        TokenizationMetadata(2, 2, False),
        TokenizationMetadata(4, 4, False),
        TokenizationMetadata(6, 6, False),
        TokenizationMetadata(7, 6, True),
        TokenizationMetadata(6, 6, False),
    ]


def test_embedding_compatibility_wrapper_matches_new_method(monkeypatch):
    embedder = make_fake_embedder(monkeypatch, max_length=6, use_float16_output=True)
    texts = ["one two", "one two three four five"]

    old_style = embedder.embed_texts(texts, batch_size=1)
    new_style, metadata = embedder.embed_texts_with_metadata(texts, batch_size=1)

    assert old_style.shape == new_style.shape
    assert old_style.dtype == new_style.dtype
    assert torch.equal(old_style, new_style)
    assert len(metadata) == len(texts)


def test_embed_and_write_raw_batch_attaches_tokenization_metadata_to_samples():
    samples = [
        RawSample(
            sample_id="sample-1",
            source_row_index=0,
            repo="owner/repo",
            commit_sha="abc",
            parent_commit_sha=None,
            build_id=None,
            label="passed",
            commit_message=TextArtifact(text="message one", provenance={}),
            diff=TextArtifact(text="diff one", provenance={}),
            context=TextArtifact(text="context one", provenance={}),
        ),
        RawSample(
            sample_id="sample-2",
            source_row_index=1,
            repo="owner/repo",
            commit_sha="def",
            parent_commit_sha=None,
            build_id=None,
            label="failed",
            commit_message=TextArtifact(text="message two long", provenance={}),
            diff=TextArtifact(text="diff two long", provenance={}),
            context=TextArtifact(text="context two long", provenance={}),
        ),
    ]
    seen_inputs = []

    class FakeEmbedder:
        def embed_texts_with_metadata(self, texts, batch_size):
            seen_inputs.append(list(texts))
            call_index = len(seen_inputs)
            embeddings = torch.tensor(
                [[call_index, i] for i, _ in enumerate(texts)],
                dtype=torch.float32,
            )
            metadata = [
                TokenizationMetadata(
                    token_count_before_truncation=(call_index * 100) + i,
                    retained_token_count=len(text),
                    was_tokenizer_truncated=False,
                )
                for i, text in enumerate(texts)
            ]
            return embeddings, metadata

    class FakeWriter:
        def __init__(self):
            self.records = []

        def add(self, record):
            self.records.append(record)

    writer = FakeWriter()
    process_module.embed_and_write_raw_batch(
        raw_buffer=samples,
        embedder=FakeEmbedder(),
        writer=writer,
        embed_batch_size=1,
    )

    assert seen_inputs == [
        ["message one", "message two long"],
        ["diff one", "diff two long"],
        ["context one", "context two long"],
    ]
    assert samples[0].commit_message.tokenization == TokenizationMetadata(100, 11, False)
    assert samples[1].commit_message.tokenization == TokenizationMetadata(101, 16, False)
    assert samples[0].diff.tokenization == TokenizationMetadata(200, 8, False)
    assert samples[1].diff.tokenization == TokenizationMetadata(201, 13, False)
    assert samples[0].context.tokenization == TokenizationMetadata(300, 11, False)
    assert samples[1].context.tokenization == TokenizationMetadata(301, 16, False)
    assert writer.records[0].message_embedding.tolist() == [1, 0]
    assert writer.records[1].context_embedding.tolist() == [3, 1]
