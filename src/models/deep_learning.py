"""PyTorch sequence models and vocabulary tools for Stage 3."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
TOKEN_PATTERN = re.compile(r"\b\w+\b")


def tokenize(text: str) -> list[str]:
    """Tokenize normalized article text into word tokens."""
    return TOKEN_PATTERN.findall(text.lower())


@dataclass(frozen=True)
class Vocabulary:
    """Training-only word vocabulary with reserved padding and unknown tokens."""

    token_to_id: dict[str, int]

    @property
    def size(self) -> int:
        """Return the number of vocabulary entries."""
        return len(self.token_to_id)

    @property
    def pad_id(self) -> int:
        """Return the padding-token id."""
        return self.token_to_id[PAD_TOKEN]

    @property
    def unknown_id(self) -> int:
        """Return the unknown-token id."""
        return self.token_to_id[UNK_TOKEN]

    def encode(self, text: str, max_length: int) -> list[int]:
        """Encode text as a fixed-length padded sequence of token ids."""
        if max_length < 1:
            raise ValueError("max_length must be positive")
        token_ids = [self.token_to_id.get(token, self.unknown_id) for token in tokenize(text)]
        token_ids = token_ids[:max_length]
        return token_ids + [self.pad_id] * (max_length - len(token_ids))


def build_vocabulary(texts: list[str], config: dict[str, Any]) -> Vocabulary:
    """Fit a deterministic vocabulary on training texts only."""
    max_size = int(config.get("max_size", 30000))
    min_frequency = int(config.get("min_frequency", 1))
    if max_size < 3:
        raise ValueError("vocabulary.max_size must be at least 3")

    counts = Counter(token for text in texts for token in tokenize(text))
    ranked = sorted(
        (token for token, count in counts.items() if count >= min_frequency),
        key=lambda token: (-counts[token], token),
    )
    tokens = [PAD_TOKEN, UNK_TOKEN, *ranked[: max_size - 2]]
    return Vocabulary({token: index for index, token in enumerate(tokens)})


class RecurrentTextClassifier(nn.Module):
    """Shared implementation for bidirectional LSTM and GRU classifiers."""

    def __init__(self, vocabulary_size: int, config: dict[str, Any], rnn_type: str) -> None:
        super().__init__()
        embedding_dim = int(config.get("embedding_dim", 128))
        hidden_dim = int(config.get("hidden_dim", 128))
        num_layers = int(config.get("num_layers", 1))
        dropout = float(config.get("dropout", 0.0))
        bidirectional = bool(config.get("bidirectional", True))

        self.embedding = nn.Embedding(vocabulary_size, embedding_dim, padding_idx=0)
        recurrent_class = nn.LSTM if rnn_type == "bilstm" else nn.GRU
        self.recurrent = recurrent_class(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        output_dim = hidden_dim * (2 if bidirectional else 1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(output_dim, 2)
        self.bidirectional = bidirectional

    def forward(self, token_ids: Tensor) -> Tensor:
        """Return binary-class logits for padded token-id sequences."""
        embedded = self.embedding(token_ids)
        _, hidden = self.recurrent(embedded)
        if isinstance(hidden, tuple):
            hidden = hidden[0]
        representation = torch.cat((hidden[-2], hidden[-1]), dim=1) if self.bidirectional else hidden[-1]
        return self.classifier(self.dropout(representation))


def build_deep_model(
    model_name: str,
    vocabulary_size: int,
    config: dict[str, Any],
) -> RecurrentTextClassifier:
    """Build a configured Stage 3 BiLSTM or GRU classifier."""
    if model_name not in {"bilstm", "gru"}:
        raise ValueError("Unsupported deep model. Supported: bilstm, gru")
    return RecurrentTextClassifier(vocabulary_size, config, model_name)


def resolve_training_device(training_config: dict[str, Any]) -> torch.device:
    """Resolve the configured device and fail clearly when CUDA is required."""
    requested = str(training_config.get("device", "auto")).lower()
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but PyTorch cannot access a GPU")
        return torch.device("cuda")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cpu":
        return torch.device("cpu")
    raise ValueError("training.device must be one of: cuda, auto, cpu")
