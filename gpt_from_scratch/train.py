"""Train a GPT model on the tiny Shakespeare dataset."""

from pathlib import Path
from typing import Literal

import torch
from torch import nn, Tensor
from torch.nn import functional as F

torch.manual_seed(1337)

with open(Path("data/tinyshakespeare/input.txt"), "r", encoding="utf-8") as file:
    TEXT = file.read()

# character tokens
VOCABULARY = sorted(list(set(TEXT)))
"""We're using character-level tokens for simplicity."""
VOCAB_SIZE = len(VOCABULARY)
# print(f"Chars: {''.join(vocabulary)}")
# print(f"Vocab size: {len(vocabulary)}")


def encode(text: str) -> list[int]:
    """Convert text to a list of integers."""
    char_to_int = {ch: i for i, ch in enumerate(VOCABULARY)}
    return [char_to_int[ch] for ch in text]


def decode(ints: list[int]) -> str:
    """Convert a list of integers to text."""
    int_to_char = {i: ch for i, ch in enumerate(VOCABULARY)}
    return "".join([int_to_char[i] for i in ints])


# encoded = encode("hello")
# print(encoded)
# print(decode(encoded))


def get_batch(
    split: Literal["train", "val"], block_size: int, batch_size: int
) -> tuple[Tensor, Tensor]:
    """Generate a small batch of data of input x and targets y."""
    data = train_data if split == "train" else val_data
    ix = torch.randint(0, data.size(0) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x, y


class BigramLanguageModel(nn.Module):
    """A simple bigram language model. A bigram model predicts the next token based on just the previous token—essentially equivalent to having context length of 1."""

    def __init__(self, vocab_size: int):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    # idx and targets are both dim (Batch, Time) tensor of integers
    def forward(
        self, prompt: Tensor, targets: Tensor | None = None
    ) -> tuple[Tensor, Tensor | None]:
        """Return the logits for the next token for each token in each block."""
        logits: Tensor = self.token_embedding_table(
            prompt
        )  # dim (Batch, Time, Channel)
        if targets is None:
            return logits, None
        B, T, C = logits.shape
        logits = logits.view(B * T, C)  # basically combine all batches
        targets = targets.view(B * T)
        loss = F.cross_entropy(
            logits, targets
        )  # loss is based on differences in distribution, not actual observations/predictions as would be in regression
        return logits, loss # ? Seems like an issue since logits has been squeezed here

        # Batch: batch size
        # Time: block size
        # Channel: vocab size

    def generate(self, context: Tensor, max_new_tokens: int) -> Tensor:
        """Generate new tokens given a prompt."""
        for _ in range(max_new_tokens):
            logits, _ = self(context, context)
            logits = logits[
                :, -1, :
            ]  # focus on last time step, i.e. last token of each batch # (B, C)
            probs = F.softmax(
                logits, dim=-1
            )  # probability of each token for each batch # (B, C)
            prediction = torch.multinomial(
                probs, num_samples=1
            )  # sample from probability distribution for each batch # (B, 1)
            context = torch.cat(
                (context, prediction), dim=1
            )  # append to context # (B, T) -> (B, T+1)
        return context


data = torch.tensor(encode(text=TEXT), dtype=torch.long)
TRAINING_PCT = 0.9
train_data = data[: int(len(data) * TRAINING_PCT)]
val_data = data[int(len(data) * TRAINING_PCT) :]
block_size = 8  # context size
batch_size = 4
xb, yb = get_batch("train", block_size, batch_size)
model = BigramLanguageModel(VOCAB_SIZE)
logits, loss = model(xb, yb) # lowest loss should be -log(1/vocab_size)

# model.generate(xb, 10)
breakpoint()
