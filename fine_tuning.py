#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Code to fine-tune a transformer model on scientific abstracts. We use a
masked language modeling object as the fine-tuning task."""

from typing import Generator

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm  # type: ignore
from transformers import DebertaForMaskedLM  # type: ignore
from transformers import DebertaTokenizer  # type: ignore


class DatasetCorpus(Dataset):
    """Class to create a huggingface dataset object from text corpus"""

    def __init__(self, abstracts, tokenizer, max_length=512):
        self.abstracts = abstracts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.abstracts)

    def __getitem__(self, idx):
        abstract = str(self.abstracts[idx])
        inputs = self.tokenizer(
            abstract,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        inputs["labels"] = inputs.input_ids.clone()
        return inputs


def stream_abstracts(dataset_file: str) -> Generator[str, None, None]:
    """Produces a generator that streams abstracts one by one"""
    with open(dataset_file, "r") as f:
        for line in f:
            yield line.strip()


def main() -> None:
    """Main function"""
    # load DeBERTa model and tokenizer
    model_name = "microsoft/deberta-base"
    model = DebertaForMaskedLM.from_pretrained(model_name)
    tokenizer = DebertaTokenizer.from_pretrained(model_name)

    # load dataset generator
    dataset = DatasetCorpus(stream_abstracts("data/abstracts.txt"), tokenizer)

    # set up dataloader
    data_loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=True,
        pin_memory=True,
        prefetch_factor=2,
        num_workers=8,
    )

    # Set up the optimizer and learning rate scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3, verbose=True
    )

    epochs = 5
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch in tqdm(data_loader, desc=f"Epoch {epoch}"):
            outputs = model(**batch, output_attentions=True)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            model.zero_grad()
            train_loss += loss.item()

        print(f"Epoch {epoch} loss: {train_loss}")

    # save model
    model.save_pretrained("models/deberta_abstracts")


if __name__ == "__main__":
    main()
