#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Code to fine-tune a transformer model on scientific abstracts. We use a
masked language modeling object as the fine-tuning task."""

import argparse
import pickle
from typing import Generator

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm  # type: ignore
from transformers import DataCollatorForLanguageModeling  # type: ignore
from transformers import DebertaV2ForMaskedLM  # type: ignore
from transformers import DebertaV2Tokenizer  # type: ignore
from transformers import Trainer  # type: ignore
from transformers import TrainingArguments  # type: ignore

from utils import _chunk_locator


def _write_abstracts_to_text(abstracts_dir: str, prefix: str) -> None:
    """Write chunks of abstracts to text, where each newline delimits a full
    abstract."""
    filenames = _chunk_locator(abstracts_dir, prefix)
    with open(f"{abstracts_dir}/combined/{prefix}_combined.txt", "w") as output:
        for filename in filenames:
            with open(filename, "rb") as file:
                abstracts = pickle.load(file)
                for abstract in abstracts:
                    line = " ".join(
                        [" ".join(sentence) for sentence in abstract]
                    ).strip()
                    output.write(f"{line}\n")


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
    # load classified abstracts
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root_dir", type=str, default="/ocean/projects/bio210019p/stevesho/nlp"
    )
    args = parser.parse_args()

    abstracts_dir = f"{args.root_dir}/data"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # write abstracts to text
    # _write_abstracts_to_text(
    #     abstracts_dir=abstracts_dir, prefix="tokens_cleaned_abstracts_casefold_finetune"
    # )

    # load DeBERTa model and tokenizer
    model_name = "microsoft/deberta-v3-base"

    # model_name = "microsoft/deberta-base"
    model = DebertaV2ForMaskedLM.from_pretrained(model_name)
    tokenizer = DebertaV2Tokenizer.from_pretrained(model_name)
    model.to(device)

    # load dataset generator
    dataset = DatasetCorpus(
        stream_abstracts(
            f"{abstracts_dir}/combined/tokens_cleaned_abstracts_casefold_finetune_combined.txt"
        ),
        tokenizer,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15,
    )

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=f"{args.root_dir}/models/deberta",
        overwrite_output_dir=True,
        num_train_epochs=5,
        per_device_train_batch_size=64,
        save_steps=10_000,
        save_total_limit=2,
        prediction_loss_only=True,
        logging_dir="/ocean/projects/bio210019p/stevesho/nlp/models/logs",
        logging_steps=500,
        max_steps=len(dataset) * 5 // 64,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    trainer.train()

    # # set up dataloader
    # data_loader = DataLoader(
    #     dataset,
    #     batch_size=64,
    #     shuffle=True,
    #     pin_memory=True,
    #     prefetch_factor=2,
    #     num_workers=4,
    # )

    # # Set up the optimizer and learning rate scheduler
    # optim = torch.optim.AdamW(model.parameters(), lr=2e-5)

    # epochs = 5
    # for epoch in range(epochs):
    #     total_loss = 0.0
    #     batch_count = 0

    #     for batch in tqdm(data_loader, desc=f"Epoch {epoch}"):
    #         optim.zero_grad()

    #         batch = {k: v.to(device) for k, v in batch.items() if v is not None}

    #         outputs = model(**batch)
    #         loss = outputs.loss
    #         loss.backward()
    #         optim.step()

    #         total_loss += loss.item()
    #         batch_count += 1

    #     avg_loss = total_loss / batch_count
    #     print(f"Epoch {epoch} average loss: {avg_loss}")

    # # save model
    # model_dir = f"{args.root_dir}/models/deberta"
    # model.save_pretrained(model_dir)


if __name__ == "__main__":
    main()
