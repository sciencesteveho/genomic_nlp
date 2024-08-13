#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Code to fine-tune a transformer model on scientific abstracts. We use a
masked language modeling object as the fine-tuning task."""

import argparse
import math
import os
import pickle

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data import IterableDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm  # type: ignore
from transformers import DataCollatorForLanguageModeling  # type: ignore
from transformers import DebertaV2ForMaskedLM  # type: ignore
from transformers import DebertaV2Tokenizer  # type: ignore
from transformers import Trainer  # type: ignore
from transformers import TrainingArguments  # type: ignore

from utils import _chunk_locator

# from datasets import Dataset
# from datasets import Features
# from datasets import Value


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


# class StreamingCorpus(IterableDataset):
#     """Class to create a Hugging Face dataset object from text corpus as an iterable"""

#     def __init__(self, dataset_file, tokenizer, data_collator, max_length=512):
#         self.dataset_file = dataset_file
#         self.tokenizer = tokenizer
#         self.data_collator = data_collator
#         self.max_length = max_length

#     def __iter__(self):
#         """Iterate over the dataset file and yield tokenized examples"""
#         # Opens the file, ensuring it's closed after iteration
#         with open(self.dataset_file, "r", encoding="utf-8") as file_iterator:
#             for line in file_iterator:
#                 abstract = line.strip()
#                 tokenized = self.tokenizer(
#                     abstract,
#                     padding="max_length",
#                     truncation=True,
#                     max_length=self.max_length,
#                     return_tensors="pt",
#                 )

#                 # inputs = self.data_collator([tokenized])
#                 yield {k: v.squeeze(0) for k, v in tokenized.items()}
#                 # yield {k: v.squeeze(0) for k, v in inputs.items()}


class StreamingCorpus(IterableDataset):
    """Class to create a Hugging Face dataset object from text corpus as an iterable"""

    def __init__(self, dataset_file, tokenizer, data_collator, max_length=512):
        self.dataset_file = dataset_file
        self.tokenizer = tokenizer
        self.data_collator = data_collator
        self.max_length = max_length

    def __iter__(self):
        """Iterate over the dataset file and yield tokenized examples"""
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:
            # single-process data loading, return the full iterator
            start_position = 0
            end_position = None  # Read until the end of the file
        else:
            # Calculate start and end positions for this worker's shard
            start_position = worker_info.id * self._shard_size(worker_info.num_workers)
            end_position = start_position + self._shard_size(worker_info.num_workers)

        # Opens the file, ensuring it's closed after iteration
        with open(self.dataset_file, "r", encoding="utf-8") as file_iterator:
            if start_position > 0:
                # Skip lines up to the start position
                for _ in range(start_position):
                    next(file_iterator)

            for line_number, line in enumerate(file_iterator):
                if end_position is not None and line_number >= end_position:
                    break

                abstract = line.strip()
                tokenized = self.tokenizer(
                    abstract,
                    padding="max_length",
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                yield {k: v.squeeze(0) for k, v in tokenized.items()}

    def _shard_size(self, num_workers):
        """Estimate shard size based on the total number of workers"""
        with open(self.dataset_file, "r", encoding="utf-8") as f:
            total_lines = sum(1 for _ in f)

        return math.ceil(total_lines / num_workers)


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
    args.local_rank = int(os.environ.get("LOCAL_RANK", -1))

    # Set up distributed training environment
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)  # Set the device
        dist.init_process_group(backend="nccl")  # Initialize process group

    # write abstracts to text
    # _write_abstracts_to_text(
    #     abstracts_dir=abstracts_dir, prefix="tokens_cleaned_abstracts_casefold_finetune"
    # )

    # load DeBERTa model and tokenizer
    # model_name = "microsoft/deberta-base"
    model_name = "microsoft/deberta-v3-base"
    model = DebertaV2ForMaskedLM.from_pretrained(model_name)
    tokenizer = DebertaV2Tokenizer.from_pretrained(model_name)

    # wrap model in ddp
    if args.local_rank != -1:
        model.to(torch.device(f"cuda:{args.local_rank}"))
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank
        )
    # model.to(device)

    # check ddp
    print(
        f"Process {args.local_rank} is using {torch.cuda.get_device_name(args.local_rank)}"
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15,
    )

    # load dataset generator
    abstracts = f"{abstracts_dir}/combined/tokens_cleaned_abstracts_casefold_finetune_combined.txt"

    streaming_dataset = StreamingCorpus(
        dataset_file=abstracts,
        tokenizer=tokenizer,
        data_collator=data_collator,
        max_length=512,
    )

    # set up total steps
    # num_gpus = torch.cuda.device_count()
    num_gpus = 8
    num_epochs = 3
    batch_size = 12
    total_abstracts = 3889578
    max_steps = (total_abstracts * num_epochs) // batch_size
    ddp_max_steps = ((total_abstracts * num_epochs) // batch_size) // num_gpus

    # current num steps = total abstracts * num epochs / gpus
    # new formula // max steps = num epochs (total_abstracts * num_epochs) // batch_size // gpus

    # set up dataloader
    data_loader = DataLoader(
        streaming_dataset,
        batch_size=batch_size,
        collate_fn=data_collator,
        num_workers=4,
    )

    class StreamingTrainer(Trainer):
        def get_train_dataloader(self):
            return data_loader

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=f"{args.root_dir}/models/deberta",
        overwrite_output_dir=True,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        # auto_find_batch_size=True,
        save_steps=10_000,
        save_total_limit=2,
        prediction_loss_only=True,
        logging_dir="/ocean/projects/bio210019p/stevesho/nlp/models/logs",
        logging_steps=500,
        max_steps=ddp_max_steps,
        fp16=True,  # mixed precision training
        local_rank=args.local_rank,
    )

    # Initialize Trainer
    trainer = StreamingTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(f"{args.root_dir}/models/deberta")

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
