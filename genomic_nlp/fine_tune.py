#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Code to fine-tune a transformer model on scientific abstracts. We use a
masked language modeling object as the fine-tuning task."""


import argparse
import os
import pickle

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from tqdm import tqdm  # type: ignore
from transformers import DataCollatorForLanguageModeling  # type: ignore
from transformers import DebertaV2ForMaskedLM  # type: ignore
from transformers import DebertaV2Tokenizer  # type: ignore
from transformers import Trainer  # type: ignore
from transformers import TrainingArguments  # type: ignore

from streaming_corpus import FinetuneStreamingCorpus
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

    streaming_dataset = FinetuneStreamingCorpus(
        dataset_file=abstracts,
        tokenizer=tokenizer,
        data_collator=data_collator,
        max_length=512,
    )

    # set up total steps
    num_gpus = 8  # num_gpus = torch.cuda.device_count()
    num_epochs = 3
    batch_size = 12
    total_abstracts = 3889578
    if num_gpus > 1:
        max_steps = ((total_abstracts * num_epochs) // batch_size) // num_gpus
    else:
        max_steps = (total_abstracts * num_epochs) // batch_size

    # set up dataloader
    data_loader = DataLoader(
        streaming_dataset,
        batch_size=batch_size,
        collate_fn=data_collator,
        num_workers=4,
    )

    class StreamingTrainer(Trainer):
        """Helper class to override get_train_dataloader method"""

        def get_train_dataloader(self) -> DataLoader:
            """Return the training dataloader"""
            return data_loader

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=f"{args.root_dir}/models/deberta",
        overwrite_output_dir=True,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        save_steps=10_000,
        save_total_limit=2,
        prediction_loss_only=True,
        logging_dir="/ocean/projects/bio210019p/stevesho/nlp/models/logs",
        logging_steps=500,
        max_steps=max_steps,
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


if __name__ == "__main__":
    main()
