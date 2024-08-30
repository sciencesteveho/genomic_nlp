#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Code to fine-tune a transformer model on scientific abstracts. We use a
masked language modeling object as the fine-tuning task.

To ensure we can extract embeddings for genes present in our texts, we use a
custom tokenizer that adds gene names to the vocabulary."""


import argparse
import os
import pickle
from typing import Set, Union

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm  # type: ignore
from transformers import DataCollatorForLanguageModeling  # type: ignore
from transformers import DebertaV2ForMaskedLM  # type: ignore
from transformers import DebertaV2Tokenizer  # type: ignore
from transformers import Trainer  # type: ignore
from transformers import TrainingArguments  # type: ignore

from streaming_corpus import FinetuneStreamingCorpus
from streaming_corpus import StreamingCorpus
from utils import _chunk_locator


def load_tokens(filename: str) -> Set[str]:
    """Load gene tokens from a file."""
    with open(filename, "r") as f:
        return {line.strip().lower() for line in f}


def custom_gene_tokenizer(
    genes: Set[str], base_model_name: str = "microsoft/deberta-v3-base"
) -> DebertaV2Tokenizer:
    """Create a custom tokenizer and add gene tokens"""
    # load base tokenizer
    tokenizer = DebertaV2Tokenizer.from_pretrained(base_model_name)

    # add gene names to the vocabulary
    new_tokens = list(genes)
    tokenizer.add_tokens(new_tokens)

    # save the tokenizer
    save_dir = (
        "/ocean/projects/bio210019p/stevesho/genomic_nlp/models/deberta/gene_tokenizer"
    )
    tokenizer.save_pretrained(save_dir)
    return DebertaV2Tokenizer.from_pretrained(save_dir)


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

    # load gene tokens
    token_file = "/ocean/projects/bio210019p/stevesho/genomic_nlp/embeddings/gene_tokens_nosyn.txt"
    genes = load_tokens(token_file)

    # custom tokenizer
    tokenizer = custom_gene_tokenizer(genes=genes)

    # load DeBERTa model
    model_name = "microsoft/deberta-v3-base"
    model = DebertaV2ForMaskedLM.from_pretrained(model_name)
    model.resize_token_embeddings(len(tokenizer))

    # wrap model in ddp
    if args.local_rank != -1:
        model.to(torch.device(f"cuda:{args.local_rank}"))
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank
        )

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
        file_path=abstracts,
        tokenizer=tokenizer,
        max_length=512,
    )

    # set up total steps
    num_gpus = 2
    num_epochs = 3
    batch_size = 32
    total_abstracts = 3889578
    if num_gpus > 1:
        max_steps = ((total_abstracts * num_epochs) // batch_size) // num_gpus
    else:
        max_steps = (total_abstracts * num_epochs) // batch_size

    # set up dataloader
    sampler: Union[DistributedSampler, None] = (
        DistributedSampler(streaming_dataset) if args.local_rank != -1 else None
    )
    data_loader = DataLoader(
        streaming_dataset,
        batch_size=batch_size,
        collate_fn=data_collator,
        num_workers=4,
        sampler=sampler,
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
        # fp16=True,  # mixed precision training
        local_rank=args.local_rank,
    )

    # Initialize Trainer
    # trainer = StreamingTrainer(
    #     model=model,
    #     args=training_args,
    #     data_collator=data_collator,
    # )
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=streaming_dataset,
    )

    trainer.train()

    # save model on the main process
    if args.local_rank in {0, -1}:
        trainer.save_model(f"{args.root_dir}/models/deberta")


if __name__ == "__main__":
    main()
