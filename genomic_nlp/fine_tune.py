#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Code to fine-tune a transformer model on scientific abstracts. We use a
masked language modeling object as the fine-tuning task. We fine-tune on DeBERTa
V3, a bi-directional transformer model designed for natural language
understanding (NLU). We choose the base model size for a mix of performance and speed.

To ensure we can extract embeddings for genes present in our texts, we use a
custom tokenizer that adds gene names to the vocabulary."""


import argparse
import json
import logging
import os
import pickle
from typing import Set, Union

import deepspeed  # type: ignore
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm import tqdm  # type: ignore
from transformers import AdamW  # type: ignore
from transformers import DataCollatorForLanguageModeling  # type: ignore
from transformers import DebertaV2ForMaskedLM  # type: ignore
from transformers import DebertaV2TokenizerFast  # type: ignore
from transformers import get_linear_schedule_with_warmup  # type: ignore

from streaming_corpus import StreamingCorpus
from utils import _chunk_locator

logging.basicConfig(level=logging.INFO)


def load_tokens(filename: str) -> Set[str]:
    """Load gene tokens from a file."""
    with open(filename, "r") as f:
        return {line.strip().lower() for line in f}


def parse_deepspeed_config(config_file: str) -> dict:
    """Parse the DeepSpeed configuration file."""
    with open(config_file, "r") as f:
        return json.load(f)


def _get_total_steps(
    num_gpus: int, num_epochs: int, batch_size: int, total_abstracts: int
) -> int:
    """Get total steps for the trainer."""
    if num_gpus > 1:
        return ((total_abstracts * num_epochs) // batch_size) // num_gpus
    else:
        return (total_abstracts * num_epochs) // batch_size


def custom_gene_tokenizer(
    genes: Set[str], base_model_name: str = "microsoft/deberta-v3-base"
) -> DebertaV2TokenizerFast:
    """Create a custom tokenizer and add gene tokens"""
    # load base tokenizer
    tokenizer = DebertaV2TokenizerFast.from_pretrained(base_model_name)

    # add gene names to the vocabulary
    new_tokens = list(genes)
    tokenizer.add_tokens(new_tokens)

    # save the tokenizer
    save_dir = (
        "/ocean/projects/bio210019p/stevesho/genomic_nlp/models/deberta/gene_tokenizer"
    )
    tokenizer.save_pretrained(save_dir)
    return tokenizer


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
    """Main function to fine-tune a transformer model on scientific abstracts."""
    # set some params
    model_name = "microsoft/deberta-v3-base"
    token_file = "/ocean/projects/bio210019p/stevesho/genomic_nlp/embeddings/gene_tokens_nosyn.txt"
    ds_config_file = (
        "/ocean/projects/bio210019p/stevesho/genomic_nlp/deepspeed_config.json"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root_dir",
        type=str,
        default="/ocean/projects/bio210019p/stevesho/genomic_nlp",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank for distributed training on GPUs",
    )
    parser.add_argument(
        "--deepspeed_config",
        type=str,
        default="ds_config.json",
        help="DeepSpeed configuration file",
    )
    args = parser.parse_args()
    abstracts_dir = f"{args.root_dir}/data"
    abstracts = f"{abstracts_dir}/combined/tokens_cleaned_abstracts_casefold_finetune_combined.txt"
    model_out = f"{args.root_dir}/models/deberta"

    # get val needed for total steps calculation
    parsed_config = parse_deepspeed_config(ds_config_file)
    train_micro_batch_size_per_gpu = parsed_config["train_micro_batch_size_per_gpu"]

    # initialize deepspeed
    deepspeed.init_distributed(dist_backend="nccl")

    # load gene tokens and make custom tokenizer
    genes = load_tokens(token_file)
    tokenizer = custom_gene_tokenizer(genes=genes)

    # load DeBERTa model
    model = DebertaV2ForMaskedLM.from_pretrained(model_name)
    model.resize_token_embeddings(len(tokenizer))
    model.gradient_checkpointing_enable()
    logging.info(
        f"Loaded DeBERTa model and resized token embeddings to {len(tokenizer)}"
    )

    # optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=2e-5, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01
    )

    # load dataset generator
    streaming_dataset = StreamingCorpus(
        file_path=abstracts,
        tokenizer=tokenizer,
        max_length=512,
    )
    logging.info(f"Created StreamingCorpus with {len(streaming_dataset)} abstracts")

    # scheduler with warmup
    total_steps = len(streaming_dataset) // (
        torch.distributed.get_world_size() * train_micro_batch_size_per_gpu
    )
    warmup_steps = int(0.1 * total_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # set up DeepSpeed
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        config=ds_config_file,
        lr_scheduler=scheduler,
    )

    # set up collator for MLM
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    dataloader = DataLoader(
        streaming_dataset,
        batch_size=model_engine.train_micro_batch_size_per_gpu(),
        num_workers=4,
        collate_fn=data_collator,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True,
        shuffle=False,
    )

    for epoch in range(3):
        model_engine.train()
        if args.local_rank in {0, -1}:
            epoch_iterator = tqdm(
                total=total_steps,
                desc=f"Epoch {epoch+1}",
                position=0,
                leave=True,
            )
        for step, batch in enumerate(dataloader):
            # move batch to device
            batch = {k: v.to(model_engine.local_rank) for k, v in batch.items()}

            # forward pass
            outputs = model_engine(**batch)
            loss = outputs.loss

            # backward pass
            model_engine.backward(loss)
            model_engine.step()

            # update progress bar on main process
            if args.local_rank in {0, -1}:
                epoch_iterator.update(1)
                epoch_iterator.set_postfix({"loss": f"{loss.item():.4f}"})

                if step % 100 == 0:
                    logging.info(f"Epoch {epoch}, Step {step}, Loss: {loss.item():.4f}")

            # break if we've processed all steps
            if step >= total_steps:
                break

    # save the model
    if args.local_rank in {0, -1}:
        model_engine.save_checkpoint(model_out)


if __name__ == "__main__":
    main()
