#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Code to fine-tune a transformer model on scientific abstracts. We use a
masked language modeling object as the fine-tuning task.

To ensure we can extract embeddings for genes present in our texts, we use a
custom tokenizer that adds gene names to the vocabulary."""


import argparse
import logging
import os
import pickle
from typing import Set, Union

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm import tqdm  # type: ignore
from transformers import AdamW  # type: ignore
from transformers import DataCollatorForLanguageModeling  # type: ignore
from transformers import DebertaV2ForMaskedLM  # type: ignore
from transformers import DebertaV2Tokenizer  # type: ignore
from transformers import get_linear_schedule_with_warmup  # type: ignore

from streaming_corpus import StreamingCorpus
from utils import _chunk_locator

logging.basicConfig(level=logging.INFO)


def load_tokens(filename: str) -> Set[str]:
    """Load gene tokens from a file."""
    with open(filename, "r") as f:
        return {line.strip().lower() for line in f}


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
    """Main function to fine-tune a transformer model on scientific abstracts."""
    # set some params
    model_name = "microsoft/deberta-v3-base"
    token_file = "/ocean/projects/bio210019p/stevesho/genomic_nlp/embeddings/gene_tokens_nosyn.txt"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root_dir",
        type=str,
        default="/ocean/projects/bio210019p/stevesho/genomic_nlp",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
    )
    args = parser.parse_args()
    abstracts_dir = f"{args.root_dir}/data"
    abstracts = f"{abstracts_dir}/combined/tokens_cleaned_abstracts_casefold_finetune_combined.txt"
    model_out = f"{args.root_dir}/models/deberta"

    # set up distributed training environment
    args.local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend="nccl")
        logging.info(f"Process {args.local_rank} initialized")

    # load gene tokens and make custom tokenizer
    genes = load_tokens(token_file)
    tokenizer = custom_gene_tokenizer(genes=genes)

    # load DeBERTa model
    model = DebertaV2ForMaskedLM.from_pretrained(model_name)
    model.resize_token_embeddings(len(tokenizer))
    logging.info(
        f"Loaded DeBERTa model and resized token embeddings to {len(tokenizer)}"
    )
    device = torch.device(f"cuda:{args.local_rank}" if args.local_rank != -1 else "cpu")
    model.to(device)

    # wrap model in ddp
    if args.local_rank != -1:
        model.to(torch.device(f"cuda:{args.local_rank}"))
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank
        )
    print(
        f"Process {args.local_rank} is using "
        f"{torch.cuda.get_device_name(args.local_rank)}"
    )
    world_size = (
        torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    )

    # set up collator for MLM
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    # load dataset generator
    streaming_dataset = StreamingCorpus(
        file_path=abstracts,
        tokenizer=tokenizer,
        max_length=512,
    )
    logging.info(f"Created StreamingCorpus with {len(streaming_dataset)} abstracts")

    dataloader = DataLoader(
        streaming_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=data_collator,
        pin_memory=True,
    )

    # optimizer and scheduler w/ warmup
    total_steps = (len(dataloader) // world_size) * 3  # 3 epochs
    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    # mixed precision training
    scaler = torch.cuda.amp.GradScaler()

    # training loop
    for epoch in range(3):
        model.train()
        if args.local_rank in {0, -1}:
            epoch_iterator = tqdm(
                total=len(dataloader) // world_size,
                desc=f"Epoch {epoch+1}",
                position=0,
                leave=True,
            )
        for step, batch in enumerate(dataloader):
            batch = {k: v.to(f"cuda:{args.local_rank}") for k, v in batch.items()}

            with torch.autocast(device_type="cuda"):
                outputs = model(**batch)
                loss = (
                    outputs.loss / args.gradient_accumulation_steps
                )  # normalize loss for gradient accumulation

            scaler.scale(loss).backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

                # update progress bar on main process
                if args.local_rank in {0, -1}:
                    epoch_iterator.update(1)
                    epoch_iterator.set_postfix(
                        {
                            "loss": f"{loss.item() * args.gradient_accumulation_steps:.4f}"
                        }
                    )

                    if step % (100 * args.gradient_accumulation_steps) == 0:
                        logging.info(
                            f"Epoch {epoch}, Step {step // world_size}, "
                            f"Loss: {loss.item() * args.gradient_accumulation_steps:.4f}"
                        )

    if args.local_rank in {0, -1}:
        model.save_pretrained(model_out)


if __name__ == "__main__":
    main()
