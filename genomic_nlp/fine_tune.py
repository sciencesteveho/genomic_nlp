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
from torch.cuda import autocast  # type: ignore
from torch.cuda import GradScaler  # type: ignore
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm  # type: ignore
from transformers import AdamW  # type: ignore
from transformers import DataCollatorForLanguageModeling  # type: ignore
from transformers import DebertaV2ForMaskedLM  # type: ignore
from transformers import DebertaV2Tokenizer  # type: ignore
from transformers import get_linear_schedule_with_warmup  # type: ignore
from transformers import Trainer  # type: ignore
from transformers import TrainingArguments  # type: ignore

from streaming_corpus import FinetuneStreamingCorpus
from streaming_corpus import RobustDataCollator
from streaming_corpus import SimpleStreamingCorpus
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
    """Main function"""
    # load classified abstracts
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root_dir",
        type=str,
        default="/ocean/projects/bio210019p/stevesho/genomic_nlp",
    )
    args = parser.parse_args()

    abstracts_dir = f"{args.root_dir}/data"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.local_rank = int(os.environ.get("LOCAL_RANK", -1))

    # set up distributed training environment
    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)  # set the device
        dist.init_process_group(backend="nccl")  # initialize process group
        logging.info(f"Process {args.local_rank} initialized")

    # load gene tokens
    token_file = "/ocean/projects/bio210019p/stevesho/genomic_nlp/embeddings/gene_tokens_nosyn.txt"
    genes = load_tokens(token_file)

    # custom tokenizer
    tokenizer = custom_gene_tokenizer(genes=genes)

    # load DeBERTa model
    model_name = "microsoft/deberta-v3-base"
    model = DebertaV2ForMaskedLM.from_pretrained(model_name)
    model.resize_token_embeddings(len(tokenizer))
    logging.info(
        f"Loaded DeBERTa model and resized token embeddings to {len(tokenizer)}"
    )

    # wrap model in ddp
    if args.local_rank != -1:
        model.to(torch.device(f"cuda:{args.local_rank}"))
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank
        )
    print(
        f"Process {args.local_rank} is using {torch.cuda.get_device_name(args.local_rank)}"
    )

    # data_collator = RobustDataCollator(
    #     tokenizer=tokenizer,
    #     mlm=True,
    #     mlm_probability=0.15,
    # )
    # logging.info("Created data collator.")
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    # load dataset generator
    abstracts = f"{abstracts_dir}/combined/tokens_cleaned_abstracts_casefold_finetune_combined.txt"
    streaming_dataset = StreamingCorpus(
        file_path=abstracts,
        tokenizer=tokenizer,
        max_length=512,
    )
    logging.info(f"Created StreamingCorpus with {len(streaming_dataset)} abstracts")

    dataloader = DataLoader(
        streaming_dataset, batch_size=8, num_workers=4, collate_fn=data_collator
    )

    # optimizer
    optimizer = AdamW(model.parameters(), lr=5e-5)

    # gradient accumulation
    accumulation_steps = 4  # Effective batch size will be 4 * 4 = 16

    # mixed precision training
    scaler = GradScaler()

    # get steps and scheduler
    total_steps = len(dataloader) * 3  # 3 epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )

    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # training loop
    for epoch in range(3):
        model.train()
        for step, batch in enumerate(dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}

            with autocast():
                outputs = model(**batch)
                loss = outputs.loss / accumulation_steps  # normalize loss

            scaler.scale(loss).backward()

            if (step + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            if step % 100 == 0:
                print(
                    f"Epoch {epoch}, Step {step}, Loss: {loss.item() * accumulation_steps}"
                )

    if args.local_rank in {0, -1}:
        model.save_pretrained(f"{args.root_dir}/models/deberta")

    # get max steps fro trainer
    # num_gpus = 2
    # num_epochs = 3
    # batch_size = 16
    # total_abstracts = 3889578
    # max_steps = _get_total_steps(num_gpus, num_epochs, batch_size, total_abstracts)
    # logging.info(f"Calculated max_steps: {max_steps}")

    # # test actual training loop behavior
    # logging.info("Testing training loop behavior")
    # train_dataloader = DataLoader(
    #     streaming_dataset,
    #     batch_size=16,
    #     collate_fn=data_collator,
    # )

    # for i, batch in enumerate(train_dataloader):
    #     logging.info(f"Batch {i} keys: {batch.keys()}")
    #     if "input_ids" in batch:
    #         logging.info(f"Batch {i} input_ids shape: {batch['input_ids'].shape}")
    #     else:
    #         logging.warning(f"Batch {i} is missing input_ids")
    #     if i >= 5:  # test first 5 batches
    #         break

    # # define training arguments
    # training_args = TrainingArguments(
    #     output_dir=f"{args.root_dir}/models/deberta",
    #     overwrite_output_dir=True,
    #     num_train_epochs=num_epochs,
    #     per_device_train_batch_size=batch_size,
    #     save_steps=10_000,
    #     save_total_limit=2,
    #     prediction_loss_only=True,
    #     logging_dir=f"{args.root_dir}p/models/logs",
    #     logging_steps=500,
    #     max_steps=max_steps,
    #     # fp16=True,  # mixed precision training
    #     local_rank=args.local_rank,
    # )

    # # initialize trainer
    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     data_collator=data_collator,
    #     train_dataset=streaming_dataset,
    # )
    # logging.info("Starting training...")
    # try:
    #     trainer.train()
    # except Exception as e:
    #     logging.error(f"Error during training: {str(e)}")
    #     raise

    # save model on the main process
    # if args.local_rank in {0, -1}:
    #     trainer.save_model(f"{args.root_dir}/models/deberta")


if __name__ == "__main__":
    main()


# class StreamingTrainer(Trainer):
#     """Helper class to override get_train_dataloader method"""

#     def get_train_dataloader(self) -> DataLoader:
#         """Return the training dataloader"""
#         return data_loader

# set up dataloader
# sampler: Union[DistributedSampler, None] = (
#     DistributedSampler(streaming_dataset) if args.local_rank != -1 else None
# )
# data_loader = DataLoader(
#     streaming_dataset,
#     batch_size=batch_size,
#     collate_fn=data_collator,
#     num_workers=4,
#     # sampler=sampler,
# )

# trainer = StreamingTrainer(
#     model=model,
#     args=training_args,
#     data_collator=data_collator,
# )
