#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Code to fine-tune a transformer model on scientific abstracts. We use a
masked language modeling objective as the fine-tuning task. We fine-tune on
a bi-directional transformer model (BiomedBERT) designed for natural language
understanding (NLU). We choose the base model size for a mix of performance and
speed.

To ensure we can extract embeddings for normalized_tokens in our texts, we use a
custom tokenizer that adds gene/disease tokens to the vocabulary.
"""

import argparse
import json
import logging
import os
from typing import Set

import deepspeed  # type: ignore
import torch  # type: ignore
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler
from tqdm import tqdm  # type: ignore
from transformers import BertForMaskedLM  # type: ignore
from transformers import BertTokenizerFast  # type: ignore
from transformers import DataCollatorForLanguageModeling  # type: ignore
from transformers import get_linear_schedule_with_warmup  # type: ignore

from genomic_nlp.utils.streaming_corpus import MLMTextDataset

logging.basicConfig(level=logging.INFO)


def load_tokens(filename: str) -> Set[str]:
    """Load tokens from a file."""
    with open(filename, "r") as f:
        return {line.strip().lower() for line in f}


def parse_deepspeed_config(config_file: str) -> dict:
    """Parse the DeepSpeed configuration file."""
    with open(config_file, "r") as f:
        return json.load(f)


def _get_total_steps(
    num_gpus: int, num_epochs: int, batch_size: int, total_abstracts: int
) -> int:
    """Compute total training steps for the trainer."""
    if num_gpus > 1:
        return ((total_abstracts * num_epochs) // batch_size) // num_gpus
    else:
        return (total_abstracts * num_epochs) // batch_size


def custom_gene_tokenizer(
    normalized_tokens: Set[str],
    base_model_name: str,
    save_dir: str,
) -> BertTokenizerFast:
    """Create a custom tokenizer and add given tokens to ensure consistent vocab."""
    # load base tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(base_model_name)

    # add tokens
    new_tokens = list(normalized_tokens)
    tokenizer.add_tokens(new_tokens)

    # save the tokenizer
    tokenizer.save_pretrained(save_dir)
    return tokenizer


def main() -> None:
    """Main function to fine-tune a transformer model on scientific abstracts."""
    model_name = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"
    gene_token_file = "/ocean/projects/bio210019p/stevesho/genomic_nlp/embeddings/gene_tokens_nosyn.txt"
    disease_token_file = "/ocean/projects/bio210019p/stevesho/genomic_nlp/training_data/disease/disease_tokens_nosyn.txt"
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
    abstracts = f"{abstracts_dir}/combined/processed_abstracts_finetune_combined.txt"
    model_out = f"{args.root_dir}/models/finetuned_biomedbert"
    best_model_out_dir = os.path.join(model_out, "best_model")
    final_model_out_dir = os.path.join(model_out, "final_model")

    parsed_config = parse_deepspeed_config(ds_config_file)
    train_micro_batch_size_per_gpu = parsed_config["train_micro_batch_size_per_gpu"]
    deepspeed.init_distributed(dist_backend="nccl")
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    gene_tokens = load_tokens(gene_token_file)
    disease_tokens = load_tokens(disease_token_file)
    all_entity_tokens = gene_tokens.union(disease_tokens)

    tokenizer = custom_gene_tokenizer(
        normalized_tokens=all_entity_tokens,
        base_model_name=model_name,
        save_dir=os.path.join(model_out, "gene_tokenizer"),
    )

    model = BertForMaskedLM.from_pretrained(model_name)
    model.resize_token_embeddings(len(tokenizer))

    # force contiguous
    for _, param in model.named_parameters():
        if not param.data.is_contiguous():
            param.data = param.data.contiguous()

    logging.info(
        f"Loaded BiomedBERT model and resized token embeddings to {len(tokenizer)}"
    )

    # load data
    streaming_dataset = MLMTextDataset(
        file_path=abstracts,
        tokenizer=tokenizer,
        max_length=512,
    )
    total_abstracts = len(streaming_dataset)
    logging.info(f"Created MLMTextDataset with {total_abstracts} abstracts")

    num_epochs = 3
    total_steps = _get_total_steps(
        world_size, num_epochs, train_micro_batch_size_per_gpu, total_abstracts
    )
    warmup_steps = int(0.1 * total_steps)
    logging.info(f"total steps: {total_steps}, warmup steps: {warmup_steps}")

    sampler = None
    if dist.is_initialized() and world_size > 1:
        sampler = DistributedSampler(streaming_dataset, shuffle=True)
        logging.info("Using DistributedSampler")
    else:
        logging.info("Not using DistributedSampler")

    dataloader = DataLoader(
        streaming_dataset,
        batch_size=train_micro_batch_size_per_gpu,
        sampler=sampler,
        num_workers=4,
        collate_fn=DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=True, mlm_probability=0.15
        ),
        pin_memory=True,
        persistent_workers=True,
    )

    # deep speed engine
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        config=ds_config_file,
    )
    base_optimizer = optimizer.optimizer
    scheduler = get_linear_schedule_with_warmup(
        optimizer=base_optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # early stopping - now just best model saving
    best_train_loss = float("inf")
    patience = 50

    steps_since_last_improvement = 0

    for epoch in range(num_epochs):
        model_engine.train()

        if sampler is not None:
            sampler.set_epoch(epoch)

        if args.local_rank in {0, -1}:
            epoch_iterator = tqdm(
                dataloader,
                desc=f"epoch {epoch+1}",
                position=0,
                leave=True,
                total=len(dataloader),
            )
        else:
            epoch_iterator = dataloader

        for step, batch in enumerate(epoch_iterator):
            # move batch to correct device
            batch = {k: v.to(model_engine.local_rank) for k, v in batch.items()}

            outputs = model_engine(**batch)
            loss = outputs.loss

            model_engine.backward(loss)
            model_engine.step()
            scheduler.step()

            # only rank 0 logs and checks best model
            if args.local_rank in {0, -1}:
                epoch_iterator.set_postfix({"loss": f"{loss.item():.4f}"})

                # check improvement every 100 steps
                if step % 100 == 0:
                    current_loss = loss.item()
                    logging.info(
                        f"epoch {epoch}, step {step}, loss: {current_loss:.4f}"
                    )

                    if current_loss < best_train_loss:
                        best_train_loss = current_loss
                        steps_since_last_improvement = 0
                        # save best model so far
                        model_engine.save_checkpoint(best_model_out_dir)
                        logging.info(
                            f"Best model updated at step={step}, "
                            f"loss={best_train_loss:.4f}, saved to {best_model_out_dir}"
                        )
                    else:
                        steps_since_last_improvement += 1
                        logging.info(
                            f"No improvement in training loss for "
                            f"{steps_since_last_improvement} checks."
                        )
                        if steps_since_last_improvement >= patience:
                            logging.info(
                                f"Patience reached at step {step} "
                                f"due to no improvement for {patience} checks, but continuing training."
                            )

    if dist.is_initialized():
        dist.barrier()

    # save final model on rank 0
    if args.local_rank in {0, -1}:
        model_engine.save_checkpoint(final_model_out_dir)
        logging.info(f"Final model saved to {final_model_out_dir}")
        logging.info(
            f"Best model (based on training loss) saved to {best_model_out_dir}"
        )


if __name__ == "__main__":
    main()
