#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Code to fine-tune a transformer model on scientific abstracts. We use a
masked language modeling object as the fine-tuning task. We fine-tune on DeBERTa
V3, a bi-directional transformer model designed for natural language
understanding (NLU). We choose the base model size for a mix of performance and speed.

To ensure we can extract embeddings for normalized_tokens present in our texts, we use a
custom tokenizer that adds gene names to the vocabulary."""


import argparse
import json
import logging
import os
from typing import Set

import deepspeed  # type: ignore
import torch  # type: ignore
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler
from tqdm import tqdm  # type: ignore
from transformers import BertForMaskedLM  # type: ignore
from transformers import BertTokenizerFast  # type: ignore
from transformers import DataCollatorForLanguageModeling  # type: ignore
from transformers import get_linear_schedule_with_warmup  # type: ignore

from genomic_nlp.utils.streaming_corpus import MLMTextDataset
from genomic_nlp.utils.streaming_corpus import StreamingCorpus

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
    normalized_tokens: Set[str],
    base_model_name: str,
    save_dir: str,
) -> BertTokenizerFast:
    """Create a custom tokenizer and add NEN tokens to ensure consistent
    vocabulary.
    """
    # load base tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(base_model_name)

    # add gene names to the vocabulary
    new_tokens = list(normalized_tokens)
    tokenizer.add_tokens(new_tokens)

    # save the tokenizer
    tokenizer.save_pretrained(save_dir)
    return tokenizer


def main() -> None:
    """Main function to fine-tune a transformer model on scientific abstracts."""
    # set some params
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

    # get val needed for total steps calculation
    parsed_config = parse_deepspeed_config(ds_config_file)
    train_micro_batch_size_per_gpu = parsed_config["train_micro_batch_size_per_gpu"]

    # initialize deepspeed
    deepspeed.init_distributed(dist_backend="nccl")

    # load gene tokens and make custom tokenizer
    gene_tokens = load_tokens(gene_token_file)
    disease_tokens = load_tokens(disease_token_file)
    all_entity_tokens = gene_tokens.union(disease_tokens)
    tokenizer = custom_gene_tokenizer(
        normalized_tokens=all_entity_tokens,
        base_model_name=model_name,
        save_dir=f"{args.root_dir}/models/finetuned_biomedbert/gene_tokenizer",
    )

    # load biomedbert model
    model = BertForMaskedLM.from_pretrained(model_name)
    model.resize_token_embeddings(len(tokenizer))

    # force contiguity of model parameters
    for name, param in model.named_parameters():
        if not param.data.is_contiguous():
            param.data = param.data.contiguous()

    # model.gradient_checkpointing_enable()
    logging.info(
        f"Loaded DeBERTa model and resized token embeddings to {len(tokenizer)}"
    )

    # load dataset generator
    # streaming_dataset = StreamingCorpus(
    #     file_path=abstracts,
    #     tokenizer=tokenizer,
    #     max_length=512,
    # )
    # logging.info(f"Created StreamingCorpus with {len(streaming_dataset)} abstracts")
    streaming_dataset = MLMTextDataset(
        file_path=abstracts,
        tokenizer=tokenizer,
        max_length=512,
    )
    logging.info(f"Created MLMTextDataset with {len(streaming_dataset)} abstracts")

    # get total steps from dataset size
    total_abstracts = len(streaming_dataset)
    num_gpus = (
        torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    )
    num_epochs = 3
    total_steps = _get_total_steps(
        num_gpus, num_epochs, train_micro_batch_size_per_gpu, total_abstracts
    )
    warmup_steps = int(0.1 * total_steps)
    logging.info(f"total steps: {total_steps}, warmup steps: {warmup_steps}")

    # create distributed sampler if in multi-gpu mode
    if torch.distributed.is_initialized():
        sampler = DistributedSampler(streaming_dataset, shuffle=True)  # type: ignore
        print("Distributed Sampler")
    else:
        sampler = None
        print("No Distributed Sampler")

    # set up DeepSpeed
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        config=ds_config_file,
    )

    # try using base optimizer
    base_optimizer = optimizer.optimizer

    scheduler = get_linear_schedule_with_warmup(
        optimizer=base_optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # set up collator for MLM
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    dataloader = DataLoader(
        streaming_dataset,
        batch_size=model_engine.train_micro_batch_size_per_gpu(),
        sampler=sampler,
        num_workers=4,
        collate_fn=data_collator,
        pin_memory=True,
        # prefetch_factor=4,
        persistent_workers=True,
    )

    # early stopping parameters
    best_train_loss = float("inf")
    patience = 30  # total steps to wait = patience * 100
    steps_since_last_improvement = 0

    # training loop!
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
            # move batch to device
            batch = {k: v.to(model_engine.local_rank) for k, v in batch.items()}

            # forward pass
            outputs = model_engine(**batch)
            loss = outputs.loss

            # backward pass
            model_engine.backward(loss)
            model_engine.step()
            scheduler.step()

            # update progress bar on main process
            if args.local_rank in {0, -1}:
                epoch_iterator.set_postfix({"loss": f"{loss.item():.4f}"})
                if step % 100 == 0:
                    current_loss = loss.item()
                    logging.info(
                        f"epoch {epoch}, step {step}, loss: {current_loss:.4f}"
                    )

                    # early stopping check
                    if current_loss < best_train_loss:
                        best_train_loss = current_loss
                        steps_since_last_improvement = 0
                        # save best model so far
                        if args.local_rank in {0, -1}:
                            model_engine.save_checkpoint(
                                best_model_out_dir
                            )  # save to best_model directory
                            logging.info(
                                f"best model updated at step {step}, "
                                f"loss: {best_train_loss:.4f}, "
                                f"saved to {best_model_out_dir}"
                            )
                    else:
                        steps_since_last_improvement += 1
                        logging.info(
                            "no improvement in training loss for "
                            f"{steps_since_last_improvement} checks."
                        )
                        if steps_since_last_improvement >= patience:
                            logging.info(
                                f"early stopping triggered at step {step} "
                                f"due to no improvement in training loss for {patience} checks."
                            )
                            if args.local_rank in {0, -1}:
                                logging.info(
                                    f"loading best model from {best_model_out_dir}..."
                                )
                                model_engine.load_checkpoint(best_model_out_dir)
                            break

        else:
            continue
        break

    if args.local_rank in {0, -1}:
        model_engine.save_checkpoint(final_model_out_dir)
        logging.info(f"final model saved to {final_model_out_dir}")
        logging.info(
            "best model (based on training loss during training) "
            f"saved to {best_model_out_dir}"
        )


if __name__ == "__main__":
    main()
