#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Code to fine-tune a transformer model on scientific abstracts using Hugging Face Trainer."""

import argparse
import json
import logging
import os
from typing import Any, Dict, Set

import torch  # type: ignore
from transformers import AdamW  # type: ignore
from transformers import BertForMaskedLM  # type: ignore
from transformers import BertTokenizerFast  # type: ignore
from transformers import DataCollatorForLanguageModeling  # type: ignore
from transformers import get_linear_schedule_with_warmup  # type: ignore
from transformers import Trainer  # type: ignore
from transformers import TrainerCallback  # type: ignore
from transformers import TrainerControl  # type: ignore
from transformers import TrainerState  # type: ignore
from transformers import TrainingArguments  # type: ignore

from genomic_nlp.utils.streaming_corpus import MLMTextDataset

logging.basicConfig(level=logging.INFO)


class SaveBestTrainingLossCallback(TrainerCallback):
    """A custom callback to save the best model checkpoint during training based on training loss.
    (Corrected version using state.global_step)
    """

    def __init__(self, save_path: str):
        self.best_loss = float("inf")
        self.save_path = save_path

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Dict[str, Any],
    ) -> None:
        """
        Event called at the end of each training step.
        (Corrected version using state.global_step)
        """
        if state.global_step % args.logging_steps == 0:
            if state.log_history:
                current_log = state.log_history[-1]
                if "loss" in current_log:
                    current_loss = current_log["loss"]
                    if current_loss < self.best_loss:
                        self.best_loss = current_loss
                        logging.info(
                            f"Training loss improved at step {state.global_step} to {self.best_loss:.4f}. Saving model..."
                        )
                        kwargs["model"].save_pretrained(self.save_path)
                else:
                    logging.warning(
                        f"Loss key not found in log history at step {state.global_step}"
                    )
            else:
                logging.warning(f"Log history is empty at step {state.global_step}")


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
    """Create a custom tokenizer."""
    tokenizer = BertTokenizerFast.from_pretrained(base_model_name)
    new_tokens = list(normalized_tokens)
    tokenizer.add_tokens(new_tokens)
    tokenizer.save_pretrained(save_dir)
    return tokenizer


def main() -> None:
    """Main function to fine-tune a transformer model using HF Trainer."""
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        logging.info("CUDA_VISIBLE_DEVICES env var is set. Unsetting in script...")
        del os.environ["CUDA_VISIBLE_DEVICES"]
        logging.info(
            f"CUDA_VISIBLE_DEVICES is now: {os.environ.get('CUDA_VISIBLE_DEVICES')}"
        )
    else:
        logging.info("CUDA_VISIBLE_DEVICES env var is NOT set initially.")

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
        default=int(os.environ.get("LOCAL_RANK", "-1")),
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
    model_out = f"{args.root_dir}/models/finetuned_biomedbert_hf"
    best_model_out_dir = os.path.join(model_out, "best_model")
    final_model_out_dir = os.path.join(model_out, "final_model")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))

    if world_size > 1:
        torch.distributed.init_process_group(backend="nccl")

    logging.info(f"Rank {rank}: device count: {torch.cuda.device_count()}")
    logging.info(f"Rank {rank}: CUDA current device: {torch.cuda.current_device()}")
    logging.info(
        f"Rank {rank}: CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}"
    )

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

    for _, param in model.named_parameters():
        if not param.data.is_contiguous():
            param.data = param.data.contiguous()

    logging.info(
        f"Loaded BiomedBERT model and resized token embeddings to {len(tokenizer)}"
    )

    streaming_dataset = MLMTextDataset(
        file_path=abstracts,
        tokenizer=tokenizer,
        max_length=512,
    )
    total_abstracts = len(streaming_dataset)
    logging.info(f"Created MLMTextDataset with {total_abstracts} abstracts")

    num_epochs = 3
    train_batch_size = 128
    train_micro_batch_size_per_gpu = 32
    gradient_accumulation_steps = 2
    total_steps = _get_total_steps(
        world_size,
        num_epochs,
        train_batch_size,
        total_abstracts,
    )
    warmup_steps = int(0.1 * total_steps)
    logging.info(f"total steps: {total_steps}, warmup steps: {warmup_steps}")
    logging.info(
        f"Effective train batch size: {train_batch_size}, Micro batch size per GPU: {train_micro_batch_size_per_gpu}, Gradient Accumulation Steps: {gradient_accumulation_steps}"
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    best_train_loss_callback = SaveBestTrainingLossCallback(best_model_out_dir)

    training_args = TrainingArguments(
        output_dir=model_out,
        overwrite_output_dir=True,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=train_micro_batch_size_per_gpu,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_steps=100,
        save_strategy="epoch",
        save_total_limit=2,
        dataloader_num_workers=4,
        max_grad_norm=1.0,
        fp16=False,
        bf16=True,
        dataloader_pin_memory=True,
        report_to=None,
        dataloader_persistent_workers=True,
        gradient_checkpointing=False,
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=streaming_dataset,
        tokenizer=tokenizer,
        optimizers=(None, None),
        callbacks=[best_train_loss_callback],
    )

    logging.info("--- Starting Training with Hugging Face Trainer ---")
    trainer.train()
    logging.info("--- Training Finished ---")

    logging.info(f"Final model saved to {final_model_out_dir} (Trainer output_dir)")
    logging.info(
        f"Best model (if using evaluation and best_model selection) saved to {best_model_out_dir} (Trainer output_dir)"
    )


if __name__ == "__main__":
    main()
