#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Extract embeddings for gene/disease tokens (Vectorized Attention Pooling + Dict Output - v6 - MYPY Fixed)."""

import pickle
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm  # type: ignore
from transformers import BertForMaskedLM  # type: ignore
from transformers import BertTokenizerFast
from transformers.modeling_outputs import MaskedLMOutput  # type: ignore


class AttentionPooling(nn.Module):
    """Layer to implement a simple, single-headed attention-weighted pooling.
    Linear followed by softmax.
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.attention_layer = nn.Linear(hidden_dim, 1, bias=False)

    def forward(
        self,
        embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
        selected_token_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass for attention pooling.

        Args:
            embeddings: Tensor of shape [batch, seq_len, hidden_dim]
            attention_mask: Tensor of shape [batch, seq_len], with 1 for valid
            tokens, 0 otherwise.
            selected_token_indices: Optional tensor [batch, num_selected]
            specifying indices of tokens to pool.

        Returns:
            Pooled embedding tensor of shape [batch, hidden_dim].
        """
        if selected_token_indices is not None:
            batch_size, _, hidden_dim = embeddings.size()
            indices = selected_token_indices.unsqueeze(-1).expand(-1, -1, hidden_dim)
            embeddings = torch.gather(embeddings, dim=1, index=indices)
            attention_mask = torch.ones(
                embeddings.size()[:2], device=embeddings.device, dtype=torch.long
            )

        scores = self.attention_layer(embeddings).squeeze(-1)
        scores = scores.masked_fill(attention_mask == 0, float("-inf"))
        attn_weights = F.softmax(scores, dim=1).unsqueeze(-1)
        return torch.sum(attn_weights * embeddings, dim=1)


def load_tokens(filename: str) -> Set[str]:
    """Load gene tokens from a file."""
    with open(filename, "r") as f:
        return {line.strip().lower() for line in f}


def load_abstracts(filename: str) -> List[str]:
    """Load abstracts from a file, one abstract per line."""
    abstracts: List[str] = []
    with open(filename, "r") as f:
        abstracts.extend(line.strip() for line in f)
    return abstracts


def main() -> None:
    """Main function to extract embeddings using vectorized attention pooling."""
    root_dir = "/ocean/projects/bio210019p/stevesho/genomic_nlp"
    data_dir = f"{root_dir}/data/combined"
    model_path = f"{root_dir}/models/finetuned_biomedbert_hf/best_model"
    tokenizer_path = f"{root_dir}/models/finetuned_biomedbert_hf/gene_tokenizer"
    output_dir = f"{root_dir}/embeddings"
    output_file_avg = f"{output_dir}/averaged_gene_disease_embeddings_v6.pkl"  # Updated filename to v6
    output_file_attn = f"{output_dir}/attention_gene_disease_embeddings_v6.pkl"  # Updated filename to v6
    gene_token_file = "/ocean/projects/bio210019p/stevesho/genomic_nlp/embeddings/gene_tokens_nosyn.txt"
    disease_token_file = "/ocean/projects/bio210019p/stevesho/genomic_nlp/training_data/disease/disease_tokens_nosyn.txt"
    abstract_file_path = f"{data_dir}/processed_abstracts_finetune_combined.txt"

    batch_size = 32
    num_abstracts_check = 1000

    # load tokens
    gene_tokens = load_tokens(gene_token_file)
    disease_tokens = load_tokens(disease_token_file)
    all_entity_tokens = gene_tokens.union(disease_tokens)

    # initialize tokenizer and model.
    tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path)
    model = BertForMaskedLM.from_pretrained(
        model_path, output_hidden_states=True
    ).eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    pooling = AttentionPooling(hidden_dim=model.config.hidden_size).to(device)

    print("Model hidden size:", model.config.hidden_size)
    print(f"Number of gene/disease tokens: {len(all_entity_tokens)}")

    # load abstracts (for initial check - first 1000 abstracts)
    abstracts_check = load_abstracts(abstract_file_path)[:num_abstracts_check]
    print(
        f"Loaded {len(abstracts_check)} abstracts for initial check, tokenizer, and model."
    )

    special_token_ids = tokenizer.convert_tokens_to_ids(list(all_entity_tokens))

    avg_token_embeddings_dict_check: Dict[str, List[float]] = {}
    attn_token_embeddings_dict_check: Dict[str, List[float]] = {}

    verification_passed = True

    # process abstracts in batches (for initial check - first 1000 abstracts)
    for i in tqdm(
        range(0, len(abstracts_check), batch_size),
        desc="Processing Batches (Check Run)",
    ):
        batch_abstracts = abstracts_check[i : i + batch_size]
        actual_batch_size = len(
            batch_abstracts
        )  # get actual number of abstracts in this batch

        # create tensor with actual batch size
        special_token_ids_tensor_batched = (
            torch.tensor(special_token_ids, dtype=torch.long, device=device)
            .unsqueeze(0)
            .repeat(actual_batch_size, 1)
        )

        inputs = tokenizer(
            batch_abstracts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512,
        ).to(device)

        with torch.no_grad():
            outputs_check: MaskedLMOutput = model(**inputs)
            all_token_embeddings_check: torch.Tensor = outputs_check.hidden_states[-1]
            batch_input_ids = inputs["input_ids"]
            batch_attention_mask = inputs["attention_mask"]

            special_token_mask = torch.isin(
                batch_input_ids.unsqueeze(-1), special_token_ids_tensor_batched
            )
            batch_special_token_indices_list = [
                torch.nonzero(special_token_mask[batch_index], as_tuple=False)[:, 0]
                for batch_index in range(actual_batch_size)
            ]

            max_special_tokens = max(
                (len(indices) for indices in batch_special_token_indices_list) or [0]
            )
            padded_special_token_indices = [
                F.pad(indices, (0, max_special_tokens - len(indices)), value=-1)
                for indices in batch_special_token_indices_list
            ]
            batch_special_token_indices_tensor = torch.stack(
                padded_special_token_indices
            )

            valid_indices_mask = batch_special_token_indices_tensor != -1

            for batch_index in range(actual_batch_size):
                special_token_indices = batch_special_token_indices_tensor[batch_index]
                valid_mask = valid_indices_mask[batch_index]
                valid_special_token_indices = special_token_indices[valid_mask]

                if valid_special_token_indices.numel() > 0:
                    text_embeddings = all_token_embeddings_check[
                        batch_index
                    ]  # Use renamed variable
                    special_token_embeddings = text_embeddings[
                        valid_special_token_indices
                    ]

                    avg_pooled_embeddings = torch.mean(special_token_embeddings, dim=0)
                    attn_pooled_embeddings = pooling(
                        special_token_embeddings.unsqueeze(0),
                        torch.ones(
                            (1, special_token_embeddings.size(0)),
                            device=device,
                            dtype=torch.long,
                        ),
                    ).squeeze(0)

                    input_ids_current_abstract = inputs["input_ids"][batch_index]
                    abstract_special_token_ids = input_ids_current_abstract[
                        valid_special_token_indices
                    ]
                    abstract_special_tokens = tokenizer.convert_ids_to_tokens(
                        abstract_special_token_ids
                    )

                    # populate dictionaries with token-embedding pairs (check run dictionaries)
                    for token_str in abstract_special_tokens:
                        token_str_lower = token_str.lower()
                        if token_str_lower in all_entity_tokens:
                            avg_token_embeddings_dict_check[token_str_lower] = (
                                avg_pooled_embeddings.cpu().numpy().tolist()
                            )
                            attn_token_embeddings_dict_check[token_str_lower] = (
                                attn_pooled_embeddings.cpu().numpy().tolist()
                            )

    print(
        f"Number of unique gene/disease tokens with average embeddings (check run): {len(avg_token_embeddings_dict_check)}"
    )
    print(
        f"Number of unique gene/disease tokens with attention embeddings (check run): {len(attn_token_embeddings_dict_check)}"
    )

    print("\n--- VERIFICATION RUN COMPLETE. INSPECTING OUTPUT DICTIONARIES ---")
    if len(avg_token_embeddings_dict_check) > 0:
        print(
            "Verification check passed. Proceeding to full embedding extraction on entire dataset."
        )
        print("\n--- STARTING FULL EMBEDDING EXTRACTION ON ENTIRE DATASET ---")

        # load abstracts (full dataset)
        abstracts_full = load_abstracts(abstract_file_path)
        print(f"Loaded full dataset of {len(abstracts_full)} abstracts.")

        avg_token_embeddings_dict: Dict[str, List[float]] = {}
        attn_token_embeddings_dict: Dict[str, List[float]] = {}

        # process abstracts in batches (full dataset)
        for i in tqdm(
            range(0, len(abstracts_full), batch_size),
            desc="Processing Batches (Full Dataset)",
        ):
            batch_abstracts = abstracts_full[i : i + batch_size]
            actual_batch_size = len(
                batch_abstracts
            )  # get actual number of abstracts in this batch

            # create tensor with actual batch size
            special_token_ids_tensor_batched = (
                torch.tensor(special_token_ids, dtype=torch.long, device=device)
                .unsqueeze(0)
                .repeat(actual_batch_size, 1)
            )

            inputs = tokenizer(
                batch_abstracts,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512,
            ).to(device)

            with torch.no_grad():
                outputs: MaskedLMOutput = model(**inputs)
                all_token_embeddings: torch.Tensor = outputs.hidden_states[-1]
                batch_input_ids = inputs["input_ids"]
                batch_attention_mask = inputs["attention_mask"]

                special_token_mask = torch.isin(
                    batch_input_ids.unsqueeze(-1), special_token_ids_tensor_batched
                )
                batch_special_token_indices_list = [
                    torch.nonzero(special_token_mask[batch_index], as_tuple=False)[:, 0]
                    for batch_index in range(actual_batch_size)  # Use actual batch size
                ]

                max_special_tokens = max(
                    (len(indices) for indices in batch_special_token_indices_list)
                    or [0]
                )
                padded_special_token_indices = [
                    F.pad(indices, (0, max_special_tokens - len(indices)), value=-1)
                    for indices in batch_special_token_indices_list
                ]
                batch_special_token_indices_tensor = torch.stack(
                    padded_special_token_indices
                )

                valid_indices_mask = batch_special_token_indices_tensor != -1

                for batch_index in range(actual_batch_size):  # use actual batch size
                    special_token_indices = batch_special_token_indices_tensor[
                        batch_index
                    ]
                    valid_mask = valid_indices_mask[batch_index]
                    valid_special_token_indices = special_token_indices[valid_mask]

                    if valid_special_token_indices.numel() > 0:
                        text_embeddings = all_token_embeddings[batch_index]
                        special_token_embeddings = text_embeddings[
                            valid_special_token_indices
                        ]

                        avg_pooled_embeddings = torch.mean(
                            special_token_embeddings, dim=0
                        )
                        attn_pooled_embeddings = pooling(
                            special_token_embeddings.unsqueeze(0),
                            torch.ones(
                                (1, special_token_embeddings.size(0)),
                                device=device,
                                dtype=torch.long,
                            ),
                        ).squeeze(0)

                        input_ids_current_abstract = inputs["input_ids"][batch_index]
                        abstract_special_token_ids = input_ids_current_abstract[
                            valid_special_token_indices
                        ]
                        abstract_special_tokens = tokenizer.convert_ids_to_tokens(
                            abstract_special_token_ids
                        )

                        # populate dictionaries with token-embedding pairs
                        for token_str in abstract_special_tokens:
                            token_str_lower = token_str.lower()
                            if token_str_lower in all_entity_tokens:
                                avg_token_embeddings_dict[token_str_lower] = (
                                    avg_pooled_embeddings.cpu().numpy().tolist()
                                )
                                attn_token_embeddings_dict[token_str_lower] = (
                                    attn_pooled_embeddings.cpu().numpy().tolist()
                                )

        print(
            f"Number of unique gene/disease tokens with average embeddings (full dataset): {len(avg_token_embeddings_dict)}"
        )
        print(
            f"Number of unique gene/disease tokens with attention embeddings (full dataset): {len(attn_token_embeddings_dict)}"
        )

        with open(output_file_avg, "wb") as f_avg:
            pickle.dump(avg_token_embeddings_dict, f_avg)
        with open(output_file_attn, "wb") as f_attn:
            pickle.dump(attn_token_embeddings_dict, f_attn)

        print(f"Average pooled embeddings (full dataset) saved to: {output_file_avg}")
        print(
            f"Attention pooled embeddings (full dataset) saved to: {output_file_attn}"
        )

    else:
        print(
            "Verification check failed. Please inspect *_check_v5.pkl files and code before running on full dataset."
        )


if __name__ == "__main__":
    main()
