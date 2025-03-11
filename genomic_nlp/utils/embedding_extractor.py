#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Extract embeddings from finetuned model (Vectorized Attention Pooling)."""

import pickle
import time
from typing import List, Optional, Set, Tuple

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
        super(AttentionPooling, self).__init__()
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
    output_file_avg = f"{output_dir}/averaged_embeddings.pkl"
    output_file_attn = f"{output_dir}/attention_embeddings.pkl"
    gene_token_file = "/ocean/projects/bio210019p/stevesho/genomic_nlp/embeddings/gene_tokens_nosyn.txt"
    disease_token_file = "/ocean/projects/bio210019p/stevesho/genomic_nlp/training_data/disease/disease_tokens_nosyn.txt"
    abstract_file_path = f"{data_dir}/processed_abstracts_finetune_combined.txt"

    batch_size = 32

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

    # load abstracts
    abstracts = load_abstracts(abstract_file_path)
    print("Loaded abstracts, tokenizer, and model.")

    # get token IDs for special tokens
    special_token_ids = tokenizer.convert_tokens_to_ids(list(all_entity_tokens))
    special_token_ids_tensor_batched = (
        torch.tensor(special_token_ids, dtype=torch.long, device=device)
        .unsqueeze(0)
        .repeat(batch_size, 1)
    )  # [batch_size, num_special_tokens] - batched special_token_ids

    avg_pooled_embedding_vectors = []
    attn_pooled_embedding_vectors = []

    # process abstracts in batches
    for i in tqdm(range(0, len(abstracts), batch_size), desc="Processing Batches"):
        batch_abstracts = abstracts[i : i + batch_size]
        inputs = tokenizer(
            batch_abstracts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512,
        ).to(device)

        with torch.no_grad():
            outputs: MaskedLMOutput = model(**inputs)
            all_token_embeddings: torch.Tensor = outputs.hidden_states[
                -1
            ]  # [batch_size, seq_len, hidden_dim]
            batch_input_ids = inputs["input_ids"]  # [batch_size, seq_len]
            batch_attention_mask = inputs["attention_mask"]  # [batch_size, seq_len]

            special_token_mask = torch.isin(
                batch_input_ids.unsqueeze(-1), special_token_ids_tensor_batched
            )  # [batch_size, seq_len, num_special_tokens]
            batch_special_token_indices_list = [
                torch.nonzero(special_token_mask[batch_index], as_tuple=False)[:, 0]
                for batch_index in range(batch_size)
            ]  # list of [num_special_tokens_in_abstract] tensors

            # pad special token indices to max length in batch for efficient batch gather
            max_special_tokens = max(
                (len(indices) for indices in batch_special_token_indices_list),
                default=0,
            )
            padded_special_token_indices = [
                F.pad(indices, (0, max_special_tokens - len(indices)), value=-1)
                for indices in batch_special_token_indices_list
            ]  # pad with -1
            batch_special_token_indices_tensor = torch.stack(
                padded_special_token_indices
            )  # [batch_size, max_special_tokens]

            # mask for valid indices (not padding -1)
            valid_indices_mask = (
                batch_special_token_indices_tensor != -1
            )  # [batch_size, max_special_tokens]

            # Vectorized gather and attention pooling
            batch_avg_pooled_embeddings = []
            batch_attn_pooled_embeddings = []

            for batch_index in range(batch_size):
                special_token_indices = batch_special_token_indices_tensor[
                    batch_index
                ]  # [max_special_tokens]
                valid_mask = valid_indices_mask[batch_index]  # [max_special_tokens]

                valid_special_token_indices = special_token_indices[
                    valid_mask
                ]  # [num_valid_special_tokens]
                if valid_special_token_indices.numel() == 0:  # Handle no special tokens
                    avg_pooled_embeddings = torch.zeros(
                        model.config.hidden_size, device=device
                    )
                    attn_pooled_embeddings = torch.zeros(
                        model.config.hidden_size, device=device
                    )
                else:
                    text_embeddings = all_token_embeddings[
                        batch_index
                    ]  # [seq_len, hidden_dim]
                    special_token_embeddings = text_embeddings[
                        valid_special_token_indices
                    ]  # [num_valid_special_tokens, hidden_dim] - Indexing

                    avg_pooled_embeddings = torch.mean(special_token_embeddings, dim=0)

                    attn_pooled_embeddings = pooling(
                        special_token_embeddings.unsqueeze(0),
                        torch.ones(
                            (1, special_token_embeddings.size(0)),
                            device=device,
                            dtype=torch.long,
                        ),
                    ).squeeze(0)

                batch_avg_pooled_embeddings.append(avg_pooled_embeddings.cpu().numpy())
                batch_attn_pooled_embeddings.append(
                    attn_pooled_embeddings.cpu().numpy()
                )

            avg_pooled_embedding_vectors.extend(batch_avg_pooled_embeddings)
            attn_pooled_embedding_vectors.extend(batch_attn_pooled_embeddings)

    print(
        "Average Pooled Embeddings Shape (first abstract):",
        (
            avg_pooled_embedding_vectors[0].shape
            if avg_pooled_embedding_vectors
            else "N/A"
        ),
    )
    print(
        "Attention Pooled Embeddings Shape (first abstract):",
        (
            attn_pooled_embedding_vectors[0].shape
            if attn_pooled_embedding_vectors
            else "N/A"
        ),
    )

    with open(output_file_avg, "wb") as f_avg:
        pickle.dump(avg_pooled_embedding_vectors, f_avg)
    with open(output_file_attn, "wb") as f_attn:
        pickle.dump(attn_pooled_embedding_vectors, f_attn)

    print(f"Average pooled embeddings saved to: {output_file_avg}")
    print(f"Attention pooled embeddings saved to: {output_file_attn}")


if __name__ == "__main__":
    main()
