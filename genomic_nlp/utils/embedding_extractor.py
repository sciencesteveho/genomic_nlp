#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Extract embeddings from finetuned model."""


import pickle
from typing import List, Optional, Set

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm  # type: ignore
from transformers import BertForMaskedLM  # type: ignore
from transformers import BertTokenizerFast  # type: ignore

# from genomic_nlp.embedding_utils.embedding_extractors import TokenizedDataset


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
        """
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
            _, hidden_dim = embeddings.size(0), embeddings.size(2)
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
    """Main function to extract embeddings. We will extract static embeddings
    via average pooling and attention pooling.
    """
    # paths
    root_dir = "/ocean/projects/bio210019p/stevesho/genomic_nlp"
    data_dir = f"{root_dir}/data/combined"
    model_path = f"{root_dir}/models/finetuned_biomedbert"
    tokenizer_path = f"{model_path}/gene_tokenizer"
    output_dir = f"{root_dir}/embeddings"
    output_file_avg = f"{output_dir}/averaged_embeddings.pkl"
    output_file_attn = f"{output_dir}/attention_embeddings.pkl"
    gene_token_file = "/ocean/projects/bio210019p/stevesho/genomic_nlp/embeddings/gene_tokens_nosyn.txt"
    disease_token_file = "/ocean/projects/bio210019p/stevesho/genomic_nlp/training_data/disease/disease_tokens_nosyn.txt"
    abstract_file_path = f"{data_dir}/processed_abstracts_finetune_combined.txt"

    # load tokens
    gene_tokens = load_tokens(gene_token_file)
    disease_tokens = load_tokens(disease_token_file)
    all_entity_tokens = gene_tokens.union(disease_tokens)

    # initialize tokenizer and model.
    tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path)
    model = BertForMaskedLM.from_pretrained(model_path).eval()

    # instantiate attention pooling layer
    pooling = AttentionPooling(hidden_dim=model.config.hidden_size)

    # load abstracts
    abstracts = load_abstracts(abstract_file_path)
    print("Loaded abstracts, tokenizer, and model.")

    # tokenize the texts
    inputs = tokenizer(abstracts, padding=True, truncation=True, return_tensors="pt")

    # get token IDs for special tokens
    special_token_ids = tokenizer.convert_tokens_to_ids(list(all_entity_tokens))

    avg_pooled_embedding_vectors = []
    attn_pooled_embedding_vectors = []

    # run forward pass to get embeddings
    with torch.no_grad():
        outputs = model(**inputs)
        all_token_embeddings = outputs.last_hidden_state

        for i in tqdm(range(len(abstracts)), desc="Processing Abstracts"):
            input_ids = inputs["input_ids"][i]
            text_embeddings = all_token_embeddings[i]

            special_token_indices_for_text = []
            for token_id in special_token_ids:
                indices = (input_ids == token_id).nonzero(as_tuple=True)[0].tolist()
                special_token_indices_for_text.extend(indices)

            if not special_token_indices_for_text:
                print(f"No special tokens found in abstract (index {i})")
                avg_pooled_embeddings = torch.zeros(model.config.hidden_size)
                attn_pooled_embeddings = torch.zeros(model.config.hidden_size)
            else:
                special_token_indices_tensor = torch.tensor(
                    special_token_indices_for_text, dtype=torch.long
                ).unsqueeze(0)

                _, hidden_dim = text_embeddings.size()
                indices_expanded = special_token_indices_tensor.unsqueeze(-1).expand(
                    -1, -1, hidden_dim
                )
                special_token_embeddings = torch.gather(
                    text_embeddings.unsqueeze(0), dim=1, index=indices_expanded
                ).squeeze(0)

                avg_pooled_embeddings = torch.mean(special_token_embeddings, dim=0)

                attn_pooled_embeddings = pooling(
                    special_token_embeddings.unsqueeze(0),
                    torch.ones(
                        (1, special_token_embeddings.size(0)),
                        device=text_embeddings.device,
                        dtype=torch.long,
                    ),
                ).squeeze(0)

            avg_pooled_embedding_vectors.append(avg_pooled_embeddings.cpu().numpy())
            attn_pooled_embedding_vectors.append(attn_pooled_embeddings.cpu().numpy())

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
