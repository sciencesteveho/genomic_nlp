import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity


class PairwiseSimilarityScorer:
    def __init__(self, embeddings, gene_ids):
        self.embeddings = embeddings
        self.gene_ids = gene_ids
        self.gene_to_index = {gene: idx for idx, gene in enumerate(gene_ids)}

    def score_pair(self, gene1, gene2):
        idx1 = self.gene_to_index[gene1]
        idx2 = self.gene_to_index[gene2]
        return cosine_similarity(
            self.embeddings[idx1].reshape(1, -1), self.embeddings[idx2].reshape(1, -1)
        )[0][0]

    def evaluate_interactions(self, positive_pairs, all_possible_pairs=None):
        scores = []
        labels = []

        # Score positive pairs
        for gene1, gene2 in positive_pairs:
            scores.append(self.score_pair(gene1, gene2))
            labels.append(1)

        # If all_possible_pairs is not provided, we'll only use positive examples
        if all_possible_pairs is None:
            return np.mean(scores)

        # Score a sample of negative pairs
        num_negative = len(positive_pairs)
        negative_pairs = set(all_possible_pairs) - set(positive_pairs)
        sampled_negative = random.sample(
            negative_pairs, min(num_negative, len(negative_pairs))
        )

        for gene1, gene2 in sampled_negative:
            scores.append(self.score_pair(gene1, gene2))
            labels.append(0)

        return {
            "auc_roc": roc_auc_score(labels, scores),
            "avg_precision": average_precision_score(labels, scores),
        }


# Example usage
embeddings = np.random.rand(1000, 300)  # Replace with actual embeddings
gene_ids = [f"gene_{i}" for i in range(1000)]
positive_pairs = [
    ("gene_1", "gene_2"),
    ("gene_3", "gene_4"),
]  # Replace with actual positive pairs

scorer = PairwiseSimilarityScorer(embeddings, gene_ids)
results = scorer.evaluate_interactions(positive_pairs)
print(f"Evaluation results: {results}")
