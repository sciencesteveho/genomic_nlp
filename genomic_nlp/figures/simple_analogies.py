#! /usr/bin/env python
# -*- coding: utf-8 -*-


"""Code used to test and visualize simple analogies for figures."""


from typing import List, Tuple

from adjustText import adjust_text  # type: ignore
from gensim.models import Word2Vec  # type: ignore
from matplotlib import colors as mcolors  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np
from sklearn.decomposition import PCA  # type: ignore

from genomic_nlp.visualization import set_matplotlib_publication_parameters


def find_partial_matches(
    partial_word: str,
    model: Word2Vec,
    top_n: int = 50,
) -> None:
    """Find the top n word vectors that contain the partial word and print them
    out, sorted by similarity.
    """
    matches = [word for word in model.wv.key_to_index if partial_word in word]
    matched = sorted(
        matches,
        key=lambda w: model.wv.get_vector(w).dot(
            model.wv.get_vector(partial_word, norm=True)
        ),
        reverse=True,
    )[:top_n]
    print(matched)


def lighten_color(color: str, amount: float = 0.0001) -> np.ndarray:
    """Lightens the given color by mixing it with white.

    Parameters:
        color: Matplotlib color string (e.g., 'C0', '#ff0000').
        amount: 0 and 1, where higher values make the color lighter.
    """
    color_array = np.array(mcolors.to_rgb(color))
    return np.clip((1 - amount) + amount * color_array, 0, 1)


def compute_analogy(
    model: Word2Vec,
    analogy: Tuple[str, str, str],
    top_n: int = 5,
) -> None:
    """Given an analogy, use cosine similarity to compute the top n words that
    best complete the analogy.
    """
    result = model.wv.most_similar(
        positive=[analogy[0], analogy[1]], negative=[analogy[2]], topn=top_n
    )
    print(f"Analogy: {analogy[0]} is to {analogy[2]} as {analogy[1]} is to ?")
    for word, score in result:
        print(f"{word} ({score:.2f})\n")


def collect_vectors(model: Word2Vec, words: List[str]) -> np.ndarray:
    """Fetch vectors for each word in the list from the model. If a word is not
    present in the model vocabulary, ignore it.
    """
    in_vocab = [w for w in words if w in model.wv]
    if len(in_vocab) < len(words):
        missing = set(words) - set(in_vocab)
        print(
            f"Warning: The following words are not in the model vocabulary: {missing}"
        )
    return np.array([model.wv[word] for word in in_vocab])


def project_vectors_2d(vectors: np.ndarray) -> np.ndarray:
    """Project the given word vectors down to 2D with PCA."""
    pca = PCA(n_components=2)
    return pca.fit_transform(vectors)


def plot_analogies_2d(
    model: Word2Vec,
    analogies: List[Tuple[str, str, str, str]],
    shapes: List[str],
    output_filename: str = "word_analogies_2d.png",
) -> None:
    """Plot the words involved in each analogy in 2D space."""
    # collect all unique words from the analogies
    words = list({word for analogy in analogies for word in analogy})
    vectors = collect_vectors(model, words)

    if len(vectors) == 0:
        print("No vectors to plot (all words missing in model). Exiting plot function.")
        return

    # map each word to its 2D coordinates
    vectors_2d = project_vectors_2d(vectors)
    in_vocab_words = [w for w in words if w in model.wv]
    word_to_2d = dict(zip(in_vocab_words, vectors_2d))
    fig, ax = plt.subplots(constrained_layout=True, figsize=(4, 4))

    base_marker_size = 25
    face_light_factor = 0.95
    texts = []

    # plot each analogy
    for i, (w1, w2, w3, w4) in enumerate(analogies):
        # skip analogies with missing words
        if any(w not in word_to_2d for w in (w1, w2, w3, w4)):
            print(f"Skipping analogy {w1}, {w2}, {w3}, {w4} - not all in vocabulary.")
            continue

        v1 = word_to_2d[w1]
        v2 = word_to_2d[w2]
        v3 = word_to_2d[w3]
        v4 = word_to_2d[w4]

        color = f"C{i}"  # cycle through matplotlib default colors
        light_color = lighten_color(color, amount=face_light_factor)
        shape = shapes[i % len(shapes)]  # cycle through shapes if list is shorter

        for w_coord, word_label in [(v1, w1), (v2, w2), (v3, w3), (v4, w4)]:
            ax.scatter(
                w_coord[0],
                w_coord[1],
                facecolors=light_color,
                edgecolors=color,
                marker=shape,
                s=base_marker_size,
                linewidths=0.5,
                zorder=3,
                alpha=0.45,
            )
            texts.append(ax.text(w_coord[0], w_coord[1], word_label, zorder=4))

        # draw arrows between pairs illustrating the analogy
        for start, end in [(v1, v3), (v2, v4)]:
            ax.arrow(
                start[0],
                start[1],
                end[0] - start[0],
                end[1] - start[1],
                color=color,
                width=0.0003,
                head_width=0.02,
                head_length=0.02,
                length_includes_head=True,
                zorder=2,
                alpha=0.70,
            )

    # minimize text overlap
    adjust_text(
        texts,
        ax=ax,
        expand_text=(1, 1),
        expand_points=(1, 1),
        force_text=1.0,
        force_points=1.0,
        force_explode=1.0,
        only_move={"texts": "xy", "points": "xy"},
    )

    # ax.set_title("Word Analogies in 2D Space")
    ax.set_xlabel("First principal component")
    ax.set_ylabel("Second principal component")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="major", length=6)
    ax.set_xticks(np.linspace(-3, 3, 7))
    ax.set_yticks(np.linspace(-3, 3, 7))
    ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig(output_filename, dpi=450, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Main function for testing simple analogies."""
    set_matplotlib_publication_parameters()

    # load full models
    model = Word2Vec.load("word2vec_300_dimensions_2023.model")
    # model = Word2Vec.load("word2vec_300_dimensions_2003.model")

    broad_analogies = [
        # broader domain knowledge
        ("genes", "rna", "dna", "transcripts"),
        ("euchromatin", "closed", "open", "heterochromatin"),
        ("exonic", "splice", "expressed", "intronic"),
        ("acetylation", "silencing", "activation", "deacetylation"),
        ("ligase", "unwinding", "join", "helicase"),
        # ("h3k4me1", "silencer", "enhancer", "h3k9me3"),
        # ("h3k4me1", "promoter", "enhancer", "histone_h3_trimethyl_lys4"),
    ]

    interact_analogies = [
        # specific gene/gene protein/protein interactions
    ]

    gda_analogies = [
        # gene-disease relationships
        ("app", "huntington", "alzheimers", "htt"),
        ("capn3", "muscular_dystrophy_duchenne", "lgmd", "dmd"),
        ("smn1", "mapt", "spinal_muscular_atrophy", "tauopathies"),
        ("snca", "hemiplegic_migraine", "synucleinopathies", "cacna1a"),
        ("park7", "huntington", "htt", "parkinson"),
        ("abcd1", "mucopolysaccharidosis", "adrenoleukodystrophy", "idua"),
        ("idua", "tay_sachs_disease", "mucopolysaccharidosis", "hexosaminidase"),
        ("tsc1", "neurofibromatosis", "tuberous_sclerosis", "nf1"),
    ]

    # test analogies
    for analogy in broad_analogies:
        test_analogy = (analogy[0], analogy[1], analogy[2])
        compute_analogy(model, test_analogy, top_n=1)

    # broad
    # codon - mRNA + tRNA ≈ anticodon
    model.wv.most_similar(positive=["codon", "tRNA"], negative=["mRNA"], topn=5)

    # phosphorylation - kinase + phosphatase ≈ dephosphorylation
    model.wv.most_similar(
        positive=["phosphorylation", "phosphatase"], negative=["kinase"], topn=5
    )

    # interact
    # il10 activates stat3 as il6 activates tnf
    model.wv.most_similar(positive=["il10", "il6"], negative=["stat3"], topn=5)

    # atm activates chk1 as atr activates chk2
    model.wv.most_similar(positive=["atm", "atr"], negative=["chek1"], topn=5)

    # p53 -> cdkn1a (p21); RB1 -> E2F1
    model.wv.most_similar(positive=["tp53", "e2f1"], negative=["cdkn1a"], topn=5)

    # SMAD2/3 both interact downstream of TGF-beta receptor subunits
    model.wv.most_similar(positive=["smad2", "tgfbr2"], negative=["tgfbr1"], topn=5)

    # MYC-MAX and JUN-FOS are classic transcription factor dimers
    model.wv.most_similar(positive=["myc", "fos"], negative=["max"], topn=5)

    # gda
    # htt is to huntington_disease as park7 is to parkinson
    model.wv.most_similar(
        positive=["huntington_disease", "park7"], negative=["htt"], topn=5
    )

    # hbb is to hemophilia as f9 is to thalassemia
    model.wv.most_similar(
        positive=["hbb", "hemophilia"], negative=["thalassemia"], topn=5
    )

    # fragile_x_syndrome is to fmr1 as rett is to mecp2
    model.wv.most_similar(
        positive=["mecp2", "fragile_x_syndrome"], negative=["rett"], topn=5
    )

    model.wv.most_similar(
        positive=["brca1", "colon_cancer"], negative=["carcinoma_ductal_breast"], topn=5
    )

    plot_analogies_2d(
        model=model,
        analogies=gda_analogies,
        shapes=["o", "s", "^", "D", "p", "*"],
        output_filename="gda_analogies_2d.png",
    )

    # plot analogies
    plot_analogies_2d(
        model=model,
        analogies=broad_analogies,
        shapes=["o", "s", "^", "D", "p", "*"],
        output_filename="broad_analogies_2d.png",
    )

    plot_analogies_2d(
        model=model,
        analogies=interact_analogies,
        shapes=["o", "s", "^", "D", "p", "*"],
        output_filename="interact_analogies_2d.png",
    )
