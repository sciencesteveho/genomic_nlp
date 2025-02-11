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

ENTITY_MAPPINGS = {
    "il10": "IL10",
    "il6": "IL6",
    "il4": "IL4",
    "il2": "IL2",
    "tnf": "TNF",
    "vegfa": "VEGFA",
    "fgf2": "FGF2",
    "stat3": "STAT3",
    "stat6": "STAT6",
    "tp53": "TP53",
    "myc": "MYC",
    "max": "MAX",
    "fos": "FOS",
    "jun": "JUN",
    "pik3ca": "PIK3CA",
    "akt1": "AKT1",
    "raf1": "RAF1",
    "kras": "KRAS",
    "map2k1": "MAP2K1",
    "map2k2": "MAP2K2",
    "braf": "BRAF",
    "vegfr2": "VEGFR2",
    "fgfr1": "FGFR1",
    "fgfr2": "FGFR2",
    "egfr": "EGFR",
    "erbb2": "ERBB2",
    "tgfbr1": "TGFBR1",
    "tgfbr2": "TGFBR2",
    "smad2": "SMAD2",
    "smad3": "SMAD3",
    "brca1": "BRCA1",
    "bard1": "BARD1",
    "mdm2": "MDM2",
    "mdm4": "MDM4",
    "cdkn1a": "CDKN1A",
    "ctnnb1": "CTNNB1",
    "apc": "APC",
    "sufu": "SUFU",
    "gli1": "GLI1",
    "frs2": "FRS2",
    "ifng": "IFNG",
    "htt": "HTT",
    "park7": "PARK7",
    "hbb": "HBB",
    "f9": "F9",
    "f8": "F8",
    "cftr": "CFTR",
    "smn1": "SMN1",
    "tsc1": "TSC1",
    "nf1": "NF1",
    "bcr": "BCR",
    "pml": "PML",
    "rb1": "RB1",
    "vhl": "VHL",
    "col1a1": "COL1A1",
    "fbn1": "FBN1",
    "hexa": "HEXA",
    "hexb": "HEXB",
    "idua": "IDUA",
    "gba": "GBA",
    "huntington_disease": "Huntington's Disease",
    "parkinson_disease": "Parkinson's Disease",
    "hemophilia": "Hemophilia",
    "thalassemia": "Thalassemia",
    "cystic_fibrosis": "Cystic Fibrosis",
    "muscular_atrophy_spinal": "Spinal Muscular Atrophy",
    "tuberous_sclerosis": "Tuberous Sclerosis",
    "neurofibromatosis_1": "Neurofibromatosis Type 1",
    "cml": "Chronic Myeloid Leukemia",
    "leukemia_promyelocytic_acute": "Acute Promyelocytic Leukemia",
    "retinoblastoma": "Retinoblastoma",
    "von_hippel_lindau_disease": "Von Hippel-Lindau Disease",
    "osteogenesis_imperfecta": "Osteogenesis Imperfecta",
    "marfan_syndrome": "Marfan Syndrome",
    "tay_sachs_disease": "Tay-Sachs Disease",
    "sandhoff_disease": "Sandhoff Disease",
    "mucopolysaccharidosis_i": "Mucopolysaccharidosis Type I",
    "gaucher_disease": "Gaucher Disease",
    "hemophilia_a": "Hemophilia A",
    "hemophilia_b": "Hemophilia B",
}


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
    fig, ax = plt.subplots(constrained_layout=True, figsize=(3.5, 3.5))
    ax.tick_params(axis="both", which="major", width=0.5)

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
            label = ENTITY_MAPPINGS.get(word_label, word_label)
            texts.append(ax.text(w_coord[0], w_coord[1], label, zorder=4))

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
    ax.spines["bottom"].set_linewidth(0.5)
    ax.spines["left"].set_linewidth(0.5)
    ax.tick_params(axis="both", which="major", length=6)

    # set x and yticks to 0.5
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
        # 1) genes is to dna as rna is to transcripts
        # ("genes", "rna", "dna", "transcripts"),
        # 2) euchromatin is to open as closed is to heterochromatin
        ("euchromatin", "closed", "open", "heterochromatin"),
        # 3) exonic is to expressed as splice is to intronic
        # ("exonic", "splice", "expressed", "intronic"),
        # 4) acetylation is to activation as silencing is to deacetylation
        ("acetylation", "silencing", "activation", "deacetylation"),
        # 5) ligase is to join as unwinding is to helicase
        # ("ligase", "unwinding", "join", "helicase"),
        # 8) codon is to mrna as trna is to anticodon
        ("codon", "trna", "mrna", "anticodon"),
        # 9) phosphorylation is to kinase as phosphatase is to dephosphorylation
        # ("phosphorylation", "phosphatase", "kinase", "dephosphorylation"),
        # 10) ribosome is to translation as spliceosome is to splicing
        ("ribosome", "spliceosome", "translation", "splicing"),
        # 11) metaphase is to alignment as separation is to anaphase
        ("metaphase", "separation", "alignment", "anaphase"),
    ]

    gda_analogies = [
        # 1) htt is to huntington_disease as park7 is to parkinson_disease
        # ("htt", "park7", "huntington_disease", "parkinson_disease"),
        # 2) hbb is to hemophilia as f9 is to thalassemia
        # ("hbb", "f9", "hemophilia", "thalassemia"),
        # 3) cftr is to cystic_fibrosis as smn1 is to muscular_atrophy_spinal
        ("cftr", "smn1", "cystic_fibrosis", "muscular_atrophy_spinal"),
        # 4) tsc1 is to tuberous_sclerosis as nf1 is to neurofibromatosis_1
        ("tsc1", "nf1", "tuberous_sclerosis", "neurofibromatosis_1"),
        # 5) bcr is to cml as pml is to leukemia_promyelocytic_acute
        ("bcr", "pml", "cml", "leukemia_promyelocytic_acute"),
        # 6) retinoblastoma is to rb1 as vhl is to von_hippel_lindau_disease
        ("retinoblastoma", "vhl", "rb1", "von_hippel_lindau_disease"),
        # 7) col1a1 is to osteogenesis_imperfecta as fbn1 is to marfan_syndrome
        ("col1a1", "fbn1", "osteogenesis_imperfecta", "marfan_syndrome"),
        # 8) hexa is to tay_sachs_disease as hexb is to sandhoff_disease
        ("hexa", "hexb", "tay_sachs_disease", "sandhoff_disease"),
        # 9) idua is to mucopolysaccharidosis_i as gba is to gaucher_disease
        ("idua", "gba", "mucopolysaccharidosis_i", "gaucher_disease"),
        # 10) f8 is to hemophilia_a as f9 is to hemophilia_b
        ("f8", "f9", "hemophilia_a", "hemophilia_b"),
    ]

    interact_analogies = [
        # 1) il10 is to stat3 as il6 is to tnf
        ("il10", "il6", "stat3", "tnf"),
        # 2) tp53 is to cdkn1a as il6 is to tnf
        # ("tp53", "il6", "cdkn1a", "tnf"),
        # 3) smad2 is to tgfbr1 as tgfbr2 is to smad3  (downstream TGF-beta signaling)
        ("smad2", "tgfbr2", "tgfbr1", "smad3"),
        # 4) myc is to max as fos is to jun  (classic TF dimers)
        ("myc", "fos", "max", "jun"),
        # 5) il4 is to stat6 as il2 is to ifng
        # ("il4", "il2", "stat6", "ifng"),
        # 6) pik3ca is to akt1 as raf1 is to kras
        ("pik3ca", "raf1", "akt1", "kras"),
        # 7) vegfa is to vegfr2 as fgf2 is to fgfr1
        # ("vegfa", "fgfr1", "vegfr2", "fgf2"),
        # 8) brca1 is to bard1 as mdm4 is to mdm2
        # ("brca1", "mdm4", "bard1", "mdm2"),
        # 9) ctnnb1 is to apc as sufu is to gli1  (degradation complexes)
        ("ctnnb1", "sufu", "apc", "gli1"),
        # 10) egfr is to kras as frs2 is to fgfr1  (downstream adaptors)
        ("egfr", "frs2", "kras", "fgfr1"),
        # 11) raf1 is to braf as map2k1 is to map2k2  (related kinases -> MAP2Ks)
        ("raf1", "map2k1", "braf", "map2k2"),
        # 12) egfr is to erbb2 as fgfr2 is to fgfr1 (RTK dimer partners)
        ("egfr", "fgfr2", "erbb2", "fgfr1"),
    ]

    # # test analogies
    # for analogy in broad_analogies:
    #     test_analogy = (analogy[0], analogy[1], analogy[2])
    #     compute_analogy(model, test_analogy, top_n=1)

    # # euchromatin is to open as closed is to heterochromatin
    # model.wv.most_similar(positive=["euchromatic", "closed"], negative=["open"], topn=5)

    # # exonic -> expressed as intronic -> splice
    # model.wv.most_similar(positive=["exonic", "splice"], negative=["expressed"], topn=5)

    # # acetylation -> activation as deacetylation -> silencing
    # model.wv.most_similar(
    #     positive=["acetylation", "silencing"], negative=["activation"], topn=5
    # )

    # # ligase -> join as helicase -> unwinding
    # model.wv.most_similar(positive=["ligase", "unwinding"], negative=["join"], topn=5)

    # # codon - mRNA + tRNA ≈ anticodon
    # model.wv.most_similar(positive=["codon", "trna"], negative=["mrna"], topn=5)

    # # phosphorylation - kinase + phosphatase ≈ dephosphorylation
    # model.wv.most_similar(
    #     positive=["phosphorylation", "phosphatase"], negative=["kinase"], topn=5
    # )

    # # ribosome is to translation as spliceosome is to splicing
    # model.wv.most_similar(
    #     positive=["ribosome", "splicing"], negative=["translation"], topn=5
    # )

    # # metaphase is to alignment as separation is to anaphase
    # model.wv.most_similar(
    #     positive=["metaphase", "separation"], negative=["alignment"], topn=5
    # )

    # # interact
    # # il10 activates stat3 as il6 activates tnf
    # model.wv.most_similar(positive=["il10", "il6"], negative=["stat3"], topn=5)

    # # tp53 upregulates cdkn1a, and il6 can drive the expression of inflammatory mediators (e.g., tnf).
    # model.wv.most_similar(positive=["tp53", "il6"], negative=["cdkn1a"], topn=5)

    # # SMAD2/3 both interact downstream of TGF-beta receptor subunits
    # model.wv.most_similar(positive=["smad2", "tgfbr2"], negative=["tgfbr1"], topn=5)

    # # MYC-MAX and JUN-FOS are classic transcription factor dimers
    # model.wv.most_similar(positive=["myc", "fos"], negative=["max"], topn=5)

    # # il4 activates stat6 as il2 activates ifng
    # model.wv.most_similar(positive=["il4", "il2"], negative=["stat6"], topn=5)

    # # pik3ca activates akt1 as kras activates raf1
    # model.wv.most_similar(positive=["pik3ca", "raf1"], negative=["akt1"], topn=5)

    # # VEGFA binds VEGFR2 as FGF2 binds FGFR1
    # model.wv.most_similar(positive=["vegfa", "fgfr1"], negative=["vegfr2"], topn=5)

    # # BRCA1 interacts with BARD1 as MDM2 interacts with MDM4
    # model.wv.most_similar(positive=["brca1", "mdm4"], negative=["bard1"], topn=5)

    # # CTNNB1 is degraded by APC as GLI1 is degraded by SUFU
    # model.wv.most_similar(positive=["ctnnb1", "sufu"], negative=["apc"], topn=5)

    # # EGFR uses KRAS downstream, while fgfr1 uses FRS2 as an adaptor
    # model.wv.most_similar(positive=["egfr", "frs2"], negative=["kras"], topn=5)

    # # RAF1 and BRAF are related kinases that activate MAP2Ks
    # model.wv.most_similar(positive=["raf1", "map2k1"], negative=["braf"], topn=5)

    # # EGFR dimerizes with ERBB2 as FGFR1 dimerizes with FGFR2
    # model.wv.most_similar(positive=["egfr", "fgfr2"], negative=["erbb2"], topn=5)

    # # gda
    # # htt is to huntington_disease as park7 is to parkinson_disease
    # model.wv.most_similar(
    #     positive=["huntington_disease", "park7"], negative=["htt"], topn=5
    # )

    # # hbb is to hemophilia as f9 is to thalassemia
    # model.wv.most_similar(
    #     positive=["hbb", "hemophilia"], negative=["thalassemia"], topn=5
    # )

    # # cftr is to cystic_fibrosis as smn1 is to muscular_atrophy_spinal
    # model.wv.most_similar(
    #     positive=["cystic_fibrosis", "smn1"], negative=["cftr"], topn=5
    # )

    # # tsc1 is to tuberous_sclerosis as nf1 is to neurofibromatosis_1
    # model.wv.most_similar(
    #     positive=["tuberous_sclerosis", "nf1"], negative=["tsc1"], topn=5
    # )

    # # brc is to CML as pml is to 'leukemia_promyelocytic_acute'
    # model.wv.most_similar(
    #     positive=["leukemia_myelogenous_chronic_bcr_abl_positive", "pml"],
    #     negative=["bcr"],
    #     topn=5,
    # )

    # # retinoblastoma is to rb1 as vhl is to von_hippel_lindau_disease
    # model.wv.most_similar(positive=["retinoblastoma", "vhl"], negative=["rb1"], topn=5)

    # # col1a1 is to osteogenesis_imperfecta as fbn1 is to marfan_syndrome
    # model.wv.most_similar(
    #     positive=["osteogenesis_imperfecta", "fbn1"], negative=["col1a1"], topn=5
    # )

    # # hexa is to tay_sachs_disease as hexb is to sandhoff_disease
    # model.wv.most_similar(
    #     positive=["tay_sachs_disease", "hexb"], negative=["hexa"], topn=5
    # )

    # # idua is to mucopolysaccharidosis_i as gba is to gaucher_disease
    # model.wv.most_similar(
    #     positive=["gaucher_disease", "idua"], negative=["gba"], topn=5
    # )

    # # f8 is to hemophilia_a as f9 is to hemophilia_b
    # model.wv.most_similar(positive=["hemophilia_a", "f9"], negative=["f8"], topn=5)

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


if __name__ == "__main__":
    main()
