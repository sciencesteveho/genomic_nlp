#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# // TO-DO //
# - [ ] to-do

"""Mine abstracts from scopus API"""

import argparse
import os
import pandas as pd
import pickle

from pybliometrics.scopus import ScopusSearch

general_search_terms = [
    'ATAC-seq',
    'ChIA-PET',
    'DNA',
    'DNase',
    'GWAS',
    'Hi-C',
    'Pseudogene',
    'QTL',
    'RNA',
    'RNAi',
    'Repli-seq',
    'SNPs',
    'WGBS',
    'acetylation',
    'allele',
    'alleles',
    'biolog*',
    'ChIP-seq',
    'chromatid',
    'chromatin',
    'chromatin',
    'eCLIP',
    'eQTL',
    'epigenetics',
    'epigenome',
    'epigenomic',
    'epigenomics',
    'gene',
    'genes',
    'genetic',
    'genetics',
    'genome',
    'genomic',
    'genomics',
    'genotype',
    'haplotype',
    'helicase',
    'ligase',
    'lncRNA',
    'lncRNAs',
    'mRNA',
    'methylation',
    'methylome',
    'noncoding',
    'phenotype',
    'polymerase',
    'proteome',
    'retrotransposon',
    'sRNAs',
    'telomerase',
    'transcription',
    'transcriptional',
    'transcriptome',
    'transcriptome',
    'transcriptomic',
    'transcriptomics',
    'transposon',
    '{chromatin modification}',
    '{3D chromatin interactions}',
    '{DNA accessibility}',
    '{DNA condensation}',
    '{DNA damage}',
    '{DNA elements}',
    '{DNA polymerase}',
    '{DNA repair}',
    '{DNA replication}',
    '{DNA sequencing}',
    '{DNA supercoiling}',
    '{RNA decay}',
    '{RNA interference}',
    '{RNA modification}',
    '{RNA polymerase}',
    '{RNA processing}',
    '{RNA replication}',
    '{X-inactivation}',
    '{biological mechanisms}',
    '{biological mechanism}',
    '{chromatin remodeling}',
    '{chromatin states}',
    '{chromosome condensation}',
    '{chromosome segregation}',
    '{cis-acting}',
    '{cis-regulatory}',
    '{conserved sequence}',
    '{copy number variation}',
    '{distal enhancers}',
    '{dna damage}',
    '{dna recombination}',
    '{functional genomics}',
    '{gene expression}',
    '{gene function}',
    '{genetic mutation}',
    '{gene regulation}',
    '{gene regulatory networks}',
    '{gene regulatory network}',
    '{genetic mechanisms}',
    '{genetic mechanisms}',
    '{genome sequencing}',
    '{histone}',
    '{long non-coding RNA}',
    '{massively parallel reporter assays}',
    '{massively parallel sequencing}',
    '{messenger RNA}',
    '{microRNA}',
    '{molecular biology}',
    '{molecular cell biology}',
    '{noncoding elements}',
    '{next generation sequencing}',
    '{open chromatin}',
    '{origin of replication}',
    '{polyA RNA}',
    '{post-transcriptional modification}',
    '{post-translational modification}',
    '{protein activation}',
    '{protein coding}',
    '{protein decay}',
    '{protein modification}',
    '{protein translation}',
    '{protein-coding}',
    '{reporter activity}',
    '{regulatory elements}',
    '{regulatory element}',
    '{repeat DNA}',
    '{repetitive DNA}',
    '{RNA binding}',
    '{segmental duplication}',
    '{short hairpin RNA}',
    '{single nucleotide polymorphism}',
    '{small RNA}',
    '{small inhibitory RNA}',
    '{tRNA}',
    '{tandem repeat}',
    '{topologically associating domains}',
    '{trans-acting}',
    '{trans-regulatory}',
    '{transcription factors}',
    '{transcription factor}',
    '{transcriptional modification}',
    '{transcriptional regulation}',
    '{transcriptional regulation}',
    '{translational modification}',
    '{translational regulation}'
]

def make_directories(dir: str) -> None:
    try:
        os.makedirs(dir)
    except FileExistsError:
        pass


def main() -> None:
    """Download some abstracts!"""
    make_directories('abstracts')

    parser = argparse.ArgumentParser(description='Graph regression attempts')
    parser.add_argument('--start', type=int, default=1898,
                        help='random seed to use (default: 1898)')
    parser.add_argument('--end', type=int, default=2023,
                        help='random seed to use (default: 2023)')
    args = parser.parse_args()

    scopus_general = ScopusSearch(f"TITLE-ABS-KEY({' OR '.join(general_search_terms)}) AND (DOCTYPE(ar) OR DOCTYPE(le) OR DOCTYPE(re) AND (PUBYEAR > 2019) AND (PUBYEAR < 2023))", cursor=True, refresh=False, verbose=True, download=True)

    ### save the named tuples
    output = open(f'abstracts/abstract_retrieval_{args.start}_{args.end}.pkl', 'wb')
    try:
        pickle.dump(scopus_general.results(), output)
    except:
        pass
    finally:
        output.close()

    ### also save as dataframe
    df = pd.DataFrame(pd.DataFrame(scopus_general.results))
    df['description'].to_pickle(f'abstracts/df_abstracts_{args.start}_{args.end}.pkl')

if __name__ == '__main__':
    main()