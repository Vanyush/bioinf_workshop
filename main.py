import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt


def parse_raw_data(expr_table, attr_table) -> pd.DataFrame:
    df = pd.read_csv(expr_table, sep="\t")
    names = pd.read_csv(attr_table, sep='\t')['gene_short_name']
    time_samples = df.columns[1:]
    labels = ['gene']

    for s in time_samples:
        match = re.search(r'CT(\d+)', s)
        labels.append(int(match.group(1)))

    df['tracking_id'] = names
    df.columns = labels
    time_cols = sorted([c for c in df.columns if isinstance(c, int)])
    df = df[['gene'] + time_cols]
    df = df.set_index('gene')
    return df


def expr_log(df) -> pd.DataFrame:
    expr_log = np.log2(df + 1)
    return expr_log


def gene_plot(expr_log_df, gene):
    if gene not in df.index:
        print(f'Gene {gene} is not provided by dataset')
    else:
        row = expr_log_df.loc[gene]
        plt.figure()
        plt.plot(row.index, row.values, marker='o')
        plt.title(gene)
        plt.xlabel("CT (circadian time)")
        plt.ylabel("log2(FPKM + 1)")
        plt.grid(True)
        plt.savefig(f'{gene}_expr_plot.png')


df = parse_raw_data('GSE74439_DMSO_genes.fpkm_table.txt', 'GSE74439_DMSO_genes.attr_table.txt')
expr_log = expr_log(df)
for gene in ['CLOCK', 'BMAL1', 'PER1', 'PER2', 'CRY1', 'CRY2']:
    gene_plot(expr_log, gene)
expr_log.to_csv('expr_log.tsv', sep='\t')