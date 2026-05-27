import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


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
    df.columns = ['gene'] + [float(c) for c in time_cols]
    df = df.set_index('gene')
    df = df.groupby(df.index).mean()
    return df


def expr_log(df) -> pd.DataFrame:
    expr_log = np.log2(df + 1)
    return expr_log


def gene_plot(expr_log_df, gene):
    if gene not in expr_log_df.index:
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
        plt.close()


def cosine_model(t, A, phi, C):
    return A * np.cos(2 * np.pi * t / 24 - phi) + C


def fit_gene(times, values):
    try:
        popt, _ = curve_fit(
            cosine_model,
            times,
            values,
            p0=[np.std(values), 0, np.mean(values)],
            maxfev=10000
        )
        fitted = cosine_model(times, *popt)
        ss_res = np.sum((values - fitted) ** 2)
        ss_tot = np.sum((values - np.mean(values)) ** 2)

        if ss_tot == 0:
            return None

        r2 = 1 - ss_res / ss_tot

        return {
            "amplitude": float(popt[0]),
            "phase": float(popt[1]),
            "mean": float(popt[2]),
            "r2": float(r2)
        }

    except Exception as e:
        print("ERROR:", e)
        return None


def screen_rhythmic_genes(expr_log_df):
    times = np.asarray(expr_log_df.columns, dtype=np.float64)
    results = []

    for gene in expr_log_df.index:
        values = np.asarray(expr_log_df.loc[gene], dtype=float).flatten()
        fit = fit_gene(times, values)
        if fit:
            results.append({
                "gene": gene,
                **fit
            })
    return pd.DataFrame(results)


def select_rhythmic(results, r2_threshold=0.35, amp_threshold=0.15):
    return results[
        (results["r2"] > r2_threshold) &
        (np.abs(results["amplitude"]) > amp_threshold)
    ]


def rhythmic_heatmap(expr_log_df, rhythmic_df):
    rhythmic_sorted = rhythmic_df.sort_values("phase")
    genes = rhythmic_sorted["gene"]
    data = expr_log_df.loc[genes]
    data = data.sub(data.mean(axis=1), axis=0)
    data = data.div(data.std(axis=1), axis=0)
    plt.figure(figsize=(12, 20))
    plt.imshow(data, aspect='auto')
    plt.colorbar(label='log2(FPKM + 1)')
    plt.xlabel("CT")
    plt.ylabel("Genes")
    plt.title("Rhythmic genes")
    plt.savefig("rhythmic_heatmap.png")
    plt.close()


df = parse_raw_data('GSE74439_DMSO_genes.fpkm_table.txt', 'GSE74439_DMSO_genes.attr_table.txt')
expr_log_df = expr_log(df)
for gene in ['CLOCK', 'ARNTL', 'PER1', 'PER2', 'CRY1', 'CRY2']:
    gene_plot(expr_log_df, gene)
expr_log_df.to_csv('expr_log.tsv', sep='\t')
results = screen_rhythmic_genes(expr_log_df)
rhythmic = select_rhythmic(results)
rhythmic.to_csv("rhythmic_genes.tsv", sep='\t', index=False)
rhythmic_heatmap(expr_log_df, rhythmic)

# control_genes = ["CLOCK", "ARNTL", "PER1", "PER2", "CRY1", "CRY2"]
# for gene in control_genes:
#     if gene in rhythmic["gene"].values:
#         row = rhythmic[rhythmic["gene"] == gene]
#         print(gene, row[["amplitude", "r2"]].to_dict('records')[0])
#     else:
#         print(f"{gene} not rhythmic")