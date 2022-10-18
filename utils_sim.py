from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mantel
import glob


def plot_cosine_matrix(matrix, name_f_plot, title_, yaxis, xaxis, output_dir):

    fig, ax = plt.subplots()

    matrix = np.tril(matrix).astype('float')
    matrix[matrix == 0] = 'nan'  # or use np.nan

    ax.matshow(matrix, cmap=plt.cm.Blues)

    for i in range(matrix.shape[1]):
        for j in range(matrix.shape[0]):
            c = matrix[j, i]
            if str(c)!="nan":
                ax.text(i, j, str(round(c, 2)), va='center', ha='center')  #parties[i], parties[j], str(float(c)),

    plt.title(title_)
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticklabels([i for i in xaxis])
    ax.set_yticklabels([i for i in yaxis])

    plt.show()
    Path(f"{output_dir}/plots").mkdir(exist_ok=True, parents=True)
    plt.savefig(f"{output_dir}/plots/{name_f_plot}.jpeg")
    plt.close()

def save_cosine_csv(df_matrix, name_f, output_dir):
    Path(f"{output_dir}/cosine_scores").mkdir(exist_ok=True, parents=True)
    df_matrix.to_csv(f"{output_dir}/cosine_scores/"+str(name_f)+".csv")

def save_cosine_between_sentences(cosine_scores, sentences, parties, name_f, output_dir):
    # Find the pairs with the highest cosine similarity scores
    Path(f"{output_dir}/cos_sentences").mkdir(exist_ok=True, parents=True)
    f_out = open(f"{output_dir}/cos_sentences/{name_f}.txt", "w")
    pairs = []
    for i in range(len(cosine_scores) - 1):
        for j in range(i + 1, len(cosine_scores)):
            pairs.append({'index': [i, j], 'score': cosine_scores[i][j]})

    # Sort scores in decreasing order
    pairs = sorted(pairs, key=lambda x: x['score'], reverse=True)
    for pair in pairs:
        i, j = pair['index']
        f_out.write(f"Sent1 of party {parties[i].upper()}= {sentences[i]}\nSent2 of party {parties[j].upper()}= {sentences[j]}\nScore= {pair['score']}\n\n")
    f_out.close()

def compute_kernel_bias(vecs, k=None):
    """计算kernel和bias
    最后的变换：y = (x + bias).dot(kernel)
    Code taken from: https://github.com/bojone/BERT-whitening
    """
    mu = vecs.mean(axis=0, keepdims=True)
    cov = np.cov(vecs.T)
    u, s, vh = np.linalg.svd(cov)
    W = np.dot(u, np.diag(1 / np.sqrt(s)))
    if k:
        return W[:,:k], -mu
    else:
        return W, -mu


def transform_and_normalize(vecs, kernel=None, bias=None):
    """应用变换，然后标准化
    Code taken from: https://github.com/bojone/BERT-whitening
    """
    if not (kernel is None or bias is None):
        vecs = (vecs + bias).dot(kernel)
    return vecs / (vecs**2).sum(axis=1, keepdims=True)**0.5


def calculate_sim_score(cos_results):
    results = []
    for parties in cos_results.keys():
        sim_score = np.mean(cos_results[parties]["across_parties"])/np.mean(cos_results[parties]["party1"]+cos_results[parties]["party2"])
        results.append((parties[0], parties[1], sim_score))
        results.append((parties[1], parties[0], sim_score))
    return results


def categorical_distance_matrices_claims():
    dist_cat={}
    for dataset in ["macov", "wom"]:
        df = pd.read_csv(f"./data/categorical_distance/{dataset}_claim.csv")
        dist_cat[f"part_{dataset}_claim"]=df.party.tolist()
        df = 1-df.drop(columns=["party"]).to_numpy()
        dist_cat[f"dist_{dataset}_claim"] = df
    return dist_cat

def compute_mantel(cat_arr, text_arr):
    r, pval, z = mantel.test(cat_arr, text_arr, perms=10000, method='spearman', tail='two-tail')
    return r, pval


def correlation_matrices(analysis_year):
    results=[]
    files = glob.glob(f"./results/*/*/*/*/*/cosine_scores/*.csv")
    print(files)
    # files.extend(glob.glob(f"./results_papermodel/*/*/*/*/cosine_scores/*.csv"))

    df_gold = pd.read_csv(f"./data/ground_truth/hamming_wom_claim_{analysis_year}.csv", index_col=0)
    cat_dist_mat = df_gold.to_numpy()
    cat_parties = df_gold.columns

    for file in files:

        f = file.split("/")
        df = pd.read_csv(file, index_col=0)
        df.index = df.columns

        text_arr = df.reindex(cat_parties)  #reordering indexes to correspond with categorical distance matrix
        text_arr = 1-text_arr[cat_parties].to_numpy()  #reordering columns

        r_corr, p_val = compute_mantel(cat_dist_mat, text_arr)

        results.append({"dataset":f[-3], "fine_tuning":f[-5], "post_proc":f[-6], "similarity_comput":f[-7],   #.split("_")[0]
                           "analysis_year":analysis_year, "finetuning_year":f[-4], "r":round(r_corr, 2), "pval":round(p_val, 4)})

    df = pd.DataFrame.from_dict(results)
    df.to_csv(f"./results/correlation.csv")
    for dataset in [["manifesto", "manifesto_claim_quote"], ["claim_identifier", "claim_identifier_claim_quote"]]:
        tmp = df[df["dataset"].isin(dataset)]
        tmp = tmp.sort_values(by=['fine_tuning'], ascending=False)
        print(tmp.head(40))
        print()
