from sentence_transformers import util  #SentenceTransformer
import pandas as pd
import torch
from os.path import join
import pickle
from collections import defaultdict
from itertools import combinations
import numpy as np
from utils_sim import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_distance_seq(mat, output_dir, kernel=None, bias=None, save_cosine=True):
    metadata = mat[:, -2:]
    mat = mat[:, :-2].astype(float)

    cos_results= {}
    parties = set(list(metadata[:, 0]))

    for p1, p2 in combinations(parties, 2):
        cos_results[(p1, p2)] = defaultdict(list)
        ind1 = [i[0] for i in np.argwhere(metadata == p1)]
        ind2 = [i[0] for i in np.argwhere(metadata == p2)]

        embeddings1 = mat[ind1,:]
        embeddings2 = mat[ind2, :]
        if kernel is not None:
            embeddings1 = transform_and_normalize(embeddings1, kernel, bias)
            embeddings2 = transform_and_normalize(embeddings2, kernel, bias)

        #Compute cosine-similarits
        cosine_scores1 = util.cos_sim(embeddings1, embeddings1).numpy()   #compute_distances_no_loops(embeddings1, embeddings1) #
        cosine_scores2 = util.cos_sim(embeddings2, embeddings2).numpy()
        cosine_scores_across =  util.cos_sim(embeddings1, embeddings2).numpy()

        dic={'party1':cosine_scores1, 'party2':cosine_scores2, 'across_parties':cosine_scores_across}

        for name, cosine_scores in dic.items():
            sent_index = defaultdict(list)
            if name == "across_parties":
                for i in range(cosine_scores.shape[0]):
                    for j in range(cosine_scores.shape[1]):
                        sent_index[i].append(cosine_scores[i][j])
                        sent_index[j].append(cosine_scores[i][j])

            else:
                for i in range(cosine_scores.shape[0] - 1):  
                    for j in range(i + 1, cosine_scores.shape[1]):
                        sent_index[i].append(cosine_scores[i][j])

            for idx, score in sent_index.items():
                # print(len(score), len(set(score)))
                cos_results[(p1, p2)][name].append(max(score))  #append the max score of every sentence

    sim_scores=calculate_sim_score(cos_results)
    df_matrix = pd.DataFrame.from_records(sim_scores, columns=['party1', 'party2', 'sim_score'])
    df_matrix = df_matrix.pivot('party1', 'party2', 'sim_score')
    df_matrix = df_matrix.fillna(1)
    print("df_matrix:\n", df_matrix)
    matrix = np.array(df_matrix)
    yaxis = df_matrix.index.tolist()
    xaxis = df_matrix.columns.tolist()

    plot_cosine_matrix(matrix, "computed_cosine", "Average over all sequences", ["1"] + yaxis, ["1"] + xaxis, output_dir)

    if save_cosine:
        save_cosine_csv(df_matrix, "computed_cosine", output_dir)


def compute_distance_seq_with_domain(mat, output_dir, kernel=None, bias=None, save_cosine=True):

    avg_scores = defaultdict(list)
    metadata = mat[:, -2:]
    mat = mat[:, :-2].astype(float)

    categories = set(list(metadata[:, 1]))
    for cat in categories:
        ind = [i[0] for i in np.argwhere(metadata == cat)]
        embeddings_cat = mat[ind,:]
        metadata_cat = metadata[ind,:]

        parties = set(list(metadata_cat[:,0]))
        new_emb = np.empty((0, embeddings_cat.shape[1]))
        new_meta = np.empty((0, metadata_cat.shape[1]))

        for p in parties:
            ind = [i[0] for i in np.argwhere(metadata_cat == p)]
            embeddings_cat_p = embeddings_cat[ind, :]
            metadata_cat_p = metadata_cat[ind, :]

            if embeddings_cat_p.shape[0]>1:
                new_emb=np.vstack((new_emb, np.mean(embeddings_cat_p, axis=0)))
                new_meta = np.vstack((new_meta, metadata_cat_p[0, :]))
            elif embeddings_cat_p.shape[0]==1:
                new_emb = np.vstack((new_emb, embeddings_cat_p))
                new_meta = np.vstack((new_meta, metadata_cat_p))
            else:
                continue

        if kernel is not None:
            new_emb = transform_and_normalize(new_emb, kernel, bias)
        cosine_scores = util.cos_sim(new_emb, new_emb)
        # cosine_scores = compute_distances_no_loops(new_emb, new_emb)

        parties = list(new_meta[:, 0])
        for i in range(len(cosine_scores) - 1):
            for j in range(i + 1, len(cosine_scores)):
                avg_scores[(parties[i], parties[j])].append(cosine_scores[i][j])

    results=[]
    for k in avg_scores.keys():
        results.append((k[0], k[1], np.mean(avg_scores[k])))

    df_matrix = pd.DataFrame.from_records(results, columns=['party1', 'party2', 'sim_score'])
    df_matrix = df_matrix.pivot('party1', 'party2', 'sim_score')
    df_matrix = df_matrix.reindex(df_matrix.columns)
    df_matrix = df_matrix.fillna(1)
    print(df_matrix)
    matrix = np.array(df_matrix)
    yaxis = df_matrix.index.tolist()
    xaxis = df_matrix.columns.tolist()

    plot_cosine_matrix(matrix, "computed_cosine", "Average over all sequences", [1] + yaxis, [1] + xaxis, output_dir)
    if save_cosine:
        save_cosine_csv(df_matrix, "computed_cosine", output_dir)


def compute_distances(do_whiten=False):
    models_emb = glob.glob("./embeddings/*/*/*/embeddings.p")

    for model_emb in models_emb:
        print(model_emb)

        mat = pickle.load(open(model_emb, "rb"))
        kernel, bias = compute_kernel_bias(mat[:, :-2].astype(float), None)  # k=None - no dimensionality reduction
        output_dir = f"./results/no_domain/" + "whiten" + "/" + join(*list(Path(model_emb).parts[1:-1]))

        if do_whiten:
            print(output_dir)
            compute_distance_seq(mat, output_dir, kernel=kernel, bias=bias)
            output_dir1 = output_dir.replace("/no_domain/", "/with_domain/")
            print("output folder: ", output_dir1)
            compute_distance_seq_with_domain(mat, output_dir1, kernel=kernel, bias=bias)
        else:
            output_dir = output_dir.replace("whiten", "no_whiten")
            compute_distance_seq(mat, output_dir)
            output_dir = output_dir.replace("/no_domain/", "/with_domain/")
            print("output folder: ", output_dir)
            compute_distance_seq_with_domain(mat, output_dir)
