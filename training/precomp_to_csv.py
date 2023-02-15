"""
Examples

CUDA_VISIBLE_DEVICES=0 python dataset_to_clusters.py --dataset CF --s2only

CUDA_VISIBLE_DEVICES=0 python dataset_to_clusters.py --s2only --lastdense --dataset S
CUDA_VISIBLE_DEVICES=1 python dataset_to_clusters.py --s2only --lastdense --dataset SICK
CUDA_VISIBLE_DEVICES=2 python dataset_to_clusters.py --s2only --lastdense --dataset MU
CUDA_VISIBLE_DEVICES=3 python dataset_to_clusters.py --s2only --lastdense --dataset MB
CUDA_VISIBLE_DEVICES=0 python dataset_to_clusters.py --s2only --lastdense --dataset SdbA
CUDA_VISIBLE_DEVICES=1 python dataset_to_clusters.py --s2only --lastdense --dataset MdbA
CUDA_VISIBLE_DEVICES=1 python dataset_to_clusters.py --s1only --lastdense --dataset F

"""
import click
from sklearn import cluster

from train_classifier import *
from manage_settings import get_write_settings, read_models_csv, lazymkdir
from transformers import RobertaTokenizer, RobertaForSequenceClassification, BertTokenizer, BertForSequenceClassification

from collections import defaultdict

from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import pickle

import numpy as np

from tqdm import tqdm

import matplotlib.pyplot as plt

from dataset_to_clusters import *


def tsne_csv(embs, labels, cluster_ids, cluster_norms, tmp_save_dir, perp=30, threshold=2.5):
    embs_tsne = tsne_fit_transform(embs, perp, tmp_save_dir)
    under_thresh = defaultdict(list)
    above_thresh = defaultdict(list)
    for i in range(embs.shape[0]):
        if cluster_norms[cluster_ids[i]] > threshold:
            above_thresh[labels[i]].append(embs_tsne[i,:])
        else:
            under_thresh[labels[i]].append(embs_tsne[i,:])
    # entailment:0, contradict:1, neutral:2

    fig, ax = plt.subplots(figsize=(4,4))
    colors = ['blue', 'orange', 'black']
    for i in range(3):
        above_thresh[i] = np.stack(above_thresh[i])
        under_thresh[i] = np.stack(under_thresh[i])
        print(above_thresh[i].shape)
        print(under_thresh[i].shape)
        ax.scatter(above_thresh[i][:,0], above_thresh[i][:,1],
        c = colors[i], marker="x", alpha=0.4)
        ax.scatter(under_thresh[i][:,0], under_thresh[i][:,1],
        c = colors[i], marker="x", s=1, alpha=0.4)
    return fig


# should clean out biased vs """""extreme_bias""""""" and make function clearer. 
# s2only/s1only implemented as maps somehow? extreme_bias should be zero_tokens
@click.command()
@click.option('--dataset', help="S, M, A1, A2, A3, OC, SICK, etc")
@click.option('--s1only', is_flag=True)
@click.option('--s2only', is_flag=True)
@click.option('--lastdense', is_flag=True)
@click.option('--n_clusters', default=50)
@click.option('--n_components', default=50)
@click.option('--tsne_thresh', default=2.5)
@click.option('--tsne', is_flag=True)
def main(dataset, s1only, s2only, n_clusters, n_components, lastdense, tsne_thresh, tsne):
    dir_settings = get_write_settings(["dataset_dir", "intermed_comp_dir_base", "model_ckpts_path"])

    skip_pca = n_components == 0

    intermed_comp_dir = setup_intermed_comp_dir(dir_settings["intermed_comp_dir_base"], dataset, lastdense = lastdense)

    # collect lists of numpy arrays
    if s1only:
        affix = "-s1"
    elif s2only:
        affix = "-s2"
    else:
        affix = ""
    embs, labs = get_numpy_embs(tmp_save_dir=intermed_comp_dir, lastdense = lastdense, affix=affix)
    # pca transformed embeddings
    if not skip_pca:
        embs_pca = pca_fit_transform(embs, n_components=n_components, tmp_save_dir=intermed_comp_dir)
        pca_affix = f"-pca{n_components}"
    else:
        embs_pca = embs
        pca_affix = ""
    # cluster-labeled embeddings
    embs_cll = kmeans_fit_transform(embs_pca, tmp_save_dir=intermed_comp_dir, n_clusters = n_clusters, pca_affix=pca_affix)
    # vectors = get_cluster_vectors(embs, embs_cll)
    ## evaluate the PECO measure
    # get the cluster cross entropies
    cluster_dists, global_dist = cluster_preds_to_dists(embs_cll, labs, n_clusters = n_clusters)
    clusters_xHs = cluster_xH(cluster_dists, global_dist)
    clusters_L2 = cluster_L2(cluster_dists, global_dist)
    peco_xH = peco_score(clusters_xHs, scale = 5)
    peco_L2 = peco_score(clusters_L2)
    threshscore_xH_25 = threshold_score(clusters_xHs, .25 * 10)
    threshscore_L2_25 = threshold_score(clusters_L2, .25)
    # generate AUC plots
    # generate T-SNE plot
    print("##### REPORT #####")
    lines = ["peco_xH,peco_L2,thresh_xH_25,thresh_L2_25", 
        f"{peco_xH:.4f},{peco_L2:.4f},{threshscore_xH_25:.4f},{threshscore_L2_25:.4f}"]
    with open(PurePath(intermed_comp_dir + f"/results-{'s2' * s2only + 's1' * s1only}-pc{n_components}-k{n_clusters}.csv"), "w") as f:
        for line in lines:
            print(line)
            f.write(line + "\n")
    if tsne:
        fig = plot_outliers(embs_pca, labs, embs_cll, clusters_xHs, tmp_save_dir=intermed_comp_dir, threshold=tsne_thresh)
        fig.savefig(f"{intermed_comp_dir}/test.pdf")

    print(global_dist)
    print("##### HIGHEST BIAS ClUSTERS #####")
    max_bias_clusters = sorted(clusters_L2, reverse=True)
    for i in range(5):
        bias_cluster = np.where(clusters_L2 == max_bias_clusters[i])[0][0]
        print(f"#{i} bias cluster : {bias_cluster}, {max_bias_clusters[i]}")
        print("distribution:")
        print(cluster_dists[i])
        idces = list(np.arange(embs_cll.shape[0])[embs_cll == bias_cluster])
        with open(f"{intermed_comp_dir}/{i}-idces.csv","w") as f:
            f.writelines(map(lambda x: str(x) + "\n", idces))
        
    # print the number 1


if __name__ == "__main__":
    main()