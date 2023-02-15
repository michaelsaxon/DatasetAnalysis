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


def tsne_csv(embs, tmp_save_dir, perp=30, threshold=2.5):
    embs_tsne = tsne_fit_transform(embs, perp, tmp_save_dir)
    return embs_tsne


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

    clusters_L2 = cluster_L2(cluster_dists, global_dist)
    peco_L2 = peco_score(clusters_L2)

    tsne_coords = tsne_fit_transform(embs, 30, intermed_comp_dir)

    print(clusters_L2.shape)
    print(peco_L2.shape)
    print(tsne_coords.shape)
    print(labs.shape)
    print(embs_cll.shape)




if __name__ == "__main__":
    main()