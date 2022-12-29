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
from dataset_to_clusters import *
from manage_settings import get_write_settings, read_models_csv, lazymkdir

from collections import defaultdict

from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import pickle

import numpy as np

from tqdm import tqdm

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from cartography import cartography_from_dir, find_cartography_dir


# map val set samples into an existing test set clustering (so the clusters can be characterized)

# for each cluster in our existing clustering, map and assign each sample in the dev set for cartography


# continuous version of dataset_to_clusters.cluster_preds_to_dists()
def cluster_values_to_dists(embs_cll, values, n_clusters):
    cluster_dists = [[] for i in range(n_clusters)]
    for i in range(embs_cll.shape[0]):
        cluster_dists[embs_cll[i]].append(values[i])
    return cluster_dists


def hist_by_L2_bin(df):
    fig, axs = plt.subplots(2, 1)
    sns.histplot(df[df["outlier"] == True], x="mus", element="step", ax = axs[0], bins=50, binrange=(0,1), color="orange", kde=True)
    sns.histplot(df[df["outlier"] == False], x="mus", element="step", ax = axs[1], bins=50, binrange=(0,1), color="blue", kde=True)
    fig.savefig("histogram_cartography.png")


@click.command()
@click.option('--skip_gpu', is_flag=True)
@click.option('--dataset', help="S, M, A1, A2, A3, OC, SICK, etc")
@click.option('--batch_size', default=48)
@click.option('--biased', is_flag=True)
@click.option('--extreme_bias', is_flag=True)
@click.option('--s2only', is_flag=True)
@click.option('--s1only', is_flag=True)
@click.option('--lastdense', is_flag=True)
@click.option('--n_clusters', default=50)
@click.option('--n_epochs', default=3)
@click.option('--threshold', default=0.25)
def main(skip_gpu, dataset, biased, batch_size, extreme_bias, s1only, s2only, n_clusters, lastdense, n_epochs, threshold):
    model_id, pretrained_path = read_models_csv(dataset)
    model, tokenizer = choose_load_model_tokenizer(model_id, dataset)
    ltmodel = RobertaClassifier(model, learning_rate=0)
    print("Init litmodel...")
    if pretrained_path != "":
        ckpt = torch.load(pretrained_path)
        ltmodel.load_state_dict(ckpt["state_dict"])
    print("Init dataset...")
    if extreme_bias:
        factor = 0
    else:
        factor = 1

    if not skip_gpu:
        model.cuda()
        ltmodel.cuda()

    if s1only:
        s2only = False

    dir_settings = get_write_settings(["dataset_dir", "intermed_comp_dir_base", "cartography_save_dir"])

    intermed_comp_dir = setup_intermed_comp_dir(dir_settings["intermed_comp_dir_base"], dataset,
        n_clusters, lastdense, (biased, extreme_bias, s2only or s1only))

    nli_data = plNLIDataModule(tokenizer, dir_settings["dataset_dir"], dataset, batch_size, biased, factor, s2only, s1only)
    # we want to embed and PCA the train and val sets in order to assign clusters to samples
    nli_data.prepare_data(test_only = False)


    # load up an existing embedding model, PCA, clustering->apply to the val set

    embs, labs = get_numpy_embs(nli_data, ltmodel, tmp_save_dir=intermed_comp_dir, lastdense = lastdense, partition = "val")
    # pca transformed embeddings

    embs_pca = pca_fit_transform(embs, tmp_save_dir=intermed_comp_dir, force_load = True)

    # cluster-labeled embeddings
    embs_cll = kmeans_fit_transform(embs_pca, tmp_save_dir=intermed_comp_dir, n_clusters = n_clusters, force_load = True)

    mus, sigmas = cartography_from_dir(
            find_cartography_dir(dir_settings["cartography_save_dir"], dataset, model_id),
            n_epochs=n_epochs,
            key="val"
        )

    mus_dist = cluster_values_to_dists(embs_cll, mus, n_clusters = n_clusters)
    sigmas_dist = cluster_values_to_dists(embs_cll, sigmas, n_clusters = n_clusters)

    cluster_dists, global_dist = cluster_preds_to_dists(embs_cll, labs, n_clusters = n_clusters)
    clusters_L2 = cluster_L2(cluster_dists, global_dist)
    
    outlier_cluster_ids = np.arange(clusters_L2.shape[0]) * (clusters_L2 > threshold) + -1 * (clusters_L2 <= threshold)

    is_outlier = (np.expand_dims(outlier_cluster_ids, 0) == np.expand_dims(embs_cll, -1)).sum(-1)

    sample_cluster_L2 = (
        (np.expand_dims(embs_cll, -1) == np.expand_dims(np.arange(clusters_L2.shape[0]), 0)) * np.expand_dims(clusters_L2, 0)
    ).sum(-1)


    # print(is_outlier)

    #is_outlier = "outlier" * is_outlier + "not outlier" * (1-is_outlier)

    df_sample = pd.DataFrame(np.stack([embs_cll, mus, sigmas, is_outlier, sample_cluster_L2], axis=-1), columns = ["cluster", "mus", "sigmas", "outlier", "L2"])

    df_cluster = pd.DataFrame(np.stack([np.arange(clusters_L2.shape[0]), clusters_L2, mus_dist, sigmas_dist], axis=-1))
    # assign "outlier" to some clusters



    print("Plotting...")

    #sns.histplot(df, x="mus", y="L2", hue="outlier", bins=10, legend=False)
    #plt.savefig("histogram_cartography.png")

    # manual histogram algo
    hist_by_L2_bin(df_sample)


if __name__ == "__main__":
    main()