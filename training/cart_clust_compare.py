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
def main(skip_gpu, dataset, biased, batch_size, extreme_bias, s1only, s2only, n_clusters, lastdense, n_epochs):
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

    df = pd.DataFrame(np.stack([embs_cll, mus, sigmas], axis=-1), columns = ["cluster", "mus", "sigmas"])

    sns.histplot(df, x="mus", y="cluster", kde=True)
    plt.savefig("histogram_cartography.png")


if __name__ == "__main__":
    main()