import click
from sklearn import cluster

from train_classifier import *
from manage_settings import get_write_settings, read_models_csv, lazymkdir

from collections import defaultdict

import numpy as np

from tqdm import tqdm

import matplotlib.pyplot as plt

from dataset_to_clusters import *

def cluster_relationship_experiment(ltmodel, nli_data, lastdense, intermed_comp_dir, try_clusters_range = range(10,110,10)):
    pecos_xH = []
    pecos_L2 = []
    ts25s_xH = []
    ts25s_L2 = []
    cluster_vectors = []
    for n_clusters in try_clusters_range:
        # collect lists of numpy arrays
        embs, labs = get_numpy_embs(nli_data, ltmodel, tmp_save_dir=intermed_comp_dir, lastdense = lastdense)
        # pca transformed embeddings
        embs_pca = pca_fit_transform(embs, tmp_save_dir=intermed_comp_dir)
        # cluster-labeled embeddings
        embs_cll = kmeans_fit_transform(embs_pca, n_clusters = n_clusters)
        vectors = get_cluster_vectors(embs, embs_cll)
        cluster_vectors.append(vectors)
        ## evaluate the PECO measure
        # get the cluster cross entropies
        cluster_dists, global_dist = cluster_preds_to_dists(embs_cll, labs, n_clusters = n_clusters)
        clusters_xHs = cluster_xH(cluster_dists, global_dist)
        clusters_L2 = cluster_L2(cluster_dists, global_dist)
        peco_xH = peco_score(clusters_xHs, scale = 5)
        peco_L2 = peco_score(clusters_L2)
        threshscore_xH_25 = threshold_score(clusters_xHs, .25 * 10)
        threshscore_L2_25 = threshold_score(clusters_L2, .25)
        pecos_xH.append(peco_xH)
        pecos_L2.append(peco_L2)
        ts25s_xH.append(threshscore_xH_25)
        ts25s_L2.append(threshscore_L2_25)
    return try_clusters_range, pecos_xH, pecos_L2, ts25s_xH, ts25s_L2, cluster_vectors



@click.command()
@click.option('--n_gpus', default=1, help='number of gpus')
@click.option('--dataset', help="S, M, A1, A2, A3, OC, SICK, etc")
@click.option('--batch_size', default=48)
@click.option('--biased', is_flag=True)
@click.option('--extreme_bias', is_flag=True)
@click.option('--s2only', is_flag=True)
@click.option('--s1only', is_flag=True)
@click.option('--lastdense', is_flag=True)
@click.option('--n_clusters', default=50)
@click.option('--tsne_thresh', default=2.5)
def main(n_gpus, dataset, biased, batch_size, extreme_bias, s1only, s2only, n_clusters, lastdense, tsne_thresh):
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

    model.cuda()
    ltmodel.cuda()

    if s1only:
        s2only = False

    dir_settings = get_write_settings(["data_save_dir", "dataset_dir", "intermed_comp_dir_base"])
    
    intermed_comp_dir = setup_intermed_comp_dir(dir_settings["intermed_comp_dir_base"], dataset,
        n_clusters, lastdense, (biased, extreme_bias, s2only or s1only))

    nli_data = plNLIDataModule(tokenizer, dir_settings["dataset_dir"], dataset, batch_size, biased, factor, s2only, s1only)
    nli_data.prepare_data(test_only = True)

    _, pecos_xH, pecos_L2, ts25s_xH, ts25s_L2, vectors = cluster_relationship_experiment(ltmodel, nli_data, lastdense, intermed_comp_dir)



if __name__ == "__main__":
    main()