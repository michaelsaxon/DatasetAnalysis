"""
CUDA_VISIBLE_DEVICES=0 python tsne_kmeans.py --cuda \
--modelpath /mnt/hdd/saxon/roberta-nli/weights/roberta_nli/ \
--outpath /mnt/hdd/saxon/roberta-nli/ \
--basepath /mnt/hdd/saxon/snli_1.0/ --dataset S --partition train

CUDA_VISIBLE_DEVICES=1 python tsne_kmeans.py --cuda \
--modelpath /mnt/hdd/saxon/roberta-nli/weights/roberta_nli/ \
--outpath /mnt/hdd/saxon/roberta-nli/ \
--basepath /mnt/hdd/saxon/anli_v1.0/ --dataset A1 --partition train

CUDA_VISIBLE_DEVICES=2 python tsne_kmeans.py --cuda \
--modelpath /mnt/hdd/saxon/roberta-nli/weights/roberta_nli/ \
--outpath /mnt/hdd/saxon/roberta-nli/ \
--basepath /mnt/hdd/saxon/anli_v1.0/ --dataset A2 --partition train

CUDA_VISIBLE_DEVICES=3 python tsne_kmeans.py --cuda \
--modelpath /mnt/hdd/saxon/roberta-nli/weights/roberta_nli/ \
--outpath /mnt/hdd/saxon/roberta-nli/ \
--basepath /mnt/hdd/saxon/anli_v1.0/ --dataset A3 --partition train


"""
import pandas as pd
import numpy as np
import pickle

import click

from sklearn.metrics.pairwise import cosine_similarity

import math
import random
import re
import os
import json

NUM_USE = 50000

@click.command()
@click.option('--fname')
@click.option('--train', default=False)
@click.option('--n_clusters', default=50)
def main(fname, train, n_clusters):
    assert "X" in fname

    ident = fname.split("/")[-1].split(".")[0].strip("_")
    ds = ident.split("_")[0]
    bert = ident.split("_")[2] == "BERT"
    part = ident.split("_")[1]
    if bert:
        model = "BERT"
    else:
        model = "RB"


    X = np.load(fname)

    """subset = X.shape[0] > NUM_USE

    if subset:
        idces = list(range(X.shape[0]))
        random.shuffle(idces)
        idces = np.array(idces[0:NUM_USE])
        X_use = X[idces,:]
        np.save(f"../{ds}_{part}_{model}_idces.npy", idces)
    else:
        X_use = X"""
    X_use = X

    # parse the 
    pcapath = f"tmp/_{ds}_{model}_pca.tmp.npy"
    pca = pickle.load(open(pcapath, "rb"))

    X_pc = pca.transform(X_use)
    print(X_pc.shape)
    # get cosine similarity

    kmpath = f"tmp/_{ds}_{model}_kms.tmp.npy"
    if os.path.exists(kmpath):
        kms = pickle.load(open(kmpath, "rb"))
    elif train:
        kms = KMeans(n_clusters=n_clusters, verbose=True, init='k-means++').fit(X_pc[selected, :])
        pickle.dump(kms, open(kmpath, "wb"))
    else:
        print("Please train first")
        assert False

    cluster_ids = kms.predict(X_pc)


    labarray = np.load(f"tmp/_{ds}_{part}_{model}_l.tmp.npy")


    cluster_counts = [np.array([0,0,0]) for i in range(n_clusters)]
    global_counts = np.array([0,0,0])
    for i in range(cluster_ids.shape[0]):
        cluster_counts[cluster_ids[i]][labarray[i]] += 1
        global_counts[labarray[i]] += 1

    global_balance = global_counts / np.sum(global_counts)


    imbalances = []


    cluster_corrects = []
    cluster_sums = []

    print(f"{ds}, {model}")
    for first, second in [(0,1), (1,2), (0,2)]:
        corrects = 0
        totals = 0
        for cluster_count in cluster_counts:
            largest = np.max(cluster_count[np.array([first, second])])
            total = np.sum(cluster_count[np.array([first, second])])
            corrects += largest
            totals += total
        print(f"{first}, {second}, {corrects / totals:.3f}")



if __name__ == "__main__":
    main()