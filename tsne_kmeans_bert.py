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

import torch
from torch.utils.data import DataLoader, TensorDataset, random_split, RandomSampler, Dataset
import pandas as pd
import numpy as np
import pickle

import transformers
import click
#from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import BertTokenizer, BertModel
import torch.nn.functional as F

from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import math
import random
import re
import os
import json

import matplotlib.pyplot as plt

SNLI_PATH = "/Users/mssaxon/data/snli_1.0/"
ANLI_PATH = "/Users/mssaxon/data/anli_v1.0/"
MNLI_PATH = "/Users/mssaxon/data/multinli_1.0/"
MODEL_PATH = '/Users/mssaxon/Documents/github/DatasetAnalysis/classifier/weights/roberta_nli'

#/mnt/hdd/saxon/anli_v1.0/

# tab separated gold_label s1bp s2bp s1p s2p s1 s2 captionID pairID label1 label2 label3 label4 label5

# generate sentensewise pair matrix

VALID_DATASETS = {
    "S" : ("snli", None, ["sentence1", "sentence2"]),
    "A1": ("anli", 1, ["context", "hypothesis"]),
    "A2": ("anli", 2, ["context", "hypothesis"]),
    "A3": ("anli", 3, ["context", "hypothesis"]),
    "M" : ("mnli", None, ["sentence1", "sentence2"])
}

FULL_LABEL_MAP = {
    "e" : "entailment",
    "c" : "contradiction",
    "n" : "neutral"
}


class lazydict():
    def __init__(self):
        self.indict = {}

    def add(self, key, val):
        if key not in self.indict.keys():
            self.indict[key] = []
        self.indict[key].append(val)


def load_sentences_str(registered_path, dataset, partition, sentencemap):
        lines = open(registered_path[partition]).readlines()
        #random.shuffle(lines)
        n = len(lines)

        sents = lazydict()

        for i, line in enumerate(lines):
            linecopy = line
            line = json.loads(line)
            if dataset == "S":
                lst = list(line["annotator_labels"])
                label = max(set(lst), key=lst.count)
            elif dataset == "M":
                label = line["gold_label"]
            else:
                label = FULL_LABEL_MAP[line["label"]]
            #label = line["gold_label"]
            s1 = line[sentencemap[0]] 
            s2 = line[sentencemap[1]]
            #print(label + "," + eval_sent)
            sents.add(label, (s1, s2, linecopy))
            # get the embedding
            print(f"{i}/{n}", end="\r")

        print("")
        return sents

@click.command()
@click.option('--basepath', default=SNLI_PATH)
@click.option('--model', default="BERT")
@click.option('--outpath', default="")
@click.option('--dataset', default="S")
@click.option('--partition', default="train")
@click.option('--debug', is_flag=True)
@click.option('--hides2', is_flag=True)
@click.option('--perp', default=30)
@click.option('--cuda', is_flag=True)
@click.option('--plot', is_flag=True)
@click.option('--redo_tsne', is_flag=True)
@click.option('--redo_model', is_flag=True)
@click.option('--cluster_subset', default=None)
@click.option('--apply_pca', is_flag=True)
@click.option('--save_text', is_flag=True)
@click.option('--n_clusters', default=50)
@click.option('--threshold', default=0.25)
def main(basepath, model, outpath, dataset, partition, debug, hides2, perp, cuda, plot, redo_tsne, redo_model, cluster_subset, apply_pca, save_text, n_clusters, threshold):
    if cluster_subset is not None:
        num_subset = int(cluster_subset)
        cluster_subset = True
    else:
        num_subset = -1
        cluster_subset = False
    debugplots = False

    ds, r, sentencemap = VALID_DATASETS[dataset]
    train = partition == "train"

    if ds == "anli" and basepath == SNLI_PATH:
        basepath = ANLI_PATH
    elif ds == "mnli" and basepath == SNLI_PATH:
        basepath = MNLI_PATH

    # SNLI
    registered_path = {
        'snli_train': basepath + "snli_1.0_" + "train.jsonl",
        'snli_dev': basepath + "snli_1.0_" + "dev.jsonl",
        'snli_test': basepath + "snli_1.0_" + "test.jsonl",
        'anli_train': basepath + f"R{r}/" + "train.jsonl",
        'anli_dev': basepath + f"R{r}/" + "dev.jsonl",
        'anli_test': basepath + f"R{r}/" + "test.jsonl",
        'mnli_train': basepath + f"multinli_1.0_" + "train.jsonl",
        'mnli_dev': basepath + f"multinli_1.0_" + "dev.jsonl",
        'mnli_test': basepath + f"multinli_1.0_" + "test.jsonl",
    }

    hides1 = not hides2
    if os.path.exists(f"{outpath}_{dataset}_{partition}_{model}_X.tmp.npy") and not redo_model:
        print("Loading temporary x file")
        X = np.load(f"{outpath}_{dataset}_{partition}_{model}_X.tmp.npy")
        labels = list(np.load(f"{outpath}_{dataset}_{partition}_{model}_l.tmp.npy"))
    else:
        print("failed, no pregen'd x...")
        assert False

    print("Loading sentence strs")

    ##################
    """
    sents = load_sentences_str(registered_path, dataset, f"{ds}_{partition}", sentencemap)


    ####################################### LEXICAL FEATURE COMPUTATION GOES HERE #####################################

    sents_l = []

    lens_hyps = []
    lablist = []

    for i, key in enumerate(sents.indict.keys()):
            lens_hyps += map(lambda x: np.log(float(len(x[1].split(" ")))), sents.indict[key])
            sents_l += sents.indict[key]
            lablist += len(sents.indict[key]) * [key]

    #for x in X_list:
    #    print(x.shape)


    ###################################### ARRAY OF LABELS BY ID #######################################################

    labarray = np.array(labels)
    global_balance = []
    for i in range(3):
        num_this_label = np.sum(labarray == i)
        print(num_this_label)
        global_balance.append(num_this_label)

    global_balance = np.array(global_balance)
    global_balance = global_balance / np.sum(global_balance)
    """
    global_balance = np.array([0.33297707,0.3330116,0.33401133])
    labarray = np.array(labels)
    ##################

    """
    p1 = np.array([-1.457,1.867])
    p2 = np.array([-0.465,-0.194])
    p3 = np.array([-1.265,-1.938])

    def get_theta(origin, point):
        delt = origin - point
        normed = delt / np.sqrt(np.sum(delt*delt))
        return np.arctan(normed[1] / normed[0]) + 3.14159265 / 2

    t1 = get_theta(p2, p1)
    t2 = get_theta(p2, p3)

    use_points = []
    bad_points = []

    use_counts = {'contradiction':0, 'neutral':0, 'entailment':0}
    bad_counts = {'contradiction':0, 'neutral':0, 'entailment':0}

    use_pt = []

    for row in range(X_tsne.shape[0]):
        xy = X_tsne[row,:]
        theta = get_theta(p2, xy)
        #print(theta)
        if theta < t2 and theta > t1 and xy[0] < p2[0]:
            # in the section
            use_points.append(sents_l[row])
            #print(labels[row])
            use_counts[lablist[row]] += 1
            use_pt.append("blue")
        else:
            bad_points.append(sents_l[row])
            bad_counts[lablist[row]] += 1
            use_pt.append("red")

    print(use_counts)
    print(bad_counts)
    """


    #################################### ANALYSIS OF CLUSTERS ##########################################################

    # possible steps:
    # cluster subset selection -> create a selection of a subset of the points for making PCA, KNN more tractable
    # PCA
    # KNN
    # over all points:
    # Assign KNN ID
    # count distribution in each cluster
    # clusterwise distance to global distribution
    # get % clusters over thresh, mark outliers
    # get AUC by varying thresh
    # analysis of outliers:
    # lexical analysis
    # if plotting:
    # plot subset selection -> if cluster subset not used, pick a subset for TSNE and readable plots
    # TSNE
    # plot


    #print(X)

    pklpth = f"{outpath}_{dataset}"

    # cluster subset selection: if cluster_subset, will be 50k random sents, else will be all
    selected = np.arange(0,X.shape[0])
    if cluster_subset:
        np.random.shuffle(selected)
        selected = selected[0:min(num_subset,X.shape[0])]
    
    # evaluate PCA over the selected points, and then transform all points according to found PCs
    pcapath = pklpth + f"_{model}_pca.tmp.npy"
    if os.path.exists(pcapath):
        pca = pickle.load(open(pcapath, "rb"))
    else:
        pca = PCA(n_components=50).fit(X[selected, :])
        pickle.dump(pca, open(pcapath, "wb"))

    X_pc = pca.transform(X)

    if debugplots:
        plt.plot(np.cumsum(pca.explained_variance_ratio_))
        plt.xlabel('number of components')
        plt.ylabel('cumulative explained variance');
        plt.show()


    # generate K Means clustering over the selected points, projected according to PCs
    kmpath = pklpth + f"_{model}_kms.tmp.npy"
    if os.path.exists(kmpath):
        kms = pickle.load(open(kmpath, "rb"))
    elif train:
        kms = KMeans(n_clusters=n_clusters, verbose=True, init='k-means++').fit(X_pc[selected, :])
        pickle.dump(kms, open(kmpath, "wb"))
    else:
        print("Please train first")
        assert False

    cluster_ids = kms.predict(X_pc)

    # count distribution in each cluster
    cluster_counts = [np.array([0,0,0]) for i in range(n_clusters)]
    for i in range(cluster_ids.shape[0]):
        cluster_counts[cluster_ids[i]][labarray[i]] += 1


    # compare clusterwise distribution against global distribution
    # get pseudoaccuracy for this set
    print(global_balance)

    imbalances = []

    print(cluster_ids)

    cluster_corrects = []
    cluster_sums = []


    if train:
        cluster_labels = []
    else:
        cluster_labels = np.load(f"{outpath}_{dataset}_train_{model}_cllb.tmp.npy")
    # train test

    for i in range(n_clusters):
        # check balance of this cluster
        this_balance = np.array(cluster_counts[i]) / sum(cluster_counts[i])
        #print(cluster_counts[i])
        imbalance = this_balance - global_balance
        imbalance = np.sqrt(np.sum(imbalance * imbalance))
        #print(imbalance)
        imbalances.append(imbalance)
        if train:
            cluster_corrects.append(max(cluster_counts[i]))
            cluster_labels.append(np.argmax(cluster_counts[i][0]))
        else:
            cluster_corrects.append(cluster_counts[i][cluster_labels[i]])
        cluster_sums.append(sum(cluster_counts[i]))

    if train:
        np.save(f"{outpath}_{dataset}_{partition}_{model}_cllb.tmp", np.array(cluster_labels))

    #   # imbalances contains the distance metric between each cluster and the global
    imbalances = np.array(imbalances)

    print(imbalances.max())
    print(imbalances.mean())
    print(imbalances.min())

    # cluster corrects contains the number of points in each cluster that agree with the cluster label
    correct_classification = sum(cluster_corrects) / float(sum(cluster_sums))

    print(f"Fit for this clustering: {correct_classification}")

    # outlier is a binary tag for each example: true if in outlier cluster, false else
    outlier = list(map(lambda x: list(imbalances > threshold)[x], list(cluster_ids)))

    outliers = []
    regulars = []
    #print(len(keys))
    #print(len(outlier))
    #print(X_tsne.shape[0])

    # compute cluster curve, AUC
    cumulative_outlier = []
    cumulative_x = []
    for i in range(n_clusters):
        cumulative_outlier.append(np.sum(imbalances > float(i) / n_clusters))
        cumulative_x.append( float(i) / n_clusters )

    auc = float(sum(cumulative_outlier)) / n_clusters

    print(f"AUC for this clustering: {auc}")


    # if we want to save the selected dataset lines for the outliers, do so now

    if save_text:
        outlier_lines = []
        for i in range(len(outlier)):
            if outlier[i]:
                outlier_lines.append(sents_l[i][2])
        with open(f"{outpath}_{dataset}_{partition}_{model}_outliers.jsonl", "w") as outfile:
            outfile.writelines(outlier_lines)

    ####################################### PLOTTING ########################################################

    if plot:
        # randomly select points to use again
        np.random.shuffle(selected)
        selected = selected[0:min(20000,X.shape[0])]

        # TSNE can probably go later
        if os.path.exists(f"{outpath}_{dataset}_{partition}_{model}_T{perp}.tmp.npy") and not redo_tsne:
            print(f"Loading temporary tsne, perp={perp}")
            X_tsne = np.load(f"{outpath}_{dataset}_{partition}_{model}_T{perp}.tmp.npy")
            selected = np.load(f"{outpath}_{dataset}_{partition}_{model}_T{perp}s.tmp.npy")
        else:
            print(f"Fitting TSNE, perp={perp}...")
            X_tsne = TSNE(perplexity=perp, verbose=2).fit_transform(X_pc[selected,:])
            if redo_tsne:
                np.save(f"{outpath}_{dataset}_{partition}_{model}_T{perp}r.tmp", X_tsne)
                np.save(f"{outpath}_{dataset}_{partition}_{model}_T{perp}sr.tmp", selected)
            else:
                np.save(f"{outpath}_{dataset}_{partition}_{model}_T{perp}.tmp", X_tsne)
                np.save(f"{outpath}_{dataset}_{partition}_{model}_T{perp}s.tmp", selected)

        #keys = list(sents.indict.keys())
        # # # # # # # # # # # HACK # # # # # # # # # # # 
        keys = ["neutral", "contradiction", "entailment"]
        colors = {'contradiction':'orange', 'neutral':'black', 'entailment':'blue'}

        # for plotting the outliers vs regulars
        outlier = list(np.array(outlier)[selected])
        labels = list(np.array(labels)[selected])
        for i in range(len(outlier)):
            if outlier[i]:
                outliers.append((X_tsne[i,:], labels[i]))
            else:
                regulars.append((X_tsne[i,:], labels[i]))
        # group by condition

        #plt.scatter(np.flip(X_tsne[:,0]), np.flip(X_tsne[:,1]), c=list(map(lambda x: colors[keys[x]], list(np.flip(np.array(labels))))))
        X_tsne, labels = zip(*outliers)
        X_tsne = np.stack(X_tsne)
        plt.scatter(X_tsne[:,0], X_tsne[:,1], c=list(map(lambda x: colors[keys[x]], list(np.array(labels)))), marker="x")
        X_tsne, labels = zip(*regulars)
        X_tsne = np.stack(X_tsne)
        idces = list(range(len(labels)))
        random.shuffle(idces)
        idces = np.array(idces)
        plt.scatter(X_tsne[:,0], X_tsne[:,1], c=list(map(lambda x: colors[keys[x]], list(np.array(labels)))), marker="x", s=1)
        #plt.scatter(X_tsne[:,0], X_tsne[:,1], c=cluster_ids)
        #plt.scatter(X_tsne[:,0], X_tsne[:,1], c=use_pt)
        #plt.scatter(X_tsne[:,0], X_tsne[:,1], c=lens_hyps)
        #plt.plot([-1.265,-0.465,-1.457],[-1.938,-0.194,1.867])
        plt.show()

        plt.plot(cumulative_x, cumulative_outlier)
        plt.title(ds)
        plt.xlabel("Threshold (L2 distance from global distribution of labels)")
        plt.ylabel("No. Clusters exceeding Threshold")
        plt.show()



    #plt.scatter(np.flip(X_tsne[:,0]), np.flip(X_tsne[:,1]), c=list(map(lambda x: colors[keys[x]], list(np.flip(np.array(labels))))))
    #plt.scatter(X_tsne[:,0], X_tsne[:,1], c=list(map(lambda x: colors[keys[x]], labels)))
    #plt.scatter(X_tsne[:,0], X_tsne[:,1], c=use_pt)
    #plt.scatter(X_tsne[:,0], X_tsne[:,1], c=lens_hyps)
    #plt.plot([-1.265,-0.465,-1.457],[-1.938,-0.194,1.867])
    #plt.show()
    #plt.



    #tok()
    """
    def preprocess_function(examples):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tok(*args, padding=padding, max_length=max_length, truncation=True)
        return result




    scorer = bertscorer(model, tok)
    for i, paper in enumerate(readsplit(basepath + "/" + evalfile, split_on_line = False)):
        # get pairwise for each sentence (slow as fucking shit)
        s1s = {}
        for k, s2 in enumerate(paper[1]):
            max_k = 0
            best_k = ""
            s2_t, _ = scorer.getencoding(s2)
            scores = []
            for j, s1 in enumerate(paper[0]):
                if len(s1.split(" ")) < 5:
                    continue
                    scores.append(0)
                print(f"{k}/{len(paper[1])}, {j}/{len(paper[0])}", end="\r")
                if s1 not in s1s.keys():
                    s1s[s1], _ = scorer.getencoding(s1)
                    s1_t = s1s[s1]
                else:
                    s1_t = s1s[s1]
                this_score = scorer.pbert(s1_t, s2_t)
                #print(f"{s1}, {s2}, {this_score}")
                scores.append(-this_score)
            top_k = np.argsort(scores)[0:3]
            #print(top_k)
            best_k = list(np.array(paper[0])[top_k])
            print(f"{s2}||||||||{best_k}")
        if debug and i == 0:
            break
    """

    #return X, colors, keys, labels


if __name__ == "__main__":
    main()