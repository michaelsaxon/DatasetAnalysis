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

def single_set(fname):
    lines = open(fname,"r").readlines()
    return lines[1].split(",")[0:2]


def collate(dataset, ld = True, part = 's2'):
    dir_settings = get_write_settings(["intermed_comp_dir_base"])
    cd_base = dir_settings["intermed_comp_dir_base"]
    outs = []
    for k in [10, 25]:
        for pc in [0, 50, 100]:
            set = single_set(f"{cd_base}/{ld * 'ld-'}{dataset}-test/results-{part}-pc{pc}-k{k}.csv")
            outs += set
    print(f"{dataset} & {part} & {' & '.join(outs)}")
    outs = []
    for k in [50, 100]:
        for pc in [0, 50, 100]:
            set = single_set(f"{cd_base}/{ld * 'ld-'}{dataset}-test/results-{part}-pc{pc}-k{k}.csv")
            outs += set
    print(f"{dataset} & {part} & {' & '.join(outs)}")


# ld s2only
folders = 'A1 A2 A3 AA CF MB MdbA MU SdbA SICK S X'.split(" ")
for dataset in folders:
    collate(dataset)

# ld s1only
folders = 'CF F MdbA SdbA'.split(" ")
for dataset in folders:
    collate(dataset, part='s1')

# s2only
folders = ['OC']
for dataset in folders:
    collate(dataset, ld=False)
