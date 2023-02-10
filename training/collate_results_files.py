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
    line = list(map(lambda x: float(x), lines[1].strip().split(",")))
    line = [f"{line[0]*100:.1f}", f"{line[1]:.3f}"]
    return lines[1].split(",")[0:2]

def print_files(cd_base, ld, dataset, part, pcs = [0, 50, 100], ks = [10, 25]):
    outs = []
    for k in ks:
      
        for pc in pcs:
            set = single_set(f"{cd_base}/{ld * 'ld-'}{dataset}-test/results-{part}-pc{pc}-k{k}.csv")
            outs += set
    print(f"{dataset} & {part} & {' & '.join(outs)} \\\\")


def collate(dataset, ld = True, part = 's2'):
    dir_settings = get_write_settings(["intermed_comp_dir_base"])
    cd_base = dir_settings["intermed_comp_dir_base"]
    print_files(cd_base, ld, dataset, part)
    print_files(cd_base, ld, dataset, part, ks = [50, 100])


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
