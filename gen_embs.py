"""
CUDA_VISIBLE_DEVICES=0 python gen_embs.py --cuda \
--outpath /mnt/hdd/saxon/roberta-nli/ \
--basepath /mnt/hdd/saxon/snli_1.0/ --dataset S --partition train

CUDA_VISIBLE_DEVICES=1 python gen_embs.py --cuda \
--outpath /mnt/hdd/saxon/roberta-nli/ \
--basepath /mnt/hdd/saxon/anli_v1.0/ --dataset A1 --partition train

CUDA_VISIBLE_DEVICES=2 python gen_embs.py --cuda \
--outpath /mnt/hdd/saxon/roberta-nli/ \
--basepath /mnt/hdd/saxon/anli_v1.0/ --dataset A2 --partition train

CUDA_VISIBLE_DEVICES=3 python gen_embs.py --cuda \
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

################### FOR NOW BERT ONLY
#@click.option('--modelpath', default=MODEL_PATH)
@click.command()
@click.option('--basepath', default=SNLI_PATH)
@click.option('--outpath', default="")
@click.option('--dataset', default="S")
@click.option('--partition', default="train")
@click.option('--debug', is_flag=True)
@click.option('--hides2', is_flag=True)
@click.option('--cuda', is_flag=True)
@click.option('--redo_model', is_flag=True)
def main(basepath, outpath, dataset, partition, debug, hides2, cuda, redo_model):
    ds, r, sentencemap = VALID_DATASETS[dataset]

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
    print("Loading model...")
    #model = RobertaForSequenceClassification.from_pretrained(modelpath)
    model = BertModel.from_pretrained("bert-large-uncased")

    if cuda:
        model = model.cuda()

    print("Loading tokenizer...")
    #tok = RobertaTokenizer.from_pretrained(modelpath)
    tok = BertTokenizer.from_pretrained('bert-large-uncased')
    print("Loading scorer...")

    """
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
    """
    lines = open(registered_path[f"{ds}_{partition}"]).readlines()
    #random.shuffle(lines)
    n = len(lines)
    eval_dataset = map(lambda line: json.loads(line), lines)


    sep = tok(tok.sep_token)


    embs = lazydict()

    for i, line in enumerate(eval_dataset):
        # if SNLI, we need to get the label through fuckery
        if dataset == "S":
            lst = list(line["annotator_labels"])
            label = max(set(lst), key=lst.count)
        elif dataset == "M":
            label = line["gold_label"]
        else:
            label = FULL_LABEL_MAP[line["label"]]
        s1 = line[sentencemap[0]] 
        s2 = line[sentencemap[1]]
        #print(label + "," + eval_sent)
        eval_sent = s1 + " </s> " + s2
        s1t = tok(s1, return_tensors="pt")
        s2t = tok(s2, return_tensors="pt")
        # to combine, truncate BOS token from s2 and add to s1
        #print(s1t)
        #print(s2t)
        if hides1:
            inseq = {
                "input_ids" : torch.cat([s1t["input_ids"], s2t["input_ids"][:,1:]], dim=1),
                "attention_mask" : torch.cat([s1t["attention_mask"][:,0].unsqueeze(-1), 
                    0*s1t["attention_mask"][:,1:], s2t["attention_mask"][:,1:]], dim=1)
            }
        else:
            raise NotImplemented
        if cuda:
            for key in inseq.keys():
                inseq[key] = inseq[key].cuda()
        out = model(**inseq, output_hidden_states = True)
        emb = out["hidden_states"][-1]
        emb = torch.tanh(model.classifier.dense(emb[:,0,:]))
        emb = emb.squeeze().detach().cpu().numpy().flatten()
        #print(inseq)
        #print(emb)
        #print(emb.shape)
        embs.add(label, emb)
        # get the embedding
        print(f"{i}/{n}", end="\r")
        if debug and i > 10:
            break



    X_list = []
    labels = []
    for i, key in enumerate(embs.indict.keys()):
        X_list += embs.indict[key]
        labels += [i] * len(embs.indict[key])

    X = np.stack(X_list)
    np.save(f"{outpath}_{dataset}_{partition}_BERT_X.tmp", X)
    np.save(f"{outpath}_{dataset}_{partition}_BERT_l.tmp", np.array(labels))
    print(f"Success. Arrays saved to '{outpath}_{dataset}_{partition}_BERT_X.tmp'")

if __name__ == "__main__":
    main()