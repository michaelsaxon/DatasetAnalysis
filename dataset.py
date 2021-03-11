
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split, RandomSampler, Dataset
import pandas as pd
import numpy as np

import transformers
import click
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch.nn.functional as F

from sklearn.manifold import TSNE

import math
import random
import re
import os
import json

import matplotlib.pyplot as plt

BASEPATH = ""
SNLI_PATH = "/Users/mssaxon/data/snli_1.0/"
MODEL_PATH = '/Users/mssaxon/Documents/github/DatasetAnalysis/classifier/weights/roberta_nli'

# tab separated gold_label s1bp s2bp s1p s2p s1 s2 captionID pairID label1 label2 label3 label4 label5

# generate sentensewise pair matrix

basepath = SNLI_PATH + "snli_1.0_"

registered_path = {
    'snli_train': basepath + "train.jsonl",
    'snli_dev': basepath + "dev.jsonl",
    'snli_test': basepath + "test.jsonl",
}


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    #classifier/weights/roberta_nli
    """
    config = AutoConfig.from_pretrained(
        "classifier/weights/roberta_nli",
        num_labels=num_labels,
        cache_dir="classifier/cache/",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "classifier/weights/roberta_nli",
        cache_dir="classifier/cache/",
        use_fast=model_args.use_fast_tokenizer,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        "classifier/weights/roberta_nli",
        from_tf=False,
        config=config,
        cache_dir="classifier/cache/",
    )"""




    # Log a few random samples from the training set:


class lazydict():
    def __init__(self):
        self.indict = {}

    def add(self, key, val):
        if key not in self.indict.keys():
            self.indict[key] = []
        self.indict[key].append(val)


def load_sentences_str():
        lines = open(registered_path["snli_dev"]).readlines()
        #random.shuffle(lines)
        n = len(lines)
        eval_dataset = map(lambda line: json.loads(line), lines)

        sents = lazydict()

        for i, line in enumerate(eval_dataset):
            lst = list(line["annotator_labels"])
            label = max(set(lst), key=lst.count)
            #label = line["gold_label"]
            s1 = line["sentence1"] 
            s2 = line["sentence2"]
            #print(label + "," + eval_sent)
            sents.add(label, (s1, s2))
            # get the embedding
            print(f"{i}/{n}", end="\r")

        print("")
        return sents


@click.command()
@click.option('--basepath', default=BASEPATH)
@click.option('--evalfile', default="data_labels.csv")
@click.option('--debug', default=False)
@click.option('--outfile', default="final_out.ckpt")
@click.option('--hides1', default=True)
@click.option('--perp', default=30)
def main(basepath, evalfile, debug, outfile, hides1, perp):
    if os.path.exists("_X.tmp.npy"):
        print("Loading temporary x file")
        X = np.load("_X.tmp.npy")
        labels = list(np.load("_l.tmp.npy"))
    else:
        print("Loading model...")
        model = RobertaForSequenceClassification.from_pretrained(MODEL_PATH)
        print("Loading tokenizer...")
        tok = RobertaTokenizer.from_pretrained(MODEL_PATH)
        print("Loading scorer...")

        """
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name)
        """
        lines = open(registered_path["snli_dev"]).readlines()
        #random.shuffle(lines)
        n = len(lines)
        eval_dataset = map(lambda line: json.loads(line), lines)


        sep = tok(tok.sep_token)


        embs = lazydict()

        for i, line in enumerate(eval_dataset):
            lst = list(line["annotator_labels"])
            label = max(set(lst), key=lst.count)
            s1 = line["sentence1"] 
            s2 = line["sentence2"]
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
            out = model(**inseq, output_hidden_states = True)
            emb = out["hidden_states"][0]
            emb = torch.tanh(model.classifier.dense(emb[:,0,:]))
            emb = emb.squeeze().detach().cpu().numpy().flatten()
            #print(emb)
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
        np.save("_X.tmp", X)
        np.save("_l.tmp", np.array(labels))

    print("")

    sents = load_sentences_str()

    sents_l = []

    lens_hyps = []
    lablist = []

    for i, key in enumerate(sents.indict.keys()):
            lens_hyps += map(lambda x: np.log(float(len(x[1].split(" ")))), sents.indict[key])
            sents_l += sents.indict[key]
            lablist += len(sents.indict[key]) * [key]

    #for x in X_list:
    #    print(x.shape)


    labarray = np.array(labels)
    for i in range(3):
        print(np.sum(labarray == i))


    if os.path.exists(f"_T{perp}.tmp.npy"):
        print(f"Loading temporary tsne, perp={perp}")
        X_tsne = np.load(f"_T{perp}.tmp.npy")
    else:
        print(f"Fitting TSNE, perp={perp}...")
        X_tsne = TSNE(perplexity=perp).fit_transform(X)
        np.save(f"_T{perp}.tmp.npy", X_tsne)

    keys = list(sents.indict.keys())
    colors = {'contradiction':'red', 'neutral':'black', 'entailment':'green'}

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

    #plt.scatter(np.flip(X_tsne[:,0]), np.flip(X_tsne[:,1]), c=list(map(lambda x: colors[keys[x]], list(np.flip(np.array(labels))))))
    plt.scatter(X_tsne[:,0], X_tsne[:,1], c=list(map(lambda x: colors[keys[x]], labels)))
    #plt.scatter(X_tsne[:,0], X_tsne[:,1], c=use_pt)
    #plt.scatter(X_tsne[:,0], X_tsne[:,1], c=lens_hyps)
    plt.plot([-1.265,-0.465,-1.457],[-1.938,-0.194,1.867])
    plt.show()
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