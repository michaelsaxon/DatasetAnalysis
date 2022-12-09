'''
CUDA_VISIBLE_DEVICES=1 python train_classifier.py --dataset S --batch_size 64 --biased

SEE RECIPES.txt
'''

from ctypes.wintypes import tagSIZE
from email.policy import default
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split, RandomSampler, Dataset
import numpy as np

import click
from transformers import RobertaTokenizer, RobertaForSequenceClassification, BertTokenizer, BertForSequenceClassification
from transformers import AdamW, AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger
import wandb
import json
from tqdm import tqdm
from pathlib import PurePath

from collections import defaultdict
import os

import time

# is this really the way to pull out of manage settings?
from manage_settings import get_write_settings, lazymkdir

from cartography import CartographyCallback


MAX_MODEL_LEN_HACK = 512

# (name, number (only for ANLI), [premise_name, hypothesis_name], dev_only)
VALID_DATASETS = {
    "S" : ("snli", None, ["sentence1", "sentence2"], False),
    "A1": ("anli", 1, ["context", "hypothesis"], False),
    "A2": ("anli", 2, ["context", "hypothesis"], False),
    "A3": ("anli", 3, ["context", "hypothesis"], False),
    "AA": ("anli_all", None, ["context", "hypothesis"], False),
    "M" : ("mnli", None, ["sentence1", "sentence2"], True),
    "OC" : ("ocnli", None, ["sentence1", "sentence2"], False),
    "F" : ("fever", None, ["query", "context"], True),
    "CF" : ("counterfactual", None, ["sentence1", "sentence2"], False),
    "X" : ("xnli", None, ["sentence1", "sentence2"], False),
    "MU" : ("mnli_u", None, ["sentence1", "sentence2"], True),
    "MB" : ("mnli_b", None, ["sentence1", "sentence2"], True),
    "SdbA" : ("debiased_snli_aug", None, ["premise", "hypothesis"], True),
    "MdbA" : ("debiased_mnli_aug", None, ["premise", "hypothesis"], True)
}

FULL_LABEL_MAP = {
    "e" : "entailment",
    "c" : "contradiction",
    "n" : "neutral"
}

FEVER_LABEL_MAP = {
    "SUPPORTS" : "entailment",
    "REFUTES" : "contradiction",
    "NOT ENOUGH INFO" : "neutral"
}

WU_LABEL_MAP = ["entailment", "neutral", "contradiction"]

# THIS IS THE IDX REVERSAL DICTIONARY
# IF THIS IS MODIFIED ALL PRIOR PROG IS LOST
LABEL_IDS = {}
for i, label in enumerate(FULL_LABEL_MAP.keys()):
    LABEL_IDS[FULL_LABEL_MAP[label]] = i




class RobertaClassifier(pl.LightningModule):
    def __init__(self, roberta_for_seq, learning_rate, loss_fct = torch.nn.CrossEntropyLoss()):
        super().__init__()
        self.model = roberta_for_seq
        self.learning_rate = learning_rate
        self.loss_fct = loss_fct

    def forward(self, **kwargs):
        return self.model(**kwargs)


    def forward_get_embs(self, batch):
        return self.model(input_ids = batch['input_ids'], attention_mask=batch['attention_mask'], 
            output_hidden_states = True)["hidden_states"][-1]


    def forward_loss_acc(self, batch):
        outputs = self(input_ids = batch['input_ids'], attention_mask=batch['attention_mask'])
        targets = batch['labels'].squeeze()
        logits = outputs.logits
        preds = torch.max(logits, dim=-1).indices
        loss = self.loss_fct(logits, targets)
        acc = torch.sum(torch.eq(preds, targets)) / targets.shape[0]
        return preds, loss, acc


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr = self.learning_rate)
        return optimizer

    def training_step(self, batch, batch_idx):
        _, loss, acc = self.forward_loss_acc(batch)
        self.log('train_loss', loss)
        self.log('train_accuracy', acc)
        return loss

    def validation_step(self, batch, batch_idx):
        preds, loss, acc = self.forward_loss_acc(batch)
        self.log('val_loss', loss)
        self.log('val_accuracy', acc)
        self.log('val_best_acc', acc)
        return preds

    def test_step(self, batch, batch_idx):
        preds, loss, acc = self.forward_loss_acc(batch)
        self.log('test_loss', loss)
        self.log('test_accuracy', acc)
        return preds


def load_sick_data(basepath, label_id = True):
    path_affix = "SICK/SICK.txt"
    with open(PurePath(basepath + path_affix)) as f:
        lines = f.readlines()[1:]
    # keys will be TRAIN TRIAL TEST
    lines_sorted = defaultdict(list)
    for line in lines:
        line = line.strip().split("\t")
        partition = line[-1]
        sentence_a = line[1]
        sentence_b = line[2]
        # labels are allcaps vers of others
        label = LABEL_IDS.get(line[3].lower(), LABEL_IDS["neutral"])
        lines_sorted[partition].append((sentence_a, sentence_b, label))
    return lines_sorted["TRAIN"], lines_sorted["TRIAL"], lines_sorted["TEST"]


def load_counterfact_nli_data(basepath, partition, label_id = True):
    fname = PurePath(basepath + f"/counterfactually-augmented-data/NLI/all_combined/{partition}.tsv")
    lines = open(fname, "r").readlines()[1:]
    sents = []
    for line in tqdm(lines):
        line = line.strip().split("\t")
        label = line[2]
        if label_id:
            label = LABEL_IDS.get(label, LABEL_IDS["neutral"])
        sents.append((line[0], line[1], label))
    return sents


def load_nli_data(basepath, dataset, partition, label_id = True):
    # hack to add in nonstandard formatted datasets (not in json)
    if dataset == "CF":
        return load_counterfact_nli_data(basepath, partition, label_id)
    
    ds, r, sentencemap, dev_only = VALID_DATASETS[dataset]

    # lightest weight hack
    if dev_only and partition == "test":
        partition = "dev"

    registered_path = {
        'snli': f'snli_1.0/snli_1.0_{partition}.jsonl',
        'mnli': f'multinli_1.0/multinli_1.0_{partition}.jsonl',
        'anli': f'anli_v1.0/R{r}/{partition}.jsonl',
        'ocnli': f'OCNLI/data/ocnli/{partition}.json',
        'fever': f'nli_fever/{partition}_labels.jsonl',
        'xnli': f'XNLI-1.0/xnli.{partition}.jsonl',
        'mnli_u' : f'multinli_1.0/unmatched/{partition}.jsonl',
        'mnli_b' : f'multinli_1.0/matched/{partition}.jsonl',
        'debiased_snli_aug' : f'debiased_wu/snli_z-aug_{partition}.jsonl',
        'debiased_mnli_aug' : f'debiased_wu/mnli_z-aug_{partition}.jsonl',
    }

    if dataset == "AA":
        lines = []
        for r in [1, 2, 3]:
            with open(PurePath(basepath + f'/anli_v1.0/R{r}/{partition}.jsonl')) as f:
                lines += f.readlines()
    else:
        with open(PurePath(basepath + "/" + registered_path[ds])) as f:
            lines = f.readlines()

    sents = []

    for i, line in enumerate(tqdm(lines)):
        line = json.loads(line)
        if dataset == "S":
            lst = list(line["annotator_labels"])
            label = max(set(lst), key=lst.count)
        elif dataset == "M" or dataset == "X" or dataset == "MU" or dataset == "MB":
            label = line["gold_label"]
        elif dataset == "OC":
            label = line["label"].lower()
        elif dataset == "F":
            label = FEVER_LABEL_MAP[line["label"]]
        elif dataset == "SdbA" or dataset == "MdbA":
            label = WU_LABEL_MAP[int(line["label"])]
        else:
            label = FULL_LABEL_MAP[line["label"]]
        s1 = line[sentencemap[0]]
        s2 = line[sentencemap[1]]
        if label_id:
            # defaults to neutral
            label = LABEL_IDS.get(label, LABEL_IDS["neutral"])
        sents.append((s1, s2, label, i))
    
    return sents


class NLIDataset(Dataset):
    def __init__(self, sents, tokenizer, bias, bias_factor = 1, s2only = False, s1only = False):
        # fuck it just store all the sentences in memory lmao
        self.sents = sents
        self.length = len(sents)
        self.tok = tokenizer
        self.bias = bias
        self.factor = bias_factor
        self.s2only = s2only
        self.s1only = s1only

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        s1, s2, label, idx_src = self.sents[idx]
        s1t = self.tok(s1, return_tensors="pt")
        s2t = self.tok(s2, return_tensors="pt")
        if self.bias:
            datum = {
                "input_ids" : torch.cat([self.factor * s1t["input_ids"], s2t["input_ids"][:,1:]], dim=1),
                "attention_mask" : torch.cat([s1t["attention_mask"][:,0].unsqueeze(-1), 
                    0*s1t["attention_mask"][:,1:], s2t["attention_mask"][:,1:]], dim=1)
            }
        elif self.s2only:
            datum = {
                "input_ids" : s2t["input_ids"],
                "attention_mask" : s2t["attention_mask"]
            }
        elif self.s1only:
            datum = {
                "input_ids" : s1t["input_ids"],
                "attention_mask" : s1t["attention_mask"]
            }
        else:
            datum = {
                "input_ids" : torch.cat([s1t["input_ids"], s2t["input_ids"][:,1:]], dim=1),
                "attention_mask" : torch.cat([s1t["attention_mask"][:,0].unsqueeze(-1), 
                    s1t["attention_mask"][:,1:], s2t["attention_mask"][:,1:]], dim=1)
            }
        datum["labels"] = torch.tensor([label])
        datum["idx"] = torch.tensor(idx_src)
        return datum


def pad_seq_collate_fn(seq_of_samples):
    data = defaultdict(list)
    for datum in seq_of_samples:
        for key in datum.keys():
            if len(datum[key].shape) > 1:
                datum[key] = datum[key].squeeze()
            data[key].append(datum[key])
    for key in data.keys():
        data[key] = torch.nn.utils.rnn.pad_sequence(data[key], batch_first = True)
        shape = data[key].shape
        if shape[1] > MAX_MODEL_LEN_HACK:
            data[key] = data[key][:,:MAX_MODEL_LEN_HACK]
    return data


class plNLIDataModule(pl.LightningDataModule):
    def __init__(self, tokenizer, basepath, dataset, batch_size, bias, bias_factor = 1, s2only = False, s1only = False):
        super().__init__()
        self.tokenizer = tokenizer
        self.basepath = basepath
        self.dataset = dataset
        self.batch_size = batch_size
        self.bias = bias
        self.bias_factor = bias_factor
        self.s2only = s2only
        self.s1only = s1only

    # Loads and splits the data into training, validation and test sets with a 60/20/20 split
    def prepare_data(self, test_only = False):
        print("Preparing data...")
        if self.dataset == "SICK":
            self.train, self.valid, self.test = load_sick_data(self.basepath)
        else:
            if not test_only:
                self.train = load_nli_data(self.basepath, self.dataset, "train")
                self.valid = load_nli_data(self.basepath, self.dataset, "dev")
            self.test = load_nli_data(self.basepath, self.dataset, "test")
        #print("Preparing input vectors...")

    # deprecated, for now I'm doing in-time tokenization with NLIDataset
    # encode the sentences using the tokenizer  
    #def setup(self, stage):
    #    self.train = encode_sentences(self.tokenizer, self.train)
    #    self.valid = encode_sentences(self.tokenizer, self.validate)
    #    self.test = encode_sentences(self.tokenizer, self.test)

    # Load the training, validation and test sets in Pytorch Dataset objects
    def train_dataloader(self):
        dataset = NLIDataset(self.train, self.tokenizer, self.bias, self.bias_factor, self.s2only, self.s1only)                   
        train_data = DataLoader(dataset, 
            sampler = RandomSampler(dataset), 
            batch_size = self.batch_size,
            collate_fn = pad_seq_collate_fn,
            num_workers = 16)
        return train_data

    def val_dataloader(self):
        dataset = NLIDataset(self.valid, self.tokenizer, self.bias, self.bias_factor, self.s2only, self.s1only)
        val_data = DataLoader(dataset, 
            batch_size = self.batch_size,
            collate_fn = pad_seq_collate_fn,
            num_workers = 16)
        return val_data

    def test_dataloader(self):
        dataset = NLIDataset(self.test, self.tokenizer, self.bias, self.bias_factor, self.s2only, self.s1only)
        test_data = DataLoader(dataset, 
            batch_size = self.batch_size,
            collate_fn = pad_seq_collate_fn,
            num_workers = 16)
        return test_data


# pre-encode using the tokenizer into pt files first
#def encode_sentences(tokenizer, source_sentences, return_tensors="pt"):

def load_model_tokenizer(model_id, model_class, tokenizer_class, num_labels=3):
    print("Loading model...")
    model = model_class.from_pretrained(model_id, num_labels=num_labels)
    print("Loading tokenizer...")
    tokenizer = tokenizer_class.from_pretrained(model_id)
    return model, tokenizer

def choose_load_model_tokenizer(model_id, dataset):
    if dataset == "OC":
        lang = "zh"
    elif dataset == "X":
        lang = "multi"
    else:
        lang = "en"
    if lang == "en":
        return load_model_tokenizer(model_id, RobertaForSequenceClassification, RobertaTokenizer)
    elif lang == "zh":
        return load_model_tokenizer(model_id, BertForSequenceClassification, BertTokenizer)
    elif lang == "multi":
        return load_model_tokenizer(model_id, AutoModelForSequenceClassification, AutoTokenizer)
    else:
        raise NotImplementedError


#@click.argument('name')
# abs_split.txt final_lines.txt
@click.command()
@click.option('--n_gpus', default=1, help='number of gpus')
@click.option('--n_epochs', default=25, help='max number of epochs')
@click.option('--dataset', default="S")
@click.option('--lr', default=1e-5)
@click.option('--model_id', default="roberta-large")
@click.option('--pretrained_path', help="location of an instance of this model that has already been fine-tuned", default="")
@click.option('--batch_size', default=48)
@click.option('--biased', is_flag=True)
@click.option('--extreme_bias', is_flag=True)
@click.option('--s2only', is_flag=True)
@click.option('--s1only', is_flag=True)
@click.option('--collect_cartography', is_flag=True, help="run the dataset cartography metrics, tracking samplewise confidence and correctness")
def main(n_gpus, n_epochs, dataset, lr, biased, model_id, batch_size, extreme_bias, s2only, s1only, pretrained_path, collect_cartography):
    dir_settings = get_write_settings(["data_save_dir", "dataset_dir"])

    wandb.login()

    projectname = "DatasetAnalysis-NLIbias"
    
    start_time_str = time.strftime('%y%m%d-%H%M')

    if s2only:
        extreme_bias = False
        biased = True
    if extreme_bias:
        biased = True
    
    if dataset == "OC":
        lang = "zh"
    elif dataset == "X":
        lang = "multi"
    else:
        lang = "en"

    # generate name for W&B tracking
    name = ""
    if s2only:
        name += "tests2-"
    elif s1only:
        name += "tests1-"
    else:
        name += "TRAIN-"
    dsname, dsnum, _, _ = VALID_DATASETS[dataset]
    full_dsname = dsname
    if dsnum is not None:
        full_dsname = f"{dsname}{dsnum}"
    name += full_dsname + f"-lr{lr:.1E}"
    name += "-r0310"
    # generate tags for W&B tracking
    tags = {full_dsname : 1, dsname : 1, "has_test" : 1}
    if s2only:
        tags["s2only"] = 1
    elif s1only:
        tags["s1only"] = 1
    else:
        tags["train"] = 1
        tags["baseline"] = 1
    if pretrained_path != "":
        tags["from_pretrained"] = 1
    tags = list(tags.keys())

    # generate wandb config details
    wandb.init(
        project = projectname,
        entity="saxon",
        config = {
            "learning_rate" : lr,
            "batch_size" : batch_size,
            "epochs" : n_epochs,
            "dataset" : dataset,
            "model" : model_id,
            "biased": biased,
            "extreme_bias" : extreme_bias,
            "lang": lang,
            "start" : start_time_str,
            "s2only" : s2only,
            "s1only" : s1only,
            "pretrained_path" : pretrained_path
        },
        tags = tags,
        name = name
    )
    # setting up global metrics
    wandb.define_metric('val_best_acc', summary="max")

    wandb_logger = WandbLogger(log_model=True)

    run_name = f"{dataset}-{model_id}-{lr}-{start_time_str}"
    if biased:
        run_name = f"Bias-{run_name}"
    if extreme_bias:
        run_name = f"Extreme{run_name}"
    if s2only:
        run_name = f"S2only{run_name}"

    model, tokenizer = choose_load_model_tokenizer(model_id, dataset)

    print("Init litmodel...")
    ltmodel = RobertaClassifier(model, lr)
    if pretrained_path != "":
        ckpt = torch.load(pretrained_path)
        ltmodel.load_state_dict(ckpt["state_dict"])
    print("Init dataset...")

    wandb_logger.watch(ltmodel.model, log_freq=500)

    if extreme_bias:
        factor = 0
    else:
        factor = 1
    nli_data = plNLIDataModule(tokenizer, dir_settings["dataset_dir"], dataset, batch_size, biased, factor, s2only, s1only)

    run_path = PurePath(dir_settings["data_save_dir"] + "/" + run_name)
    lazymkdir(run_path)
    ckpts_path = PurePath(str(run_path) + "/ckpts")
    lazymkdir(ckpts_path)

    callbacks = [TQDMProgressBar(refresh_rate=4)]
    if collect_cartography:
        callbacks.append(CartographyCallback(
            
            f"{ckpts_path}/{run_name}_cart"
        ))

    print("Loading model...")
    checkpoint = ModelCheckpoint(dirpath=ckpts_path, monitor="val_accuracy", mode="max",
        save_last=True, save_top_k=2)
    print("Init trainer...")

    trainer = pl.Trainer(gpus = n_gpus, max_epochs = n_epochs, enable_checkpointing = checkpoint, logger = wandb_logger, callbacks = callbacks)
    print("Training...")
    trainer.fit(ltmodel, nli_data)
    trainer.test(ltmodel, nli_data)

    wandb.finish()


if __name__ == "__main__":
    main()