'''
CUDA_VISIBLE_DEVICES=0 python train_classifier.py --n_gpus 1 \
    --dataset S --batch_size 16

CUDA_VISIBLE_DEVICES=1 python train_classifier.py --n_gpus 1 \
    --dataset M --batch_size 16


'''

from email.policy import default
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split, RandomSampler, Dataset
import numpy as np

import click
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import AdamW
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import wandb
import json
from tqdm import tqdm
from pathlib import PurePath

from collections import defaultdict
import os

import time

from manage_settings import get_write_settings, lazymkdir

BASEPATH = "/data2/saxon/bart_test"

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


def load_nli_data(basepath, dataset, partition, label_id = True):
    ds, r, sentencemap = VALID_DATASETS[dataset]

    registered_path = {
        'snli': f'snli_1.0/snli_1.0_{partition}.jsonl',
        'mnli': f'multinli_1.0/multinli_1.0_{partition}.jsonl',
        'anli': f'anli_v1.0/R{r}/{partition}.jsonl'
    }

    with open(PurePath(basepath + "/" + registered_path[ds])) as f:
        lines = f.readlines()

    sents = []

    for line in tqdm(lines):
        line = json.loads(line)
        if dataset == "S":
            lst = list(line["annotator_labels"])
            label = max(set(lst), key=lst.count)
        elif dataset == "M":
            label = line["gold_label"]
        else:
            label = FULL_LABEL_MAP[line["label"]]
        s1 = line[sentencemap[0]]
        s2 = line[sentencemap[1]]
        if label_id:
            label = LABEL_IDS[label]
        sents.append((s1, s2, label))
    
    return sents


class NLIDataset(Dataset):
    def __init__(self, sents, tokenizer, bias, bias_factor = 1):
        # fuck it just store all the sentences in memory lmao
        self.sents = sents
        self.length = len(sents)
        self.tok = tokenizer
        self.bias = bias
        self.factor = bias_factor

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        s1, s2, label = self.sents[idx]
        s1t = self.tok(s1, return_tensors="pt")
        s2t = self.tok(s2, return_tensors="pt")
        if self.bias:
            datum = {
                "input_ids" : torch.cat([self.factor * s1t["input_ids"], s2t["input_ids"][:,1:]], dim=1),
                "attention_mask" : torch.cat([s1t["attention_mask"][:,0].unsqueeze(-1), 
                    0*s1t["attention_mask"][:,1:], s2t["attention_mask"][:,1:]], dim=1)
            }
        else:
            datum = {
                "input_ids" : torch.cat([s1t["input_ids"], s2t["input_ids"][:,1:]], dim=1),
                "attention_mask" : torch.cat([s1t["attention_mask"][:,0].unsqueeze(-1), 
                    s1t["attention_mask"][:,1:], s2t["attention_mask"][:,1:]], dim=1)
            }
        datum["labels"] = torch.tensor([label])
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
    return data


class plNLIDataModule(pl.LightningDataModule):
    def __init__(self, tokenizer, basepath, dataset, batch_size, bias):
        super().__init__()
        self.tokenizer = tokenizer
        self.basepath = basepath
        self.dataset = dataset
        self.batch_size = batch_size
        self.bias = bias

    # Loads and splits the data into training, validation and test sets with a 60/20/20 split
    def prepare_data(self):
        print("Preparing data...")
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
        dataset = NLIDataset(self.train, self.tokenizer, self.bias)                   
        train_data = DataLoader(dataset, 
            sampler = RandomSampler(dataset), 
            batch_size = self.batch_size,
            collate_fn = pad_seq_collate_fn)
        return train_data

    def val_dataloader(self):
        dataset = NLIDataset(self.valid, self.tokenizer, self.bias)
        val_data = DataLoader(dataset, 
            sampler = RandomSampler(dataset), 
            batch_size = self.batch_size,
            collate_fn = pad_seq_collate_fn)
        return val_data

    def test_dataloader(self):
        dataset = NLIDataset(self.test, self.tokenizer, self.bias)
        test_data = DataLoader(dataset, 
            sampler = RandomSampler(dataset), 
            batch_size = self.batch_size,
            collate_fn = pad_seq_collate_fn)
        return test_data


# pre-encode using the tokenizer into pt files first
#def encode_sentences(tokenizer, source_sentences, return_tensors="pt"):


#@click.argument('name')
# abs_split.txt final_lines.txt
@click.command()
@click.option('--n_gpus', default=1, help='number of gpus')
@click.option('--n_epochs', default=25, help='max number of epochs')
@click.option('--dataset', default="snli")
@click.option('--lr', default=2e-5)
@click.option('--model_id', default="roberta-large")
@click.option('--batch_size', default=16)
@click.option('--biased', is_flag=True)
def main(n_gpus, n_epochs, dataset, lr, biased, model_id, batch_size):
    dir_settings = get_write_settings(["data_save_dir", "dataset_dir"])

    wandb.login()

    projectname = "DatasetAnalysis-NLIbias"
    
    start_time_str = time.strftime('%y%m%d:%H:%M')

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
            "lang": "en",
            "start" : start_time_str
        }
    )
    # setting up global metrics
    wandb.define_metric('val_best_acc', summary="max")

    wandb_logger = WandbLogger(log_model=True)

    run_name = f"{dataset}-{model_id}-{lr}-{start_time_str}"

    print("Loading model...")
    model = RobertaForSequenceClassification.from_pretrained(model_id)
    print("Loading tokenizer...")
    tokenizer = RobertaTokenizer.from_pretrained(model_id)
    print("Init litmodel...")
    ltmodel = RobertaClassifier(model, lr)
    print("Init dataset...")

    wandb_logger.watch(ltmodel.model, log_freq=500)

    nli_data = plNLIDataModule(tokenizer, dir_settings["dataset_dir"], dataset, batch_size, biased)

    run_path = PurePath(dir_settings["data_save_dir"] + "/" + run_name)
    lazymkdir(run_path)
    ckpts_path = PurePath(str(run_path) + "/ckpts")
    lazymkdir(ckpts_path)

    print("Loading model...")
    checkpoint = ModelCheckpoint(dirpath=ckpts_path, monitor="val_accuracy", mode="max",
        save_last=True, save_top_k=2)
    print("Init trainer...")

    trainer = pl.Trainer(gpus = n_gpus, max_epochs = n_epochs, 
        checkpoint_callback = checkpoint, progress_bar_refresh_rate = 4,
        logger = wandb_logger)
    print("Training...")
    trainer.fit(ltmodel, nli_data)

    wandb_logger.unwatch(ltmodel.model)

    wandb.finish()


if __name__ == "__main__":
    main()