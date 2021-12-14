#CUDA_VISIBLE_DEVICES=0,1,2,3 python train_bart.py --n_gpus 4 --basepath /mnt/hdd/saxon/bart/ --n_epochs 20
#CUDA_VISIBLE_DEVICES=0,1,2,3 python train_bart.py --n_gpus 4 --basepath /mnt/hdd/saxon/bart/ --n_epochs 20 --trainfile final_lines.txt

import torch
from torch.utils.data import DataLoader, TensorDataset, random_split, RandomSampler, Dataset
import pandas as pd
import numpy as np

import transformers
import click
from transformers import RobertaTokenizer, RobertaForSequenceClassification
#from transformers import RobertaTokenizer, RobertaModel
from transformers import AdamW
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

import math
import random
import re
import os
#import argparse

BASEPATH = "/data2/saxon/bart_test"

def lazymkdir(file):
    head = os.path.split(file)[0]
    if not os.path.isdir(head):
        os.mkdir(head)
    return file

# a lot stolen from https://towardsdatascience.com/teaching-bart-to-rap-fine-tuning-hugging-faces-bart-model-41749d38f3ef

def line_len(line):
    sents = line.split(". ")
    words = line.split(" ")
    return len(sents), len(words)


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


class RobertaClassifier(pl.LightningModule):
    def __init__(self, roberta_for_seq, roberta_tok, learning_rate):
        super().__init__()
        self.model = roberta_for_seq
        self.tokenizer = roberta_tok
        self.learning_rate = learning_rate


    def forward(self, x, **kwargs):
        return self.model(x, **kwargs)        

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr = self.learning_rate)
        return optimizer

    def training_step(self, batch, batch_idx):
        src_ids, src_mask = batch[0], batch[1]
        tgt_ids = batch[2]
        decoder_input_ids = shift_tokens_right(tgt_ids, self.tokenizer.pad_token_id)

        outputs = self(src_ids, attention_mask=src_mask, decoder_input_ids=decoder_input_ids, use_cache=False)
        lm_logits = outputs[0]
        ce_loss_fct = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        loss = ce_loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), tgt_ids.view(-1))

        return {'loss':loss}


    def validation_step(self, batch, batch_idx):
        src_ids, src_mask = batch[0], batch[1]
        tgt_ids = batch[2]

        decoder_input_ids = shift_tokens_right(tgt_ids, self.tokenizer.pad_token_id)
        
        # Run the model and get the logits
        outputs = self(src_ids, attention_mask=src_mask, decoder_input_ids=decoder_input_ids, use_cache=False)
        lm_logits = outputs[0]

        ce_loss_fct = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        val_loss = ce_loss_fct(lm_logits.view(-1, lm_logits.shape[-1]), tgt_ids.view(-1))

        return {'loss': val_loss}

    def generate_text(self, text, eval_beams, early_stopping = True, max_len = 1024, min_len = 10, temp = 0.4):
        ''' Function to generate text '''
        generated_ids = self.model.generate(
            text["input_ids"],
            attention_mask=text["attention_mask"],
            use_cache=True,
            decoder_start_token_id = self.tokenizer.pad_token_id,
            num_beams= eval_beams,
            max_length = max_len,
            min_length = min_len,
            early_stopping = early_stopping,
            temperature = temp,
        )
        return [self.tokenizer.decode(w, skip_special_tokens=True, clean_up_tokenization_spaces=True) for w in generated_ids]


def load_sentences_str(registered_path, dataset, partition, sentencemap):
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



class SummaryDataModule(pl.LightningDataModule):
    def __init__(self, tokenizer, data_file, batch_size, num_examples = 8192):
        super().__init__()
        self.tokenizer = tokenizer
        self.data_file = data_file
        self.batch_size = batch_size
        self.num_examples = num_examples

    # Loads and splits the data into training, validation and test sets with a 60/20/20 split
    def prepare_data(self):
        print("Preparing data...")
        self.data = load_sentences_str(self.data_file, self.num_examples)
        print("Preparing input vectors...")
        self.train, self.validate, self.test = np.split(self.data.sample(frac=1), [int(.9*len(self.data)), int(.95*len(self.data))])

    # encode the sentences using the tokenizer  
    def setup(self, stage):
        self.train = encode_sentences(self.tokenizer, self.train['source'], self.train['target'])
        self.validate = encode_sentences(self.tokenizer, self.validate['source'], self.validate['target'])
        self.test = encode_sentences(self.tokenizer, self.test['source'], self.test['target'])

    # Load the training, validation and test sets in Pytorch Dataset objects
    def train_dataloader(self):
        dataset = TensorDataset(self.train['input_ids'], self.train['attention_mask'], self.train['labels'])                          
        train_data = DataLoader(dataset, sampler = RandomSampler(dataset), batch_size = self.batch_size)
        return train_data

    def val_dataloader(self):
        dataset = TensorDataset(self.validate['input_ids'], self.validate['attention_mask'], self.validate['labels']) 
        val_data = DataLoader(dataset, batch_size = self.batch_size)                       
        return val_data

    def test_dataloader(self):
        dataset = TensorDataset(self.test['input_ids'], self.test['attention_mask'], self.test['labels']) 
        test_data = DataLoader(dataset, batch_size = self.batch_size)                   
        return test_data

def shift_tokens_right(input_ids, pad_token_id):
    """ Shift input ids one token to the right, and wrap the last non pad token (usually <eos>).
      This is taken directly from modeling_bart.py
    """
    prev_output_tokens = input_ids.clone()
    index_of_eos = (input_ids.ne(pad_token_id).sum(dim=1) - 1).unsqueeze(-1)
    prev_output_tokens[:, 0] = input_ids.gather(1, index_of_eos).squeeze()
    prev_output_tokens[:, 1:] = input_ids[:, :-1]
    return prev_output_tokens

def encode_sentences(tokenizer, source_sentences, target_sentences, max_length=1024, pad_to_max_length=True, return_tensors="pt"):
    ''' Function that tokenizes a sentence 
      Args: tokenizer - the BART tokenizer; source and target sentences are the source and target sentences
      Returns: Dictionary with keys: input_ids, attention_mask, target_ids
    '''

    input_ids = []
    attention_masks = []
    target_ids = []
    tokenized_sentences = {}


    skips = []

    for i, sentence in enumerate(source_sentences):
        #try:
        encoded_dict = tokenizer(
            sentence,
            max_length=max_length,
            padding="max_length" if pad_to_max_length else None,
            truncation=True,
            return_tensors=return_tensors,
            add_prefix_space = True
          )
        #except:
        #    #print(i)
        #    skips.append(i)
        #    continue
  

        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    input_ids = torch.cat(input_ids, dim = 0)
    attention_masks = torch.cat(attention_masks, dim = 0)

    MAX_LENGTH_ABS = 400
    for i, sentence in enumerate(target_sentences):
        if i in skips:
            continue
        encoded_dict = tokenizer(
            sentence,
            max_length=MAX_LENGTH_ABS,
            padding="max_length" if pad_to_max_length else None,
            truncation=True,
            return_tensors=return_tensors,
            add_prefix_space = True
          )
        # Shift the target ids to the right
        # shifted_target_ids = shift_tokens_right(encoded_dict['input_ids'], tokenizer.pad_token_id)
        target_ids.append(encoded_dict['input_ids'])

    target_ids = torch.cat(target_ids, dim = 0)


    batch = {
        "input_ids": input_ids,
        "attention_mask": attention_masks,
        "labels": target_ids,
    }

    return batch


def noise_sentence(sentence_, percent_words, replacement_token = "<mask>"):
    '''
    Function that noises a sentence by adding <mask> tokens
    Args: sentence - the sentence to noise
          percent_words - the percent of words to replace with <mask> tokens; the number is rounded up using math.ceil
    Returns a noised sentence
    '''
    # Create a list item and copy
    sentence_ = sentence_.split(' ')
    sentence = sentence_.copy()

    num_words = math.ceil(len(sentence) * percent_words)

    # Create an array of tokens to sample from; don't include the last word as an option because in the case of lyrics
    # that word is often a rhyming word and plays an important role in song construction
    sample_tokens = set(np.arange(0, np.maximum(1, len(sentence)-1)))

    words_to_noise = random.sample(sample_tokens, num_words)

    # Swap out words, but not full stops
    for pos in words_to_noise:
        if sentence[pos] != '.':
            sentence[pos] = replacement_token

    # Remove redundant spaces
    sentence = re.sub(r' {2,5}', ' ', ' '.join(sentence))

    # Combine concurrent <mask> tokens into a single token; this just does two rounds of this; more could be done
    sentence = re.sub(r'<mask> <mask>', "<mask>", sentence)
    sentence = re.sub(r'<mask> <mask>', "<mask>", sentence)
    return sentence


#@click.argument('name')
# abs_split.txt final_lines.txt
@click.command()
@click.option('--n_gpus', default=1, help='number of gpus')
@click.option('--n_epochs', default=2, help='max number of epochs')
@click.option('--basepath', default=BASEPATH)
@click.option('--outfile', default="final_out.ckpt")
@click.option('--datasets', default="../datasets.csv")
@click.option('--dataset', default="snli")
@click.option('--lr', default=2e-5)
def main(n_gpus, n_epochs, basepath, trainfile, n_examples, outfile, lr):
    # if r-en
    model_id = "roberta-large"
    print("Loading model...")
    model = RobertaForSequenceClassification.from_pretrained(model_id)
    print("Loading tokenizer...")
    tok = RobertaTokenizer.from_pretrained(model_id)
    print("Init litmodel...")
    ltmodel = RobertaClassifier(model, tok, lr)
    print("Init dataset...")

    #######

    summary_data = SummaryDataModule(tok, basepath + "/" + trainfile, batch_size = 1, num_examples = n_examples)
    path = basepath + "/ckpts"
    lazymkdir(path)
    print("Loading model...")
    checkpoint = ModelCheckpoint(path)
    print("Init trainer...")
    trainer = pl.Trainer(gpus = n_gpus, max_epochs = n_epochs, checkpoint_callback = checkpoint, progress_bar_refresh_rate = 4)
    print("Training...")
    trainer.fit(ltmodel, summary_data)
    trainer.save_checkpoint(basepath + +"/" + outfile)


if __name__ == "__main__":
    main()