import torch
import torch.nn.functional as F
import numpy as np
from pytorch_lightning.callbacks import Callback
from manage_settings import get_write_settings, lazymkdir
import glob
from pathlib import PurePath

def define_samplewise_metric(key_nums_dict):
    out = {}
    for key in key_nums_dict.keys():
        out[key] = np.zeros(key_nums_dict[key])
    return out

# confidence
# C_i = 1/E \sum_epochs p_model(label | input)
def confidence_elementwise(targets, logits):
    return torch.gather(F.softmax(logits, dim=-1), -1, targets.unsqueeze(-1)).detach().cpu().numpy()

def correct_elementwise(targets, preds):
    return torch.eq(preds, targets).squeeze().detach().cpu().numpy()

# a model will contain self.confidences and self.correctnesses, two dicts of dicts
# integrate a new value into it from the returned elements

class CartographyCallback(Callback):
    def __init__(self, output_base):
        super().__init__()
        self.output_base = output_base

    def init_buffers(self, trainer):
        key_nums = {
            "train" : len(trainer.datamodule.train), 
            "val" : len(trainer.datamodule.valid), 
            "test" : len(trainer.datamodule.test)
        }
        self.confidences = define_samplewise_metric(key_nums)
        self.correctnesses = define_samplewise_metric(key_nums)

    def cartography_save(self, epoch, key):
        np.save(f"{self.output_base}/conf_{key}_{epoch}.npy", self.confidences[key])
        np.save(f"{self.output_base}/corr_{key}_{epoch}.npy", self.correctnesses[key])

    def batch_end_accumulate(self, batch, outputs, key):
        targets = batch['labels'].squeeze()
        logits = outputs['logits']
        preds = torch.max(logits, dim=-1).indices
        batch_idces = batch['idx'].cpu().numpy().squeeze()
        self.confidences[key][batch_idces] = confidence_elementwise(targets, logits).squeeze()
        self.correctnesses[key][batch_idces] = correct_elementwise(targets, preds).squeeze()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.batch_end_accumulate(batch, outputs, "train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self.batch_end_accumulate(batch, outputs, "val")

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self.batch_end_accumulate(batch, outputs, "test")

    def on_train_epoch_end(self, trainer, pl_module):
        self.cartography_save(trainer.current_epoch, "train")

    def on_validation_epoch_end(self, trainer, pl_module):
        self.cartography_save(trainer.current_epoch, "val")
    
    def on_test_epoch_end(self, trainer, pl_module):
        self.cartography_save(trainer.current_epoch, "test")

    def on_train_start(self, trainer, pl_module):
        self.init_buffers(trainer)
    
    def on_sanity_check_start(self, trainer, pl_module):
        self.init_buffers(trainer)

def cartography_from_dir(folder, n_epochs, key):
    if key == 'test':
        epochs = [n_epochs]
    else:
        epochs = list(range(n_epochs))
    confs = []
    #corrs = []
    for epoch in epochs:
        confs.append(np.load(PurePath(folder + f"/conf_{key}_{epoch}.npy")))
        #corrs.append(np.load(folder + f"/corr_{key}_{epoch}.npy"))
    confs = np.stack(confs)
    #corrs = np.stack(corrs)
    mus = confs.sum(0) / n_epochs
    sigmas = np.sqrt(np.sum(np.power(confs - mus, 2) / n_epochs, 0))
    return mus, sigmas

def find_cartography_dir(folder, dataset, model):
    dataset_model_stub = str(PurePath(folder + f"{dataset}-{model}-"))
    folder_options = glob.glob(f"{dataset_model_stub}*")
    if len(folder_options) == 1:
        return folder_options[0]
    elif len(folder_options) > 1:
        print("Please select one from:")
        for i in range(len(folder_options)):
            print(f"{i} : {folder_options[i]}")
        choice = input("Choice: ")
        return folder_options[choice]
    else:
        raise FileNotFoundError(f"Couldn't find a matching checkpoint folder for dataset {dataset}, model {model} in path {folder}")